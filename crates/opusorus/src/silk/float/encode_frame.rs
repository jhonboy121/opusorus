//! Port of `silk/float/encode_frame_FLP.c`: VAD decision, frame encoding with the rate control
//! loop, and LBRR encoding.

use crate::celt::entenc::EcEnc;
use crate::silk::coding::{
    silk_encode_pulses, silk_gains_dequant, silk_gains_id, silk_gains_quant,
};
use crate::silk::define::{
    CODE_CONDITIONALLY, LA_PITCH_MAX, LA_SHAPE_MS, MAX_CONSECUTIVE_DTX, MAX_FRAME_LENGTH,
    MAX_NB_SUBFR, N_LEVELS_QGAIN, NB_SPEECH_FRAMES_BEFORE_DTX, TYPE_NO_VOICE_ACTIVITY,
    TYPE_UNVOICED, VAD_NO_ACTIVITY,
};
use crate::silk::encoder_common::silk_encode_indices;
use crate::silk::float::find::{silk_find_pitch_lags_flp, silk_find_pred_coefs_flp};
use crate::silk::float::noise_shape::{silk_noise_shape_analysis_flp, silk_process_gains_flp};
use crate::silk::float::sigproc::{silk_max_float, silk_short2float_array};
use crate::silk::float::structs::{SilkEncoderControlFlp, SilkEncoderStateFlp};
use crate::silk::float::wrappers::silk_nsq_wrapper_flp;
use crate::silk::macros::{
    silk_add_rshift32, silk_fix_const, silk_lshift_sat32, silk_max_32, silk_min_32, silk_min_int,
    silk_rshift, silk_smulwb, silk_sub_rshift32,
};
use crate::silk::nsq::NsqEncParams;
use crate::silk::sigproc::silk_lp_variable_cutoff;
use crate::silk::structs::SilkNsqState;
use crate::silk::tuning_parameters::{LBRR_SPEECH_ACTIVITY_THRES, SPEECH_ACTIVITY_DTX_THRES};
use crate::silk::vad::silk_vad_get_sa_q8_input_buf;

const MNSF: usize = MAX_NB_SUBFR as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

/// Size of the local `res_pitch` buffer: `2 * MAX_FRAME_LENGTH + LA_PITCH_MAX`.
const RES_PITCH_LEN: usize = 2 * MFL + LA_PITCH_MAX as usize;

/// Size of `ec_buf_copy` in `silk_encode_frame_FLP`.
const EC_BUF_COPY_LEN: usize = 1275;

/// Port of `silk/float/encode_frame_FLP.c:silk_encode_do_VAD_FLP` — voice activity detection
/// and conversion of the speech activity into VAD and DTX flags.
pub fn silk_encode_do_vad_flp(ps_enc: &mut SilkEncoderStateFlp, activity: i32) {
    let activity_threshold = silk_fix_const(SPEECH_ACTIVITY_DTX_THRES as f64, 8);
    let s = &mut ps_enc.s_cmn;

    // Voice Activity Detection
    silk_vad_get_sa_q8_input_buf(s);
    // If Opus VAD is inactive and Silk VAD is active: lower Silk VAD to just under the threshold
    if activity == VAD_NO_ACTIVITY && s.speech_activity_q8 >= activity_threshold {
        s.speech_activity_q8 = activity_threshold - 1;
    }

    // Convert speech activity into VAD and DTX flags
    let nfe = s.n_frames_encoded as usize;
    if s.speech_activity_q8 < activity_threshold {
        s.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        s.no_speech_counter += 1;
        if s.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            s.in_dtx = 0;
        } else if s.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX {
            s.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            s.in_dtx = 0;
        }
        s.vad_flags[nfe] = 0;
    } else {
        s.no_speech_counter = 0;
        s.in_dtx = 0;
        s.indices.signal_type = TYPE_UNVOICED as i8;
        s.vad_flags[nfe] = 1;
    }
}

/// Port of `silk/float/encode_frame_FLP.c:silk_encode_frame_FLP` — encode one frame.
///
/// Always returns 0 (C `ret` is never changed). `ps_range_enc` is not touched when the state's
/// `prefill_flag` is set (the Opus encoder passes `NULL` in that case).
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_encode_frame_flp(
    ps_enc: &mut SilkEncoderStateFlp,
    pn_bytes_out: &mut i32,
    ps_range_enc: &mut EcEnc<'_>,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut s_enc_ctrl = SilkEncoderControlFlp::new();
    let ret = 0;
    let mut res_pitch = [0f32; RES_PITCH_LEN];
    let mut p_gains_q16 = [0i32; MNSF];
    let mut gain_lock = [false; MNSF];
    let mut best_gain_mult = [0i16; MNSF];
    let mut best_sum = [0i32; MNSF];

    // For CBR, 5 bits below budget is close enough. For VBR, allow up to 25% below the cap if we
    // initially busted the budget.
    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };
    // This is totally unnecessary but many compilers (including gcc) are too dumb to realise it
    let mut last_gain_index_copy2: i8 = 0;
    let mut n_bits_lower: i32 = 0;
    let mut n_bits_upper: i32 = 0;
    let mut gain_mult_lower: i32 = 0;
    let mut gain_mult_upper: i32 = 0;

    {
        let s = &mut ps_enc.s_cmn;
        s.indices.seed = (s.frame_counter & 3) as i8;
        s.frame_counter += 1;
    }

    // Set up Input Pointers, and insert frame in input buffer
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let fs_khz = ps_enc.s_cmn.fs_khz as usize;
    // pointers aligned with start of frame to encode
    let x_frame = ltp_mem_length; // start of frame to encode (index into x_buf)
    let res_pitch_frame = ltp_mem_length; // start of pitch LPC residual frame

    // Ensure smooth bandwidth transitions
    {
        let s = &mut ps_enc.s_cmn;
        silk_lp_variable_cutoff(&mut s.s_lp, &mut s.input_buf[1..], frame_length);
    }

    // Copy new frame to front of input buffer
    let la = LA_SHAPE_MS as usize * fs_khz;
    silk_short2float_array(
        &mut ps_enc.x_buf[x_frame + la..],
        &ps_enc.s_cmn.input_buf[1..],
        frame_length,
    );

    // Add tiny signal to avoid high CPU load from denormalized floating point numbers
    for i in 0..8 {
        ps_enc.x_buf[x_frame + la + i * (frame_length >> 3)] +=
            (1 - (i as i32 & 2)) as f32 * 1e-6f32;
    }

    if ps_enc.s_cmn.prefill_flag == 0 {
        let SilkEncoderStateFlp {
            s_cmn,
            s_shape,
            x_buf,
            ltp_corr,
        } = ps_enc;

        // Find pitch lags, initial LPC analysis
        silk_find_pitch_lags_flp(
            s_cmn,
            ltp_corr,
            &mut s_enc_ctrl,
            &mut res_pitch,
            x_buf,
            x_frame,
        );

        // Noise shape analysis
        silk_noise_shape_analysis_flp(
            s_cmn,
            s_shape,
            *ltp_corr,
            &mut s_enc_ctrl,
            &res_pitch[res_pitch_frame..],
            x_buf,
            x_frame,
        );

        // Find linear prediction coefficients (LPC + LTP)
        silk_find_pred_coefs_flp(
            s_cmn,
            &mut s_enc_ctrl,
            &res_pitch,
            res_pitch_frame,
            x_buf,
            x_frame,
            cond_coding,
        );

        // Process gains
        silk_process_gains_flp(s_cmn, s_shape, &mut s_enc_ctrl, cond_coding);

        // Low Bitrate Redundant Encoding
        silk_lbrr_encode_flp(ps_enc, &mut s_enc_ctrl, cond_coding);

        let SilkEncoderStateFlp {
            s_cmn,
            s_shape,
            x_buf,
            ..
        } = ps_enc;
        let nb_subfr = s_cmn.nb_subfr as usize;
        let subfr_length = s_cmn.subfr_length as usize;

        // Loop over quantizer and entroy coding to control bitrate
        let max_iter = 6;
        let mut gain_mult_q8: i16 = silk_fix_const(1.0, 8) as i16;
        let mut found_lower = 0i32;
        let mut found_upper = 0i32;
        let mut gains_id = silk_gains_id(&s_cmn.indices.gains_indices, nb_subfr);
        let mut gains_id_lower = -1i32;
        let mut gains_id_upper = -1i32;
        // Copy part of the input state
        let s_range_enc_copy = ps_range_enc.snapshot();
        let mut s_range_enc_copy2 = s_range_enc_copy;
        let mut s_nsq_copy: [SilkNsqState; 2] = [s_cmn.s_nsq.clone(), SilkNsqState::new()];
        let seed_copy = s_cmn.indices.seed;
        let ec_prev_lag_index_copy = s_cmn.ec_prev_lag_index;
        let ec_prev_signal_type_copy = s_cmn.ec_prev_signal_type;
        let mut ec_buf_copy = [0u8; EC_BUF_COPY_LEN];
        let mut iter = 0i32;
        loop {
            let n_bits;
            if gains_id == gains_id_lower {
                n_bits = n_bits_lower;
            } else if gains_id == gains_id_upper {
                n_bits = n_bits_upper;
            } else {
                // Restore part of the input state
                if iter > 0 {
                    ps_range_enc.restore(&s_range_enc_copy);
                    s_cmn.s_nsq.clone_from(&s_nsq_copy[0]);
                    s_cmn.indices.seed = seed_copy;
                    s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                    s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
                }

                // Noise shaping quantization
                let params = NsqEncParams::from_enc(s_cmn);
                silk_nsq_wrapper_flp(
                    &params,
                    &s_enc_ctrl,
                    &mut s_cmn.indices,
                    &mut s_cmn.s_nsq,
                    &mut s_cmn.pulses,
                    &x_buf[x_frame..],
                );

                if iter == max_iter && found_lower == 0 {
                    s_range_enc_copy2 = ps_range_enc.snapshot();
                }

                // Encode Parameters
                let nfe = s_cmn.n_frames_encoded as usize;
                silk_encode_indices(s_cmn, ps_range_enc, nfe, 0, cond_coding);

                // Encode Excitation Signal
                silk_encode_pulses(
                    ps_range_enc,
                    s_cmn.indices.signal_type as i32,
                    s_cmn.indices.quant_offset_type as i32,
                    &mut s_cmn.pulses,
                    s_cmn.frame_length,
                );

                let mut nb = ps_range_enc.tell();

                // If we still bust after the last iteration, do some damage control.
                if iter == max_iter && found_lower == 0 && nb > max_bits {
                    ps_range_enc.restore(&s_range_enc_copy2);

                    // Keep gains the same as the last frame.
                    s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                    for i in 0..nb_subfr {
                        s_cmn.indices.gains_indices[i] = 4;
                    }
                    if cond_coding != CODE_CONDITIONALLY {
                        s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev;
                    }
                    s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                    s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
                    // Clear all pulses.
                    s_cmn.pulses[..frame_length].fill(0);

                    silk_encode_indices(s_cmn, ps_range_enc, nfe, 0, cond_coding);

                    silk_encode_pulses(
                        ps_range_enc,
                        s_cmn.indices.signal_type as i32,
                        s_cmn.indices.quant_offset_type as i32,
                        &mut s_cmn.pulses,
                        s_cmn.frame_length,
                    );

                    nb = ps_range_enc.tell();
                }

                n_bits = nb;
                if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                    break;
                }
            }

            if iter == max_iter {
                if found_lower != 0 && (gains_id == gains_id_lower || n_bits > max_bits) {
                    // Restore output state from earlier iteration that did meet the bitrate
                    // budget
                    ps_range_enc.restore(&s_range_enc_copy2);
                    let offs = ps_range_enc.offs as usize; // sRangeEnc_copy2.offs
                    celt_assert!(offs <= EC_BUF_COPY_LEN);
                    ps_range_enc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                    s_cmn.s_nsq.clone_from(&s_nsq_copy[1]);
                    s_shape.last_gain_index = last_gain_index_copy2;
                }
                break;
            }

            if n_bits > max_bits {
                if found_lower == 0 && iter >= 2 {
                    // Adjust the quantizer's rate/distortion tradeoff and discard previous
                    // "upper" results
                    s_enc_ctrl.lambda = silk_max_float(s_enc_ctrl.lambda * 1.5f32, 1.5f32);
                    // Reducing dithering can help us hit the target.
                    s_cmn.indices.quant_offset_type = 0;
                    found_upper = 0;
                    gains_id_upper = -1;
                } else {
                    found_upper = 1;
                    n_bits_upper = n_bits;
                    gain_mult_upper = gain_mult_q8 as i32;
                    gains_id_upper = gains_id;
                }
            } else if n_bits < max_bits - bits_margin {
                found_lower = 1;
                n_bits_lower = n_bits;
                gain_mult_lower = gain_mult_q8 as i32;
                if gains_id != gains_id_lower {
                    gains_id_lower = gains_id;
                    // Copy part of the output state
                    s_range_enc_copy2 = ps_range_enc.snapshot();
                    let offs = ps_range_enc.offs as usize;
                    celt_assert!(offs <= EC_BUF_COPY_LEN);
                    ec_buf_copy[..offs].copy_from_slice(&ps_range_enc.buf[..offs]);
                    s_nsq_copy[1].clone_from(&s_cmn.s_nsq);
                    last_gain_index_copy2 = s_shape.last_gain_index;
                }
            } else {
                // Close enough
                break;
            }

            if found_lower == 0 && n_bits > max_bits {
                for i in 0..nb_subfr {
                    let mut sum = 0i32;
                    for j in i * subfr_length..(i + 1) * subfr_length {
                        sum += (s_cmn.pulses[j] as i32).abs();
                    }
                    if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                        best_sum[i] = sum;
                        best_gain_mult[i] = gain_mult_q8;
                    } else {
                        gain_lock[i] = true;
                    }
                }
            }
            if (found_lower & found_upper) == 0 {
                // Adjust gain according to high-rate rate/distortion curve
                if n_bits > max_bits {
                    gain_mult_q8 = silk_min_32(1024, gain_mult_q8 as i32 * 3 / 2) as i16;
                } else {
                    gain_mult_q8 = silk_max_32(64, gain_mult_q8 as i32 * 4 / 5) as i16;
                }
            } else {
                // Adjust gain by interpolating
                gain_mult_q8 = (gain_mult_lower
                    + ((gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower))
                        / (n_bits_upper - n_bits_lower)) as i16;
                // New gain multiplier must be between 25% and 75% of old range (note that
                // gainMult_upper < gainMult_lower)
                if gain_mult_q8 as i32
                    > silk_add_rshift32(gain_mult_lower, gain_mult_upper - gain_mult_lower, 2)
                {
                    gain_mult_q8 =
                        silk_add_rshift32(gain_mult_lower, gain_mult_upper - gain_mult_lower, 2)
                            as i16;
                } else if (gain_mult_q8 as i32)
                    < silk_sub_rshift32(gain_mult_upper, gain_mult_upper - gain_mult_lower, 2)
                {
                    gain_mult_q8 =
                        silk_sub_rshift32(gain_mult_upper, gain_mult_upper - gain_mult_lower, 2)
                            as i16;
                }
            }

            for i in 0..nb_subfr {
                let tmp = if gain_lock[i] {
                    best_gain_mult[i]
                } else {
                    gain_mult_q8
                };
                p_gains_q16[i] =
                    silk_lshift_sat32(silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp as i32), 8);
            }

            // Quantize gains
            s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
            silk_gains_quant(
                &mut s_cmn.indices.gains_indices,
                &mut p_gains_q16,
                &mut s_shape.last_gain_index,
                (cond_coding == CODE_CONDITIONALLY) as i32,
                nb_subfr,
            );

            // Unique identifier of gains vector
            gains_id = silk_gains_id(&s_cmn.indices.gains_indices, nb_subfr);

            // Overwrite unquantized gains with quantized gains and convert back to Q0 from Q16
            for i in 0..nb_subfr {
                s_enc_ctrl.gains[i] = p_gains_q16[i] as f32 / 65536.0f32;
            }
            iter += 1;
        }
    }

    // Update input buffer
    ps_enc
        .x_buf
        .copy_within(frame_length..frame_length + ltp_mem_length + la, 0);

    // Exit without entropy coding
    if ps_enc.s_cmn.prefill_flag != 0 {
        // No payload
        *pn_bytes_out = 0;
        return ret;
    }

    let s = &mut ps_enc.s_cmn;
    // Parameters needed for next frame
    s.prev_lag = s_enc_ctrl.pitch_l[s.nb_subfr as usize - 1];
    s.prev_signal_type = s.indices.signal_type;

    // Finalize payload
    s.first_frame_after_reset = 0;
    // Payload size
    *pn_bytes_out = silk_rshift(ps_range_enc.tell() + 7, 3);

    ret
}

/// Port of the static `encode_frame_FLP.c:silk_LBRR_encode_FLP` — Low-Bitrate Redundancy (LBRR)
/// encoding. Reuses all parameters but encodes the excitation at a lower bitrate. The input
/// signal `xfw` is `x_frame` (`ps_enc.x_buf[ltp_mem_length..]`).
fn silk_lbrr_encode_flp(
    ps_enc: &mut SilkEncoderStateFlp,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
    cond_coding: i32,
) {
    let SilkEncoderStateFlp {
        s_cmn,
        s_shape,
        x_buf,
        ..
    } = ps_enc;
    let nfe = s_cmn.n_frames_encoded as usize;
    let nb_subfr = s_cmn.nb_subfr as usize;
    let mut gains_q16 = [0i32; MNSF];

    // Control use of inband LBRR
    if s_cmn.lbrr_enabled != 0
        && s_cmn.speech_activity_q8 > silk_fix_const(LBRR_SPEECH_ACTIVITY_THRES as f64, 8)
    {
        s_cmn.lbrr_flags[nfe] = 1;

        // Copy noise shaping quantizer state and quantization indices from regular encoding
        let mut s_nsq_lbrr = s_cmn.s_nsq.clone();
        s_cmn.indices_lbrr[nfe] = s_cmn.indices;

        // Save original gains
        let temp_gains = ps_enc_ctrl.gains;

        if nfe == 0 || s_cmn.lbrr_flags[nfe - 1] == 0 {
            // First frame in packet or previous frame not LBRR coded
            s_cmn.lbrr_prev_last_gain_index = s_shape.last_gain_index;

            // Increase Gains to get target LBRR rate
            let psi = &mut s_cmn.indices_lbrr[nfe];
            psi.gains_indices[0] = (psi.gains_indices[0] as i32 + s_cmn.lbrr_gain_increases) as i8;
            psi.gains_indices[0] =
                silk_min_int(psi.gains_indices[0] as i32, N_LEVELS_QGAIN - 1) as i8;
        }

        // Decode to get gains in sync with decoder
        silk_gains_dequant(
            &mut gains_q16,
            &s_cmn.indices_lbrr[nfe].gains_indices,
            &mut s_cmn.lbrr_prev_last_gain_index,
            (cond_coding == CODE_CONDITIONALLY) as i32,
            nb_subfr,
        );

        // Overwrite unquantized gains with quantized gains and convert back to Q0 from Q16
        for k in 0..nb_subfr {
            ps_enc_ctrl.gains[k] = gains_q16[k] as f32 * (1.0f32 / 65536.0f32);
        }

        // Noise shaping quantization
        let params = NsqEncParams::from_enc(s_cmn);
        silk_nsq_wrapper_flp(
            &params,
            ps_enc_ctrl,
            &mut s_cmn.indices_lbrr[nfe],
            &mut s_nsq_lbrr,
            &mut s_cmn.pulses_lbrr[nfe],
            &x_buf[s_cmn.ltp_mem_length as usize..],
        );

        // Restore original gains
        ps_enc_ctrl.gains[..nb_subfr].copy_from_slice(&temp_gains[..nb_subfr]);
    }
}

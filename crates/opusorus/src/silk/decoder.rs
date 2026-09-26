//! Port of the SILK decoder: silk/dec_API.c (the `silk_decoder` super-struct and
//! `silk_Get_Decoder_Size` / `silk_InitDecoder` / `silk_ResetDecoder` / `silk_Decode`),
//! silk/init_decoder.c, silk/decoder_set_fs.c, silk/decode_frame.c, silk/decode_core.c,
//! silk/decode_indices.c, silk/decode_parameters.c and silk/stereo_MS_to_LR.c (the decoder
//! declarations of silk/API.h and silk/main.h).
//!
//! DNN (features `deep-plc` / `osce`, C `ENABLE_DEEP_PLC` / `ENABLE_OSCE` + `ENABLE_OSCE_BWE`):
//! * [`SilkDecoder::silk_decode_dnn`] is `silk_Decode` with the C `lpcnet` argument (the Opus
//!   decoder's LPCNet PLC state, used by the SILK PLC of channel 0) and the OSCE fields of
//!   `silk_DecControlStruct` (`SilkOsceControl`; `silk/structs.rs` keeps the base struct).
//!   [`SilkDecoder::silk_decode`] is the same call with C `lpcnet == NULL` and zeroed OSCE
//!   control fields.
//! * The OSCE model (`silk_decoder.osce_model`) and the per-channel `silk_decoder_state.osce` /
//!   `osce_bwe` states live in [`SilkDecoder`] (`osce_model`, `osce`, `osce_bwe`) next to
//!   `channel_state`; `silk_init_decoder` / `silk_reset_decoder` of a channel through the super
//!   struct reset them as C does. The OSCE model is not compiled in: [`SilkDecoder::init`]
//!   leaves it unloaded (upstream `USE_WEIGHTS_FILE`) until
//!   [`SilkDecoder::load_osce_models`] binds a weight blob.

use crate::celt::arch::{OpusRes, int16tores};
use crate::celt::entdec::EcDec;
use crate::silk::cng::{silk_cng, silk_cng_reset};
use crate::silk::coding::{
    silk_decode_pitch, silk_decode_pulses, silk_gains_dequant, silk_stereo_decode_mid_only,
    silk_stereo_decode_pred,
};
use crate::silk::define::{
    BWE_AFTER_LOSS_Q16, CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    DECODER_NUM_CHANNELS, LTP_MEM_LENGTH_MS, LTP_ORDER, MAX_API_FS_KHZ, MAX_FRAME_LENGTH,
    MAX_FRAME_LENGTH_MS, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SUB_FRAME_LENGTH, MIN_LPC_ORDER,
    NLSF_QUANT_MAX_AMPLITUDE, QUANT_LEVEL_ADJUST_Q10, STEREO_INTERP_LEN_MS, SUB_FRAME_LENGTH_MS,
    TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};
use crate::silk::errors::{
    SILK_DEC_INVALID_FRAME_SIZE, SILK_DEC_INVALID_SAMPLING_FREQUENCY, SILK_NO_ERROR,
};
use crate::silk::macros::{
    silk_add_lshift32, silk_add_sat32, silk_add32_ovflw, silk_div32, silk_div32_16,
    silk_div32_varq, silk_fix_const, silk_inverse32_varq, silk_lshift, silk_lshift_sat32, silk_min,
    silk_mul, silk_rand, silk_rshift, silk_rshift_round, silk_sat16, silk_smlawb,
    silk_smlawb_chain, silk_smulbb, silk_smulwb, silk_smulww,
};
use crate::silk::nlsf::{silk_nlsf_decode, silk_nlsf_unpack, silk_nlsf2a};
use crate::silk::plc::{silk_plc, silk_plc_glue_frames, silk_plc_reset};
use crate::silk::resampler::{silk_resampler, silk_resampler_init};
use crate::silk::sigproc::{silk_bwexpander, silk_lpc_analysis_filter};
use crate::silk::structs::{
    FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, SilkDecControlStruct,
    SilkDecoderControl, SilkDecoderState, StereoDecState,
};
use crate::silk::tables::{
    SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF, SILK_LBRR_FLAGS_ICDF_PTR, SILK_LTP_GAIN_ICDF_PTRS,
    SILK_LTP_PER_INDEX_ICDF, SILK_LTP_VQ_PTRS_Q7, SILK_LTPSCALE_ICDF, SILK_LTPSCALES_TABLE_Q14,
    SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_PITCH_CONTOUR_10_MS_ICDF, SILK_PITCH_CONTOUR_10_MS_NB_ICDF, SILK_PITCH_CONTOUR_ICDF,
    SILK_PITCH_CONTOUR_NB_ICDF, SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF,
    SILK_QUANTIZATION_OFFSETS_Q10, SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF,
    SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF,
};

#[cfg(feature = "deep-plc")]
use crate::dnn::lpcnet_plc::LpcnetPlcState;
#[cfg(feature = "osce")]
use crate::dnn::osce::{
    OSCE_DEFAULT_METHOD, OSCE_MODE_HYBRID, OSCE_MODE_SILK_BBWE, OSCE_MODE_SILK_ONLY, OsceModel,
    SilkOsceBweStruct, SilkOsceStruct, osce_bwe, osce_bwe_reset, osce_enhance_frame,
    osce_load_models, osce_reset,
};
#[cfg(feature = "osce")]
use crate::dnn::osce_features::{OsceDecInfo, osce_bwe_cross_fade_10ms};

const MFL: usize = MAX_FRAME_LENGTH as usize;
const MSFL: usize = MAX_SUB_FRAME_LENGTH as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const LTPO: usize = LTP_ORDER as usize;
const NCH: usize = DECODER_NUM_CHANNELS as usize;
/// Largest `*nSamplesOut` of one `silk_Decode` call (one 20 ms frame at the highest API rate).
const MAX_SAMPLES_OUT: usize = (MAX_API_FS_KHZ * MAX_FRAME_LENGTH_MS) as usize;

// ---------------------------------------------------------------------------------------------
// init_decoder.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/init_decoder.c:silk_reset_decoder: resets a channel decoder state (clears
/// everything from `prev_gain_Q16` on, then resets the CNG and PLC states). Returns 0.
pub fn silk_reset_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    // Clear the entire encoder state, except anything copied
    ps_dec.reset();

    // Used to deactivate LSF interpolation
    ps_dec.first_frame_after_reset = 1;
    ps_dec.prev_gain_q16 = 65536;
    // `arch = opus_select_arch()`: dropped.

    // Reset CNG state
    silk_cng_reset(ps_dec);

    // Reset PLC state
    silk_plc_reset(ps_dec);

    // ENABLE_OSCE: `osce_reset(&psDec->osce, OSCE_DEFAULT_METHOD)` is done by the owner of the
    // OSCE state ([`SilkDecoder`]).

    0
}

/// Port of silk/init_decoder.c:silk_init_decoder: clears the whole channel decoder state and
/// resets it. Returns 0.
pub fn silk_init_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    // Clear the entire encoder state, except anything copied
    *ps_dec = SilkDecoderState::new();

    silk_reset_decoder(ps_dec);

    0
}

// ---------------------------------------------------------------------------------------------
// decoder_set_fs.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decoder_set_fs.c:silk_decoder_set_fs: sets the decoder sampling rate
/// (`fs_khz` is 8, 12 or 16) and the API rate. Returns the resampler init status (0 or -1).
pub fn silk_decoder_set_fs(ps_dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32 {
    let mut ret = 0;

    debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
    debug_assert!(ps_dec.nb_subfr == MAX_NB_SUBFR || ps_dec.nb_subfr == MAX_NB_SUBFR / 2);

    // New (sub)frame length
    ps_dec.subfr_length = silk_smulbb(SUB_FRAME_LENGTH_MS, fs_khz);
    let frame_length = silk_smulbb(ps_dec.nb_subfr, ps_dec.subfr_length);

    // Initialize resampler when switching internal or external sampling frequency
    if ps_dec.fs_khz != fs_khz || ps_dec.fs_api_hz != fs_api_hz {
        // Initialize the resampler for dec_API.c preparing resampling from fs_kHz to API_fs_Hz
        ret += silk_resampler_init(
            &mut ps_dec.resampler_state,
            silk_smulbb(fs_khz, 1000),
            fs_api_hz,
            0,
        );

        ps_dec.fs_api_hz = fs_api_hz;
    }

    if ps_dec.fs_khz != fs_khz || frame_length != ps_dec.frame_length {
        if fs_khz == 8 {
            if ps_dec.nb_subfr == MAX_NB_SUBFR {
                ps_dec.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_dec.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if ps_dec.nb_subfr == MAX_NB_SUBFR {
            ps_dec.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            ps_dec.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
        if ps_dec.fs_khz != fs_khz {
            ps_dec.ltp_mem_length = silk_smulbb(LTP_MEM_LENGTH_MS, fs_khz);
            if fs_khz == 8 || fs_khz == 12 {
                ps_dec.lpc_order = MIN_LPC_ORDER;
                ps_dec.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
            } else {
                ps_dec.lpc_order = MAX_LPC_ORDER;
                ps_dec.ps_nlsf_cb = &SILK_NLSF_CB_WB;
            }
            if fs_khz == 16 {
                ps_dec.pitch_lag_low_bits_icdf = &SILK_UNIFORM8_ICDF;
            } else if fs_khz == 12 {
                ps_dec.pitch_lag_low_bits_icdf = &SILK_UNIFORM6_ICDF;
            } else if fs_khz == 8 {
                ps_dec.pitch_lag_low_bits_icdf = &SILK_UNIFORM4_ICDF;
            } else {
                // unsupported sampling rate
                debug_assert!(false, "unsupported sampling rate");
            }
            ps_dec.first_frame_after_reset = 1;
            ps_dec.lag_prev = 100;
            ps_dec.last_gain_index = 10;
            ps_dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            ps_dec.out_buf.fill(0);
            ps_dec.s_lpc_q14_buf.fill(0);
        }

        ps_dec.fs_khz = fs_khz;
        ps_dec.frame_length = frame_length;
    }

    // Check that settings are valid
    debug_assert!(ps_dec.frame_length > 0 && ps_dec.frame_length <= MAX_FRAME_LENGTH);

    ret
}

// ---------------------------------------------------------------------------------------------
// decode_frame.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_frame.c:silk_decode_frame: decodes one frame (or conceals it) into
/// `p_out` (`frame_length` samples) and writes the frame length to `p_n`. Returns 0.
///
/// DNN: `lpcnet` is the C `ENABLE_DEEP_PLC` argument (`None` = `NULL`), `osce` the
/// `ENABLE_OSCE` model and this channel's enhancer state.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn silk_decode_frame(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut EcDec<'_>,
    p_out: &mut [i16],
    p_n: &mut i32,
    lost_flag: i32,
    cond_coding: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LpcnetPlcState>,
    #[cfg(feature = "osce")] osce: (&OsceModel, &mut SilkOsceStruct),
) -> i32 {
    let mut ps_dec_ctrl = SilkDecoderControl::default();
    let ret = 0;

    let l = ps_dec.frame_length;
    ps_dec_ctrl.ltp_scale_q14 = 0;

    // Safety checks
    debug_assert!(l > 0 && l <= MAX_FRAME_LENGTH);
    let lu = l as usize;

    if lost_flag == FLAG_DECODE_NORMAL
        || (lost_flag == FLAG_DECODE_LBRR
            && ps_dec.lbrr_flags[ps_dec.n_frames_decoded as usize] == 1)
    {
        // `(L + SHELL_CODEC_FRAME_LENGTH - 1) & ~(SHELL_CODEC_FRAME_LENGTH - 1)` <= MAX_FRAME_LENGTH
        let mut pulses = [0i16; MFL];
        #[cfg(feature = "osce")]
        let ec_start = ps_range_dec.tell();

        // Decode quantization indices of side info
        let frame_index = ps_dec.n_frames_decoded;
        silk_decode_indices(ps_dec, ps_range_dec, frame_index, lost_flag, cond_coding);

        // Decode quantization indices of excitation
        silk_decode_pulses(
            ps_range_dec,
            &mut pulses,
            ps_dec.indices.signal_type as i32,
            ps_dec.indices.quant_offset_type as i32,
            ps_dec.frame_length,
        );

        // Decode parameters and pulse signal
        silk_decode_parameters(ps_dec, &mut ps_dec_ctrl, cond_coding);

        // Run inverse NSQ
        silk_decode_core(ps_dec, &mut ps_dec_ctrl, p_out, &pulses);

        // Update output buffer.
        update_out_buf(ps_dec, p_out);

        // Run SILK enhancer
        #[cfg(feature = "osce")]
        {
            let info = OsceDecInfo::from_decoder(ps_dec);
            let num_bits = ps_range_dec.tell() - ec_start;
            osce_enhance_frame(osce.0, osce.1, info, &ps_dec_ctrl, p_out, num_bits);
        }

        // Update PLC state
        silk_plc(
            ps_dec,
            &mut ps_dec_ctrl,
            p_out,
            0,
            #[cfg(feature = "deep-plc")]
            lpcnet,
        );

        ps_dec.loss_cnt = 0;
        ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
        debug_assert!(ps_dec.prev_signal_type >= 0 && ps_dec.prev_signal_type <= 2);

        // A frame has been decoded without errors
        ps_dec.first_frame_after_reset = 0;
    } else {
        // Handle packet loss by extrapolation
        silk_plc(
            ps_dec,
            &mut ps_dec_ctrl,
            p_out,
            1,
            #[cfg(feature = "deep-plc")]
            lpcnet,
        );

        #[cfg(feature = "osce")]
        {
            let method = osce.1.method;
            osce_reset(osce.1, method);
        }

        // Update output buffer.
        update_out_buf(ps_dec, p_out);
    }

    // Comfort noise generation / estimation
    silk_cng(ps_dec, &ps_dec_ctrl, p_out, lu);

    // Ensure smooth connection of extrapolated and good frames
    silk_plc_glue_frames(ps_dec, p_out, lu);

    // Update some decoder state variables
    ps_dec.lag_prev = ps_dec_ctrl.pitch_l[(ps_dec.nb_subfr - 1) as usize];

    // Set output frame length
    *p_n = l;

    ret
}

/// The "Update output buffer" block of silk_decode_frame: shifts `outBuf` by one frame and
/// appends the new frame.
fn update_out_buf(ps_dec: &mut SilkDecoderState, p_out: &[i16]) {
    debug_assert!(ps_dec.ltp_mem_length >= ps_dec.frame_length);
    let frame_length = ps_dec.frame_length as usize;
    let mv_len = (ps_dec.ltp_mem_length - ps_dec.frame_length) as usize;
    ps_dec
        .out_buf
        .copy_within(frame_length..frame_length + mv_len, 0);
    ps_dec.out_buf[mv_len..mv_len + frame_length].copy_from_slice(&p_out[..frame_length]);
}

// ---------------------------------------------------------------------------------------------
// decode_core.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_core.c:silk_decode_core: core decoder, performs the inverse NSQ
/// operation (LTP + LPC synthesis). Writes `frame_length` samples to `xq`.
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn silk_decode_core(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;
    let frame_length = ps_dec.frame_length as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;
    let mut lag: i32 = 0;
    let mut a_q12_tmp = [0i16; MLPC];
    let mut s_ltp = [0i16; MFL];
    let mut s_ltp_q15 = [0i32; 2 * MFL];
    let mut res_q14 = [0i32; MSFL];
    let mut s_lpc_q14 = [0i32; MSFL + MLPC];

    debug_assert!(ps_dec.prev_gain_q16 != 0);

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_dec.indices.signal_type >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    let nlsf_interpolation_flag = i32::from((ps_dec.indices.nlsf_interp_coef_q2 as i32) < 1 << 2);

    // Decode excitation
    let mut rand_seed = ps_dec.indices.seed as i32;
    for i in 0..frame_length {
        rand_seed = silk_rand(rand_seed);
        let mut e = silk_lshift(pulses[i] as i32, 14);
        if e > 0 {
            e -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if e < 0 {
            e += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        e += offset_q10 << 4;
        if rand_seed < 0 {
            e = -e;
        }
        ps_dec.exc_q14[i] = e;

        rand_seed = silk_add32_ovflw(rand_seed, pulses[i] as i32);
    }

    // Copy LPC state
    s_lpc_q14[..MLPC].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    let mut pexc = 0usize; // pexc_Q14 = psDec->exc_Q14
    let mut pxq = 0usize; // pxq = xq
    let mut s_ltp_buf_idx = ltp_mem_length;
    // Loop over subframes
    for k in 0..ps_dec.nb_subfr as usize {
        let a_q12 = ps_dec_ctrl.pred_coef_q12[k >> 1];

        // Preload LPC coefficients to array on stack. Gives small performance gain
        a_q12_tmp[..lpc_order].copy_from_slice(&a_q12[..lpc_order]);
        let b_off = k * LTPO; // B_Q14 = &psDecCtrl->LTPCoef_Q14[ k * LTP_ORDER ]
        let mut signal_type = ps_dec.indices.signal_type as i32;

        let gain_q10 = silk_rshift(ps_dec_ctrl.gains_q16[k], 6);
        let mut inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);

        // Calculate gain adjustment factor
        let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
            let gain_adj_q16 = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);

            // Scale short term state
            for i in 0..MLPC {
                s_lpc_q14[i] = silk_smulww(gain_adj_q16, s_lpc_q14[i]);
            }
            gain_adj_q16
        } else {
            1i32 << 16
        };

        // Save inv_gain
        debug_assert!(inv_gain_q31 != 0);
        ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

        // Avoid abrupt transition from voiced PLC to unvoiced normal decoding
        if ps_dec.loss_cnt != 0
            && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < (MAX_NB_SUBFR / 2) as usize
        {
            ps_dec_ctrl.ltp_coef_q14[b_off..b_off + LTPO].fill(0);
            ps_dec_ctrl.ltp_coef_q14[b_off + LTPO / 2] = silk_fix_const(0.25, 14) as i16;

            signal_type = TYPE_VOICED;
            ps_dec_ctrl.pitch_l[k] = ps_dec.lag_prev;
        }

        if signal_type == TYPE_VOICED {
            // Voiced
            lag = ps_dec_ctrl.pitch_l[k];

            // Re-whitening
            if k == 0 || (k == 2 && nlsf_interpolation_flag != 0) {
                // Rewhiten with new A coefs
                let start_idx = ps_dec.ltp_mem_length - lag - ps_dec.lpc_order - LTP_ORDER / 2;
                debug_assert!(start_idx > 0);
                let start_idx = start_idx as usize;

                if k == 2 {
                    ps_dec.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                        .copy_from_slice(&xq[..2 * subfr_length]);
                }

                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &ps_dec.out_buf[start_idx + k * subfr_length..],
                    &a_q12,
                    ltp_mem_length - start_idx,
                    lpc_order,
                );

                // After rewhitening the LTP state is unscaled
                if k == 0 {
                    // Do LTP downscaling to reduce inter-packet dependency
                    inv_gain_q31 =
                        silk_lshift(silk_smulwb(inv_gain_q31, ps_dec_ctrl.ltp_scale_q14), 2);
                }
                for i in 0..(lag + LTP_ORDER / 2) as usize {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        silk_smulwb(inv_gain_q31, s_ltp[ltp_mem_length - i - 1] as i32);
                }
            } else {
                // Update LTP state when Gain changes
                if gain_adj_q16 != 1i32 << 16 {
                    for i in 0..(lag + LTP_ORDER / 2) as usize {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            silk_smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                    }
                }
            }
        }

        let b_q14 = &ps_dec_ctrl.ltp_coef_q14[b_off..b_off + LTPO];
        // Long-term prediction
        let pres_q14: &[i32] = if signal_type == TYPE_VOICED {
            // Set up pointer
            let p0 = s_ltp_buf_idx - lag as usize + LTPO / 2;
            for i in 0..subfr_length {
                let p = p0 + i;
                // Unrolled loop
                // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
                let mut ltp_pred_q13: i32 = 2;
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[p], b_q14[0] as i32);
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[p - 1], b_q14[1] as i32);
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[p - 2], b_q14[2] as i32);
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[p - 3], b_q14[3] as i32);
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[p - 4], b_q14[4] as i32);

                // Generate LPC excitation
                res_q14[i] = silk_add_lshift32(ps_dec.exc_q14[pexc + i], ltp_pred_q13, 1);

                // Update states
                s_ltp_q15[s_ltp_buf_idx] = silk_lshift(res_q14[i], 1);
                s_ltp_buf_idx += 1;
            }
            &res_q14[..subfr_length]
        } else {
            &ps_dec.exc_q14[pexc..pexc + subfr_length]
        };

        // Short-term prediction
        debug_assert!(lpc_order == 10 || lpc_order == 16);
        let xq_sub = &mut xq[pxq..pxq + subfr_length];
        if lpc_order == 16 {
            decode_core_lpc_synthesis::<16>(&mut s_lpc_q14, &a_q12_tmp, pres_q14, xq_sub, gain_q10);
        } else {
            decode_core_lpc_synthesis::<10>(&mut s_lpc_q14, &a_q12_tmp, pres_q14, xq_sub, gain_q10);
        }

        // Update LPC filter state
        s_lpc_q14.copy_within(subfr_length..subfr_length + MLPC, 0);
        pexc += subfr_length;
        pxq += subfr_length;
    }

    // Save LPC state
    ps_dec.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MLPC]);
}

/// The short-term prediction loop of silk_decode_core for one subframe
/// (`xq.len() == subfr_length`), specialised on the LPC order (the C `if( LPC_order == 16 )`
/// inside the loop). Perf: the `silk_SMLAWB` chain goes through [`silk_smlawb_chain`], which
/// is bit-identical but keeps the sample produced by the previous iteration off the serial
/// add chain.
#[inline(always)]
fn decode_core_lpc_synthesis<const ORDER: usize>(
    s_lpc_q14: &mut [i32; MSFL + MLPC],
    a_q12_tmp: &[i16; MLPC],
    pres_q14: &[i32],
    xq: &mut [i16],
    gain_q10: i32,
) {
    let pres_q14 = &pres_q14[..xq.len()];
    for (i, (x, &pres)) in xq.iter_mut().zip(pres_q14).enumerate() {
        // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
        // s_lpc_q14[i + MLPC - ORDER..i + MLPC] = sLPC_Q14[MAX_LPC_ORDER + i - ORDER ..= - 1]
        let lpc_pred_q10 = silk_smlawb_chain(
            (ORDER >> 1) as i32,
            &s_lpc_q14[i + MLPC - ORDER..i + MLPC],
            a_q12_tmp,
        );

        // Add prediction to LPC excitation
        let v = silk_add_sat32(pres, silk_lshift_sat32(lpc_pred_q10, 4));
        s_lpc_q14[MLPC + i] = v;

        // Scale with gain
        *x = silk_sat16(silk_rshift_round(silk_smulww(v, gain_q10), 8)) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// decode_indices.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_indices.c:silk_decode_indices: decodes side-information parameters
/// from the payload into `ps_dec.indices`.
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn silk_decode_indices(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut EcDec<'_>,
    frame_index: i32,
    decode_lbrr: i32,
    cond_coding: i32,
) {
    let mut ec_ix = [0i16; MLPC];
    let mut pred_q8 = [0u8; MLPC];

    // Decode signal type and quantizer offset
    let ix = if decode_lbrr != 0 || ps_dec.vad_flags[frame_index as usize] != 0 {
        ps_range_dec.dec_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
    } else {
        ps_range_dec.dec_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
    };
    ps_dec.indices.signal_type = silk_rshift(ix, 1) as i8;
    ps_dec.indices.quant_offset_type = (ix & 1) as i8;

    // Decode gains
    // First subframe
    if cond_coding == CODE_CONDITIONALLY {
        // Conditional coding
        ps_dec.indices.gains_indices[0] = ps_range_dec.dec_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        // Independent coding, in two stages: MSB bits followed by 3 LSBs
        let msb =
            ps_range_dec.dec_icdf(&SILK_GAIN_ICDF[ps_dec.indices.signal_type as usize], 8) as i32;
        ps_dec.indices.gains_indices[0] = silk_lshift(msb, 3) as i8;
        let lsb = ps_range_dec.dec_icdf(&SILK_UNIFORM8_ICDF, 8) as i8;
        ps_dec.indices.gains_indices[0] =
            (ps_dec.indices.gains_indices[0] as i32 + lsb as i32) as i8;
    }

    // Remaining subframes
    for i in 1..ps_dec.nb_subfr as usize {
        ps_dec.indices.gains_indices[i] = ps_range_dec.dec_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    // Decode LSF Indices
    let cb = ps_dec.ps_nlsf_cb;
    let cb1_off = (ps_dec.indices.signal_type >> 1) as usize * cb.n_vectors as usize;
    ps_dec.indices.nlsf_indices[0] = ps_range_dec.dec_icdf(&cb.cb1_icdf[cb1_off..], 8) as i8;
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        cb,
        ps_dec.indices.nlsf_indices[0] as i32,
    );
    debug_assert!(cb.order as i32 == ps_dec.lpc_order);
    for i in 0..cb.order as usize {
        let mut ix = ps_range_dec.dec_icdf(&cb.ec_icdf[ec_ix[i] as usize..], 8) as i32;
        if ix == 0 {
            ix -= ps_range_dec.dec_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
        } else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix += ps_range_dec.dec_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
        }
        ps_dec.indices.nlsf_indices[i + 1] = (ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    // Decode LSF interpolation factor
    if ps_dec.nb_subfr == MAX_NB_SUBFR {
        ps_dec.indices.nlsf_interp_coef_q2 =
            ps_range_dec.dec_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Decode pitch lags
        // Get lag index
        let mut decode_absolute_lag_index = 1;
        if cond_coding == CODE_CONDITIONALLY && ps_dec.ec_prev_signal_type == TYPE_VOICED {
            // Decode Delta index
            let mut delta_lag_index =
                ps_range_dec.dec_icdf(&SILK_PITCH_DELTA_ICDF, 8) as i16 as i32;
            if delta_lag_index > 0 {
                delta_lag_index -= 9;
                ps_dec.indices.lag_index =
                    (ps_dec.ec_prev_lag_index as i32 + delta_lag_index) as i16;
                decode_absolute_lag_index = 0;
            }
        }
        if decode_absolute_lag_index != 0 {
            // Absolute decoding
            ps_dec.indices.lag_index = (ps_range_dec.dec_icdf(&SILK_PITCH_LAG_ICDF, 8) as i16
                as i32
                * silk_rshift(ps_dec.fs_khz, 1)) as i16;
            let low = ps_range_dec.dec_icdf(ps_dec.pitch_lag_low_bits_icdf, 8) as i16;
            ps_dec.indices.lag_index = (ps_dec.indices.lag_index as i32 + low as i32) as i16;
        }
        ps_dec.ec_prev_lag_index = ps_dec.indices.lag_index;

        // Get contour index
        ps_dec.indices.contour_index = ps_range_dec.dec_icdf(ps_dec.pitch_contour_icdf, 8) as i8;

        // Decode LTP gains
        // Decode PERIndex value
        ps_dec.indices.per_index = ps_range_dec.dec_icdf(&SILK_LTP_PER_INDEX_ICDF, 8) as i8;

        for k in 0..ps_dec.nb_subfr as usize {
            ps_dec.indices.ltp_index[k] = ps_range_dec.dec_icdf(
                SILK_LTP_GAIN_ICDF_PTRS[ps_dec.indices.per_index as usize],
                8,
            ) as i8;
        }

        // Decode LTP scaling
        if cond_coding == CODE_INDEPENDENTLY {
            ps_dec.indices.ltp_scale_index = ps_range_dec.dec_icdf(&SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            ps_dec.indices.ltp_scale_index = 0;
        }
    }
    ps_dec.ec_prev_signal_type = ps_dec.indices.signal_type as i32;

    // Decode seed
    ps_dec.indices.seed = ps_range_dec.dec_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
}

// ---------------------------------------------------------------------------------------------
// decode_parameters.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_parameters.c:silk_decode_parameters: decodes parameters (gains, LPC,
/// pitch lags, LTP coefficients) from the indices.
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn silk_decode_parameters(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    cond_coding: i32,
) {
    let mut p_nlsf_q15 = [0i16; MLPC];
    let mut p_nlsf0_q15 = [0i16; MLPC];
    let lpc_order = ps_dec.lpc_order as usize;
    let nb_subfr = ps_dec.nb_subfr as usize;

    // Dequant Gains
    silk_gains_dequant(
        &mut ps_dec_ctrl.gains_q16,
        &ps_dec.indices.gains_indices,
        &mut ps_dec.last_gain_index,
        i32::from(cond_coding == CODE_CONDITIONALLY),
        nb_subfr,
    );

    // Decode NLSFs
    silk_nlsf_decode(
        &mut p_nlsf_q15,
        &ps_dec.indices.nlsf_indices,
        ps_dec.ps_nlsf_cb,
    );

    // Convert NLSF parameters to AR prediction filter coefficients
    silk_nlsf2a(&mut ps_dec_ctrl.pred_coef_q12[1], &p_nlsf_q15, lpc_order);

    // If just reset, e.g., because internal Fs changed, do not allow interpolation
    // improves the case of packet loss in the first frame after a switch
    if ps_dec.first_frame_after_reset == 1 {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.nlsf_interp_coef_q2 < 4 {
        // Calculation of the interpolated NLSF0 vector from the interpolation factor,
        // the previous NLSF1, and the current NLSF1
        for i in 0..lpc_order {
            p_nlsf0_q15[i] = (ps_dec.prev_nlsf_q15[i] as i32
                + silk_rshift(
                    silk_mul(
                        ps_dec.indices.nlsf_interp_coef_q2 as i32,
                        p_nlsf_q15[i] as i32 - ps_dec.prev_nlsf_q15[i] as i32,
                    ),
                    2,
                )) as i16;
        }

        // Convert NLSF parameters to AR prediction filter coefficients
        silk_nlsf2a(&mut ps_dec_ctrl.pred_coef_q12[0], &p_nlsf0_q15, lpc_order);
    } else {
        // Copy LPC coefficients for first half from second half
        let [first, second] = &mut ps_dec_ctrl.pred_coef_q12;
        first[..lpc_order].copy_from_slice(&second[..lpc_order]);
    }

    ps_dec.prev_nlsf_q15[..lpc_order].copy_from_slice(&p_nlsf_q15[..lpc_order]);

    // After a packet loss do BWE of LPC coefs
    if ps_dec.loss_cnt != 0 {
        silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[0],
            lpc_order,
            BWE_AFTER_LOSS_Q16,
        );
        silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[1],
            lpc_order,
            BWE_AFTER_LOSS_Q16,
        );
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Decode pitch lags

        // Decode pitch values
        silk_decode_pitch(
            ps_dec.indices.lag_index,
            ps_dec.indices.contour_index,
            &mut ps_dec_ctrl.pitch_l,
            ps_dec.fs_khz,
            ps_dec.nb_subfr,
        );

        // Decode Codebook Index
        let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[ps_dec.indices.per_index as usize]; // set pointer to start of codebook

        for k in 0..nb_subfr {
            let ix = ps_dec.indices.ltp_index[k] as usize;
            for i in 0..LTPO {
                ps_dec_ctrl.ltp_coef_q14[k * LTPO + i] =
                    silk_lshift(cbk_ptr_q7[ix][i] as i32, 7) as i16;
            }
        }

        // Decode LTP scaling
        let ix = ps_dec.indices.ltp_scale_index as usize;
        ps_dec_ctrl.ltp_scale_q14 = SILK_LTPSCALES_TABLE_Q14[ix] as i32;
    } else {
        ps_dec_ctrl.pitch_l[..nb_subfr].fill(0);
        ps_dec_ctrl.ltp_coef_q14[..LTPO * nb_subfr].fill(0);
        ps_dec.indices.per_index = 0;
        ps_dec_ctrl.ltp_scale_q14 = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// stereo_MS_to_LR.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/stereo_MS_to_LR.c:silk_stereo_MS_to_LR: converts the adaptive Mid/Side
/// representation to a Left/Right stereo signal.
///
/// `x1` (mid → left) and `x2` (side → right) hold `frame_length + 2` samples each: the first
/// two are filled from the state buffers, the decoded signal starts at index 2.
pub fn silk_stereo_ms_to_lr(
    state: &mut StereoDecState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_khz: i32,
    frame_length: i32,
) {
    let fl = frame_length as usize;
    let (x1, x2) = (&mut x1[..fl + 2], &mut x2[..fl + 2]);

    // Buffering
    x1[..2].copy_from_slice(&state.s_mid);
    x2[..2].copy_from_slice(&state.s_side);
    state.s_mid.copy_from_slice(&x1[fl..fl + 2]);
    state.s_side.copy_from_slice(&x2[fl..fl + 2]);

    // Interpolate predictors and add prediction to side channel
    let mut pred0_q13 = state.pred_prev_q13[0] as i32;
    let mut pred1_q13 = state.pred_prev_q13[1] as i32;
    let denom_q16 = silk_div32_16(1i32 << 16, STEREO_INTERP_LEN_MS * fs_khz);
    let delta0_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        let mut sum = silk_lshift(
            silk_add_lshift32(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1),
            9,
        ); // Q11
        sum = silk_smlawb(silk_lshift(x2[n + 1] as i32, 8), sum, pred0_q13); // Q8
        sum = silk_smlawb(sum, silk_lshift(x1[n + 1] as i32, 11), pred1_q13); // Q8
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }
    pred0_q13 = pred_q13[0];
    pred1_q13 = pred_q13[1];
    for n in interp_len..fl {
        let mut sum = silk_lshift(
            silk_add_lshift32(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1),
            9,
        ); // Q11
        sum = silk_smlawb(silk_lshift(x2[n + 1] as i32, 8), sum, pred0_q13); // Q8
        sum = silk_smlawb(sum, silk_lshift(x1[n + 1] as i32, 11), pred1_q13); // Q8
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;

    // Convert to left/right signals
    for n in 0..fl {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;
        x1[n + 1] = silk_sat16(sum) as i16;
        x2[n + 1] = silk_sat16(diff) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// dec_API.c
// ---------------------------------------------------------------------------------------------

/// `silk_decoder`: the decoder super struct (one state per internal channel plus the stereo
/// state).
#[derive(Debug, Clone)]
pub struct SilkDecoder {
    pub channel_state: [SilkDecoderState; NCH],
    pub s_stereo: StereoDecState,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub prev_decode_only_middle: i32,
    /// `osce_model` (`ENABLE_OSCE`).
    #[cfg(feature = "osce")]
    pub osce_model: OsceModel,
    /// `channel_state[n].osce` (`ENABLE_OSCE`; C keeps it in `silk_decoder_state`).
    #[cfg(feature = "osce")]
    pub osce: [SilkOsceStruct; NCH],
    /// `channel_state[n].osce_bwe` (`ENABLE_OSCE_BWE`; C keeps it in `silk_decoder_state`).
    #[cfg(feature = "osce")]
    pub osce_bwe: [SilkOsceBweStruct; NCH],
}

/// The `ENABLE_OSCE` / `ENABLE_OSCE_BWE` fields of `silk_DecControlStruct` (silk/control.h),
/// passed to [`SilkDecoder::silk_decode_dnn`] next to the base [`SilkDecControlStruct`].
#[cfg(feature = "osce")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkOsceControl {
    /// `osce_method` (I): `OSCE_METHOD_NONE` / `LACE` / `NOLACE`.
    pub osce_method: i32,
    /// `enable_osce_bwe` (I): set by `OPUS_SET_OSCE_BWE`.
    pub enable_osce_bwe: i32,
    /// `osce_extended_mode` (I): `OSCE_MODE_*` of this call.
    pub osce_extended_mode: i32,
    /// `prev_osce_extended_mode` (I/O).
    pub prev_osce_extended_mode: i32,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Port of silk/dec_API.c:silk_Get_Decoder_Size: writes the size in bytes of the decoder
/// state to `dec_size_bytes` and returns `SILK_NO_ERROR`.
pub const fn silk_get_decoder_size(dec_size_bytes: &mut i32) -> i32 {
    let ret = SILK_NO_ERROR;

    *dec_size_bytes = size_of::<SilkDecoder>() as i32;

    ret
}

impl SilkDecoder {
    /// A decoder initialized with `silk_InitDecoder` (the fields that function does not touch,
    /// `nChannelsAPI` / `nChannelsInternal`, are zero as after the Opus decoder's `OPUS_CLEAR`).
    #[must_use]
    pub fn new() -> Self {
        let mut d = Self {
            channel_state: [SilkDecoderState::new(), SilkDecoderState::new()],
            s_stereo: StereoDecState::default(),
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: 0,
            #[cfg(feature = "osce")]
            osce_model: OsceModel::default(),
            #[cfg(feature = "osce")]
            osce: [SilkOsceStruct::default(), SilkOsceStruct::default()],
            #[cfg(feature = "osce")]
            osce_bwe: [SilkOsceBweStruct::default(), SilkOsceBweStruct::default()],
        };
        d.init();
        d
    }

    /// Port of silk/dec_API.c:silk_LoadOSCEModels. Without `ENABLE_OSCE` this ignores its
    /// arguments and returns `SILK_NO_ERROR`.
    #[cfg(not(feature = "osce"))]
    pub const fn load_osce_models(&mut self, _data: Option<&[u8]>) -> i32 {
        SILK_NO_ERROR
    }

    /// Port of silk/dec_API.c:silk_LoadOSCEModels: binds LACE, NoLACE and BBWENet from a weight
    /// blob and sets `osce_model.loaded` from the result. Returns 0, or -1 when a model fails
    /// to load (always for `None`: the weights are not compiled in, as with upstream
    /// `USE_WEIGHTS_FILE`).
    #[cfg(feature = "osce")]
    pub fn load_osce_models(&mut self, data: Option<&[u8]>) -> i32 {
        let ret = osce_load_models(&mut self.osce_model, data);
        self.osce_model.loaded = ret.is_ok();
        match ret {
            Ok(()) => SILK_NO_ERROR,
            Err(_) => -1,
        }
    }

    /// `silk_init_decoder( &channel_state[ n ] )` including the OSCE state C keeps in the
    /// channel state (cleared, then `osce_reset(.., OSCE_DEFAULT_METHOD)`).
    fn init_channel(&mut self, n: usize) -> i32 {
        let ret = silk_init_decoder(&mut self.channel_state[n]);
        #[cfg(feature = "osce")]
        {
            self.osce[n] = SilkOsceStruct::default();
            self.osce_bwe[n] = SilkOsceBweStruct::default();
            osce_reset(&mut self.osce[n], OSCE_DEFAULT_METHOD);
        }
        ret
    }

    /// `silk_reset_decoder( &channel_state[ n ] )` including its `osce_reset`.
    fn reset_channel(&mut self, n: usize) -> i32 {
        let ret = silk_reset_decoder(&mut self.channel_state[n]);
        #[cfg(feature = "osce")]
        osce_reset(&mut self.osce[n], OSCE_DEFAULT_METHOD);
        ret
    }

    /// Port of silk/dec_API.c:silk_ResetDecoder: resets the decoder state (keeps the channel
    /// configuration).
    pub fn reset(&mut self) -> i32 {
        let mut ret = SILK_NO_ERROR;

        for n in 0..NCH {
            ret = self.reset_channel(n);
        }
        self.s_stereo = StereoDecState::default();
        // Not strictly needed, but it's cleaner that way
        self.prev_decode_only_middle = 0;

        ret
    }

    /// Port of silk/dec_API.c:silk_InitDecoder: initializes the decoder state.
    pub fn init(&mut self) -> i32 {
        let mut ret = SILK_NO_ERROR;
        #[cfg(feature = "osce")]
        {
            self.osce_model.loaded = false;
        }
        // load osce models (C discards the return value; without compiled-in weights this
        // leaves the model unloaded, like upstream USE_WEIGHTS_FILE)
        c_discard(self.load_osce_models(None));

        for n in 0..NCH {
            ret = self.init_channel(n);
        }
        self.s_stereo = StereoDecState::default();
        // Not strictly needed, but it's cleaner that way
        self.prev_decode_only_middle = 0;

        ret
    }

    /// Port of silk/dec_API.c:silk_Decode: decodes one frame (10 or 20 ms) of a SILK payload.
    ///
    /// * `lost_flag`: 0 = no loss ([`FLAG_DECODE_NORMAL`]), 1 = loss ([`FLAG_PACKET_LOST`]),
    ///   2 = decode FEC ([`FLAG_DECODE_LBRR`]).
    /// * `new_packet_flag`: nonzero on the first call for a packet.
    /// * `samples_out`: receives `*n_samples_out` samples per API channel (interleaved when
    ///   `n_channels_api == 2`), as `opus_res` (`INT16TORES`: float in the float build,
    ///   16-bit PCM with `fixed-point`, Q8-scaled `i32` with `fixed-res24`).
    ///
    /// Returns a SILK error code (0 on success; C `int` codes from `silk/errors.h`, or the sum
    /// of the negative resampler init codes, exactly as C).
    pub fn silk_decode(
        &mut self,
        dec_control: &mut SilkDecControlStruct,
        lost_flag: i32,
        new_packet_flag: i32,
        ps_range_dec: &mut EcDec<'_>,
        samples_out: &mut [OpusRes],
        n_samples_out: &mut i32,
    ) -> i32 {
        #[cfg(feature = "osce")]
        let mut osce_ctl = SilkOsceControl::default();
        self.silk_decode_dnn(
            dec_control,
            #[cfg(feature = "osce")]
            &mut osce_ctl,
            lost_flag,
            new_packet_flag,
            ps_range_dec,
            samples_out,
            n_samples_out,
            #[cfg(feature = "deep-plc")]
            None,
        )
    }

    /// Port of silk/dec_API.c:silk_Decode with the DNN arguments: `osce_ctl` holds the
    /// `ENABLE_OSCE` / `ENABLE_OSCE_BWE` fields of `decControl` (method, extended mode;
    /// `prev_osce_extended_mode` is updated) and `lpcnet` is the `ENABLE_DEEP_PLC` LPCNet PLC
    /// state (`None` = C `NULL`), used for channel 0. Otherwise as [`Self::silk_decode`].
    #[expect(
        clippy::needless_range_loop,
        reason = "index arithmetic mirrors C across several arrays"
    )]
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub fn silk_decode_dnn(
        &mut self,
        dec_control: &mut SilkDecControlStruct,
        #[cfg(feature = "osce")] osce_ctl: &mut SilkOsceControl,
        lost_flag: i32,
        new_packet_flag: i32,
        ps_range_dec: &mut EcDec<'_>,
        samples_out: &mut [OpusRes],
        n_samples_out: &mut i32,
        #[cfg(feature = "deep-plc")] mut lpcnet: Option<&mut LpcnetPlcState>,
    ) -> i32 {
        let mut decode_only_middle: i32 = 0;
        let mut ret = SILK_NO_ERROR;
        let mut ms_pred_q13 = [0i32; 2];
        let mut n_samples_out_dec: i32 = 0;

        debug_assert!(dec_control.n_channels_internal == 1 || dec_control.n_channels_internal == 2);
        let nci = dec_control.n_channels_internal as usize;
        let nca = dec_control.n_channels_api as usize;

        // Test if first frame in payload
        if new_packet_flag != 0 {
            for n in 0..nci {
                self.channel_state[n].n_frames_decoded = 0; // Used to count frames in packet
            }
        }

        // If Mono -> Stereo transition in bitstream: init state of second channel
        if dec_control.n_channels_internal > self.n_channels_internal {
            ret += self.init_channel(1);
        }

        let stereo_to_mono = dec_control.n_channels_internal == 1
            && self.n_channels_internal == 2
            && (dec_control.internal_sample_rate == 1000 * self.channel_state[0].fs_khz);

        if self.channel_state[0].n_frames_decoded == 0 {
            for n in 0..nci {
                let cs = &mut self.channel_state[n];
                let (n_frames_per_packet, nb_subfr) = match dec_control.payload_size_ms {
                    // Assuming packet loss, use 10 ms
                    0 | 10 => (1, 2),
                    20 => (1, 4),
                    40 => (2, 4),
                    60 => (3, 4),
                    // C: celt_assert( 0 ) before returning (not reproduced so the error path
                    // stays reachable, as in a C build without assertions).
                    _ => return SILK_DEC_INVALID_FRAME_SIZE,
                };
                cs.n_frames_per_packet = n_frames_per_packet;
                cs.nb_subfr = nb_subfr;
                let fs_khz_dec = (dec_control.internal_sample_rate >> 10) + 1;
                if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                    // C: celt_assert( 0 ) (see above)
                    return SILK_DEC_INVALID_SAMPLING_FREQUENCY;
                }
                ret += silk_decoder_set_fs(cs, fs_khz_dec, dec_control.api_sample_rate);
            }
        }

        if dec_control.n_channels_api == 2
            && dec_control.n_channels_internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.s_stereo.pred_prev_q13 = [0; 2];
            self.s_stereo.s_side = [0; 2];
            let [cs0, cs1] = &mut self.channel_state;
            cs1.resampler_state.clone_from(&cs0.resampler_state);
        }
        self.n_channels_api = dec_control.n_channels_api;
        self.n_channels_internal = dec_control.n_channels_internal;

        if dec_control.api_sample_rate > MAX_API_FS_KHZ * 1000 || dec_control.api_sample_rate < 8000
        {
            ret = SILK_DEC_INVALID_SAMPLING_FREQUENCY;
            return ret;
        }

        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            // First decoder call for this payload
            // Decode VAD flags and LBRR flag
            for n in 0..nci {
                let cs = &mut self.channel_state[n];
                for i in 0..cs.n_frames_per_packet as usize {
                    cs.vad_flags[i] = i32::from(ps_range_dec.dec_bit_logp(1));
                }
                cs.lbrr_flag = i32::from(ps_range_dec.dec_bit_logp(1));
            }
            // Decode LBRR flags
            for n in 0..nci {
                let cs = &mut self.channel_state[n];
                cs.lbrr_flags = [0; 3];
                if cs.lbrr_flag != 0 {
                    if cs.n_frames_per_packet == 1 {
                        cs.lbrr_flags[0] = 1;
                    } else {
                        let lbrr_symbol = ps_range_dec.dec_icdf(
                            SILK_LBRR_FLAGS_ICDF_PTR[(cs.n_frames_per_packet - 2) as usize],
                            8,
                        ) as i32
                            + 1;
                        for i in 0..cs.n_frames_per_packet as usize {
                            cs.lbrr_flags[i] = silk_rshift(lbrr_symbol, i as i32) & 1;
                        }
                    }
                }
            }

            if lost_flag == FLAG_DECODE_NORMAL {
                // Regular decoding: skip all LBRR data
                for i in 0..self.channel_state[0].n_frames_per_packet as usize {
                    for n in 0..nci {
                        if self.channel_state[n].lbrr_flags[i] != 0 {
                            let mut pulses = [0i16; MFL];

                            if nci == 2 && n == 0 {
                                silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
                                if self.channel_state[1].lbrr_flags[i] == 0 {
                                    decode_only_middle = silk_stereo_decode_mid_only(ps_range_dec);
                                }
                            }
                            let cs = &mut self.channel_state[n];
                            // Use conditional coding if previous frame available
                            let cond_coding = if i > 0 && cs.lbrr_flags[i - 1] != 0 {
                                CODE_CONDITIONALLY
                            } else {
                                CODE_INDEPENDENTLY
                            };
                            silk_decode_indices(cs, ps_range_dec, i as i32, 1, cond_coding);
                            silk_decode_pulses(
                                ps_range_dec,
                                &mut pulses,
                                cs.indices.signal_type as i32,
                                cs.indices.quant_offset_type as i32,
                                cs.frame_length,
                            );
                        }
                    }
                }
            }
        }

        // Get MS predictor index
        if nci == 2 {
            let nfd = self.channel_state[0].n_frames_decoded as usize;
            if lost_flag == FLAG_DECODE_NORMAL
                || (lost_flag == FLAG_DECODE_LBRR && self.channel_state[0].lbrr_flags[nfd] == 1)
            {
                silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
                // For LBRR data, decode mid-only flag only if side-channel's LBRR flag is false
                if (lost_flag == FLAG_DECODE_NORMAL && self.channel_state[1].vad_flags[nfd] == 0)
                    || (lost_flag == FLAG_DECODE_LBRR && self.channel_state[1].lbrr_flags[nfd] == 0)
                {
                    decode_only_middle = silk_stereo_decode_mid_only(ps_range_dec);
                } else {
                    decode_only_middle = 0;
                }
            } else {
                for n in 0..2 {
                    ms_pred_q13[n] = self.s_stereo.pred_prev_q13[n] as i32;
                }
            }
        }

        // Reset side channel decoder prediction memory for first frame with side coding
        if nci == 2 && decode_only_middle == 0 && self.prev_decode_only_middle == 1 {
            let cs1 = &mut self.channel_state[1];
            cs1.out_buf.fill(0);
            cs1.s_lpc_q14_buf.fill(0);
            cs1.lag_prev = 100;
            cs1.last_gain_index = 10;
            cs1.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            cs1.first_frame_after_reset = 1;
        }

        // Temp buffers `samplesOut1_tmp[ n ]`, each `frame_length + 2` samples.
        let mut samples_out1_tmp = [[0i16; MFL + 2]; NCH];

        let has_side = if lost_flag == FLAG_DECODE_NORMAL {
            decode_only_middle == 0
        } else {
            self.prev_decode_only_middle == 0
                || (nci == 2
                    && lost_flag == FLAG_DECODE_LBRR
                    && self.channel_state[1].lbrr_flags
                        [self.channel_state[1].n_frames_decoded as usize]
                        == 1)
        };
        self.channel_state[0].s_plc.enable_deep_plc = dec_control.enable_deep_plc;
        // Call decoder for one frame
        for n in 0..nci {
            if n == 0 || has_side {
                let frame_index = self.channel_state[0].n_frames_decoded - n as i32;
                #[cfg(feature = "osce")]
                if self.osce[n].method != osce_ctl.osce_method {
                    osce_reset(&mut self.osce[n], osce_ctl.osce_method);
                }
                let cs = &mut self.channel_state[n];
                // Use independent coding if no previous frame available
                let cond_coding = if frame_index <= 0 {
                    CODE_INDEPENDENTLY
                } else if lost_flag == FLAG_DECODE_LBRR {
                    if cs.lbrr_flags[(frame_index - 1) as usize] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    }
                } else if n > 0 && self.prev_decode_only_middle != 0 {
                    // If we skipped a side frame in this packet, we don't
                    // need LTP scaling; the LTP state is well-defined.
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                ret += silk_decode_frame(
                    cs,
                    ps_range_dec,
                    &mut samples_out1_tmp[n][2..],
                    &mut n_samples_out_dec,
                    lost_flag,
                    cond_coding,
                    #[cfg(feature = "deep-plc")]
                    if n == 0 { lpcnet.as_deref_mut() } else { None },
                    #[cfg(feature = "osce")]
                    (&self.osce_model, &mut self.osce[n]),
                );
            } else {
                samples_out1_tmp[n][2..2 + n_samples_out_dec as usize].fill(0);
            }
            self.channel_state[n].n_frames_decoded += 1;
        }

        let nsod = n_samples_out_dec as usize;
        if dec_control.n_channels_api == 2 && dec_control.n_channels_internal == 2 {
            // Convert Mid/Side to Left/Right
            let [t0, t1] = &mut samples_out1_tmp;
            silk_stereo_ms_to_lr(
                &mut self.s_stereo,
                t0,
                t1,
                &ms_pred_q13,
                self.channel_state[0].fs_khz,
                n_samples_out_dec,
            );
        } else {
            // Buffering
            samples_out1_tmp[0][..2].copy_from_slice(&self.s_stereo.s_mid);
            self.s_stereo
                .s_mid
                .copy_from_slice(&samples_out1_tmp[0][nsod..nsod + 2]);
        }

        // Number of output samples
        *n_samples_out = silk_div32(
            n_samples_out_dec * dec_control.api_sample_rate,
            silk_smulbb(self.channel_state[0].fs_khz, 1000),
        );
        let nso = *n_samples_out as usize;

        // Set up pointers to temp buffers
        let mut samples_out2_tmp = [0i16; MAX_SAMPLES_OUT];
        let resample_out = &mut samples_out2_tmp[..];

        #[cfg(feature = "osce")]
        let mut resamp_buffer = [0i16; 3 * MFL];
        for n in 0..silk_min(dec_control.n_channels_api, dec_control.n_channels_internal) as usize {
            #[cfg(feature = "osce")]
            if osce_ctl.osce_extended_mode == OSCE_MODE_SILK_BBWE {
                // Resample or extend decoded signal to API_sampleRate
                debug_assert!(dec_control.api_sample_rate == 48000);

                if osce_ctl.prev_osce_extended_mode != OSCE_MODE_SILK_BBWE {
                    // Reset the BWE state
                    osce_bwe_reset(&mut self.osce_bwe[n]);
                }

                osce_bwe(
                    &self.osce_model,
                    &mut self.osce_bwe[n],
                    resample_out,
                    &samples_out1_tmp[n][1..],
                    nsod,
                );

                if osce_ctl.prev_osce_extended_mode == OSCE_MODE_SILK_ONLY
                    || osce_ctl.prev_osce_extended_mode == OSCE_MODE_HYBRID
                {
                    // cross-fade with upsampled signal (C ignores the resampler's return value)
                    c_discard(silk_resampler(
                        &mut self.channel_state[n].resampler_state,
                        &mut resamp_buffer,
                        &samples_out1_tmp[n][1..],
                        n_samples_out_dec,
                    ));
                    osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer, 480);
                }
            } else {
                ret += silk_resampler(
                    &mut self.channel_state[n].resampler_state,
                    resample_out,
                    &samples_out1_tmp[n][1..],
                    n_samples_out_dec,
                );
                if osce_ctl.prev_osce_extended_mode == OSCE_MODE_SILK_BBWE
                    && dec_control.internal_sample_rate == 16000
                {
                    // fade out if internal sample rate did not change
                    osce_bwe(
                        &self.osce_model,
                        &mut self.osce_bwe[n],
                        &mut resamp_buffer,
                        &samples_out1_tmp[n][1..],
                        nsod,
                    );
                    // cross-fade with upsampled signal
                    osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer, 480);
                }
            }
            // Resample decoded signal to API_sampleRate
            #[cfg(not(feature = "osce"))]
            {
                ret += silk_resampler(
                    &mut self.channel_state[n].resampler_state,
                    resample_out,
                    &samples_out1_tmp[n][1..],
                    n_samples_out_dec,
                );
            }

            // Interleave if stereo output and stereo stream
            if nca == 2 {
                for i in 0..nso {
                    samples_out[n + 2 * i] = int16tores(resample_out[i]);
                }
            } else {
                for i in 0..nso {
                    samples_out[i] = int16tores(resample_out[i]);
                }
            }
        }

        #[cfg(feature = "osce")]
        {
            osce_ctl.prev_osce_extended_mode = osce_ctl.osce_extended_mode;
        }

        // Create two channel output from mono stream
        if dec_control.n_channels_api == 2 && dec_control.n_channels_internal == 1 {
            if stereo_to_mono {
                // Resample right channel for newly collapsed stereo just in case
                // we weren't doing collapsing when switching to mono
                ret += silk_resampler(
                    &mut self.channel_state[1].resampler_state,
                    resample_out,
                    &samples_out1_tmp[0][1..],
                    n_samples_out_dec,
                );

                for i in 0..nso {
                    samples_out[1 + 2 * i] = int16tores(resample_out[i]);
                }
            } else {
                for i in 0..nso {
                    samples_out[1 + 2 * i] = samples_out[2 * i];
                }
            }
        }

        // Export pitch lag, measured at 48 kHz sampling rate
        if self.channel_state[0].prev_signal_type == TYPE_VOICED {
            const MULT_TAB: [i32; 3] = [6, 4, 3];
            dec_control.prev_pitch_lag = self.channel_state[0].lag_prev
                * MULT_TAB[((self.channel_state[0].fs_khz - 8) >> 2) as usize];
        } else {
            dec_control.prev_pitch_lag = 0;
        }

        if lost_flag == FLAG_PACKET_LOST {
            // On packet loss, remove the gain clamping to prevent having the energy "bounce back"
            // if we lose packets when the energy is going down
            for i in 0..self.n_channels_internal as usize {
                self.channel_state[i].last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        ret
    }
}

/// C calls whose `int` return value is discarded (`silk_LoadOSCEModels` in `silk_InitDecoder`,
/// the cross-fade `silk_resampler` of the BWE path).
#[inline]
const fn c_discard(_ret: i32) {}

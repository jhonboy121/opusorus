//! Port of `silk/enc_API.c`, `silk/init_encoder.c` and `silk/control_codec.c`: the SILK encoder
//! API (`silk_Get_Encoder_Size`, `silk_InitEncoder`, `silk_Encode`), encoder state
//! initialization, and encoder control (`silk_control_encoder` with its static setup helpers).
//!
//! The encoder super-struct `silk_encoder` (declared in `silk/float/structs_FLP.h`) is
//! [`SilkEncoder`].
//!
//! * FIXED_POINT: not ported (float build) — the C files select `main_FIX.h` /
//!   `silk_encoder_state_FIX` there; only `silk_setup_resamplers` has a fixed-point branch.
//! * DNN: ENABLE_DRED only adds `#include "dred_encoder.h"` to `enc_API.c` and
//!   `init_encoder.c` (no code in these files); nothing to hook here.

#![allow(
    clippy::too_many_arguments,
    reason = "function signatures mirror the C sources one-to-one"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::celt::arch::{OpusRes, res2int16};
use crate::celt::entenc::EcEnc;
use crate::silk::coding::{
    silk_encode_pulses, silk_stereo_encode_mid_only, silk_stereo_encode_pred,
};
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    ENCODER_NUM_CHANNELS, FIND_PITCH_LPC_WIN_MS, FIND_PITCH_LPC_WIN_MS_2_SF, LA_PITCH_MS,
    LA_SHAPE_MAX, LA_SHAPE_MS, LTP_MEM_LENGTH_MS, MAX_API_FS_KHZ, MAX_DEL_DEC_STATES,
    MAX_FIND_PITCH_LPC_ORDER, MAX_FRAME_LENGTH_MS, MAX_LPC_ORDER, MAX_NB_SUBFR,
    MAX_SHAPE_LPC_ORDER, MIN_LPC_ORDER, SHAPE_LPC_WIN_MAX, SUB_FRAME_LENGTH_MS,
    TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::encoder_common::{
    check_control_input, silk_control_audio_bandwidth, silk_control_snr, silk_encode_indices,
    silk_hp_variable_cutoff, silk_stereo_lr_to_ms,
};
use crate::silk::errors::{
    SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES, SILK_ENC_PACKET_SIZE_NOT_SUPPORTED, SILK_NO_ERROR,
};
use crate::silk::float::{
    SilkEncoderStateFlp, SilkShapeStateFlp, silk_encode_do_vad_flp, silk_encode_frame_flp,
    silk_float2short_array, silk_short2float_array,
};
use crate::silk::macros::{
    silk_div32_16, silk_fix_const, silk_limit, silk_lshift, silk_max_int, silk_min, silk_min_int,
    silk_mul, silk_rshift, silk_rshift_round, silk_smlawb, silk_smulbb, silk_smulwb,
};
use crate::silk::resampler::{SilkResamplerState, silk_resampler, silk_resampler_init};
use crate::silk::sigproc::silk_lin2log;
use crate::silk::structs::{SilkEncControlStruct, SilkNsqState, StereoEncState};
use crate::silk::tables::{
    SILK_LBRR_FLAGS_ICDF_PTR, SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB, SILK_PE_MAX_COMPLEX,
    SILK_PE_MID_COMPLEX, SILK_PE_MIN_COMPLEX, SILK_PITCH_CONTOUR_10_MS_ICDF,
    SILK_PITCH_CONTOUR_10_MS_NB_ICDF, SILK_PITCH_CONTOUR_ICDF, SILK_PITCH_CONTOUR_NB_ICDF,
    SILK_QUANTIZATION_OFFSETS_Q10, SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF,
};
use crate::silk::tuning_parameters::{
    BITRESERVOIR_DECAY_TIME_MS, MAX_BANDWIDTH_SWITCH_DELAY_MS, SPEECH_ACTIVITY_DTX_THRES,
    VARIABLE_HP_MIN_CUTOFF_HZ, WARPING_MULTIPLIER,
};
use crate::silk::vad::silk_vad_init;

const NCH: usize = ENCODER_NUM_CHANNELS as usize;

/// Maximum number of API-rate samples per channel handled by one pass of the `silk_Encode`
/// input loop (at most one 20 ms frame): the size of the C `buf` VLA actually used.
const ENC_BUF_MAX: usize = (MAX_API_FS_KHZ * MAX_FRAME_LENGTH_MS) as usize;

/// Longest `x_buf` history re-sampled by `silk_setup_resamplers`:
/// `buf_length_ms = 2 * nb_subfr * 5 + LA_SHAPE_MS` ≤ 45 ms.
const SETUP_BUF_MS: usize = (2 * MAX_NB_SUBFR * SUB_FRAME_LENGTH_MS + LA_SHAPE_MS) as usize;

// ---------------------------------------------------------------------------------------------
// structs_FLP.h: silk_encoder
// ---------------------------------------------------------------------------------------------

/// `silk_encoder`: the SILK encoder super struct (`silk/float/structs_FLP.h`).
///
/// Roughly 20 KB (two channel states); the Opus encoder should embed or `Box` it.
#[derive(Debug, Clone)]
pub struct SilkEncoder {
    pub s_stereo: StereoEncState,
    pub n_bits_used_lbrr: i32,
    pub n_bits_exceeded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub n_prev_channels_internal: i32,
    pub time_since_switch_allowed_ms: i32,
    pub allow_bandwidth_switch: i32,
    pub prev_decode_only_middle: i32,
    /// This needs to be last so we can skip the second state for mono (C layout note).
    pub state_fxx: [SilkEncoderStateFlp; NCH],
}

impl SilkEncoder {
    /// All-zero encoder (C: the memory handed to `silk_InitEncoder`, cleared by the Opus
    /// encoder). Call [`Self::init`] before encoding.
    #[must_use]
    pub fn new() -> Self {
        Self {
            s_stereo: StereoEncState::default(),
            n_bits_used_lbrr: 0,
            n_bits_exceeded: 0,
            n_channels_api: 0,
            n_channels_internal: 0,
            n_prev_channels_internal: 0,
            time_since_switch_allowed_ms: 0,
            allow_bandwidth_switch: 0,
            prev_decode_only_middle: 0,
            state_fxx: [SilkEncoderStateFlp::new(), SilkEncoderStateFlp::new()],
        }
    }

    /// Port of `silk/enc_API.c:silk_InitEncoder` — init or reset the encoder for `channels`
    /// channels and read the control structure into `enc_status`.
    ///
    /// For mono the second channel state is left untouched, like the C `memset` that skips it.
    pub fn init(&mut self, channels: i32, enc_status: &mut SilkEncControlStruct) -> i32 {
        let mut ret = SILK_NO_ERROR;

        // Reset encoder. Skip second encoder state for mono.
        self.s_stereo = StereoEncState::default();
        self.n_bits_used_lbrr = 0;
        self.n_bits_exceeded = 0;
        self.n_channels_api = 0;
        self.n_channels_internal = 0;
        self.n_prev_channels_internal = 0;
        self.time_since_switch_allowed_ms = 0;
        self.allow_bandwidth_switch = 0;
        self.prev_decode_only_middle = 0;
        self.state_fxx[0] = SilkEncoderStateFlp::new();
        if channels != 1 {
            self.state_fxx[1] = SilkEncoderStateFlp::new();
        }
        for n in 0..channels as usize {
            ret += silk_init_encoder(&mut self.state_fxx[n]);
            debug_assert!(ret == 0);
        }

        self.n_channels_api = 1;
        self.n_channels_internal = 1;

        // Read control structure
        ret += self.query_encoder(enc_status);
        debug_assert!(ret == 0);

        ret
    }

    /// Port of the static `silk/enc_API.c:silk_QueryEncoder` — read control structure from
    /// encoder.
    const fn query_encoder(&self, enc_status: &mut SilkEncControlStruct) -> i32 {
        let s = &self.state_fxx[0].s_cmn;
        enc_status.n_channels_api = self.n_channels_api;
        enc_status.n_channels_internal = self.n_channels_internal;
        enc_status.api_sample_rate = s.api_fs_hz;
        enc_status.max_internal_sample_rate = s.max_internal_fs_hz;
        enc_status.min_internal_sample_rate = s.min_internal_fs_hz;
        enc_status.desired_internal_sample_rate = s.desired_internal_fs_hz;
        enc_status.payload_size_ms = s.packet_size_ms;
        enc_status.bit_rate = s.target_rate_bps;
        enc_status.packet_loss_percentage = s.packet_loss_perc;
        enc_status.complexity = s.complexity;
        enc_status.use_in_band_fec = s.use_in_band_fec;
        enc_status.use_dtx = s.use_dtx;
        enc_status.use_cbr = s.use_cbr;
        enc_status.internal_sample_rate = silk_smulbb(s.fs_khz, 1000);
        enc_status.allow_bandwidth_switch = s.allow_bandwidth_switch;
        enc_status.in_wb_mode_without_variable_lp = (s.fs_khz == 16 && s.s_lp.mode == 0) as i32;
        SILK_NO_ERROR
    }

    /// Port of `silk/enc_API.c:silk_Encode` — encode frame(s) with SILK.
    ///
    /// `samples_in` holds `n_samples_in` samples per API channel (interleaved for stereo),
    /// as `opus_res` (float in the float build, converted with `RES2INT16`). `n_bytes_out` is
    /// the payload size (input: max bytes). If `prefill_flag` is set, the input must contain
    /// 10 ms of audio irrespective of `enc_control.payload_size_ms`, and `ps_range_enc` is never
    /// touched (the Opus encoder passes `NULL`; pass any encoder, e.g. one over an empty
    /// buffer). Returns a C error code (`silk::errors`).
    #[allow(
        clippy::too_many_lines,
        reason = "one C function; splitting it would obscure the correspondence"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "one C function; splitting it would obscure the correspondence"
    )]
    pub fn silk_encode(
        &mut self,
        enc_control: &mut SilkEncControlStruct,
        samples_in: &[OpusRes],
        mut n_samples_in: i32,
        ps_range_enc: &mut EcEnc<'_>,
        n_bytes_out: &mut i32,
        prefill_flag: i32,
        activity: i32,
    ) -> i32 {
        let mut ret: i32;
        let mut tmp_payload_size_ms = 0;
        let mut tmp_complexity = 0;
        let mut ms_target_rates_bps = [0i32; 2];
        let mut buf = [0i16; ENC_BUF_MAX];
        let mut samples_off = 0usize;

        debug_assert!(
            enc_control.n_channels_api >= enc_control.n_channels_internal
                && enc_control.n_channels_api >= self.n_channels_internal
        );
        if enc_control.reduced_dependency != 0 {
            for n in 0..enc_control.n_channels_api as usize {
                self.state_fxx[n].s_cmn.first_frame_after_reset = 1;
            }
        }
        for n in 0..enc_control.n_channels_api as usize {
            self.state_fxx[n].s_cmn.n_frames_encoded = 0;
        }
        // Check values in encoder control structure
        ret = check_control_input(enc_control);
        if ret != 0 {
            // C: celt_assert( 0 ) (aborts in a hardened build; the error return is kept)
            return ret;
        }

        enc_control.switch_ready = 0;

        if enc_control.n_channels_internal > self.n_channels_internal {
            // Mono -> Stereo transition: init state of second channel and stereo state
            ret += silk_init_encoder(&mut self.state_fxx[1]);
            self.s_stereo.pred_prev_q13 = [0; 2];
            self.s_stereo.s_side = [0; 2];
            self.s_stereo.mid_side_amp_q0 = [0, 1, 0, 1];
            self.s_stereo.width_prev_q14 = 0;
            self.s_stereo.smth_width_q14 = silk_fix_const(1.0, 14) as i16;
            if self.n_channels_api == 2 {
                let [s0, s1] = &mut self.state_fxx;
                s1.s_cmn
                    .resampler_state
                    .clone_from(&s0.s_cmn.resampler_state);
                s1.s_cmn.in_hp_state = s0.s_cmn.in_hp_state;
            }
        }

        let transition = (enc_control.payload_size_ms != self.state_fxx[0].s_cmn.packet_size_ms)
            || (self.n_channels_internal != enc_control.n_channels_internal);

        self.n_channels_api = enc_control.n_channels_api;
        self.n_channels_internal = enc_control.n_channels_internal;

        let n_blocks_of_10ms = 100 * n_samples_in / enc_control.api_sample_rate;
        let tot_blocks = if n_blocks_of_10ms > 1 {
            n_blocks_of_10ms >> 1
        } else {
            1
        };
        let mut curr_block = 0;
        if prefill_flag != 0 {
            // Only accept input length of 10 ms
            if n_blocks_of_10ms != 1 {
                // C: celt_assert( 0 )
                return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            }
            let mut save_lp = self.state_fxx[0].s_cmn.s_lp;
            if prefill_flag == 2 {
                // Save the sampling rate so the bandwidth switching code can keep handling
                // transitions.
                save_lp.saved_fs_khz = self.state_fxx[0].s_cmn.fs_khz;
            }
            // Reset Encoder
            for n in 0..enc_control.n_channels_internal as usize {
                ret = silk_init_encoder(&mut self.state_fxx[n]);
                // Restore the variable LP state.
                if prefill_flag == 2 {
                    self.state_fxx[n].s_cmn.s_lp = save_lp;
                }
                debug_assert!(ret == 0);
            }
            tmp_payload_size_ms = enc_control.payload_size_ms;
            enc_control.payload_size_ms = 10;
            tmp_complexity = enc_control.complexity;
            enc_control.complexity = 0;
            for n in 0..enc_control.n_channels_internal as usize {
                self.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
                self.state_fxx[n].s_cmn.prefill_flag = 1;
            }
        } else {
            // Only accept input lengths that are a multiple of 10 ms
            if n_blocks_of_10ms * enc_control.api_sample_rate != 100 * n_samples_in
                || n_samples_in < 0
            {
                // C: celt_assert( 0 )
                return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            }
            // Make sure no more than one packet can be produced
            if 1000 * n_samples_in > enc_control.payload_size_ms * enc_control.api_sample_rate {
                // C: celt_assert( 0 )
                return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            }
        }

        for n in 0..enc_control.n_channels_internal as usize {
            // Force the side channel to the same rate as the mid
            let force_fs_khz = if n == 1 {
                self.state_fxx[0].s_cmn.fs_khz
            } else {
                0
            };
            ret = silk_control_encoder(
                &mut self.state_fxx[n],
                enc_control,
                self.allow_bandwidth_switch,
                n as i32,
                force_fs_khz,
            );
            if ret != 0 {
                return ret;
            }
            if self.state_fxx[n].s_cmn.first_frame_after_reset != 0 || transition {
                for i in 0..self.state_fxx[0].s_cmn.n_frames_per_packet as usize {
                    self.state_fxx[n].s_cmn.lbrr_flags[i] = 0;
                }
            }
            self.state_fxx[n].s_cmn.in_dtx = self.state_fxx[n].s_cmn.use_dtx;
        }
        debug_assert!(
            enc_control.n_channels_internal == 1
                || self.state_fxx[0].s_cmn.fs_khz == self.state_fxx[1].s_cmn.fs_khz
        );

        // Input buffering/resampling and encoding
        let n_samples_to_buffer_max = 10 * n_blocks_of_10ms * self.state_fxx[0].s_cmn.fs_khz;
        loop {
            let mut curr_n_bits_used_lbrr = 0;
            let n_ch_api = enc_control.n_channels_api;
            let n_ch_int = enc_control.n_channels_internal;
            {
                let [s0, s1] = &mut self.state_fxx;
                let mut n_samples_to_buffer = s0.s_cmn.frame_length - s0.s_cmn.input_buf_ix;
                n_samples_to_buffer = silk_min(n_samples_to_buffer, n_samples_to_buffer_max);
                let n_samples_from_input = silk_div32_16(
                    n_samples_to_buffer * s0.s_cmn.api_fs_hz,
                    s0.s_cmn.fs_khz * 1000,
                );
                let nfi = n_samples_from_input as usize;
                let input = &samples_in[samples_off..];
                // Resample and write to buffer
                if n_ch_api == 2 && n_ch_int == 2 {
                    let id = s0.s_cmn.n_frames_encoded;
                    for n in 0..nfi {
                        buf[n] = res2int16(input[2 * n]);
                    }
                    // Making sure to start both resamplers from the same state when switching
                    // from mono to stereo
                    if self.n_prev_channels_internal == 1 && id == 0 {
                        s1.s_cmn
                            .resampler_state
                            .clone_from(&s0.s_cmn.resampler_state);
                    }

                    let ix = s0.s_cmn.input_buf_ix as usize;
                    ret += silk_resampler(
                        &mut s0.s_cmn.resampler_state,
                        &mut s0.s_cmn.input_buf[ix + 2..],
                        &buf,
                        n_samples_from_input,
                    );
                    s0.s_cmn.input_buf_ix += n_samples_to_buffer;

                    n_samples_to_buffer = s1.s_cmn.frame_length - s1.s_cmn.input_buf_ix;
                    n_samples_to_buffer =
                        silk_min(n_samples_to_buffer, 10 * n_blocks_of_10ms * s1.s_cmn.fs_khz);
                    for n in 0..nfi {
                        buf[n] = res2int16(input[2 * n + 1]);
                    }
                    let ix = s1.s_cmn.input_buf_ix as usize;
                    ret += silk_resampler(
                        &mut s1.s_cmn.resampler_state,
                        &mut s1.s_cmn.input_buf[ix + 2..],
                        &buf,
                        n_samples_from_input,
                    );

                    s1.s_cmn.input_buf_ix += n_samples_to_buffer;
                } else if n_ch_api == 2 && n_ch_int == 1 {
                    // Combine left and right channels before resampling
                    for n in 0..nfi {
                        let sum = res2int16(input[2 * n] + input[2 * n + 1]) as i32;
                        buf[n] = silk_rshift_round(sum, 1) as i16;
                    }
                    let ix = s0.s_cmn.input_buf_ix as usize;
                    ret += silk_resampler(
                        &mut s0.s_cmn.resampler_state,
                        &mut s0.s_cmn.input_buf[ix + 2..],
                        &buf,
                        n_samples_from_input,
                    );
                    // On the first mono frame, average the results for the two resampler states
                    if self.n_prev_channels_internal == 2 && s0.s_cmn.n_frames_encoded == 0 {
                        let ix1 = s1.s_cmn.input_buf_ix as usize;
                        ret += silk_resampler(
                            &mut s1.s_cmn.resampler_state,
                            &mut s1.s_cmn.input_buf[ix1 + 2..],
                            &buf,
                            n_samples_from_input,
                        );
                        for n in 0..s0.s_cmn.frame_length as usize {
                            s0.s_cmn.input_buf[ix + n + 2] = silk_rshift(
                                s0.s_cmn.input_buf[ix + n + 2] as i32
                                    + s1.s_cmn.input_buf[ix1 + n + 2] as i32,
                                1,
                            ) as i16;
                        }
                    }
                    s0.s_cmn.input_buf_ix += n_samples_to_buffer;
                } else {
                    debug_assert!(n_ch_api == 1 && n_ch_int == 1);
                    for n in 0..nfi {
                        buf[n] = res2int16(input[n]);
                    }
                    let ix = s0.s_cmn.input_buf_ix as usize;
                    ret += silk_resampler(
                        &mut s0.s_cmn.resampler_state,
                        &mut s0.s_cmn.input_buf[ix + 2..],
                        &buf,
                        n_samples_from_input,
                    );
                    s0.s_cmn.input_buf_ix += n_samples_to_buffer;
                }

                samples_off += nfi * n_ch_api as usize;
                n_samples_in -= n_samples_from_input;
            }

            // Default
            self.allow_bandwidth_switch = 0;

            // Silk encoder
            if self.state_fxx[0].s_cmn.input_buf_ix >= self.state_fxx[0].s_cmn.frame_length {
                // Enough data in input buffer, so encode
                debug_assert!(
                    self.state_fxx[0].s_cmn.input_buf_ix == self.state_fxx[0].s_cmn.frame_length
                );
                debug_assert!(
                    n_ch_int == 1
                        || self.state_fxx[1].s_cmn.input_buf_ix
                            == self.state_fxx[1].s_cmn.frame_length
                );

                // Deal with LBRR data
                if self.state_fxx[0].s_cmn.n_frames_encoded == 0 && prefill_flag == 0 {
                    // Create space at start of payload for VAD and FEC flags
                    let mut icdf = [0u8; 2];
                    icdf[0] = (256
                        - silk_rshift(
                            256,
                            (self.state_fxx[0].s_cmn.n_frames_per_packet + 1) * n_ch_int,
                        )) as u8;
                    ps_range_enc.enc_icdf(0, &icdf, 8);
                    curr_n_bits_used_lbrr = ps_range_enc.tell();

                    // Encode any LBRR data from previous packet
                    // Encode LBRR flags
                    for n in 0..n_ch_int as usize {
                        let s = &mut self.state_fxx[n].s_cmn;
                        let mut lbrr_symbol = 0i32;
                        for i in 0..s.n_frames_per_packet as usize {
                            lbrr_symbol |= silk_lshift(s.lbrr_flags[i], i as i32);
                        }
                        s.lbrr_flag = (lbrr_symbol > 0) as i8;
                        if lbrr_symbol != 0 && s.n_frames_per_packet > 1 {
                            ps_range_enc.enc_icdf(
                                (lbrr_symbol - 1) as usize,
                                SILK_LBRR_FLAGS_ICDF_PTR[(s.n_frames_per_packet - 2) as usize],
                                8,
                            );
                        }
                    }

                    // Code LBRR indices and excitation signals
                    for i in 0..self.state_fxx[0].s_cmn.n_frames_per_packet as usize {
                        for n in 0..n_ch_int as usize {
                            if self.state_fxx[n].s_cmn.lbrr_flags[i] != 0 {
                                if n_ch_int == 2 && n == 0 {
                                    silk_stereo_encode_pred(
                                        ps_range_enc,
                                        &self.s_stereo.pred_ix[i],
                                    );
                                    // For LBRR data there's no need to code the mid-only flag if
                                    // the side-channel LBRR flag is set
                                    if self.state_fxx[1].s_cmn.lbrr_flags[i] == 0 {
                                        silk_stereo_encode_mid_only(
                                            ps_range_enc,
                                            self.s_stereo.mid_only_flags[i],
                                        );
                                    }
                                }
                                let s = &mut self.state_fxx[n].s_cmn;
                                // Use conditional coding if previous frame available
                                let cond_coding = if i > 0 && s.lbrr_flags[i - 1] != 0 {
                                    CODE_CONDITIONALLY
                                } else {
                                    CODE_INDEPENDENTLY
                                };
                                silk_encode_indices(s, ps_range_enc, i, 1, cond_coding);
                                let signal_type = s.indices_lbrr[i].signal_type as i32;
                                let quant_offset_type = s.indices_lbrr[i].quant_offset_type as i32;
                                silk_encode_pulses(
                                    ps_range_enc,
                                    signal_type,
                                    quant_offset_type,
                                    &mut s.pulses_lbrr[i],
                                    s.frame_length,
                                );
                            }
                        }
                    }

                    // Reset LBRR flags
                    for n in 0..n_ch_int as usize {
                        self.state_fxx[n].s_cmn.lbrr_flags = [0; 3];
                    }
                    curr_n_bits_used_lbrr = ps_range_enc.tell() - curr_n_bits_used_lbrr;
                }

                silk_hp_variable_cutoff(&mut self.state_fxx[0].s_cmn);

                // Total target bits for packet
                let mut n_bits = silk_div32_16(
                    silk_mul(enc_control.bit_rate, enc_control.payload_size_ms),
                    1000,
                );
                // Subtract bits used for LBRR
                if prefill_flag == 0 {
                    // psEnc->nBitsUsedLBRR is an exponential moving average of the LBRR usage,
                    // except that for the first LBRR frame it does no averaging and for the
                    // first frame after after LBRR, it goes back to zero immediately.
                    if curr_n_bits_used_lbrr < 10 {
                        self.n_bits_used_lbrr = 0;
                    } else if self.n_bits_used_lbrr < 10 {
                        self.n_bits_used_lbrr = curr_n_bits_used_lbrr;
                    } else {
                        self.n_bits_used_lbrr = (self.n_bits_used_lbrr + curr_n_bits_used_lbrr) / 2;
                    }
                    n_bits -= self.n_bits_used_lbrr;
                }
                // Divide by number of uncoded frames left in packet
                n_bits = silk_div32_16(n_bits, self.state_fxx[0].s_cmn.n_frames_per_packet);
                // Convert to bits/second
                let mut target_rate_bps = if enc_control.payload_size_ms == 10 {
                    silk_smulbb(n_bits, 100)
                } else {
                    silk_smulbb(n_bits, 50)
                };
                // Subtract fraction of bits in excess of target in previous frames and packets
                target_rate_bps -= silk_div32_16(
                    silk_mul(self.n_bits_exceeded, 1000),
                    BITRESERVOIR_DECAY_TIME_MS,
                );
                if prefill_flag == 0 && self.state_fxx[0].s_cmn.n_frames_encoded > 0 {
                    // Compare actual vs target bits so far in this packet
                    let bits_balance = ps_range_enc.tell()
                        - self.n_bits_used_lbrr
                        - n_bits * self.state_fxx[0].s_cmn.n_frames_encoded;
                    target_rate_bps -=
                        silk_div32_16(silk_mul(bits_balance, 1000), BITRESERVOIR_DECAY_TIME_MS);
                }
                // Never exceed input bitrate
                target_rate_bps = silk_limit(target_rate_bps, enc_control.bit_rate, 5000);

                // Convert Left/Right to Mid/Side
                let nfe = self.state_fxx[0].s_cmn.n_frames_encoded as usize;
                if n_ch_int == 2 {
                    {
                        let [s0, s1] = &mut self.state_fxx;
                        let mut ix = self.s_stereo.pred_ix[nfe];
                        let mut mid_only = self.s_stereo.mid_only_flags[nfe];
                        silk_stereo_lr_to_ms(
                            &mut self.s_stereo,
                            &mut s0.s_cmn.input_buf[..],
                            &mut s1.s_cmn.input_buf[..],
                            &mut ix,
                            &mut mid_only,
                            &mut ms_target_rates_bps,
                            target_rate_bps,
                            s0.s_cmn.speech_activity_q8,
                            enc_control.to_mono,
                            s0.s_cmn.fs_khz,
                            s0.s_cmn.frame_length,
                        );
                        self.s_stereo.pred_ix[nfe] = ix;
                        self.s_stereo.mid_only_flags[nfe] = mid_only;
                    }
                    if self.s_stereo.mid_only_flags[nfe] == 0 {
                        // Reset side channel encoder memory for first frame with side coding
                        if self.prev_decode_only_middle == 1 {
                            let s1 = &mut self.state_fxx[1];
                            s1.s_shape = SilkShapeStateFlp::default();
                            s1.s_cmn.s_nsq = SilkNsqState::new();
                            s1.s_cmn.prev_nlsfq_q15 = [0; MAX_LPC_ORDER as usize];
                            s1.s_cmn.s_lp.in_lp_state = [0; 2];
                            s1.s_cmn.prev_lag = 100;
                            s1.s_cmn.s_nsq.lag_prev = 100;
                            s1.s_shape.last_gain_index = 10;
                            s1.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
                            s1.s_cmn.s_nsq.prev_gain_q16 = 65536;
                            s1.s_cmn.first_frame_after_reset = 1;
                        }
                        silk_encode_do_vad_flp(&mut self.state_fxx[1], activity);
                    } else {
                        self.state_fxx[1].s_cmn.vad_flags[nfe] = 0;
                    }
                    if prefill_flag == 0 {
                        silk_stereo_encode_pred(ps_range_enc, &self.s_stereo.pred_ix[nfe]);
                        if self.state_fxx[1].s_cmn.vad_flags[nfe] == 0 {
                            silk_stereo_encode_mid_only(
                                ps_range_enc,
                                self.s_stereo.mid_only_flags[nfe],
                            );
                        }
                    }
                } else {
                    // Buffering
                    let s0 = &mut self.state_fxx[0].s_cmn;
                    s0.input_buf[..2].copy_from_slice(&self.s_stereo.s_mid);
                    let fl = s0.frame_length as usize;
                    self.s_stereo
                        .s_mid
                        .copy_from_slice(&s0.input_buf[fl..fl + 2]);
                }
                silk_encode_do_vad_flp(&mut self.state_fxx[0], activity);

                // Encode
                for n in 0..n_ch_int as usize {
                    // Handling rate constraints
                    let mut max_bits = enc_control.max_bits;
                    if tot_blocks == 2 && curr_block == 0 {
                        max_bits = max_bits * 3 / 5;
                    } else if tot_blocks == 3 {
                        if curr_block == 0 {
                            max_bits = max_bits * 2 / 5;
                        } else if curr_block == 1 {
                            max_bits = max_bits * 3 / 4;
                        }
                    }
                    let mut use_cbr =
                        (enc_control.use_cbr != 0 && curr_block == tot_blocks - 1) as i32;

                    let channel_rate_bps;
                    if n_ch_int == 1 {
                        channel_rate_bps = target_rate_bps;
                    } else {
                        channel_rate_bps = ms_target_rates_bps[n];
                        if n == 0 && ms_target_rates_bps[1] > 0 {
                            use_cbr = 0;
                            // Give mid up to 1/2 of the max bits for that frame
                            max_bits -= enc_control.max_bits / (tot_blocks * 2);
                        }
                    }

                    if channel_rate_bps > 0 {
                        silk_control_snr(&mut self.state_fxx[n].s_cmn, channel_rate_bps);

                        // Use independent coding if no previous frame available
                        let cond_coding =
                            if self.state_fxx[0].s_cmn.n_frames_encoded - n as i32 <= 0 {
                                CODE_INDEPENDENTLY
                            } else if n > 0 && self.prev_decode_only_middle != 0 {
                                // If we skipped a side frame in this packet, we don't need LTP
                                // scaling; the LTP state is well-defined.
                                CODE_INDEPENDENTLY_NO_LTP_SCALING
                            } else {
                                CODE_CONDITIONALLY
                            };
                        ret = silk_encode_frame_flp(
                            &mut self.state_fxx[n],
                            n_bytes_out,
                            ps_range_enc,
                            cond_coding,
                            max_bits,
                            use_cbr,
                        );
                        debug_assert!(ret == 0);
                    }
                    let s = &mut self.state_fxx[n].s_cmn;
                    s.controlled_since_last_payload = 0;
                    s.input_buf_ix = 0;
                    s.n_frames_encoded += 1;
                }
                self.prev_decode_only_middle = self.s_stereo.mid_only_flags
                    [self.state_fxx[0].s_cmn.n_frames_encoded as usize - 1]
                    as i32;

                // Insert VAD and FEC flags at beginning of bitstream
                if *n_bytes_out > 0
                    && self.state_fxx[0].s_cmn.n_frames_encoded
                        == self.state_fxx[0].s_cmn.n_frames_per_packet
                {
                    let mut flags = 0i32;
                    for n in 0..n_ch_int as usize {
                        let s = &self.state_fxx[n].s_cmn;
                        for i in 0..s.n_frames_per_packet as usize {
                            flags = silk_lshift(flags, 1);
                            flags |= s.vad_flags[i] as i32;
                        }
                        flags = silk_lshift(flags, 1);
                        flags |= s.lbrr_flag as i32;
                    }
                    if prefill_flag == 0 {
                        ps_range_enc.patch_initial_bits(
                            flags as u32,
                            ((self.state_fxx[0].s_cmn.n_frames_per_packet + 1) * n_ch_int) as u32,
                        );
                    }

                    // Return zero bytes if all channels DTXed
                    if self.state_fxx[0].s_cmn.in_dtx != 0
                        && (n_ch_int == 1 || self.state_fxx[1].s_cmn.in_dtx != 0)
                    {
                        *n_bytes_out = 0;
                    }

                    self.n_bits_exceeded += *n_bytes_out * 8;
                    self.n_bits_exceeded -= silk_div32_16(
                        silk_mul(enc_control.bit_rate, enc_control.payload_size_ms),
                        1000,
                    );
                    self.n_bits_exceeded = silk_limit(self.n_bits_exceeded, 0, 10000);

                    // Update flag indicating if bandwidth switching is allowed
                    let speech_act_thr_for_switch_q8 = silk_smlawb(
                        silk_fix_const(SPEECH_ACTIVITY_DTX_THRES as f64, 8),
                        speech_act_switch_coef_q24(),
                        self.time_since_switch_allowed_ms,
                    );
                    if self.state_fxx[0].s_cmn.speech_activity_q8 < speech_act_thr_for_switch_q8 {
                        self.allow_bandwidth_switch = 1;
                        self.time_since_switch_allowed_ms = 0;
                    } else {
                        self.allow_bandwidth_switch = 0;
                        self.time_since_switch_allowed_ms += enc_control.payload_size_ms;
                    }
                }

                if n_samples_in == 0 {
                    break;
                }
            } else {
                break;
            }
            curr_block += 1;
        }

        self.n_prev_channels_internal = enc_control.n_channels_internal;

        let s0 = &self.state_fxx[0].s_cmn;
        enc_control.allow_bandwidth_switch = self.allow_bandwidth_switch;
        enc_control.in_wb_mode_without_variable_lp = (s0.fs_khz == 16 && s0.s_lp.mode == 0) as i32;
        enc_control.internal_sample_rate = silk_smulbb(s0.fs_khz, 1000);
        enc_control.stereo_width_q14 = if enc_control.to_mono != 0 {
            0
        } else {
            self.s_stereo.smth_width_q14 as i32
        };
        if prefill_flag != 0 {
            enc_control.payload_size_ms = tmp_payload_size_ms;
            enc_control.complexity = tmp_complexity;
            for n in 0..enc_control.n_channels_internal as usize {
                self.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
                self.state_fxx[n].s_cmn.prefill_flag = 0;
            }
        }

        let s0 = &self.state_fxx[0].s_cmn;
        enc_control.signal_type = s0.indices.signal_type as i32;
        enc_control.offset = SILK_QUANTIZATION_OFFSETS_Q10[(s0.indices.signal_type >> 1) as usize]
            [s0.indices.quant_offset_type as usize] as i32;
        ret
    }
}

impl Default for SilkEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// `SILK_FIX_CONST( ( 1 - SPEECH_ACTIVITY_DTX_THRES ) / MAX_BANDWIDTH_SWITCH_DELAY_MS, 16 + 8 )`,
/// with the C float arithmetic (`(1 - 0.05f) / 5000` is a float division).
#[inline]
fn speech_act_switch_coef_q24() -> i32 {
    let c = (1.0f32 - SPEECH_ACTIVITY_DTX_THRES) / MAX_BANDWIDTH_SWITCH_DELAY_MS as f32;
    ((c * (1i64 << 24) as f32) as f64 + 0.5) as i32
}

/// Port of `silk/enc_API.c:silk_Get_Encoder_Size` — number of bytes in the SILK encoder state
/// (the Rust struct size; the second channel state is skipped for mono as in C).
pub const fn silk_get_encoder_size(enc_size_bytes: &mut i32, channels: i32) -> i32 {
    *enc_size_bytes = size_of::<SilkEncoder>() as i32;
    // Skip second encoder state for mono.
    if channels == 1 {
        *enc_size_bytes -= size_of::<SilkEncoderStateFlp>() as i32;
    }
    SILK_NO_ERROR
}

// ---------------------------------------------------------------------------------------------
// init_encoder.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/init_encoder.c:silk_init_encoder` — initialize the SILK encoder state.
pub fn silk_init_encoder(ps_enc: &mut SilkEncoderStateFlp) -> i32 {
    let mut ret = 0;

    // Clear the entire encoder state
    *ps_enc = SilkEncoderStateFlp::new();

    ps_enc.s_cmn.variable_hp_smth1_q15 = silk_lshift(
        silk_lin2log(silk_fix_const(VARIABLE_HP_MIN_CUTOFF_HZ as f64, 16)) - (16 << 7),
        8,
    );
    ps_enc.s_cmn.variable_hp_smth2_q15 = ps_enc.s_cmn.variable_hp_smth1_q15;

    // Used to deactivate LSF interpolation, pitch prediction
    ps_enc.s_cmn.first_frame_after_reset = 1;

    // Initialize Silk VAD
    ret += silk_vad_init(&mut ps_enc.s_cmn.s_vad);

    // DNN: ENABLE_DRED (init_encoder.c only includes dred_encoder.h) not ported yet.
    ret
}

// ---------------------------------------------------------------------------------------------
// control_codec.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/control_codec.c:silk_control_encoder` — control the SILK encoder.
pub fn silk_control_encoder(
    ps_enc: &mut SilkEncoderStateFlp,
    enc_control: &mut SilkEncControlStruct,
    allow_bw_switch: i32,
    channel_nb: i32,
    force_fs_khz: i32,
) -> i32 {
    let mut ret = 0;
    {
        let s = &mut ps_enc.s_cmn;
        s.use_dtx = enc_control.use_dtx;
        s.use_cbr = enc_control.use_cbr;
        s.api_fs_hz = enc_control.api_sample_rate;
        s.max_internal_fs_hz = enc_control.max_internal_sample_rate;
        s.min_internal_fs_hz = enc_control.min_internal_sample_rate;
        s.desired_internal_fs_hz = enc_control.desired_internal_sample_rate;
        s.use_in_band_fec = enc_control.use_in_band_fec;
        s.n_channels_api = enc_control.n_channels_api;
        s.n_channels_internal = enc_control.n_channels_internal;
        s.allow_bandwidth_switch = allow_bw_switch;
        s.channel_nb = channel_nb;
    }

    if ps_enc.s_cmn.controlled_since_last_payload != 0 && ps_enc.s_cmn.prefill_flag == 0 {
        if ps_enc.s_cmn.api_fs_hz != ps_enc.s_cmn.prev_api_fs_hz && ps_enc.s_cmn.fs_khz > 0 {
            // Change in API sampling rate in the middle of encoding a packet
            let fs_khz = ps_enc.s_cmn.fs_khz;
            ret += silk_setup_resamplers(ps_enc, fs_khz);
        }
        return ret;
    }

    // Beyond this point we know that there are no previously coded frames in the payload
    // buffer

    // Determine internal sampling rate
    let mut fs_khz = silk_control_audio_bandwidth(&mut ps_enc.s_cmn, enc_control);
    if force_fs_khz != 0 {
        fs_khz = force_fs_khz;
    }
    // Prepare resampler and buffered data
    ret += silk_setup_resamplers(ps_enc, fs_khz);

    // Set internal sampling frequency
    ret += silk_setup_fs(ps_enc, fs_khz, enc_control.payload_size_ms);

    // Set encoding complexity
    ret += silk_setup_complexity(&mut ps_enc.s_cmn, enc_control.complexity);

    // Set packet loss rate measured by farend
    ps_enc.s_cmn.packet_loss_perc = enc_control.packet_loss_percentage;

    // Set LBRR usage
    ret += silk_setup_lbrr(&mut ps_enc.s_cmn, enc_control);

    ps_enc.s_cmn.controlled_since_last_payload = 1;

    ret
}

/// Port of the static `silk/control_codec.c:silk_setup_resamplers` — prepare the input
/// resampler and re-sample the buffered analysis data (`x_buf`) when the internal or API
/// sampling rate changes.
pub fn silk_setup_resamplers(ps_enc: &mut SilkEncoderStateFlp, fs_khz: i32) -> i32 {
    let mut ret = SILK_NO_ERROR;
    let s = &mut ps_enc.s_cmn;

    if s.fs_khz != fs_khz || s.prev_api_fs_hz != s.api_fs_hz {
        if s.fs_khz == 0 {
            // Initialize the resampler for enc_API.c preparing resampling from API_fs_Hz to
            // fs_kHz
            ret += silk_resampler_init(&mut s.resampler_state, s.api_fs_hz, fs_khz * 1000, 1);
        } else {
            const XBUF_MAX: usize = SETUP_BUF_MS * crate::silk::define::MAX_FS_KHZ as usize;
            let mut x_buf_fix = [0i16; XBUF_MAX];
            let mut x_buf_api_fs_hz = [0i16; SETUP_BUF_MS * MAX_API_FS_KHZ as usize];
            let mut temp_resampler_state = SilkResamplerState::default();

            let buf_length_ms = silk_lshift(s.nb_subfr * 5, 1) + LA_SHAPE_MS;
            let old_buf_samples = buf_length_ms * s.fs_khz;

            // FIXED_POINT: not ported (float build) — x_bufFIX aliases psEnc->x_buf there.
            let new_buf_samples = buf_length_ms * fs_khz;
            silk_float2short_array(&mut x_buf_fix, &ps_enc.x_buf, old_buf_samples as usize);

            // Initialize resampler for temporary resampling of x_buf data to API_fs_Hz
            ret += silk_resampler_init(
                &mut temp_resampler_state,
                silk_smulbb(s.fs_khz, 1000),
                s.api_fs_hz,
                0,
            );

            // Calculate number of samples to temporarily upsample
            let api_buf_samples = buf_length_ms * silk_div32_16(s.api_fs_hz, 1000);

            // Temporary resampling of x_buf data to API_fs_Hz
            ret += silk_resampler(
                &mut temp_resampler_state,
                &mut x_buf_api_fs_hz,
                &x_buf_fix,
                old_buf_samples,
            );

            // Initialize the resampler for enc_API.c preparing resampling from API_fs_Hz to
            // fs_kHz
            ret += silk_resampler_init(
                &mut s.resampler_state,
                s.api_fs_hz,
                silk_smulbb(fs_khz, 1000),
                1,
            );

            // Correct resampler state by resampling buffered data from API_fs_Hz to fs_kHz
            ret += silk_resampler(
                &mut s.resampler_state,
                &mut x_buf_fix,
                &x_buf_api_fs_hz,
                api_buf_samples,
            );

            silk_short2float_array(&mut ps_enc.x_buf, &x_buf_fix, new_buf_samples as usize);
        }
    }

    ps_enc.s_cmn.prev_api_fs_hz = ps_enc.s_cmn.api_fs_hz;

    ret
}

/// Port of the static `silk/control_codec.c:silk_setup_fs` — set packet size and internal
/// sampling frequency.
pub fn silk_setup_fs(ps_enc: &mut SilkEncoderStateFlp, fs_khz: i32, packet_size_ms: i32) -> i32 {
    let mut ret = SILK_NO_ERROR;
    let s = &mut ps_enc.s_cmn;

    // Set packet size
    if packet_size_ms != s.packet_size_ms {
        if packet_size_ms != 10
            && packet_size_ms != 20
            && packet_size_ms != 40
            && packet_size_ms != 60
        {
            ret = SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
        }
        if packet_size_ms <= 10 {
            s.n_frames_per_packet = 1;
            s.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
            s.frame_length = silk_smulbb(packet_size_ms, fs_khz);
            s.pitch_lpc_win_length = silk_smulbb(FIND_PITCH_LPC_WIN_MS_2_SF, fs_khz);
            if s.fs_khz == 8 {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            } else {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
            }
        } else {
            s.n_frames_per_packet = silk_div32_16(packet_size_ms, MAX_FRAME_LENGTH_MS);
            s.nb_subfr = MAX_NB_SUBFR;
            s.frame_length = silk_smulbb(20, fs_khz);
            s.pitch_lpc_win_length = silk_smulbb(FIND_PITCH_LPC_WIN_MS, fs_khz);
            if s.fs_khz == 8 {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
            }
        }
        s.packet_size_ms = packet_size_ms;
        s.target_rate_bps = 0; // trigger new SNR computation
    }

    // Set internal sampling frequency
    debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
    debug_assert!(s.nb_subfr == 2 || s.nb_subfr == 4);
    if s.fs_khz != fs_khz {
        // reset part of the state
        ps_enc.s_shape = SilkShapeStateFlp::default();
        let s = &mut ps_enc.s_cmn;
        s.s_nsq = SilkNsqState::new();
        s.prev_nlsfq_q15 = [0; MAX_LPC_ORDER as usize];
        s.s_lp.in_lp_state = [0; 2];
        s.input_buf_ix = 0;
        s.n_frames_encoded = 0;
        s.target_rate_bps = 0; // trigger new SNR computation

        // Initialize non-zero parameters
        s.prev_lag = 100;
        s.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        let s = &mut ps_enc.s_cmn;
        s.s_nsq.lag_prev = 100;
        s.s_nsq.prev_gain_q16 = 65536;
        s.prev_signal_type = TYPE_NO_VOICE_ACTIVITY as i8;

        s.fs_khz = fs_khz;
        if s.fs_khz == 8 {
            if s.nb_subfr == MAX_NB_SUBFR {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if s.nb_subfr == MAX_NB_SUBFR {
            s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            s.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
        if s.fs_khz == 8 || s.fs_khz == 12 {
            s.predict_lpc_order = MIN_LPC_ORDER;
            s.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
        } else {
            s.predict_lpc_order = MAX_LPC_ORDER;
            s.ps_nlsf_cb = &SILK_NLSF_CB_WB;
        }
        s.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz;
        s.frame_length = silk_smulbb(s.subfr_length, s.nb_subfr);
        s.ltp_mem_length = silk_smulbb(LTP_MEM_LENGTH_MS, fs_khz);
        s.la_pitch = silk_smulbb(LA_PITCH_MS, fs_khz);
        s.max_pitch_lag = silk_smulbb(18, fs_khz);
        if s.nb_subfr == MAX_NB_SUBFR {
            s.pitch_lpc_win_length = silk_smulbb(FIND_PITCH_LPC_WIN_MS, fs_khz);
        } else {
            s.pitch_lpc_win_length = silk_smulbb(FIND_PITCH_LPC_WIN_MS_2_SF, fs_khz);
        }
        if s.fs_khz == 16 {
            s.pitch_lag_low_bits_icdf = &SILK_UNIFORM8_ICDF;
        } else if s.fs_khz == 12 {
            s.pitch_lag_low_bits_icdf = &SILK_UNIFORM6_ICDF;
        } else {
            s.pitch_lag_low_bits_icdf = &SILK_UNIFORM4_ICDF;
        }
    }

    // Check that settings are valid
    debug_assert!(ps_enc.s_cmn.subfr_length * ps_enc.s_cmn.nb_subfr == ps_enc.s_cmn.frame_length);

    ret
}

/// Port of the static `silk/control_codec.c:silk_setup_complexity` — set encoding complexity.
pub fn silk_setup_complexity(
    ps_enc_c: &mut crate::silk::structs::SilkEncoderState,
    complexity: i32,
) -> i32 {
    let ret = 0;
    let warping = silk_fix_const(WARPING_MULTIPLIER as f64, 16);

    // Set encoding complexity
    debug_assert!((0..=10).contains(&complexity));
    let s = ps_enc_c;
    if complexity < 1 {
        s.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.8, 16);
        s.pitch_estimation_lpc_order = 6;
        s.shaping_lpc_order = 12;
        s.la_shape = 3 * s.fs_khz;
        s.n_states_delayed_decision = 1;
        s.use_interpolated_nlsfs = 0;
        s.nlsf_msvq_survivors = 2;
        s.warping_q16 = 0;
    } else if complexity < 2 {
        s.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.76, 16);
        s.pitch_estimation_lpc_order = 8;
        s.shaping_lpc_order = 14;
        s.la_shape = 5 * s.fs_khz;
        s.n_states_delayed_decision = 1;
        s.use_interpolated_nlsfs = 0;
        s.nlsf_msvq_survivors = 3;
        s.warping_q16 = 0;
    } else if complexity < 3 {
        s.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.8, 16);
        s.pitch_estimation_lpc_order = 6;
        s.shaping_lpc_order = 12;
        s.la_shape = 3 * s.fs_khz;
        s.n_states_delayed_decision = 2;
        s.use_interpolated_nlsfs = 0;
        s.nlsf_msvq_survivors = 2;
        s.warping_q16 = 0;
    } else if complexity < 4 {
        s.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.76, 16);
        s.pitch_estimation_lpc_order = 8;
        s.shaping_lpc_order = 14;
        s.la_shape = 5 * s.fs_khz;
        s.n_states_delayed_decision = 2;
        s.use_interpolated_nlsfs = 0;
        s.nlsf_msvq_survivors = 4;
        s.warping_q16 = 0;
    } else if complexity < 6 {
        s.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.74, 16);
        s.pitch_estimation_lpc_order = 10;
        s.shaping_lpc_order = 16;
        s.la_shape = 5 * s.fs_khz;
        s.n_states_delayed_decision = 2;
        s.use_interpolated_nlsfs = 1;
        s.nlsf_msvq_survivors = 6;
        s.warping_q16 = s.fs_khz * warping;
    } else if complexity < 8 {
        s.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.72, 16);
        s.pitch_estimation_lpc_order = 12;
        s.shaping_lpc_order = 20;
        s.la_shape = 5 * s.fs_khz;
        s.n_states_delayed_decision = 3;
        s.use_interpolated_nlsfs = 1;
        s.nlsf_msvq_survivors = 8;
        s.warping_q16 = s.fs_khz * warping;
    } else {
        s.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX;
        s.pitch_estimation_threshold_q16 = silk_fix_const(0.7, 16);
        s.pitch_estimation_lpc_order = 16;
        s.shaping_lpc_order = 24;
        s.la_shape = 5 * s.fs_khz;
        s.n_states_delayed_decision = MAX_DEL_DEC_STATES;
        s.use_interpolated_nlsfs = 1;
        s.nlsf_msvq_survivors = 16;
        s.warping_q16 = s.fs_khz * warping;
    }

    // Do not allow higher pitch estimation LPC order than predict LPC order
    s.pitch_estimation_lpc_order = silk_min_int(s.pitch_estimation_lpc_order, s.predict_lpc_order);
    s.shape_win_length = SUB_FRAME_LENGTH_MS * s.fs_khz + 2 * s.la_shape;
    s.complexity = complexity;

    debug_assert!(s.pitch_estimation_lpc_order <= MAX_FIND_PITCH_LPC_ORDER);
    debug_assert!(s.shaping_lpc_order <= MAX_SHAPE_LPC_ORDER);
    debug_assert!(s.n_states_delayed_decision <= MAX_DEL_DEC_STATES);
    debug_assert!(s.warping_q16 <= 32767);
    debug_assert!(s.la_shape <= LA_SHAPE_MAX);
    debug_assert!(s.shape_win_length <= SHAPE_LPC_WIN_MAX);

    ret
}

/// Port of the static inline `silk/control_codec.c:silk_setup_LBRR` — set LBRR usage.
pub const fn silk_setup_lbrr(
    ps_enc_c: &mut crate::silk::structs::SilkEncoderState,
    enc_control: &SilkEncControlStruct,
) -> i32 {
    let ret = SILK_NO_ERROR;

    let lbrr_in_previous_packet = ps_enc_c.lbrr_enabled;
    ps_enc_c.lbrr_enabled = enc_control.lbrr_coded;
    if ps_enc_c.lbrr_enabled != 0 {
        // Set gain increase for coding LBRR excitation
        if lbrr_in_previous_packet == 0 {
            // Previous packet did not have LBRR, and was therefore coded at a higher bitrate
            ps_enc_c.lbrr_gain_increases = 7;
        } else {
            ps_enc_c.lbrr_gain_increases = silk_max_int(
                7 - silk_smulwb(ps_enc_c.packet_loss_perc, silk_fix_const(0.2, 16)),
                3,
            );
        }
    }

    ret
}

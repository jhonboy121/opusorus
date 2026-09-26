//! Port of `silk/structs.h` and `silk/control.h`.
//!
//! Field names are the C names converted to snake case (`prev_gain_Q16` → `prev_gain_q16`,
//! `sLPC_Q14_buf` → `s_lpc_q14_buf`). Array sizes use the C defines. C pointer fields that point
//! into constant tables become `&'static` references; where C initializes them to `NULL` (by
//! `memset`), the Rust default is an empty table / the NB-MB codebook, which is never read before
//! the owning `*_set_fs` / `control_codec` function assigns the real table (same as in C).
//! The `arch` fields are dropped (no run-time CPU dispatch in the port).

use crate::silk::define::{
    LTP_ORDER, MAX_FRAME_LENGTH, MAX_FRAMES_PER_PACKET, MAX_LPC_ORDER, MAX_NB_SUBFR,
    MAX_SHAPE_LPC_ORDER, MAX_SUB_FRAME_LENGTH, NSQ_LPC_BUF_LENGTH, VAD_N_BANDS,
};
use crate::silk::resampler::SilkResamplerState;
use crate::silk::tables::SILK_NLSF_CB_NB_MB;

const MFL: usize = MAX_FRAME_LENGTH as usize;
const MSFL: usize = MAX_SUB_FRAME_LENGTH as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MFPP: usize = MAX_FRAMES_PER_PACKET as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const VADB: usize = VAD_N_BANDS as usize;

// ---------------------------------------------------------------------------------------------
// silk/control.h
// ---------------------------------------------------------------------------------------------

/// `FLAG_DECODE_NORMAL`: decoder API flag.
pub const FLAG_DECODE_NORMAL: i32 = 0;
/// `FLAG_PACKET_LOST`: decoder API flag.
pub const FLAG_PACKET_LOST: i32 = 1;
/// `FLAG_DECODE_LBRR`: decoder API flag.
pub const FLAG_DECODE_LBRR: i32 = 2;

/// `silk_EncControlStruct`: structure for controlling encoder operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkEncControlStruct {
    /// I: number of channels; 1/2.
    pub n_channels_api: i32,
    /// I: number of channels; 1/2.
    pub n_channels_internal: i32,
    /// I: input signal sampling rate in Hertz; 8000/12000/16000/24000/32000/44100/48000.
    pub api_sample_rate: i32,
    /// I: maximum internal sampling rate in Hertz; 8000/12000/16000.
    pub max_internal_sample_rate: i32,
    /// I: minimum internal sampling rate in Hertz; 8000/12000/16000.
    pub min_internal_sample_rate: i32,
    /// I: soft request for internal sampling rate in Hertz; 8000/12000/16000.
    pub desired_internal_sample_rate: i32,
    /// I: number of samples per packet in milliseconds; 10/20/40/60.
    pub payload_size_ms: i32,
    /// I: bitrate during active speech in bits/second; internally limited.
    pub bit_rate: i32,
    /// I: uplink packet loss in percent (0-100).
    pub packet_loss_percentage: i32,
    /// I: complexity mode; 0 is lowest, 10 is highest complexity.
    pub complexity: i32,
    /// I: flag to enable in-band Forward Error Correction (FEC); 0/1.
    pub use_in_band_fec: i32,
    /// I: flag to enable in-band Deep REDundancy (DRED); 0/1.
    pub use_dred: i32,
    /// I: flag to actually code in-band FEC in the current packet; 0/1.
    pub lbrr_coded: i32,
    /// I: flag to enable discontinuous transmission (DTX); 0/1.
    pub use_dtx: i32,
    /// I: flag to use constant bitrate.
    pub use_cbr: i32,
    /// I: maximum number of bits allowed for the frame.
    pub max_bits: i32,
    /// I: causes a smooth downmix to mono.
    pub to_mono: i32,
    /// I: Opus encoder is allowing us to switch bandwidth.
    pub opus_can_switch: i32,
    /// I: make frames as independent as possible (but still use LPC).
    pub reduced_dependency: i32,
    /// O: internal sampling rate used, in Hertz; 8000/12000/16000.
    pub internal_sample_rate: i32,
    /// O: flag that bandwidth switching is allowed (because low voice activity).
    pub allow_bandwidth_switch: i32,
    /// O: flag that SILK runs in WB mode without variable LP filter.
    pub in_wb_mode_without_variable_lp: i32,
    /// O: stereo width.
    pub stereo_width_q14: i32,
    /// O: tells the Opus encoder we're ready to switch.
    pub switch_ready: i32,
    /// O: SILK signal type.
    pub signal_type: i32,
    /// O: SILK offset (dithering).
    pub offset: i32,
}

/// `silk_DecControlStruct`: structure for controlling decoder operation and reading status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkDecControlStruct {
    /// I: number of channels; 1/2.
    pub n_channels_api: i32,
    /// I: number of channels; 1/2.
    pub n_channels_internal: i32,
    /// I: output signal sampling rate in Hertz; 8000/12000/16000/24000/32000/44100/48000.
    pub api_sample_rate: i32,
    /// I: internal sampling rate used, in Hertz; 8000/12000/16000.
    pub internal_sample_rate: i32,
    /// I: number of samples per packet in milliseconds; 10/20/40/60.
    pub payload_size_ms: i32,
    /// O: pitch lag of previous frame (0 if unvoiced), measured in samples at 48 kHz.
    pub prev_pitch_lag: i32,
    /// I: enable Deep PLC.
    pub enable_deep_plc: i32,
    // DNN: ENABLE_OSCE (osce_method) / ENABLE_OSCE_BWE (enable_osce_bwe, osce_extended_mode,
    // prev_osce_extended_mode) not ported yet.
}

// ---------------------------------------------------------------------------------------------
// silk/structs.h
// ---------------------------------------------------------------------------------------------

/// `silk_nsq_state`: noise shaping quantization state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SilkNsqState {
    /// Buffer for quantized output signal.
    pub xq: [i16; 2 * MFL],
    pub s_ltp_shp_q14: [i32; 2 * MFL],
    pub s_lpc_q14: [i32; MSFL + NSQ_LPC_BUF_LENGTH as usize],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER as usize],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
}

impl SilkNsqState {
    /// All-zero state (C `memset(0)`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            xq: [0; 2 * MFL],
            s_ltp_shp_q14: [0; 2 * MFL],
            s_lpc_q14: [0; MSFL + NSQ_LPC_BUF_LENGTH as usize],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER as usize],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 0,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 0,
            rewhite_flag: 0,
        }
    }
}

impl Default for SilkNsqState {
    fn default() -> Self {
        Self::new()
    }
}

/// `silk_VAD_state`: voice activity detector state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkVadState {
    /// Analysis filterbank state: 0-8 kHz.
    pub ana_state: [i32; 2],
    /// Analysis filterbank state: 0-4 kHz.
    pub ana_state1: [i32; 2],
    /// Analysis filterbank state: 0-2 kHz.
    pub ana_state2: [i32; 2],
    /// Subframe energies.
    pub xnrg_subfr: [i32; VADB],
    /// Smoothed energy level in each band.
    pub nrg_ratio_smth_q8: [i32; VADB],
    /// State of differentiator in the lowest band.
    pub hp_state: i16,
    /// Noise energy level in each band.
    pub nl: [i32; VADB],
    /// Inverse noise energy level in each band.
    pub inv_nl: [i32; VADB],
    /// Noise level estimator bias/offset.
    pub noise_level_bias: [i32; VADB],
    /// Frame counter used in the initial phase.
    pub counter: i32,
}

/// `silk_LP_state`: variable cut-off low-pass filter state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkLpState {
    /// Low pass filter state.
    pub in_lp_state: [i32; 2],
    /// Counter which is mapped to a cut-off frequency.
    pub transition_frame_no: i32,
    /// Operating mode, <0: switch down, >0: switch up; 0: do nothing.
    pub mode: i32,
    /// If non-zero, holds the last sampling rate before a bandwidth switching reset.
    pub saved_fs_khz: i32,
}

/// `silk_NLSF_CB_struct`: structure containing an NLSF codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkNlsfCbStruct {
    pub n_vectors: i16,
    pub order: i16,
    pub quant_step_size_q16: i16,
    pub inv_quant_step_size_q6: i16,
    pub cb1_nlsf_q8: &'static [u8],
    pub cb1_wght_q9: &'static [i16],
    pub cb1_icdf: &'static [u8],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_icdf: &'static [u8],
    pub ec_rates_q5: &'static [u8],
    pub delta_min_q15: &'static [i16],
}

/// `stereo_enc_state`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MFPP],
    pub mid_only_flags: [i8; MFPP],
}

/// `stereo_dec_state`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StereoDecState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

/// `SideInfoIndices`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MNSF],
    pub ltp_index: [i8; MNSF],
    pub nlsf_indices: [i8; MLPC + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

impl SideInfoIndices {
    /// All-zero indices (C `memset(0)`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gains_indices: [0; MNSF],
            ltp_index: [0; MNSF],
            nlsf_indices: [0; MLPC + 1],
            lag_index: 0,
            contour_index: 0,
            signal_type: 0,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 0,
            per_index: 0,
            ltp_scale_index: 0,
            seed: 0,
        }
    }
}

/// `silk_encoder_state`: encoder state common to the fixed and float encoders.
#[derive(Debug, Clone)]
pub struct SilkEncoderState {
    /// High pass filter state.
    pub in_hp_state: [i32; 2],
    /// State of first smoother.
    pub variable_hp_smth1_q15: i32,
    /// State of second smoother.
    pub variable_hp_smth2_q15: i32,
    /// Low pass filter state.
    pub s_lp: SilkLpState,
    /// Voice activity detector state.
    pub s_vad: SilkVadState,
    /// Noise Shape Quantizer State.
    pub s_nsq: SilkNsqState,
    /// Previously quantized NLSF vector.
    pub prev_nlsfq_q15: [i16; MLPC],
    /// Speech activity.
    pub speech_activity_q8: i32,
    /// Flag indicating that switching of internal bandwidth is allowed.
    pub allow_bandwidth_switch: i32,
    pub lbrr_prev_last_gain_index: i8,
    pub prev_signal_type: i8,
    pub prev_lag: i32,
    pub pitch_lpc_win_length: i32,
    /// Highest possible pitch lag (samples).
    pub max_pitch_lag: i32,
    /// API sampling frequency (Hz).
    pub api_fs_hz: i32,
    /// Previous API sampling frequency (Hz).
    pub prev_api_fs_hz: i32,
    /// Maximum internal sampling frequency (Hz).
    pub max_internal_fs_hz: i32,
    /// Minimum internal sampling frequency (Hz).
    pub min_internal_fs_hz: i32,
    /// Soft request for internal sampling frequency (Hz).
    pub desired_internal_fs_hz: i32,
    /// Internal sampling frequency (kHz).
    pub fs_khz: i32,
    /// Number of 5 ms subframes in a frame.
    pub nb_subfr: i32,
    /// Frame length (samples).
    pub frame_length: i32,
    /// Subframe length (samples).
    pub subfr_length: i32,
    /// Length of LTP memory.
    pub ltp_mem_length: i32,
    /// Look-ahead for pitch analysis (samples).
    pub la_pitch: i32,
    /// Look-ahead for noise shape analysis (samples).
    pub la_shape: i32,
    /// Window length for noise shape analysis (samples).
    pub shape_win_length: i32,
    /// Target bitrate (bps).
    pub target_rate_bps: i32,
    /// Number of milliseconds to put in each packet.
    pub packet_size_ms: i32,
    /// Packet loss rate measured by farend.
    pub packet_loss_perc: i32,
    pub frame_counter: i32,
    /// Complexity setting.
    pub complexity: i32,
    /// Number of states in delayed decision quantization.
    pub n_states_delayed_decision: i32,
    /// Flag for using NLSF interpolation.
    pub use_interpolated_nlsfs: i32,
    /// Filter order for noise shaping filters.
    pub shaping_lpc_order: i32,
    /// Filter order for prediction filters.
    pub predict_lpc_order: i32,
    /// Complexity level for pitch estimator.
    pub pitch_estimation_complexity: i32,
    /// Whitening filter order for pitch estimator.
    pub pitch_estimation_lpc_order: i32,
    /// Threshold for pitch estimator.
    pub pitch_estimation_threshold_q16: i32,
    /// Cumulative max prediction gain.
    pub sum_log_gain_q7: i32,
    /// Number of survivors in NLSF MSVQ.
    pub nlsf_msvq_survivors: i32,
    /// Flag for deactivating NLSF interpolation, pitch prediction.
    pub first_frame_after_reset: i32,
    /// Flag for ensuring codec_control only runs once per packet.
    pub controlled_since_last_payload: i32,
    /// Warping parameter for warped noise shaping.
    pub warping_q16: i32,
    /// Flag to enable constant bitrate.
    pub use_cbr: i32,
    /// Flag to indicate that only buffers are prefilled, no coding.
    pub prefill_flag: i32,
    /// iCDF table for low bits of pitch lag index (C pointer; `NULL` → empty).
    pub pitch_lag_low_bits_icdf: &'static [u8],
    /// iCDF table for pitch contour index (C pointer; `NULL` → empty).
    pub pitch_contour_icdf: &'static [u8],
    /// NLSF codebook (C pointer; `NULL` → NB/MB codebook, never read before being set).
    pub ps_nlsf_cb: &'static SilkNlsfCbStruct,
    pub input_quality_bands_q15: [i32; VADB],
    pub input_tilt_q15: i32,
    /// Quality setting.
    pub snr_db_q7: i32,
    pub vad_flags: [i8; MFPP],
    pub lbrr_flag: i8,
    pub lbrr_flags: [i32; MFPP],
    pub indices: SideInfoIndices,
    pub pulses: [i8; MFL],
    // `arch`: dropped (no run-time dispatch).
    /// Buffer containing input signal.
    pub input_buf: [i16; MFL + 2],
    pub input_buf_ix: i32,
    pub n_frames_per_packet: i32,
    /// Number of frames analyzed in current packet.
    pub n_frames_encoded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub channel_nb: i32,
    /// Parameters for LTP scaling control.
    pub frames_since_onset: i32,
    /// Specifically for entropy coding.
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub resampler_state: SilkResamplerState,
    /// Flag to enable DTX.
    pub use_dtx: i32,
    /// Flag to signal DTX period.
    pub in_dtx: i32,
    /// Counts consecutive nonactive frames, used by DTX.
    pub no_speech_counter: i32,
    /// Saves the API setting for query.
    pub use_in_band_fec: i32,
    /// Depends on useInBandFRC, bitrate and packet loss rate.
    pub lbrr_enabled: i32,
    /// Gains increment for coding LBRR frames.
    pub lbrr_gain_increases: i32,
    pub indices_lbrr: [SideInfoIndices; MFPP],
    pub pulses_lbrr: [[i8; MFL]; MFPP],
}

impl SilkEncoderState {
    /// All-zero state, like the C `silk_memset(psEnc, 0, sizeof(...))` in `silk_init_encoder`
    /// (the remaining initialization belongs to `silk_init_encoder`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_hp_state: [0; 2],
            variable_hp_smth1_q15: 0,
            variable_hp_smth2_q15: 0,
            s_lp: SilkLpState::default(),
            s_vad: SilkVadState::default(),
            s_nsq: SilkNsqState::new(),
            prev_nlsfq_q15: [0; MLPC],
            speech_activity_q8: 0,
            allow_bandwidth_switch: 0,
            lbrr_prev_last_gain_index: 0,
            prev_signal_type: 0,
            prev_lag: 0,
            pitch_lpc_win_length: 0,
            max_pitch_lag: 0,
            api_fs_hz: 0,
            prev_api_fs_hz: 0,
            max_internal_fs_hz: 0,
            min_internal_fs_hz: 0,
            desired_internal_fs_hz: 0,
            fs_khz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            la_pitch: 0,
            la_shape: 0,
            shape_win_length: 0,
            target_rate_bps: 0,
            packet_size_ms: 0,
            packet_loss_perc: 0,
            frame_counter: 0,
            complexity: 0,
            n_states_delayed_decision: 0,
            use_interpolated_nlsfs: 0,
            shaping_lpc_order: 0,
            predict_lpc_order: 0,
            pitch_estimation_complexity: 0,
            pitch_estimation_lpc_order: 0,
            pitch_estimation_threshold_q16: 0,
            sum_log_gain_q7: 0,
            nlsf_msvq_survivors: 0,
            first_frame_after_reset: 0,
            controlled_since_last_payload: 0,
            warping_q16: 0,
            use_cbr: 0,
            prefill_flag: 0,
            pitch_lag_low_bits_icdf: &[],
            pitch_contour_icdf: &[],
            ps_nlsf_cb: &SILK_NLSF_CB_NB_MB,
            input_quality_bands_q15: [0; VADB],
            input_tilt_q15: 0,
            snr_db_q7: 0,
            vad_flags: [0; MFPP],
            lbrr_flag: 0,
            lbrr_flags: [0; MFPP],
            indices: SideInfoIndices::new(),
            pulses: [0; MFL],
            input_buf: [0; MFL + 2],
            input_buf_ix: 0,
            n_frames_per_packet: 0,
            n_frames_encoded: 0,
            n_channels_api: 0,
            n_channels_internal: 0,
            channel_nb: 0,
            frames_since_onset: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            resampler_state: SilkResamplerState::default(),
            use_dtx: 0,
            in_dtx: 0,
            no_speech_counter: 0,
            use_in_band_fec: 0,
            lbrr_enabled: 0,
            lbrr_gain_increases: 0,
            indices_lbrr: [SideInfoIndices::new(); MFPP],
            pulses_lbrr: [[0; MFL]; MFPP],
        }
    }

    /// C `silk_memset(psEnc, 0, sizeof(...))`.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SilkEncoderState {
    fn default() -> Self {
        Self::new()
    }
}

// DNN: ENABLE_OSCE (silk_OSCE_struct, silk_OSCE_BWE_struct) not ported yet.

/// `silk_PLC_struct`: struct for packet loss concealment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkPlcStruct {
    /// Pitch lag to use for voiced concealment.
    pub pitch_l_q8: i32,
    /// LTP coefficients to use for voiced concealment.
    pub ltp_coef_q14: [i16; LTP_ORDER as usize],
    pub prev_lpc_q12: [i16; MLPC],
    /// Was previous frame lost.
    pub last_frame_lost: i32,
    /// Seed for unvoiced signal generation.
    pub rand_seed: i32,
    /// Scaling of unvoiced random signal.
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
    pub enable_deep_plc: i32,
}

/// `silk_CNG_struct`: struct for comfort noise generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SilkCngStruct {
    pub cng_exc_buf_q14: [i32; MFL],
    pub cng_smth_nlsf_q15: [i16; MLPC],
    pub cng_synth_state: [i32; MLPC],
    pub cng_smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl SilkCngStruct {
    /// All-zero state (C `memset(0)`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cng_exc_buf_q14: [0; MFL],
            cng_smth_nlsf_q15: [0; MLPC],
            cng_synth_state: [0; MLPC],
            cng_smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

impl Default for SilkCngStruct {
    fn default() -> Self {
        Self::new()
    }
}

/// `silk_decoder_state`: decoder state.
#[derive(Debug, Clone)]
pub struct SilkDecoderState {
    // DNN: ENABLE_OSCE (osce) / ENABLE_OSCE_BWE (osce_bwe) not ported yet. In C these fields
    // precede `SILK_DECODER_STATE_RESET_START` (= `prev_gain_Q16`) and survive `reset()`.
    pub prev_gain_q16: i32,
    pub exc_q14: [i32; MFL],
    pub s_lpc_q14_buf: [i32; MLPC],
    /// Buffer for output signal.
    pub out_buf: [i16; MFL + 2 * MSFL],
    /// Previous lag.
    pub lag_prev: i32,
    /// Previous gain index.
    pub last_gain_index: i8,
    /// Sampling frequency in kHz.
    pub fs_khz: i32,
    /// API sample frequency (Hz).
    pub fs_api_hz: i32,
    /// Number of 5 ms subframes in a frame.
    pub nb_subfr: i32,
    /// Frame length (samples).
    pub frame_length: i32,
    /// Subframe length (samples).
    pub subfr_length: i32,
    /// Length of LTP memory.
    pub ltp_mem_length: i32,
    /// LPC order.
    pub lpc_order: i32,
    /// Used to interpolate LSFs.
    pub prev_nlsf_q15: [i16; MLPC],
    /// Flag for deactivating NLSF interpolation.
    pub first_frame_after_reset: i32,
    /// iCDF table for low bits of pitch lag index (C pointer; `NULL` → empty).
    pub pitch_lag_low_bits_icdf: &'static [u8],
    /// iCDF table for pitch contour index (C pointer; `NULL` → empty).
    pub pitch_contour_icdf: &'static [u8],
    /// For buffering payload in case of more frames per packet.
    pub n_frames_decoded: i32,
    pub n_frames_per_packet: i32,
    /// Specifically for entropy coding.
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub vad_flags: [i32; MFPP],
    pub lbrr_flag: i32,
    pub lbrr_flags: [i32; MFPP],
    pub resampler_state: SilkResamplerState,
    /// NLSF codebook (C pointer; `NULL` → NB/MB codebook, never read before being set).
    pub ps_nlsf_cb: &'static SilkNlsfCbStruct,
    /// Quantization indices.
    pub indices: SideInfoIndices,
    /// CNG state.
    pub s_cng: SilkCngStruct,
    /// Stuff used for PLC.
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    // `arch`: dropped (no run-time dispatch).
    pub s_plc: SilkPlcStruct,
}

impl SilkDecoderState {
    /// All-zero state (C `memset(0)` of the whole struct).
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_gain_q16: 0,
            exc_q14: [0; MFL],
            s_lpc_q14_buf: [0; MLPC],
            out_buf: [0; MFL + 2 * MSFL],
            lag_prev: 0,
            last_gain_index: 0,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MLPC],
            first_frame_after_reset: 0,
            pitch_lag_low_bits_icdf: &[],
            pitch_contour_icdf: &[],
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [0; MFPP],
            lbrr_flag: 0,
            lbrr_flags: [0; MFPP],
            resampler_state: SilkResamplerState::default(),
            ps_nlsf_cb: &SILK_NLSF_CB_NB_MB,
            indices: SideInfoIndices::new(),
            s_cng: SilkCngStruct::new(),
            loss_cnt: 0,
            prev_signal_type: 0,
            s_plc: SilkPlcStruct {
                pitch_l_q8: 0,
                ltp_coef_q14: [0; LTP_ORDER as usize],
                prev_lpc_q12: [0; MLPC],
                last_frame_lost: 0,
                rand_seed: 0,
                rand_scale_q14: 0,
                conc_energy: 0,
                conc_energy_shift: 0,
                prev_ltp_scale_q14: 0,
                prev_gain_q16: [0; 2],
                fs_khz: 0,
                nb_subfr: 0,
                subfr_length: 0,
                enable_deep_plc: 0,
            },
        }
    }

    /// C `silk_memset(&psDec->SILK_DECODER_STATE_RESET_START, 0, ...)` as done by
    /// `silk_reset_decoder`: zeroes every field from `prev_gain_Q16` to the end of the struct
    /// (which, without the DNN fields, is the whole struct). The follow-up initialization
    /// (`first_frame_after_reset`, `prev_gain_Q16`, CNG/PLC reset) belongs to
    /// `silk_reset_decoder`.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SilkDecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// `silk_decoder_control`: decoder control.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkDecoderControl {
    /// Prediction and coding parameters.
    pub pitch_l: [i32; MNSF],
    pub gains_q16: [i32; MNSF],
    /// Holds interpolated and final coefficients.
    pub pred_coef_q12: [[i16; MLPC]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER as usize * MNSF],
    pub ltp_scale_q14: i32,
}

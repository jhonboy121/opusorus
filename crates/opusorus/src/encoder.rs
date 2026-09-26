//! The Opus encoder: port of `src/opus_encoder.c`.
//!
//! [`Encoder`] is the C `OpusEncoder`. It combines a SILK encoder (speech, up to wideband) and a
//! CELT encoder (music / full band) and decides per frame which one (or both, "hybrid") to use,
//! the audio bandwidth, the channel count and the bit allocation.
//!
//! * `opus_encoder_create` / `opus_encoder_init` → [`Encoder::new`] / [`Encoder::init`].
//! * `opus_encode` / `opus_encode24` / `opus_encode_float` → [`Encoder::encode`] /
//!   [`Encoder::encode24`] / [`Encoder::encode_float`].
//! * Every `opus_encoder_ctl` request is a typed method (`set_bitrate`, `bitrate`, ...,
//!   [`Encoder::reset`] for `OPUS_RESET_STATE`), plus the numeric escape hatches
//!   [`Encoder::ctl_set`] / [`Encoder::ctl_get`] for the C ABI layer.
//! * `opus_encoder_get_size` → [`encoder_get_size`] (Rust footprint, not the C `sizeof`).
//! * `opus_encoder_destroy` → `Drop`.
//!
//! Both builds are ported: the float build (`opus_res` = `f32`) and the fixed-point build
//! (features `fixed-point` / `fixed-res24`: `opus_res` = `i16` / `i32`, the `FIXED_POINT`
//! branches of the C file: integer high-pass / DC filters, gain and stereo fades, stereo width,
//! frame energy, surround masking, `opus_encode` / `opus_encode24` passing their input through
//! without conversion when it already is `opus_res`). Each is bit-exact with the matching C
//! build.
//!
//! Deviations from C (none observable in the output):
//! * The energy mask (`OPUS_SET_ENERGY_MASK`, used by the surround multistream encoder) is
//!   copied instead of stored as a pointer; the multistream encoder sets it before every frame,
//!   as C does.
//! * All C VLAs are scratch buffers owned by the encoder; encoding does not allocate once the
//!   buffers have grown to the size a call needs (the input conversion buffer and the multi-frame
//!   packet buffer grow on first use).
//! * Rust-only argument checks: a PCM slice shorter than `frame_size * channels` returns
//!   [`Error::BadArg`] (C reads out of bounds).
//!
//! DRED (feature `dred`, C `ENABLE_DRED`): with `Encoder::set_dred_duration` > 0 the encoder
//! reserves part of the bitrate for Deep REDundancy and appends a DRED extension (RDOVAE latents
//! of up to `duration * 10 ms` of past audio) to the first non-DTX frame of each packet. The
//! RDOVAE encoder model is not compiled in (upstream `USE_WEIGHTS_FILE`, PLAN D-015) unless the
//! `dnn-weights-embedded` feature embeds a blob (bound by [`Encoder::new`]): load it
//! with `Encoder::set_dnn_blob`. As in upstream builds without loaded weights, the DRED
//! bitrate is still reserved when no model is loaded, but no DRED data is produced. The loaded
//! model survives [`Encoder::reset`] and [`Encoder::init`].

#![allow(
    clippy::too_many_arguments,
    reason = "internal functions mirror the C signatures"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "`!(x < y)` is the C idiom that also catches NaN"
)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::analysis::{AnalysisInfo, DownmixFunc, downmix_int, downmix_int24, is_digital_silence};
#[cfg(not(feature = "disable-float-api"))]
use crate::analysis::{
    TonalityAnalysisState, downmix_float, run_analysis, tonality_analysis_init,
    tonality_analysis_reset, tonality_get_info,
};
#[cfg(all(feature = "fixed-point", not(feature = "disable-float-api")))]
use crate::celt::arch::float2res;
#[cfg(not(feature = "fixed-res24"))]
use crate::celt::arch::int24tores;
use crate::celt::arch::{
    CeltCoef, CeltGlog, MAX_ENCODING_DEPTH, OpusRes, OpusVal16, OpusVal32, Q15ONE, abs16, half32,
    imax, imin, int16tores, max16, max32, maxg, min16, min32, ming,
};
#[cfg(feature = "fixed-point")]
use crate::celt::arch::{
    DB_SHIFT, EPSILON, RES_SHIFT, coef2val16, extract16, gconst, mac16_16, mult16_16,
    mult16_16_q15, mult16_32_q15, mult16_res_q15, pshr32, qconst16, qconst32, res2int16, res2val16,
    saturate, shl16, shl32, shr32,
};
#[cfg(not(feature = "fixed-point"))]
use crate::celt::arch::{VERY_SMALL, celt_isnan};
use crate::celt::celt::{SilkInfo, bitrate_to_bits, bits_to_bitrate};
use crate::celt::celt_encoder::CeltEncoder;
use crate::celt::entenc::EcEnc;
use crate::celt::mathops::{celt_exp2, celt_sqrt, frac_div32};
#[cfg(feature = "fixed-point")]
use crate::celt::mathops::{celt_ilog2, celt_maxabs_res};
#[cfg(feature = "qext")]
use crate::celt::modes::QEXT_PACKET_SIZE_CAP;
#[cfg(not(feature = "fixed-point"))]
use crate::celt::pitch::celt_inner_prod;
use crate::celt::static_modes::CeltMode;
use crate::constants::raw::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_CELT, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND, OPUS_BITRATE_MAX, OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_40_MS,
    OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_ARG, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
};
use crate::packet::{MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, len_i32};
use crate::repacketizer::Repacketizer;
use crate::silk::define::{
    DTX_ACTIVITY_THRESHOLD, MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX,
    TYPE_NO_VOICE_ACTIVITY, VAD_NO_DECISION,
};
use crate::silk::encoder::SilkEncoder;
use crate::silk::macros::{
    silk_div32_16, silk_fix_const, silk_lshift, silk_min, silk_mul, silk_rshift, silk_smlawb,
    silk_smulbb, silk_smulwb, silk_smulww,
};
#[cfg(feature = "fixed-point")]
use crate::silk::macros::{silk_rshift_round, silk_sat16};
#[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
use crate::silk::sigproc::{silk_biquad_alt_stride1, silk_biquad_alt_stride2};
use crate::silk::sigproc::{silk_lin2log, silk_log2lin};
use crate::silk::structs::SilkEncControlStruct;
use crate::silk::tuning_parameters::{VARIABLE_HP_MIN_CUTOFF_HZ, VARIABLE_HP_SMTH_COEF2};
use crate::{Application, Bandwidth, Bitrate, Error, FrameSize, Result, Signal};

#[cfg(feature = "dred")]
use crate::celt::entcode::ec_ilog;
#[cfg(feature = "dred")]
use crate::dnn::dred_coding::{
    DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID, DRED_MAX_DATA_SIZE,
    DRED_MAX_FRAMES, DRED_MIN_BYTES, DRED_NUM_REDUNDANCY_FRAMES, compute_quantizer,
};
#[cfg(feature = "dred")]
use crate::dnn::dred_encoder::{
    DRED_ACTIVITY_MEM_SIZE, DredEnc, dred_compute_latents, dred_encode_silk_frame,
};
#[cfg(feature = "dred")]
use crate::extensions::Extension;

/// CTL request numbers handled by the encoders (`opus_defines.h`, `opus_private.h`, `celt.h`,
/// `opus_multistream.h`, `opus_projection.h`), for use with [`Encoder::ctl_set`] /
/// [`Encoder::ctl_get`] and the multistream / projection equivalents.
pub mod request {
    #![allow(missing_docs, reason = "values mirror the C headers one-to-one")]
    pub const OPUS_SET_APPLICATION_REQUEST: i32 = 4000;
    pub const OPUS_GET_APPLICATION_REQUEST: i32 = 4001;
    pub const OPUS_SET_BITRATE_REQUEST: i32 = 4002;
    pub const OPUS_GET_BITRATE_REQUEST: i32 = 4003;
    pub const OPUS_SET_MAX_BANDWIDTH_REQUEST: i32 = 4004;
    pub const OPUS_GET_MAX_BANDWIDTH_REQUEST: i32 = 4005;
    pub const OPUS_SET_VBR_REQUEST: i32 = 4006;
    pub const OPUS_GET_VBR_REQUEST: i32 = 4007;
    pub const OPUS_SET_BANDWIDTH_REQUEST: i32 = 4008;
    pub const OPUS_GET_BANDWIDTH_REQUEST: i32 = 4009;
    pub const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
    pub const OPUS_GET_COMPLEXITY_REQUEST: i32 = 4011;
    pub const OPUS_SET_INBAND_FEC_REQUEST: i32 = 4012;
    pub const OPUS_GET_INBAND_FEC_REQUEST: i32 = 4013;
    pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: i32 = 4014;
    pub const OPUS_GET_PACKET_LOSS_PERC_REQUEST: i32 = 4015;
    pub const OPUS_SET_DTX_REQUEST: i32 = 4016;
    pub const OPUS_GET_DTX_REQUEST: i32 = 4017;
    pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: i32 = 4020;
    pub const OPUS_GET_VBR_CONSTRAINT_REQUEST: i32 = 4021;
    pub const OPUS_SET_FORCE_CHANNELS_REQUEST: i32 = 4022;
    pub const OPUS_GET_FORCE_CHANNELS_REQUEST: i32 = 4023;
    pub const OPUS_SET_SIGNAL_REQUEST: i32 = 4024;
    pub const OPUS_GET_SIGNAL_REQUEST: i32 = 4025;
    pub const OPUS_GET_LOOKAHEAD_REQUEST: i32 = 4027;
    pub const OPUS_RESET_STATE: i32 = 4028;
    pub const OPUS_GET_SAMPLE_RATE_REQUEST: i32 = 4029;
    pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
    pub const OPUS_SET_LSB_DEPTH_REQUEST: i32 = 4036;
    pub const OPUS_GET_LSB_DEPTH_REQUEST: i32 = 4037;
    pub const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: i32 = 4040;
    pub const OPUS_GET_EXPERT_FRAME_DURATION_REQUEST: i32 = 4041;
    pub const OPUS_SET_PREDICTION_DISABLED_REQUEST: i32 = 4042;
    pub const OPUS_GET_PREDICTION_DISABLED_REQUEST: i32 = 4043;
    pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4046;
    pub const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4047;
    pub const OPUS_GET_IN_DTX_REQUEST: i32 = 4049;
    pub const OPUS_SET_DRED_DURATION_REQUEST: i32 = 4050;
    pub const OPUS_GET_DRED_DURATION_REQUEST: i32 = 4051;
    pub const OPUS_SET_DNN_BLOB_REQUEST: i32 = 4052;
    pub const OPUS_SET_QEXT_REQUEST: i32 = 4056;
    pub const OPUS_GET_QEXT_REQUEST: i32 = 4057;
    pub const OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST: i32 = 5120;
    pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST: i32 = 6001;
    pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST: i32 = 6003;
    pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST: i32 = 6005;
    pub const CELT_GET_MODE_REQUEST: i32 = 10015;
    pub const OPUS_SET_LFE_REQUEST: i32 = 10024;
    pub const OPUS_SET_ENERGY_MASK_REQUEST: i32 = 10026;
    pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
    pub const OPUS_SET_VOICE_RATIO_REQUEST: i32 = 11018;
    pub const OPUS_GET_VOICE_RATIO_REQUEST: i32 = 11019;
}
use request::*;

/// `MAX_ENCODER_BUFFER`: delay buffer length per channel.
#[cfg(feature = "qext")]
const MAX_ENCODER_BUFFER: usize = 960;
/// `MAX_ENCODER_BUFFER`: delay buffer length per channel.
#[cfg(not(feature = "qext"))]
const MAX_ENCODER_BUFFER: usize = 480;

/// `PSEUDO_SNR_THRESHOLD`: 10^(25/10).
const PSEUDO_SNR_THRESHOLD: f32 = 316.23;

/// The `run_analysis` call of `opus_encode_native` with the input-type dependent arguments
/// (`analysis_pcm`, `analysis_size`, `c1`, `c2`, `analysis_channels`, `downmix`) bound:
/// `(analysis, celt_mode, frame_size, Fs, lsb_depth, analysis_info)`.
#[cfg(not(feature = "disable-float-api"))]
type AnalyzeFn<'a> =
    &'a mut dyn FnMut(&mut TonalityAnalysisState, &CeltMode, i32, i32, i32, &mut AnalysisInfo);

/// Largest packet a single call may produce (`packet_size_cap*6` in `opus_encode_native`).
#[cfg(feature = "qext")]
const MAX_PACKET_BYTES: usize = 6 * QEXT_PACKET_SIZE_CAP as usize;
/// Largest packet a single call may produce (`packet_size_cap*6` in `opus_encode_native`).
#[cfg(not(feature = "qext"))]
const MAX_PACKET_BYTES: usize = 6 * 1276;

/// `StereoWidthState` (src/opus_encoder.c).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[doc(hidden)]
pub struct StereoWidthState {
    pub xx: OpusVal32,
    pub xy: OpusVal32,
    pub yy: OpusVal32,
    pub smoothed_width: OpusVal16,
    pub max_follower: OpusVal16,
}

// Transition tables for the voice and music. First column is the middle (memoriless)
// threshold. The second column is the hysteresis (difference with the middle).
static MONO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB<->MB
    9000, 700, // MB<->WB
    13500, 1000, // WB<->SWB
    14000, 2000, // SWB<->FB
];
static MONO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB<->MB
    9000, 700, // MB<->WB
    11000, 1000, // WB<->SWB
    12000, 2000, // SWB<->FB
];
static STEREO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB<->MB
    9000, 700, // MB<->WB
    13500, 1000, // WB<->SWB
    14000, 2000, // SWB<->FB
];
static STEREO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB<->MB
    9000, 700, // MB<->WB
    11000, 1000, // WB<->SWB
    12000, 2000, // SWB<->FB
];
/// Threshold bit-rates for switching between mono and stereo.
const STEREO_VOICE_THRESHOLD: i32 = 19000;
const STEREO_MUSIC_THRESHOLD: i32 = 17000;

/// Threshold bit-rate for switching between SILK/hybrid and CELT-only
/// (`[mono, stereo][voice, music]`).
static MODE_THRESHOLDS: [[i32; 2]; 2] = [[64000, 10000], [44000, 10000]];

static FEC_THRESHOLDS: [i32; 10] = [
    12000, 1000, // NB
    14000, 1000, // MB
    16000, 1000, // WB
    20000, 1000, // SWB
    22000, 1000, // FB
];

/// Scratch buffers standing in for the C VLAs (`ALLOC`).
#[derive(Debug, Clone, Default)]
struct Scratch {
    /// `in` of `opus_encode` / `opus_encode24` (int to `opus_res` conversion). Grows on demand.
    input: Vec<OpusRes>,
    /// `pcm_buf` of `opus_encode_frame_native`: `(total_buffer+frame_size)*channels`.
    pcm_buf: Vec<OpusRes>,
    /// `tmp_prefill`: `channels*Fs/400`.
    tmp_prefill: Vec<OpusRes>,
    /// `tmp_data` of the multi-frame path. Grows on demand.
    tmp_data: Vec<u8>,
    /// Copy of the packet for in-place padding (`opus_packet_pad`). Grows on demand.
    pad: Vec<u8>,
}

/// An Opus encoder (port of `struct OpusEncoder`, `src/opus_encoder.c`).
///
/// Encodes 16-bit, 24-bit or float PCM into Opus packets. One encoder handles one stream of
/// one or two channels; see [`crate::ms_encoder::MsEncoder`] for more channels.
///
/// ```
/// use opusorus::encoder::Encoder;
/// use opusorus::{Application, Bitrate};
///
/// let mut enc = Encoder::new(48000, 2, Application::Audio)?;
/// enc.set_bitrate(Bitrate::Bits(64000))?;
/// enc.set_complexity(10)?;
/// // 20 ms of silence (960 samples per channel, interleaved).
/// let pcm = vec![0i16; 960 * 2];
/// let mut packet = [0u8; 1500];
/// let len = enc.encode(&pcm, 960, &mut packet)?;
/// assert!(len >= 1);
/// # Ok::<(), opusorus::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Encoder {
    silk_mode: SilkEncControlStruct,
    /// `dred_encoder` (boxed: RDOVAE/pitch models and ~70 KB of buffers).
    #[cfg(feature = "dred")]
    dred_encoder: Box<DredEnc>,
    application: i32,
    channels: i32,
    delay_compensation: i32,
    force_channels: i32,
    signal_type: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    fs: i32,
    use_vbr: i32,
    vbr_constraint: i32,
    variable_duration: i32,
    bitrate_bps: i32,
    user_bitrate_bps: i32,
    lsb_depth: i32,
    encoder_buffer: i32,
    lfe: i32,
    /// General DTX for both SILK and CELT.
    use_dtx: i32,
    fec_config: i32,
    /// `analysis` (not with `DISABLE_FLOAT_API`, like the tonality analysis itself).
    #[cfg(not(feature = "disable-float-api"))]
    analysis: Box<TonalityAnalysisState>,
    #[cfg(feature = "qext")]
    enable_qext: i32,

    // ---- OPUS_ENCODER_RESET_START: everything below is cleared by OPUS_RESET_STATE ----
    stream_channels: i32,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: OpusVal16,
    hp_mem: [OpusVal32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: i32,
    /// Bandwidth determined automatically from the rate (before any other adjustment).
    auto_bandwidth: i32,
    silk_bw_switch: i32,
    first: i32,
    /// `energy_masking != NULL`.
    has_energy_mask: bool,
    /// Copy of the `energy_masking` values (`21*channels`).
    energy_masking: [CeltGlog; 42],
    width_mem: StereoWidthState,
    #[cfg(not(feature = "disable-float-api"))]
    detected_bandwidth: i32,
    nb_no_activity_ms_q1: i32,
    peak_signal_energy: OpusVal32,
    /// `dred_duration` (in the C reset region: OPUS_RESET_STATE clears it).
    #[cfg(feature = "dred")]
    dred_duration: i32,
    /// `dred_q0`.
    #[cfg(feature = "dred")]
    dred_q0: i32,
    /// `dred_dQ`.
    #[cfg(feature = "dred")]
    dred_dq: i32,
    /// `dred_qmax`.
    #[cfg(feature = "dred")]
    dred_qmax: i32,
    /// `dred_target_chunks`.
    #[cfg(feature = "dred")]
    dred_target_chunks: i32,
    /// `activity_mem[DRED_MAX_FRAMES*4]`: voice activity at 2.5 ms resolution, newest first.
    #[cfg(feature = "dred")]
    activity_mem: [u8; DRED_ACTIVITY_MEM_SIZE],
    /// The C local `dred_bitrate_bps` of `opus_encode_native`, passed down to the frame
    /// encoder (`opus_encode_frame_native`'s argument); not part of the C struct.
    #[cfg(feature = "dred")]
    dred_bitrate_bps: i32,
    /// Current frame is not the final in a packet.
    nonfinal_frame: i32,
    range_final: u32,
    /// `delay_buffer[MAX_ENCODER_BUFFER*2]`.
    delay_buffer: Vec<OpusRes>,

    /// The SILK encoder (absent for `OPUS_APPLICATION_RESTRICTED_CELT`).
    silk_enc: Option<Box<SilkEncoder>>,
    /// The CELT encoder (absent for `OPUS_APPLICATION_RESTRICTED_SILK`).
    celt_enc: Option<Box<CeltEncoder>>,
    scratch: Scratch,
}

/// Port of src/opus_encoder.c:opus_encoder_get_size: the memory footprint of an encoder with
/// `channels` channels (1 or 2), or 0 for other channel counts.
///
/// The value is the Rust footprint (the encoder struct plus its heap state), not the C `sizeof`.
#[must_use]
pub fn encoder_get_size(channels: i32) -> usize {
    if channels != 1 && channels != 2 {
        return 0;
    }
    let ch = channels as usize;
    let celt = crate::celt::celt_encoder::celt_encoder_get_size(channels).max(0) as usize;
    // The DRED encoder state (without the heap-allocated model weights of a loaded blob).
    #[cfg(feature = "dred")]
    let celt = celt + size_of::<DredEnc>();
    #[cfg(not(feature = "disable-float-api"))]
    let celt = celt + size_of::<TonalityAnalysisState>();
    size_of::<Encoder>()
        + size_of::<SilkEncoder>()
        + celt
        + MAX_ENCODER_BUFFER * 2 * size_of::<OpusRes>()
        + ch * (MAX_ENCODER_BUFFER / 10 * 76) * size_of::<OpusRes>()
}

/// Maps a C return value (bytes or a negative `OPUS_*` code) to a [`Result`].
#[inline]
pub(crate) const fn code_to_result(ret: i32) -> Result<usize> {
    match Error::from_code(ret) {
        Some(e) => Err(e),
        None => Ok(ret as usize),
    }
}

/// C ignores the return value of most internal CTL calls; the only error they can produce is
/// `OPUS_BAD_ARG` (e.g. `OPUS_SET_BITRATE` of 500 b/s or less), which leaves the setting
/// unchanged. Other errors cannot occur and are propagated.
#[inline]
pub(crate) const fn c_ignored(r: Result<()>) -> Result<()> {
    match r {
        Ok(()) | Err(Error::BadArg) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Converts a validated stored value back to its enum.
#[inline]
fn validated<T>(r: Result<T>) -> T {
    #[expect(
        clippy::expect_used,
        reason = "the encoder only stores values validated by its setters"
    )]
    r.expect("encoder state holds a validated value")
}

/// The argument checks of `opus_encoder_init` (C `opus_encoder_init(NULL, ...)` is used by
/// the multistream encoder to validate the rate and application).
pub(crate) const fn check_init_args(fs: i32, channels: i32, application: i32) -> Result<()> {
    if !valid_fs(fs) || (channels != 1 && channels != 2) || !valid_application(application) {
        return Err(Error::BadArg);
    }
    Ok(())
}

/// Checks the application argument of `opus_encoder_init`.
const fn valid_application(application: i32) -> bool {
    application == OPUS_APPLICATION_VOIP
        || application == OPUS_APPLICATION_AUDIO
        || application == OPUS_APPLICATION_RESTRICTED_LOWDELAY
        || application == OPUS_APPLICATION_RESTRICTED_SILK
        || application == OPUS_APPLICATION_RESTRICTED_CELT
}

/// Checks the sampling rate argument of `opus_encoder_init`.
const fn valid_fs(fs: i32) -> bool {
    #[cfg(feature = "qext")]
    if fs == 96000 {
        return true;
    }
    fs == 48000 || fs == 24000 || fs == 16000 || fs == 12000 || fs == 8000
}

/// Port of src/opus_encoder.c:gen_toc: the TOC byte for a frame.
#[must_use]
#[doc(hidden)]
pub const fn gen_toc(mode: i32, framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    let mut framerate = framerate;
    let mut period = 0;
    while framerate < 400 {
        framerate <<= 1;
        period += 1;
    }
    // C stores into an unsigned char at each step; truncating once at the end is equivalent.
    let mut toc: i32;
    if mode == MODE_SILK_ONLY {
        toc = (bandwidth - OPUS_BANDWIDTH_NARROWBAND) << 5;
        toc |= (period - 2) << 3;
    } else if mode == MODE_CELT_ONLY {
        let mut tmp = bandwidth - OPUS_BANDWIDTH_MEDIUMBAND;
        if tmp < 0 {
            tmp = 0;
        }
        toc = 0x80;
        toc |= tmp << 5;
        toc |= period << 3;
    } else {
        // Hybrid
        toc = 0x60;
        toc |= (bandwidth - OPUS_BANDWIDTH_SUPERWIDEBAND) << 4;
        toc |= (period - 2) << 3;
    }
    toc |= ((channels == 2) as i32) << 2;
    toc as u8
}

/// Port of src/opus_encoder.c:silk_biquad_res (fixed-point build): second order ARMA filter
/// (direct form II transposed, the `silk_biquad_alt` arithmetic on `RES2INT16` samples) on
/// `len` samples read/written with `stride`. Used by 24-bit resolution builds.
#[cfg(feature = "fixed-point")]
#[doc(hidden)]
pub fn silk_biquad_res(
    input: &[OpusRes],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [OpusVal32],
    out: &mut [OpusRes],
    len: i32,
    stride: i32,
) {
    // DIRECT FORM II TRANSPOSED (uses 2 element state vector)

    // Negate A_Q28 values and split in two parts
    let a0_l_q28 = (-a_q28[0]) & 0x0000_3FFF; // lower part
    let a0_u_q28 = silk_rshift(-a_q28[0], 14); // upper part
    let a1_l_q28 = (-a_q28[1]) & 0x0000_3FFF; // lower part
    let a1_u_q28 = silk_rshift(-a_q28[1], 14); // upper part

    let stride = stride as usize;
    for k in 0..len as usize {
        // S[ 0 ], S[ 1 ]: Q12
        let inval = i32::from(res2int16(input[k * stride]));
        let out32_q14 = silk_lshift(silk_smlawb(s[0], b_q28[0], inval), 2);

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        // Scale back to Q0 and saturate
        out[k * stride] = int16tores(silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16);
    }
}

/// Port of src/opus_encoder.c:silk_biquad_res (float build): second order ARMA filter
/// (direct form II transposed) on `len` samples read/written with `stride`.
#[cfg(not(feature = "fixed-point"))]
#[doc(hidden)]
pub fn silk_biquad_res(
    input: &[OpusRes],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [OpusVal32],
    out: &mut [OpusRes],
    len: i32,
    stride: i32,
) {
    let a = [
        a_q28[0] as f32 * (1.0f32 / (1i32 << 28) as f32),
        a_q28[1] as f32 * (1.0f32 / (1i32 << 28) as f32),
    ];
    let b = [
        b_q28[0] as f32 * (1.0f32 / (1i32 << 28) as f32),
        b_q28[1] as f32 * (1.0f32 / (1i32 << 28) as f32),
        b_q28[2] as f32 * (1.0f32 / (1i32 << 28) as f32),
    ];
    let stride = stride as usize;
    for k in 0..len as usize {
        // S[ 0 ], S[ 1 ]: Q12
        let inval: OpusVal32 = input[k * stride];
        let vout: OpusVal32 = s[0] + b[0] * inval;

        s[0] = s[1] - vout * a[0] + b[1] * inval;

        s[1] = -vout * a[1] + b[2] * inval + VERY_SMALL;

        // Scale back to Q0 and saturate
        out[k * stride] = vout;
    }
}

/// Port of src/opus_encoder.c:hp_cutoff: variable-cutoff high-pass filter (VoIP input).
#[doc(hidden)]
pub fn hp_cutoff(
    input: &[OpusRes],
    cutoff_hz: i32,
    out: &mut [OpusRes],
    hp_mem: &mut [OpusVal32; 4],
    len: i32,
    channels: i32,
    fs: i32,
) {
    let fc_q19 = silk_div32_16(
        silk_smulbb(silk_fix_const(1.5 * 3.14159 / 1000.0, 19), cutoff_hz),
        fs / 1000,
    );
    // C: silk_assert( Fc_Q19 > 0 && Fc_Q19 < 32768 ) (a no-op in libopus builds; it holds
    // for the cutoffs the encoder produces).

    let r_q28 = silk_fix_const(1.0, 28) - silk_mul(silk_fix_const(0.92, 9), fc_q19);

    // b = r * [ 1; -2; 1 ];
    // a = [ 1; -2 * r * ( 1 - 0.5 * Fc^2 ); r^2 ];
    let b_q28 = [r_q28, silk_lshift(-r_q28, 1), r_q28];

    // -r * ( 2 - Fc * Fc );
    let r_q22 = silk_rshift(r_q28, 6);
    let a_q28 = [
        silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - silk_fix_const(2.0, 22)),
        silk_smulww(r_q22, r_q22),
    ];

    // 16-bit fixed-point builds use the SILK biquad directly on the `opus_int16` samples.
    #[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
    {
        if channels == 1 {
            silk_biquad_alt_stride1(input, &b_q28, &a_q28, hp_mem, out, len as usize);
        } else {
            silk_biquad_alt_stride2(input, &b_q28, &a_q28, hp_mem, out, len as usize);
        }
    }
    #[cfg(any(not(feature = "fixed-point"), feature = "fixed-res24"))]
    {
        let (m01, m23) = hp_mem.split_at_mut(2);
        silk_biquad_res(input, &b_q28, &a_q28, m01, out, len, channels);
        if channels == 2 {
            silk_biquad_res(
                &input[1..],
                &b_q28,
                &a_q28,
                m23,
                &mut out[1..],
                len,
                channels,
            );
        }
    }
}

/// Port of src/opus_encoder.c:dc_reject (fixed-point build): one-pole DC rejection filter
/// (`hp_mem` in Q14 of the 16-bit scale).
#[cfg(feature = "fixed-point")]
#[doc(hidden)]
pub fn dc_reject(
    input: &[OpusRes],
    cutoff_hz: i32,
    out: &mut [OpusRes],
    hp_mem: &mut [OpusVal32; 4],
    len: i32,
    channels: i32,
    fs: i32,
) {
    // Approximates -round(log2(6.3*cutoff_Hz/Fs))
    let shift = celt_ilog2(fs / (cutoff_hz * 4));
    let ch = channels as usize;
    for c in 0..ch {
        for i in 0..len as usize {
            // Saturate at +6 dBFS to avoid any wrap-around.
            let mut x: OpusVal32 = saturate(input[ch * i + c], (1 << 16 << RES_SHIFT) - 1);
            x = shl32(x, 14 - RES_SHIFT);
            let y: OpusVal32 = x - hp_mem[2 * c];
            hp_mem[2 * c] += pshr32(x - hp_mem[2 * c], shift);
            // Don't saturate if we have the headroom to avoid it (24-bit resolution).
            #[cfg(feature = "fixed-res24")]
            {
                out[ch * i + c] = pshr32(y, 14 - RES_SHIFT);
            }
            #[cfg(not(feature = "fixed-res24"))]
            {
                out[ch * i + c] = saturate(pshr32(y, 14 - RES_SHIFT), 32767) as OpusRes;
            }
        }
    }
}

/// Port of src/opus_encoder.c:dc_reject (float build): one-pole DC rejection filter.
#[cfg(not(feature = "fixed-point"))]
#[doc(hidden)]
pub fn dc_reject(
    input: &[OpusVal16],
    cutoff_hz: i32,
    out: &mut [OpusVal16],
    hp_mem: &mut [OpusVal32; 4],
    len: i32,
    channels: i32,
    fs: i32,
) {
    let coef: f32 = 6.3f32 * cutoff_hz as f32 / fs as f32;
    let coef2: f32 = 1.0 - coef;
    let len = len as usize;
    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for i in 0..len {
            let x0: OpusVal32 = input[2 * i];
            let x1: OpusVal32 = input[2 * i + 1];
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = coef * x0 + VERY_SMALL + coef2 * m0;
            m2 = coef * x1 + VERY_SMALL + coef2 * m2;
            out[2 * i] = out0;
            out[2 * i + 1] = out1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0 = hp_mem[0];
        for i in 0..len {
            let x: OpusVal32 = input[i];
            let y = x - m0;
            m0 = coef * x + VERY_SMALL + coef2 * m0;
            out[i] = y;
        }
        hp_mem[0] = m0;
    }
}

/// The squared window gain `w` and the crossfaded gain between `g1` and `g2` of
/// `stereo_fade` / `gain_fade` (fixed-point build): `w = MULT16_16_Q15(w, w)`,
/// `g = SHR32(MAC16_16(MULT16_16(w,g2), Q15ONE-w, g1), 15)`.
#[cfg(feature = "fixed-point")]
#[inline]
fn fade_gain(window: CeltCoef, g1: OpusVal16, g2: OpusVal16) -> OpusVal16 {
    let mut w: OpusVal16 = coef2val16(window);
    w = mult16_16_q15(w, w) as OpusVal16;
    shr32(
        mac16_16(mult16_16(w, g2), i32::from(Q15ONE) - i32::from(w), g1),
        15,
    ) as OpusVal16
}

/// Port of src/opus_encoder.c:stereo_fade (fixed-point build), in place (the encoder always
/// calls it with `in == out`): crossfades the stereo width from `g1` to `g2` (Q15).
#[cfg(feature = "fixed-point")]
#[doc(hidden)]
#[allow(
    clippy::useless_conversion,
    reason = "`opus_res` is already 32-bit with fixed-res24"
)]
pub fn stereo_fade(
    buf: &mut [OpusRes],
    g1: OpusVal16,
    g2: OpusVal16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[CeltCoef],
    fs: i32,
) {
    let inc = imax(1, 48000 / fs);
    let overlap = (overlap48 / inc) as usize;
    let inc = inc as usize;
    let ch = channels as usize;
    let g1 = (i32::from(Q15ONE) - i32::from(g1)) as OpusVal16;
    let g2 = (i32::from(Q15ONE) - i32::from(g2)) as OpusVal16;
    // `out[k] = out[k] -/+ diff` with the implicit C narrowing to `opus_res`.
    let apply = |buf: &mut [OpusRes], i: usize, g: OpusVal16| {
        // C: `(opus_val32)` casts and implicit conversions (no `EXTEND32`).
        let mut diff: OpusVal32 = half32(i32::from(buf[i * ch]) - i32::from(buf[i * ch + 1]));
        // MULT16_RES_Q15 takes the difference as an `opus_res` (MULT16_16_Q15 truncates it to
        // 16 bits in 16-bit builds; it fits).
        diff = i32::from(mult16_res_q15(g, diff as OpusRes));
        buf[i * ch] = (i32::from(buf[i * ch]) - diff) as OpusRes;
        buf[i * ch + 1] = (i32::from(buf[i * ch + 1]) + diff) as OpusRes;
    };
    let mut i = 0usize;
    while i < overlap {
        let g = fade_gain(window[i * inc], g1, g2);
        apply(buf, i, g);
        i += 1;
    }
    while i < frame_size as usize {
        apply(buf, i, g2);
        i += 1;
    }
}

/// Port of src/opus_encoder.c:gain_fade (fixed-point build), in place (the encoder always calls
/// it with `in == out`): crossfades the gain from `g1` to `g2` (Q15) over the CELT overlap.
#[cfg(feature = "fixed-point")]
#[doc(hidden)]
pub fn gain_fade(
    buf: &mut [OpusRes],
    g1: OpusVal16,
    g2: OpusVal16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[CeltCoef],
    fs: i32,
) {
    let inc = imax(1, 48000 / fs);
    let overlap = (overlap48 / inc) as usize;
    let inc = inc as usize;
    if channels == 1 {
        for i in 0..overlap {
            let g = fade_gain(window[i * inc], g1, g2);
            buf[i] = mult16_res_q15(g, buf[i]);
        }
    } else {
        for i in 0..overlap {
            let g = fade_gain(window[i * inc], g1, g2);
            buf[i * 2] = mult16_res_q15(g, buf[i * 2]);
            buf[i * 2 + 1] = mult16_res_q15(g, buf[i * 2 + 1]);
        }
    }
    let ch = channels as usize;
    for c in 0..ch {
        for i in overlap..frame_size as usize {
            buf[i * ch + c] = mult16_res_q15(g2, buf[i * ch + c]);
        }
    }
}

/// Port of src/opus_encoder.c:stereo_fade, in place (the encoder always calls it with
/// `in == out`): crossfades the stereo width from `g1` to `g2`.
#[cfg(not(feature = "fixed-point"))]
#[doc(hidden)]
pub fn stereo_fade(
    buf: &mut [OpusRes],
    g1: OpusVal16,
    g2: OpusVal16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[CeltCoef],
    fs: i32,
) {
    let inc = imax(1, 48000 / fs);
    let overlap = (overlap48 / inc) as usize;
    let inc = inc as usize;
    let ch = channels as usize;
    let g1 = Q15ONE - g1;
    let g2 = Q15ONE - g2;
    let mut i = 0usize;
    while i < overlap {
        let mut w: OpusVal16 = window[i * inc];
        w = w * w;
        let g: OpusVal16 = w * g2 + (Q15ONE - w) * g1;
        let mut diff: OpusVal32 = half32(buf[i * ch] - buf[i * ch + 1]);
        diff *= g;
        buf[i * ch] -= diff;
        buf[i * ch + 1] += diff;
        i += 1;
    }
    while i < frame_size as usize {
        let mut diff: OpusVal32 = half32(buf[i * ch] - buf[i * ch + 1]);
        diff *= g2;
        buf[i * ch] -= diff;
        buf[i * ch + 1] += diff;
        i += 1;
    }
}

/// Port of src/opus_encoder.c:gain_fade, in place (the encoder always calls it with
/// `in == out`): crossfades the gain from `g1` to `g2` over the CELT overlap.
#[cfg(not(feature = "fixed-point"))]
#[doc(hidden)]
pub fn gain_fade(
    buf: &mut [OpusRes],
    g1: OpusVal16,
    g2: OpusVal16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[CeltCoef],
    fs: i32,
) {
    let inc = imax(1, 48000 / fs);
    let overlap = (overlap48 / inc) as usize;
    let inc = inc as usize;
    if channels == 1 {
        for i in 0..overlap {
            let mut w: OpusVal16 = window[i * inc];
            w = w * w;
            let g: OpusVal16 = w * g2 + (Q15ONE - w) * g1;
            buf[i] *= g;
        }
    } else {
        for i in 0..overlap {
            let mut w: OpusVal16 = window[i * inc];
            w = w * w;
            let g: OpusVal16 = w * g2 + (Q15ONE - w) * g1;
            buf[i * 2] *= g;
            buf[i * 2 + 1] *= g;
        }
    }
    let ch = channels as usize;
    for c in 0..ch {
        for i in overlap..frame_size as usize {
            buf[i * ch + c] *= g2;
        }
    }
}

/// Port of src/opus_encoder.c:frame_size_select: the frame size to encode for an input of
/// `frame_size` samples given `OPUS_SET_EXPERT_FRAME_DURATION`, or -1 if invalid.
#[must_use]
pub fn frame_size_select(
    application: i32,
    frame_size: i32,
    variable_duration: i32,
    fs: i32,
) -> i32 {
    if frame_size < fs / 400 {
        return -1;
    }
    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if (OPUS_FRAMESIZE_2_5_MS..=OPUS_FRAMESIZE_120_MS).contains(&variable_duration) {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (fs / 400) << (variable_duration - OPUS_FRAMESIZE_2_5_MS)
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * fs / 50
        }
    } else {
        return -1;
    };
    if new_size > frame_size {
        return -1;
    }
    // The products overflow in C for absurd frame sizes (undefined behaviour); wrap like the
    // usual C build instead of panicking.
    let m = |k: i32| k.wrapping_mul(new_size);
    if m(400) != fs
        && m(200) != fs
        && m(100) != fs
        && m(50) != fs
        && m(25) != fs
        && m(50) != 3 * fs
        && m(50) != 4 * fs
        && m(50) != 5 * fs
        && m(50) != 6 * fs
    {
        return -1;
    }
    if application == OPUS_APPLICATION_RESTRICTED_SILK && new_size < fs / 100 {
        return -1;
    }
    new_size
}

/// Port of src/opus_encoder.c:compute_stereo_width (fixed-point build): smoothed estimate of
/// the stereo width (Q15) of `pcm` (interleaved stereo, `frame_size` samples per channel).
#[cfg(feature = "fixed-point")]
#[must_use]
#[doc(hidden)]
pub fn compute_stereo_width(
    pcm: &[OpusRes],
    frame_size: i32,
    fs: i32,
    mem: &mut StereoWidthState,
) -> OpusVal16 {
    let shift = celt_ilog2(frame_size) - 2;
    let frame_rate = fs / frame_size;
    let short_alpha = (mult16_16(25, Q15ONE) / imax(50, frame_rate)) as OpusVal16;
    let mut xx: OpusVal32 = 0;
    let mut xy: OpusVal32 = 0;
    let mut yy: OpusVal32 = 0;
    // Unroll by 4. The frame size is always a multiple of 4 *except* for 2.5 ms frames at
    // 12 kHz. Since this setting is very rare (and very stupid), we just discard the last two
    // samples.
    let mut i = 0i32;
    while i < frame_size - 3 {
        let iu = i as usize;
        let mut pxx: OpusVal32 = 0;
        let mut pxy: OpusVal32 = 0;
        let mut pyy: OpusVal32 = 0;
        for k in 0..4 {
            let x: OpusVal16 = res2val16(pcm[2 * iu + 2 * k]);
            let y: OpusVal16 = res2val16(pcm[2 * iu + 2 * k + 1]);
            pxx += shr32(mult16_16(x, x), 2);
            pxy += shr32(mult16_16(x, y), 2);
            pyy += shr32(mult16_16(y, y), 2);
        }
        xx += shr32(pxx, shift);
        xy += shr32(pxy, shift);
        yy += shr32(pyy, shift);
        i += 4;
    }
    mem.xx += mult16_32_q15(short_alpha, xx - mem.xx);
    // mem->XY += MULT16_32_Q15(short_alpha, xy-mem->XY);
    // Rewritten to avoid overflows on abrupt sign change.
    mem.xy = mult16_32_q15(i32::from(Q15ONE) - i32::from(short_alpha), mem.xy)
        + mult16_32_q15(short_alpha, xy);
    mem.yy += mult16_32_q15(short_alpha, yy - mem.yy);
    mem.xx = max32(0, mem.xx);
    mem.xy = max32(0, mem.xy);
    mem.yy = max32(0, mem.yy);
    if max32(mem.xx, mem.yy) > i32::from(qconst16(8e-4f32 as f64, 18)) {
        // C stores the `opus_val32` square roots in `opus_val16` variables.
        let sqrt_xx = celt_sqrt(mem.xx) as OpusVal16;
        let sqrt_yy = celt_sqrt(mem.yy) as OpusVal16;
        let qrrt_xx = celt_sqrt(i32::from(sqrt_xx)) as OpusVal16;
        let qrrt_yy = celt_sqrt(i32::from(sqrt_yy)) as OpusVal16;
        // Inter-channel correlation
        mem.xy = min32(mem.xy, i32::from(sqrt_xx) * i32::from(sqrt_yy));
        let corr = shr32(
            frac_div32(mem.xy, EPSILON + mult16_16(sqrt_xx, sqrt_yy)),
            16,
        ) as OpusVal16;
        // Approximate loudness difference
        let ldiff = (mult16_16(Q15ONE, abs16(i32::from(qrrt_xx) - i32::from(qrrt_yy)))
            / (EPSILON + i32::from(qrrt_xx) + i32::from(qrrt_yy))) as OpusVal16;
        let width = mult16_16_q15(
            min16(
                i32::from(Q15ONE),
                celt_sqrt(qconst32(1.0, 30) - mult16_16(corr, corr)),
            ),
            ldiff,
        ) as OpusVal16;
        // Smoothing over one second
        mem.smoothed_width = (i32::from(mem.smoothed_width)
            + (i32::from(width) - i32::from(mem.smoothed_width)) / frame_rate)
            as OpusVal16;
        // Peak follower
        mem.max_follower = max16(
            i32::from(mem.max_follower) - i32::from(qconst16(0.02f32 as f64, 15)) / frame_rate,
            i32::from(mem.smoothed_width),
        ) as OpusVal16;
    }
    extract16(min32(i32::from(Q15ONE), mult16_16(20, mem.max_follower)))
}

/// Port of src/opus_encoder.c:compute_stereo_width (float build): smoothed estimate of the
/// stereo width of `pcm` (interleaved stereo, `frame_size` samples per channel).
#[cfg(not(feature = "fixed-point"))]
#[must_use]
#[doc(hidden)]
pub fn compute_stereo_width(
    pcm: &[OpusRes],
    frame_size: i32,
    fs: i32,
    mem: &mut StereoWidthState,
) -> OpusVal16 {
    let frame_rate = fs / frame_size;
    let short_alpha: OpusVal16 = (25.0f32 * Q15ONE) / imax(50, frame_rate) as f32;
    let mut xx: OpusVal32 = 0.0;
    let mut xy: OpusVal32 = 0.0;
    let mut yy: OpusVal32 = 0.0;
    // Unroll by 4. The frame size is always a multiple of 4 *except* for 2.5 ms frames at
    // 12 kHz. Since this setting is very rare (and very stupid), we just discard the last two
    // samples.
    let mut i = 0i32;
    while i < frame_size - 3 {
        let iu = i as usize;
        let mut x: OpusVal16 = pcm[2 * iu];
        let mut y: OpusVal16 = pcm[2 * iu + 1];
        let mut pxx: OpusVal32 = x * x;
        let mut pxy: OpusVal32 = x * y;
        let mut pyy: OpusVal32 = y * y;
        x = pcm[2 * iu + 2];
        y = pcm[2 * iu + 3];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;
        x = pcm[2 * iu + 4];
        y = pcm[2 * iu + 5];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;
        x = pcm[2 * iu + 6];
        y = pcm[2 * iu + 7];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;

        xx += pxx;
        xy += pxy;
        yy += pyy;
        i += 4;
    }
    if !(xx < 1e9f32) || celt_isnan(xx) || !(yy < 1e9f32) || celt_isnan(yy) {
        xy = 0.0;
        xx = 0.0;
        yy = 0.0;
    }
    mem.xx += short_alpha * (xx - mem.xx);
    // mem->XY += MULT16_32_Q15(short_alpha, xy-mem->XY);
    // Rewritten to avoid overflows on abrupt sign change.
    mem.xy = (Q15ONE - short_alpha) * mem.xy + short_alpha * xy;
    mem.yy += short_alpha * (yy - mem.yy);
    mem.xx = max32(0.0, mem.xx);
    mem.xy = max32(0.0, mem.xy);
    mem.yy = max32(0.0, mem.yy);
    if max32(mem.xx, mem.yy) > 8e-4f32 {
        let sqrt_xx: OpusVal16 = celt_sqrt(mem.xx);
        let sqrt_yy: OpusVal16 = celt_sqrt(mem.yy);
        let qrrt_xx: OpusVal16 = celt_sqrt(sqrt_xx);
        let qrrt_yy: OpusVal16 = celt_sqrt(sqrt_yy);
        // Inter-channel correlation
        mem.xy = min32(mem.xy, sqrt_xx * sqrt_yy);
        let corr: OpusVal16 = frac_div32(mem.xy, crate::celt::arch::EPSILON + sqrt_xx * sqrt_yy);
        // Approximate loudness difference
        let ldiff: OpusVal16 =
            Q15ONE * abs16(qrrt_xx - qrrt_yy) / (crate::celt::arch::EPSILON + qrrt_xx + qrrt_yy);
        let width: OpusVal16 = min16(Q15ONE, celt_sqrt(1.0f32 - corr * corr)) * ldiff;
        // Smoothing over one second
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        // Peak follower
        mem.max_follower = max16(
            mem.max_follower - 0.02f32 / frame_rate as f32,
            mem.smoothed_width,
        );
    }
    min32(Q15ONE, 20.0f32 * mem.max_follower)
}

/// Port of src/opus_encoder.c:decide_fec: whether to code LBRR (in-band FEC) at the current
/// rate; may lower `bandwidth` to make room for it.
#[doc(hidden)]
pub fn decide_fec(
    use_in_band_fec: i32,
    packet_loss_perc: i32,
    last_fec: i32,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> i32 {
    if use_in_band_fec == 0 || packet_loss_perc == 0 || mode == MODE_CELT_ONLY {
        return 0;
    }
    let orig_bandwidth = *bandwidth;
    loop {
        // Compute threshold for using FEC at the current bandwidth setting
        let idx = (2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND)) as usize;
        let mut lbrr_rate_thres_bps = FEC_THRESHOLDS[idx];
        let hysteresis = FEC_THRESHOLDS[idx + 1];
        if last_fec == 1 {
            lbrr_rate_thres_bps -= hysteresis;
        }
        if last_fec == 0 {
            lbrr_rate_thres_bps += hysteresis;
        }
        lbrr_rate_thres_bps = silk_smulwb(
            silk_mul(lbrr_rate_thres_bps, 125 - silk_min(packet_loss_perc, 25)),
            silk_fix_const(0.01, 16),
        );
        // If loss <= 5%, we look at whether we have enough rate to enable FEC.
        // If loss > 5%, we decrease the bandwidth until we can enable FEC.
        if rate > lbrr_rate_thres_bps {
            return 1;
        } else if packet_loss_perc <= 5 {
            return 0;
        } else if *bandwidth > OPUS_BANDWIDTH_NARROWBAND {
            *bandwidth -= 1;
        } else {
            break;
        }
    }
    // Couldn't find any bandwidth to enable FEC, keep original bandwidth.
    *bandwidth = orig_bandwidth;
    0
}

/// Port of src/opus_encoder.c:compute_silk_rate_for_hybrid: the SILK share of a hybrid rate.
#[must_use]
#[doc(hidden)]
pub fn compute_silk_rate_for_hybrid(
    rate: i32,
    bandwidth: i32,
    frame20ms: i32,
    vbr: i32,
    fec: i32,
    channels: i32,
) -> i32 {
    static RATE_TABLE: [[i32; 5]; 7] = [
        //  |total| |-------- SILK------------|
        //          |-- No FEC -| |--- FEC ---|
        //           10ms   20ms   10ms   20ms
        [0, 0, 0, 0, 0],
        [12000, 10000, 10000, 11000, 11000],
        [16000, 13500, 13500, 15000, 15000],
        [20000, 16000, 16000, 18000, 18000],
        [24000, 18000, 18000, 21000, 21000],
        [32000, 22000, 22000, 28000, 28000],
        [64000, 38000, 38000, 50000, 50000],
    ];
    // Do the allocation per-channel.
    let rate = rate / channels;
    let entry = (1 + frame20ms + 2 * fec) as usize;
    let n = RATE_TABLE.len();
    let mut i = 1;
    while i < n {
        if RATE_TABLE[i][0] > rate {
            break;
        }
        i += 1;
    }
    let mut silk_rate;
    if i == n {
        silk_rate = RATE_TABLE[i - 1][entry];
        // For now, just give 50% of the extra bits to SILK.
        silk_rate += (rate - RATE_TABLE[i - 1][0]) / 2;
    } else {
        let lo = RATE_TABLE[i - 1][entry];
        let hi = RATE_TABLE[i][entry];
        let x0 = RATE_TABLE[i - 1][0];
        let x1 = RATE_TABLE[i][0];
        silk_rate = (lo * (x1 - rate) + hi * (rate - x0)) / (x1 - x0);
    }
    if vbr == 0 {
        // Tiny boost to SILK for CBR. We should probably tune this better.
        silk_rate += 100;
    }
    if bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND {
        silk_rate += 300;
    }
    silk_rate *= channels;
    // Small adjustment for stereo (calibrated for 32 kb/s, haven't tried other bitrates).
    if channels == 2 && rate >= 12000 {
        silk_rate -= 1000;
    }
    silk_rate
}

/// Port of src/opus_encoder.c:compute_equiv_rate: the equivalent bitrate corresponding to
/// 20 ms frames, complexity 10 VBR operation.
#[must_use]
#[doc(hidden)]
pub const fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: i32,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    // Take into account overhead from smaller frames.
    if frame_rate > 50 {
        equiv -= (40 * channels + 20) * (frame_rate - 50);
    }
    // CBR is about a 8% penalty for both SILK and CELT.
    if vbr == 0 {
        equiv -= equiv / 12;
    }
    // Complexity makes about 10% difference (from 0 to 10) in general.
    equiv = equiv * (90 + complexity) / 100;
    if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
        // SILK complexity 0-1 uses the non-delayed-decision NSQ, which costs about 20%.
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * loss / (6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        // CELT complexity 0-4 doesn't have the pitch filter, which costs about 10%.
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        // Mode not known yet
        // Half the SILK loss
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

/// Port of src/opus_encoder.c:compute_frame_energy (fixed-point build): mean energy per
/// sample of the 16-bit scaled signal (shifted to avoid overflows in the accumulation).
#[cfg(feature = "fixed-point")]
#[must_use]
#[doc(hidden)]
pub fn compute_frame_energy(pcm: &[OpusRes], frame_size: i32, channels: i32) -> OpusVal32 {
    let len = frame_size * channels;
    let pcm = &pcm[..len as usize];
    // Max amplitude in the signal
    #[cfg(feature = "fixed-res24")]
    let sample_max: OpusVal32 = i32::from(res2int16(celt_maxabs_res(pcm)));
    // RES2INT16 is the identity with a 16-bit `opus_res` (the maximum may be 32768).
    #[cfg(not(feature = "fixed-res24"))]
    let sample_max: OpusVal32 = celt_maxabs_res(pcm);

    // Compute the right shift required in the MAC to avoid an overflow
    let max_shift = celt_ilog2(len);
    let shift = imax(0, (celt_ilog2(1 + sample_max) << 1) + max_shift - 28);

    // Compute the energy
    let mut energy: OpusVal32 = 0;
    for &v in pcm {
        energy += shr32(mult16_16(res2int16(v), res2int16(v)), shift);
    }

    // Normalize energy by the frame size and left-shift back to the original position
    energy /= len;
    shl32(energy, shift)
}

/// Port of src/opus_encoder.c:compute_frame_energy (float build): mean energy per sample.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
#[doc(hidden)]
pub fn compute_frame_energy(pcm: &[OpusVal16], frame_size: i32, channels: i32) -> OpusVal32 {
    let len = frame_size * channels;
    celt_inner_prod(pcm, pcm, len as usize) / len as f32
}

/// Port of src/opus_encoder.c:decide_dtx_mode: decides if DTX should be turned on (1) or off
/// (0) for this frame.
#[doc(hidden)]
pub const fn decide_dtx_mode(
    activity: i32,
    nb_no_activity_ms_q1: &mut i32,
    frame_size_ms_q1: i32,
) -> i32 {
    if activity == 0 {
        // The number of consecutive DTX frames should be within the allowed bounds. Note that
        // the allowed bound is defined in the SILK headers and assumes 20 ms frames. As this
        // function can be called with any frame length, a conversion to milliseconds is done
        // before the comparisons.
        *nb_no_activity_ms_q1 += frame_size_ms_q1;
        if *nb_no_activity_ms_q1 > NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2 {
            if *nb_no_activity_ms_q1 <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2
            {
                // Valid frame for DTX!
                return 1;
            } else {
                *nb_no_activity_ms_q1 = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
            }
        }
    } else {
        *nb_no_activity_ms_q1 = 0;
    }
    0
}

/// Port of src/opus_encoder.c:compute_redundancy_bytes: size of a 5 ms redundant CELT frame.
#[must_use]
#[doc(hidden)]
pub const fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    let base_bits = 40 * channels + 20;

    // Equivalent rate for 5 ms frames.
    let mut redundancy_rate = bitrate_bps + base_bits * (200 - frame_rate);
    // For VBR, further increase the bitrate if we can afford it. It's pretty short and we'll
    // avoid artefacts.
    redundancy_rate = 3 * redundancy_rate / 2;
    let mut redundancy_bytes = redundancy_rate / 1600;

    // Compute the max rate we can use given CBR or VBR with cap.
    let available_bits = max_data_bytes * 8 - 2 * base_bits;
    let redundancy_bytes_cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    redundancy_bytes = imin(redundancy_bytes, redundancy_bytes_cap);
    // It we can't get enough bits for redundancy to be worth it, rely on the decoder PLC.
    if redundancy_bytes > 4 + 8 * channels {
        redundancy_bytes = imin(257, redundancy_bytes);
    } else {
        redundancy_bytes = 0;
    }
    redundancy_bytes
}

/// Port of src/repacketizer.c:opus_packet_pad using a caller-owned scratch buffer for the copy
/// of the packet (C uses a stack VLA): pads `data[..len]` to `new_len` bytes in place.
pub(crate) fn packet_pad_with(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    copy: &mut Vec<u8>,
) -> Result<()> {
    if len < 1 {
        return Err(Error::BadArg);
    }
    if len == new_len {
        return Ok(());
    } else if len > new_len {
        return Err(Error::BadArg);
    }
    let Some(out) = data.get_mut(..new_len as usize) else {
        return Err(Error::BadArg);
    };
    // Moving payload to the end of the packet so we can do in-place padding
    copy.clear();
    copy.extend_from_slice(&out[..len as usize]);
    let mut rp = Repacketizer::new();
    rp.cat(copy)?;
    rp.out_range_impl(0, rp.nb_frames(), out, false, true, &[])?;
    Ok(())
}

/// `opus_packet_pad_impl(data, len, new_len, pad, &extension, 1)` using a reusable copy buffer:
/// adds one extension (and, with `pad`, padding up to `new_len`). Returns the new length (C:
/// `OPUS_OK` = 0 when `len == new_len`).
#[cfg(feature = "dred")]
fn packet_pad_ext_with(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    pad: bool,
    extension: &Extension<'_>,
    copy: &mut Vec<u8>,
) -> Result<i32> {
    if len < 1 {
        return Err(Error::BadArg);
    }
    if len == new_len {
        return Ok(0);
    } else if len > new_len {
        return Err(Error::BadArg);
    }
    let Some(out) = data.get_mut(..new_len as usize) else {
        return Err(Error::BadArg);
    };
    // Moving payload to the end of the packet so we can do in-place padding
    copy.clear();
    copy.extend_from_slice(&out[..len as usize]);
    let mut rp = Repacketizer::new();
    rp.cat(copy)?;
    rp.out_range_impl(
        0,
        rp.nb_frames(),
        out,
        false,
        pad,
        core::slice::from_ref(extension),
    )
}

/// `dred_bits_table` (src/opus_encoder.c).
#[cfg(feature = "dred")]
const DRED_BITS_TABLE: [f32; 16] = [
    73.2, 68.1, 62.5, 57.0, 51.5, 45.7, 39.9, 32.4, 26.4, 20.4, 16.3, 13.0, 9.3, 8.2, 7.2, 6.4,
];

/// Port of src/opus_encoder.c:estimate_dred_bitrate: the bits of a DRED payload with
/// `duration` (10 ms units) of redundancy, and (C `*target_chunks`) the number of chunks that
/// fit in `target_bits`.
#[cfg(feature = "dred")]
fn estimate_dred_bitrate(
    q0: i32,
    dq: i32,
    qmax: i32,
    duration: i32,
    target_bits: i32,
) -> (i32, i32) {
    // Signaling DRED costs 3 bytes.
    let mut bits: f32 = (8 * (3 + DRED_EXPERIMENTAL_BYTES)) as f32;
    // Approximation for the size of the IS.
    bits += 50.0f32 + DRED_BITS_TABLE[q0 as usize];
    let dred_chunks = imin((duration + 5) / 4, DRED_NUM_REDUNDANCY_FRAMES as i32 / 2);
    let mut target_chunks = 0;
    for i in 0..dred_chunks {
        let q = compute_quantizer(q0, dq, qmax, i);
        bits += DRED_BITS_TABLE[q as usize];
        if bits < target_bits as f32 {
            target_chunks = i + 1;
        }
    }
    (
        crate::math::floor(f64::from(0.5f32 + bits)) as i32,
        target_chunks,
    )
}

/// Calls `celt_encode_with_ec`, passing the TOC byte only with QEXT.
#[inline]
fn celt_encode(
    celt: &mut CeltEncoder,
    pcm: &[OpusRes],
    frame_size: i32,
    compressed: Option<&mut [u8]>,
    nb_compressed_bytes: i32,
    enc: Option<&mut EcEnc<'_>>,
    toc: Option<&mut u8>,
) -> i32 {
    #[cfg(feature = "qext")]
    {
        celt.celt_encode_with_ec(pcm, frame_size, compressed, nb_compressed_bytes, enc, toc)
    }
    #[cfg(not(feature = "qext"))]
    {
        // Only QEXT signals through the TOC byte.
        let _ = toc;
        celt.celt_encode_with_ec(pcm, frame_size, compressed, nb_compressed_bytes, enc)
    }
}

/// Returns the CELT encoder, which exists unless the application is
/// `OPUS_APPLICATION_RESTRICTED_SILK` (callers only ask for it when C would use it).
#[inline]
fn celt_of(celt: &mut Option<Box<CeltEncoder>>) -> Result<&mut CeltEncoder> {
    celt.as_deref_mut().ok_or(Error::InternalError)
}

/// Returns the SILK encoder, which exists unless the application is
/// `OPUS_APPLICATION_RESTRICTED_CELT` (callers only ask for it when C would use it).
#[inline]
fn silk_of(silk: &mut Option<Box<SilkEncoder>>) -> Result<&mut SilkEncoder> {
    silk.as_deref_mut().ok_or(Error::InternalError)
}

/// A state value of [`Encoder::debug_state_dump`]: the bit pattern of a float, or the
/// sign-extended value of a fixed-point integer.
#[cfg(feature = "internals")]
trait DumpBits {
    fn dump_bits(self) -> u32;
}
#[cfg(feature = "internals")]
impl DumpBits for f32 {
    fn dump_bits(self) -> u32 {
        self.to_bits()
    }
}
#[cfg(feature = "internals")]
impl DumpBits for i32 {
    fn dump_bits(self) -> u32 {
        self as u32
    }
}
#[cfg(feature = "internals")]
impl DumpBits for i16 {
    fn dump_bits(self) -> u32 {
        i32::from(self) as u32
    }
}
/// [`DumpBits::dump_bits`].
#[cfg(feature = "internals")]
fn dump_bits(v: impl DumpBits) -> u32 {
    v.dump_bits()
}

/// Converts a `usize` sample count / length to the C `int`.
#[inline]
fn usize_to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::BadArg)
}

impl Encoder {
    /// Port of src/opus_encoder.c:opus_encoder_create / opus_encoder_init: creates an encoder.
    ///
    /// * `fs`: input sampling rate: 8000, 12000, 16000, 24000 or 48000 Hz (96000 with the
    ///   `qext` feature).
    /// * `channels`: 1 or 2.
    /// * `application`: the intended application.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an unsupported rate or channel count; [`Error::InternalError`] if a
    /// codec fails to initialise.
    pub fn new(fs: i32, channels: i32, application: Application) -> Result<Self> {
        Self::new_raw(fs, channels, application.to_raw())
    }

    /// [`Encoder::new`] taking the raw `OPUS_APPLICATION_*` value (C ABI layer).
    ///
    /// # Errors
    /// As [`Encoder::new`]; [`Error::BadArg`] for unknown applications.
    #[doc(hidden)]
    pub fn new_raw(fs: i32, channels: i32, application: i32) -> Result<Self> {
        check_init_args(fs, channels, application)?;
        let mut silk_mode = SilkEncControlStruct::default();

        // Create SILK encoder
        let silk_enc = if application != OPUS_APPLICATION_RESTRICTED_CELT {
            let mut s = Box::new(SilkEncoder::new());
            if s.init(channels, &mut silk_mode) != 0 {
                return Err(Error::InternalError);
            }
            Some(s)
        } else {
            None
        };

        // default SILK parameters
        silk_mode.n_channels_api = channels;
        silk_mode.n_channels_internal = channels;
        silk_mode.api_sample_rate = fs;
        silk_mode.max_internal_sample_rate = 16000;
        silk_mode.min_internal_sample_rate = 8000;
        silk_mode.desired_internal_sample_rate = 16000;
        silk_mode.payload_size_ms = 20;
        silk_mode.bit_rate = 25000;
        silk_mode.packet_loss_percentage = 0;
        silk_mode.complexity = 9;
        silk_mode.use_in_band_fec = 0;
        silk_mode.use_dred = 0;
        silk_mode.use_dtx = 0;
        silk_mode.use_cbr = 0;
        silk_mode.reduced_dependency = 0;

        // Create CELT encoder
        let celt_enc = if application != OPUS_APPLICATION_RESTRICTED_SILK {
            let Ok(mut c) = CeltEncoder::celt_encoder_init(fs, channels) else {
                return Err(Error::InternalError);
            };
            c.set_signalling(0);
            c.set_complexity(silk_mode.complexity)?;
            Some(Box::new(c))
        } else {
            None
        };

        // Initialize DRED Encoder
        #[cfg(feature = "dred")]
        let dred_encoder = DredEnc::new(fs, channels);

        let encoder_buffer = if application != OPUS_APPLICATION_RESTRICTED_CELT
            && application != OPUS_APPLICATION_RESTRICTED_SILK
        {
            fs / 100
        } else {
            0
        };
        let ch = channels as usize;
        #[cfg(not(feature = "disable-float-api"))]
        let mut analysis = Box::new(TonalityAnalysisState::new(fs));
        #[cfg(not(feature = "disable-float-api"))]
        {
            tonality_analysis_init(&mut analysis, fs);
            analysis.application = application;
        }

        #[allow(unused_mut, reason = "only mutated with compiled-in DNN weights")]
        let mut enc = Self {
            silk_mode,
            #[cfg(feature = "dred")]
            dred_encoder,
            application,
            channels,
            // Delay compensation of 4 ms (2.5 ms for SILK's extra look-ahead + 1.5 ms for SILK
            // resamplers and stereo prediction)
            delay_compensation: fs / 250,
            force_channels: OPUS_AUTO,
            signal_type: OPUS_AUTO,
            user_bandwidth: OPUS_AUTO,
            max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            user_forced_mode: OPUS_AUTO,
            voice_ratio: -1,
            fs,
            use_vbr: 1,
            // Makes constrained VBR the default (safer for real-time use)
            vbr_constraint: 1,
            variable_duration: OPUS_FRAMESIZE_ARG,
            bitrate_bps: 3000 + fs * channels,
            user_bitrate_bps: OPUS_AUTO,
            lsb_depth: 24,
            encoder_buffer,
            lfe: 0,
            use_dtx: 0,
            fec_config: 0,
            #[cfg(not(feature = "disable-float-api"))]
            analysis,
            #[cfg(feature = "qext")]
            enable_qext: 0,
            stream_channels: channels,
            hybrid_stereo_width_q14: 1 << 14,
            variable_hp_smth2_q15: silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8),
            prev_hb_gain: Q15ONE,
            hp_mem: [OpusVal32::default(); 4],
            mode: MODE_HYBRID,
            prev_mode: 0,
            prev_channels: 0,
            prev_framesize: 0,
            bandwidth: OPUS_BANDWIDTH_FULLBAND,
            auto_bandwidth: 0,
            silk_bw_switch: 0,
            first: 1,
            has_energy_mask: false,
            energy_masking: [CeltGlog::default(); 42],
            width_mem: StereoWidthState::default(),
            #[cfg(not(feature = "disable-float-api"))]
            detected_bandwidth: 0,
            nb_no_activity_ms_q1: 0,
            peak_signal_energy: OpusVal32::default(),
            #[cfg(feature = "dred")]
            dred_duration: 0,
            #[cfg(feature = "dred")]
            dred_q0: 0,
            #[cfg(feature = "dred")]
            dred_dq: 0,
            #[cfg(feature = "dred")]
            dred_qmax: 0,
            #[cfg(feature = "dred")]
            dred_target_chunks: 0,
            #[cfg(feature = "dred")]
            activity_mem: [0; DRED_ACTIVITY_MEM_SIZE],
            #[cfg(feature = "dred")]
            dred_bitrate_bps: 0,
            nonfinal_frame: 0,
            range_final: 0,
            delay_buffer: vec![OpusRes::default(); MAX_ENCODER_BUFFER * 2],
            silk_enc,
            celt_enc,
            scratch: Scratch {
                input: Vec::new(),
                pcm_buf: vec![OpusRes::default(); ((fs / 250 + 3 * fs / 50) as usize) * ch],
                tmp_prefill: vec![OpusRes::default(); ch * (fs / 400) as usize],
                tmp_data: Vec::new(),
                pad: Vec::new(),
            },
        };
        // Compiled-in weights (C: `dred_encoder_init` binds `rdovaeenc_arrays`).
        #[cfg(all(feature = "dnn-weights-embedded", feature = "dred"))]
        enc.set_dnn_blob(crate::dnn::embedded::DNN_BLOB)
            .map_err(|_| Error::InternalError)?;
        Ok(enc)
    }

    /// Port of src/opus_encoder.c:opus_encoder_init: re-initialises this encoder in place
    /// (all settings return to their defaults).
    ///
    /// # Errors
    /// As [`Encoder::new`]; on error the encoder is left unchanged.
    pub fn init(&mut self, fs: i32, channels: i32, application: Application) -> Result<()> {
        #[allow(unused_mut, reason = "only mutated with the dred feature")]
        let mut new = Self::new(fs, channels, application)?;
        // Keep a DRED model loaded by `set_dnn_blob` (it plays the role of upstream's
        // compiled-in weights, which `dred_encoder_init` binds again).
        #[cfg(feature = "dred")]
        {
            core::mem::swap(&mut new.dred_encoder, &mut self.dred_encoder);
            let loaded = new.dred_encoder.loaded;
            new.dred_encoder.dred_encoder_init(fs, channels);
            new.dred_encoder.loaded = loaded;
        }
        *self = new;
        Ok(())
    }

    /// Number of channels (1 or 2).
    #[must_use]
    pub const fn channels(&self) -> i32 {
        self.channels
    }

    // -----------------------------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------------------------

    /// Port of src/opus_encoder.c:opus_encode: encodes a frame of 16-bit PCM.
    ///
    /// * `pcm`: `frame_size * channels` interleaved samples.
    /// * `frame_size`: samples per channel: 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms at the
    ///   encoder rate (with `OPUS_SET_EXPERT_FRAME_DURATION`, the input may be longer than the
    ///   frame actually encoded).
    /// * `out`: output buffer; its length is the maximum packet size (`max_data_bytes`).
    ///
    /// Returns the packet length in bytes.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an invalid frame size or a too-short `pcm`;
    /// [`Error::BufferTooSmall`] if `out` cannot hold the packet; [`Error::InternalError`] on
    /// internal failures (as libopus).
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let analysis_frame_size = usize_to_i32(frame_size)?;
        // 16-bit fixed-point builds pass the input through (`opus_res` is `opus_int16`).
        #[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
        {
            self.encode_passthrough(pcm, pcm, analysis_frame_size, out, 16, downmix_int, 0)
        }
        #[cfg(any(not(feature = "fixed-point"), feature = "fixed-res24"))]
        {
            self.encode_converted(pcm, analysis_frame_size, out, 16, downmix_int, int16tores)
        }
    }

    /// Port of src/opus_encoder.c:opus_encode24: encodes a frame of 24-bit PCM (in `i32`,
    /// nominal range ±2^23). Otherwise as [`Encoder::encode`].
    ///
    /// # Errors
    /// As [`Encoder::encode`].
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let analysis_frame_size = usize_to_i32(frame_size)?;
        // 24-bit resolution builds pass the input through (`opus_res` is `opus_int32`).
        #[cfg(feature = "fixed-res24")]
        {
            self.encode_passthrough(
                pcm,
                pcm,
                analysis_frame_size,
                out,
                MAX_ENCODING_DEPTH,
                downmix_int24,
                0,
            )
        }
        #[cfg(not(feature = "fixed-res24"))]
        {
            self.encode_converted(
                pcm,
                analysis_frame_size,
                out,
                MAX_ENCODING_DEPTH,
                downmix_int24,
                int24tores,
            )
        }
    }

    /// Port of src/opus_encoder.c:opus_encode_float: encodes a frame of float PCM (nominal
    /// range ±1.0). Otherwise as [`Encoder::encode`]. Not available with the
    /// `disable-float-api` feature (libopus `DISABLE_FLOAT_API`).
    ///
    /// # Errors
    /// As [`Encoder::encode`].
    #[cfg(not(feature = "disable-float-api"))]
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        let analysis_frame_size = usize_to_i32(frame_size)?;
        // The float build passes the input through; the fixed-point build converts it with
        // FLOAT2RES.
        #[cfg(not(feature = "fixed-point"))]
        {
            self.encode_passthrough(
                pcm,
                pcm,
                analysis_frame_size,
                out,
                MAX_ENCODING_DEPTH,
                downmix_float,
                1,
            )
        }
        #[cfg(feature = "fixed-point")]
        {
            self.encode_converted(
                pcm,
                analysis_frame_size,
                out,
                MAX_ENCODING_DEPTH,
                downmix_float,
                float2res,
            )
        }
    }

    /// An `opus_encode*` entry point whose input already is `opus_res`: C passes it to
    /// `opus_encode_native` unchanged (`pcm` and `res` are the same samples; `res` is the
    /// `opus_res` view), without checking the frame size first.
    fn encode_passthrough<T>(
        &mut self,
        pcm: &[T],
        res: &[OpusRes],
        analysis_frame_size: i32,
        out: &mut [u8],
        lsb_depth: i32,
        downmix: DownmixFunc<T>,
        float_api: i32,
    ) -> Result<usize> {
        let frame_size = frame_size_select(
            self.application,
            analysis_frame_size,
            self.variable_duration,
            self.fs,
        );
        let ch = self.channels as usize;
        if frame_size > 0 && pcm.len() < analysis_frame_size as usize * ch {
            // Rust-only check (C reads out of bounds).
            return Err(Error::BadArg);
        }
        let n = frame_size.max(0) as usize * ch;
        let ret = self.opus_encode_native(
            &res[..n],
            frame_size,
            out,
            len_i32(out.len()),
            lsb_depth,
            Some(pcm),
            analysis_frame_size,
            0,
            -2,
            self.channels,
            downmix,
            float_api,
        );
        code_to_result(ret)
    }

    /// An `opus_encode*` entry point that converts its input to `opus_res` (C: `ALLOC(in, ...)`
    /// + `INT16TORES` / `INT24TORES` / `FLOAT2RES`) before `opus_encode_native` (`float_api` 1).
    fn encode_converted<T: Copy>(
        &mut self,
        pcm: &[T],
        analysis_frame_size: i32,
        out: &mut [u8],
        lsb_depth: i32,
        downmix: DownmixFunc<T>,
        conv: fn(T) -> OpusRes,
    ) -> Result<usize> {
        let frame_size = frame_size_select(
            self.application,
            analysis_frame_size,
            self.variable_duration,
            self.fs,
        );
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        let ch = self.channels as usize;
        if pcm.len() < analysis_frame_size as usize * ch {
            // Rust-only check (C reads out of bounds).
            return Err(Error::BadArg);
        }
        let n = frame_size as usize * ch;
        let mut input = core::mem::take(&mut self.scratch.input);
        if input.len() < n {
            input.resize(n, OpusRes::default());
        }
        for (d, &s) in input[..n].iter_mut().zip(&pcm[..n]) {
            *d = conv(s);
        }
        let ret = self.opus_encode_native(
            &input[..n],
            frame_size,
            out,
            len_i32(out.len()),
            lsb_depth,
            Some(pcm),
            analysis_frame_size,
            0,
            -2,
            self.channels,
            downmix,
            1,
        );
        self.scratch.input = input;
        code_to_result(ret)
    }

    /// Port of src/opus_encoder.c:user_bitrate_to_bitrate.
    const fn user_bitrate_to_bitrate(&self, frame_size: i32, max_data_bytes: i32) -> i32 {
        let frame_size = if frame_size == 0 {
            self.fs / 400
        } else {
            frame_size
        };
        let max_bitrate = bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);
        let user_bitrate = if self.user_bitrate_bps == OPUS_AUTO {
            60 * self.fs / frame_size + self.fs * self.channels
        } else if self.user_bitrate_bps == OPUS_BITRATE_MAX {
            1500000
        } else {
            self.user_bitrate_bps
        };
        imin(user_bitrate, max_bitrate)
    }

    /// Port of src/opus_encoder.c:opus_encode_native: the shared encoder entry point, with the
    /// C semantics (returns the packet length or a negative `OPUS_*` error code).
    ///
    /// * `pcm`: `frame_size*channels` samples (`opus_res`).
    /// * `data` / `out_data_bytes`: output buffer and the C `out_data_bytes` (at most
    ///   `data.len()` bytes are written).
    /// * `analysis_pcm` / `analysis_size` / `c1` / `c2` / `analysis_channels` / `downmix`: the
    ///   original input for the tonality analysis (see `run_analysis`).
    /// * `float_api`: 1 when the input may contain non-finite / huge values.
    ///
    /// Used by the multistream and projection encoders and the C ABI layer.
    #[doc(hidden)]
    pub fn opus_encode_native<T>(
        &mut self,
        pcm: &[OpusRes],
        frame_size: i32,
        data: &mut [u8],
        out_data_bytes: i32,
        lsb_depth: i32,
        analysis_pcm: Option<&[T]>,
        analysis_size: i32,
        c1: i32,
        c2: i32,
        analysis_channels: i32,
        downmix: DownmixFunc<T>,
        float_api: i32,
    ) -> i32 {
        let out_data_bytes = imin(out_data_bytes, len_i32(data.len()));
        // DISABLE_FLOAT_API: no tonality analysis; C: `(void)analysis_pcm; (void)analysis_size;
        // (void)c1; (void)c2; (void)analysis_channels; (void)downmix;`.
        #[cfg(feature = "disable-float-api")]
        let _ = (
            analysis_pcm,
            analysis_size,
            c1,
            c2,
            analysis_channels,
            downmix,
        );
        // Size: only the tonality analysis depends on the input sample type, so it is passed
        // to the (large, non-generic) body as a type-erased closure instead of monomorphizing
        // the whole encoder per sample type.
        #[cfg(not(feature = "disable-float-api"))]
        let mut analyze = |analysis: &mut TonalityAnalysisState,
                           celt_mode: &CeltMode,
                           frame_size: i32,
                           fs: i32,
                           lsb_depth: i32,
                           analysis_info: &mut AnalysisInfo| {
            run_analysis(
                analysis,
                celt_mode,
                analysis_pcm,
                analysis_size,
                frame_size,
                c1,
                c2,
                analysis_channels,
                fs,
                lsb_depth,
                downmix,
                analysis_info,
            );
        };
        match self.encode_native_impl(
            pcm,
            frame_size,
            data,
            out_data_bytes,
            lsb_depth,
            #[cfg(not(feature = "disable-float-api"))]
            &mut analyze,
            float_api,
        ) {
            Ok(n) => n,
            Err(e) => e.code(),
        }
    }

    #[expect(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "one C function; splitting it would obscure the correspondence"
    )]
    fn encode_native_impl(
        &mut self,
        pcm: &[OpusRes],
        frame_size: i32,
        data: &mut [u8],
        out_data_bytes: i32,
        lsb_depth: i32,
        #[cfg(not(feature = "disable-float-api"))] analyze: AnalyzeFn<'_>,
        float_api: i32,
    ) -> Result<i32> {
        let mut redundancy = 0;
        let mut celt_to_silk = 0;
        let mut to_celt = 0;
        let mut prefill = 0;
        let mut cbr_bytes = -1;
        #[allow(unused_mut, reason = "only changed with qext")]
        let mut packet_size_cap = 1276;
        #[cfg(feature = "qext")]
        if self.enable_qext != 0 {
            packet_size_cap = QEXT_PACKET_SIZE_CAP;
        }

        // Just avoid insane packet sizes here, but the real bounds are applied later on.
        let mut max_data_bytes = imin(packet_size_cap * 6, out_data_bytes);

        self.range_final = 0;
        if frame_size <= 0 || max_data_bytes <= 0 {
            return Err(Error::BadArg);
        }

        // Cannot encode 100 ms in 1 byte
        if max_data_bytes == 1 && self.fs == frame_size * 10 {
            return Err(Error::BufferTooSmall);
        }
        let ch = self.channels;
        if pcm.len() < (frame_size * ch) as usize {
            // Rust-only check (C reads out of bounds).
            return Err(Error::BadArg);
        }

        let lsb_depth = imin(lsb_depth, self.lsb_depth);

        let is_silence = is_digital_silence(pcm, frame_size, ch, lsb_depth) as i32;
        // With DISABLE_FLOAT_API C has no `analysis_info`; here it stays invalid (never read:
        // every C use is compiled out below and in `opus_encode_frame_native`).
        let mut analysis_info = AnalysisInfo::default();
        #[cfg_attr(
            feature = "disable-float-api",
            allow(unused_mut, reason = "only set by the tonality analysis")
        )]
        let mut analysis_read_pos_bak = -1;
        #[cfg(not(feature = "disable-float-api"))]
        let mut analysis_read_subframe_bak = -1;
        // The fixed-point build only runs the (float) analysis at complexity 10.
        #[cfg(all(feature = "fixed-point", not(feature = "disable-float-api")))]
        let analysis_complexity = 10;
        #[cfg(not(feature = "fixed-point"))]
        let analysis_complexity = 7;
        #[cfg(not(feature = "disable-float-api"))]
        if self.silk_mode.complexity >= analysis_complexity
            && self.fs >= 16000
            && self.fs <= 48000
            && self.application != OPUS_APPLICATION_RESTRICTED_SILK
        {
            analysis_read_pos_bak = self.analysis.read_pos;
            analysis_read_subframe_bak = self.analysis.read_subframe;
            let celt = celt_of(&mut self.celt_enc)?;
            // run_analysis(&st->analysis, celt_mode, analysis_pcm, analysis_size, frame_size,
            //              c1, c2, analysis_channels, st->Fs, lsb_depth, downmix,
            //              &analysis_info)
            analyze(
                &mut self.analysis,
                celt.mode(),
                frame_size,
                self.fs,
                lsb_depth,
                &mut analysis_info,
            );
        } else if self.analysis.initialized != 0 {
            tonality_analysis_reset(&mut self.analysis);
        }

        // Reset voice_ratio if this frame is not silent or if analysis is disabled.
        // Otherwise, preserve voice_ratio from the last non-silent frame
        if is_silence == 0 {
            self.voice_ratio = -1;
        }
        #[cfg(feature = "disable-float-api")]
        {
            self.voice_ratio = -1;
        }
        #[cfg(not(feature = "disable-float-api"))]
        {
            self.detected_bandwidth = 0;
        }
        #[cfg(not(feature = "disable-float-api"))]
        if analysis_info.valid != 0 {
            if self.signal_type == OPUS_AUTO {
                let prob: f32 = if self.prev_mode == 0 {
                    analysis_info.music_prob
                } else if self.prev_mode == MODE_CELT_ONLY {
                    analysis_info.music_prob_max
                } else {
                    analysis_info.music_prob_min
                };
                self.voice_ratio =
                    crate::math::floor(0.5f64 + (100.0f32 * (1.0f32 - prob)) as f64) as i32;
            }

            let analysis_bandwidth = analysis_info.bandwidth;
            self.detected_bandwidth = if analysis_bandwidth <= 12 {
                OPUS_BANDWIDTH_NARROWBAND
            } else if analysis_bandwidth <= 14 {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if analysis_bandwidth <= 16 {
                OPUS_BANDWIDTH_WIDEBAND
            } else if analysis_bandwidth <= 18 {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };
        }

        // Track the peak signal energy (DISABLE_FLOAT_API: without the analysis condition)
        if (cfg!(feature = "disable-float-api")
            || analysis_info.valid == 0
            || analysis_info.activity_probability > DTX_ACTIVITY_THRESHOLD)
            && is_silence == 0
        {
            #[cfg(feature = "fixed-point")]
            let decayed = mult16_32_q15(qconst16(0.999f32 as f64, 15), self.peak_signal_energy);
            #[cfg(not(feature = "fixed-point"))]
            let decayed = 0.999f32 * self.peak_signal_energy;
            self.peak_signal_energy = max32(decayed, compute_frame_energy(pcm, frame_size, ch));
        }
        let stereo_width: OpusVal16 = if ch == 2 && self.force_channels != 1 {
            compute_stereo_width(pcm, frame_size, self.fs, &mut self.width_mem)
        } else {
            OpusVal16::default()
        };
        self.bitrate_bps = self.user_bitrate_to_bitrate(frame_size, max_data_bytes);

        let mut frame_rate = self.fs / frame_size;
        if self.use_vbr == 0 {
            cbr_bytes = imin(
                (bitrate_to_bits(self.bitrate_bps, self.fs, frame_size) + 4) / 8,
                max_data_bytes,
            );
            self.bitrate_bps = bits_to_bitrate(cbr_bytes * 8, self.fs, frame_size);
            // Make sure we provide at least one byte to avoid failing.
            max_data_bytes = imax(1, cbr_bytes);
        }
        // Allocate some of the bits to DRED if needed.
        #[cfg(feature = "dred")]
        {
            self.dred_bitrate_bps = self.compute_dred_bitrate(self.bitrate_bps, frame_size);
            self.bitrate_bps -= self.dred_bitrate_bps;
        }
        if max_data_bytes < 3
            || self.bitrate_bps < 3 * frame_rate * 8
            || (frame_rate < 50 && (max_data_bytes * frame_rate < 300 || self.bitrate_bps < 2400))
        {
            // If the space is too low to do something useful, emit 'PLC' frames.
            let mut tocmode = self.mode;
            let mut bw = if self.bandwidth == 0 {
                OPUS_BANDWIDTH_NARROWBAND
            } else {
                self.bandwidth
            };
            let mut packet_code = 0;
            let mut num_multiframes = 0;

            if tocmode == 0 {
                tocmode = MODE_SILK_ONLY;
            }
            if frame_rate > 100 {
                tocmode = MODE_CELT_ONLY;
            }
            // 40 ms -> 2 x 20 ms if in CELT_ONLY or HYBRID mode
            if frame_rate == 25 && tocmode != MODE_SILK_ONLY {
                frame_rate = 50;
                packet_code = 1;
            }

            // >= 60 ms frames
            if frame_rate <= 16 {
                // 1 x 60 ms, 2 x 40 ms, 2 x 60 ms
                if out_data_bytes == 1 || (tocmode == MODE_SILK_ONLY && frame_rate != 10) {
                    tocmode = MODE_SILK_ONLY;

                    packet_code = (frame_rate <= 12) as i32;
                    frame_rate = if frame_rate == 12 { 25 } else { 16 };
                } else {
                    num_multiframes = 50 / frame_rate;
                    frame_rate = 50;
                    packet_code = 3;
                }
            }

            if tocmode == MODE_SILK_ONLY && bw > OPUS_BANDWIDTH_WIDEBAND {
                bw = OPUS_BANDWIDTH_WIDEBAND;
            } else if tocmode == MODE_CELT_ONLY && bw == OPUS_BANDWIDTH_MEDIUMBAND {
                bw = OPUS_BANDWIDTH_NARROWBAND;
            } else if tocmode == MODE_HYBRID && bw <= OPUS_BANDWIDTH_SUPERWIDEBAND {
                bw = OPUS_BANDWIDTH_SUPERWIDEBAND;
            }

            data[0] = gen_toc(tocmode, frame_rate, bw, self.stream_channels);
            data[0] |= packet_code as u8;

            let mut ret = if packet_code <= 1 { 1 } else { 2 };

            max_data_bytes = imax(max_data_bytes, ret);

            if packet_code == 3 {
                data[1] = num_multiframes as u8;
            }

            if self.use_vbr == 0 {
                ret = match packet_pad_with(data, ret, max_data_bytes, &mut self.scratch.pad) {
                    Ok(()) => max_data_bytes,
                    Err(_) => return Err(Error::InternalError),
                };
            }
            return Ok(ret);
        }
        let max_rate = bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);

        // Equivalent 20-ms rate for mode/channel/bandwidth decisions
        let mut equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            ch,
            self.fs / frame_size,
            self.use_vbr,
            0,
            self.silk_mode.complexity,
            self.silk_mode.packet_loss_percentage,
        );

        let voice_est: i32 = if self.signal_type == OPUS_SIGNAL_VOICE {
            127
        } else if self.signal_type == OPUS_SIGNAL_MUSIC {
            0
        } else if self.voice_ratio >= 0 {
            let v = (self.voice_ratio * 327) >> 8;
            // For AUDIO, never be more than 90% confident of having speech
            if self.application == OPUS_APPLICATION_AUDIO {
                imin(v, 115)
            } else {
                v
            }
        } else if self.application == OPUS_APPLICATION_VOIP {
            115
        } else {
            48
        };

        if self.force_channels != OPUS_AUTO && ch == 2 {
            self.stream_channels = self.force_channels;
        } else {
            // FUZZING: random mono/stereo decision not ported.
            // Rate-dependent mono-stereo decision
            if ch == 2 {
                let mut stereo_threshold = STEREO_MUSIC_THRESHOLD
                    + ((voice_est * voice_est * (STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD))
                        >> 14);
                if self.stream_channels == 2 {
                    stereo_threshold -= 1000;
                } else {
                    stereo_threshold += 1000;
                }
                self.stream_channels = if equiv_rate > stereo_threshold { 2 } else { 1 };
            } else {
                self.stream_channels = ch;
            }
        }
        // Update equivalent rate for channels decision.
        equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            self.stream_channels,
            self.fs / frame_size,
            self.use_vbr,
            0,
            self.silk_mode.complexity,
            self.silk_mode.packet_loss_percentage,
        );

        // Allow SILK DTX if DTX is enabled but the generalized DTX cannot be used,
        // e.g. because of the complexity setting or sample rate.
        #[cfg(not(feature = "disable-float-api"))]
        {
            self.silk_mode.use_dtx =
                (self.use_dtx != 0 && !(analysis_info.valid != 0 || is_silence != 0)) as i32;
        }
        #[cfg(feature = "disable-float-api")]
        {
            self.silk_mode.use_dtx = (self.use_dtx != 0 && is_silence == 0) as i32;
        }

        // Mode selection depending on application and signal type
        if self.application == OPUS_APPLICATION_RESTRICTED_SILK {
            self.mode = MODE_SILK_ONLY;
        } else if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY
            || self.application == OPUS_APPLICATION_RESTRICTED_CELT
        {
            self.mode = MODE_CELT_ONLY;
        } else if self.user_forced_mode == OPUS_AUTO {
            // FUZZING: random mode switching not ported.
            // Interpolate based on stereo width
            #[cfg(feature = "fixed-point")]
            let (mode_voice, mode_music) = {
                let inv = i32::from(Q15ONE) - i32::from(stereo_width);
                (
                    mult16_32_q15(inv, MODE_THRESHOLDS[0][0])
                        + mult16_32_q15(stereo_width, MODE_THRESHOLDS[1][0]),
                    // C quirk: MODE_THRESHOLDS[1][1] is used for both terms.
                    mult16_32_q15(inv, MODE_THRESHOLDS[1][1])
                        + mult16_32_q15(stereo_width, MODE_THRESHOLDS[1][1]),
                )
            };
            #[cfg(not(feature = "fixed-point"))]
            let mode_voice = ((Q15ONE - stereo_width) * MODE_THRESHOLDS[0][0] as f32
                + stereo_width * MODE_THRESHOLDS[1][0] as f32) as i32;
            // C quirk: MODE_THRESHOLDS[1][1] is used for both terms.
            #[cfg(not(feature = "fixed-point"))]
            let mode_music = ((Q15ONE - stereo_width) * MODE_THRESHOLDS[1][1] as f32
                + stereo_width * MODE_THRESHOLDS[1][1] as f32) as i32;
            // Interpolate based on speech/music probability
            let mut threshold =
                mode_music + ((voice_est * voice_est * (mode_voice - mode_music)) >> 14);
            // Bias towards SILK for VoIP because of some useful features
            if self.application == OPUS_APPLICATION_VOIP {
                threshold += 8000;
            }

            // Hysteresis
            if self.prev_mode == MODE_CELT_ONLY {
                threshold -= 4000;
            } else if self.prev_mode > 0 {
                threshold += 4000;
            }

            self.mode = if equiv_rate >= threshold {
                MODE_CELT_ONLY
            } else {
                MODE_SILK_ONLY
            };

            // When FEC is enabled and there's enough packet loss, use SILK. Unless the FEC is
            // set to 2, in which case we don't switch to SILK if we're confident we have music.
            if self.silk_mode.use_in_band_fec != 0
                && self.silk_mode.packet_loss_percentage > (128 - voice_est) >> 4
                && (self.fec_config != 2 || voice_est > 25)
            {
                self.mode = MODE_SILK_ONLY;
            }
            // When encoding voice and DTX is enabled but the generalized DTX cannot be used,
            // use SILK in order to make use of its DTX.
            if self.silk_mode.use_dtx != 0 && voice_est > 100 {
                self.mode = MODE_SILK_ONLY;
            }

            // If max_data_bytes represents less than 6 kb/s, switch to CELT-only mode
            if max_data_bytes
                < bitrate_to_bits(
                    if frame_rate > 50 { 9000 } else { 6000 },
                    self.fs,
                    frame_size,
                ) / 8
            {
                self.mode = MODE_CELT_ONLY;
            }
        } else {
            self.mode = self.user_forced_mode;
        }

        // Override the chosen mode to make sure we meet the requested frame size
        if self.mode != MODE_CELT_ONLY && frame_size < self.fs / 100 {
            debug_assert!(self.application != OPUS_APPLICATION_RESTRICTED_SILK);
            self.mode = MODE_CELT_ONLY;
        }
        if self.lfe != 0 && self.application != OPUS_APPLICATION_RESTRICTED_SILK {
            self.mode = MODE_CELT_ONLY;
        }

        if self.prev_mode > 0
            && ((self.mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY)
                || (self.mode == MODE_CELT_ONLY && self.prev_mode != MODE_CELT_ONLY))
        {
            redundancy = 1;
            celt_to_silk = (self.mode != MODE_CELT_ONLY) as i32;
            if celt_to_silk == 0 {
                // Switch to SILK/hybrid if frame size is 10 ms or more
                if frame_size >= self.fs / 100 {
                    self.mode = self.prev_mode;
                    to_celt = 1;
                } else {
                    redundancy = 0;
                }
            }
        }

        // When encoding multiframes, we can ask for a switch to CELT only in the last frame.
        // This switch is processed above as the requested mode shouldn't interrupt
        // stereo->mono transition.
        if self.stream_channels == 1
            && self.prev_channels == 2
            && self.silk_mode.to_mono == 0
            && self.mode != MODE_CELT_ONLY
            && self.prev_mode != MODE_CELT_ONLY
        {
            // Delay stereo->mono transition by two frames so that SILK can do a smooth downmix
            self.silk_mode.to_mono = 1;
            self.stream_channels = 2;
        } else {
            self.silk_mode.to_mono = 0;
        }

        // Update equivalent rate with mode decision.
        equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            self.stream_channels,
            self.fs / frame_size,
            self.use_vbr,
            self.mode,
            self.silk_mode.complexity,
            self.silk_mode.packet_loss_percentage,
        );

        if self.mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY {
            let mut dummy = SilkEncControlStruct::default();
            // C ignores the return value.
            let _init_ret: i32 = silk_of(&mut self.silk_enc)?.init(ch, &mut dummy);
            prefill = 1;
        }

        // Automatic (rate-dependent) bandwidth selection
        if self.mode == MODE_CELT_ONLY
            || self.first != 0
            || self.silk_mode.allow_bandwidth_switch != 0
        {
            let (voice_bandwidth_thresholds, music_bandwidth_thresholds) =
                if ch == 2 && self.force_channels != 1 {
                    (
                        &STEREO_VOICE_BANDWIDTH_THRESHOLDS,
                        &STEREO_MUSIC_BANDWIDTH_THRESHOLDS,
                    )
                } else {
                    (
                        &MONO_VOICE_BANDWIDTH_THRESHOLDS,
                        &MONO_MUSIC_BANDWIDTH_THRESHOLDS,
                    )
                };
            let mut bandwidth_thresholds = [0i32; 8];
            // Interpolate bandwidth thresholds depending on voice estimation
            for i in 0..8 {
                bandwidth_thresholds[i] = music_bandwidth_thresholds[i]
                    + ((voice_est
                        * voice_est
                        * (voice_bandwidth_thresholds[i] - music_bandwidth_thresholds[i]))
                        >> 14);
            }
            let mut bandwidth = OPUS_BANDWIDTH_FULLBAND;
            loop {
                let idx = (2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND)) as usize;
                let mut threshold = bandwidth_thresholds[idx];
                let hysteresis = bandwidth_thresholds[idx + 1];
                if self.first == 0 {
                    if self.auto_bandwidth >= bandwidth {
                        threshold -= hysteresis;
                    } else {
                        threshold += hysteresis;
                    }
                }
                if equiv_rate >= threshold {
                    break;
                }
                bandwidth -= 1;
                if bandwidth <= OPUS_BANDWIDTH_NARROWBAND {
                    break;
                }
            }
            // We don't use mediumband anymore, except when explicitly requested or during mode
            // transitions.
            if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
            self.bandwidth = bandwidth;
            self.auto_bandwidth = bandwidth;
            // Prevents any transition to SWB/FB until the SILK layer has fully switched to WB
            // mode and turned the variable LP filter off
            if self.first == 0
                && self.mode != MODE_CELT_ONLY
                && self.silk_mode.in_wb_mode_without_variable_lp == 0
                && self.bandwidth > OPUS_BANDWIDTH_WIDEBAND
            {
                self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        }

        if self.bandwidth > self.max_bandwidth {
            self.bandwidth = self.max_bandwidth;
        }

        if self.user_bandwidth != OPUS_AUTO {
            self.bandwidth = self.user_bandwidth;
        }

        // This prevents us from using hybrid at unsafe CBR/max rates
        if self.mode != MODE_CELT_ONLY && max_rate < 15000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_WIDEBAND);
        }

        // Prevents Opus from wasting bits on frequencies that are above the Nyquist rate of
        // the input signal
        if self.fs <= 24000 && self.bandwidth > OPUS_BANDWIDTH_SUPERWIDEBAND {
            self.bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
        }
        if self.fs <= 16000 && self.bandwidth > OPUS_BANDWIDTH_WIDEBAND {
            self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
        if self.fs <= 12000 && self.bandwidth > OPUS_BANDWIDTH_MEDIUMBAND {
            self.bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
        }
        if self.fs <= 8000 && self.bandwidth > OPUS_BANDWIDTH_NARROWBAND {
            self.bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        }
        // Use detected bandwidth to reduce the encoded bandwidth.
        #[cfg(not(feature = "disable-float-api"))]
        if self.detected_bandwidth != 0 && self.user_bandwidth == OPUS_AUTO {
            // Makes bandwidth detection more conservative just in case the detector gets it
            // wrong when we could have coded a high bandwidth transparently. When operating in
            // SILK/hybrid mode, we don't go below wideband to avoid more complicated switches
            // that require redundancy.
            let min_detected_bandwidth = if equiv_rate <= 18000 * self.stream_channels
                && self.mode == MODE_CELT_ONLY
            {
                OPUS_BANDWIDTH_NARROWBAND
            } else if equiv_rate <= 24000 * self.stream_channels && self.mode == MODE_CELT_ONLY {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if equiv_rate <= 30000 * self.stream_channels {
                OPUS_BANDWIDTH_WIDEBAND
            } else if equiv_rate <= 44000 * self.stream_channels {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };

            self.detected_bandwidth = imax(self.detected_bandwidth, min_detected_bandwidth);
            self.bandwidth = imin(self.bandwidth, self.detected_bandwidth);
        }
        self.silk_mode.lbrr_coded = decide_fec(
            self.silk_mode.use_in_band_fec,
            self.silk_mode.packet_loss_percentage,
            self.silk_mode.lbrr_coded,
            self.mode,
            &mut self.bandwidth,
            equiv_rate,
        );
        if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
            c_ignored(celt_of(&mut self.celt_enc)?.set_lsb_depth(lsb_depth))?;
        }

        // CELT mode doesn't support mediumband, use wideband instead
        if self.mode == MODE_CELT_ONLY && self.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
        if self.lfe != 0 {
            self.bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        }

        let mut curr_bandwidth = self.bandwidth;

        if self.application == OPUS_APPLICATION_RESTRICTED_SILK
            && curr_bandwidth > OPUS_BANDWIDTH_WIDEBAND
        {
            curr_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            self.bandwidth = curr_bandwidth;
        }
        // Chooses the appropriate mode for speech
        // *NEVER* switch to/from CELT-only mode here as this will invalidate some assumptions
        if self.mode == MODE_SILK_ONLY && curr_bandwidth > OPUS_BANDWIDTH_WIDEBAND {
            self.mode = MODE_HYBRID;
        }
        if self.mode == MODE_HYBRID && curr_bandwidth <= OPUS_BANDWIDTH_WIDEBAND {
            self.mode = MODE_SILK_ONLY;
        }

        // Can't support higher than >60 ms frames, and >20 ms when in Hybrid or CELT-only modes
        if (frame_size > self.fs / 50 && self.mode != MODE_SILK_ONLY)
            || frame_size > 3 * self.fs / 50
        {
            let enc_frame_size = if self.mode == MODE_SILK_ONLY {
                if frame_size == 2 * self.fs / 25 {
                    // 80 ms -> 2x 40 ms
                    self.fs / 25
                } else if frame_size == 3 * self.fs / 25 {
                    // 120 ms -> 2x 60 ms
                    3 * self.fs / 50
                } else {
                    // 100 ms -> 5x 20 ms
                    self.fs / 50
                }
            } else {
                self.fs / 50
            };

            let nb_frames = frame_size / enc_frame_size;

            #[cfg(not(feature = "disable-float-api"))]
            if analysis_read_pos_bak != -1 {
                // Reset analysis position to the beginning of the first frame so we can use it
                // one frame at a time.
                self.analysis.read_pos = analysis_read_pos_bak;
                self.analysis.read_subframe = analysis_read_subframe_bak;
            }

            // Worst cases:
            // 2 frames: Code 2 with different compressed sizes
            // >2 frames: Code 3 VBR
            #[allow(unused_mut, reason = "only changed with qext")]
            let mut max_header_bytes = if nb_frames == 2 {
                3
            } else {
                2 + (nb_frames - 1) * 2
            };
            // Cover the use of the separators that are the only thing that can get us over
            // once we consider that we need to subtract the extension overhead in each of the
            // individual frames. Also consider that a separator can get our padding from 254
            // to 255, which costs an extra length byte (at most once).
            #[cfg(feature = "qext")]
            if self.enable_qext != 0 {
                max_header_bytes += (nb_frames - 1) + 1;
            }

            let repacketize_len = if self.use_vbr != 0 || self.user_bitrate_bps == OPUS_BITRATE_MAX
            {
                out_data_bytes
            } else {
                debug_assert!(cbr_bytes >= 0);
                imin(cbr_bytes, out_data_bytes)
            };
            let max_len_sum = nb_frames + repacketize_len - max_header_bytes;

            let mut tmp_data = core::mem::take(&mut self.scratch.tmp_data);
            let need = max_len_sum.max(0) as usize;
            if tmp_data.len() < need {
                tmp_data.resize(need.max(MAX_PACKET_BYTES), 0);
            }
            let ret = self.encode_multiframe(
                pcm,
                &mut tmp_data[..need],
                data,
                frame_size,
                enc_frame_size,
                nb_frames,
                max_len_sum,
                repacketize_len,
                float_api,
                analysis_read_pos_bak,
                &mut analysis_info,
                lsb_depth,
                redundancy,
                celt_to_silk,
                prefill,
                equiv_rate,
                to_celt,
            );
            self.scratch.tmp_data = tmp_data;
            ret
        } else {
            self.opus_encode_frame_native(
                pcm,
                frame_size,
                data,
                max_data_bytes,
                float_api,
                1,
                &analysis_info,
                is_silence,
                redundancy,
                celt_to_silk,
                prefill,
                equiv_rate,
                to_celt,
            )
        }
    }

    /// The multi-frame branch of `opus_encode_native` (frames longer than 60 ms, or longer
    /// than 20 ms outside SILK-only mode): encodes `nb_frames` frames into `tmp_data` and
    /// repacketizes them into `data`.
    fn encode_multiframe(
        &mut self,
        pcm: &[OpusRes],
        tmp_data: &mut [u8],
        data: &mut [u8],
        #[cfg_attr(
            not(feature = "dred"),
            allow(unused_variables, reason = "only used by the dred feature")
        )]
        frame_size: i32,
        enc_frame_size: i32,
        nb_frames: i32,
        max_len_sum: i32,
        repacketize_len: i32,
        float_api: i32,
        #[cfg_attr(
            feature = "disable-float-api",
            allow(unused_variables, reason = "only used by the tonality analysis")
        )]
        analysis_read_pos_bak: i32,
        #[cfg_attr(
            feature = "disable-float-api",
            allow(
                clippy::needless_pass_by_ref_mut,
                reason = "only written by the tonality analysis"
            )
        )]
        analysis_info: &mut AnalysisInfo,
        lsb_depth: i32,
        redundancy: i32,
        celt_to_silk: i32,
        prefill: i32,
        equiv_rate: i32,
        to_celt: i32,
    ) -> Result<i32> {
        let ch = self.channels;
        let mut tot_size: i32 = 0;
        let mut dtx_count = 0;
        let mut rp = Repacketizer::new();

        let bak_to_mono = self.silk_mode.to_mono;
        if bak_to_mono != 0 {
            self.force_channels = 1;
        } else {
            self.prev_channels = self.stream_channels;
        }

        let mut remaining: &mut [u8] = tmp_data;
        for i in 0..nb_frames {
            // Attempt DRED encoding until we have a non-DTX frame. In case of DTX refresh,
            // that allows for DRED not to be in the first frame.
            let first_frame = (i == 0 || i == dtx_count) as i32;
            self.silk_mode.to_mono = 0;
            self.nonfinal_frame = (i < nb_frames - 1) as i32;

            // When switching from SILK/Hybrid to CELT, only ask for a switch at the last frame
            let frame_to_celt = (to_celt != 0 && i == nb_frames - 1) as i32;
            let frame_redundancy =
                (redundancy != 0 && (frame_to_celt != 0 || (to_celt == 0 && i == 0))) as i32;

            let mut curr_max = imin(
                bitrate_to_bits(self.bitrate_bps, self.fs, enc_frame_size) / 8,
                max_len_sum / nb_frames,
            );
            #[cfg(feature = "dred")]
            {
                let dred_bytes = bitrate_to_bits(self.dred_bitrate_bps, self.fs, frame_size) / 8;
                curr_max = imin(curr_max, (max_len_sum - dred_bytes) / nb_frames);
                if first_frame != 0 {
                    curr_max += dred_bytes;
                }
            }
            // Leave room for signaling the extension size once we repacketize.
            #[cfg(feature = "qext")]
            if self.enable_qext != 0 {
                curr_max -= curr_max / 254;
            }
            curr_max = imin(max_len_sum - tot_size, curr_max);
            #[cfg(not(feature = "disable-float-api"))]
            if analysis_read_pos_bak != -1 {
                // Get analysis for current frame.
                tonality_get_info(&mut self.analysis, analysis_info, enc_frame_size);
            }
            let off = (i * ch * enc_frame_size) as usize;
            let frame_pcm = &pcm[off..off + (ch * enc_frame_size) as usize];
            let is_silence = is_digital_silence(frame_pcm, enc_frame_size, ch, lsb_depth) as i32;

            let buf = core::mem::take(&mut remaining);
            let Ok(tmp_len) = self.opus_encode_frame_native(
                frame_pcm,
                enc_frame_size,
                buf,
                curr_max,
                float_api,
                first_frame,
                analysis_info,
                is_silence,
                frame_redundancy,
                celt_to_silk,
                prefill,
                equiv_rate,
                frame_to_celt,
            ) else {
                return Err(Error::InternalError);
            };
            if tmp_len == 1 {
                dtx_count += 1;
            }
            let (done, rest) = buf.split_at_mut(tmp_len as usize);
            if rp.cat(done).is_err() {
                return Err(Error::InternalError);
            }
            remaining = rest;
            tot_size += tmp_len;
        }
        let pad = self.use_vbr == 0 && dtx_count != nb_frames;
        let ret = match rp.out_range_impl(
            0,
            nb_frames,
            &mut data[..repacketize_len.max(0) as usize],
            false,
            pad,
            &[],
        ) {
            Ok(n) => n,
            Err(_) => Error::InternalError.code(),
        };
        self.silk_mode.to_mono = bak_to_mono;
        if ret < 0 {
            return Err(Error::InternalError);
        }
        Ok(ret)
    }

    /// Port of src/opus_encoder.c:opus_encode_frame_native: encodes one frame (at most 60 ms
    /// SILK / 20 ms CELT or hybrid) into `data[..orig_max_data_bytes]`.
    fn opus_encode_frame_native(
        &mut self,
        pcm: &[OpusRes],
        frame_size: i32,
        data: &mut [u8],
        orig_max_data_bytes: i32,
        float_api: i32,
        first_frame: i32,
        analysis_info: &AnalysisInfo,
        is_silence: i32,
        redundancy: i32,
        celt_to_silk: i32,
        prefill: i32,
        equiv_rate: i32,
        to_celt: i32,
    ) -> Result<i32> {
        let mut pcm_buf = core::mem::take(&mut self.scratch.pcm_buf);
        let mut tmp_prefill = core::mem::take(&mut self.scratch.tmp_prefill);
        let ret = self.frame_native_impl(
            pcm,
            frame_size,
            data,
            orig_max_data_bytes,
            float_api,
            first_frame,
            analysis_info,
            is_silence,
            redundancy,
            celt_to_silk,
            prefill,
            equiv_rate,
            to_celt,
            &mut pcm_buf,
            &mut tmp_prefill,
        );
        self.scratch.pcm_buf = pcm_buf;
        self.scratch.tmp_prefill = tmp_prefill;
        ret
    }

    #[expect(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "one C function; splitting it would obscure the correspondence"
    )]
    fn frame_native_impl(
        &mut self,
        pcm: &[OpusRes],
        frame_size: i32,
        data: &mut [u8],
        orig_max_data_bytes: i32,
        float_api: i32,
        first_frame: i32,
        analysis_info: &AnalysisInfo,
        is_silence: i32,
        mut redundancy: i32,
        mut celt_to_silk: i32,
        mut prefill: i32,
        equiv_rate: i32,
        to_celt: i32,
        pcm_buf: &mut [OpusRes],
        tmp_prefill: &mut [OpusRes],
    ) -> Result<i32> {
        let ch = self.channels;
        let chu = ch as usize;
        let fs = self.fs;
        let mut ret: i32 = 0;
        let mut redundancy_bytes: i32 = 0; // Number of bytes to use for redundancy frame
        let mut redundant_rng: u32 = 0;
        let mut activity = VAD_NO_DECISION;
        #[cfg(not(feature = "dred"))]
        let _ = first_frame; // Avoids a warning about first_frame being unused (C).

        let max_data_bytes = imin(orig_max_data_bytes, 1276);
        self.range_final = 0;
        if orig_max_data_bytes < 1 || data.len() < orig_max_data_bytes as usize {
            // Rust-only check (C writes out of bounds).
            return Err(Error::BadArg);
        }
        let data = &mut data[..orig_max_data_bytes as usize];
        let has_celt = self.celt_enc.is_some();
        let mut curr_bandwidth = self.bandwidth;
        let delay_compensation = if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY
            || self.application == OPUS_APPLICATION_RESTRICTED_CELT
            || self.application == OPUS_APPLICATION_RESTRICTED_SILK
        {
            0
        } else {
            self.delay_compensation
        };
        let total_buffer = delay_compensation;

        let frame_rate = fs / frame_size;

        if is_silence != 0 {
            activity = (is_silence == 0) as i32;
        } else if cfg!(not(feature = "disable-float-api")) && analysis_info.valid != 0 {
            activity = (analysis_info.activity_probability >= DTX_ACTIVITY_THRESHOLD) as i32;
            if activity == 0 {
                // Mark as active if this noise frame is sufficiently loud
                let noise_energy = compute_frame_energy(pcm, frame_size, ch);
                // C compares in float in both builds (PSEUDO_SNR_THRESHOLD is a float).
                #[cfg(feature = "fixed-point")]
                {
                    activity = ((self.peak_signal_energy as f32)
                        < PSEUDO_SNR_THRESHOLD * noise_energy as f32)
                        as i32;
                }
                #[cfg(not(feature = "fixed-point"))]
                {
                    activity =
                        (self.peak_signal_energy < PSEUDO_SNR_THRESHOLD * noise_energy) as i32;
                }
            }
        } else if self.mode == MODE_CELT_ONLY {
            let noise_energy = compute_frame_energy(pcm, frame_size, ch);
            // Boosting peak energy a bit because we didn't just average the active frames.
            // C: QCONST16(PSEUDO_SNR_THRESHOLD, 0) * (opus_val64)HALF32(noise_energy).
            #[cfg(feature = "fixed-point")]
            {
                activity = (i64::from(self.peak_signal_energy)
                    < i64::from(qconst16(f64::from(PSEUDO_SNR_THRESHOLD), 0))
                        * i64::from(half32(noise_energy))) as i32;
            }
            #[cfg(not(feature = "fixed-point"))]
            {
                activity =
                    (self.peak_signal_energy < PSEUDO_SNR_THRESHOLD * half32(noise_energy)) as i32;
            }
        }

        // For the first frame at a new SILK bandwidth
        if self.silk_bw_switch != 0 {
            redundancy = 1;
            celt_to_silk = 1;
            self.silk_bw_switch = 0;
            // Do a prefill without resetting the sampling rate control.
            prefill = 2;
        }

        // If we decided to go with CELT, make sure redundancy is off, no matter what we
        // decided earlier.
        if self.mode == MODE_CELT_ONLY {
            redundancy = 0;
        }

        if redundancy != 0 {
            redundancy_bytes = compute_redundancy_bytes(
                max_data_bytes,
                self.bitrate_bps,
                frame_rate,
                self.stream_channels,
            );
            if redundancy_bytes == 0 {
                redundancy = 0;
            }
        }
        if self.application == OPUS_APPLICATION_RESTRICTED_SILK {
            redundancy = 0;
            redundancy_bytes = 0;
        }

        let bits_target = imin(
            8 * (max_data_bytes - redundancy_bytes),
            bitrate_to_bits(self.bitrate_bps, fs, frame_size),
        ) - 8;

        let (toc, payload) = data.split_at_mut(1);
        let toc = &mut toc[0];
        let mut enc = EcEnc::new(payload);

        let tb = (total_buffer * ch) as usize;
        let pcm_buf = &mut pcm_buf[..((total_buffer + frame_size) * ch) as usize];
        let db_off = ((self.encoder_buffer - total_buffer) * ch) as usize;
        pcm_buf[..tb].copy_from_slice(&self.delay_buffer[db_off..db_off + tb]);

        let hp_freq_smth1 = if self.mode == MODE_CELT_ONLY {
            silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8)
        } else {
            silk_of(&mut self.silk_enc)?.state_fxx[0]
                .s_cmn
                .variable_hp_smth1_q15
        };

        self.variable_hp_smth2_q15 = silk_smlawb(
            self.variable_hp_smth2_q15,
            hp_freq_smth1 - self.variable_hp_smth2_q15,
            silk_fix_const(VARIABLE_HP_SMTH_COEF2 as f64, 16),
        );

        // convert from log scale to Hertz
        let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

        let n = (frame_size * ch) as usize;
        if self.application == OPUS_APPLICATION_VOIP {
            hp_cutoff(
                pcm,
                cutoff_hz,
                &mut pcm_buf[tb..],
                &mut self.hp_mem,
                frame_size,
                ch,
                fs,
            );
            #[cfg(feature = "osce-training-data")]
            {
                // write out high pass filtered clean signal
                // (C indexes `pcm_buf[total_buffer + idx]`, i.e. without the channel stride.)
                let tbs = total_buffer as usize;
                let mut b = Vec::with_capacity(2 * frame_size as usize);
                for &x in &pcm_buf[tbs..tbs + frame_size as usize] {
                    // `(opus_int16) (32768 * pcm_buf[..] + 0.5f)`: float to int, truncated to
                    // 16 bits (out-of-range values are UB in C; this matches aarch64 gcc).
                    let tmp = ((32768.0f32 * x + 0.5f32) as i32) as i16;
                    b.extend_from_slice(&tmp.to_ne_bytes());
                }
                crate::osce_training_data::write(crate::osce_training_data::CLEAN_HP, &b);
            }
        } else {
            #[cfg(feature = "qext")]
            let qext_copy = self.enable_qext != 0;
            #[cfg(not(feature = "qext"))]
            let qext_copy = false;
            // FIXME (C): Avoid glitching when we switch qext on/off dynamically.
            if qext_copy {
                pcm_buf[tb..tb + n].copy_from_slice(&pcm[..n]);
            } else {
                dc_reject(
                    pcm,
                    3,
                    &mut pcm_buf[tb..],
                    &mut self.hp_mem,
                    frame_size,
                    ch,
                    fs,
                );
            }
        }
        #[cfg(not(feature = "fixed-point"))]
        if float_api != 0 {
            let x = &pcm_buf[tb..tb + n];
            let sum = celt_inner_prod(x, x, n);
            // This should filter out both NaNs and ridiculous signals that could cause NaNs
            // further down.
            if !(sum < 1e9f32) || celt_isnan(sum) {
                pcm_buf[tb..tb + n].fill(0.0);
                self.hp_mem = [0.0; 4];
            }
        }
        #[cfg(feature = "fixed-point")]
        let _ = float_api; // C: (void)float_api;

        // Compute the DRED features. Needs to be before SILK because of DTX.
        #[cfg(feature = "dred")]
        if self.dred_duration > 0 && self.dred_encoder.loaded {
            // DRED Encoder
            dred_compute_latents(
                &mut self.dred_encoder,
                &pcm_buf[total_buffer as usize * chu..],
                frame_size,
                total_buffer,
            );
            let frame_size_400hz = (frame_size * 400 / fs) as usize;
            self.activity_mem.copy_within(
                ..DRED_ACTIVITY_MEM_SIZE - frame_size_400hz,
                frame_size_400hz,
            );
            // C stores the int `activity` (possibly VAD_NO_DECISION = -1) in unsigned chars.
            self.activity_mem[..frame_size_400hz].fill(activity as u8);
        } else {
            self.dred_encoder.latents_buffer_fill = 0;
            self.activity_mem = [0; DRED_ACTIVITY_MEM_SIZE];
        }

        // SILK processing
        let mut hb_gain: OpusVal16 = Q15ONE;
        if self.mode != MODE_CELT_ONLY {
            // Distribute bits between SILK and CELT
            let total_bit_rate = bits_to_bitrate(bits_target, fs, frame_size);
            if self.mode == MODE_HYBRID {
                // Base rate for SILK
                self.silk_mode.bit_rate = compute_silk_rate_for_hybrid(
                    total_bit_rate,
                    curr_bandwidth,
                    (fs == 50 * frame_size) as i32,
                    self.use_vbr,
                    self.silk_mode.lbrr_coded,
                    self.stream_channels,
                );
                if !self.has_energy_mask {
                    // Increasingly attenuate high band when it gets allocated fewer bits
                    let celt_rate = total_bit_rate - self.silk_mode.bit_rate;
                    // C passes the int product to the `opus_val16` argument of celt_exp2 (Q10)
                    // and stores the difference in the `opus_val16` HB_gain.
                    #[cfg(feature = "fixed-point")]
                    {
                        hb_gain = (i32::from(Q15ONE)
                            - shr32(
                                celt_exp2(
                                    (-celt_rate * i32::from(qconst16(1.0f32 as f64 / 1024.0, 10)))
                                        as OpusVal16,
                                ),
                                1,
                            )) as OpusVal16;
                    }
                    #[cfg(not(feature = "fixed-point"))]
                    {
                        hb_gain = Q15ONE - celt_exp2((-celt_rate) as f32 * (1.0f32 / 1024.0));
                    }
                }
            } else {
                // SILK gets all bits
                self.silk_mode.bit_rate = total_bit_rate;
            }

            // Surround masking for SILK
            #[cfg(feature = "fixed-point")]
            if self.has_energy_mask && self.use_vbr != 0 && self.lfe == 0 {
                let mut mask_sum: OpusVal32 = 0;
                let mut end = 17;
                let mut srate: i16 = 16000;
                if self.bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                    end = 13;
                    srate = 8000;
                } else if self.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                    end = 15;
                    srate = 12000;
                }
                for c in 0..chu {
                    for i in 0..end {
                        let mut mask: CeltGlog = maxg(
                            ming(self.energy_masking[21 * c + i], gconst(0.5)),
                            -gconst(2.0),
                        );
                        if mask > 0 {
                            mask = half32(mask);
                        }
                        mask_sum += mask;
                    }
                }
                // Conservative rate reduction, we cut the masking in half
                let mut masking_depth: CeltGlog = mask_sum / end as i32 * ch;
                masking_depth += gconst(0.2);
                let mut rate_offset =
                    pshr32(mult16_16(srate, shr32(masking_depth, DB_SHIFT - 10)), 10);
                rate_offset = max32(rate_offset, -2 * self.silk_mode.bit_rate / 3);
                // Split the rate change between the SILK and CELT part for hybrid.
                if self.bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND
                    || self.bandwidth == OPUS_BANDWIDTH_FULLBAND
                {
                    self.silk_mode.bit_rate += 3 * rate_offset / 5;
                } else {
                    self.silk_mode.bit_rate += rate_offset;
                }
            }
            #[cfg(not(feature = "fixed-point"))]
            if self.has_energy_mask && self.use_vbr != 0 && self.lfe == 0 {
                let mut mask_sum: OpusVal32 = 0.0;
                let mut end = 17;
                let mut srate: i16 = 16000;
                if self.bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                    end = 13;
                    srate = 8000;
                } else if self.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                    end = 15;
                    srate = 12000;
                }
                for c in 0..chu {
                    for i in 0..end {
                        let mut mask: CeltGlog =
                            maxg(ming(self.energy_masking[21 * c + i], 0.5f32), -2.0f32);
                        if mask > 0.0 {
                            mask = half32(mask);
                        }
                        mask_sum += mask;
                    }
                }
                // Conservative rate reduction, we cut the masking in half
                let mut masking_depth: CeltGlog = mask_sum / end as f32 * ch as f32;
                masking_depth += 0.2f32;
                let mut rate_offset = (srate as f32 * masking_depth) as i32;
                rate_offset = if rate_offset > -2 * self.silk_mode.bit_rate / 3 {
                    rate_offset
                } else {
                    -2 * self.silk_mode.bit_rate / 3
                };
                // Split the rate change between the SILK and CELT part for hybrid.
                if self.bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND
                    || self.bandwidth == OPUS_BANDWIDTH_FULLBAND
                {
                    self.silk_mode.bit_rate += 3 * rate_offset / 5;
                } else {
                    self.silk_mode.bit_rate += rate_offset;
                }
            }

            self.silk_mode.payload_size_ms = 1000 * frame_size / fs;
            self.silk_mode.n_channels_api = ch;
            self.silk_mode.n_channels_internal = self.stream_channels;
            if curr_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                self.silk_mode.desired_internal_sample_rate = 8000;
            } else if curr_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                self.silk_mode.desired_internal_sample_rate = 12000;
            } else {
                debug_assert!(
                    self.mode == MODE_HYBRID || curr_bandwidth == OPUS_BANDWIDTH_WIDEBAND
                );
                self.silk_mode.desired_internal_sample_rate = 16000;
            }
            if self.mode == MODE_HYBRID {
                // Don't allow bandwidth reduction at lowest bitrates in hybrid mode
                self.silk_mode.min_internal_sample_rate = 16000;
            } else {
                self.silk_mode.min_internal_sample_rate = 8000;
            }

            self.silk_mode.max_internal_sample_rate = 16000;
            if self.mode == MODE_SILK_ONLY {
                let mut effective_max_rate = bits_to_bitrate(max_data_bytes * 8, fs, frame_size);
                if frame_rate > 50 {
                    effective_max_rate = effective_max_rate * 2 / 3;
                }
                if effective_max_rate < 8000 {
                    self.silk_mode.max_internal_sample_rate = 12000;
                    self.silk_mode.desired_internal_sample_rate =
                        imin(12000, self.silk_mode.desired_internal_sample_rate);
                }
                if effective_max_rate < 7000 {
                    self.silk_mode.max_internal_sample_rate = 8000;
                    self.silk_mode.desired_internal_sample_rate =
                        imin(8000, self.silk_mode.desired_internal_sample_rate);
                }
                // At 96 kHz, we don't have the input resampler to do 8 or 12 kHz.
                #[cfg(feature = "qext")]
                if fs == 96000 {
                    self.silk_mode.max_internal_sample_rate = 16000;
                    self.silk_mode.desired_internal_sample_rate = 16000;
                }
            }

            self.silk_mode.use_cbr = (self.use_vbr == 0) as i32;

            // Call SILK encoder for the low band

            // Max bits for SILK, counting ToC, redundancy bytes, and optionally redundancy.
            self.silk_mode.max_bits = (max_data_bytes - 1) * 8;
            if redundancy != 0 && redundancy_bytes >= 2 {
                // Counting 1 bit for redundancy position and 20 bits for flag+size (only for
                // hybrid).
                self.silk_mode.max_bits -= redundancy_bytes * 8 + 1;
                if self.mode == MODE_HYBRID {
                    self.silk_mode.max_bits -= 20;
                }
            }
            if self.silk_mode.use_cbr != 0 {
                // When we're in CBR mode, but we have non-SILK data to encode, switch SILK to
                // VBR with cap to save on complexity. Any variations will be absorbed by CELT
                // and/or DRED and we can still produce a constant bitrate without wasting bits.
                #[cfg(feature = "dred")]
                let steal = self.mode == MODE_HYBRID || self.dred_bitrate_bps > 0;
                #[cfg(not(feature = "dred"))]
                let steal = self.mode == MODE_HYBRID;
                if steal {
                    // Allow SILK to steal up to 25% of the remaining bits
                    // C stores the value in an opus_int16.
                    let other_bits = imax(
                        0,
                        self.silk_mode.max_bits - self.silk_mode.bit_rate * frame_size / fs,
                    ) as i16 as i32;
                    self.silk_mode.max_bits = imax(0, self.silk_mode.max_bits - other_bits * 3 / 4);
                    self.silk_mode.use_cbr = 0;
                }
            } else {
                // Constrained VBR.
                if self.mode == MODE_HYBRID {
                    // Compute SILK bitrate corresponding to the max total bits available
                    let max_bit_rate = compute_silk_rate_for_hybrid(
                        self.silk_mode.max_bits * fs / frame_size,
                        curr_bandwidth,
                        (fs == 50 * frame_size) as i32,
                        self.use_vbr,
                        self.silk_mode.lbrr_coded,
                        self.stream_channels,
                    );
                    self.silk_mode.max_bits = bitrate_to_bits(max_bit_rate, fs, frame_size);
                }
            }

            if prefill != 0 && self.application != OPUS_APPLICATION_RESTRICTED_SILK {
                let mut zero: i32 = 0;
                // Use a smooth onset for the SILK prefill to avoid the encoder trying to
                // encode a discontinuity. The exact location is what we need to avoid leaving
                // any "gap" in the audio when mixing with the redundant CELT frame. Here we
                // can afford to overwrite st->delay_buffer because the only thing that uses it
                // before it gets rewritten is tmp_prefill[] and even then only the part after
                // the ramp really gets used (rather than sent to the encoder and discarded)
                let prefill_offset =
                    (ch * (self.encoder_buffer - self.delay_compensation - fs / 400)) as usize;
                let Some(celt) = self.celt_enc.as_deref() else {
                    return Err(Error::InternalError);
                };
                let mode = celt.mode();
                gain_fade(
                    &mut self.delay_buffer[prefill_offset..],
                    OpusVal16::default(),
                    Q15ONE,
                    mode.overlap,
                    fs / 400,
                    ch,
                    &mode.window,
                    fs,
                );
                self.delay_buffer[..prefill_offset].fill(OpusRes::default());
                let silk = silk_of(&mut self.silk_enc)?;
                // C passes a NULL range coder (never touched for a prefill) and ignores the
                // return value.
                let _prefill_ret: i32 = silk.silk_encode(
                    &mut self.silk_mode,
                    &self.delay_buffer,
                    self.encoder_buffer,
                    &mut EcEnc::new(&mut []),
                    &mut zero,
                    prefill,
                    activity,
                );
                // Prevent a second switch in the real encode call.
                self.silk_mode.opus_can_switch = 0;
            }

            let mut n_bytes: i32 = 0;
            let silk = silk_of(&mut self.silk_enc)?;
            ret = silk.silk_encode(
                &mut self.silk_mode,
                &pcm_buf[tb..],
                frame_size,
                &mut enc,
                &mut n_bytes,
                0,
                activity,
            );
            if ret != 0 {
                // Handle error
                return Err(Error::InternalError);
            }

            // Extract SILK internal bandwidth for signaling in first byte
            if self.mode == MODE_SILK_ONLY {
                if self.silk_mode.internal_sample_rate == 8000 {
                    curr_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
                } else if self.silk_mode.internal_sample_rate == 12000 {
                    curr_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
                } else if self.silk_mode.internal_sample_rate == 16000 {
                    curr_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
                }
            } else {
                debug_assert!(self.silk_mode.internal_sample_rate == 16000);
            }

            self.silk_mode.opus_can_switch =
                (self.silk_mode.switch_ready != 0 && self.nonfinal_frame == 0) as i32;

            if activity == VAD_NO_DECISION {
                activity = (self.silk_mode.signal_type != TYPE_NO_VOICE_ACTIVITY) as i32;
                #[cfg(feature = "dred")]
                {
                    let n = (frame_size * 400 / fs) as usize;
                    self.activity_mem[..n].fill(activity as u8);
                }
            }
            if n_bytes == 0 {
                self.range_final = 0;
                *toc = gen_toc(
                    self.mode,
                    fs / frame_size,
                    curr_bandwidth,
                    self.stream_channels,
                );
                return Ok(1);
            }

            // FIXME (C): How do we allocate the redundancy for CBR?
            if self.silk_mode.opus_can_switch != 0 {
                if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
                    redundancy_bytes = compute_redundancy_bytes(
                        max_data_bytes,
                        self.bitrate_bps,
                        frame_rate,
                        self.stream_channels,
                    );
                    redundancy = (redundancy_bytes != 0) as i32;
                }
                celt_to_silk = 0;
                self.silk_bw_switch = 1;
            }
        }

        // CELT processing
        if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
            let endband = match curr_bandwidth {
                OPUS_BANDWIDTH_NARROWBAND => 13,
                OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => 17,
                OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                _ => 21,
            };
            let celt = celt_of(&mut self.celt_enc)?;
            c_ignored(celt.set_end_band(endband))?;
            c_ignored(celt.set_channels(self.stream_channels))?;
            c_ignored(celt.set_bitrate(OPUS_BITRATE_MAX))?;
        }
        if self.mode != MODE_SILK_ONLY {
            // We may still decide to disable prediction later
            let celt_pred = if self.silk_mode.reduced_dependency != 0 {
                0
            } else {
                2
            };
            c_ignored(celt_of(&mut self.celt_enc)?.set_prediction(celt_pred))?;
        }

        let np = chu * (fs / 400) as usize;
        let tmp_prefill = &mut tmp_prefill[..np];
        if self.mode != MODE_SILK_ONLY
            && self.mode != self.prev_mode
            && self.prev_mode > 0
            && self.application != OPUS_APPLICATION_RESTRICTED_CELT
        {
            let off = ((self.encoder_buffer - total_buffer - fs / 400) * ch) as usize;
            tmp_prefill.copy_from_slice(&self.delay_buffer[off..off + np]);
        }

        let eb = self.encoder_buffer;
        if ch * (eb - (frame_size + total_buffer)) > 0 {
            let src = chu * frame_size as usize;
            let cnt = (ch * (eb - frame_size - total_buffer)) as usize;
            self.delay_buffer.copy_within(src..src + cnt, 0);
            let m = ((frame_size + total_buffer) * ch) as usize;
            self.delay_buffer[cnt..cnt + m].copy_from_slice(&pcm_buf[..m]);
        } else {
            let off = ((frame_size + total_buffer - eb) * ch) as usize;
            let cnt = (eb * ch) as usize;
            self.delay_buffer[..cnt].copy_from_slice(&pcm_buf[off..off + cnt]);
        }
        // gain_fade() and stereo_fade() need to be after the buffer copying because we don't
        // want any of this to affect the SILK part
        if (self.prev_hb_gain < Q15ONE || hb_gain < Q15ONE)
            && has_celt
            && let Some(celt) = self.celt_enc.as_deref()
        {
            let mode = celt.mode();
            gain_fade(
                pcm_buf,
                self.prev_hb_gain,
                hb_gain,
                mode.overlap,
                frame_size,
                ch,
                &mode.window,
                fs,
            );
        }
        self.prev_hb_gain = hb_gain;
        if self.mode != MODE_HYBRID || self.stream_channels == 1 {
            self.silk_mode.stereo_width_q14 = if equiv_rate > 32000 {
                16384
            } else if equiv_rate < 16000 {
                0
            } else {
                16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000)
            };
        }
        if !self.has_energy_mask && ch == 2 {
            // Apply stereo width reduction (at low bitrates)
            if self.hybrid_stereo_width_q14 < (1 << 14)
                || self.silk_mode.stereo_width_q14 < (1 << 14)
            {
                #[cfg(feature = "fixed-point")]
                let (g1, g2) = {
                    let g1: OpusVal16 = self.hybrid_stereo_width_q14;
                    let g2 = self.silk_mode.stereo_width_q14 as OpusVal16;
                    (
                        if g1 == 16384 { Q15ONE } else { shl16(g1, 1) },
                        if g2 == 16384 { Q15ONE } else { shl16(g2, 1) },
                    )
                };
                #[cfg(not(feature = "fixed-point"))]
                let (g1, g2) = {
                    let mut g1: OpusVal16 = self.hybrid_stereo_width_q14 as f32;
                    let mut g2: OpusVal16 = self.silk_mode.stereo_width_q14 as f32;
                    g1 *= 1.0f32 / 16384.0;
                    g2 *= 1.0f32 / 16384.0;
                    (g1, g2)
                };
                if let Some(celt) = self.celt_enc.as_deref() {
                    let mode = celt.mode();
                    stereo_fade(
                        pcm_buf,
                        g1,
                        g2,
                        mode.overlap,
                        frame_size,
                        ch,
                        &mode.window,
                        fs,
                    );
                }
                // C stores the int into an opus_int16.
                self.hybrid_stereo_width_q14 = self.silk_mode.stereo_width_q14 as i16;
            }
        }

        if self.mode != MODE_CELT_ONLY
            && enc.tell() + 17 + 20 * (self.mode == MODE_HYBRID) as i32 <= 8 * (max_data_bytes - 1)
        {
            // For SILK mode, the redundancy is inferred from the length
            if self.mode == MODE_HYBRID {
                enc.enc_bit_logp(redundancy != 0, 12);
            }
            if redundancy != 0 {
                enc.enc_bit_logp(celt_to_silk != 0, 1);
                let max_redundancy = if self.mode == MODE_HYBRID {
                    // Reserve the 8 bits needed for the redundancy length, and at least a few
                    // bits for CELT if possible
                    (max_data_bytes - 1) - ((enc.tell() + 8 + 3 + 7) >> 3)
                } else {
                    (max_data_bytes - 1) - ((enc.tell() + 7) >> 3)
                };
                // Target the same bit-rate for redundancy as for the rest, up to a max of 257
                // bytes
                redundancy_bytes = imin(max_redundancy, redundancy_bytes);
                redundancy_bytes = imin(257, imax(2, redundancy_bytes));
                if self.mode == MODE_HYBRID {
                    enc.enc_uint((redundancy_bytes - 2) as u32, 256);
                }
            }
        } else {
            redundancy = 0;
        }

        if redundancy == 0 {
            self.silk_bw_switch = 0;
            redundancy_bytes = 0;
        }
        let start_band = if self.mode != MODE_CELT_ONLY { 17 } else { 0 };

        let mut nb_compr_bytes: i32;
        if self.mode == MODE_SILK_ONLY {
            ret = (enc.tell() + 7) >> 3;
            enc.done();
            nb_compr_bytes = ret;
        } else {
            nb_compr_bytes = (max_data_bytes - 1) - redundancy_bytes;
            #[cfg(feature = "qext")]
            if self.mode == MODE_CELT_ONLY && self.enable_qext != 0 {
                debug_assert!(redundancy_bytes == 0);
                nb_compr_bytes = orig_max_data_bytes - 1;
            }
            #[cfg(feature = "dred")]
            if self.dred_duration > 0 {
                let dred_bytes = bitrate_to_bits(self.dred_bitrate_bps, fs, frame_size) / 8;
                // Allow CELT to steal up to 25% of the remaining bits.
                let mut max_celt_bytes = nb_compr_bytes - dred_bytes * 3 / 4;
                // But try to give CELT at least 5 bytes to prevent a mismatch with the
                // redundancy signaling.
                max_celt_bytes = imax((enc.tell() + 7) / 8 + 5, max_celt_bytes);
                // Subject to the original max.
                nb_compr_bytes = imin(nb_compr_bytes, max_celt_bytes);
            }
            enc.shrink(nb_compr_bytes as u32);
        }

        #[cfg(not(feature = "disable-float-api"))]
        if redundancy != 0 || self.mode != MODE_SILK_ONLY {
            celt_of(&mut self.celt_enc)?.set_analysis(Some(analysis_info));
        }
        if self.mode == MODE_HYBRID {
            let info = SilkInfo {
                signal_type: self.silk_mode.signal_type,
                offset: self.silk_mode.offset,
            };
            celt_of(&mut self.celt_enc)?.set_silk_info(Some(&info));
        }

        // 5 ms redundant frame for CELT->SILK
        if redundancy != 0 && celt_to_silk != 0 {
            let celt = celt_of(&mut self.celt_enc)?;
            c_ignored(celt.set_start_band(0))?;
            celt.set_vbr(0);
            c_ignored(celt.set_bitrate(OPUS_BITRATE_MAX))?;
            let nb = nb_compr_bytes as usize;
            let err = celt_encode(
                celt,
                pcm_buf,
                fs / 200,
                Some(&mut enc.buf[nb..nb + redundancy_bytes as usize]),
                redundancy_bytes,
                None,
                None,
            );
            if err < 0 {
                return Err(Error::InternalError);
            }
            redundant_rng = celt.final_range();
            celt.reset();
        }

        if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
            c_ignored(celt_of(&mut self.celt_enc)?.set_start_band(start_band))?;
        }

        *toc = 0;
        if self.mode != MODE_SILK_ONLY {
            let celt = celt_of(&mut self.celt_enc)?;
            celt.set_vbr(self.use_vbr);
            if self.mode == MODE_HYBRID {
                if self.use_vbr != 0 {
                    c_ignored(celt.set_bitrate(self.bitrate_bps - self.silk_mode.bit_rate))?;
                    celt.set_vbr_constraint(0);
                }
            } else if self.use_vbr != 0 {
                celt.set_vbr(1);
                celt.set_vbr_constraint(self.vbr_constraint);
                c_ignored(celt.set_bitrate(self.bitrate_bps))?;
            }
            // When Using DRED CBR, we can actually make the CELT part VBR and have DRED pick up
            // the slack.
            #[cfg(feature = "dred")]
            if self.use_vbr == 0 && self.dred_duration > 0 {
                let mut celt_bitrate = self.bitrate_bps;
                celt.set_vbr(1);
                celt.set_vbr_constraint(0);
                if self.mode == MODE_HYBRID {
                    celt_bitrate -= self.silk_mode.bit_rate;
                }
                c_ignored(celt.set_bitrate(celt_bitrate))?;
            }
            if self.mode != self.prev_mode
                && self.prev_mode > 0
                && self.application != OPUS_APPLICATION_RESTRICTED_CELT
            {
                let mut dummy = [0u8; 2];
                celt.reset();

                // Prefilling (C ignores the return value)
                let _prefill_ret: i32 =
                    celt_encode(celt, tmp_prefill, fs / 400, Some(&mut dummy), 2, None, None);
                c_ignored(celt.set_prediction(0))?;
            }
            // If false, we already busted the budget and we'll end up with a "PLC frame"
            if enc.tell() <= 8 * nb_compr_bytes {
                #[cfg(feature = "qext")]
                if self.mode == MODE_CELT_ONLY {
                    c_ignored(celt.set_qext(self.enable_qext))?;
                }
                ret = celt_encode(
                    celt,
                    pcm_buf,
                    frame_size,
                    None,
                    nb_compr_bytes,
                    Some(&mut enc),
                    Some(&mut *toc),
                );
                #[cfg(feature = "qext")]
                c_ignored(celt.set_qext(0))?;
                if ret < 0 {
                    return Err(Error::InternalError);
                }
                // Put CELT->SILK redundancy data in the right place.
                if redundancy != 0
                    && celt_to_silk != 0
                    && self.mode == MODE_HYBRID
                    && nb_compr_bytes != ret
                {
                    let nb = nb_compr_bytes as usize;
                    enc.buf
                        .copy_within(nb..nb + redundancy_bytes as usize, ret as usize);
                    nb_compr_bytes = ret + redundancy_bytes;
                }
            }
            self.range_final = celt.final_range();
        } else {
            self.range_final = enc.rng;
        }

        // 5 ms redundant frame for SILK->CELT
        if redundancy != 0 && celt_to_silk == 0 {
            let n2 = fs / 200;
            let n4 = fs / 400;
            let celt = celt_of(&mut self.celt_enc)?;

            celt.reset();
            c_ignored(celt.set_start_band(0))?;
            c_ignored(celt.set_prediction(0))?;
            celt.set_vbr(0);
            c_ignored(celt.set_bitrate(OPUS_BITRATE_MAX))?;

            if self.mode == MODE_HYBRID {
                // Shrink packet to what the encoder actually used.
                nb_compr_bytes = ret;
                enc.shrink(nb_compr_bytes as u32);
            }
            // NOTE: We could speed this up slightly (at the expense of code size) by just
            // adding a function that prefills the buffer
            let mut dummy = [0u8; 2];
            let _prefill_ret: i32 = celt_encode(
                celt,
                &pcm_buf[chu * (frame_size - n2 - n4) as usize..],
                n4,
                Some(&mut dummy),
                2,
                None,
                None,
            );

            let nb = nb_compr_bytes as usize;
            let err = celt_encode(
                celt,
                &pcm_buf[chu * (frame_size - n2) as usize..],
                n2,
                Some(&mut enc.buf[nb..nb + redundancy_bytes as usize]),
                redundancy_bytes,
                None,
                None,
            );
            if err < 0 {
                return Err(Error::InternalError);
            }
            redundant_rng = celt.final_range();
        }

        // Signalling the mode in the first byte
        *toc |= gen_toc(
            self.mode,
            fs / frame_size,
            curr_bandwidth,
            self.stream_channels,
        );

        self.range_final ^= redundant_rng;

        self.prev_mode = if to_celt != 0 {
            MODE_CELT_ONLY
        } else {
            self.mode
        };
        self.prev_channels = self.stream_channels;
        self.prev_framesize = frame_size;

        self.first = 0;

        // DTX decision
        if self.use_dtx != 0 && self.silk_mode.use_dtx == 0 {
            if decide_dtx_mode(
                activity,
                &mut self.nb_no_activity_ms_q1,
                2 * 1000 * frame_size / fs,
            ) != 0
            {
                self.range_final = 0;
                *toc = gen_toc(
                    self.mode,
                    fs / frame_size,
                    curr_bandwidth,
                    self.stream_channels,
                );
                return Ok(1);
            }
        } else {
            self.nb_no_activity_ms_q1 = 0;
        }

        // In the unlikely case that the SILK encoder busted its target, tell the decoder to
        // call the PLC
        // Last use of the range coder (its borrow of `data` ends here).
        let tell = enc.tell();
        if tell > (max_data_bytes - 1) * 8 {
            if max_data_bytes < 2 {
                return Err(Error::BufferTooSmall);
            }
            data[1] = 0;
            ret = 1;
            self.range_final = 0;
        } else if self.mode == MODE_SILK_ONLY && redundancy == 0 {
            // When in LPC only mode it's perfectly reasonable to strip off trailing zero bytes
            // as the required range decoder behavior is to fill these in. This can't be done
            // when the MDCT modes are used because the decoder needs to know the actual length
            // for allocation purposes.
            while ret > 2 && data[ret as usize] == 0 {
                ret -= 1;
            }
        }
        // Count ToC and redundancy
        ret += 1 + redundancy_bytes;
        #[allow(unused_mut, reason = "only cleared by the dred feature")]
        let mut apply_padding = self.use_vbr == 0;
        #[cfg(feature = "dred")]
        if self.dred_duration > 0 && self.dred_encoder.loaded && first_frame != 0 {
            let mut buf = [0u8; DRED_MAX_DATA_SIZE];
            let mut dred_chunks = imin(
                (self.dred_duration + 5) / 4,
                DRED_NUM_REDUNDANCY_FRAMES as i32 / 2,
            );
            if self.use_vbr != 0 {
                dred_chunks = imin(dred_chunks, self.dred_target_chunks);
            }
            // Remaining space for DRED, accounting for cost the 3 extra bytes for code 3,
            // padding length, and extension number.
            let mut dred_bytes_left =
                imin(DRED_MAX_DATA_SIZE as i32, orig_max_data_bytes - ret - 3);
            // Account for the extra bytes required to signal large padding length.
            dred_bytes_left -= (dred_bytes_left + 1 + DRED_EXPERIMENTAL_BYTES) / 255;
            // Check whether we actually have something to encode.
            if dred_chunks >= 1 && dred_bytes_left >= DRED_MIN_BYTES + DRED_EXPERIMENTAL_BYTES {
                // Add temporary extension type and version. These bytes will be removed once
                // extension is finalized.
                buf[0] = b'D';
                buf[1] = DRED_EXPERIMENTAL_VERSION as u8;
                // C passes `st->activity_mem`, and dred_voice_active can read up to 8 bytes past
                // it: the following struct fields `nonfinal_frame` and `rangeFinal` (native
                // byte order). Reproduce that memory.
                let mut activity = [0u8; DRED_ACTIVITY_MEM_SIZE + 8];
                activity[..DRED_ACTIVITY_MEM_SIZE].copy_from_slice(&self.activity_mem);
                activity[DRED_ACTIVITY_MEM_SIZE..DRED_ACTIVITY_MEM_SIZE + 4]
                    .copy_from_slice(&self.nonfinal_frame.to_ne_bytes());
                activity[DRED_ACTIVITY_MEM_SIZE + 4..]
                    .copy_from_slice(&self.range_final.to_ne_bytes());
                let eb = DRED_EXPERIMENTAL_BYTES as usize;
                let mut dred_bytes = dred_encode_silk_frame(
                    &mut self.dred_encoder,
                    &mut buf[eb..],
                    dred_chunks,
                    dred_bytes_left - DRED_EXPERIMENTAL_BYTES,
                    self.dred_q0,
                    self.dred_dq,
                    self.dred_qmax,
                    &activity,
                );
                if dred_bytes > 0 {
                    dred_bytes += DRED_EXPERIMENTAL_BYTES;
                    debug_assert!(dred_bytes <= dred_bytes_left);
                    let extension = Extension {
                        id: DRED_EXTENSION_ID,
                        frame: 0,
                        data: &buf[..dred_bytes as usize],
                    };
                    ret = match packet_pad_ext_with(
                        data,
                        ret,
                        orig_max_data_bytes,
                        self.use_vbr == 0,
                        &extension,
                        &mut self.scratch.pad,
                    ) {
                        Ok(r) => r,
                        Err(_) => return Err(Error::InternalError),
                    };
                    apply_padding = false;
                }
            }
        }
        if apply_padding {
            if packet_pad_with(data, ret, orig_max_data_bytes, &mut self.scratch.pad).is_err() {
                return Err(Error::InternalError);
            }
            ret = orig_max_data_bytes;
        }
        Ok(ret)
    }

    // -----------------------------------------------------------------------------------------
    // CTLs (src/opus_encoder.c:opus_encoder_ctl)
    // -----------------------------------------------------------------------------------------

    /// `OPUS_SET_APPLICATION`: changes the application. Only VoIP, Audio and restricted
    /// low-delay can be selected, and only before the first frame is encoded (or to the
    /// current value).
    ///
    /// # Errors
    /// [`Error::BadArg`] for restricted SILK/CELT encoders, a restricted SILK/CELT target, or a
    /// change after encoding has started.
    pub fn set_application(&mut self, application: Application) -> Result<()> {
        self.set_application_raw(application.to_raw())
    }

    #[cfg_attr(
        feature = "disable-float-api",
        expect(
            clippy::missing_const_for_fn,
            reason = "could only be const without the tonality analysis"
        )
    )]
    fn set_application_raw(&mut self, value: i32) -> Result<()> {
        if self.application == OPUS_APPLICATION_RESTRICTED_SILK
            || self.application == OPUS_APPLICATION_RESTRICTED_CELT
        {
            return Err(Error::BadArg);
        }
        if (value != OPUS_APPLICATION_VOIP
            && value != OPUS_APPLICATION_AUDIO
            && value != OPUS_APPLICATION_RESTRICTED_LOWDELAY)
            || (self.first == 0 && self.application != value)
        {
            return Err(Error::BadArg);
        }
        self.application = value;
        #[cfg(not(feature = "disable-float-api"))]
        {
            self.analysis.application = value;
        }
        Ok(())
    }

    /// `OPUS_GET_APPLICATION`.
    #[must_use]
    pub fn application(&self) -> Application {
        validated(Application::from_raw(self.application))
    }

    /// `OPUS_SET_BITRATE`: target bitrate. Explicit values are clamped to 500..=750000 b/s per
    /// channel.
    ///
    /// # Errors
    /// [`Error::BadArg`] for explicit values <= 0.
    pub const fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()> {
        let mut value = bitrate.to_raw();
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            if value <= 0 {
                return Err(Error::BadArg);
            } else if value <= 500 {
                value = 500;
            } else if value > 750000 * self.channels {
                value = 750000 * self.channels;
            }
        }
        self.user_bitrate_bps = value;
        Ok(())
    }

    /// `OPUS_GET_BITRATE`: the bitrate in b/s the encoder would use for the last frame size
    /// (resolving `OPUS_AUTO` / `OPUS_BITRATE_MAX`).
    #[must_use]
    pub const fn bitrate(&self) -> i32 {
        self.user_bitrate_to_bitrate(self.prev_framesize, 1276)
    }

    /// `OPUS_SET_FORCE_CHANNELS`: forces mono or stereo coding (`None` = automatic).
    ///
    /// # Errors
    /// [`Error::BadArg`] unless the value is 1 or at most the channel count.
    pub fn set_force_channels(&mut self, channels: Option<u8>) -> Result<()> {
        self.set_force_channels_raw(match channels {
            None => OPUS_AUTO,
            Some(c) => i32::from(c),
        })
    }

    const fn set_force_channels_raw(&mut self, value: i32) -> Result<()> {
        if (value < 1 || value > self.channels) && value != OPUS_AUTO {
            return Err(Error::BadArg);
        }
        self.force_channels = value;
        Ok(())
    }

    /// `OPUS_GET_FORCE_CHANNELS` (`None` = automatic).
    #[must_use]
    pub const fn force_channels(&self) -> Option<u8> {
        if self.force_channels == OPUS_AUTO {
            None
        } else {
            Some(self.force_channels as u8)
        }
    }

    /// `OPUS_SET_MAX_BANDWIDTH`: the maximum bandwidth the encoder may select automatically.
    pub const fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.max_bandwidth = bandwidth.to_raw();
        self.silk_mode.max_internal_sample_rate = match bandwidth {
            Bandwidth::Narrowband => 8000,
            Bandwidth::Mediumband => 12000,
            _ => 16000,
        };
    }

    /// `OPUS_GET_MAX_BANDWIDTH`.
    #[must_use]
    pub fn max_bandwidth(&self) -> Bandwidth {
        validated(Bandwidth::from_raw(self.max_bandwidth))
    }

    /// `OPUS_SET_BANDWIDTH`: forces the coded bandwidth (`None` = automatic, `OPUS_AUTO`).
    pub const fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        self.user_bandwidth = match bandwidth {
            None => OPUS_AUTO,
            Some(b) => b.to_raw(),
        };
        self.silk_mode.max_internal_sample_rate = match bandwidth {
            Some(Bandwidth::Narrowband) => 8000,
            Some(Bandwidth::Mediumband) => 12000,
            _ => 16000,
        };
    }

    /// `OPUS_GET_BANDWIDTH`: the bandwidth of the last encoded frame (or the initial
    /// full band).
    #[must_use]
    pub fn bandwidth(&self) -> Bandwidth {
        validated(Bandwidth::from_raw(self.bandwidth))
    }

    /// `OPUS_SET_DTX`: enables discontinuous transmission.
    pub const fn set_dtx(&mut self, enabled: bool) {
        self.use_dtx = enabled as i32;
    }

    /// `OPUS_GET_DTX`.
    #[must_use]
    pub const fn dtx(&self) -> bool {
        self.use_dtx != 0
    }

    /// `OPUS_SET_COMPLEXITY`: computational complexity 0..=10.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=10`.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<()> {
        if !(0..=10).contains(&complexity) {
            return Err(Error::BadArg);
        }
        self.silk_mode.complexity = complexity;
        if let Some(celt) = self.celt_enc.as_deref_mut() {
            c_ignored(celt.set_complexity(complexity))?;
        }
        Ok(())
    }

    /// `OPUS_GET_COMPLEXITY`.
    #[must_use]
    pub const fn complexity(&self) -> i32 {
        self.silk_mode.complexity
    }

    /// `OPUS_SET_INBAND_FEC`: 0 = off, 1 = on, 2 = on but not forcing SILK for music.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=2`.
    pub const fn set_inband_fec(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 2 {
            return Err(Error::BadArg);
        }
        self.fec_config = value;
        self.silk_mode.use_in_band_fec = (value != 0) as i32;
        Ok(())
    }

    /// `OPUS_GET_INBAND_FEC`.
    #[must_use]
    pub const fn inband_fec(&self) -> i32 {
        self.fec_config
    }

    /// `OPUS_SET_PACKET_LOSS_PERC`: expected packet loss percentage 0..=100.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=100`.
    pub fn set_packet_loss_perc(&mut self, percentage: i32) -> Result<()> {
        if !(0..=100).contains(&percentage) {
            return Err(Error::BadArg);
        }
        self.silk_mode.packet_loss_percentage = percentage;
        if let Some(celt) = self.celt_enc.as_deref_mut() {
            c_ignored(celt.set_packet_loss_perc(percentage))?;
        }
        Ok(())
    }

    /// `OPUS_GET_PACKET_LOSS_PERC`.
    #[must_use]
    pub const fn packet_loss_perc(&self) -> i32 {
        self.silk_mode.packet_loss_percentage
    }

    /// `OPUS_SET_VBR`: variable (true, default) or constant bitrate.
    pub const fn set_vbr(&mut self, enabled: bool) {
        self.use_vbr = enabled as i32;
        self.silk_mode.use_cbr = 1 - self.use_vbr;
    }

    /// `OPUS_GET_VBR`.
    #[must_use]
    pub const fn vbr(&self) -> bool {
        self.use_vbr != 0
    }

    /// `OPUS_SET_VBR_CONSTRAINT`: constrained VBR (true, default) or unconstrained.
    pub const fn set_vbr_constraint(&mut self, enabled: bool) {
        self.vbr_constraint = enabled as i32;
    }

    /// `OPUS_GET_VBR_CONSTRAINT`.
    #[must_use]
    pub const fn vbr_constraint(&self) -> bool {
        self.vbr_constraint != 0
    }

    /// `OPUS_SET_SIGNAL`: signal type hint.
    pub const fn set_signal(&mut self, signal: Signal) {
        self.signal_type = signal.to_raw();
    }

    /// `OPUS_GET_SIGNAL`.
    #[must_use]
    pub fn signal(&self) -> Signal {
        validated(Signal::from_raw(self.signal_type))
    }

    /// `OPUS_GET_LOOKAHEAD`: the encoder look-ahead in samples at the encoder rate.
    #[must_use]
    pub const fn lookahead(&self) -> usize {
        let mut value = self.fs / 400;
        if self.application != OPUS_APPLICATION_RESTRICTED_LOWDELAY
            && self.application != OPUS_APPLICATION_RESTRICTED_CELT
        {
            value += self.delay_compensation;
        }
        value as usize
    }

    /// `OPUS_GET_SAMPLE_RATE`: the rate the encoder was created with.
    #[must_use]
    pub const fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// `OPUS_GET_FINAL_RANGE`: the final range coder state of the last packet (for
    /// comparing with the decoder).
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.range_final
    }

    /// `OPUS_SET_LSB_DEPTH`: significant bits of the input (8..=24).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `8..=24`.
    pub const fn set_lsb_depth(&mut self, depth: i32) -> Result<()> {
        if depth < 8 || depth > 24 {
            return Err(Error::BadArg);
        }
        self.lsb_depth = depth;
        Ok(())
    }

    /// `OPUS_GET_LSB_DEPTH`.
    #[must_use]
    pub const fn lsb_depth(&self) -> i32 {
        self.lsb_depth
    }

    /// `OPUS_SET_EXPERT_FRAME_DURATION`: frame duration to encode (or [`FrameSize::Arg`] to
    /// use the `frame_size` argument).
    pub const fn set_expert_frame_duration(&mut self, duration: FrameSize) {
        self.variable_duration = duration.to_raw();
    }

    /// `OPUS_GET_EXPERT_FRAME_DURATION`.
    #[must_use]
    pub fn expert_frame_duration(&self) -> FrameSize {
        validated(FrameSize::from_raw(self.variable_duration))
    }

    /// `OPUS_SET_PREDICTION_DISABLED`: disables inter-frame prediction (more robust to loss,
    /// lower quality).
    pub const fn set_prediction_disabled(&mut self, disabled: bool) {
        self.silk_mode.reduced_dependency = disabled as i32;
    }

    /// `OPUS_GET_PREDICTION_DISABLED`.
    #[must_use]
    pub const fn prediction_disabled(&self) -> bool {
        self.silk_mode.reduced_dependency != 0
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED`: disables the use of phase inversion for intensity
    /// stereo.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        if let Some(celt) = self.celt_enc.as_deref_mut() {
            // Cannot fail for 0/1.
            if celt.set_phase_inversion_disabled(disabled as i32).is_err() {
                debug_assert!(false, "0/1 is always accepted");
            }
        }
    }

    /// `OPUS_GET_PHASE_INVERSION_DISABLED` (always false for restricted-SILK encoders).
    #[must_use]
    pub fn phase_inversion_disabled(&self) -> bool {
        match self.celt_enc.as_deref() {
            Some(celt) => celt.phase_inversion_disabled() != 0,
            None => false,
        }
    }

    /// `OPUS_GET_IN_DTX`: whether the last frame was a DTX frame (or the encoder is in DTX).
    #[must_use]
    pub fn in_dtx(&self) -> bool {
        if self.silk_mode.use_dtx != 0
            && (self.prev_mode == MODE_SILK_ONLY || self.prev_mode == MODE_HYBRID)
        {
            // DTX determined by Silk.
            let Some(silk) = self.silk_enc.as_deref() else {
                return false;
            };
            let mut value =
                silk.state_fxx[0].s_cmn.no_speech_counter >= NB_SPEECH_FRAMES_BEFORE_DTX;
            // Stereo: check second channel unless only the middle channel was encoded.
            if value && self.silk_mode.n_channels_internal == 2 && silk.prev_decode_only_middle == 0
            {
                value = silk.state_fxx[1].s_cmn.no_speech_counter >= NB_SPEECH_FRAMES_BEFORE_DTX;
            }
            value
        } else if self.use_dtx != 0 {
            // DTX determined by Opus.
            self.nb_no_activity_ms_q1 >= NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2
        } else {
            false
        }
    }

    /// `OPUS_SET_QEXT`: enables the quality extension (Opus HD). This hurts quality unless
    /// operating at very high bitrates.
    #[cfg(feature = "qext")]
    pub const fn set_qext(&mut self, enabled: bool) {
        self.enable_qext = enabled as i32;
    }

    /// `OPUS_GET_QEXT`.
    #[cfg(feature = "qext")]
    #[must_use]
    pub const fn qext(&self) -> bool {
        self.enable_qext != 0
    }

    /// `OPUS_SET_DRED_DURATION`: the amount of Deep REDundancy to send, in 10 ms units
    /// (0 = off, up to 104 = 1.04 s). DRED also needs the RDOVAE encoder model
    /// (`Encoder::set_dnn_blob`) and a packet loss percentage
    /// ([`Encoder::set_packet_loss_perc`]) to get a share of the bitrate.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=DRED_MAX_FRAMES` (104).
    #[cfg(feature = "dred")]
    pub const fn set_dred_duration(&mut self, duration: i32) -> Result<()> {
        if duration < 0 || duration > DRED_MAX_FRAMES as i32 {
            return Err(Error::BadArg);
        }
        self.dred_duration = duration;
        self.silk_mode.use_dred = (duration != 0) as i32;
        Ok(())
    }

    /// `OPUS_GET_DRED_DURATION`.
    #[cfg(feature = "dred")]
    #[must_use]
    pub const fn dred_duration(&self) -> i32 {
        self.dred_duration
    }

    /// `OPUS_SET_DNN_BLOB` (upstream `USE_WEIGHTS_FILE` builds): loads the DRED encoder
    /// models (RDOVAE encoder and pitch DNN arrays) from a libopus weight blob
    /// (`dred_encoder_load_model`).
    ///
    /// # Errors
    /// [`Error::BadArg`] if the blob cannot be parsed or lacks a required layer.
    #[cfg(feature = "dred")]
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        self.dred_encoder.load_model(data)
    }

    /// Whether the DRED encoder model is loaded (C `dred_encoder.loaded`).
    #[cfg(feature = "dred")]
    #[must_use]
    pub fn dred_loaded(&self) -> bool {
        self.dred_encoder.loaded
    }

    /// Port of src/opus_encoder.c:compute_dred_bitrate: the bitrate to reserve for DRED at
    /// `bitrate_bps` (also sets `dred_q0`, `dred_dQ`, `dred_qmax`, `dred_target_chunks`).
    #[cfg(feature = "dred")]
    fn compute_dred_bitrate(&mut self, bitrate_bps: i32, frame_size: i32) -> i32 {
        let mut dred_frac: f32;
        let bitrate_offset: i32;
        let plp = self.silk_mode.packet_loss_percentage;
        if self.silk_mode.use_in_band_fec != 0 {
            dred_frac = min16(0.7f32, 3.0f32 * plp as f32 / 100.0f32);
            bitrate_offset = 20000;
        } else {
            if plp > 5 {
                dred_frac = min16(0.8f32, 0.55f32 + plp as f32 / 100.0f32);
            } else {
                dred_frac = (12 * plp) as f32 / 100.0f32;
            }
            bitrate_offset = 12000;
        }
        // Account for the fact that longer packets require less redundancy.
        dred_frac = dred_frac
            / (dred_frac + (1.0 - dred_frac) * (frame_size as f32 * 50.0f32) / self.fs as f32);
        // Approximate fit based on a few experiments. Could probably be improved.
        let q0 = imin(
            15,
            imax(
                4,
                51 - 3 * ec_ilog(imax(1, bitrate_bps - bitrate_offset) as u32),
            ),
        );
        let dq = if bitrate_bps - bitrate_offset > 36000 {
            3
        } else {
            5
        };
        let qmax = 15;
        let target_dred_bitrate = imax(
            0,
            (dred_frac * (bitrate_bps - bitrate_offset) as f32) as i32,
        );
        let (max_dred_bits, target_chunks) = if self.dred_duration > 0 {
            let target_bits = bitrate_to_bits(target_dred_bitrate, self.fs, frame_size);
            estimate_dred_bitrate(q0, dq, qmax, self.dred_duration, target_bits)
        } else {
            (0, 0)
        };
        let mut dred_bitrate = imin(
            target_dred_bitrate,
            bits_to_bitrate(max_dred_bits, self.fs, frame_size),
        );
        // If we can't afford enough bits, don't bother with DRED at all.
        if target_chunks < 2 {
            dred_bitrate = 0;
        }
        self.dred_q0 = q0;
        self.dred_dq = dq;
        self.dred_qmax = qmax;
        self.dred_target_chunks = target_chunks;
        dred_bitrate
    }

    /// `OPUS_SET_VOICE_RATIO` (private): expected voice percentage 0..=100, or -1 for
    /// automatic.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `-1..=100`.
    #[doc(hidden)]
    pub const fn set_voice_ratio(&mut self, ratio: i32) -> Result<()> {
        if ratio < -1 || ratio > 100 {
            return Err(Error::BadArg);
        }
        self.voice_ratio = ratio;
        Ok(())
    }

    /// `OPUS_GET_VOICE_RATIO` (private).
    #[doc(hidden)]
    #[must_use]
    pub const fn voice_ratio(&self) -> i32 {
        self.voice_ratio
    }

    /// `OPUS_SET_FORCE_MODE` (private, request 11002): forces `MODE_SILK_ONLY` (1000),
    /// `MODE_HYBRID` (1001), `MODE_CELT_ONLY` (1002) or `OPUS_AUTO` (-1000).
    ///
    /// # Errors
    /// [`Error::BadArg`] for other values.
    #[doc(hidden)]
    pub const fn set_force_mode(&mut self, mode: i32) -> Result<()> {
        if (mode < MODE_SILK_ONLY || mode > MODE_CELT_ONLY) && mode != OPUS_AUTO {
            return Err(Error::BadArg);
        }
        self.user_forced_mode = mode;
        Ok(())
    }

    /// `OPUS_SET_LFE` (private): marks this stream as a low-frequency-effects channel.
    #[doc(hidden)]
    pub fn set_lfe(&mut self, lfe: i32) {
        self.lfe = lfe;
        if let Some(celt) = self.celt_enc.as_deref_mut() {
            celt.set_lfe(lfe);
        }
    }

    /// `OPUS_SET_ENERGY_MASK` (private): the surround masking curve (`21*channels` values),
    /// or `None` to disable it. The values are copied.
    #[doc(hidden)]
    pub fn set_energy_mask(&mut self, mask: Option<&[CeltGlog]>) {
        match mask {
            None => {
                self.has_energy_mask = false;
            }
            Some(m) => {
                let n = m.len().min(self.energy_masking.len());
                self.energy_masking[..n].copy_from_slice(&m[..n]);
                self.energy_masking[n..].fill(CeltGlog::default());
                self.has_energy_mask = true;
            }
        }
        if let Some(celt) = self.celt_enc.as_deref_mut() {
            celt.set_energy_mask(mask);
        }
    }

    /// `CELT_GET_MODE`: the CELT mode (absent for restricted-SILK encoders).
    pub(crate) fn celt_mode(&self) -> Option<&CeltMode> {
        self.celt_enc.as_deref().map(CeltEncoder::mode)
    }

    /// `OPUS_RESET_STATE`: resets the codec state to that of a freshly initialised encoder,
    /// keeping all settings.
    pub fn reset(&mut self) {
        #[cfg(not(feature = "disable-float-api"))]
        tonality_analysis_reset(&mut self.analysis);

        // OPUS_CLEAR from OPUS_ENCODER_RESET_START (stream_channels) to the codec states.
        self.stream_channels = 0;
        self.hybrid_stereo_width_q14 = 0;
        self.variable_hp_smth2_q15 = 0;
        self.prev_hb_gain = OpusVal16::default();
        self.hp_mem = [OpusVal32::default(); 4];
        self.mode = 0;
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.bandwidth = 0;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = 0;
        self.first = 0;
        self.has_energy_mask = false;
        self.energy_masking = [CeltGlog::default(); 42];
        self.width_mem = StereoWidthState::default();
        #[cfg(not(feature = "disable-float-api"))]
        {
            self.detected_bandwidth = 0;
        }
        self.nb_no_activity_ms_q1 = 0;
        self.peak_signal_energy = OpusVal32::default();
        #[cfg(feature = "dred")]
        {
            self.dred_duration = 0;
            self.dred_q0 = 0;
            self.dred_dq = 0;
            self.dred_qmax = 0;
            self.dred_target_chunks = 0;
            self.activity_mem = [0; DRED_ACTIVITY_MEM_SIZE];
        }
        self.nonfinal_frame = 0;
        self.range_final = 0;
        self.delay_buffer.fill(OpusRes::default());

        if let Some(celt) = self.celt_enc.as_deref_mut() {
            celt.reset();
        }
        if let Some(silk) = self.silk_enc.as_deref_mut() {
            let mut dummy = SilkEncControlStruct::default();
            // C ignores the return value.
            let _init_ret: i32 = silk.init(self.channels, &mut dummy);
        }
        // Initialize DRED Encoder
        #[cfg(feature = "dred")]
        self.dred_encoder.dred_encoder_reset();
        self.stream_channels = self.channels;
        self.hybrid_stereo_width_q14 = 1 << 14;
        self.prev_hb_gain = Q15ONE;
        self.first = 1;
        self.mode = MODE_HYBRID;
        self.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.variable_hp_smth2_q15 = silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8);
    }

    /// Flat dump of the encoder state in the order of the oracle shim's `oracle_oe_dump`
    /// (floats as bit patterns, fixed-point values sign-extended to 32 bits), plus the delay
    /// buffer (`encoder_buffer*channels` samples). For differential tests only.
    #[cfg(feature = "internals")]
    #[doc(hidden)]
    #[must_use]
    pub fn debug_state_dump(&self) -> (Vec<u32>, Vec<OpusRes>) {
        let s = &self.silk_mode;
        let v = vec![
            self.application as u32,
            self.channels as u32,
            self.delay_compensation as u32,
            self.force_channels as u32,
            self.signal_type as u32,
            self.user_bandwidth as u32,
            self.max_bandwidth as u32,
            self.user_forced_mode as u32,
            self.voice_ratio as u32,
            self.fs as u32,
            self.use_vbr as u32,
            self.vbr_constraint as u32,
            self.variable_duration as u32,
            self.bitrate_bps as u32,
            self.user_bitrate_bps as u32,
            self.lsb_depth as u32,
            self.encoder_buffer as u32,
            self.lfe as u32,
            self.use_dtx as u32,
            self.fec_config as u32,
            self.stream_channels as u32,
            self.hybrid_stereo_width_q14 as i32 as u32,
            self.variable_hp_smth2_q15 as u32,
            dump_bits(self.prev_hb_gain),
            dump_bits(self.hp_mem[0]),
            dump_bits(self.hp_mem[1]),
            dump_bits(self.hp_mem[2]),
            dump_bits(self.hp_mem[3]),
            self.mode as u32,
            self.prev_mode as u32,
            self.prev_channels as u32,
            self.prev_framesize as u32,
            self.bandwidth as u32,
            self.auto_bandwidth as u32,
            self.silk_bw_switch as u32,
            self.first as u32,
            dump_bits(self.width_mem.xx),
            dump_bits(self.width_mem.xy),
            dump_bits(self.width_mem.yy),
            dump_bits(self.width_mem.smoothed_width),
            dump_bits(self.width_mem.max_follower),
            // DISABLE_FLOAT_API: no `detected_bandwidth` (the C dump writes 0).
            #[cfg(not(feature = "disable-float-api"))]
            {
                self.detected_bandwidth as u32
            },
            #[cfg(feature = "disable-float-api")]
            0,
            self.nb_no_activity_ms_q1 as u32,
            dump_bits(self.peak_signal_energy),
            self.nonfinal_frame as u32,
            self.range_final,
            s.bit_rate as u32,
            s.max_bits as u32,
            s.to_mono as u32,
            s.lbrr_coded as u32,
            s.use_dtx as u32,
            s.use_cbr as u32,
            s.stereo_width_q14 as u32,
            s.internal_sample_rate as u32,
            s.opus_can_switch as u32,
            s.desired_internal_sample_rate as u32,
            s.max_internal_sample_rate as u32,
            s.min_internal_sample_rate as u32,
            s.payload_size_ms as u32,
            s.n_channels_internal as u32,
            s.allow_bandwidth_switch as u32,
            s.in_wb_mode_without_variable_lp as u32,
            s.switch_ready as u32,
        ];
        let n = (self.encoder_buffer * self.channels) as usize;
        (v, self.delay_buffer[..n].to_vec())
    }

    /// Numeric `opus_encoder_ctl` for requests taking one `opus_int32` argument (every
    /// `OPUS_SET_*` request, plus `OPUS_RESET_STATE`, whose `value` is ignored). Meant for the C
    /// ABI layer; Rust code should use the typed methods.
    ///
    /// # Errors
    /// The request's error ([`Error::BadArg`] for out-of-range values), or
    /// [`Error::Unimplemented`] for requests `opus_encoder_ctl` does not handle (including
    /// `OPUS_GET_*` requests and requests taking pointers).
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        match request {
            OPUS_SET_APPLICATION_REQUEST => self.set_application_raw(value),
            OPUS_SET_BITRATE_REQUEST => self.set_bitrate(Bitrate::from_raw(value)),
            OPUS_SET_FORCE_CHANNELS_REQUEST => self.set_force_channels_raw(value),
            OPUS_SET_MAX_BANDWIDTH_REQUEST => {
                let bw = Bandwidth::from_raw(value)?;
                self.set_max_bandwidth(bw);
                Ok(())
            }
            OPUS_SET_BANDWIDTH_REQUEST => {
                let bw = if value == OPUS_AUTO {
                    None
                } else {
                    Some(Bandwidth::from_raw(value)?)
                };
                self.set_bandwidth(bw);
                Ok(())
            }
            OPUS_SET_DTX_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_dtx(value != 0);
                Ok(())
            }
            OPUS_SET_COMPLEXITY_REQUEST => self.set_complexity(value),
            OPUS_SET_INBAND_FEC_REQUEST => self.set_inband_fec(value),
            OPUS_SET_PACKET_LOSS_PERC_REQUEST => self.set_packet_loss_perc(value),
            OPUS_SET_VBR_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_vbr(value != 0);
                Ok(())
            }
            OPUS_SET_VOICE_RATIO_REQUEST => self.set_voice_ratio(value),
            OPUS_SET_VBR_CONSTRAINT_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_vbr_constraint(value != 0);
                Ok(())
            }
            OPUS_SET_SIGNAL_REQUEST => {
                self.set_signal(Signal::from_raw(value)?);
                Ok(())
            }
            OPUS_SET_LSB_DEPTH_REQUEST => self.set_lsb_depth(value),
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => {
                self.set_expert_frame_duration(FrameSize::from_raw(value)?);
                Ok(())
            }
            OPUS_SET_PREDICTION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_prediction_disabled(value != 0);
                Ok(())
            }
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_phase_inversion_disabled(value != 0);
                Ok(())
            }
            #[cfg(feature = "dred")]
            OPUS_SET_DRED_DURATION_REQUEST => self.set_dred_duration(value),
            #[cfg(feature = "qext")]
            OPUS_SET_QEXT_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_qext(value != 0);
                Ok(())
            }
            OPUS_RESET_STATE => {
                self.reset();
                Ok(())
            }
            OPUS_SET_FORCE_MODE_REQUEST => self.set_force_mode(value),
            OPUS_SET_LFE_REQUEST => {
                self.set_lfe(value);
                Ok(())
            }
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_encoder_ctl` for requests writing one `opus_int32` (every `OPUS_GET_*`
    /// request; `OPUS_GET_FINAL_RANGE` returns the `u32` bits as `i32`). Meant for the C ABI
    /// layer; Rust code should use the typed methods.
    ///
    /// # Errors
    /// [`Error::Unimplemented`] for requests `opus_encoder_ctl` does not handle (including
    /// `OPUS_SET_*` requests and requests taking pointers).
    pub fn ctl_get(&self, request: i32) -> Result<i32> {
        Ok(match request {
            OPUS_GET_APPLICATION_REQUEST => self.application,
            OPUS_GET_BITRATE_REQUEST => self.bitrate(),
            OPUS_GET_FORCE_CHANNELS_REQUEST => self.force_channels,
            OPUS_GET_MAX_BANDWIDTH_REQUEST => self.max_bandwidth,
            OPUS_GET_BANDWIDTH_REQUEST => self.bandwidth,
            OPUS_GET_DTX_REQUEST => self.use_dtx,
            OPUS_GET_COMPLEXITY_REQUEST => self.silk_mode.complexity,
            OPUS_GET_INBAND_FEC_REQUEST => self.fec_config,
            OPUS_GET_PACKET_LOSS_PERC_REQUEST => self.silk_mode.packet_loss_percentage,
            OPUS_GET_VBR_REQUEST => self.use_vbr,
            OPUS_GET_VOICE_RATIO_REQUEST => self.voice_ratio,
            OPUS_GET_VBR_CONSTRAINT_REQUEST => self.vbr_constraint,
            OPUS_GET_SIGNAL_REQUEST => self.signal_type,
            OPUS_GET_LOOKAHEAD_REQUEST => self.lookahead() as i32,
            OPUS_GET_SAMPLE_RATE_REQUEST => self.fs,
            OPUS_GET_FINAL_RANGE_REQUEST => self.range_final as i32,
            OPUS_GET_LSB_DEPTH_REQUEST => self.lsb_depth,
            OPUS_GET_EXPERT_FRAME_DURATION_REQUEST => self.variable_duration,
            OPUS_GET_PREDICTION_DISABLED_REQUEST => self.silk_mode.reduced_dependency,
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => self.phase_inversion_disabled() as i32,
            #[cfg(feature = "dred")]
            OPUS_GET_DRED_DURATION_REQUEST => self.dred_duration,
            #[cfg(feature = "qext")]
            OPUS_GET_QEXT_REQUEST => self.enable_qext,
            OPUS_GET_IN_DTX_REQUEST => self.in_dtx() as i32,
            _ => return Err(Error::Unimplemented),
        })
    }
}

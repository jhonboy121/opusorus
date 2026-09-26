//! Port of src/analysis.c, src/analysis.h: the tonality / speech-music analysis that drives the
//! encoder's mode, bandwidth and CELT tuning decisions.
//!
//! The analysis itself is float in both builds (as in C, where `analysis.c` is part of the
//! float API that fixed-point builds keep). In fixed-point builds (`FIXED_POINT`) it is fed
//! fixed-point signal: the downmixed input and `inmem` are `celt_sig` (Q27, `SIG_SHIFT` above
//! 16-bit PCM), `silk_resampler_down2_hp` is integer, the FFT is the fixed-point kiss FFT
//! (`kiss_fft_cpx` of `i32`), and the energies are rescaled (`SCALE_ENER`, the high-band energy
//! compensation).
//!
//! Also contains:
//! * [`AnalysisInfo`] and [`LEAK_BANDS`] from `celt/celt.h` (the `celt_bands` unit owns
//!   `celt.rs`; the struct lives here so both the CELT encoder and the Opus encoder can use it);
//! * the downmix callbacks [`downmix_float`], [`downmix_int`], [`downmix_int24`] and
//!   [`is_digital_silence`] from `src/opus_encoder.c`, which the Opus encoder reuses.
//!
//! The whole file is `#ifndef DISABLE_FLOAT_API` in C; the float build always has it.
//!
//! The C `downmix_func` callback takes an untyped `const void *` PCM buffer. In Rust it is
//! [`DownmixFunc<T>`], generic over the PCM sample type, and [`run_analysis`] is generic over
//! the same `T`.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::celt::arch::{
    OpusRes, OpusVal32, float2sig, half32, imax, imin, int16tosig, int24tosig, max16, max32, min16,
    min32, mult16_32_q15, qconst16,
};
#[cfg(feature = "fixed-point")]
use crate::celt::arch::{OpusVal16, SIG_SHIFT, shr64};
#[cfg(not(feature = "fixed-point"))]
use crate::celt::arch::{abs16, celt_isnan};
use crate::celt::kiss_fft::opus_fft;
#[cfg(feature = "fixed-point")]
use crate::celt::mathops::celt_maxabs32;
use crate::celt::mathops::{PI, celt_maxabs_res, fast_atan2f, float2int};
use crate::celt::static_modes::{CeltMode, KissFftCpx, KissFftScalar};
use crate::math;
use crate::mlp::{
    LAYER0, LAYER1, LAYER2, MAX_NEURONS, analysis_compute_dense, analysis_compute_gru,
};

#[cfg(not(feature = "fixed-point"))]
pub use crate::celt::celt::LEAK_BANDS;

/// `NB_FRAMES`.
pub const NB_FRAMES: usize = 8;
/// `NB_TBANDS`.
pub const NB_TBANDS: usize = 18;
/// `ANALYSIS_BUF_SIZE`: 30 ms at 24 kHz.
pub const ANALYSIS_BUF_SIZE: usize = 720;
/// `ANALYSIS_COUNT_MAX`: at that point we can stop counting frames because it no longer matters.
pub const ANALYSIS_COUNT_MAX: i32 = 10000;
/// `DETECT_SIZE`.
pub const DETECT_SIZE: usize = 100;

/// `TRANSITION_PENALTY`.
const TRANSITION_PENALTY: i32 = 10;
/// `NB_TONAL_SKIP_BANDS`.
const NB_TONAL_SKIP_BANDS: usize = 9;
/// `LEAKAGE_OFFSET`.
const LEAKAGE_OFFSET: f32 = 2.5;
/// `LEAKAGE_SLOPE`.
const LEAKAGE_SLOPE: f32 = 2.0;

#[cfg(not(feature = "fixed-point"))]
pub use crate::celt::celt::AnalysisInfo;

/// `LEAK_BANDS` (celt/celt.h).
///
/// Fixed-point builds: private copy while `celt/celt.rs` is not converted yet (the float build
/// re-exports `crate::celt::celt::LEAK_BANDS`); to be replaced by that re-export once it is.
#[cfg(feature = "fixed-point")]
pub const LEAK_BANDS: usize = 19;

/// Port of celt/celt.h:AnalysisInfo: tonality analysis results passed from the Opus encoder to
/// CELT.
///
/// Fixed-point builds: private copy while `celt/celt.rs` is not converted yet (the float build
/// re-exports `crate::celt::celt::AnalysisInfo`); to be replaced by that re-export once it is.
#[cfg(feature = "fixed-point")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AnalysisInfo {
    pub valid: i32,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    /// Store as Q6 char to save space.
    pub leak_boost: [u8; LEAK_BANDS],
}

/// `ABS16` on a float in a fixed-point build: the `fixed_generic.h` ternary
/// `((x) < 0 ? (-(x)) : (x))` (keeps the sign of `-0.0`, unlike `fabsf`).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn abs16(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// C `(float)x` of an `opus_val32` / `kiss_fft_scalar` (identity in the float build).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn to_float(x: f32) -> f32 {
    x
}

/// C `(float)x` of an `opus_val32` / `kiss_fft_scalar` (`i32` in fixed-point builds).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn to_float(x: i32) -> f32 {
    x as f32
}

/// C `(kiss_fft_scalar)(w*x)`: windowed FFT input sample.
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn windowed(w: f32, x: OpusVal32) -> KissFftScalar {
    w * x
}

/// C `(kiss_fft_scalar)(w*x)`: windowed FFT input sample (float product truncated to `i32`).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn windowed(w: f32, x: OpusVal32) -> KissFftScalar {
    (w * x as f32) as KissFftScalar
}

/// `downmix_func` (src/opus_private.h): `downmix(x, y, subframe, offset, c1, c2, C)` mixes
/// `subframe` samples starting at sample `offset` of the interleaved `C`-channel buffer `x`
/// into `y` (see [`downmix_float`]).
pub type DownmixFunc<T> = fn(&[T], &mut [OpusVal32], i32, i32, i32, i32, i32);

/// `TonalityAnalysisState` (src/analysis.h).
///
/// Field names follow C in snake case (`E` → `e`, `logE` → `log_e`, `Etracker` → `e_tracker`,
/// ...). The C `arch` field is dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct TonalityAnalysisState {
    pub application: i32,
    pub fs: i32,
    // ---- TONALITY_ANALYSIS_RESET_START: everything below is cleared by reset ----
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [OpusVal32; ANALYSIS_BUF_SIZE],
    /// Number of usable samples in the buffer.
    pub mem_fill: i32,
    pub prev_band_tonality: [f32; NB_TBANDS],
    pub prev_tonality: f32,
    pub prev_bandwidth: i32,
    pub e: [[f32; NB_TBANDS]; NB_FRAMES],
    pub log_e: [[f32; NB_TBANDS]; NB_FRAMES],
    pub low_e: [f32; NB_TBANDS],
    pub high_e: [f32; NB_TBANDS],
    pub mean_e: [f32; NB_TBANDS + 1],
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    pub e_tracker: f32,
    pub low_e_count: f32,
    pub e_count: i32,
    pub count: i32,
    pub analysis_offset: i32,
    pub write_pos: i32,
    pub read_pos: i32,
    pub read_subframe: i32,
    pub hp_ener_accum: f32,
    pub initialized: i32,
    pub rnn_state: [f32; MAX_NEURONS],
    pub downmix_state: [OpusVal32; 3],
    pub info: [AnalysisInfo; DETECT_SIZE],
}

impl TonalityAnalysisState {
    /// Creates a state initialized with [`tonality_analysis_init`].
    #[must_use]
    pub fn new(fs: i32) -> Self {
        let mut s = Self {
            application: 0,
            fs: 0,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [OpusVal32::default(); ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            e: [[0.0; NB_TBANDS]; NB_FRAMES],
            log_e: [[0.0; NB_TBANDS]; NB_FRAMES],
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            e_tracker: 0.0,
            low_e_count: 0.0,
            e_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: 0,
            rnn_state: [0.0; MAX_NEURONS],
            downmix_state: [OpusVal32::default(); 3],
            info: [AnalysisInfo::default(); DETECT_SIZE],
        };
        tonality_analysis_init(&mut s, fs);
        s
    }
}

#[rustfmt::skip]
static DCT_TABLE: [f32; 128] = [
    0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
    0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
    0.351851, 0.338330, 0.311806, 0.273300, 0.224292, 0.166664, 0.102631, 0.034654,
    -0.034654,-0.102631,-0.166664,-0.224292,-0.273300,-0.311806,-0.338330,-0.351851,
    0.346760, 0.293969, 0.196424, 0.068975,-0.068975,-0.196424,-0.293969,-0.346760,
    -0.346760,-0.293969,-0.196424,-0.068975, 0.068975, 0.196424, 0.293969, 0.346760,
    0.338330, 0.224292, 0.034654,-0.166664,-0.311806,-0.351851,-0.273300,-0.102631,
    0.102631, 0.273300, 0.351851, 0.311806, 0.166664,-0.034654,-0.224292,-0.338330,
    0.326641, 0.135299,-0.135299,-0.326641,-0.326641,-0.135299, 0.135299, 0.326641,
    0.326641, 0.135299,-0.135299,-0.326641,-0.326641,-0.135299, 0.135299, 0.326641,
    0.311806, 0.034654,-0.273300,-0.338330,-0.102631, 0.224292, 0.351851, 0.166664,
    -0.166664,-0.351851,-0.224292, 0.102631, 0.338330, 0.273300,-0.034654,-0.311806,
    0.293969,-0.068975,-0.346760,-0.196424, 0.196424, 0.346760, 0.068975,-0.293969,
    -0.293969, 0.068975, 0.346760, 0.196424,-0.196424,-0.346760,-0.068975, 0.293969,
    0.273300,-0.166664,-0.338330, 0.034654, 0.351851, 0.102631,-0.311806,-0.224292,
    0.224292, 0.311806,-0.102631,-0.351851,-0.034654, 0.338330, 0.166664,-0.273300,
];

#[rustfmt::skip]
static ANALYSIS_WINDOW: [f32; 240] = [
    0.000043, 0.000171, 0.000385, 0.000685, 0.001071, 0.001541, 0.002098, 0.002739,
    0.003466, 0.004278, 0.005174, 0.006156, 0.007222, 0.008373, 0.009607, 0.010926,
    0.012329, 0.013815, 0.015385, 0.017037, 0.018772, 0.020590, 0.022490, 0.024472,
    0.026535, 0.028679, 0.030904, 0.033210, 0.035595, 0.038060, 0.040604, 0.043227,
    0.045928, 0.048707, 0.051564, 0.054497, 0.057506, 0.060591, 0.063752, 0.066987,
    0.070297, 0.073680, 0.077136, 0.080665, 0.084265, 0.087937, 0.091679, 0.095492,
    0.099373, 0.103323, 0.107342, 0.111427, 0.115579, 0.119797, 0.124080, 0.128428,
    0.132839, 0.137313, 0.141849, 0.146447, 0.151105, 0.155823, 0.160600, 0.165435,
    0.170327, 0.175276, 0.180280, 0.185340, 0.190453, 0.195619, 0.200838, 0.206107,
    0.211427, 0.216797, 0.222215, 0.227680, 0.233193, 0.238751, 0.244353, 0.250000,
    0.255689, 0.261421, 0.267193, 0.273005, 0.278856, 0.284744, 0.290670, 0.296632,
    0.302628, 0.308658, 0.314721, 0.320816, 0.326941, 0.333097, 0.339280, 0.345492,
    0.351729, 0.357992, 0.364280, 0.370590, 0.376923, 0.383277, 0.389651, 0.396044,
    0.402455, 0.408882, 0.415325, 0.421783, 0.428254, 0.434737, 0.441231, 0.447736,
    0.454249, 0.460770, 0.467298, 0.473832, 0.480370, 0.486912, 0.493455, 0.500000,
    0.506545, 0.513088, 0.519630, 0.526168, 0.532702, 0.539230, 0.545751, 0.552264,
    0.558769, 0.565263, 0.571746, 0.578217, 0.584675, 0.591118, 0.597545, 0.603956,
    0.610349, 0.616723, 0.623077, 0.629410, 0.635720, 0.642008, 0.648271, 0.654508,
    0.660720, 0.666903, 0.673059, 0.679184, 0.685279, 0.691342, 0.697372, 0.703368,
    0.709330, 0.715256, 0.721144, 0.726995, 0.732807, 0.738579, 0.744311, 0.750000,
    0.755647, 0.761249, 0.766807, 0.772320, 0.777785, 0.783203, 0.788573, 0.793893,
    0.799162, 0.804381, 0.809547, 0.814660, 0.819720, 0.824724, 0.829673, 0.834565,
    0.839400, 0.844177, 0.848895, 0.853553, 0.858151, 0.862687, 0.867161, 0.871572,
    0.875920, 0.880203, 0.884421, 0.888573, 0.892658, 0.896677, 0.900627, 0.904508,
    0.908321, 0.912063, 0.915735, 0.919335, 0.922864, 0.926320, 0.929703, 0.933013,
    0.936248, 0.939409, 0.942494, 0.945503, 0.948436, 0.951293, 0.954072, 0.956773,
    0.959396, 0.961940, 0.964405, 0.966790, 0.969096, 0.971321, 0.973465, 0.975528,
    0.977510, 0.979410, 0.981228, 0.982963, 0.984615, 0.986185, 0.987671, 0.989074,
    0.990393, 0.991627, 0.992778, 0.993844, 0.994826, 0.995722, 0.996534, 0.997261,
    0.997902, 0.998459, 0.998929, 0.999315, 0.999615, 0.999829, 0.999957, 1.000000,
];

static TBANDS: [i32; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

static STD_FEATURE_BIAS: [f32; 9] = [
    5.684947, 3.475288, 1.770634, 1.599784, 3.773215, 2.163313, 1.260756, 1.116868, 1.918795,
];

/// Maximum `subframe` handled by [`downmix_and_resample`] (20 ms at 48 kHz).
const MAX_DOWNMIX: usize = 960;

/// `SCALE_ENER` (float build): `(1.f/32768/32768)*e`.
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn scale_ener(e: f32) -> f32 {
    (1.0f32 / 32768.0 / 32768.0) * e
}

/// `SCALE_COMPENS` (fixed-point build): the input is ±2^15 shifted up by `SIG_SHIFT`, so the
/// energy is compensated for that.
#[cfg(feature = "fixed-point")]
const SCALE_COMPENS: f32 = 1.0f32 / (1i32 << (15 + SIG_SHIFT)) as f32;

/// `SCALE_ENER` (fixed-point build): `(SCALE_COMPENS*SCALE_COMPENS)*(e)`.
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn scale_ener(e: f32) -> f32 {
    (SCALE_COMPENS * SCALE_COMPENS) * e
}

/// Port of src/opus_encoder.c:downmix_float. `x` is interleaved float PCM (±1.0 full scale).
///
/// In fixed-point builds `FLOAT2SIG` converts (and clamps) each sample to `celt_sig`, and the
/// float-only +6 dBFS cap / NaN removal is not applied (as in C).
pub fn downmix_float(
    x: &[f32],
    y: &mut [OpusVal32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    let n = subframe as usize;
    let y = &mut y[..n];
    for (j, yj) in y.iter_mut().enumerate() {
        *yj = float2sig(x[((j as i32 + offset) * c + c1) as usize]);
    }
    if c2 > -1 {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj += float2sig(x[((j as i32 + offset) * c + c2) as usize]);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += float2sig(x[((j as i32 + offset) * c + ch) as usize]);
            }
        }
    }
    // Cap signal to +6 dBFS to avoid problems in the analysis.
    #[cfg(not(feature = "fixed-point"))]
    for yj in y.iter_mut() {
        // Same as the C pair of comparisons (a NaN passes through and is zeroed below).
        *yj = (*yj).clamp(-65536.0, 65536.0);
        if celt_isnan(*yj) {
            *yj = 0.0;
        }
    }
}

/// Port of src/opus_encoder.c:downmix_int. `x` is interleaved 16-bit PCM.
pub fn downmix_int(
    x: &[i16],
    y: &mut [OpusVal32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    let n = subframe as usize;
    let y = &mut y[..n];
    for (j, yj) in y.iter_mut().enumerate() {
        *yj = int16tosig(x[((j as i32 + offset) * c + c1) as usize]);
    }
    if c2 > -1 {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj += int16tosig(x[((j as i32 + offset) * c + c2) as usize]);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += int16tosig(x[((j as i32 + offset) * c + ch) as usize]);
            }
        }
    }
}

/// Port of src/opus_encoder.c:downmix_int24. `x` is interleaved 24-bit PCM in `i32`.
pub fn downmix_int24(
    x: &[i32],
    y: &mut [OpusVal32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    let n = subframe as usize;
    let y = &mut y[..n];
    for (j, yj) in y.iter_mut().enumerate() {
        *yj = int24tosig(x[((j as i32 + offset) * c + c1) as usize]);
    }
    if c2 > -1 {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj += int24tosig(x[((j as i32 + offset) * c + c2) as usize]);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += int24tosig(x[((j as i32 + offset) * c + ch) as usize]);
            }
        }
    }
}

/// Port of src/opus_encoder.c:is_digital_silence: true when every sample of `pcm`
/// (`frame_size * channels` values) is within one LSB at `lsb_depth` bits (float build), or
/// exactly zero (fixed-point builds, where `lsb_depth` is unused).
#[must_use]
pub fn is_digital_silence(pcm: &[OpusRes], frame_size: i32, channels: i32, lsb_depth: i32) -> bool {
    // MLP_TRAINING: not ported (training-only instrumentation).
    let sample_max = celt_maxabs_res(&pcm[..(frame_size * channels) as usize]);
    #[cfg(feature = "fixed-point")]
    {
        let _ = lsb_depth;
        sample_max == 0
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        sample_max <= 1.0f32 / (1i32 << lsb_depth) as f32
    }
}

/// Port of src/analysis.c:is_digital_silence32 (fixed-point build): true when every sample of
/// the `opus_val32` buffer is zero.
#[cfg(feature = "fixed-point")]
#[must_use]
fn is_digital_silence32(pcm: &[OpusVal32], frame_size: i32, channels: i32, lsb_depth: i32) -> bool {
    // MLP_TRAINING: not ported (training-only instrumentation).
    let sample_max = celt_maxabs32(&pcm[..(frame_size * channels) as usize]);
    let _ = lsb_depth;
    sample_max == 0
}

/// `is_digital_silence32` is `is_digital_silence` in the float build.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
fn is_digital_silence32(pcm: &[OpusVal32], frame_size: i32, channels: i32, lsb_depth: i32) -> bool {
    is_digital_silence(pcm, frame_size, channels, lsb_depth)
}

/// `QCONST16(0.6074371f, 15)`: first all-pass coefficient of [`silk_resampler_down2_hp`].
#[cfg(feature = "fixed-point")]
const DOWN2_COEF0: OpusVal16 = qconst16(0.6074371f32 as f64, 15);
/// `QCONST16(0.15063f, 15)`: second all-pass coefficient of [`silk_resampler_down2_hp`].
#[cfg(feature = "fixed-point")]
const DOWN2_COEF1: OpusVal16 = qconst16(0.15063f32 as f64, 15);
/// `QCONST16(0.6074371f, 15)` (float build: the value itself).
#[cfg(not(feature = "fixed-point"))]
const DOWN2_COEF0: f32 = qconst16(0.6074371, 15);
/// `QCONST16(0.15063f, 15)` (float build: the value itself).
#[cfg(not(feature = "fixed-point"))]
const DOWN2_COEF1: f32 = qconst16(0.15063, 15);

/// Port of src/analysis.c:silk_resampler_down2_hp: 2x downsampler that also returns the energy
/// of the high-pass (upper half-band) signal. `s` is the 3-value state, `out` receives
/// `in_len/2` samples. In fixed-point builds the energy is accumulated in 64 bits (each term
/// shifted right by 8), then shifted right by `2*SIG_SHIFT` and saturated to 32 bits.
pub fn silk_resampler_down2_hp(
    s: &mut [OpusVal32; 3],
    out: &mut [OpusVal32],
    input: &[OpusVal32],
    in_len: i32,
) -> OpusVal32 {
    let len2 = (in_len / 2) as usize;
    let out = &mut out[..len2];
    let input = &input[..2 * len2];
    #[cfg(not(feature = "fixed-point"))]
    let mut hp_ener: f32 = 0.0;
    #[cfg(feature = "fixed-point")]
    let mut hp_ener: i64 = 0;
    // Internal variables and state are in Q10 format
    for k in 0..len2 {
        // Convert to Q10
        let mut in32 = input[2 * k];

        // All-pass section for even input sample
        let mut y = in32 - s[0];
        let mut x = mult16_32_q15(DOWN2_COEF0, y);
        let mut out32 = s[0] + x;
        s[0] = in32 + x;
        let mut out32_hp = out32;
        // Convert to Q10
        in32 = input[2 * k + 1];

        // All-pass section for odd input sample, and add to output of previous section
        y = in32 - s[1];
        x = mult16_32_q15(DOWN2_COEF1, y);
        out32 += s[1];
        out32 += x;
        s[1] = in32 + x;

        y = -in32 - s[2];
        x = mult16_32_q15(DOWN2_COEF1, y);
        out32_hp += s[2];
        out32_hp += x;
        s[2] = -in32 + x;

        // len2 can be up to 480, so we shift by 8 to make it fit (SHR64 is a no-op in float).
        #[cfg(not(feature = "fixed-point"))]
        {
            hp_ener += out32_hp * out32_hp;
        }
        #[cfg(feature = "fixed-point")]
        {
            hp_ener += shr64(i64::from(out32_hp) * i64::from(out32_hp), 8);
        }
        // Add, convert back to int16 and store to output
        out[k] = half32(out32);
    }
    #[cfg(feature = "fixed-point")]
    let hp_ener = {
        // Fitting in 32 bits.
        hp_ener >>= 2 * SIG_SHIFT;
        if hp_ener > 2147483647 {
            hp_ener = 2147483647;
        }
        hp_ener as OpusVal32
    };
    hp_ener
}

/// Port of src/analysis.c:downmix_and_resample: downmixes `subframe` samples (at the 24 kHz
/// analysis rate) starting at `offset` of `x` with `downmix`, resamples to 24 kHz into `y`,
/// and returns the (scaled) high-band energy at 48 kHz (0 otherwise).
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn downmix_and_resample<T>(
    downmix: DownmixFunc<T>,
    x: &[T],
    y: &mut [OpusVal32],
    s: &mut [OpusVal32; 3],
    mut subframe: i32,
    mut offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
) -> OpusVal32 {
    let mut ret = OpusVal32::default();

    if subframe == 0 {
        return OpusVal32::default();
    }
    if fs == 48000 {
        subframe *= 2;
        offset *= 2;
    } else if fs == 16000 {
        subframe = subframe * 2 / 3;
        offset = offset * 2 / 3;
    } else if fs != 24000 {
        debug_assert!(false, "downmix_and_resample: unsupported Fs {fs}");
    }
    let n = subframe as usize;
    debug_assert!(n <= MAX_DOWNMIX);
    let mut tmp_buf = [OpusVal32::default(); MAX_DOWNMIX];
    let tmp = &mut tmp_buf[..n];

    downmix(x, tmp, subframe, offset, c1, c2, c);
    if (c2 == -2 && c == 2) || c2 > -1 {
        for t in tmp.iter_mut() {
            *t = half32(*t);
        }
    }
    if fs == 48000 {
        ret = silk_resampler_down2_hp(s, y, tmp, subframe);
    } else if fs == 24000 {
        y[..n].copy_from_slice(tmp);
    } else if fs == 16000 {
        let mut tmp3x = [OpusVal32::default(); 3 * MAX_DOWNMIX / 2];
        let tmp3x = &mut tmp3x[..3 * n];
        // Don't do this at home! This resampler is horrible and it's only (barely)
        // usable for the purpose of the analysis because we don't care about all
        // the aliasing between 8 kHz and 12 kHz.
        for (j, &t) in tmp.iter().enumerate() {
            tmp3x[3 * j] = t;
            tmp3x[3 * j + 1] = t;
            tmp3x[3 * j + 2] = t;
        }
        silk_resampler_down2_hp(s, y, tmp3x, 3 * subframe);
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        ret *= 1.0f32 / 32768.0 / 32768.0;
    }
    ret
}

/// Port of src/analysis.c:tonality_analysis_init.
pub fn tonality_analysis_init(tonal: &mut TonalityAnalysisState, fs: i32) {
    // Initialize reusable fields.
    tonal.fs = fs;
    // Clear remaining fields.
    tonality_analysis_reset(tonal);
}

/// Port of src/analysis.c:tonality_analysis_reset: clears every field from `angle` onward.
pub fn tonality_analysis_reset(tonal: &mut TonalityAnalysisState) {
    // Clear non-reusable fields.
    tonal.angle = [0.0; 240];
    tonal.d_angle = [0.0; 240];
    tonal.d2_angle = [0.0; 240];
    tonal.inmem = [OpusVal32::default(); ANALYSIS_BUF_SIZE];
    tonal.mem_fill = 0;
    tonal.prev_band_tonality = [0.0; NB_TBANDS];
    tonal.prev_tonality = 0.0;
    tonal.prev_bandwidth = 0;
    tonal.e = [[0.0; NB_TBANDS]; NB_FRAMES];
    tonal.log_e = [[0.0; NB_TBANDS]; NB_FRAMES];
    tonal.low_e = [0.0; NB_TBANDS];
    tonal.high_e = [0.0; NB_TBANDS];
    tonal.mean_e = [0.0; NB_TBANDS + 1];
    tonal.mem = [0.0; 32];
    tonal.cmean = [0.0; 8];
    tonal.std = [0.0; 9];
    tonal.e_tracker = 0.0;
    tonal.low_e_count = 0.0;
    tonal.e_count = 0;
    tonal.count = 0;
    tonal.analysis_offset = 0;
    tonal.write_pos = 0;
    tonal.read_pos = 0;
    tonal.read_subframe = 0;
    tonal.hp_ener_accum = 0.0;
    tonal.initialized = 0;
    tonal.rnn_state = [0.0; MAX_NEURONS];
    tonal.downmix_state = [OpusVal32::default(); 3];
    tonal.info = [AnalysisInfo::default(); DETECT_SIZE];
}

/// Port of src/analysis.c:tonality_get_info: fills `info_out` with the analysis for the next
/// `len` samples and advances the read position.
pub fn tonality_get_info(tonal: &mut TonalityAnalysisState, info_out: &mut AnalysisInfo, len: i32) {
    const DS: i32 = DETECT_SIZE as i32;
    let mut pos = tonal.read_pos;
    let mut curr_lookahead = tonal.write_pos - tonal.read_pos;
    if curr_lookahead < 0 {
        curr_lookahead += DS;
    }

    tonal.read_subframe += len / (tonal.fs / 400);
    while tonal.read_subframe >= 8 {
        tonal.read_subframe -= 8;
        tonal.read_pos += 1;
    }
    if tonal.read_pos >= DS {
        tonal.read_pos -= DS;
    }

    // On long frames, look at the second analysis window rather than the first.
    if len > tonal.fs / 50 && pos != tonal.write_pos {
        pos += 1;
        if pos == DS {
            pos = 0;
        }
    }
    if pos == tonal.write_pos {
        pos -= 1;
    }
    if pos < 0 {
        pos = DS - 1;
    }
    let pos0 = pos;
    *info_out = tonal.info[pos as usize];
    if info_out.valid == 0 {
        return;
    }
    let mut tonality_max = info_out.tonality;
    let mut tonality_avg = info_out.tonality;
    let mut tonality_count: i32 = 1;
    // Look at the neighbouring frames and pick largest bandwidth found (to be safe).
    let mut bandwidth_span = 6;
    // If possible, look ahead for a tone to compensate for the delay in the tone detector.
    for _ in 0..3 {
        pos += 1;
        if pos == DS {
            pos = 0;
        }
        if pos == tonal.write_pos {
            break;
        }
        let inf = &tonal.info[pos as usize];
        tonality_max = max32(tonality_max, inf.tonality);
        tonality_avg += inf.tonality;
        tonality_count += 1;
        info_out.bandwidth = imax(info_out.bandwidth, inf.bandwidth);
        bandwidth_span -= 1;
    }
    pos = pos0;
    // Look back in time to see if any has a wider bandwidth than the current frame.
    for _ in 0..bandwidth_span {
        pos -= 1;
        if pos < 0 {
            pos = DS - 1;
        }
        if pos == tonal.write_pos {
            break;
        }
        info_out.bandwidth = imax(info_out.bandwidth, tonal.info[pos as usize].bandwidth);
    }
    info_out.tonality = max32(tonality_avg / tonality_count as f32, tonality_max - 0.2);

    let mut mpos = pos0;
    let mut vpos = pos0;
    // If we have enough look-ahead, compensate for the ~5-frame delay in the music prob and
    // ~1 frame delay in the VAD prob.
    if curr_lookahead > 15 {
        mpos += 5;
        if mpos >= DS {
            mpos -= DS;
        }
        vpos += 1;
        if vpos >= DS {
            vpos -= DS;
        }
    }

    // The following calculations attempt to minimize a "badness function" for the
    // transition (see the long comment in src/analysis.c:tonality_get_info). The result is
    // music_prob_min, the threshold for switching to music if we're currently encoding for
    // speech, and music_prob_max, used for switching from music to speech.
    let mut prob_min: f32 = 1.0;
    let mut prob_max: f32 = 0.0;
    let vad_prob = tonal.info[vpos as usize].activity_probability;
    let mut prob_count = max16(0.1, vad_prob);
    let mut prob_avg = max16(0.1, vad_prob) * tonal.info[mpos as usize].music_prob;
    loop {
        mpos += 1;
        if mpos == DS {
            mpos = 0;
        }
        if mpos == tonal.write_pos {
            break;
        }
        vpos += 1;
        if vpos == DS {
            vpos = 0;
        }
        if vpos == tonal.write_pos {
            break;
        }
        let pos_vad = tonal.info[vpos as usize].activity_probability;
        prob_min = min16(
            (prob_avg - TRANSITION_PENALTY as f32 * (vad_prob - pos_vad)) / prob_count,
            prob_min,
        );
        prob_max = max16(
            (prob_avg + TRANSITION_PENALTY as f32 * (vad_prob - pos_vad)) / prob_count,
            prob_max,
        );
        prob_count += max16(0.1, pos_vad);
        prob_avg += max16(0.1, pos_vad) * tonal.info[mpos as usize].music_prob;
    }
    info_out.music_prob = prob_avg / prob_count;
    prob_min = min16(prob_avg / prob_count, prob_min);
    prob_max = max16(prob_avg / prob_count, prob_max);
    prob_min = max16(prob_min, 0.0);
    prob_max = min16(prob_max, 1.0);

    // If we don't have enough look-ahead, do our best to make a decent decision.
    if curr_lookahead < 10 {
        let mut pmin = prob_min;
        let mut pmax = prob_max;
        pos = pos0;
        // Look for min/max in the past.
        for _ in 0..imin(tonal.count - 1, 15) {
            pos -= 1;
            if pos < 0 {
                pos = DS - 1;
            }
            pmin = min16(pmin, tonal.info[pos as usize].music_prob);
            pmax = max16(pmax, tonal.info[pos as usize].music_prob);
        }
        // Bias against switching on active audio.
        pmin = max16(0.0, pmin - 0.1 * vad_prob);
        pmax = min16(1.0, pmax + 0.1 * vad_prob);
        prob_min += (1.0 - 0.1 * curr_lookahead as f32) * (pmin - prob_min);
        prob_max += (1.0 - 0.1 * curr_lookahead as f32) * (pmax - prob_max);
    }
    info_out.music_prob_min = prob_min;
    info_out.music_prob_max = prob_max;
}

/// Port of src/analysis.c:tonality_analysis: consumes `len` samples (at `tonal.fs`) of `x`
/// starting at sample `offset`, and runs one analysis step each time 20 ms (at 24 kHz) have
/// been buffered.
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn tonality_analysis<T>(
    tonal: &mut TonalityAnalysisState,
    celt_mode: &CeltMode,
    x: &[T],
    len: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    lsb_depth: i32,
    downmix: DownmixFunc<T>,
) {
    let mut resample = |y: &mut [OpusVal32], s: &mut [OpusVal32; 3], subframe, offset, fs| {
        downmix_and_resample(downmix, x, y, s, subframe, offset, c1, c2, c, fs)
    };
    tonality_analysis_impl(tonal, celt_mode, len, offset, lsb_depth, &mut resample);
}

/// `downmix_and_resample(downmix, x, y, s, subframe, offset, c1, c2, C, Fs)` with the
/// input-type dependent arguments (`downmix`, `x`, `c1`, `c2`, `C`) bound:
/// `(y, s, subframe, offset, Fs)`.
type DownmixResampleFn<'a> =
    &'a mut dyn FnMut(&mut [OpusVal32], &mut [OpusVal32; 3], i32, i32, i32) -> OpusVal32;

/// Body of [`tonality_analysis`]. Size: only the input downmix depends on the sample type, so
/// it is passed as a type-erased closure instead of monomorphizing the analysis per type.
#[expect(clippy::too_many_lines, reason = "mirrors the C function")]
fn tonality_analysis_impl(
    tonal: &mut TonalityAnalysisState,
    celt_mode: &CeltMode,
    mut len: i32,
    mut offset: i32,
    lsb_depth: i32,
    downmix_and_resample: DownmixResampleFn<'_>,
) {
    const N: usize = 480;
    const N2: usize = 240;
    const BUF: i32 = ANALYSIS_BUF_SIZE as i32;
    let pi4: f32 = (PI * PI * PI * PI) as f32;
    let mut band_tonality = [0.0f32; NB_TBANDS];
    let mut log_e = [0.0f32; NB_TBANDS];
    let mut bfcc = [0.0f32; 8];
    let mut features = [0.0f32; 25];
    let mut slope: f32 = 0.0;
    let mut frame_probs = [0.0f32; 2];
    let mut is_masked = [false; NB_TBANDS + 1];
    let mut tonality2 = [0.0f32; 240];
    let mut mid_e = [0.0f32; 8];
    let mut spec_variability: f32 = 0.0;
    let mut band_log2 = [0.0f32; NB_TBANDS + 1];
    let mut leakage_from = [0.0f32; NB_TBANDS + 1];
    let mut leakage_to = [0.0f32; NB_TBANDS + 1];
    let mut layer_out = [0.0f32; MAX_NEURONS];

    if tonal.initialized == 0 {
        tonal.mem_fill = 240;
        tonal.initialized = 1;
    }
    let alpha = 1.0f32 / imin(10, 1 + tonal.count) as f32;
    let alpha_e = 1.0f32 / imin(25, 1 + tonal.count) as f32;
    // Noise floor related decay for bandwidth detection: -2.2 dB/second
    let mut alpha_e2 = 1.0f32 / imin(100, 1 + tonal.count) as f32;
    if tonal.count <= 1 {
        alpha_e2 = 1.0;
    }

    if tonal.fs == 48000 {
        // len and offset are now at 24 kHz.
        len /= 2;
        offset /= 2;
    } else if tonal.fs == 16000 {
        len = 3 * len / 2;
        offset = 3 * offset / 2;
    }

    let kfft = &*celt_mode.mdct.kfft[0];
    {
        let mf = tonal.mem_fill as usize;
        tonal.hp_ener_accum += to_float(downmix_and_resample(
            &mut tonal.inmem[mf..],
            &mut tonal.downmix_state,
            imin(len, BUF - tonal.mem_fill),
            offset,
            tonal.fs,
        ));
    }
    if tonal.mem_fill + len < BUF {
        tonal.mem_fill += len;
        // Don't have enough to update the analysis
        return;
    }
    let hp_ener = tonal.hp_ener_accum;
    let info_idx = tonal.write_pos as usize;
    tonal.write_pos += 1;
    if tonal.write_pos >= DETECT_SIZE as i32 {
        tonal.write_pos -= DETECT_SIZE as i32;
    }

    let is_silence = is_digital_silence32(&tonal.inmem, BUF, 1, lsb_depth);

    let mut input = [KissFftCpx::default(); N];
    let mut out = [KissFftCpx::default(); N];
    let mut tonality = [0.0f32; 240];
    let mut noisiness = [0.0f32; 240];
    for i in 0..N2 {
        let w = ANALYSIS_WINDOW[i];
        input[i].r = windowed(w, tonal.inmem[i]);
        input[i].i = windowed(w, tonal.inmem[N2 + i]);
        input[N - i - 1].r = windowed(w, tonal.inmem[N - i - 1]);
        input[N - i - 1].i = windowed(w, tonal.inmem[N + N2 - i - 1]);
    }
    tonal.inmem.copy_within(ANALYSIS_BUF_SIZE - 240.., 0);
    let remaining = len - (BUF - tonal.mem_fill);
    tonal.hp_ener_accum = to_float(downmix_and_resample(
        &mut tonal.inmem[240..],
        &mut tonal.downmix_state,
        remaining,
        offset + BUF - tonal.mem_fill,
        tonal.fs,
    ));
    tonal.mem_fill = 240 + remaining;
    if is_silence {
        // On silence, copy the previous analysis.
        let mut prev_pos = tonal.write_pos - 2;
        if prev_pos < 0 {
            prev_pos += DETECT_SIZE as i32;
        }
        tonal.info[info_idx] = tonal.info[prev_pos as usize];
        return;
    }
    opus_fft(kfft, &input, &mut out);
    // If there's any NaN on the input, the entire output will be NaN, so we only need to check
    // one value (float build only).
    #[cfg(not(feature = "fixed-point"))]
    if celt_isnan(out[0].r) {
        tonal.info[info_idx].valid = 0;
        return;
    }

    let a = &mut tonal.angle;
    let da = &mut tonal.d_angle;
    let d2a = &mut tonal.d2_angle;
    let half_over_pi = (0.5f64 / PI) as f32;
    for i in 1..N2 {
        let x1r = to_float(out[i].r) + to_float(out[N - i].r);
        let x1i = to_float(out[i].i) - to_float(out[N - i].i);
        let x2r = to_float(out[i].i) + to_float(out[N - i].i);
        let x2i = to_float(out[N - i].r) - to_float(out[i].r);

        let angle = half_over_pi * fast_atan2f(x1i, x1r);
        let d_angle = angle - a[i];
        let d2_angle = d_angle - da[i];

        let angle2 = half_over_pi * fast_atan2f(x2i, x2r);
        let d_angle2 = angle2 - angle;
        let d2_angle2 = d_angle2 - d_angle;

        let mut mod1 = d2_angle - float2int(d2_angle) as f32;
        noisiness[i] = abs16(mod1);
        mod1 *= mod1;
        mod1 *= mod1;

        let mut mod2 = d2_angle2 - float2int(d2_angle2) as f32;
        noisiness[i] += abs16(mod2);
        mod2 *= mod2;
        mod2 *= mod2;

        let avg_mod = 0.25f32 * (d2a[i] + mod1 + 2.0 * mod2);
        // This introduces an extra delay of 2 frames in the detection.
        tonality[i] = 1.0f32 / (1.0 + 40.0f32 * 16.0 * pi4 * avg_mod) - 0.015;
        // No delay on this detection, but it's less reliable.
        tonality2[i] = 1.0f32 / (1.0 + 40.0f32 * 16.0 * pi4 * mod2) - 0.015;

        a[i] = angle2;
        da[i] = d_angle2;
        d2a[i] = mod2;
    }
    for i in 2..N2 - 1 {
        let tt = min32(tonality2[i], max32(tonality2[i - 1], tonality2[i + 1]));
        tonality[i] = 0.9f32 * max32(tonality[i], tt - 0.1);
    }
    let mut frame_tonality: f32 = 0.0;
    let mut max_frame_tonality: f32 = 0.0;
    tonal.info[info_idx].activity = 0.0;
    let mut frame_noisiness: f32 = 0.0;
    let mut frame_stationarity: f32 = 0.0;
    if tonal.count == 0 {
        for b in 0..NB_TBANDS {
            tonal.low_e[b] = 1e10;
            tonal.high_e[b] = -1e10;
        }
    }
    let mut relative_e: f32 = 0.0;
    let mut frame_loudness: f32 = 0.0;
    let bin_e = |i: usize| -> f32 {
        let (xr, xi) = (to_float(out[i].r), to_float(out[i].i));
        let (yr, yi) = (to_float(out[N - i].r), to_float(out[N - i].i));
        xr * xr + yr * yr + xi * xi + yi * yi
    };
    // The energy of the very first band is special because of DC.
    {
        let x1r = 2.0f32 * to_float(out[0].r);
        let x2r = 2.0f32 * to_float(out[0].i);
        let mut e = x1r * x1r + x2r * x2r;
        for i in 1..4 {
            e += bin_e(i);
        }
        e = scale_ener(e);
        band_log2[0] = 0.5f32 * 1.442695 * math::log((e + 1e-10) as f64) as f32;
    }
    for b in 0..NB_TBANDS {
        let mut e: f32 = 0.0;
        let mut t_e: f32 = 0.0;
        let mut n_e: f32 = 0.0;
        for i in TBANDS[b] as usize..TBANDS[b + 1] as usize {
            let be = scale_ener(bin_e(i));
            e += be;
            t_e += be * max32(0.0, tonality[i]);
            n_e += be * 2.0 * (0.5 - noisiness[i]);
        }
        // Check for extreme band energies that could cause NaNs later (float build only).
        #[cfg(not(feature = "fixed-point"))]
        {
            let below_limit = e < 1e9;
            if !below_limit || celt_isnan(e) {
                tonal.info[info_idx].valid = 0;
                return;
            }
        }

        let ec = tonal.e_count as usize;
        tonal.e[ec][b] = e;
        frame_noisiness += n_e / (1e-15 + e);

        frame_loudness += math::sqrt((e + 1e-10) as f64) as f32;
        log_e[b] = math::log((e + 1e-10) as f64) as f32;
        band_log2[b + 1] = 0.5f32 * 1.442695 * math::log((e + 1e-10) as f64) as f32;
        tonal.log_e[ec][b] = log_e[b];
        if tonal.count == 0 {
            tonal.low_e[b] = log_e[b];
            tonal.high_e[b] = log_e[b];
        }
        if tonal.high_e[b] as f64 > tonal.low_e[b] as f64 + 7.5 {
            if tonal.high_e[b] - log_e[b] > log_e[b] - tonal.low_e[b] {
                tonal.high_e[b] -= 0.01;
            } else {
                tonal.low_e[b] += 0.01;
            }
        }
        if log_e[b] > tonal.high_e[b] {
            tonal.high_e[b] = log_e[b];
            tonal.low_e[b] = max32(tonal.high_e[b] - 15.0, tonal.low_e[b]);
        } else if log_e[b] < tonal.low_e[b] {
            tonal.low_e[b] = log_e[b];
            tonal.high_e[b] = min32(tonal.low_e[b] + 15.0, tonal.high_e[b]);
        }
        relative_e += (log_e[b] - tonal.low_e[b]) / (1e-5 + (tonal.high_e[b] - tonal.low_e[b]));

        let mut l1: f32 = 0.0;
        let mut l2: f32 = 0.0;
        for i in 0..NB_FRAMES {
            l1 += math::sqrt(tonal.e[i][b] as f64) as f32;
            l2 += tonal.e[i][b];
        }

        let mut stationarity = min16(
            0.99,
            l1 / math::sqrt(1e-15 + (NB_FRAMES as f32 * l2) as f64) as f32,
        );
        stationarity *= stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;
        band_tonality[b] = max16(
            t_e / (1e-15 + e),
            stationarity * tonal.prev_band_tonality[b],
        );
        frame_tonality += band_tonality[b];
        if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
            frame_tonality -= band_tonality[b + NB_TONAL_SKIP_BANDS - NB_TBANDS];
        }
        max_frame_tonality = max16(
            max_frame_tonality,
            (1.0f32 + 0.03 * (b as i32 - NB_TBANDS as i32) as f32) * frame_tonality,
        );
        slope += band_tonality[b] * (b as i32 - 8) as f32;
        tonal.prev_band_tonality[b] = band_tonality[b];
    }

    leakage_from[0] = band_log2[0];
    leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
    for b in 1..NB_TBANDS + 1 {
        let leak_slope = LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0;
        leakage_from[b] = min16(leakage_from[b - 1] + leak_slope, band_log2[b]);
        leakage_to[b] = max16(
            leakage_to[b - 1] - leak_slope,
            band_log2[b] - LEAKAGE_OFFSET,
        );
    }
    for b in (0..=NB_TBANDS - 2).rev() {
        let leak_slope = LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0;
        leakage_from[b] = min16(leakage_from[b + 1] + leak_slope, leakage_from[b]);
        leakage_to[b] = max16(leakage_to[b + 1] - leak_slope, leakage_to[b]);
    }
    const _: () = assert!(NB_TBANDS < LEAK_BANDS);
    for b in 0..NB_TBANDS + 1 {
        // leak_boost[] is made up of two terms. The first, based on leakage_to[],
        // represents the boost needed to overcome the amount of analysis leakage
        // cause in a weaker band b by louder neighbouring bands.
        // The second, based on leakage_from[], applies to a loud band b for
        // which the quantization noise causes synthesis leakage to the weaker
        // neighbouring bands.
        let boost = max16(0.0, leakage_to[b] - band_log2[b])
            + max16(0.0, band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET));
        tonal.info[info_idx].leak_boost[b] =
            imin(255, math::floor(0.5 + (64.0f32 * boost) as f64) as i32) as u8;
    }
    // C clears leak_boost[NB_TBANDS+1..LEAK_BANDS]; that range is empty (LEAK_BANDS == 19).

    for i in 0..NB_FRAMES {
        let mut mindist: f32 = 1e15;
        for j in 0..NB_FRAMES {
            let mut dist: f32 = 0.0;
            for k in 0..NB_TBANDS {
                let tmp = tonal.log_e[i][k] - tonal.log_e[j][k];
                dist += tmp * tmp;
            }
            if j != i {
                mindist = min32(mindist, dist);
            }
        }
        spec_variability += mindist;
    }
    spec_variability =
        math::sqrt((spec_variability / NB_FRAMES as f32 / NB_TBANDS as f32) as f64) as f32;
    let mut bandwidth_mask: f32 = 0.0;
    let mut bandwidth: i32 = 0;
    let mut max_e: f32 = 0.0;
    let mut noise_floor = 5.7e-4f32 / (1i32 << imax(0, lsb_depth - 8)) as f32;
    noise_floor *= noise_floor;
    let mut below_max_pitch: f32 = 0.0;
    let mut above_max_pitch: f32 = 0.0;
    for b in 0..NB_TBANDS {
        let mut e: f32 = 0.0;
        // Keep a margin of 300 Hz for aliasing
        let band_start = TBANDS[b];
        let band_end = TBANDS[b + 1];
        for i in band_start as usize..band_end as usize {
            e += bin_e(i);
        }
        e = scale_ener(e);
        max_e = max32(max_e, e);
        if band_start < 64 {
            below_max_pitch += e;
        } else {
            above_max_pitch += e;
        }
        tonal.mean_e[b] = max32((1.0 - alpha_e2) * tonal.mean_e[b], e);
        let em = max32(e, tonal.mean_e[b]);
        // Consider the band "active" only if all these conditions are met:
        // 1) less than 90 dB below the peak band (maximal masking possible considering
        //    both the ATH and the loudness-dependent slope of the spreading function)
        // 2) above the PCM quantization noise floor
        // We use b+1 because the first CELT band isn't included in tbands[]
        if e * 1e9 > max_e
            && (em > 3.0 * noise_floor * (band_end - band_start) as f32
                || e > noise_floor * (band_end - band_start) as f32)
        {
            bandwidth = b as i32 + 1;
        }
        // Check if the band is masked (see below).
        is_masked[b] = e
            < (if tonal.prev_bandwidth > b as i32 {
                0.01f32
            } else {
                0.05
            }) * bandwidth_mask;
        // Use a simple follower with 13 dB/Bark slope for spreading function.
        bandwidth_mask = max32(0.05 * bandwidth_mask, e);
    }
    // Special case for the last two bands, for which we don't have spectrum but only
    // the energy above 12 kHz. The difficulty here is that the high-pass we use
    // leaks some LF energy, so we need to increase the threshold without accidentally cutting
    // off the band.
    if tonal.fs == 48000 {
        let b = NB_TBANDS;
        #[cfg(not(feature = "fixed-point"))]
        let e = hp_ener * (1.0f32 / (60 * 60) as f32);
        #[cfg(feature = "fixed-point")]
        let mut e = hp_ener * (1.0f32 / (60 * 60) as f32);
        let noise_ratio: f32 = if tonal.prev_bandwidth == 20 {
            10.0
        } else {
            30.0
        };

        #[cfg(feature = "fixed-point")]
        {
            // silk_resampler_down2_hp() shifted right by an extra 8 bits.
            e *= 256.0f32 * (1.0f32 / 32767.0) * (1.0f32 / 32767.0);
        }
        above_max_pitch += e;
        tonal.mean_e[b] = max32((1.0 - alpha_e2) * tonal.mean_e[b], e);
        let em = max32(e, tonal.mean_e[b]);
        if em > 3.0 * noise_ratio * noise_floor * 160.0 || e > noise_ratio * noise_floor * 160.0 {
            bandwidth = 20;
        }
        // Check if the band is masked (see below).
        is_masked[b] = e
            < (if tonal.prev_bandwidth == 20 {
                0.01f32
            } else {
                0.05
            }) * bandwidth_mask;
    }
    tonal.info[info_idx].max_pitch_ratio = if above_max_pitch > below_max_pitch {
        below_max_pitch / above_max_pitch
    } else {
        1.0
    };
    // In some cases, resampling aliasing can create a small amount of energy in the first band
    // being cut. So if the last band is masked, we don't include it.
    if bandwidth == 20 && is_masked[NB_TBANDS] {
        bandwidth -= 2;
    } else if bandwidth > 0 && bandwidth <= NB_TBANDS as i32 && is_masked[(bandwidth - 1) as usize]
    {
        bandwidth -= 1;
    }
    if tonal.count <= 2 {
        bandwidth = 20;
    }
    frame_loudness = 20.0f32 * math::log10(frame_loudness as f64) as f32;
    tonal.e_tracker = max32(tonal.e_tracker - 0.003, frame_loudness);
    tonal.low_e_count *= 1.0 - alpha_e;
    if frame_loudness < tonal.e_tracker - 30.0 {
        tonal.low_e_count += alpha_e;
    }

    for i in 0..8 {
        let mut sum: f32 = 0.0;
        for b in 0..16 {
            sum += DCT_TABLE[i * 16 + b] * log_e[b];
        }
        bfcc[i] = sum;
    }
    for i in 0..8 {
        let mut sum: f32 = 0.0;
        for b in 0..16 {
            sum += DCT_TABLE[i * 16 + b] * 0.5 * (tonal.high_e[b] + tonal.low_e[b]);
        }
        mid_e[i] = sum;
    }

    frame_stationarity /= NB_TBANDS as f32;
    relative_e /= NB_TBANDS as f32;
    if tonal.count < 10 {
        relative_e = 0.5;
    }
    frame_noisiness /= NB_TBANDS as f32;
    tonal.info[info_idx].activity = frame_noisiness + (1.0 - frame_noisiness) * relative_e;
    frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
    frame_tonality = max16(frame_tonality, tonal.prev_tonality * 0.8);
    tonal.prev_tonality = frame_tonality;

    slope /= (8 * 8) as f32;
    tonal.info[info_idx].tonality_slope = slope;

    tonal.e_count = (tonal.e_count + 1) % NB_FRAMES as i32;
    tonal.count = imin(tonal.count + 1, ANALYSIS_COUNT_MAX);
    tonal.info[info_idx].tonality = frame_tonality;

    let mem = &mut tonal.mem;
    for i in 0..4 {
        features[i] = -0.12299f32 * (bfcc[i] + mem[i + 24])
            + 0.49195f32 * (mem[i] + mem[i + 16])
            + 0.69693f32 * mem[i + 8]
            - 1.4349f32 * tonal.cmean[i];
    }

    for i in 0..4 {
        tonal.cmean[i] = (1.0 - alpha) * tonal.cmean[i] + alpha * bfcc[i];
    }

    for i in 0..4 {
        features[4 + i] =
            0.63246f32 * (bfcc[i] - mem[i + 24]) + 0.31623f32 * (mem[i] - mem[i + 16]);
    }
    for i in 0..3 {
        features[8 + i] = 0.53452f32 * (bfcc[i] + mem[i + 24])
            - 0.26726f32 * (mem[i] + mem[i + 16])
            - 0.53452f32 * mem[i + 8];
    }

    if tonal.count > 5 {
        for i in 0..9 {
            tonal.std[i] = (1.0 - alpha) * tonal.std[i] + alpha * features[i] * features[i];
        }
    }
    for i in 0..4 {
        features[i] = bfcc[i] - mid_e[i];
    }

    for i in 0..8 {
        mem[i + 24] = mem[i + 16];
        mem[i + 16] = mem[i + 8];
        mem[i + 8] = mem[i];
        mem[i] = bfcc[i];
    }
    for i in 0..9 {
        features[11 + i] = math::sqrt(tonal.std[i] as f64) as f32 - STD_FEATURE_BIAS[i];
    }
    let info = &mut tonal.info[info_idx];
    features[18] = spec_variability - 0.78;
    features[20] = info.tonality - 0.154723;
    features[21] = info.activity - 0.724643;
    features[22] = frame_stationarity - 0.743717;
    features[23] = info.tonality_slope + 0.069216;
    features[24] = tonal.low_e_count - 0.067930;

    analysis_compute_dense(&LAYER0, &mut layer_out, &features);
    analysis_compute_gru(&LAYER1, &mut tonal.rnn_state, &layer_out);
    analysis_compute_dense(&LAYER2, &mut frame_probs, &tonal.rnn_state);

    // Probability of speech or music vs noise
    info.activity_probability = frame_probs[1];
    info.music_prob = frame_probs[0];

    // MLP_TRAINING: not ported (training-only feature dump).

    info.bandwidth = bandwidth;
    tonal.prev_bandwidth = bandwidth;
    info.noisiness = frame_noisiness;
    info.valid = 1;
}

/// Port of src/analysis.c:run_analysis: analyses the new samples of `analysis_pcm` (if any)
/// and returns in `analysis_info` the analysis for the `frame_size` samples being encoded.
///
/// `celt_mode` must be the 48 kHz / 960 static mode (its 480-point FFT is used).
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn run_analysis<T>(
    analysis: &mut TonalityAnalysisState,
    celt_mode: &CeltMode,
    analysis_pcm: Option<&[T]>,
    mut analysis_frame_size: i32,
    frame_size: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
    lsb_depth: i32,
    downmix: DownmixFunc<T>,
    analysis_info: &mut AnalysisInfo,
) {
    analysis_frame_size -= analysis_frame_size & 1;
    if let Some(pcm) = analysis_pcm {
        // Avoid overflow/wrap-around of the analysis buffer
        analysis_frame_size = imin((DETECT_SIZE as i32 - 5) * fs / 50, analysis_frame_size);

        let mut pcm_len = analysis_frame_size - analysis.analysis_offset;
        let mut offset = analysis.analysis_offset;
        while pcm_len > 0 {
            tonality_analysis(
                analysis,
                celt_mode,
                pcm,
                imin(fs / 50, pcm_len),
                offset,
                c1,
                c2,
                c,
                lsb_depth,
                downmix,
            );
            offset += fs / 50;
            pcm_len -= fs / 50;
        }
        analysis.analysis_offset = analysis_frame_size;

        analysis.analysis_offset -= frame_size;
    }

    tonality_get_info(analysis, analysis_info, frame_size);
}

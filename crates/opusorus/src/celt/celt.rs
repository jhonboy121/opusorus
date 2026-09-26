//! Port of celt/celt.c, celt/celt.h: definitions shared by the CELT encoder and decoder (ICDF
//! tables, CTL request numbers, analysis/SILK side-information structs), the pitch pre/post
//! comb filter, `init_caps`, `resampling_factor`, the TF select table and the library-wide
//! `opus_strerror` / `opus_get_version_string`.
//!
//! The `ARG_QEXT`, `QEXT_SCALE` and `OPUS_CUSTOM_NOSTATIC` macros have no port: QEXT arguments
//! are `#[cfg(feature = "qext")]` parameters, and `QEXT_SCALE(x)` is written inline by its
//! users (it multiplies by a local `qext_scale`). `celt_preemphasis`, `deemphasis`,
//! `celt_synthesis` and `validate_celt_decoder` are declared in `celt.h` but defined in the
//! encoder/decoder sources and are ported there.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::celt::arch::{
    COEF_ONE, CeltCoef, OpusVal16, OpusVal32, add32, imax, mult_coef, mult_coef_32, mult_coef_taps,
    qconst16, saturate,
};
#[cfg(feature = "fixed-point")]
use crate::celt::arch::{SIG_SAT, sub32};
use crate::celt::static_modes::CeltMode;

/// `QEXT_EXTENSION_ID`: padding extension ID carrying the QEXT payload.
pub const QEXT_EXTENSION_ID: i32 = 124;

/// `LEAK_BANDS`.
pub const LEAK_BANDS: usize = 19;

/// Port of celt/celt.h:AnalysisInfo: tonality analysis results passed from the Opus encoder to
/// CELT.
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

/// Port of celt/celt.h:SILKInfo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkInfo {
    pub signal_type: i32,
    pub offset: i32,
}

// Encoder/decoder requests.

/// `CELT_SET_PREDICTION_REQUEST`: controls the use of interframe prediction (0 = independent
/// frames, 1 = short term interframe prediction allowed, 2 = long term prediction allowed).
pub const CELT_SET_PREDICTION_REQUEST: i32 = 10002;
/// `CELT_SET_INPUT_CLIPPING_REQUEST`.
pub const CELT_SET_INPUT_CLIPPING_REQUEST: i32 = 10004;
/// `CELT_GET_AND_CLEAR_ERROR_REQUEST`.
pub const CELT_GET_AND_CLEAR_ERROR_REQUEST: i32 = 10007;
/// `CELT_SET_CHANNELS_REQUEST`.
pub const CELT_SET_CHANNELS_REQUEST: i32 = 10008;
/// `CELT_SET_START_BAND_REQUEST` (internal).
pub const CELT_SET_START_BAND_REQUEST: i32 = 10010;
/// `CELT_SET_END_BAND_REQUEST` (internal).
pub const CELT_SET_END_BAND_REQUEST: i32 = 10012;
/// `CELT_GET_MODE_REQUEST`: get the `CELTMode` used by an encoder or decoder.
pub const CELT_GET_MODE_REQUEST: i32 = 10015;
/// `CELT_SET_SIGNALLING_REQUEST`.
pub const CELT_SET_SIGNALLING_REQUEST: i32 = 10016;
/// `CELT_SET_TONALITY_REQUEST`.
pub const CELT_SET_TONALITY_REQUEST: i32 = 10018;
/// `CELT_SET_TONALITY_SLOPE_REQUEST`.
pub const CELT_SET_TONALITY_SLOPE_REQUEST: i32 = 10020;
/// `CELT_SET_ANALYSIS_REQUEST`.
pub const CELT_SET_ANALYSIS_REQUEST: i32 = 10022;
/// `OPUS_SET_LFE_REQUEST`.
pub const OPUS_SET_LFE_REQUEST: i32 = 10024;
/// `OPUS_SET_ENERGY_MASK_REQUEST`.
pub const OPUS_SET_ENERGY_MASK_REQUEST: i32 = 10026;
/// `CELT_SET_SILK_INFO_REQUEST`.
pub const CELT_SET_SILK_INFO_REQUEST: i32 = 10028;

/// Port of celt/celt.h:bits_to_bitrate.
#[inline]
#[must_use]
pub const fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    bits * (6 * fs / frame_size) / 6
}

/// Port of celt/celt.h:bitrate_to_bits.
#[inline]
#[must_use]
pub const fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    bitrate * 6 / (6 * fs / frame_size)
}

/// `trim_icdf`.
pub static TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
/// `spread_icdf`. Probs: NONE: 21.875%, LIGHT: 6.25%, NORMAL: 65.625%, AGGRESSIVE: 6.25%.
pub static SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
/// `tapset_icdf`.
pub static TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// `toOpusTable` (custom modes / Opus custom API).
#[cfg(feature = "custom-modes")]
pub static TO_OPUS_TABLE: [u8; 20] = [
    0xE0, 0xE8, 0xF0, 0xF8, 0xC0, 0xC8, 0xD0, 0xD8, 0xA0, 0xA8, 0xB0, 0xB8, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x88, 0x90, 0x98,
];

/// `fromOpusTable` (custom modes / Opus custom API).
#[cfg(feature = "custom-modes")]
pub static FROM_OPUS_TABLE: [u8; 16] = [
    0x80, 0x88, 0x90, 0x98, 0x40, 0x48, 0x50, 0x58, 0x20, 0x28, 0x30, 0x38, 0x00, 0x08, 0x10, 0x18,
];

/// Port of celt/celt.h:toOpus: converts a CELT TOC byte to an Opus TOC byte (-1 if invalid).
#[cfg(feature = "custom-modes")]
#[must_use]
pub const fn to_opus(c: u8) -> i32 {
    let mut ret: i32 = 0;
    if c < 0xA0 {
        ret = TO_OPUS_TABLE[(c >> 3) as usize] as i32;
    }
    if ret == 0 { -1 } else { ret | (c & 0x7) as i32 }
}

/// Port of celt/celt.h:fromOpus: converts an Opus TOC byte to a CELT TOC byte (-1 if invalid).
#[cfg(feature = "custom-modes")]
#[must_use]
pub const fn from_opus(c: u8) -> i32 {
    if c < 0x80 {
        -1
    } else {
        FROM_OPUS_TABLE[((c >> 3) - 16) as usize] as i32 | (c & 0x7) as i32
    }
}

/// `COMBFILTER_MAXPERIOD`.
pub const COMBFILTER_MAXPERIOD: i32 = 1024;
/// `COMBFILTER_MINPERIOD`.
pub const COMBFILTER_MINPERIOD: i32 = 15;

/// Port of celt/celt.c:resampling_factor. Returns 0 for unsupported rates (a `celt_assert`
/// failure in C builds without custom modes).
#[must_use]
pub const fn resampling_factor(rate: i32) -> i32 {
    match rate {
        #[cfg(feature = "qext")]
        96000 => 1,
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => {
            #[cfg(not(feature = "custom-modes"))]
            celt_assert!(false, "resampling_factor: unsupported rate");
            0
        }
    }
}

/// `SIG_SAT` (celt/arch.h): the saturation is a no-op in the float build.
#[cfg(not(feature = "fixed-point"))]
const SIG_SAT: i32 = 0;

/// A `celt_coef` from the `int` result of a `MULT_COEF*` macro (implicit C conversion: the Q15
/// `celt_coef` of the fixed-point build without QEXT truncates it to 16 bits).
#[cfg(all(feature = "fixed-point", not(feature = "qext")))]
#[inline(always)]
const fn to_coef(x: i32) -> CeltCoef {
    x as i16
}
/// A `celt_coef` from the result of a `MULT_COEF*` macro (same type).
#[cfg(any(not(feature = "fixed-point"), feature = "qext"))]
#[inline(always)]
const fn to_coef(x: CeltCoef) -> CeltCoef {
    x
}

/// Memory access for the comb filter: either separate input (`x`, with history before its
/// origin) and output (`y`) buffers, or one buffer filtered in place (C `x == y`), where
/// already-filtered output samples are read back as input.
trait CombIo {
    /// `x[i]`, where `i` may be negative (history).
    fn x(&self, i: isize) -> OpusVal32;
    /// `y[i] = v`.
    fn set_y(&mut self, i: usize, v: OpusVal32);
    /// `OPUS_MOVE(y+from, x+from, n)` when `x != y` (a no-op in place).
    fn copy_x_to_y(&mut self, from: usize, n: usize);
}

/// Separate `y` and `x` buffers; `x[0]` is `xs[x_off]`.
struct CombSeparate<'a> {
    y: &'a mut [OpusVal32],
    xs: &'a [OpusVal32],
    x_off: usize,
}

impl CombIo for CombSeparate<'_> {
    #[inline(always)]
    fn x(&self, i: isize) -> OpusVal32 {
        self.xs[self.x_off.wrapping_add_signed(i)]
    }
    #[inline(always)]
    fn set_y(&mut self, i: usize, v: OpusVal32) {
        self.y[i] = v;
    }
    #[inline(always)]
    fn copy_x_to_y(&mut self, from: usize, n: usize) {
        self.y[from..from + n].copy_from_slice(&self.xs[self.x_off + from..self.x_off + from + n]);
    }
}

/// In-place filtering; `x[0] = y[0]` is `buf[off]`.
struct CombInPlace<'a> {
    buf: &'a mut [OpusVal32],
    off: usize,
}

impl CombIo for CombInPlace<'_> {
    #[inline(always)]
    fn x(&self, i: isize) -> OpusVal32 {
        self.buf[self.off.wrapping_add_signed(i)]
    }
    #[inline(always)]
    fn set_y(&mut self, i: usize, v: OpusVal32) {
        self.buf[self.off + i] = v;
    }
    #[inline(always)]
    fn copy_x_to_y(&mut self, _from: usize, _n: usize) {}
}

/// Port of celt/celt.c:comb_filter_const_c, starting at output index `i0`.
#[inline]
fn comb_filter_const_c<B: CombIo>(
    io: &mut B,
    i0: usize,
    t: i32,
    n: usize,
    g10: CeltCoef,
    g11: CeltCoef,
    g12: CeltCoef,
) {
    let t = t as isize;
    let b = i0 as isize;
    let mut x4 = io.x(b - t - 2);
    let mut x3 = io.x(b - t - 1);
    let mut x2 = io.x(b - t);
    let mut x1 = io.x(b - t + 1);
    for i in 0..n {
        let ii = b + i as isize;
        let x0 = io.x(ii - t + 2);
        let mut y = io.x(ii)
            + mult_coef_32(g10, x2)
            + mult_coef_32(g11, add32(x1, x3))
            + mult_coef_32(g12, add32(x0, x4));
        // A bit of bias seems to help here.
        #[cfg(feature = "fixed-point")]
        {
            y = sub32(y, 1);
        }
        y = saturate(y, SIG_SAT);
        io.set_y(i0 + i, y);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// `gains` of celt/celt.c:comb_filter.
static COMB_GAINS: [[OpusVal16; 3]; 3] = [
    [
        qconst16(0.3066406250, 15),
        qconst16(0.2170410156, 15),
        qconst16(0.1296386719, 15),
    ],
    [
        qconst16(0.4638671875, 15),
        qconst16(0.2680664062, 15),
        qconst16(0.0, 15),
    ],
    [
        qconst16(0.7998046875, 15),
        qconst16(0.1000976562, 15),
        qconst16(0.0, 15),
    ],
];

/// Port of celt/celt.c:comb_filter (without the QEXT dispatch) over a [`CombIo`].
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
fn comb_filter_io<B: CombIo>(
    io: &mut B,
    mut t0: i32,
    mut t1: i32,
    n: i32,
    g0: OpusVal16,
    g1: OpusVal16,
    tapset0: i32,
    tapset1: i32,
    window: &[CeltCoef],
    mut overlap: i32,
) {
    let nu = n as usize;
    if g0 == OpusVal16::default() && g1 == OpusVal16::default() {
        // OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder
        // already copies x to y
        io.copy_x_to_y(0, nu);
        return;
    }
    // When the gain is zero, T0 and/or T1 is set to zero. We need to have then be at least 2
    // to avoid processing garbage data.
    t0 = imax(t0, COMBFILTER_MINPERIOD);
    t1 = imax(t1, COMBFILTER_MINPERIOD);
    let gt0 = &COMB_GAINS[tapset0 as usize];
    let gt1 = &COMB_GAINS[tapset1 as usize];
    let g00: CeltCoef = to_coef(mult_coef_taps(g0, gt0[0]));
    let g01: CeltCoef = to_coef(mult_coef_taps(g0, gt0[1]));
    let g02: CeltCoef = to_coef(mult_coef_taps(g0, gt0[2]));
    let g10: CeltCoef = to_coef(mult_coef_taps(g1, gt1[0]));
    let g11: CeltCoef = to_coef(mult_coef_taps(g1, gt1[1]));
    let g12: CeltCoef = to_coef(mult_coef_taps(g1, gt1[2]));
    let (t0i, t1i) = (t0 as isize, t1 as isize);
    let mut x1 = io.x(-t1i + 1);
    let mut x2 = io.x(-t1i);
    let mut x3 = io.x(-t1i - 1);
    let mut x4 = io.x(-t1i - 2);
    // If the filter didn't change, we don't need the overlap
    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        overlap = 0;
    }
    let ov = overlap as usize;
    for i in 0..ov {
        let ii = i as isize;
        let x0 = io.x(ii - t1i + 2);
        let f: CeltCoef = to_coef(mult_coef(window[i], window[i]));
        let mut y = io.x(ii)
            + mult_coef_32(mult_coef(COEF_ONE - f, g00), io.x(ii - t0i))
            + mult_coef_32(
                mult_coef(COEF_ONE - f, g01),
                add32(io.x(ii - t0i + 1), io.x(ii - t0i - 1)),
            )
            + mult_coef_32(
                mult_coef(COEF_ONE - f, g02),
                add32(io.x(ii - t0i + 2), io.x(ii - t0i - 2)),
            )
            + mult_coef_32(mult_coef(f, g10), x2)
            + mult_coef_32(mult_coef(f, g11), add32(x1, x3))
            + mult_coef_32(mult_coef(f, g12), add32(x0, x4));
        // A bit of bias seems to help here.
        #[cfg(feature = "fixed-point")]
        {
            y = sub32(y, 3);
        }
        y = saturate(y, SIG_SAT);
        io.set_y(i, y);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
    if g1 == OpusVal16::default() {
        // OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder
        // already copies x to y
        io.copy_x_to_y(ov, nu - ov);
        return;
    }

    // Compute the part with the constant filter.
    comb_filter_const_c(io, ov, t1, nu - ov, g10, g11, g12);
}

/// Port of celt/celt.c:comb_filter_qext (QEXT, `overlap == 240`): at 96 kHz the period and
/// the spacing between taps are doubled, which is equivalent to filtering the even and odd
/// samples independently at 48 kHz. `x_off = None` means in place (C `x == y`).
#[cfg(feature = "qext")]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
fn comb_filter_qext(
    y: &mut [OpusVal32],
    x: Option<(&[OpusVal32], usize)>,
    y_off: usize,
    t0: i32,
    t1: i32,
    n: i32,
    g0: OpusVal16,
    g1: OpusVal16,
    tapset0: i32,
    tapset1: i32,
    window: &[CeltCoef],
    overlap: i32,
) {
    use crate::celt::vq::Scratch;
    const MP: usize = COMBFILTER_MAXPERIOD as usize;
    let mut new_window = [CeltCoef::default(); 120];
    let n2 = (n / 2) as usize;
    let overlap2 = overlap / 2;
    let mut mem_buf_s = Scratch::<OpusVal32, { MP + 960 }>::new();
    let mut buf_s = Scratch::<OpusVal32, { MP + 960 }>::new();
    let mem_buf = mem_buf_s.get(MP + n2);
    let buf = buf_s.get(n2);
    // At 96 kHz, we double the period and the spacing between taps, which is equivalent to
    // creating a mirror image of the filter around 24 kHz. It also means we can process the
    // even and odd samples completely independently.
    for s in 0..2usize {
        for (i, w) in new_window[..overlap2 as usize].iter_mut().enumerate() {
            *w = window[2 * i + s];
        }
        match x {
            Some((xs, x_off)) => {
                for (i, m) in mem_buf.iter_mut().enumerate() {
                    *m = xs[x_off + 2 * i + s - 2 * MP];
                }
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = y[y_off + 2 * i + s];
                }
                comb_filter(
                    buf,
                    mem_buf,
                    MP,
                    t0,
                    t1,
                    n2 as i32,
                    g0,
                    g1,
                    tapset0,
                    tapset1,
                    &new_window,
                    overlap2,
                );
                for (i, &b) in buf.iter().enumerate() {
                    y[y_off + 2 * i + s] = b;
                }
            }
            None => {
                for (i, m) in mem_buf.iter_mut().enumerate() {
                    *m = y[y_off + 2 * i + s - 2 * MP];
                }
                comb_filter_inplace(
                    mem_buf,
                    MP,
                    t0,
                    t1,
                    n2 as i32,
                    g0,
                    g1,
                    tapset0,
                    tapset1,
                    &new_window,
                    overlap2,
                );
                for i in 0..n2 {
                    y[y_off + 2 * i + s] = mem_buf[MP + i];
                }
            }
        }
    }
}

/// Port of celt/celt.c:comb_filter with separate buffers (C `x != y`).
///
/// Filters `x` into `y[..n]`. `x[0]` is `xs[x_off]`; the filter reads up to
/// `max(T0, T1, COMBFILTER_MINPERIOD) + 2` samples of history before it (twice that, plus
/// `COMBFILTER_MAXPERIOD` alignment, for the QEXT 96 kHz path: `x_off >= 2*COMBFILTER_MAXPERIOD`).
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn comb_filter(
    y: &mut [OpusVal32],
    xs: &[OpusVal32],
    x_off: usize,
    t0: i32,
    t1: i32,
    n: i32,
    g0: OpusVal16,
    g1: OpusVal16,
    tapset0: i32,
    tapset1: i32,
    window: &[CeltCoef],
    overlap: i32,
) {
    #[cfg(feature = "qext")]
    if overlap == 240 {
        comb_filter_qext(
            y,
            Some((xs, x_off)),
            0,
            t0,
            t1,
            n,
            g0,
            g1,
            tapset0,
            tapset1,
            window,
            overlap,
        );
        return;
    }
    let mut io = CombSeparate { y, xs, x_off };
    comb_filter_io(
        &mut io, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
    );
}

/// Port of celt/celt.c:comb_filter filtering in place (C `x == y`): `buf[off..off+n]` is
/// replaced by the filtered signal, with the history before `off` as input. Output samples
/// written earlier in the call are read back as (IIR) input, exactly like the aliased C call.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn comb_filter_inplace(
    buf: &mut [OpusVal32],
    off: usize,
    t0: i32,
    t1: i32,
    n: i32,
    g0: OpusVal16,
    g1: OpusVal16,
    tapset0: i32,
    tapset1: i32,
    window: &[CeltCoef],
    overlap: i32,
) {
    #[cfg(feature = "qext")]
    if overlap == 240 {
        comb_filter_qext(
            buf, None, off, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
        );
        return;
    }
    let mut io = CombInPlace { buf, off };
    comb_filter_io(
        &mut io, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
    );
}

/// `tf_select_table`: TF change table. Positive values mean better frequency resolution
/// (longer effective window), whereas negative values mean better time resolution (shorter
/// effective window). The second index is computed as:
/// `4*isTransient + 2*tf_select + per_band_flag`.
#[rustfmt::skip]
pub static TF_SELECT_TABLE: [[i8; 8]; 4] = [
    // isTransient=0     isTransient=1
    [0, -1, 0, -1,    0, -1, 0, -1], // 2.5 ms
    [0, -1, 0, -2,    1,  0, 1, -1], // 5 ms
    [0, -2, 0, -3,    2,  0, 1, -1], // 10 ms
    [0, -2, 0, -3,    3,  0, 1, -1], // 20 ms
];

/// Port of celt/celt.c:init_caps: per-band maximum allocation (in 1/8 bits).
pub fn init_caps(m: &CeltMode, cap: &mut [i32], lm: i32, c: i32) {
    let nb = m.nb_ebands;
    for i in 0..nb {
        let iu = i as usize;
        let n = i32::from(m.e_bands[iu + 1] - m.e_bands[iu]) << lm;
        cap[iu] =
            ((i32::from(m.cache.caps[(nb * (2 * lm + c - 1) + i) as usize]) + 64) * c * n) >> 2;
    }
}

/// `PACKAGE_VERSION` reported by [`opus_get_version_string`].
pub const PACKAGE_VERSION: &str = "1.6.1";

/// Port of celt/celt.c:opus_strerror: converts an Opus error code into a human readable
/// string.
#[must_use]
pub const fn opus_strerror(error: i32) -> &'static str {
    const ERROR_STRINGS: [&str; 8] = [
        "success",
        "invalid argument",
        "buffer too small",
        "internal error",
        "corrupted stream",
        "request not implemented",
        "invalid state",
        "memory allocation failed",
    ];
    if error > 0 || error < -7 {
        "unknown error"
    } else {
        ERROR_STRINGS[(-error) as usize]
    }
}

/// Port of celt/celt.c:opus_get_version_string. Applications may rely on the presence of a
/// `-fixed` suffix to detect fixed-point builds.
#[must_use]
pub const fn opus_get_version_string() -> &'static str {
    // C: "libopus " PACKAGE_VERSION (+ "-fixed" / "-fuzzing" in those builds).
    #[cfg(all(feature = "fixed-point", not(feature = "fuzzing")))]
    let s = "libopus 1.6.1-fixed";
    #[cfg(all(not(feature = "fixed-point"), not(feature = "fuzzing")))]
    let s = "libopus 1.6.1";
    #[cfg(all(feature = "fixed-point", feature = "fuzzing"))]
    let s = "libopus 1.6.1-fixed-fuzzing";
    #[cfg(all(not(feature = "fixed-point"), feature = "fuzzing"))]
    let s = "libopus 1.6.1-fuzzing";
    s
}

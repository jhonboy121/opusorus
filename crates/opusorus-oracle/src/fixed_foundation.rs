//! Oracle bindings for the `fixed_foundation` unit (feature `fixed-point`): the fixed-point
//! macros of `celt/arch.h` + `celt/fixed_generic.h`, the fixed-point mathops, the float API
//! conversions of `celt/float_cast.h`, the fixed-point static modes, CWRS and Laplace coding.
//! Shims: `csrc/fixed_foundation.c` and `csrc/fixed_foundation_int32.c` (the 32-bit
//! `OPUS_FAST_INT64 == 0` multiply forms).

use core::ffi::{c_int, c_uint};

unsafe extern "C" {
    fn ofx_op1(op: c_int, a: c_int) -> c_int;
    fn ofx_op2(op: c_int, a: c_int, b: c_int) -> c_int;
    fn ofx32_op2(op: c_int, a: c_int, b: c_int) -> c_int;
    fn ofx_op3(op: c_int, c: c_int, a: c_int, b: c_int) -> c_int;
    fn ofx_mult16_16u(a: c_uint, b: c_uint) -> c_uint;
    fn ofx_uadd32(a: c_uint, b: c_uint) -> c_uint;
    fn ofx_usub32(a: c_uint, b: c_uint) -> c_uint;
    fn ofx_shr64(a: i64, shift: c_int) -> i64;
    fn ofx_qconst16(x: f64, bits: c_int) -> c_int;
    fn ofx_qconst16f(x: f32, bits: c_int) -> c_int;
    fn ofx_qconst32(x: f64, bits: c_int) -> c_int;
    fn ofx_qconst32f(x: f32, bits: c_int) -> c_int;
    fn ofx_gconst(x: f64) -> c_int;
    fn ofx_gconst2(x: f64, bits: c_int) -> c_int;
    fn ofx_frac_mul16(a: c_int, b: c_int) -> c_int;
    fn ofx_res2float(a: c_int) -> f32;
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_float2res(a: f32) -> c_int;
    fn ofx_float2int(x: f32) -> c_int;
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_float2int16(x: f32) -> c_int;
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_float2int24(x: f32) -> c_int;
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_float2sig(x: f32) -> c_int;
    fn ofx_fast_atan2f(y: f32, x: f32) -> f32;
    #[cfg(feature = "qext")]
    fn ofx_celt_cos_norm2(x: f32) -> f32;
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_celt_float2int16(input: *const f32, out: *mut i16, cnt: c_int);
    #[cfg(not(feature = "disable-float-api"))]
    fn ofx_opus_limit2_checkwithin1(samples: *mut f32, cnt: c_int) -> c_int;
    fn ofx_constants(out: *mut c_int);
    fn ofx_math1(op: c_int, x: c_int) -> c_int;
    fn ofx_math2(op: c_int, a: c_int, b: c_int) -> c_int;
    fn ofx_isqrt32(x: c_uint) -> c_uint;
    fn ofx_maxabs16(x: *const i16, len: c_int) -> c_int;
    #[cfg(feature = "fixed-res24")]
    fn ofx_maxabs_res(x: *const i32, len: c_int) -> c_int;
    #[cfg(not(feature = "fixed-res24"))]
    fn ofx_maxabs_res(x: *const i16, len: c_int) -> c_int;
    fn ofx_maxabs32(x: *const i32, len: c_int) -> c_int;
    fn ofx_mode_scalars(fs: c_int, frame_size: c_int, out: *mut c_int) -> c_int;
    fn ofx_mode_array(
        fs: c_int,
        frame_size: c_int,
        which: c_int,
        out: *mut c_int,
        n: c_int,
    ) -> c_int;
    fn ofx_mode_fft(
        fs: c_int,
        frame_size: c_int,
        k: c_int,
        scalars: *mut c_int,
        bitrev: *mut c_int,
        twiddles: *mut c_int,
        ntw: c_int,
    ) -> c_int;
    fn ofx_cwrs_roundtrip(
        y: *const c_int,
        n: c_int,
        k: c_int,
        buf: *mut u8,
        size: c_int,
        y_out: *mut c_int,
        rng_out: *mut c_uint,
    ) -> c_int;
    fn ofx_laplace_roundtrip(
        values: *mut c_int,
        count: c_int,
        fs: *const c_uint,
        decay: *const c_int,
        buf: *mut u8,
        size: c_int,
        out: *mut c_int,
        rng_out: *mut c_uint,
    );
}

/// One-argument macro ids of `ofx_op1` (see `csrc/fixed_foundation.c`).
pub mod id1 {
    pub const NEG16: i32 = 0;
    pub const NEG32: i32 = 1;
    pub const NEG32_OVFLW: i32 = 2;
    pub const EXTRACT16: i32 = 3;
    pub const EXTEND32: i32 = 4;
    pub const SATURATE16: i32 = 5;
    pub const HALF16: i32 = 6;
    pub const HALF32: i32 = 7;
    pub const ABS16: i32 = 8;
    pub const ABS32: i32 = 9;
    pub const SAT16: i32 = 10;
    pub const SIG2WORD16: i32 = 11;
    pub const CELT_ISNAN: i32 = 12;
    pub const COEF2VAL16: i32 = 13;
    pub const SIG2RES: i32 = 14;
    pub const RES2INT16: i32 = 15;
    pub const RES2INT24: i32 = 16;
    pub const INT16TORES: i32 = 17;
    pub const INT24TORES: i32 = 18;
    pub const RES2SIG: i32 = 19;
    pub const RES2VAL16: i32 = 20;
    pub const INT16TOSIG: i32 = 21;
    pub const INT24TOSIG: i32 = 22;
}

/// Two-argument macro ids of `ofx_op2` (0..=8 also for the 32-bit forms, [`op2_int32`]).
pub mod id2 {
    pub const MULT16_16SU: i32 = 0;
    pub const MULT16_32_Q16: i32 = 1;
    pub const MULT16_32_P16: i32 = 2;
    pub const MULT16_32_Q15: i32 = 3;
    pub const MULT32_32_Q16: i32 = 4;
    pub const MULT32_32_Q31: i32 = 5;
    pub const MULT32_32_P31: i32 = 6;
    pub const MULT32_32_P31_OVFLW: i32 = 7;
    pub const MULT32_32_Q32: i32 = 8;
    pub const SHR16: i32 = 9;
    pub const SHL16: i32 = 10;
    pub const SHR32: i32 = 11;
    pub const SHL32: i32 = 12;
    pub const PSHR32: i32 = 13;
    pub const VSHR32: i32 = 14;
    pub const SHR: i32 = 15;
    pub const SHL: i32 = 16;
    pub const PSHR: i32 = 17;
    pub const SATURATE: i32 = 18;
    pub const ROUND16: i32 = 19;
    pub const SROUND16: i32 = 20;
    pub const ADD16: i32 = 21;
    pub const SUB16: i32 = 22;
    pub const ADD32: i32 = 23;
    pub const SUB32: i32 = 24;
    pub const ADD32_OVFLW: i32 = 25;
    pub const SUB32_OVFLW: i32 = 26;
    pub const SHL32_OVFLW: i32 = 27;
    pub const PSHR32_OVFLW: i32 = 28;
    pub const MULT16_16_16: i32 = 29;
    pub const MULT32_32_32: i32 = 30;
    pub const MULT16_16: i32 = 31;
    pub const MULT16_16_Q11_32: i32 = 32;
    pub const MULT16_16_Q11: i32 = 33;
    pub const MULT16_16_Q13: i32 = 34;
    pub const MULT16_16_Q14: i32 = 35;
    pub const MULT16_16_Q15: i32 = 36;
    pub const MULT16_16_P13: i32 = 37;
    pub const MULT16_16_P14: i32 = 38;
    pub const MULT16_16_P15: i32 = 39;
    pub const DIV32_16: i32 = 40;
    pub const DIV32: i32 = 41;
    pub const MIN16: i32 = 42;
    pub const MAX16: i32 = 43;
    pub const MIN32: i32 = 44;
    pub const MAX32: i32 = 45;
    pub const IMIN: i32 = 46;
    pub const IMAX: i32 = 47;
    pub const MULT_COEF_32: i32 = 48;
    pub const MULT_COEF: i32 = 49;
    pub const MULT_COEF_TAPS: i32 = 50;
    pub const IMUL32: i32 = 51;
    pub const MING: i32 = 52;
    pub const MAXG: i32 = 53;
    pub const ADD_RES: i32 = 54;
    pub const MULT16_RES_Q15: i32 = 55;
}

/// Three-argument macro ids of `ofx_op3`.
pub mod id3 {
    pub const MAC16_16: i32 = 0;
    pub const MAC16_32_Q15: i32 = 1;
    pub const MAC16_32_Q16: i32 = 2;
    pub const MAC_COEF_32_ARM: i32 = 3;
}

/// Fixed-point mathops ids of `ofx_math1`.
pub mod mid1 {
    pub const CELT_RSQRT_NORM: i32 = 0;
    pub const CELT_RSQRT_NORM32: i32 = 1;
    pub const CELT_SQRT: i32 = 2;
    pub const CELT_SQRT32: i32 = 3;
    pub const CELT_COS_NORM: i32 = 4;
    pub const CELT_COS_NORM32: i32 = 5;
    pub const CELT_LOG2: i32 = 6;
    pub const CELT_EXP2_FRAC: i32 = 7;
    pub const CELT_EXP2: i32 = 8;
    pub const CELT_LOG2_DB: i32 = 9;
    pub const CELT_EXP2_DB_FRAC: i32 = 10;
    pub const CELT_EXP2_DB: i32 = 11;
    pub const CELT_RCP: i32 = 12;
    pub const CELT_RCP_NORM16: i32 = 13;
    pub const CELT_RCP_NORM32: i32 = 14;
    pub const CELT_ATAN_NORM: i32 = 15;
    pub const CELT_ATAN01: i32 = 16;
    pub const CELT_ILOG2: i32 = 17;
    pub const CELT_ZLOG2: i32 = 18;
}

/// Fixed-point mathops ids of `ofx_math2`.
pub mod mid2 {
    pub const CELT_DIV: i32 = 0;
    pub const FRAC_DIV32_Q29: i32 = 1;
    pub const FRAC_DIV32: i32 = 2;
    pub const CELT_ATAN2P_NORM: i32 = 3;
    pub const CELT_ATAN2P: i32 = 4;
}

/// C one-argument macro `op` ([`id1`]) applied to the `int` `a`.
#[must_use]
pub fn op1(op: i32, a: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx_op1(op, a) }
}
/// C two-argument macro `op` ([`id2`]) applied to the `int`s `a`, `b`.
#[must_use]
pub fn op2(op: i32, a: i32, b: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx_op2(op, a, b) }
}
/// C 32-bit form (`OPUS_FAST_INT64 == 0`) of the multiply macro `op` ([`id2`] ids 1..=8).
#[must_use]
pub fn op2_int32(op: i32, a: i32, b: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx32_op2(op, a, b) }
}
/// C three-argument macro `op` ([`id3`]) applied to `c`, `a`, `b`.
#[must_use]
pub fn op3(op: i32, c: i32, a: i32, b: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx_op3(op, c, a, b) }
}
/// C fixed-point mathops function `op` ([`mid1`]).
#[must_use]
pub fn math1(op: i32, x: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx_math1(op, x) }
}
/// C fixed-point mathops function `op` ([`mid2`]).
#[must_use]
pub fn math2(op: i32, a: i32, b: i32) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { ofx_math2(op, a, b) }
}

macro_rules! pure {
    ($($(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),*) -> $r:ty = $c:ident;)+) => {$(
        $(#[$m])*
        #[must_use]
        pub fn $name($($a: $t),*) -> $r {
            // SAFETY: pure function of its arguments.
            unsafe { $c($($a),*) }
        }
    )+};
}
pure! {
    /// C `MULT16_16U` (32-bit configuration).
    fn mult16_16u(a: u32, b: u32) -> u32 = ofx_mult16_16u;
    /// C `UADD32`.
    fn uadd32(a: u32, b: u32) -> u32 = ofx_uadd32;
    /// C `USUB32`.
    fn usub32(a: u32, b: u32) -> u32 = ofx_usub32;
    /// C `SHR64`.
    fn shr64(a: i64, shift: i32) -> i64 = ofx_shr64;
    /// C `QCONST16` of a `double`.
    fn qconst16(x: f64, bits: i32) -> i32 = ofx_qconst16;
    /// C `QCONST16` of a `float`.
    fn qconst16f(x: f32, bits: i32) -> i32 = ofx_qconst16f;
    /// C `QCONST32` of a `double`.
    fn qconst32(x: f64, bits: i32) -> i32 = ofx_qconst32;
    /// C `QCONST32` of a `float`.
    fn qconst32f(x: f32, bits: i32) -> i32 = ofx_qconst32f;
    /// C `GCONST`.
    fn gconst(x: f64) -> i32 = ofx_gconst;
    /// C `GCONST2`.
    fn gconst2(x: f64, bits: i32) -> i32 = ofx_gconst2;
    /// C `FRAC_MUL16`.
    fn frac_mul16(a: i32, b: i32) -> i32 = ofx_frac_mul16;
    /// C `RES2FLOAT`.
    fn res2float(a: i32) -> f32 = ofx_res2float;
    #[cfg(not(feature = "disable-float-api"))]
    /// C `FLOAT2RES`.
    fn float2res(a: f32) -> i32 = ofx_float2res;
    /// C `float2int`.
    fn float2int(x: f32) -> i32 = ofx_float2int;
    #[cfg(not(feature = "disable-float-api"))]
    /// C `FLOAT2INT16`.
    fn float2int16(x: f32) -> i32 = ofx_float2int16;
    #[cfg(not(feature = "disable-float-api"))]
    /// C `FLOAT2INT24`.
    fn float2int24(x: f32) -> i32 = ofx_float2int24;
    #[cfg(not(feature = "disable-float-api"))]
    /// C `FLOAT2SIG` (fixed-point `float_cast.h`).
    fn float2sig(x: f32) -> i32 = ofx_float2sig;
    /// C `fast_atan2f` (compiled for `analysis.c` in fixed-point builds).
    fn fast_atan2f(y: f32, x: f32) -> f32 = ofx_fast_atan2f;
    /// C `isqrt32`.
    fn isqrt32(x: u32) -> u32 = ofx_isqrt32;
}

/// C `celt_cos_norm2` (float; compiled in fixed-point builds with QEXT).
#[cfg(feature = "qext")]
#[must_use]
pub fn celt_cos_norm2(x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { ofx_celt_cos_norm2(x) }
}

#[cfg(not(feature = "disable-float-api"))]
/// C `celt_float2int16_c`.
#[must_use]
pub fn celt_float2int16(input: &[f32]) -> Vec<i16> {
    let mut out = vec![0i16; input.len()];
    // SAFETY: both buffers hold `input.len()` elements.
    unsafe { ofx_celt_float2int16(input.as_ptr(), out.as_mut_ptr(), input.len() as c_int) };
    out
}

#[cfg(not(feature = "disable-float-api"))]
/// C `opus_limit2_checkwithin1_c` (clamps in place).
pub fn opus_limit2_checkwithin1(samples: &mut [f32]) -> i32 {
    // SAFETY: the buffer holds `samples.len()` elements.
    unsafe { ofx_opus_limit2_checkwithin1(samples.as_mut_ptr(), samples.len() as c_int) }
}

/// Number of values returned by [`constants`].
pub const N_CONSTANTS: usize = 27;

/// Build constants: `Q15ONE, Q31ONE, COEF_ONE, SIG_SHIFT, SIG_SAT, NORM_SHIFT, NORM_SCALING,
/// DB_SHIFT, EPSILON, VERY_SMALL, VERY_LARGE16, Q15_ONE, RES_SHIFT, MAX_ENCODING_DEPTH,
/// OPUS_FAST_INT64`, then `sizeof` of `opus_val16, opus_val32, opus_val64, celt_sig, celt_norm,
/// celt_ener, celt_glog, opus_res, celt_coef, kiss_fft_scalar, kiss_twiddle_scalar`, then
/// `GLOBAL_STACK_SIZE`.
#[must_use]
pub fn constants() -> [i32; N_CONSTANTS] {
    let mut out = [0 as c_int; N_CONSTANTS];
    // SAFETY: `out` has room for every value the shim writes.
    unsafe { ofx_constants(out.as_mut_ptr()) };
    out
}

/// C `celt_maxabs16`.
#[must_use]
pub fn maxabs16(x: &[i16]) -> i32 {
    // SAFETY: reads `x.len()` elements.
    unsafe { ofx_maxabs16(x.as_ptr(), x.len() as c_int) }
}
/// C `celt_maxabs_res` (24-bit `opus_res`).
#[cfg(feature = "fixed-res24")]
#[must_use]
pub fn maxabs_res(x: &[i32]) -> i32 {
    // SAFETY: reads `x.len()` elements.
    unsafe { ofx_maxabs_res(x.as_ptr(), x.len() as c_int) }
}
/// C `celt_maxabs_res` (16-bit `opus_res`).
#[cfg(not(feature = "fixed-res24"))]
#[must_use]
pub fn maxabs_res(x: &[i16]) -> i32 {
    // SAFETY: reads `x.len()` elements.
    unsafe { ofx_maxabs_res(x.as_ptr(), x.len() as c_int) }
}
/// C `celt_maxabs32`.
#[must_use]
pub fn maxabs32(x: &[i32]) -> i32 {
    // SAFETY: reads `x.len()` elements.
    unsafe { ofx_maxabs32(x.as_ptr(), x.len() as c_int) }
}

/// Scalar fields of the static mode `(fs, frame_size)`: `Fs, overlap, nbEBands, effEBands,
/// preemph[4], maxLM, nbShortMdcts, shortMdctSize, nbAllocVectors, mdct.n, mdct.maxshift,
/// cache.size, qext_cache.size` (-1 without QEXT); `None` if the mode does not exist.
#[must_use]
pub fn mode_scalars(fs: i32, frame_size: i32) -> Option<[i32; 16]> {
    let mut out = [0 as c_int; 16];
    // SAFETY: `out` has room for the 16 values written.
    let r = unsafe { ofx_mode_scalars(fs, frame_size, out.as_mut_ptr()) };
    (r == 0).then_some(out)
}

/// Mode array ids of [`mode_array`].
pub mod mode_arr {
    pub const E_BANDS: i32 = 0;
    pub const ALLOC_VECTORS: i32 = 1;
    pub const LOG_N: i32 = 2;
    pub const WINDOW: i32 = 3;
    pub const MDCT_TRIG: i32 = 4;
    pub const CACHE_INDEX: i32 = 5;
    pub const CACHE_BITS: i32 = 6;
    pub const CACHE_CAPS: i32 = 7;
    pub const QEXT_CACHE_INDEX: i32 = 8;
    pub const QEXT_CACHE_BITS: i32 = 9;
    pub const QEXT_CACHE_CAPS: i32 = 10;
}

/// The first `n` entries of the static mode's array `which` ([`mode_arr`]); the caller must not
/// ask for more entries than the C array holds.
#[must_use]
pub fn mode_array(fs: i32, frame_size: i32, which: i32, n: usize) -> Option<Vec<i32>> {
    let mut out = vec![0 as c_int; n];
    // SAFETY: `out` holds `n` values; the C array is at least as long (caller contract).
    let r = unsafe { ofx_mode_array(fs, frame_size, which, out.as_mut_ptr(), n as c_int) };
    (r == 0).then_some(out)
}

/// FFT state of a static mode's MDCT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftState {
    /// `nfft, scale, scale_shift, shift, factors[16]`.
    pub scalars: Vec<i32>,
    /// `bitrev[nfft]`.
    pub bitrev: Vec<i32>,
    /// First `ntw` twiddles, interleaved `r, i`.
    pub twiddles: Vec<i32>,
}

/// FFT state `k` of the static mode `(fs, frame_size)`, with its first `ntw` twiddles.
#[must_use]
pub fn mode_fft(fs: i32, frame_size: i32, k: i32, ntw: usize) -> Option<FftState> {
    let mut scalars = vec![0 as c_int; 20];
    let mut bitrev = vec![0 as c_int; 2048];
    let mut twiddles = vec![0 as c_int; 2 * ntw];
    // SAFETY: buffers sized for 20 scalars, nfft <= 2048 bit-reversal entries and `ntw`
    // twiddles (the caller must not ask for more than the table holds).
    let r = unsafe {
        ofx_mode_fft(
            fs,
            frame_size,
            k,
            scalars.as_mut_ptr(),
            bitrev.as_mut_ptr(),
            twiddles.as_mut_ptr(),
            ntw as c_int,
        )
    };
    if r != 0 {
        return None;
    }
    bitrev.truncate(scalars[0] as usize);
    Some(FftState {
        scalars,
        bitrev,
        twiddles,
    })
}

/// Result of [`cwrs_roundtrip`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CwrsOut {
    /// Encoded buffer.
    pub buf: Vec<u8>,
    /// Final encoder `rng`.
    pub rng: u32,
    /// Decoded pulses.
    pub y: Vec<i32>,
    /// `decode_pulses` return value (squared norm, `opus_val32`).
    pub yy: i32,
}

/// C `encode_pulses` then `decode_pulses` of `y` (`n` = `y.len()`, `k` pulses).
#[must_use]
pub fn cwrs_roundtrip(y: &[i32], k: i32, size: usize) -> CwrsOut {
    let mut buf = vec![0u8; size];
    let mut y_out = vec![0 as c_int; y.len()];
    let mut rng = 0;
    // SAFETY: `y`/`y_out` hold n values, `buf` `size` bytes.
    let yy = unsafe {
        ofx_cwrs_roundtrip(
            y.as_ptr(),
            y.len() as c_int,
            k,
            buf.as_mut_ptr(),
            size as c_int,
            y_out.as_mut_ptr(),
            &mut rng,
        )
    };
    CwrsOut {
        buf,
        rng,
        y: y_out,
        yy,
    }
}

/// Result of [`laplace_roundtrip`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceOut {
    /// The values after `ec_laplace_encode` (which clamps them).
    pub encoded: Vec<i32>,
    /// Encoded buffer.
    pub buf: Vec<u8>,
    /// Final encoder `rng`.
    pub rng: u32,
    /// Decoded values.
    pub decoded: Vec<i32>,
}

/// C `ec_laplace_encode` of every `(values[i], fs[i], decay[i])`, then `ec_laplace_decode`.
#[must_use]
pub fn laplace_roundtrip(values: &[i32], fs: &[u32], decay: &[i32], size: usize) -> LaplaceOut {
    assert!(fs.len() == values.len() && decay.len() == values.len());
    let mut encoded = values.to_vec();
    let mut buf = vec![0u8; size];
    let mut decoded = vec![0 as c_int; values.len()];
    let mut rng = 0;
    // SAFETY: every array holds `values.len()` entries, `buf` `size` bytes.
    unsafe {
        ofx_laplace_roundtrip(
            encoded.as_mut_ptr(),
            values.len() as c_int,
            fs.as_ptr(),
            decay.as_ptr(),
            buf.as_mut_ptr(),
            size as c_int,
            decoded.as_mut_ptr(),
            &mut rng,
        );
    }
    LaplaceOut {
        encoded,
        buf,
        rng,
        decoded,
    }
}

//! Port of `celt/mathops.h`, `celt/mathops.c` and `celt/float_cast.h`.
//!
//! The items in this file are shared by the float and the fixed-point builds (integer helpers,
//! the float API conversions `float2int*`, and the float functions that fixed-point builds still
//! compile: `fast_atan2f` for `src/analysis.c`, `celt_cos_norm2` for QEXT). The build-specific
//! functions with the same names (`celt_log2`, `celt_sqrt`, `celt_rcp`, `frac_div32`, ...) live in
//! `mathops/float.rs` (default) and `mathops/fixed.rs` (feature `fixed-point`) and are re-exported
//! here.
//!
//! Every float function preserves the C float/double promotion semantics exactly; comments show
//! the original C expression where promotion is non-obvious.

#[cfg(feature = "fixed-point")]
mod fixed;
#[cfg(not(feature = "fixed-point"))]
mod float;

#[cfg(feature = "fixed-point")]
pub use fixed::*;
#[cfg(not(feature = "fixed-point"))]
pub use float::*;

#[cfg(not(feature = "disable-float-api"))]
use crate::celt::arch::{CELT_SIG_SCALE, max32, min32};
use crate::celt::entcode::ec_ilog;
use crate::math;

/// `PI` (a `double` constant in C).
pub const PI: f64 = 3.1415926535897931;

/// `FRAC_MUL16`: multiplies two 16-bit fractional values. Bit-exactness is important.
#[inline(always)]
#[must_use]
pub const fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// Port of `isqrt32`: `floor(sqrt(val))` with exact arithmetic. `val` must be > 0.
#[must_use]
pub const fn isqrt32(mut val: u32) -> u32 {
    let mut g: u32 = 0;
    let mut bshift: i32 = (ec_ilog(val) - 1) >> 1;
    let mut b: u32 = 1u32 << bshift;
    loop {
        let t: u32 = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// Port of `fast_atan2f`.
#[inline]
#[must_use]
pub const fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.43157974;
    const CB: f32 = 0.67848403;
    const CC: f32 = 0.08595542;
    const CE: f32 = PI as f32 / 2.0;
    let x2 = x * x;
    let y2 = y * y;
    // For very small values, we don't care about the answer, so we can just return 0.
    if x2 + y2 < 1e-18 {
        return 0.0;
    }
    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + if y < 0.0 { -CE } else { CE }
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den + if y < 0.0 { -CE } else { CE }
            - if x * y < 0.0 { -CE } else { CE }
    }
}

/// Port of `celt_cos_norm2`: `cos(PI/2 * x)` via an even polynomial.
#[inline]
#[must_use]
pub fn celt_cos_norm2(mut x: f32) -> f32 {
    const A0: f32 = 9.999999403953552246093750000000e-01;
    const A2: f32 = -1.233698248863220214843750000000000;
    const A4: f32 = 2.536507546901702880859375000000e-01;
    const A6: f32 = -2.08106283098459243774414062500e-02;
    const A8: f32 = 8.581906440667808055877685546875e-04;
    // Restrict x to [-1, 3]. C: `x -= 4*floor(.25*(x+1));` (double arithmetic).
    x = (x as f64 - 4.0 * math::floor(0.25 * (x + 1.0) as f64)) as f32;
    // Negative sign for [1, 3].
    let gt1 = (x > 1.0) as i32;
    let output_sign: i32 = 1 - 2 * gt1;
    // Restrict to [-1, 1]. C: `x -= 2*(x>1);` (int converted to float).
    x -= (2 * gt1) as f32;
    let x_norm_sq = x * x;
    output_sign as f32
        * (A0 + x_norm_sq * (A2 + x_norm_sq * (A4 + x_norm_sq * (A6 + x_norm_sq * A8))))
}

/// `celt_ilog2`: integer log2, undefined for `x <= 0`.
#[inline(always)]
#[must_use]
pub const fn celt_ilog2(x: i32) -> i32 {
    celt_sig_assert!(x > 0);
    ec_ilog(x as u32) - 1
}
/// `celt_zlog2`: integer log2, defined as 0 for `x <= 0`.
#[inline(always)]
#[must_use]
pub const fn celt_zlog2(x: i32) -> i32 {
    if x <= 0 { 0 } else { celt_ilog2(x) }
}

/// `float2int`: round to nearest, ties to even (`lrintf` / AArch64 `fcvtns`).
#[inline(always)]
#[must_use]
pub fn float2int(x: f32) -> i32 {
    math::lrintf(x)
}

/// `FLOAT2INT16` (`celt/float_cast.h`; not with `DISABLE_FLOAT_API`).
#[cfg(not(feature = "disable-float-api"))]
#[inline(always)]
#[must_use]
pub fn float2int16(mut x: f32) -> i16 {
    x *= CELT_SIG_SCALE;
    x = max32(x, -32768.0);
    x = min32(x, 32767.0);
    float2int(x) as i16
}

/// `FLOAT2INT24` (not with `DISABLE_FLOAT_API`).
#[cfg(not(feature = "disable-float-api"))]
#[inline(always)]
#[must_use]
pub fn float2int24(mut x: f32) -> i32 {
    x *= CELT_SIG_SCALE * 256.0;
    x = max32(x, -16_777_216.0);
    x = min32(x, 16_777_216.0);
    float2int(x)
}

/// Port of `celt_float2int16_c` (not with `DISABLE_FLOAT_API`).
#[cfg(not(feature = "disable-float-api"))]
pub fn celt_float2int16(input: &[f32], out: &mut [i16]) {
    for (o, &i) in out.iter_mut().zip(input) {
        *o = float2int16(i);
    }
}

/// Port of `opus_limit2_checkwithin1_c`: clamps to [-2, 2]; returns whether all samples are
/// known to be within [-1, 1] (the C version never knows, so it returns `false` unless empty).
/// Not with `DISABLE_FLOAT_API`.
#[cfg(not(feature = "disable-float-api"))]
pub fn opus_limit2_checkwithin1(samples: &mut [f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    for s in samples.iter_mut() {
        let mut clipped = *s;
        clipped = crate::celt::arch::fmax(-2.0, clipped);
        clipped = crate::celt::arch::fmin(2.0, clipped);
        *s = clipped;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt32_exhaustive_small_and_edges() {
        for v in 1u32..100_000 {
            let r = isqrt32(v) as u64;
            assert!(r * r <= v as u64 && (r + 1) * (r + 1) > v as u64, "v={v}");
        }
        for v in [u32::MAX, u32::MAX - 1, 1 << 31, (1 << 31) - 1] {
            let r = isqrt32(v) as u64;
            assert!(r * r <= v as u64 && (r + 1) * (r + 1) > v as u64, "v={v}");
        }
    }

    #[test]
    #[cfg(not(feature = "disable-float-api"))]
    fn float2int16_saturates() {
        assert_eq!(float2int16(1.0), 32767);
        assert_eq!(float2int16(-1.0), -32768);
        assert_eq!(float2int16(2.0), 32767);
        assert_eq!(float2int16(0.5 / 32768.0), 0);
        assert_eq!(float2int16(1.5 / 32768.0), 2);
    }

    #[test]
    fn frac_mul16_matches_c() {
        assert_eq!(frac_mul16(16384, 16384), 8192);
        assert_eq!(frac_mul16(-32768, 32767), -32767);
    }
}

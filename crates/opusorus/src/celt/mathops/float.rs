//! Port of the float-build (`#ifndef FIXED_POINT`) parts of `celt/mathops.h`.
//!
//! Every function preserves the C float/double promotion semantics exactly; comments show the
//! original C expression where promotion is non-obvious.

use super::{PI, celt_cos_norm2};
use crate::celt::arch::{max32, min16};
use crate::math;

/// Port of `celt_maxabs16` (float): `max(max(x), -min(x))` with 0 floor.
#[inline]
#[must_use]
pub fn celt_maxabs16(x: &[f32]) -> f32 {
    let mut maxval: f32 = 0.0;
    let mut minval: f32 = 0.0;
    for &v in x {
        maxval = crate::celt::arch::max16(maxval, v);
        minval = min16(minval, v);
    }
    max32(maxval, -minval)
}

/// `celt_maxabs_res` (float build alias of [`celt_maxabs16`]).
#[inline]
#[must_use]
pub fn celt_maxabs_res(x: &[f32]) -> f32 {
    celt_maxabs16(x)
}

/// `celt_maxabs32` (float build alias of [`celt_maxabs16`]).
#[inline]
#[must_use]
pub fn celt_maxabs32(x: &[f32]) -> f32 {
    celt_maxabs16(x)
}

/// Port of `celt_atan_norm`: `atan(x)*2/pi` via an order-15 odd Remez polynomial.
#[inline]
#[must_use]
pub const fn celt_atan_norm(x: f32) -> f32 {
    const ATAN2_2_OVER_PI: f32 = 0.636619772367581;
    const A03: f32 = -3.3331659436225891113281250000e-01;
    const A05: f32 = 1.99627041816711425781250000000e-01;
    const A07: f32 = -1.3976582884788513183593750000e-01;
    const A09: f32 = 9.79423448443412780761718750000e-02;
    const A11: f32 = -5.7773590087890625000000000000e-02;
    const A13: f32 = 2.30401363223791122436523437500e-02;
    const A15: f32 = -4.3554059229791164398193359375e-03;
    let x_sq = x * x;
    ATAN2_2_OVER_PI
        * (x + x
            * x_sq
            * (A03
                + x_sq
                    * (A05
                        + x_sq * (A07 + x_sq * (A09 + x_sq * (A11 + x_sq * (A13 + x_sq * A15)))))))
}

/// Port of `celt_atan2p_norm`: `atan2(y,x)*2/pi` for `x,y >= 0`.
#[inline]
#[must_use]
pub const fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    debug_assert!(x >= 0.0 && y >= 0.0);
    // For very small values, we don't care about the answer.
    if (x * x + y * y) < 1e-18 {
        return 0.0;
    }
    if y < x {
        celt_atan_norm(y / x)
    } else {
        1.0 - celt_atan_norm(x / y)
    }
}

/// `celt_sqrt(x)` = `(float)sqrt(x)`.
#[inline(always)]
#[must_use]
pub fn celt_sqrt(x: f32) -> f32 {
    math::sqrt(x as f64) as f32
}
/// `celt_sqrt32(x)` = `(float)sqrt(x)`.
#[inline(always)]
#[must_use]
pub fn celt_sqrt32(x: f32) -> f32 {
    celt_sqrt(x)
}
/// `celt_rsqrt(x)` = `1.f/celt_sqrt(x)`.
#[inline(always)]
#[must_use]
pub fn celt_rsqrt(x: f32) -> f32 {
    1.0 / celt_sqrt(x)
}
/// `celt_rsqrt_norm`.
#[inline(always)]
#[must_use]
pub fn celt_rsqrt_norm(x: f32) -> f32 {
    celt_rsqrt(x)
}
/// `celt_rsqrt_norm32`.
#[inline(always)]
#[must_use]
pub fn celt_rsqrt_norm32(x: f32) -> f32 {
    celt_rsqrt(x)
}
/// `celt_cos_norm(x)` = `(float)cos((.5f*PI)*(x))` (double arithmetic).
#[inline(always)]
#[must_use]
pub fn celt_cos_norm(x: f32) -> f32 {
    math::cos((0.5f32 as f64 * PI) * x as f64) as f32
}
/// `celt_rcp(x)` = `1.f/(x)`.
#[inline(always)]
#[must_use]
pub const fn celt_rcp(x: f32) -> f32 {
    1.0 / x
}
/// `celt_div(a,b)` = `a/b`.
#[inline(always)]
#[must_use]
pub const fn celt_div(a: f32, b: f32) -> f32 {
    a / b
}
/// `frac_div32(a,b)` = `(float)(a)/(b)`.
#[inline(always)]
#[must_use]
pub const fn frac_div32(a: f32, b: f32) -> f32 {
    a / b
}
/// `frac_div32_q29`.
#[inline(always)]
#[must_use]
pub const fn frac_div32_q29(a: f32, b: f32) -> f32 {
    a / b
}
/// `celt_log2(x)` = `(float)(1.442695040888963387*log(x))` (non-`FLOAT_APPROX` build).
#[inline(always)]
#[must_use]
pub fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387_f64 * math::log(x as f64)) as f32
}
/// `celt_exp2(x)` = `(float)exp(0.6931471805599453094*(x))`.
#[inline(always)]
#[must_use]
pub fn celt_exp2(x: f32) -> f32 {
    math::exp(0.6931471805599453094_f64 * x as f64) as f32
}
/// `celt_exp2_db` (float alias).
#[inline(always)]
#[must_use]
pub fn celt_exp2_db(x: f32) -> f32 {
    celt_exp2(x)
}
/// `celt_log2_db` (float alias).
#[inline(always)]
#[must_use]
pub fn celt_log2_db(x: f32) -> f32 {
    celt_log2(x)
}
/// `celt_sin(x)` = `celt_cos_norm2((0.5f*PI) * (x) - 1.0f)`; the argument is computed in double
/// and converted to float at the call.
#[inline(always)]
#[must_use]
pub fn celt_sin(x: f32) -> f32 {
    celt_cos_norm2(((0.5f32 as f64 * PI) * x as f64 - 1.0) as f32)
}
/// `celt_log(x)` = `celt_log2(x) * 0.6931471805599453f`.
#[inline(always)]
#[must_use]
pub fn celt_log(x: f32) -> f32 {
    celt_log2(x) * 0.6931471805599453_f32
}
/// `celt_exp(x)` = `celt_exp2((x) * 1.4426950408889634f)`.
#[inline(always)]
#[must_use]
pub fn celt_exp(x: f32) -> f32 {
    celt_exp2(x * 1.4426950408889634_f32)
}

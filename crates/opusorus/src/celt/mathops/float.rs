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
    celt_sig_assert!(x >= 0.0 && y >= 0.0);
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
#[cfg(not(feature = "float-approx"))]
#[inline(always)]
#[must_use]
pub fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387_f64 * math::log(x as f64)) as f32
}
/// `celt_exp2(x)` = `(float)exp(0.6931471805599453094*(x))` (non-`FLOAT_APPROX` build).
#[cfg(not(feature = "float-approx"))]
#[inline(always)]
#[must_use]
pub fn celt_exp2(x: f32) -> f32 {
    math::exp(0.6931471805599453094_f64 * x as f64) as f32
}

/// `log2_x_norm_coeff` (`FLOAT_APPROX`): `1 / (1 + 0.125 * index)` in single precision.
#[cfg(feature = "float-approx")]
const LOG2_X_NORM_COEFF: [f32; 8] = [
    1.000000000000000000000000000,
    8.88888895511627197265625e-01,
    8.00000000000000000000000e-01,
    7.27272748947143554687500e-01,
    6.66666686534881591796875e-01,
    6.15384638309478759765625e-01,
    5.71428596973419189453125e-01,
    5.33333361148834228515625e-01,
];

/// `log2_y_norm_coeff` (`FLOAT_APPROX`): `log2(1 + 0.125 * index)` in single precision.
#[cfg(feature = "float-approx")]
const LOG2_Y_NORM_COEFF: [f32; 8] = [
    0.0000000000000000000000000000,
    1.699250042438507080078125e-01,
    3.219280838966369628906250e-01,
    4.594316184520721435546875e-01,
    5.849624872207641601562500e-01,
    7.004396915435791015625000e-01,
    8.073549270629882812500000e-01,
    9.068905711174011230468750e-01,
];

/// Port of the `FLOAT_APPROX` `celt_log2` (celt/mathops.h): base-2 logarithm from the IEEE 754
/// exponent plus a degree-4 polynomial of the mantissa normalized to `[1, 1.125]`. As in C,
/// zero, denormals, infinities, NaN and negative inputs are not special-cased (the bit
/// manipulation produces the same garbage as the C code; the integer steps wrap like the C
/// casts do on two's complement targets).
#[cfg(feature = "float-approx")]
#[inline]
#[must_use]
pub fn celt_log2(x: f32) -> f32 {
    const A0: f32 = 8.74628424644470214843750000e-02;
    const A1: f32 = 1.357829570770263671875000000000;
    const A2: f32 = -6.3897705078125000000000000e-01;
    const A3: f32 = 4.01971250772476196289062500e-01;
    const A4: f32 = -2.8415444493293762207031250e-01;
    let mut i = x.to_bits();
    // integer = (opus_int32)(in.i>>23)-127;
    let integer = (i >> 23) as i32 - 127;
    // in.i = (opus_int32)in.i - (opus_int32)((opus_uint32)integer<<23);
    i = (i as i32).wrapping_sub(((integer as u32) << 23) as i32) as u32;
    // Normalize the mantissa range from [1, 2] to [1,1.125], and then shift x by 1.0625 to
    // [-0.0625, 0.0625].
    let range_idx = ((i >> 20) & 0x7) as usize;
    let f = f32::from_bits(i) * LOG2_X_NORM_COEFF[range_idx] - 1.0625;
    let f = A0 + f * (A1 + f * (A2 + f * (A3 + f * A4)));
    integer as f32 + f + LOG2_Y_NORM_COEFF[range_idx]
}

/// Port of the `FLOAT_APPROX` `celt_exp2` (celt/mathops.h): `2^x` as `2^floor(x)` (added to the
/// exponent bits) times a degree-5 polynomial of the fraction; 0 below `2^-50`.
#[cfg(feature = "float-approx")]
#[inline]
#[must_use]
pub fn celt_exp2(x: f32) -> f32 {
    const A0: f32 = 9.999999403953552246093750000000e-01;
    const A1: f32 = 6.931530833244323730468750000000e-01;
    const A2: f32 = 2.401536107063293457031250000000e-01;
    const A3: f32 = 5.582631751894950866699218750000e-02;
    const A4: f32 = 8.989339694380760192871093750000e-03;
    const A5: f32 = 1.877576694823801517486572265625e-03;
    // integer = (int)floor(x); (floor of the double-promoted argument)
    let integer = math::floor(x as f64) as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    let res = A0 + frac * (A1 + frac * (A2 + frac * (A3 + frac * (A4 + frac * A5))));
    // res.i = (opus_uint32)((opus_int32)res.i + (opus_int32)((opus_uint32)integer<<23)) & 0x7fffffff;
    f32::from_bits(
        ((res.to_bits() as i32).wrapping_add(((integer as u32) << 23) as i32) as u32) & 0x7fff_ffff,
    )
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

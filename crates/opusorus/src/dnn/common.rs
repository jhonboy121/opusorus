//! Port of `dnn/common.h`: `log2_approx`, `log_approx` and the u-law conversions.

use crate::math;

/// `LOG256` (`5.5451774445f`).
pub const LOG256: f32 = 5.5451774445;

/// Port of dnn/common.h:log2_approx.
#[inline(always)]
#[must_use]
pub fn log2_approx(x: f32) -> f32 {
    let mut i = x.to_bits() as i32;
    let integer = (i >> 23) - 127;
    // C: in.i -= integer<<23 (signed; wraps for negative inputs as gcc does).
    i = i.wrapping_sub(integer << 23);
    let mut frac = f32::from_bits(i as u32) - 1.5f32;
    frac = -0.41445418f32 + frac * (0.95909232f32 + frac * (-0.33951290f32 + frac * 0.16541097f32));
    (1 + integer) as f32 + frac
}

/// Port of dnn/common.h:log_approx (`0.69315f*log2_approx(x)`).
#[inline(always)]
#[must_use]
pub fn log_approx(x: f32) -> f32 {
    0.69315f32 * log2_approx(x)
}

/// Port of dnn/common.h:ulaw2lin.
#[must_use]
pub fn ulaw2lin(u: f32) -> f32 {
    let scale_1 = 32768.0f32 / 255.0f32;
    let u = u - 128.0f32;
    let s: f32 = if u >= 0.0 { 1.0 } else { -1.0 };
    // C: u = fabs(u) (double, stored back to float).
    let u = math::fabs(u as f64) as f32;
    ((s * scale_1) as f64 * (math::exp(u as f64 / 128.0 * LOG256 as f64) - 1.0)) as f32
}

/// Port of dnn/common.h:lin2ulaw.
#[must_use]
pub fn lin2ulaw(x: f32) -> i32 {
    let scale = 255.0f32 / 32768.0f32;
    let s: i32 = if x >= 0.0 { 1 } else { -1 };
    let x = math::fabs(x as f64) as f32;
    let mut u = s as f32 * (128.0f32 * log_approx(1.0f32 + scale * x) / LOG256);
    u += 128.0f32;
    // C: if (u < 0) u = 0; if (u > 255) u = 255; (same result as clamp, NaN included).
    u = u.clamp(0.0, 255.0);
    math::floor(0.5 + u as f64) as i32
}

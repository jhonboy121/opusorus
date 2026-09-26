//! Floating-point math shim.
//!
//! Every transcendental/rounding function used by the port goes through here so that:
//! * with `std` (default) we call the platform libm — the same functions the C oracle calls,
//!   which is required for bit-exactness on the same platform;
//! * without `std` we fall back to the pure-Rust `libm` crate (`no_std`).
//!
//! C `double` functions are exposed with their C names (`cos`, `log`, ...) taking `f64`;
//! C `float` functions carry an `f` suffix (`cosf`, `sqrtf`, ...).

macro_rules! unary {
    ($(#[$m:meta])* $name:ident, $ty:ty, $std:ident, $libm:ident) => {
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub fn $name(x: $ty) -> $ty {
            #[cfg(feature = "std")]
            {
                <$ty>::$std(x)
            }
            #[cfg(not(feature = "std"))]
            {
                libm::$libm(x)
            }
        }
    };
}

unary!(/// C `cos`.
    cos, f64, cos, cos);
unary!(/// C `sin`.
    sin, f64, sin, sin);
unary!(/// C `tan`.
    tan, f64, tan, tan);
unary!(/// C `exp`.
    exp, f64, exp, exp);
unary!(/// C `log`.
    log, f64, ln, log);
unary!(/// C `log10`.
    log10, f64, log10, log10);
unary!(/// C `log2`.
    log2, f64, log2, log2);
unary!(/// C `atan`.
    atan, f64, atan, atan);
unary!(/// C `tanh`.
    tanh, f64, tanh, tanh);
unary!(/// C `sqrt`.
    sqrt, f64, sqrt, sqrt);
unary!(/// C `floor`.
    floor, f64, floor, floor);
unary!(/// C `ceil`.
    ceil, f64, ceil, ceil);
unary!(/// C `cosf`.
    cosf, f32, cos, cosf);
unary!(/// C `sinf`.
    sinf, f32, sin, sinf);
unary!(/// C `expf`.
    expf, f32, exp, expf);
unary!(/// C `logf`.
    logf, f32, ln, logf);
unary!(/// C `log10f`.
    log10f, f32, log10, log10f);
unary!(/// C `log2f`.
    log2f, f32, log2, log2f);
unary!(/// C `atanf`.
    atanf, f32, atan, atanf);
unary!(/// C `tanhf`.
    tanhf, f32, tanh, tanhf);
unary!(/// C `sqrtf`.
    sqrtf, f32, sqrt, sqrtf);
unary!(/// C `floorf`.
    floorf, f32, floor, floorf);
unary!(/// C `ceilf`.
    ceilf, f32, ceil, ceilf);

/// C `pow`.
#[inline(always)]
#[must_use]
pub fn pow(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::pow(x, y)
    }
}

/// C `powf`.
#[inline(always)]
#[must_use]
pub fn powf(x: f32, y: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::powf(x, y)
    }
}

/// C `atan2`.
#[inline(always)]
#[must_use]
pub fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2(y, x)
    }
}

/// C `fabs` (exact, no libm needed).
#[inline(always)]
#[must_use]
pub const fn fabs(x: f64) -> f64 {
    x.abs()
}

/// C `fabsf` (exact, no libm needed).
#[inline(always)]
#[must_use]
pub const fn fabsf(x: f32) -> f32 {
    x.abs()
}

/// C `lrintf` under the default rounding mode (round half to even), saturating like AArch64
/// `fcvtns`. NaN maps to 0.
#[inline(always)]
#[must_use]
pub fn lrintf(x: f32) -> i32 {
    #[cfg(feature = "std")]
    {
        x.round_ties_even() as i32
    }
    #[cfg(not(feature = "std"))]
    {
        libm::rintf(x) as i32
    }
}

/// C `lrint` (double) under the default rounding mode.
#[inline(always)]
#[must_use]
pub fn lrint(x: f64) -> i32 {
    #[cfg(feature = "std")]
    {
        x.round_ties_even() as i32
    }
    #[cfg(not(feature = "std"))]
    {
        libm::rint(x) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lrintf_ties_to_even() {
        assert_eq!(lrintf(0.5), 0);
        assert_eq!(lrintf(1.5), 2);
        assert_eq!(lrintf(2.5), 2);
        assert_eq!(lrintf(-0.5), 0);
        assert_eq!(lrintf(-1.5), -2);
        assert_eq!(lrintf(f32::NAN), 0);
        assert_eq!(lrintf(1e20), i32::MAX);
        assert_eq!(lrint(-2.5), -2);
    }

    #[test]
    fn basic_values() {
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(floorf(-1.5), -2.0);
        assert!((cos(0.0) - 1.0).abs() < 1e-15);
    }
}

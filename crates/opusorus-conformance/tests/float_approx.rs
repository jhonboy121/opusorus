//! Feature `float-approx` (libopus `FLOAT_APPROX`) vs the oracle built with the same define.
//!
//! * The float `celt_log2`, `celt_exp2` and `celt_isnan` against the oracle's over a sweep of
//!   every 97th `f32` bit pattern (all signs, exponents, denormals, infinities and NaNs) plus
//!   edge values: bit-exact. Without the feature both sides use the libm forms, with it the
//!   polynomial approximations.
//! * The codec-level effect (every `celt_log2`/`celt_exp2` user: band energies, dynalloc,
//!   analysis, PLC, gains, DNN features) is covered by running the other differential suites
//!   with `--features float-approx` (`just test-options`).
//!
//! Float build only: upstream's `FLOAT_APPROX` has no effect on a fixed-point build.
#![cfg(not(feature = "fixed-point"))]
#![allow(clippy::unwrap_used, reason = "test code")]

use opusorus::celt::arch::celt_isnan;
use opusorus::celt::mathops::{celt_exp2, celt_log2};
use opusorus_oracle::build_options as c;

#[test]
fn oracle_has_the_same_option() {
    assert_eq!(c::options().float_approx, cfg!(feature = "float-approx"));
}

/// Every 97th bit pattern (so every exponent, both signs, NaN payloads) plus edge values.
fn sweep() -> impl Iterator<Item = f32> {
    let edges = [
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        1.0,
        -1.0,
        2.0,
        0.5,
        1.0625,
        1.125,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        -50.0,
        -50.000_004,
        -49.999_996,
        -51.0,
        127.0,
        127.99999,
        128.0,
        -126.0,
        -127.0,
        -149.0,
        1e-8,
        1e-30,
        32.0,
    ];
    edges
        .into_iter()
        .chain((0..=u32::MAX / 97).map(|k| f32::from_bits(k * 97)))
}

#[test]
fn celt_isnan_matches_oracle() {
    for x in sweep() {
        assert_eq!(
            celt_isnan(x),
            c::celt_isnan(x),
            "celt_isnan({x:e}) [{:#x}]",
            x.to_bits()
        );
    }
}

#[test]
fn celt_log2_matches_oracle() {
    let mut n = 0u64;
    for x in sweep() {
        let (r, cv) = (celt_log2(x), c::celt_log2(x));
        // NaN results may differ in payload only if a libm is involved; compare bits exactly
        // for the approximation (pure bit manipulation and float arithmetic) and NaN-ness else.
        if cfg!(feature = "float-approx") || !cv.is_nan() {
            assert_eq!(
                r.to_bits(),
                cv.to_bits(),
                "celt_log2({x:e}) [{:#x}]: rust {r:e} c {cv:e}",
                x.to_bits()
            );
        } else {
            assert!(r.is_nan(), "celt_log2({x:e}): rust {r:e}, c NaN");
        }
        n += 1;
    }
    assert!(n > 44_000_000);
}

/// `celt_exp2` for every argument whose `floor` is representable as an `int`: C's `(int)floor(x)`
/// is undefined behaviour otherwise (x86 yields `INT_MIN`, AArch64 saturates like Rust).
#[test]
fn celt_exp2_matches_oracle() {
    let lim = 2_147_483_648.0f32;
    for x in sweep().filter(|x| !x.is_nan() && *x > -lim && *x < lim) {
        let (r, cv) = (celt_exp2(x), c::celt_exp2(x));
        assert_eq!(
            r.to_bits(),
            cv.to_bits(),
            "celt_exp2({x:e}) [{:#x}]: rust {r:e} c {cv:e}",
            x.to_bits()
        );
    }
}

/// The approximations are really in use (they differ from the libm forms).
#[cfg(feature = "float-approx")]
#[test]
fn approximations_are_active() {
    let libm_log2 = |x: f32| (std::f64::consts::LOG2_E * f64::from(x).ln()) as f32;
    let libm_exp2 = |x: f32| (std::f64::consts::LN_2 * f64::from(x)).exp() as f32;
    let differ = (1..1000)
        .map(|i| i as f32 * 0.37)
        .filter(|&x| celt_log2(x) != libm_log2(x) || celt_exp2(-x / 40.0) != libm_exp2(-x / 40.0))
        .count();
    assert!(differ > 500, "only {differ} of 999 values differ from libm");
    // Largest error of the approximations over the codec's range.
    for i in 1..10000 {
        let x = i as f32 * 0.01;
        assert!((celt_log2(x) - libm_log2(x)).abs() < 1e-4, "log2({x})");
        let y = -x / 4.0;
        assert!(
            (celt_exp2(y) / libm_exp2(y) - 1.0).abs() < 1e-5,
            "exp2({y})"
        );
    }
    assert_eq!(celt_exp2(-51.0), 0.0);
}

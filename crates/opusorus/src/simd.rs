//! Safe SIMD plumbing (`fearless_simd`) for the port's vertical SIMD kernels (PLAN D-031).
//!
//! Every kernel executes, per lane, exactly the scalar operation sequence of the port (same
//! operations, same order, no FMA contraction) on independent data, so results are
//! bit-identical to the scalar code and to the C oracle. The kernels run on the target's SIMD
//! baseline token (NEON on aarch64, SSE2 on x86/x86-64, SIMD128 on wasm32 built with
//! `+simd128`); on other targets [`with_simd!`] reports that no SIMD is available and the caller
//! runs the scalar code.

/// Runs `$body` with the target's baseline SIMD token bound to `$s` and evaluates to
/// `Some(result)`, or to `None` if the target has no supported SIMD baseline (the caller then
/// runs the scalar code).
#[allow(
    unused_macros,
    reason = "not every build configuration has SIMD kernels"
)]
macro_rules! with_simd {
    ($s:ident => $body:expr) => {{
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        let r = {
            #[cfg(target_arch = "aarch64")]
            let token = fearless_simd::Level::baseline().as_neon();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let token = fearless_simd::Level::baseline().as_sse2();
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            let token = fearless_simd::Level::baseline().as_wasm_simd128();
            #[cfg(test)]
            let token = token.filter(|_| !$crate::simd::scalar_only());
            token.map(|$s| {
                $s.vectorize(
                    #[inline(always)]
                    || $body,
                )
            })
        };
        // No SIMD: the caller runs the scalar code. The body is still type-checked (with the
        // scalar fallback token, never executed) so it stays warning-free on these targets.
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        let r = if false {
            let $s = fearless_simd::Fallback::new();
            Some($body)
        } else {
            None
        };
        r
    }};
}

/// The NEON token (aarch64 with NEON), for kernels that use NEON instructions directly through
/// `fearless_simd::kernel!`; `None` without NEON (and in tests while [`SCALAR_ONLY`] is set).
#[cfg(target_arch = "aarch64")]
#[allow(
    unused_macros,
    reason = "not every build configuration has NEON kernels"
)]
macro_rules! neon_token {
    () => {{
        let token = fearless_simd::Level::baseline().as_neon();
        #[cfg(test)]
        let token = token.filter(|_| !$crate::simd::scalar_only());
        token
    }};
}

/// [`with_simd!`] for kernels that do a prefix of the work: evaluates to the `$body` result
/// (how far the kernel got) with SIMD, and to `$none` (nothing done) without.
#[allow(
    unused_macros,
    reason = "not every build configuration has SIMD kernels"
)]
macro_rules! simd_done {
    ($none:expr; $s:ident => $body:expr) => {
        match with_simd!($s => $body) {
            Some(done) => done,
            None => $none,
        }
    };
}

/// Makes [`with_simd!`] report "no SIMD" (tests only), so self-tests can compare the SIMD and
/// scalar paths. Other tests running meanwhile are unaffected: both paths give the same results.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "unused in builds without SIMD kernels (fixed point)"
)]
pub(crate) static SCALAR_ONLY: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Current value of [`SCALAR_ONLY`].
#[cfg(test)]
#[allow(
    dead_code,
    reason = "unused in builds without SIMD kernels (fixed point)"
)]
pub(crate) fn scalar_only() -> bool {
    SCALAR_ONLY.load(core::sync::atomic::Ordering::Relaxed)
}

/// Runs `f` once with SIMD and once scalar-only and asserts both results are equal (tests only).
#[cfg(test)]
#[allow(
    dead_code,
    reason = "unused in builds without SIMD kernels (fixed point)"
)]
pub(crate) fn assert_simd_eq_scalar<T: PartialEq + core::fmt::Debug>(
    what: &str,
    mut f: impl FnMut() -> T,
) {
    use core::sync::atomic::Ordering;
    let simd = f();
    SCALAR_ONLY.store(true, Ordering::Relaxed);
    let scalar = f();
    SCALAR_ONLY.store(false, Ordering::Relaxed);
    assert_eq!(simd, scalar, "{what}");
}

/// `fast` builds (PLAN D-032): runs `f` with SIMD and scalar-only and asserts every SIMD output
/// is within `tol(i)` of the scalar reference (tests only).
#[cfg(all(test, feature = "fast"))]
#[allow(dead_code, reason = "unused in builds without fast kernels")]
pub(crate) fn assert_simd_close_scalar(
    what: &str,
    mut f: impl FnMut() -> alloc::vec::Vec<f32>,
    tol: impl Fn(usize) -> f32,
) {
    use core::sync::atomic::Ordering;
    let simd = f();
    SCALAR_ONLY.store(true, Ordering::Relaxed);
    let scalar = f();
    SCALAR_ONLY.store(false, Ordering::Relaxed);
    assert_eq!(simd.len(), scalar.len(), "{what}");
    for (i, (a, b)) in simd.iter().zip(&scalar).enumerate() {
        assert!(
            (a - b).abs() <= tol(i),
            "{what}: output {i}: fast {a} vs reference {b} (tolerance {})",
            tol(i)
        );
    }
}

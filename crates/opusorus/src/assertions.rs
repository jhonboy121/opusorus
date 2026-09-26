//! The libopus assertion macros (`celt/arch.h`, `silk/typedef.h`).
//!
//! libopus has three kinds of internal consistency checks: `celt_assert` / `celt_assert2`
//! (enabled by `ENABLE_ASSERTIONS` or `ENABLE_HARDENING`), `celt_sig_assert` (signal-range
//! checks, `ENABLE_ASSERTIONS` only) and `silk_assert` (`ENABLE_ASSERTIONS` only). A failed check
//! calls `celt_fatal` / `silk_fatal`, which prints the condition and aborts.
//!
//! In the port every such check is one of the macros below. By default they are
//! `debug_assert!`s: checked in debug and test builds, compiled out of release builds (the
//! port's hardening does not depend on them, safe Rust bounds-checks every access). With the
//! feature `assertions` (libopus `--enable-assertions`, `ENABLE_ASSERTIONS`) they are `assert!`s
//! in every build profile, so a failed check panics (and aborts with `panic = "abort"`) like C.
//!
//! Checks the port adds on its own (Rust-side invariants that C does not assert) stay plain
//! `debug_assert!`s. Upstream checks with no Rust counterpart: pointer, aliasing and `NULL`
//! checks (enforced by the type system), `arch` range checks (no run-time CPU dispatch), the
//! `OPUS_CHECK_ASM` comparisons (no SIMD) and compile-time constant relations (`const`
//! assertions in the port).

/// `celt_assert(cond)` / `celt_assert2(cond, message)`: see the module docs.
macro_rules! celt_assert {
    ($($arg:tt)*) => {{
        #[cfg(feature = "assertions")]
        assert!($($arg)*);
        #[cfg(not(feature = "assertions"))]
        debug_assert!($($arg)*);
    }};
}

/// `celt_sig_assert(cond)`: see the module docs.
macro_rules! celt_sig_assert {
    ($($arg:tt)*) => {{
        #[cfg(feature = "assertions")]
        assert!($($arg)*);
        #[cfg(not(feature = "assertions"))]
        debug_assert!($($arg)*);
    }};
}

/// `silk_assert(cond)`: see the module docs.
macro_rules! silk_assert {
    ($($arg:tt)*) => {{
        #[cfg(feature = "assertions")]
        assert!($($arg)*);
        #[cfg(not(feature = "assertions"))]
        debug_assert!($($arg)*);
    }};
}

/// `celt_assert(0)` / `silk_assert(0)` on a path the port keeps reachable in its default builds
/// (C returns an error right after it, or the check fails on inputs the unit tests use): a
/// panic with the feature `assertions`, like the C abort, and nothing otherwise (not even a
/// debug assertion).
macro_rules! assertion_failure {
    ($msg:literal) => {
        if cfg!(feature = "assertions") {
            panic!(concat!("assertion failed: ", $msg));
        }
    };
}

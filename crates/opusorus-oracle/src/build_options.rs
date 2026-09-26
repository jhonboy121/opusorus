//! Oracle bindings for the upstream build options (features `float-approx`, `assertions`,
//! `fuzzing`, `disable-rfc8251`: libopus `FLOAT_APPROX`, `ENABLE_ASSERTIONS`, `FUZZING`,
//! `DISABLE_UPDATE_DRAFT`), shims in `csrc/build_options.c`.

use core::ffi::c_int;

unsafe extern "C" {
    fn oracle_build_options() -> c_int;
    fn oracle_srand(seed: u32);
    fn oracle_rand() -> c_int;
    fn oracle_fail_celt_assert();
    fn oracle_fail_celt_sig_assert() -> c_int;
    fn oracle_fail_silk_assert();
}

#[cfg(not(feature = "fixed-point"))]
unsafe extern "C" {
    fn oracle_opt_celt_isnan(x: f32) -> c_int;
    fn oracle_opt_celt_log2(x: f32) -> f32;
    fn oracle_opt_celt_exp2(x: f32) -> f32;
}

/// The options the oracle was compiled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// `FLOAT_APPROX`.
    pub float_approx: bool,
    /// `ENABLE_ASSERTIONS`.
    pub assertions: bool,
    /// `FUZZING`.
    pub fuzzing: bool,
    /// `DISABLE_UPDATE_DRAFT`.
    pub disable_update_draft: bool,
}

/// The oracle's compile-time options.
#[must_use]
pub fn options() -> Options {
    // SAFETY: pure function without arguments.
    let v = unsafe { oracle_build_options() };
    Options {
        float_approx: v & 1 != 0,
        assertions: v & 2 != 0,
        fuzzing: v & 4 != 0,
        disable_update_draft: v & 8 != 0,
    }
}

/// C `srand(seed)`: seeds the C library generator the `FUZZING` encoder draws from.
pub fn srand(seed: u32) {
    // SAFETY: plain libc call.
    unsafe { oracle_srand(seed) }
}

/// C `rand()`.
#[must_use]
pub fn rand() -> i32 {
    // SAFETY: plain libc call.
    unsafe { oracle_rand() }
}

/// `celt_isnan(x)` of the float build.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub fn celt_isnan(x: f32) -> bool {
    // SAFETY: pure function of a float.
    unsafe { oracle_opt_celt_isnan(x) != 0 }
}

/// `celt_log2(x)` of the float build.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub fn celt_log2(x: f32) -> f32 {
    // SAFETY: pure function of a float.
    unsafe { oracle_opt_celt_log2(x) }
}

/// `celt_exp2(x)` of the float build.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub fn celt_exp2(x: f32) -> f32 {
    // SAFETY: pure function of a float.
    unsafe { oracle_opt_celt_exp2(x) }
}

/// Fails a `celt_assert` (`ec_enc_uint` with `ft == 1`): aborts the process when the oracle's
/// checks are enabled (`ENABLE_ASSERTIONS` or `ENABLE_HARDENING`).
pub fn fail_celt_assert() {
    // SAFETY: operates on a local buffer; the failed check aborts or is ignored.
    unsafe { oracle_fail_celt_assert() }
}

/// Fails a `celt_sig_assert` (float: `celt_atan2p_norm(-1, 1)`, fixed: `celt_ilog2(0)`):
/// aborts the process with `ENABLE_ASSERTIONS`, otherwise returns the function's value.
#[must_use]
pub fn fail_celt_sig_assert() -> i32 {
    // SAFETY: pure function; the failed check aborts or is ignored.
    unsafe { oracle_fail_celt_sig_assert() }
}

/// Fails a `silk_assert` (`silk_NLSF_stabilize` with `NDeltaMin_Q15[L] == 0`): aborts the
/// process with `ENABLE_ASSERTIONS`.
pub fn fail_silk_assert() {
    // SAFETY: operates on local arrays; the failed check aborts or is ignored.
    unsafe { oracle_fail_silk_assert() }
}

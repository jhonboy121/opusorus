//! Oracle bindings for the `fixed-point-debug` feature (libopus `FIXED_DEBUG`): the diagnostics
//! of `celt/fixed_debug.h` and `silk/MacroDebug.h` (captured per thread instead of printed), the
//! `celt_mips` operation counter and every checking macro by id. Shim: `csrc/fixed_debug.c`.

use core::ffi::{c_char, c_int};

unsafe extern "C" {
    fn oracle_fdbg_count() -> i64;
    fn oracle_fdbg_message(i: i64, buf: *mut c_char, cap: c_int) -> c_int;
    fn oracle_fdbg_clear();
    fn oracle_fdbg_celt_mips() -> i64;
    fn oracle_fdbg_set_celt_mips(v: i64);
    fn oracle_fdbg_celt(op: c_int, a: i64, b: i64, c: i64) -> i64;
    fn oracle_fdbg_silk(op: c_int, a: i64, b: i64, c: i64) -> i64;
}

/// Number of diagnostics the C checking macros printed on this thread since the last
/// [`clear`].
#[must_use]
pub fn count() -> i64 {
    // SAFETY: reads a thread-local counter.
    unsafe { oracle_fdbg_count() }
}

/// Number of diagnostics [`messages`] keeps (the first ones since the last [`clear`]).
pub const KEEP: i64 = 65536;

/// The diagnostics printed on this thread since the last [`clear`] (at most the first
/// [`KEEP`]; each ends with the C message's `\n`).
#[must_use]
pub fn messages() -> Vec<String> {
    let n = count().min(KEEP);
    (0..n)
        .map(|i| {
            let mut buf = [0u8; 256];
            // SAFETY: `buf` holds 256 bytes; the shim writes at most `cap` bytes (NUL
            // included) and returns the length or -1.
            let len = unsafe { oracle_fdbg_message(i, buf.as_mut_ptr().cast(), 256) };
            String::from_utf8_lossy(&buf[..len.max(0) as usize]).into_owned()
        })
        .collect()
}

/// Forgets the diagnostics captured on this thread.
pub fn clear() {
    // SAFETY: resets a thread-local counter.
    unsafe { oracle_fdbg_clear() }
}

/// libopus' global `celt_mips` operation counter (not thread-safe in C: callers serialize).
#[must_use]
pub fn celt_mips() -> i64 {
    // SAFETY: plain read of a C global.
    unsafe { oracle_fdbg_celt_mips() }
}

/// Sets `celt_mips`.
pub fn set_celt_mips(v: i64) {
    // SAFETY: plain write of a C global.
    unsafe { oracle_fdbg_set_celt_mips(v) }
}

/// Evaluates the `celt/fixed_debug.h` (or `arch.h`) macro `op` (ids in `csrc/fixed_debug.c`)
/// with arguments converted to the C parameter types.
#[must_use]
pub fn celt(op: i32, a: i64, b: i64, c: i64) -> i64 {
    // SAFETY: pure computation on values; ops that would be undefined in C are the caller's
    // responsibility (tests only pass defined inputs).
    unsafe { oracle_fdbg_celt(op, a, b, c) }
}

/// Evaluates the `silk/MacroDebug.h` macro `op` (ids in `csrc/fixed_debug.c`).
#[must_use]
pub fn silk(op: i32, a: i64, b: i64, c: i64) -> i64 {
    // SAFETY: as for [`celt`].
    unsafe { oracle_fdbg_silk(op, a, b, c) }
}

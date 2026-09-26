//! Port of `celt/arch.h` (+ `celt/fixed_generic.h` in fixed-point builds): numeric type aliases
//! and the arithmetic "macros" used throughout CELT.
//!
//! The float build (default) and the fixed-point build (feature `fixed-point`) define the *same
//! names* with different types and semantics, exactly like the C headers, so ported code that is
//! shared between the two builds is written once against these names:
//! * `float` (`celt/arch/float.rs`): every macro degenerates to float arithmetic,
//! * `fixed` (`celt/arch/fixed.rs`): `fixed_generic.h` integer semantics (see its module docs for
//!   the typing rules; `docs/FIXED_POINT.md` for the porting conventions).
//!
//! Naming: C macro `MULT16_32_Q15` → `mult16_32_q15`, etc. The items below are shared by both.

#[cfg(feature = "fixed-point")]
mod fixed;
#[cfg(feature = "fixed-point-debug")]
mod fixed_debug;
#[cfg(not(feature = "fixed-point"))]
mod float;

#[cfg(feature = "fixed-point")]
pub use fixed::*;
#[cfg(not(feature = "fixed-point"))]
pub use float::*;

/// `CELT_SIG_SCALE` (`32768.f` in both builds; the fixed-point build uses it for the float API).
pub const CELT_SIG_SCALE: f32 = 32768.0;
/// `DB_SHIFT`: Q format of `celt_glog` in the fixed-point build (kept for shared code).
pub const DB_SHIFT: i32 = 24;
/// `GLOBAL_STACK_SIZE`.
pub const GLOBAL_STACK_SIZE: usize = 120_000;

/// `OPUS_FAST_INT64`: whether libopus uses the 64-bit forms of the 32-bit fixed-point multiply
/// macros (`MULT16_32_Q15`, `MULT32_32_Q31`, ...). C: `__x86_64__ || __LP64__ || _WIN64 ||
/// __mips`. The 32-bit forms are *not* bit-identical for every macro (`MULT32_32_Q31`,
/// `MULT32_32_P31`, `MULT32_32_Q32` drop partial products), so the fixed-point port mirrors the
/// C selection per target to stay bit-exact with libopus built for the same platform.
pub const OPUS_FAST_INT64: bool = cfg!(any(
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
));

/// `IMIN` (C ternary semantics).
#[inline(always)]
#[must_use]
pub const fn imin(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
/// `IMAX` (C ternary semantics).
#[inline(always)]
#[must_use]
pub const fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
/// `IMUL32` (`(a)*(b)`; overflow is undefined in C, so it panics in debug builds).
#[inline(always)]
#[must_use]
pub const fn imul32(a: i32, b: i32) -> i32 {
    a * b
}
/// `UADD32` (`(a)+(b)` on unsigned operands: wraps). The checking `celt/fixed_debug.h`
/// version replaces it with `fixed-point-debug`.
#[cfg(not(feature = "fixed-point-debug"))]
#[inline(always)]
#[must_use]
pub const fn uadd32(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}
/// `USUB32` (`(a)-(b)` on unsigned operands: wraps). The checking `celt/fixed_debug.h`
/// version replaces it with `fixed-point-debug`.
#[cfg(not(feature = "fixed-point-debug"))]
#[inline(always)]
#[must_use]
pub const fn usub32(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b)
}

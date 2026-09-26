//! Port of `celt/fixed_debug.h` (feature `fixed-point-debug`, libopus `FIXED_DEBUG`): the
//! checking versions of the fixed-point macros, which replace the `celt/fixed_generic.h` ones
//! ([`super::fixed`]) of the same names.
//!
//! Every function computes what the C debug macro computes — which differs from the release
//! macro when an operand is out of the documented range (the debug macros take `int` /
//! `opus_int64` operands without truncating them to 16 bits) and for the 32-bit multiplies that
//! the debug header always builds from 16-bit partial products (`MULT16_32_Q16`,
//! `MULT32_32_Q31`, `MULT32_32_P31`, `MULT32_32_Q32`) or computes exactly in 64 bits
//! (`MULT16_32_Q15`, `MULT16_32_P16`, `MULT32_32_Q16`) — reports the C diagnostic through
//! [`crate::fixed_debug`] when a check fails (and continues, like libopus without
//! `FIXED_DEBUG_ASSERT`), and adds the C weight to `celt_mips`.
//!
//! The Rust signatures are those of the release functions (so the codec code is shared); a C
//! debug function returning `short` where the release macro is an `int` expression returns the
//! truncated value widened back to `i32` (C's promotion at the use). `__FILE__`/`__LINE__` in the
//! messages are the Rust call site (`#[track_caller]`). Signed overflow that is undefined in C
//! is computed in 64 bits and truncated (the value libopus produces in practice).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    reason = "the C conversions (int64 to int to short) are reproduced with `as`"
)]

use super::OpusVal16;
use crate::fixed_debug::{fdbg, mips};
use core::panic::Location;

/// `VERIFY_SHORT(x)`.
#[inline(always)]
const fn verify_short(x: i64) -> bool {
    x <= 32767 && x >= -32768
}
/// `VERIFY_INT(x)`.
#[inline(always)]
const fn verify_int(x: i64) -> bool {
    x <= 2_147_483_647 && x >= -2_147_483_648
}
/// `VERIFY_UINT(x)`: `(x)<=(2147483647LLU<<1)` (so `0xFFFFFFFF` fails, as in C).
#[inline(always)]
const fn verify_uint(x: u64) -> bool {
    x <= (2_147_483_647u64 << 1)
}

/// The C call site (`__FILE__`, `__LINE__`) of a checking macro.
#[inline(always)]
#[track_caller]
const fn site() -> (&'static str, u32) {
    let l = Location::caller();
    (l.file(), l.line())
}

// ---- operations without checks ----

/// `ADD32_ovflw`: add ignoring overflow (`celt_mips+=2`).
#[inline(always)]
#[must_use]
pub fn add32_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mips(2);
    a.into().wrapping_add(b.into())
}
/// `SUB32_ovflw`: subtract ignoring overflow (`celt_mips+=2`).
#[inline(always)]
#[must_use]
pub fn sub32_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mips(2);
    a.into().wrapping_sub(b.into())
}
/// `NEG32_ovflw`: negate ignoring overflow (`celt_mips+=2`).
#[inline(always)]
#[must_use]
pub fn neg32_ovflw(a: impl Into<i32>) -> i32 {
    mips(2);
    a.into().wrapping_neg()
}
/// `SHL32_ovflw`: `(opus_int32)((opus_uint32)(a)<<(shift))` (not counted).
#[inline(always)]
#[must_use]
pub fn shl32_ovflw(a: impl Into<i32>, shift: i32) -> i32 {
    ((a.into() as u32) << shift) as i32
}
/// `PSHR32_ovflw`: `SHR32(ADD32_ovflw(a, (EXTEND32(1)<<(shift)>>1)),shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn pshr32_ovflw(a: impl Into<i32>, shift: i32) -> i32 {
    shr32(add32_ovflw(a, (extend32(1) << shift) >> 1), shift)
}

// ---- checked 16/32-bit arithmetic ----

/// `NEG16(int x)`: returns a `short` (widened back to `int` at the use).
#[inline(always)]
#[must_use]
pub fn neg16(x: impl Into<i32>) -> i32 {
    let x = x.into();
    if !verify_short(i64::from(x)) {
        fdbg!("NEG16: input is not short: {}\n", x);
    }
    let res = x.wrapping_neg();
    if !verify_short(i64::from(res)) {
        fdbg!("NEG16: output is not short: {}\n", res);
    }
    mips(1);
    i32::from(res as i16)
}
/// `NEG32(opus_int64 x)` (its messages say "NEG16", as in C).
#[inline(always)]
#[must_use]
pub fn neg32(x: impl Into<i32>) -> i32 {
    let x = i64::from(x.into());
    if !verify_int(x) {
        fdbg!("NEG16: input is not int: {}\n", x as i32);
    }
    let res = -x;
    if !verify_int(res) {
        fdbg!("NEG16: output is not int: {}\n", res as i32);
    }
    mips(2);
    res as i32
}
/// `EXTRACT16(int x)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn extract16(x: impl Into<i32>) -> OpusVal16 {
    let x = x.into();
    if !verify_short(i64::from(x)) {
        let (file, line) = site();
        fdbg!("EXTRACT16: input is not short: {x} in {file}: line {line}\n");
    }
    mips(1);
    x as i16
}
/// `EXTEND32(int x)`: also checks that `x` is a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn extend32(x: impl Into<i32>) -> i32 {
    let x = x.into();
    if !verify_short(i64::from(x)) {
        let (file, line) = site();
        fdbg!("EXTEND32: input is not short: {x} in {file}: line {line}\n");
    }
    mips(1);
    x
}
/// `SHR16(int a, int shift)`: returns a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn shr16(a: impl Into<i32>, shift: i32) -> i32 {
    let a = a.into();
    if !verify_short(i64::from(a)) || !verify_short(i64::from(shift)) {
        let (file, line) = site();
        fdbg!("SHR16: inputs are not short: {a} >> {shift} in {file}: line {line}\n");
    }
    let res = a >> shift;
    if !verify_short(i64::from(res)) {
        let (file, line) = site();
        fdbg!("SHR16: output is not short: {res} in {file}: line {line}\n");
    }
    mips(1);
    i32::from(res as i16)
}
/// `SHL16(int a, int shift)`: `(opus_int32)((opus_uint32)a<<shift)`, returns a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn shl16(a: impl Into<i32>, shift: i32) -> OpusVal16 {
    let a = a.into();
    if !verify_short(i64::from(a)) || !verify_short(i64::from(shift)) {
        let (file, line) = site();
        fdbg!("SHL16: inputs are not short: {a} {shift} in {file}: line {line}\n");
    }
    let res = ((a as u32) << shift) as i32;
    if !verify_short(i64::from(res)) {
        let (file, line) = site();
        fdbg!("SHL16: output is not short: {res} in {file}: line {line}\n");
    }
    mips(1);
    res as i16
}
/// `SHR32(opus_int64 a, int shift)`.
#[inline(always)]
#[must_use]
pub fn shr32(a: impl Into<i32>, shift: i32) -> i32 {
    let a = i64::from(a.into());
    if !verify_int(a) || !verify_short(i64::from(shift)) {
        fdbg!("SHR32: inputs are not int: {} {}\n", a as i32, shift);
    }
    let res = a >> shift;
    if !verify_int(res) {
        fdbg!("SHR32: output is not int: {}\n", res as i32);
    }
    mips(2);
    res as i32
}
/// `SHL32(opus_int64 a, int shift)`: `(opus_int64)((opus_uint64)a<<shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn shl32(a: impl Into<i32>, shift: i32) -> i32 {
    let a = i64::from(a.into());
    if !verify_int(a) || !verify_short(i64::from(shift)) {
        let (file, line) = site();
        fdbg!("SHL32: inputs are not int: {a} {shift} in {file}: line {line}\n");
    }
    let res = ((a as u64) << shift) as i64;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!("SHL32: output is not int: {a}<<{shift} = {res} in {file}: line {line}\n");
    }
    mips(2);
    res as i32
}
/// `SHR(a,b)`: `SHR32(a,b)` in the debug header.
#[inline(always)]
#[must_use]
pub fn shr(a: impl Into<i32>, shift: i32) -> i32 {
    shr32(a, shift)
}
/// `PSHR32(a,shift)`: `(celt_mips--,SHR32(ADD32((a),(((opus_val32)(1)<<((shift))>>1))),shift))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn pshr32(a: impl Into<i32>, shift: i32) -> i32 {
    mips(-1);
    shr32(add32(a, (1i32 << shift) >> 1), shift)
}
/// `VSHR32(a, shift)`: `((shift)>0) ? SHR32(a, shift) : SHL32(a, -(shift))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn vshr32(a: impl Into<i32>, shift: i32) -> i32 {
    if shift > 0 {
        shr32(a, shift)
    } else {
        shl32(a, -shift)
    }
}
/// `SHR64(a,shift)`: `(celt_mips++,(a) >> (shift))`.
#[inline(always)]
#[must_use]
pub fn shr64(a: i64, shift: i32) -> i64 {
    mips(1);
    a >> shift
}
/// `ROUND16(x,a)`: `(celt_mips--,EXTRACT16(PSHR32((x),(a))))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn round16(x: impl Into<i32>, a: i32) -> OpusVal16 {
    mips(-1);
    extract16(pshr32(x, a))
}
/// `SROUND16(x,a)`: `(celt_mips--,EXTRACT16(SATURATE(PSHR32(x,a), 32767)))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn sround16(x: impl Into<i32>, a: i32) -> OpusVal16 {
    mips(-1);
    extract16(saturate(pshr32(x, a), 32767))
}
/// `HALF16(x)`: `SHR16(x,1)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn half16(x: impl Into<i32>) -> i32 {
    shr16(x, 1)
}
/// `HALF32(x)`: `SHR32(x,1)`.
#[inline(always)]
#[must_use]
pub fn half32(x: impl Into<i32>) -> i32 {
    shr32(x, 1)
}
/// `ADD16(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn add16(a: impl Into<i32>, b: impl Into<i32>) -> OpusVal16 {
    let (a, b) = (a.into(), b.into());
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        let (file, line) = site();
        fdbg!("ADD16: inputs are not short: {a} {b} in {file}: line {line}\n");
    }
    let res = a.wrapping_add(b);
    if !verify_short(i64::from(res)) {
        let (file, line) = site();
        fdbg!("ADD16: output is not short: {a}+{b}={res} in {file}: line {line}\n");
    }
    mips(1);
    res as i16
}
/// `SUB16(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn sub16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        let (file, line) = site();
        fdbg!("SUB16: inputs are not short: {a} {b} in {file}: line {line}\n");
    }
    let res = a.wrapping_sub(b);
    if !verify_short(i64::from(res)) {
        let (file, line) = site();
        fdbg!("SUB16: output is not short: {res} in {file}: line {line}\n");
    }
    mips(1);
    i32::from(res as i16)
}
/// `ADD32(opus_int64 a, opus_int64 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn add32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if !verify_int(a) || !verify_int(b) {
        let (file, line) = site();
        fdbg!(
            "ADD32: inputs are not int: {} {} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
    }
    let res = a + b;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "ADD32: output is not int: {} in {file}: line {line}\n",
            res as i32
        );
    }
    mips(2);
    res as i32
}
/// `SUB32(opus_int64 a, opus_int64 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn sub32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if !verify_int(a) || !verify_int(b) {
        let (file, line) = site();
        fdbg!(
            "SUB32: inputs are not int: {} {} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
    }
    let res = a - b;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "SUB32: output is not int: {} in {file}: line {line}\n",
            res as i32
        );
    }
    mips(2);
    res as i32
}
/// `UADD32(opus_uint64 a, opus_uint64 b)` (`#undef`s the `arch.h` one).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn uadd32(a: u32, b: u32) -> u32 {
    let (a, b) = (u64::from(a), u64::from(b));
    if !verify_uint(a) || !verify_uint(b) {
        let (file, line) = site();
        fdbg!("UADD32: inputs are not uint32: {a} {b} in {file}: line {line}\n");
    }
    let res = a + b;
    if !verify_uint(res) {
        let (file, line) = site();
        fdbg!("UADD32: output is not uint32: {res} in {file}: line {line}\n");
    }
    mips(2);
    res as u32
}
/// `USUB32(opus_uint64 a, opus_uint64 b)` (`#undef`s the `arch.h` one).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn usub32(a: u32, b: u32) -> u32 {
    let (a, b) = (u64::from(a), u64::from(b));
    if !verify_uint(a) || !verify_uint(b) {
        let (file, line) = site();
        fdbg!("USUB32: inputs are not uint32: {a} {b} in {file}: line {line}\n");
    }
    if a < b {
        let (file, line) = site();
        fdbg!("USUB32: inputs underflow: {a} < {b} in {file}: line {line}\n");
    }
    let res = a.wrapping_sub(b);
    if !verify_uint(res) {
        let (file, line) = site();
        fdbg!("USUB32: output is not uint32: {a} - {b} = {res} in {file}: line {line}\n");
    }
    mips(2);
    res as u32
}

// ---- multiplications ----

/// `MULT16_16_16(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        fdbg!("MULT16_16_16: inputs are not short: {a} {b}\n");
    }
    let res = a.wrapping_mul(b);
    if !verify_short(i64::from(res)) {
        fdbg!("MULT16_16_16: output is not short: {res}\n");
    }
    mips(1);
    i32::from(res as i16)
}
/// `MULT32_32_32(opus_int64 a, opus_int64 b)`.
#[inline(always)]
#[must_use]
pub fn mult32_32_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if !verify_int(a) || !verify_int(b) {
        fdbg!("MULT32_32_32: inputs are not int: {a} {b}\n");
    }
    let res = a * b;
    if !verify_int(res) {
        fdbg!("MULT32_32_32: output is not int: {res}\n");
    }
    mips(5);
    res as i32
}
/// `MULT32_32_Q16(opus_int64 a, opus_int64 b)`: the exact 64-bit product shifted.
#[inline(always)]
#[must_use]
pub fn mult32_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if !verify_int(a) || !verify_int(b) {
        fdbg!("MULT32_32_Q16: inputs are not int: {a} {b}\n");
    }
    let res = (a * b) >> 16;
    if !verify_int(res) {
        fdbg!("MULT32_32_Q16: output is not int: {a}*{b}={res}\n");
    }
    mips(5);
    res as i32
}
/// `MULT32_32_Q16` for products that do not fit 32 bits (not a libopus macro; where the release
/// port uses it, C calls `MULT32_32_Q16`, whose debug version already wraps and reports).
#[inline(always)]
#[must_use]
pub fn mult32_32_q16_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult32_32_q16(a, b)
}
/// `MULT16_16(int a, int b)`: no truncation of the operands to 16 bits.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult16_16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        let (file, line) = site();
        fdbg!("MULT16_16: inputs are not short: {a} {b} in {file}: line {line}\n");
    }
    let res = i64::from(a) * i64::from(b);
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "MULT16_16: output is not int: {} in {file}: line {line}\n",
            res as i32
        );
    }
    mips(1);
    res as i32
}
/// `MAC16_16(c,a,b)`: `(celt_mips-=2,ADD32((c),MULT16_16((a),(b))))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mac16_16(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mips(-2);
    add32(c, mult16_16(a, b))
}
/// `MULT16_32_QX(int a, opus_int64 b, int Q)`.
#[inline(always)]
#[track_caller]
fn mult16_32_qx(a: i32, b: i64, q: i32) -> i32 {
    if !verify_short(i64::from(a)) || !verify_int(b) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_Q{q}: inputs are not short+int: {a} {} in {file}: line {line}\n",
            b as i32
        );
    }
    if b.abs() >= 1i64 << (16 + q) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_Q{q}: second operand too large: {a} {} in {file}: line {line}\n",
            b as i32
        );
    }
    let res = (i64::from(a) * b) >> q;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_Q{q}: output is not int: {a}*{}={} in {file}: line {line}\n",
            b as i32,
            res as i32
        );
    }
    mips(if q == 15 { 3 } else { 4 });
    res as i32
}
/// `MULT16_32_PX(int a, opus_int64 b, int Q)` (its messages end with two newlines, as in C).
#[inline(always)]
#[track_caller]
fn mult16_32_px(a: i32, b: i64, q: i32) -> i32 {
    if !verify_short(i64::from(a)) || !verify_int(b) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_P{q}: inputs are not short+int: {a} {} in {file}: line {line}\n\n",
            b as i32
        );
    }
    if b.abs() >= 1i64 << (16 + q) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_Q{q}: second operand too large: {a} {} in {file}: line {line}\n\n",
            b as i32
        );
    }
    let res = (i64::from(a) * b + i64::from((1i32 << q) >> 1)) >> q;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "MULT16_32_P{q}: output is not int: {a}*{}={} in {file}: line {line}\n\n",
            b as i32,
            res as i32
        );
    }
    mips(if q == 15 { 4 } else { 5 });
    res as i32
}
/// `MULT16_32_Q15(a,b)`: `MULT16_32_QX(a,b,15)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult16_32_q15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult16_32_qx(a.into(), i64::from(b.into()), 15)
}
/// `MULT16_32_P16(a,b)`: `MULT16_32_PX(a,b,16)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult16_32_p16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult16_32_px(a.into(), i64::from(b.into()), 16)
}
/// `MULT16_32_Q16(a,b)`: `ADD32(MULT16_16((a),SHR32((b),16)),
/// SHR32(MULT16_16SU((a),((b)&0x0000ffff)),16))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult16_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    add32(
        mult16_16(a, shr32(b, 16)),
        shr32(super::fixed::mult16_16su(a, b & 0x0000ffff), 16),
    )
}
/// `MAC16_32_Q15(c,a,b)`: `(celt_mips-=2,ADD32((c),MULT16_32_Q15((a),(b))))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mac16_32_q15(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mips(-2);
    add32(c, mult16_32_q15(a, b))
}
/// `MAC16_32_Q16(c,a,b)`: `(celt_mips-=2,ADD32((c),MULT16_32_Q16((a),(b))))`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mac16_32_q16(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mips(-2);
    add32(c, mult16_32_q16(a, b))
}
/// `MULT32_32_Q31(a,b)`: always the 16-bit partial-product form in the debug header.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult32_32_q31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    use super::fixed::mult16_16su;
    let (a, b) = (a.into(), b.into());
    add32(
        add32(
            shl32(mult16_16(shr32(a, 16), shr32(b, 16)), 1),
            shr32(mult16_16su(shr32(a, 16), b & 0x0000ffff), 15),
        ),
        shr32(mult16_16su(shr32(b, 16), a & 0x0000ffff), 15),
    )
}
/// The rounding term of the debug `MULT32_32_P31`: `SHR32(128+(opus_int32)(MULT16_16U(((a)&
/// 0x0000ffff),((b)&0x0000ffff))>>(16+7)) + SHR32(MULT16_16SU(SHR((a),16),((b)&0x0000ffff)),7)
/// + SHR32(MULT16_16SU(SHR((b),16),((a)&0x0000ffff)),7), 8)` (the inner `+` are plain C).
#[inline(always)]
fn p31_low(a: i32, b: i32) -> i32 {
    use super::fixed::{mult16_16su, mult16_16u};
    let lo = (mult16_16u((a & 0x0000ffff) as u32, (b & 0x0000ffff) as u32) >> (16 + 7)) as i32;
    shr32(
        128 + lo
            + shr32(mult16_16su(shr(a, 16), b & 0x0000ffff), 7)
            + shr32(mult16_16su(shr(b, 16), a & 0x0000ffff), 7),
        8,
    )
}
/// `MULT32_32_P31(a,b)`: the 16-bit partial-product form.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult32_32_p31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    add32(shl32(mult16_16(shr(a, 16), shr(b, 16)), 1), p31_low(a, b))
}
/// `MULT32_32_P31_ovflw(a,b)`: as [`mult32_32_p31`] with a wrapping final add.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult32_32_p31_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    add32_ovflw(shl32(mult16_16(shr(a, 16), shr(b, 16)), 1), p31_low(a, b))
}
/// `MULT32_32_Q32(a,b)`: the 16-bit partial-product form.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult32_32_q32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    use super::fixed::mult16_16su;
    let (a, b) = (a.into(), b.into());
    add32(
        add32(
            mult16_16(shr(a, 16), shr(b, 16)),
            shr(mult16_16su(shr(a, 16), b & 0x0000ffff), 16),
        ),
        shr(mult16_16su(shr(b, 16), a & 0x0000ffff), 16),
    )
}
/// `SATURATE(int a, int b)`.
#[inline(always)]
#[must_use]
pub fn saturate(x: impl Into<i32>, a: impl Into<i32>) -> i32 {
    let (mut x, a) = (x.into(), a.into());
    if x > a {
        x = a;
    }
    if x < -a {
        x = -a;
    }
    mips(3);
    x
}
/// `SATURATE16(opus_int32 a)`.
#[inline(always)]
#[must_use]
pub fn saturate16(x: impl Into<i32>) -> OpusVal16 {
    let x = x.into();
    mips(3);
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x as i16
    }
}

/// The `MULT16_16_Qx` / `MULT16_16_Px` debug functions: `(opus_int64)a*b`, plus `round`, then
/// `>> q`, checked. `out16`: the output must be a `short` (the function returns one); `name`:
/// the macro name in the messages.
#[inline(always)]
fn mult16_16_shift(a: i32, b: i32, q: i32, round: i64, out16: bool, name: &str) -> i32 {
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        fdbg!("{name}: inputs are not short: {a} {b}\n");
    }
    let mut res = i64::from(a) * i64::from(b);
    if round != 0 {
        res += round;
        if !verify_int(res) {
            fdbg!("{name}: overflow: {a}*{b}={}\n", res as i32);
        }
    }
    res >>= q;
    if out16 {
        if !verify_short(res) {
            if round != 0 || q == 13 {
                fdbg!("{name}: output is not short: {a}*{b}={}\n", res as i32);
            } else {
                fdbg!("{name}: output is not short: {}\n", res as i32);
            }
        }
        i32::from(res as i16)
    } else {
        if !verify_int(res) {
            // MULT16_16_Q11_32 (its message says "short" and "Q11", as in C).
            fdbg!(
                "MULT16_16_Q11: output is not short: {a}*{b}={}\n",
                res as i32
            );
        }
        res as i32
    }
}
/// `MULT16_16_Q11_32(int a, int b)` (its messages say `MULT16_16_Q11`).
#[inline(always)]
#[must_use]
pub fn mult16_16_q11_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 11, 0, false, "MULT16_16_Q11");
    mips(3);
    r
}
/// `MULT16_16_Q13(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q13(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 13, 0, true, "MULT16_16_Q13");
    mips(3);
    r
}
/// `MULT16_16_Q14(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q14(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 14, 0, true, "MULT16_16_Q14");
    mips(3);
    r
}
/// `MULT16_16_Q15(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn mult16_16_q15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    if !verify_short(i64::from(a)) || !verify_short(i64::from(b)) {
        let (file, line) = site();
        fdbg!("MULT16_16_Q15: inputs are not short: {a} {b} in {file}: line {line}\n");
    }
    let res = (i64::from(a) * i64::from(b)) >> 15;
    if !verify_short(res) {
        let (file, line) = site();
        fdbg!(
            "MULT16_16_Q15: output is not short: {} in {file}: line {line}\n",
            res as i32
        );
    }
    mips(1);
    i32::from(res as i16)
}
/// `MULT16_16_P13(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p13(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 13, 4096, true, "MULT16_16_P13");
    mips(4);
    r
}
/// `MULT16_16_P14(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p14(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 14, 8192, true, "MULT16_16_P14");
    mips(4);
    r
}
/// `MULT16_16_P15(int a, int b)`: returns a `short`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let r = mult16_16_shift(a.into(), b.into(), 15, 16384, true, "MULT16_16_P15");
    mips(2);
    r
}

// ---- divisions ----

/// `DIV32_16(opus_int64 a, opus_int64 b)`: 0 for a zero divisor, the quotient saturated to 16
/// bits when it does not fit (both reported).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn div32_16(a: impl Into<i32>, b: impl Into<i32>) -> OpusVal16 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if b == 0 {
        let (file, line) = site();
        fdbg!(
            "DIV32_16: divide by zero: {}/{} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
        return 0;
    }
    if !verify_int(a) || !verify_short(b) {
        let (file, line) = site();
        fdbg!(
            "DIV32_16: inputs are not int/short: {} {} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
    }
    let mut res = a / b;
    if !verify_short(res) {
        let (file, line) = site();
        fdbg!(
            "DIV32_16: output is not short: {} / {} = {} in {file}: line {line}\n",
            a as i32,
            b as i32,
            res as i32
        );
        res = res.clamp(-32768, 32767);
    }
    mips(35);
    res as i16
}
/// `DIV32(opus_int64 a, opus_int64 b)`: 0 for a zero divisor (reported).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn div32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (i64::from(a.into()), i64::from(b.into()));
    if b == 0 {
        let (file, line) = site();
        fdbg!(
            "DIV32: divide by zero: {}/{} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
        return 0;
    }
    if !verify_int(a) || !verify_int(b) {
        let (file, line) = site();
        fdbg!(
            "DIV32: inputs are not int/short: {} {} in {file}: line {line}\n",
            a as i32,
            b as i32
        );
    }
    let res = a / b;
    if !verify_int(res) {
        let (file, line) = site();
        fdbg!(
            "DIV32: output is not int: {} in {file}: line {line}\n",
            res as i32
        );
    }
    mips(70);
    res as i32
}

// ---- arch.h macros whose release port is `const` ----

/// `RES2INT16` (`SAT16(PSHR32(a, RES_SHIFT))`).
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
#[track_caller]
pub fn res2int16(a: super::OpusRes) -> i16 {
    super::fixed::sat16(pshr32(a, super::RES_SHIFT))
}
/// `COEF2VAL16` (`EXTRACT16(SHR32(x, 16))`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
#[track_caller]
pub fn coef2val16(x: super::CeltCoef) -> OpusVal16 {
    extract16(shr32(x, 16))
}

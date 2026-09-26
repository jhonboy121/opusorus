//! Port of `silk/MacroDebug.h` (feature `fixed-point-debug`, libopus `FIXED_DEBUG`): the
//! checking versions of the SILK macros, which replace the ones of [`super::macros`] with the
//! same names.
//!
//! Each function computes what the C debug function computes (with the C parameter types, e.g.
//! `silk_ADD_SAT16` truncates its second operand to 16 bits and `silk_ADD_LSHIFT` /
//! `silk_ADD_RSHIFT` return a 16-bit value), reports the C message through
//! [`crate::fixed_debug`] when its check fails, and continues (signed overflow, undefined in C,
//! wraps). The Rust signatures are those of the release functions. `__FILE__` / `__LINE__` are
//! the Rust call site (`#[track_caller]`). The misspelled C messages (`silk_RSHITF32`) are kept.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    reason = "the C conversions are reproduced with `as`"
)]

use super::macros::{SILK_INT16_MAX, SILK_INT16_MIN, silk_abs, silk_sat16, silk_sat32};
use crate::fixed_debug::fdbg;
use core::panic::Location;

/// `__FILE__`, `__LINE__` of the calling macro.
#[inline(always)]
#[track_caller]
const fn site() -> (&'static str, u32) {
    let l = Location::caller();
    (l.file(), l.line())
}

/// Reports `silk_<name>(<args>) in <file>: line <line>` (the format of every message here).
macro_rules! fail {
    ($fmt:literal $(, $arg:expr)*) => {{
        let (file, line) = site();
        fdbg!(concat!($fmt, " in {}: line {}\n") $(, $arg)*, file, line);
    }};
}

/// `silk_ADD16(opus_int16 a, opus_int16 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add16(a: i16, b: i16) -> i16 {
    let ret = a.wrapping_add(b);
    if ret != silk_add_sat16(a, i32::from(b)) {
        fail!("silk_ADD16({}, {})", a, b);
    }
    ret
}
/// `silk_ADD32(opus_int32 a, opus_int32 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add32(a: i32, b: i32) -> i32 {
    let ret = a.wrapping_add(b);
    if ret != silk_add_sat32(a, b) {
        fail!("silk_ADD32({}, {})", a, b);
    }
    ret
}
/// `silk_ADD64(opus_int64 a, opus_int64 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add64(a: i64, b: i64) -> i64 {
    let ret = a.wrapping_add(b);
    if ret != silk_add_sat64(a, b) {
        fail!("silk_ADD64({}, {})", a, b);
    }
    ret
}
/// `silk_SUB16(opus_int16 a, opus_int16 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub16(a: i16, b: i16) -> i16 {
    let ret = a.wrapping_sub(b);
    if ret != silk_sub_sat16(a, i32::from(b)) {
        fail!("silk_SUB16({}, {})", a, b);
    }
    ret
}
/// `silk_SUB32(opus_int32 a, opus_int32 b)` (`ret` is 64-bit, returned truncated).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub32(a: i32, b: i32) -> i32 {
    let ret = i64::from(a) - i64::from(b);
    if ret != i64::from(silk_sub_sat32(a, b)) {
        fail!("silk_SUB32({}, {})", a, b);
    }
    ret as i32
}
/// `silk_SUB64(opus_int64 a, opus_int64 b)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub64(a: i64, b: i64) -> i64 {
    let ret = a.wrapping_sub(b);
    if ret != silk_sub_sat64(a, b) {
        fail!("silk_SUB64({}, {})", a, b);
    }
    ret
}
/// `silk_ADD_SAT16(opus_int16 a16, opus_int16 b16)`: `b` is truncated to 16 bits.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_sat16(a: i16, b: i32) -> i16 {
    let b16 = b as i16;
    let res = silk_sat16(silk_add32(i32::from(a), i32::from(b16))) as i16;
    if i32::from(res) != silk_sat16(i32::from(a) + i32::from(b16)) {
        fail!("silk_ADD_SAT16({}, {})", a, b16);
    }
    res
}
/// `silk_ADD_SAT32(opus_int32 a32, opus_int32 b32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_sat32(a32: i32, b32: i32) -> i32 {
    let sum = (a32 as u32).wrapping_add(b32 as u32);
    let res = if sum & 0x8000_0000 == 0 {
        if (a32 & b32) as u32 & 0x8000_0000 != 0 {
            i32::MIN
        } else {
            a32.wrapping_add(b32)
        }
    } else if (a32 | b32) as u32 & 0x8000_0000 == 0 {
        i32::MAX
    } else {
        a32.wrapping_add(b32)
    };
    if res != silk_sat32(i64::from(a32) + i64::from(b32)) {
        fail!("silk_ADD_SAT32({}, {})", a32, b32);
    }
    res
}
/// `silk_ADD_SAT64(opus_int64 a64, opus_int64 b64)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_sat64(a64: i64, b64: i64) -> i64 {
    let sum = a64.wrapping_add(b64);
    let res = if sum as u64 & 0x8000_0000_0000_0000 == 0 {
        if (a64 & b64) as u64 & 0x8000_0000_0000_0000 != 0 {
            i64::MIN
        } else {
            sum
        }
    } else if (a64 | b64) as u64 & 0x8000_0000_0000_0000 == 0 {
        i64::MAX
    } else {
        sum
    };
    let fail = if res != sum {
        // Check that we saturated to the correct extreme value
        !((res == i64::MAX && (a64 >> 1) + (b64 >> 1) > (i64::MAX >> 3))
            || (res == i64::MIN && (a64 >> 1) + (b64 >> 1) < (i64::MIN >> 3)))
    } else {
        // Saturation not necessary
        res != sum
    };
    if fail {
        fail!("silk_ADD_SAT64({}, {})", a64, b64);
    }
    res
}
/// `silk_SUB_SAT16(opus_int16 a16, opus_int16 b16)`: `b` is truncated to 16 bits.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub_sat16(a: i16, b: i32) -> i16 {
    let b16 = b as i16;
    let res = silk_sat16(silk_sub32(i32::from(a), i32::from(b16))) as i16;
    if i32::from(res) != silk_sat16(i32::from(a) - i32::from(b16)) {
        fail!("silk_SUB_SAT16({}, {})", a, b16);
    }
    res
}
/// `silk_SUB_SAT32(opus_int32 a32, opus_int32 b32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub_sat32(a32: i32, b32: i32) -> i32 {
    let diff = (a32 as u32).wrapping_sub(b32 as u32);
    let res = if diff & 0x8000_0000 == 0 {
        if (a32 as u32) & ((b32 as u32) ^ 0x8000_0000) & 0x8000_0000 != 0 {
            i32::MIN
        } else {
            a32.wrapping_sub(b32)
        }
    } else if ((a32 as u32) ^ 0x8000_0000) & (b32 as u32) & 0x8000_0000 != 0 {
        i32::MAX
    } else {
        a32.wrapping_sub(b32)
    };
    if res != silk_sat32(i64::from(a32) - i64::from(b32)) {
        fail!("silk_SUB_SAT32({}, {})", a32, b32);
    }
    res
}
/// `silk_SUB_SAT64(opus_int64 a64, opus_int64 b64)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub_sat64(a64: i64, b64: i64) -> i64 {
    const S: u64 = 0x8000_0000_0000_0000;
    let diff = a64.wrapping_sub(b64);
    let res = if diff as u64 & S == 0 {
        if (a64 as u64) & ((b64 as u64) ^ S) & S != 0 {
            i64::MIN
        } else {
            diff
        }
    } else if ((a64 as u64) ^ S) & (b64 as u64) & S != 0 {
        i64::MAX
    } else {
        diff
    };
    let fail = if res != diff {
        // Check that we saturated to the correct extreme value
        !((res == i64::MAX && (a64 >> 1) + (b64 >> 1) > (i64::MAX >> 3))
            || (res == i64::MIN && (a64 >> 1) + (b64 >> 1) < (i64::MIN >> 3)))
    } else {
        // Saturation not necessary
        res != diff
    };
    if fail {
        fail!("silk_SUB_SAT64({}, {})", a64, b64);
    }
    res
}
/// `silk_MUL(opus_int32 a32, opus_int32 b32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_mul(a32: i32, b32: i32) -> i32 {
    let ret = (a32 as u32).wrapping_mul(b32 as u32) as i32;
    let ret64 = i64::from(a32) * i64::from(b32);
    if i64::from(ret) != ret64 {
        fail!("silk_MUL({}, {})", a32, b32);
    }
    ret
}
/// `silk_MUL_uint(opus_uint32 a32, opus_uint32 b32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_mul_uint(a32: u32, b32: u32) -> u32 {
    let ret = a32.wrapping_mul(b32);
    if u64::from(ret) != u64::from(a32) * u64::from(b32) {
        fail!("silk_MUL_uint({}, {})", a32, b32);
    }
    ret
}
/// `silk_MLA(opus_int32 a32, opus_int32 b32, opus_int32 c32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_mla(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32.wrapping_add(b32.wrapping_mul(c32));
    if i64::from(ret) != i64::from(a32) + i64::from(b32) * i64::from(c32) {
        fail!("silk_MLA({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_MLA_uint(opus_uint32 a32, opus_uint32 b32, opus_uint32 c32)` (the message prints the
/// operands with `%d`, as in C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_mla_uint(a32: u32, b32: u32, c32: u32) -> u32 {
    let ret = a32.wrapping_add(b32.wrapping_mul(c32));
    // C: (opus_int64)a32 + (opus_int64)b32 * (opus_int64)c32 (the product can exceed int64).
    let exact = i64::from(a32).wrapping_add(i64::from(b32).wrapping_mul(i64::from(c32)));
    if i64::from(ret) != exact {
        fail!(
            "silk_MLA_uint({}, {}, {})",
            a32 as i32,
            b32 as i32,
            c32 as i32
        );
    }
    ret
}
/// `silk_SMULWB(opus_int32 a32, opus_int32 b32)`: the 16-bit partial-product form (equal to
/// the release 64-bit form).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smulwb(a32: i32, b32: i32) -> i32 {
    let b16 = i32::from(b32 as i16);
    let ret = (a32 >> 16) * b16 + (((a32 & 0x0000_FFFF) * b16) >> 16);
    if i64::from(ret) != (i64::from(a32) * i64::from(b16)) >> 16 {
        fail!("silk_SMULWB({}, {})", a32, b32);
    }
    ret
}
/// `silk_SMLAWB(opus_int32 a32, opus_int32 b32, opus_int32 c32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlawb(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32.wrapping_add(silk_smulwb(b32, c32));
    if ret != silk_add_sat32(a32, silk_smulwb(b32, c32)) {
        fail!("silk_SMLAWB({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_SMULWT(opus_int32 a32, opus_int32 b32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smulwt(a32: i32, b32: i32) -> i32 {
    let ret = (a32 >> 16) * (b32 >> 16) + (((a32 & 0x0000_FFFF) * (b32 >> 16)) >> 16);
    if i64::from(ret) != (i64::from(a32) * i64::from(b32 >> 16)) >> 16 {
        fail!("silk_SMULWT({}, {})", a32, b32);
    }
    ret
}
/// `silk_SMLAWT(opus_int32 a32, opus_int32 b32, opus_int32 c32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlawt(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32
        .wrapping_add((b32 >> 16) * (c32 >> 16))
        .wrapping_add(((b32 & 0x0000_FFFF) * (c32 >> 16)) >> 16);
    if i64::from(ret) != i64::from(a32) + ((i64::from(b32) * i64::from(c32 >> 16)) >> 16) {
        fail!("silk_SMLAWT({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_SMULL(opus_int64 a64, opus_int64 b64)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smull(a32: i32, b32: i32) -> i64 {
    let (a64, b64) = (i64::from(a32), i64::from(b32));
    let ret64 = a64.wrapping_mul(b64);
    let fail = if b64 != 0 {
        a64 != ret64 / b64
    } else if a64 != 0 {
        b64 != ret64 / a64
    } else {
        false
    };
    if fail {
        fail!("silk_SMULL({}, {})", a64, b64);
    }
    ret64
}
/// `silk_SMLABB(opus_int32 a32, opus_int32 b32, opus_int32 c32)` (the check does not truncate
/// `b32`, as in C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlabb(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32.wrapping_add(i32::from(b32 as i16) * i32::from(c32 as i16));
    if i64::from(ret) != i64::from(a32) + i64::from(b32) * i64::from(c32 as i16) {
        fail!("silk_SMLABB({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_SMLABT(opus_int32 a32, opus_int32 b32, opus_int32 c32)` (the check does not truncate
/// `b32`, as in C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlabt(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32.wrapping_add(i32::from(b32 as i16) * (c32 >> 16));
    if i64::from(ret) != i64::from(a32) + i64::from(b32) * i64::from(c32 >> 16) {
        fail!("silk_SMLABT({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_SMLATT(opus_int32 a32, opus_int32 b32, opus_int32 c32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlatt(a32: i32, b32: i32, c32: i32) -> i32 {
    let ret = a32.wrapping_add((b32 >> 16) * (c32 >> 16));
    if i64::from(ret) != i64::from(a32) + i64::from((b32 >> 16) * (c32 >> 16)) {
        fail!("silk_SMLATT({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_SMULWW(opus_int32 a32, opus_int32 b32)`: built from `silk_SMULWB`, `silk_RSHIFT_ROUND`,
/// `silk_MUL` and `silk_ADD32` (each checking itself), as in C.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smulww(a32: i32, b32: i32) -> i32 {
    let mut fail = false;
    let mut ret = silk_smulwb(a32, b32);
    let mut tmp1 = silk_rshift_round(b32, 16);
    let tmp2 = silk_mul(a32, tmp1);

    fail |= i64::from(tmp2) != i64::from(a32) * i64::from(tmp1);

    tmp1 = ret;
    ret = silk_add32(tmp1, tmp2);
    fail |= silk_add32(tmp1, tmp2) != silk_add_sat32(tmp1, tmp2);

    let ret64 = silk_rshift64(silk_smull(a32, b32), 16);
    fail |= i64::from(ret) != ret64;

    if fail {
        fail!("silk_SMULWW({}, {})", a32, b32);
    }
    ret
}
/// `silk_SMLAWW(opus_int32 a32, opus_int32 b32, opus_int32 c32)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_smlaww(a32: i32, b32: i32, c32: i32) -> i32 {
    let tmp = silk_smulww(b32, c32);
    let ret = silk_add32(a32, tmp);
    if ret != silk_add_sat32(a32, tmp) {
        fail!("silk_SMLAWW({}, {}, {})", a32, b32, c32);
    }
    ret
}
/// `silk_DIV32(opus_int32 a32, opus_int32 b32)`: reports a zero divisor (then divides, like C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_div32(a32: i32, b32: i32) -> i32 {
    if b32 == 0 {
        fail!("silk_DIV32({}, {})", a32, b32);
    }
    a32 / b32
}
/// `silk_DIV32_16(opus_int32 a32, opus_int32 b32)`: reports a zero or non-16-bit divisor.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_div32_16(a32: i32, b16: i32) -> i32 {
    let fail = b16 == 0 || !(SILK_INT16_MIN..=SILK_INT16_MAX).contains(&b16);
    if fail {
        fail!("silk_DIV32_16({}, {})", a32, b16);
    }
    a32 / b16
}
/// `silk_LSHIFT8(opus_int8 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift8(a: i8, shift: u32) -> i8 {
    // C: `(opus_int8)((opus_uint8)a << shift)`, the shift on the promoted `int`.
    let ret = (u32::from(a as u8) << shift) as i8;
    let fail = shift >= 8 || i64::from(ret) != ((a as u64) << shift) as i64;
    if fail {
        fail!("silk_LSHIFT8({}, {})", a, shift);
    }
    ret
}
/// `silk_LSHIFT16(opus_int16 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift16(a: i16, shift: u32) -> i16 {
    // C: `(opus_int16)((opus_uint16)a << shift)`, the shift on the promoted `int`.
    let ret = (u32::from(a as u16) << shift) as i16;
    let fail = shift >= 16 || i64::from(ret) != ((a as u64) << shift) as i64;
    if fail {
        fail!("silk_LSHIFT16({}, {})", a, shift);
    }
    ret
}
/// `silk_LSHIFT32(opus_int32 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift32(a: i32, shift: i32) -> i32 {
    let ret = ((a as u32) << shift) as i32;
    let fail = !(0..32).contains(&shift) || i64::from(ret) != ((a as u64) << shift) as i64;
    if fail {
        fail!("silk_LSHIFT32({}, {})", a, shift);
    }
    ret
}
/// `silk_LSHIFT(a, shift)`: `silk_LSHIFT32`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift(a: i32, shift: i32) -> i32 {
    silk_lshift32(a, shift)
}
/// `silk_LSHIFT64(opus_int64 a, opus_int shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift64(a: i64, shift: i32) -> i64 {
    let ret = ((a as u64) << shift) as i64;
    let fail = !(0..64).contains(&shift) || (ret >> shift) != a;
    if fail {
        fail!("silk_LSHIFT64({}, {})", a, shift);
    }
    ret
}
/// `silk_LSHIFT_ovflw(opus_int32 a, opus_int32 shift)`: only the shift is checked.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift_ovflw(a: i32, shift: i32) -> i32 {
    if !(0..32).contains(&shift) {
        // no check for overflow
        fail!("silk_LSHIFT_ovflw({}, {})", a, shift);
    }
    ((a as u32) << shift) as i32
}
/// `silk_LSHIFT_uint(opus_uint32 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_lshift_uint(a: u32, shift: i32) -> u32 {
    let ret = a << shift;
    if shift < 0 || i64::from(ret) != i64::from(a) << shift {
        fail!("silk_LSHIFT_uint({}, {})", a, shift);
    }
    ret
}
/// `silk_RSHIFT32(opus_int32 a, opus_int32 shift)` (message: `silk_RSHITF32`, as in C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift32(a: i32, shift: i32) -> i32 {
    if !(0..32).contains(&shift) {
        fail!("silk_RSHITF32({}, {})", a, shift);
    }
    a >> shift
}
/// `silk_RSHIFT(a, shift)`: `silk_RSHIFT32`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift(a: i32, shift: i32) -> i32 {
    silk_rshift32(a, shift)
}
/// `silk_RSHIFT64(opus_int64 a, opus_int64 shift)` (message: `silk_RSHITF64`, as in C).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift64(a: i64, shift: i32) -> i64 {
    if !(0..64).contains(&shift) {
        fail!("silk_RSHITF64({}, {})", a, i64::from(shift));
    }
    a >> shift
}
/// `silk_RSHIFT_uint(opus_uint32 a, opus_int32 shift)` (a shift of 32 passes the check).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift_uint(a: u32, shift: i32) -> u32 {
    if !(0..=32).contains(&shift) {
        fail!("silk_RSHIFT_uint({}, {})", a, shift);
    }
    a >> shift
}
/// `silk_ADD_LSHIFT(int a, int b, int shift)`: the result is a 16-bit value (`opus_int16 ret`).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_lshift(a: i32, b: i32, shift: i32) -> i32 {
    // C: `ret = a + (opus_int16)((opus_uint16)b << shift)` into an `opus_int16`.
    let ret = a.wrapping_add(i32::from((u32::from(b as u16) << shift) as i16)) as i16;
    if !(0..=15).contains(&shift) || i64::from(ret) != i64::from(a) + ((b as u64) << shift) as i64 {
        fail!("silk_ADD_LSHIFT({}, {}, {})", a, b, shift);
    }
    i32::from(ret) // shift >= 0
}
/// `silk_ADD_LSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    let ret = a.wrapping_add(((b as u32) << shift) as i32);
    if !(0..=31).contains(&shift) || i64::from(ret) != i64::from(a) + ((b as u64) << shift) as i64 {
        fail!("silk_ADD_LSHIFT32({}, {}, {})", a, b, shift);
    }
    ret // shift >= 0
}
/// `silk_ADD_LSHIFT_uint(opus_uint32 a, opus_uint32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_lshift_uint(a: u32, b: u32, shift: i32) -> u32 {
    let ret = a.wrapping_add(b << shift);
    // C: `(opus_int64)a + (((opus_int64)b) << shift)` (overflows, undefined, for large `b`).
    if !(0..=32).contains(&shift)
        || i64::from(ret) != i64::from(a).wrapping_add(i64::from(b) << shift)
    {
        fail!("silk_ADD_LSHIFT_uint({}, {}, {})", a, b, shift);
    }
    ret // shift >= 0
}
/// `silk_ADD_RSHIFT(int a, int b, int shift)`: the result is a 16-bit value (`opus_int16 ret`).
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_rshift(a: i32, b: i32, shift: i32) -> i32 {
    let ret = a.wrapping_add(b >> shift) as i16;
    if !(0..=15).contains(&shift) || i64::from(ret) != i64::from(a) + (i64::from(b) >> shift) {
        fail!("silk_ADD_RSHIFT({}, {}, {})", a, b, shift);
    }
    i32::from(ret) // shift > 0
}
/// `silk_ADD_RSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    let ret = a.wrapping_add(b >> shift);
    if !(0..=31).contains(&shift) || i64::from(ret) != i64::from(a) + (i64::from(b) >> shift) {
        fail!("silk_ADD_RSHIFT32({}, {}, {})", a, b, shift);
    }
    ret // shift > 0
}
/// `silk_ADD_RSHIFT_uint(opus_uint32 a, opus_uint32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_add_rshift_uint(a: u32, b: u32, shift: i32) -> u32 {
    let ret = a.wrapping_add(b >> shift);
    if !(0..=32).contains(&shift) || i64::from(ret) != i64::from(a) + (i64::from(b) >> shift) {
        fail!("silk_ADD_RSHIFT_uint({}, {}, {})", a, b, shift);
    }
    ret // shift > 0
}
/// `silk_SUB_LSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    let ret = a.wrapping_sub(((b as u32) << shift) as i32);
    if !(0..=31).contains(&shift) || i64::from(ret) != i64::from(a) - ((b as u64) << shift) as i64 {
        fail!("silk_SUB_LSHIFT32({}, {}, {})", a, b, shift);
    }
    ret // shift >= 0
}
/// `silk_SUB_RSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_sub_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    let ret = a.wrapping_sub(b >> shift);
    if !(0..=31).contains(&shift) || i64::from(ret) != i64::from(a) - (i64::from(b) >> shift) {
        fail!("silk_SUB_RSHIFT32({}, {}, {})", a, b, shift);
    }
    ret // shift > 0
}
/// `silk_RSHIFT_ROUND(opus_int32 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    let ret = if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    };
    // the macro definition can't handle a shift of zero
    if shift <= 0 || shift > 31 || i64::from(ret) != (i64::from(a) + (1i64 << (shift - 1))) >> shift
    {
        fail!("silk_RSHIFT_ROUND({}, {})", a, shift);
    }
    ret
}
/// `silk_RSHIFT_ROUND64(opus_int64 a, opus_int32 shift)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    // the macro definition can't handle a shift of zero
    if shift <= 0 || shift >= 64 {
        fail!("silk_RSHIFT_ROUND64({}, {})", a, shift);
    }
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}
/// `silk_abs_int64(opus_int64 a)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_abs_int64(a: i64) -> i64 {
    if a == i64::MIN {
        fail!("silk_abs_int64({})", a);
    }
    if a > 0 { a } else { a.wrapping_neg() } // Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN
}
/// `silk_abs_int32(opus_int32 a)`: `silk_abs(a)` after the check.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_abs_int32(a: i32) -> i32 {
    if a == i32::MIN {
        fail!("silk_abs_int32({})", a);
    }
    silk_abs(a)
}
/// `silk_CHECK_FIT8(opus_int64 a)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_check_fit8(a: i64) -> i8 {
    let ret = a as i8;
    if i64::from(ret) != a {
        fail!("silk_CHECK_FIT8({})", a);
    }
    ret
}
/// `silk_CHECK_FIT16(opus_int64 a)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_check_fit16(a: i64) -> i16 {
    let ret = a as i16;
    if i64::from(ret) != a {
        fail!("silk_CHECK_FIT16({})", a);
    }
    ret
}
/// `silk_CHECK_FIT32(opus_int64 a)`.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn silk_check_fit32(a: i64) -> i32 {
    let ret = a as i32;
    if i64::from(ret) != a {
        fail!("silk_CHECK_FIT32({})", a);
    }
    ret
}

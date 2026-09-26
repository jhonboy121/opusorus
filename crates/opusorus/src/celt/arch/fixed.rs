//! Port of `celt/arch.h` (`#ifdef FIXED_POINT` part) and `celt/fixed_generic.h`: the
//! fixed-point types and arithmetic macros (feature `fixed-point`; `ENABLE_RES24` with
//! `fixed-res24`, `ENABLE_QEXT` with `qext`).
//!
//! # Typing rules (see also `docs/FIXED_POINT.md`)
//!
//! The C macros are untyped: their arguments are whatever integer expression the caller wrote
//! (after C's integer promotions, an `int`). To stay bit-exact with the oracle every function
//! here mirrors the *release* expansion in `fixed_generic.h` exactly:
//! * integer arguments are `impl Into<i32>` (accepting `i16`, `i32`, `u8`, `u16`, ...), and every
//!   cast the macro performs is reproduced — e.g. `MULT16_16(a,b)` truncates both arguments to
//!   16 bits (`(opus_val16)(a)`), exactly like C. 64-bit or unsigned 32-bit values must be
//!   converted explicitly by the caller (they need an explicit decision in C too);
//! * the return type is the C type of the expansion: `int` (→ `i32`) unless the macro itself casts
//!   (`EXTRACT16`, `ADD16`, `SHL16`, `ROUND16`, `DIV32_16`, `SATURATE16`, `QCONST16`, ... return
//!   `i16`). Where C narrows implicitly (assigning an `int` to an `opus_val16` variable, passing it
//!   to an `opus_val16` parameter, storing it in an `opus_val16` array), the port writes
//!   [`extract16`], which is the identity in the float build;
//! * operations that are undefined on overflow in C (`ADD32`, `SUB32`, `MULT32_32_32`, `PSHR32`,
//!   `NEG32`, ...) use plain Rust arithmetic, so debug/test builds panic where C would be
//!   undefined; the `*_ovflw` macros and unsigned casts wrap. Out-of-range integer conversions
//!   (implementation-defined in C, modular on every supported compiler) use `as`.
//!
//! `MIN16`/`MAX16`/`MIN32`/`MAX32`/`MING`/`MAXG` are generic over one operand type: C's result
//! type is the promoted type of both operands, so mixed-width calls must widen explicitly
//! ([`extend32`]).
//!
//! `OPUS_FAST_INT64` selects the 64-bit or the 32-bit forms of the 32-bit multiplies per target
//! ([`super::OPUS_FAST_INT64`]); both forms are available as [`int64`] / [`int32`] for tests.

use super::{DB_SHIFT, OPUS_FAST_INT64};
use crate::celt::mathops::float2int;
#[cfg(not(feature = "fixed-res24"))]
use crate::celt::mathops::float2int16;
#[cfg(feature = "fixed-res24")]
use crate::celt::mathops::float2int24;

/// `opus_val16`.
pub type OpusVal16 = i16;
/// `opus_val32`.
pub type OpusVal32 = i32;
/// `opus_val64`.
pub type OpusVal64 = i64;
/// `celt_sig`: internal CELT signal, Q27 (`SIG_SHIFT` = 12 above 16-bit PCM).
pub type CeltSig = i32;
/// `celt_norm`: normalised band coefficients, Q`NORM_SHIFT` (Q24).
pub type CeltNorm = i32;
/// `celt_ener`.
pub type CeltEner = i32;
/// `celt_glog`: log-domain energy, Q`DB_SHIFT` (Q24).
pub type CeltGlog = i32;
/// `opus_res`: Opus internal resolution — 24-bit (`ENABLE_RES24`).
#[cfg(feature = "fixed-res24")]
pub type OpusRes = i32;
/// `opus_res`: Opus internal resolution — 16-bit PCM.
#[cfg(not(feature = "fixed-res24"))]
pub type OpusRes = i16;
/// `celt_coef`: window/twiddle/filter coefficients — Q31 with QEXT.
#[cfg(feature = "qext")]
pub type CeltCoef = i32;
/// `celt_coef`: window/twiddle/filter coefficients — Q15 without QEXT.
#[cfg(not(feature = "qext"))]
pub type CeltCoef = i16;

/// `RES_SHIFT`: shift between `opus_res` and 16-bit PCM.
#[cfg(feature = "fixed-res24")]
pub const RES_SHIFT: i32 = 8;
/// `RES_SHIFT`: shift between `opus_res` and 16-bit PCM.
#[cfg(not(feature = "fixed-res24"))]
pub const RES_SHIFT: i32 = 0;
/// `MAX_ENCODING_DEPTH`.
#[cfg(feature = "fixed-res24")]
pub const MAX_ENCODING_DEPTH: i32 = 24;
/// `MAX_ENCODING_DEPTH`.
#[cfg(not(feature = "fixed-res24"))]
pub const MAX_ENCODING_DEPTH: i32 = 16;

/// `NORM_SHIFT`.
pub const NORM_SHIFT: i32 = 24;
/// `Q15ONE` (C: the `int` 32767).
pub const Q15ONE: OpusVal16 = 32767;
/// `Q31ONE` (C: the `int` 2147483647).
pub const Q31ONE: OpusVal32 = 2_147_483_647;
/// `COEF_ONE`.
#[cfg(feature = "qext")]
pub const COEF_ONE: CeltCoef = Q31ONE;
/// `COEF_ONE`.
#[cfg(not(feature = "qext"))]
pub const COEF_ONE: CeltCoef = Q15ONE;
/// `SIG_SHIFT`.
pub const SIG_SHIFT: i32 = 12;
/// `SIG_SAT`: safe saturation value for 32-bit signals (2^29-1).
pub const SIG_SAT: CeltSig = 536_870_911;
/// `NORM_SCALING` (`1<<NORM_SHIFT`).
pub const NORM_SCALING: CeltNorm = 1 << NORM_SHIFT;
/// `EPSILON`.
pub const EPSILON: OpusVal32 = 1;
/// `VERY_SMALL`.
pub const VERY_SMALL: OpusVal32 = 0;
/// `VERY_LARGE16`.
pub const VERY_LARGE16: OpusVal16 = 32767;
/// `Q15_ONE`.
pub const Q15_ONE: OpusVal16 = 32767;

/// `celt_isnan` (always 0 in fixed point).
#[inline(always)]
#[must_use]
pub const fn celt_isnan(_x: i32) -> bool {
    false
}

/// `(opus_val16)(x)` of an `int`.
#[inline(always)]
const fn w16(x: i32) -> i32 {
    x as i16 as i32
}

// ---- arch.h ----

/// `MIN16` (C: `(a) < (b) ? (a) : (b)`).
#[inline(always)]
#[must_use]
pub fn min16<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}
/// `MAX16` (C: `(a) > (b) ? (a) : (b)`).
#[inline(always)]
#[must_use]
pub fn max16<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
/// `MIN32`.
#[inline(always)]
#[must_use]
pub fn min32<T: PartialOrd>(a: T, b: T) -> T {
    min16(a, b)
}
/// `MAX32`.
#[inline(always)]
#[must_use]
pub fn max32<T: PartialOrd>(a: T, b: T) -> T {
    max16(a, b)
}
/// `FMIN` (float arguments in C, kept generic).
#[inline(always)]
#[must_use]
pub fn fmin<T: PartialOrd>(a: T, b: T) -> T {
    min16(a, b)
}
/// `FMAX`.
#[inline(always)]
#[must_use]
pub fn fmax<T: PartialOrd>(a: T, b: T) -> T {
    max16(a, b)
}
/// `MING` (`MIN32`).
#[inline(always)]
#[must_use]
pub fn ming<T: PartialOrd>(a: T, b: T) -> T {
    min16(a, b)
}
/// `MAXG` (`MAX32`).
#[inline(always)]
#[must_use]
pub fn maxg<T: PartialOrd>(a: T, b: T) -> T {
    max16(a, b)
}

/// `ABS16` (C: `((x) < 0 ? (-(x)) : (x))`, an `int`).
#[inline(always)]
#[must_use]
pub fn abs16(x: impl Into<i32>) -> i32 {
    let x = x.into();
    if x < 0 { -x } else { x }
}
/// `ABS32`.
#[inline(always)]
#[must_use]
pub fn abs32(x: impl Into<i32>) -> i32 {
    abs16(x)
}
/// `SAT16`.
#[inline(always)]
#[must_use]
pub const fn sat16(x: i32) -> i16 {
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x as i16
    }
}

// ---- opus_res / celt_sig conversions (arch.h) ----

/// `SIG2RES`.
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub fn sig2res(a: CeltSig) -> OpusRes {
    pshr32(a, SIG_SHIFT - RES_SHIFT)
}
/// `SIG2RES` (`SIG2WORD16`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub fn sig2res(a: CeltSig) -> OpusRes {
    sig2word16(a)
}
/// `RES2INT16`.
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub const fn res2int16(a: OpusRes) -> i16 {
    // `SAT16(PSHR32(a, RES_SHIFT))`, with `PSHR32` written out so that this stays `const`.
    sat16((a + ((1 << RES_SHIFT) >> 1)) >> RES_SHIFT)
}
/// `RES2INT16`.
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub const fn res2int16(a: OpusRes) -> i16 {
    a
}
/// `RES2INT24`.
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub const fn res2int24(a: OpusRes) -> i32 {
    a
}
/// `RES2INT24` (`SHL32(EXTEND32(a), 8)`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub fn res2int24(a: OpusRes) -> i32 {
    shl32(a, 8)
}
/// `RES2FLOAT` (C: `(1.f/32768.f/256.f)*(a)`).
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub const fn res2float(a: OpusRes) -> f32 {
    (1.0 / 32768.0 / 256.0) * a as f32
}
/// `RES2FLOAT` (C: `(1.f/32768.f)*(a)`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub const fn res2float(a: OpusRes) -> f32 {
    (1.0 / 32768.0) * a as f32
}
/// `INT16TORES`.
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub fn int16tores(a: i16) -> OpusRes {
    shl32(a, RES_SHIFT)
}
/// `INT16TORES`.
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub const fn int16tores(a: i16) -> OpusRes {
    a
}
/// `INT24TORES`.
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub const fn int24tores(a: i32) -> OpusRes {
    a
}
/// `INT24TORES` (`SAT16(PSHR32(a, 8))`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub fn int24tores(a: i32) -> OpusRes {
    sat16(pshr32(a, 8))
}
/// `ADD_RES` (`ADD32`).
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub fn add_res(a: OpusRes, b: OpusRes) -> OpusRes {
    add32(a, b)
}
/// `ADD_RES` (`SAT16(ADD32(a, b))`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub fn add_res(a: OpusRes, b: OpusRes) -> OpusRes {
    sat16(add32(a, b))
}
/// `FLOAT2RES` (`FLOAT2INT24`).
#[cfg(feature = "fixed-res24")]
#[inline(always)]
#[must_use]
pub fn float2res(a: f32) -> OpusRes {
    float2int24(a)
}
/// `FLOAT2RES` (`FLOAT2INT16`).
#[cfg(not(feature = "fixed-res24"))]
#[inline(always)]
#[must_use]
pub fn float2res(a: f32) -> OpusRes {
    float2int16(a)
}
/// `RES2SIG`.
#[inline(always)]
#[must_use]
pub fn res2sig(a: OpusRes) -> CeltSig {
    shl32(a, SIG_SHIFT - RES_SHIFT)
}
/// `MULT16_RES_Q15`: `MULT16_32_Q15` (res24) / `MULT16_16_Q15` (res16). Every C use assigns the
/// result to an `opus_res`, so this returns that type (the implicit C narrowing included).
#[inline(always)]
#[must_use]
pub fn mult16_res_q15(a: impl Into<i32>, b: OpusRes) -> OpusRes {
    #[cfg(feature = "fixed-res24")]
    {
        mult16_32_q15(a, b)
    }
    #[cfg(not(feature = "fixed-res24"))]
    {
        mult16_16_q15(a, b) as OpusRes
    }
}
/// `RES2VAL16` (`RES2INT16`).
#[inline(always)]
#[must_use]
pub const fn res2val16(a: OpusRes) -> OpusVal16 {
    res2int16(a)
}
/// `INT16TOSIG` (`SHL32(EXTEND32(a), SIG_SHIFT)`).
#[inline(always)]
#[must_use]
pub fn int16tosig(a: i16) -> CeltSig {
    shl32(a, SIG_SHIFT)
}
/// `INT24TOSIG` (`SHL32(a, SIG_SHIFT-8)`).
#[inline(always)]
#[must_use]
pub fn int24tosig(a: i32) -> CeltSig {
    shl32(a, SIG_SHIFT - 8)
}
/// `FLOAT2SIG` (`celt/float_cast.h`, fixed-point build; float API).
#[inline(always)]
#[must_use]
pub fn float2sig(mut x: f32) -> CeltSig {
    // C: `x*((opus_int32)32768<<SIG_SHIFT)`, then clamps against the `int`s ±65536<<SIG_SHIFT
    // (converted to float).
    x *= (32768i32 << SIG_SHIFT) as f32;
    x = max32(x, -(65536i32 << SIG_SHIFT) as f32);
    x = min32(x, (65536i32 << SIG_SHIFT) as f32);
    float2int(x)
}

// ---- celt_coef (arch.h, QEXT-dependent) ----

/// `MULT_COEF_32` (`MULT32_32_P31`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
pub fn mult_coef_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult32_32_p31(a, b)
}
/// `MULT_COEF_32` (`MULT16_32_Q15`).
#[cfg(not(feature = "qext"))]
#[inline(always)]
#[must_use]
pub fn mult_coef_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult16_32_q15(a, b)
}
/// `MAC_COEF_32_ARM` (`ADD32((c), MULT32_32_Q32(a,b))`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
pub fn mac_coef_32_arm(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    add32(c, mult32_32_q32(a, b))
}
/// `MAC_COEF_32_ARM` (`MAC16_32_Q16`).
#[cfg(not(feature = "qext"))]
#[inline(always)]
#[must_use]
pub fn mac_coef_32_arm(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mac16_32_q16(c, a, b)
}
/// `MULT_COEF` (`MULT32_32_Q31`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
pub fn mult_coef(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult32_32_q31(a, b)
}
/// `MULT_COEF` (`MULT16_16_Q15`).
#[cfg(not(feature = "qext"))]
#[inline(always)]
#[must_use]
pub fn mult_coef(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult16_16_q15(a, b)
}
/// `MULT_COEF_TAPS` (`SHL32(MULT16_16(a,b), 1)`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
pub fn mult_coef_taps(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shl32(mult16_16(a, b), 1)
}
/// `MULT_COEF_TAPS` (`MULT16_16_P15`).
#[cfg(not(feature = "qext"))]
#[inline(always)]
#[must_use]
pub fn mult_coef_taps(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    mult16_16_p15(a, b)
}
/// `COEF2VAL16` (`EXTRACT16(SHR32(x, 16))`).
#[cfg(feature = "qext")]
#[inline(always)]
#[must_use]
pub const fn coef2val16(x: CeltCoef) -> OpusVal16 {
    (x >> 16) as i16
}
/// `COEF2VAL16` (identity).
#[cfg(not(feature = "qext"))]
#[inline(always)]
#[must_use]
pub const fn coef2val16(x: CeltCoef) -> OpusVal16 {
    x
}

// ---- fixed_generic.h ----

/// `MULT16_16SU`: signed 16-bit × unsigned 16-bit (`(opus_val32)(opus_val16)(a)*(opus_val32)
/// (opus_uint16)(b)`).
#[inline(always)]
#[must_use]
pub fn mult16_16su(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    w16(a.into()) * (b.into() as u16 as i32)
}
/// `MULT16_16U`: `(opus_uint32)(a)*(opus_uint32)(b)` (unsigned, wraps).
#[inline(always)]
#[must_use]
pub const fn mult16_16u(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b)
}

/// The `OPUS_FAST_INT64` (64-bit) forms of the 32-bit multiply macros.
pub mod int64 {
    /// `MULT16_32_Q16`: `(opus_val32)SHR((opus_int64)((opus_val16)(a))*(b),16)`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i16 as i64 * b.into() as i64) >> 16) as i32
    }
    /// `MULT16_32_P16`: `(opus_val32)PSHR((opus_int64)((opus_val16)(a))*(b),16)`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_p16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i16 as i64 * b.into() as i64 + 32768) >> 16) as i32
    }
    /// `MULT16_32_Q15`: `(opus_val32)SHR((opus_int64)((opus_val16)(a))*(b),15)`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_q15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i16 as i64 * b.into() as i64) >> 15) as i32
    }
    /// `MULT32_32_Q16`: `(opus_val32)SHR((opus_int64)(a)*(opus_int64)(b),16)`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i64 * b.into() as i64) >> 16) as i32
    }
    /// `MULT32_32_Q31`: `(opus_val32)SHR((opus_int64)(a)*(opus_int64)(b),31)`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i64 * b.into() as i64) >> 31) as i32
    }
    /// `MULT32_32_P31`: `(opus_val32)SHR(1073741824+(opus_int64)(a)*(opus_int64)(b),31)`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_p31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((1_073_741_824 + a.into() as i64 * b.into() as i64) >> 31) as i32
    }
    /// `MULT32_32_P31_ovflw` (= `MULT32_32_P31` in the 64-bit form).
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_p31_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        mult32_32_p31(a, b)
    }
    /// `MULT32_32_Q32`: `(opus_val32)SHR((opus_int64)(a)*(opus_int64)(b),32)`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        ((a.into() as i64 * b.into() as i64) >> 32) as i32
    }
}

/// The 32-bit (`!OPUS_FAST_INT64`) forms of the 32-bit multiply macros, built from 16-bit
/// partial products. `MULT32_32_Q31`, `MULT32_32_P31(_ovflw)` and `MULT32_32_Q32` are *not*
/// bit-identical to the [`int64`] forms.
pub mod int32 {
    use super::{add32, add32_ovflw, mult16_16, mult16_16su, mult16_16u, pshr, shl, shr, shr32};

    /// `MULT16_32_Q16`: `ADD32(MULT16_16((a),SHR((b),16)), SHR(MULT16_16SU((a),((b)&0x0000ffff)),
    /// 16))`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(
            mult16_16(a, shr(b, 16)),
            shr(mult16_16su(a, b & 0x0000ffff), 16),
        )
    }
    /// `MULT16_32_P16`: `ADD32(MULT16_16((a),SHR((b),16)), PSHR(MULT16_16SU((a),((b)&0x0000ffff)),
    /// 16))`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_p16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(
            mult16_16(a, shr(b, 16)),
            pshr(mult16_16su(a, b & 0x0000ffff), 16),
        )
    }
    /// `MULT16_32_Q15`: `ADD32(SHL(MULT16_16((a),SHR((b),16)),1), SHR(MULT16_16SU((a),((b)&
    /// 0x0000ffff)),15))`.
    #[inline(always)]
    #[must_use]
    pub fn mult16_32_q15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(
            shl(mult16_16(a, shr(b, 16)), 1),
            shr(mult16_16su(a, b & 0x0000ffff), 15),
        )
    }
    /// `MULT32_32_Q16`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        // C: `ADD32(ADD32(ADD32((opus_val32)(SHR32(((opus_uint32)((a)&0x0000ffff)*
        // (opus_uint32)((b)&0x0000ffff)),16)), MULT16_16SU(SHR32(a,16),((b)&0x0000ffff))),
        // MULT16_16SU(SHR32(b,16),((a)&0x0000ffff))), SHL32(MULT16_16(SHR32(a,16),SHR32(b,16)),
        // 16))`
        let lo = ((a & 0x0000ffff) as u32).wrapping_mul((b & 0x0000ffff) as u32) >> 16;
        add32(
            add32(
                add32(lo as i32, mult16_16su(shr32(a, 16), b & 0x0000ffff)),
                mult16_16su(shr32(b, 16), a & 0x0000ffff),
            ),
            super::shl32(mult16_16(shr32(a, 16), shr32(b, 16)), 16),
        )
    }
    /// `MULT32_32_Q31`: `ADD32(ADD32(SHL(MULT16_16(SHR((a),16),SHR((b),16)),1),
    /// SHR(MULT16_16SU(SHR((a),16),((b)&0x0000ffff)),15)), SHR(MULT16_16SU(SHR((b),16),((a)&
    /// 0x0000ffff)),15))`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(
            add32(
                shl(mult16_16(shr(a, 16), shr(b, 16)), 1),
                shr(mult16_16su(shr(a, 16), b & 0x0000ffff), 15),
            ),
            shr(mult16_16su(shr(b, 16), a & 0x0000ffff), 15),
        )
    }
    /// The rounding term of `MULT32_32_P31`: `SHR32(128+(opus_int32)SHR(MULT16_16U(((a)&
    /// 0x0000ffff),((b)&0x0000ffff)),16+7) + SHR32(MULT16_16SU(SHR((a),16),((b)&0x0000ffff)),7) +
    /// SHR32(MULT16_16SU(SHR((b),16),((a)&0x0000ffff)),7), 8)`.
    #[inline(always)]
    fn p31_low(a: i32, b: i32) -> i32 {
        let lo = (mult16_16u((a & 0x0000ffff) as u32, (b & 0x0000ffff) as u32) >> (16 + 7)) as i32;
        shr32(
            128 + lo
                + shr32(mult16_16su(shr(a, 16), b & 0x0000ffff), 7)
                + shr32(mult16_16su(shr(b, 16), a & 0x0000ffff), 7),
            8,
        )
    }
    /// `MULT32_32_P31`: `ADD32(SHL(MULT16_16(SHR((a),16),SHR((b),16)),1), <rounding term>)`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_p31(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(shl(mult16_16(shr(a, 16), shr(b, 16)), 1), p31_low(a, b))
    }
    /// `MULT32_32_P31_ovflw`: as [`mult32_32_p31`] with a wrapping final add.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_p31_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32_ovflw(shl(mult16_16(shr(a, 16), shr(b, 16)), 1), p31_low(a, b))
    }
    /// `MULT32_32_Q32`: `ADD32(ADD32(MULT16_16(SHR((a),16),SHR((b),16)), SHR(MULT16_16SU(SHR((a),
    /// 16),((b)&0x0000ffff)),16)), SHR(MULT16_16SU(SHR((b),16),((a)&0x0000ffff)),16))`.
    #[inline(always)]
    #[must_use]
    pub fn mult32_32_q32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
        let (a, b) = (a.into(), b.into());
        add32(
            add32(
                mult16_16(shr(a, 16), shr(b, 16)),
                shr(mult16_16su(shr(a, 16), b & 0x0000ffff), 16),
            ),
            shr(mult16_16su(shr(b, 16), a & 0x0000ffff), 16),
        )
    }
}

macro_rules! fast_or_generic {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub fn $name(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
            if OPUS_FAST_INT64 { int64::$name(a, b) } else { int32::$name(a, b) }
        }
    )+};
}
fast_or_generic!(
    /// `MULT16_32_Q16`: 16×32 multiplication, followed by a 16-bit shift right.
    mult16_32_q16,
    /// `MULT16_32_P16`: 16×32 multiplication, followed by a rounding 16-bit shift right.
    mult16_32_p16,
    /// `MULT16_32_Q15`: 16×32 multiplication, followed by a 15-bit shift right.
    mult16_32_q15,
    /// `MULT32_32_Q16`: 32×32 multiplication, followed by a 16-bit shift right.
    mult32_32_q16,
    /// `MULT32_32_Q31`: 32×32 multiplication, followed by a 31-bit shift right.
    mult32_32_q31,
    /// `MULT32_32_P31`: 32×32 multiplication, followed by a rounding 31-bit shift right.
    mult32_32_p31,
    /// `MULT32_32_P31_ovflw`.
    mult32_32_p31_ovflw,
    /// `MULT32_32_Q32`: 32×32 multiplication, followed by a 32-bit shift right.
    mult32_32_q32,
);

/// `QCONST16`: compile-time conversion of a float constant to Q`bits`
/// (`(opus_val16)(.5+(x)*(((opus_val32)1)<<(bits)))`). For an `f`-suffixed C literal pass
/// `LIT_f32 as f64` (the power-of-two scaling is exact in either precision).
#[inline(always)]
#[must_use]
pub const fn qconst16(x: f64, bits: i32) -> OpusVal16 {
    (0.5 + x * (1i32 << bits) as f64) as i16
}
/// `QCONST32`: `(opus_val32)(.5+(x)*(((opus_int64)1)<<(bits)))`.
#[inline(always)]
#[must_use]
pub const fn qconst32(x: f64, bits: i32) -> OpusVal32 {
    (0.5 + x * (1i64 << bits) as f64) as i32
}
/// `GCONST2`: `(celt_glog)(.5+(x)*(((celt_glog)1)<<(bits)))`.
#[inline(always)]
#[must_use]
pub const fn gconst2(x: f64, bits: i32) -> CeltGlog {
    (0.5 + x * (1i32 << bits) as f64) as i32
}
/// `GCONST`: `GCONST2((x), DB_SHIFT)`.
#[inline(always)]
#[must_use]
pub const fn gconst(x: f64) -> CeltGlog {
    gconst2(x, DB_SHIFT)
}

/// `NEG16` (`-(x)`, an `int`).
#[inline(always)]
#[must_use]
pub fn neg16(x: impl Into<i32>) -> i32 {
    -x.into()
}
/// `NEG32`.
#[inline(always)]
#[must_use]
pub fn neg32(x: impl Into<i32>) -> i32 {
    -x.into()
}
/// `EXTRACT16`: `(opus_val16)(x)` (truncating, like C).
#[inline(always)]
#[must_use]
pub fn extract16(x: impl Into<i32>) -> OpusVal16 {
    x.into() as i16
}
/// `EXTEND32`: `(opus_val32)(x)`.
#[inline(always)]
#[must_use]
pub fn extend32(x: impl Into<i32>) -> OpusVal32 {
    x.into()
}
/// `SHR16`: `(a) >> (shift)` (an `int`).
#[inline(always)]
#[must_use]
pub fn shr16(a: impl Into<i32>, shift: i32) -> i32 {
    a.into() >> shift
}
/// `SHL16`: `(opus_int16)((opus_uint16)(a)<<(shift))`.
#[inline(always)]
#[must_use]
pub fn shl16(a: impl Into<i32>, shift: i32) -> OpusVal16 {
    ((a.into() as u16 as i32) << shift) as i16
}
/// `SHR32`: `(a) >> (shift)`.
#[inline(always)]
#[must_use]
pub fn shr32(a: impl Into<i32>, shift: i32) -> i32 {
    a.into() >> shift
}
/// `SHL32`: `(opus_int32)((opus_uint32)(a)<<(shift))`.
#[inline(always)]
#[must_use]
pub fn shl32(a: impl Into<i32>, shift: i32) -> i32 {
    ((a.into() as u32) << shift) as i32
}
/// `PSHR32`: 32-bit arithmetic shift right with rounding to nearest
/// (`SHR32((a)+((EXTEND32(1)<<((shift))>>1)),shift)`).
#[inline(always)]
#[must_use]
pub fn pshr32(a: impl Into<i32>, shift: i32) -> i32 {
    (a.into() + ((1i32 << shift) >> 1)) >> shift
}
/// `VSHR32`: shift right by `shift`, or left by `-shift` when it is not positive.
#[inline(always)]
#[must_use]
pub fn vshr32(a: impl Into<i32>, shift: i32) -> i32 {
    if shift > 0 {
        shr32(a, shift)
    } else {
        shl32(a, -shift)
    }
}
/// `SHR64`: `(a) >> (shift)` on a 64-bit value.
#[inline(always)]
#[must_use]
pub const fn shr64(a: i64, shift: i32) -> i64 {
    a >> shift
}
/// `SHR` ("raw" macro; `(a) >> (shift)`).
#[inline(always)]
#[must_use]
pub fn shr(a: impl Into<i32>, shift: i32) -> i32 {
    a.into() >> shift
}
/// `SHL` (`SHL32`).
#[inline(always)]
#[must_use]
pub fn shl(a: impl Into<i32>, shift: i32) -> i32 {
    shl32(a, shift)
}
/// `PSHR` ("raw" macro; `SHR((a)+((EXTEND32(1)<<((shift))>>1)),shift)`).
#[inline(always)]
#[must_use]
pub fn pshr(a: impl Into<i32>, shift: i32) -> i32 {
    pshr32(a, shift)
}
/// `SATURATE(x,a)`: clamps `x` to `[-a, a]`.
#[inline(always)]
#[must_use]
pub fn saturate(x: impl Into<i32>, a: impl Into<i32>) -> i32 {
    let (x, a) = (x.into(), a.into());
    if x > a {
        a
    } else if x < -a {
        -a
    } else {
        x
    }
}
/// `SATURATE16`: `EXTRACT16` of `x` clamped to `[-32768, 32767]`.
#[inline(always)]
#[must_use]
pub fn saturate16(x: impl Into<i32>) -> OpusVal16 {
    sat16(x.into())
}
/// `ROUND16`: `EXTRACT16(PSHR32((x),(a)))`.
#[inline(always)]
#[must_use]
pub fn round16(x: impl Into<i32>, a: i32) -> OpusVal16 {
    pshr32(x, a) as i16
}
/// `SROUND16`: `EXTRACT16(SATURATE(PSHR32(x,a), 32767))`.
#[inline(always)]
#[must_use]
pub fn sround16(x: impl Into<i32>, a: i32) -> OpusVal16 {
    saturate(pshr32(x, a), 32767) as i16
}
/// `HALF16`: `SHR16(x,1)`.
#[inline(always)]
#[must_use]
pub fn half16(x: impl Into<i32>) -> i32 {
    shr16(x, 1)
}
/// `HALF32`: `SHR32(x,1)`.
#[inline(always)]
#[must_use]
pub fn half32(x: impl Into<i32>) -> i32 {
    shr32(x, 1)
}
/// `ADD16`: `(opus_val16)((opus_val16)(a)+(opus_val16)(b))`.
#[inline(always)]
#[must_use]
pub fn add16(a: impl Into<i32>, b: impl Into<i32>) -> OpusVal16 {
    (w16(a.into()) + w16(b.into())) as i16
}
/// `SUB16`: `(opus_val16)(a)-(opus_val16)(b)` (an `int`).
#[inline(always)]
#[must_use]
pub fn sub16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    w16(a.into()) - w16(b.into())
}
/// `ADD32`: `(opus_val32)(a)+(opus_val32)(b)`.
#[inline(always)]
#[must_use]
pub fn add32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into() + b.into()
}
/// `SUB32`: `(opus_val32)(a)-(opus_val32)(b)`.
#[inline(always)]
#[must_use]
pub fn sub32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into() - b.into()
}
/// `ADD32_ovflw`: add ignoring overflow.
#[inline(always)]
#[must_use]
pub fn add32_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into().wrapping_add(b.into())
}
/// `SUB32_ovflw`: subtract ignoring overflow.
#[inline(always)]
#[must_use]
pub fn sub32_ovflw(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into().wrapping_sub(b.into())
}
/// `NEG32_ovflw`: negate ignoring overflow.
#[inline(always)]
#[must_use]
pub fn neg32_ovflw(a: impl Into<i32>) -> i32 {
    a.into().wrapping_neg()
}
/// `SHL32_ovflw` (`SHL32`).
#[inline(always)]
#[must_use]
pub fn shl32_ovflw(a: impl Into<i32>, shift: i32) -> i32 {
    shl32(a, shift)
}
/// `PSHR32_ovflw`: `SHR32(ADD32_ovflw(a, (EXTEND32(1)<<(shift)>>1)),shift)`.
#[inline(always)]
#[must_use]
pub fn pshr32_ovflw(a: impl Into<i32>, shift: i32) -> i32 {
    add32_ovflw(a, (1i32 << shift) >> 1) >> shift
}
/// `MULT16_16_16`: `((opus_val16)(a))*((opus_val16)(b))` (an `int`).
#[inline(always)]
#[must_use]
pub fn mult16_16_16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    w16(a.into()) * w16(b.into())
}
/// `MULT32_32_32`: `((opus_val32)(a))*((opus_val32)(b))`.
#[inline(always)]
#[must_use]
pub fn mult32_32_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into() * b.into()
}
/// `MULT16_16`: `((opus_val32)(opus_val16)(a))*((opus_val32)(opus_val16)(b))`.
#[inline(always)]
#[must_use]
pub fn mult16_16(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    w16(a.into()) * w16(b.into())
}
/// `MAC16_16`: `ADD32((c),MULT16_16((a),(b)))`.
#[inline(always)]
#[must_use]
pub fn mac16_16(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    add32(c, mult16_16(a, b))
}
/// `MAC16_32_Q15`: `ADD32((c),ADD32(MULT16_16((a),SHR((b),15)), SHR(MULT16_16((a),((b)&
/// 0x00007fff)),15)))` (`b` must fit in 31 bits).
#[inline(always)]
#[must_use]
pub fn mac16_32_q15(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    add32(
        c,
        add32(
            mult16_16(a, shr(b, 15)),
            shr(mult16_16(a, b & 0x00007fff), 15),
        ),
    )
}
/// `MAC16_32_Q16`: `ADD32((c),ADD32(MULT16_16((a),SHR((b),16)), SHR(MULT16_16SU((a),((b)&
/// 0x0000ffff)),16)))`.
#[inline(always)]
#[must_use]
pub fn mac16_32_q16(c: impl Into<i32>, a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    let (a, b) = (a.into(), b.into());
    add32(
        c,
        add32(
            mult16_16(a, shr(b, 16)),
            shr(mult16_16su(a, b & 0x0000ffff), 16),
        ),
    )
}
/// `MULT16_16_Q11_32`: `SHR(MULT16_16((a),(b)),11)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q11_32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(mult16_16(a, b), 11)
}
/// `MULT16_16_Q11`: `SHR(MULT16_16((a),(b)),11)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q11(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(mult16_16(a, b), 11)
}
/// `MULT16_16_Q13`: `SHR(MULT16_16((a),(b)),13)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q13(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(mult16_16(a, b), 13)
}
/// `MULT16_16_Q14`: `SHR(MULT16_16((a),(b)),14)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q14(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(mult16_16(a, b), 14)
}
/// `MULT16_16_Q15`: `SHR(MULT16_16((a),(b)),15)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_q15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(mult16_16(a, b), 15)
}
/// `MULT16_16_P13`: `SHR(ADD32(4096,MULT16_16((a),(b))),13)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p13(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(add32(4096, mult16_16(a, b)), 13)
}
/// `MULT16_16_P14`: `SHR(ADD32(8192,MULT16_16((a),(b))),14)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p14(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(add32(8192, mult16_16(a, b)), 14)
}
/// `MULT16_16_P15`: `SHR(ADD32(16384,MULT16_16((a),(b))),15)`.
#[inline(always)]
#[must_use]
pub fn mult16_16_p15(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    shr(add32(16384, mult16_16(a, b)), 15)
}
/// `DIV32_16`: `(opus_val16)(((opus_val32)(a))/((opus_val16)(b)))`.
#[inline(always)]
#[must_use]
pub fn div32_16(a: impl Into<i32>, b: impl Into<i32>) -> OpusVal16 {
    (a.into() / w16(b.into())) as i16
}
/// `DIV32`: `((opus_val32)(a))/((opus_val32)(b))`.
#[inline(always)]
#[must_use]
pub fn div32(a: impl Into<i32>, b: impl Into<i32>) -> i32 {
    a.into() / b.into()
}
/// `SIG2WORD16` (`SIG2WORD16_generic`): rounds a `celt_sig` to 16-bit PCM with saturation.
#[inline(always)]
#[must_use]
pub fn sig2word16(x: CeltSig) -> OpusVal16 {
    let mut x = pshr32(x, SIG_SHIFT);
    x = max32(x, -32768);
    x = min32(x, 32767);
    x as i16
}

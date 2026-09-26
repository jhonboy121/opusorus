//! Port of `celt/arch.h` (float build, `#ifndef FIXED_POINT` part): numeric type aliases and the
//! arithmetic "macros" used throughout CELT.
//!
//! In the float build every fixed-point macro degenerates to plain float arithmetic. They are
//! kept as `const fn`s so that the port mirrors the C source line-for-line, which (a) keeps float
//! operation order identical to the oracle and (b) shares the code with the fixed-point build
//! (`super::fixed` defines the same names with integer semantics).
//!
//! Naming: C macro `MULT16_32_Q15` → `mult16_32_q15`, etc.

use super::CELT_SIG_SCALE;

/// `opus_val16`.
pub type OpusVal16 = f32;
/// `opus_val32`.
pub type OpusVal32 = f32;
/// `opus_val64`.
pub type OpusVal64 = f32;
/// `celt_sig`: internal CELT signal, ±32768 full scale.
pub type CeltSig = f32;
/// `celt_norm`: normalised (unit-energy) band coefficients.
pub type CeltNorm = f32;
/// `celt_ener`.
pub type CeltEner = f32;
/// `celt_glog`: log-domain energy.
pub type CeltGlog = f32;
/// `opus_res`: Opus internal resolution, ±1.0 full scale.
pub type OpusRes = f32;
/// `celt_coef`.
pub type CeltCoef = f32;

/// `Q15ONE`.
pub const Q15ONE: f32 = 1.0;
/// `Q31ONE`.
pub const Q31ONE: f32 = 1.0;
/// `COEF_ONE`.
pub const COEF_ONE: f32 = 1.0;
/// `NORM_SCALING`.
pub const NORM_SCALING: f32 = 1.0;
/// `EPSILON`.
pub const EPSILON: f32 = 1e-15;
/// `VERY_SMALL`.
pub const VERY_SMALL: f32 = 1e-30;
/// `VERY_LARGE16`.
pub const VERY_LARGE16: f32 = 1e15;
/// `Q15_ONE`.
pub const Q15_ONE: f32 = 1.0;
/// `MAX_ENCODING_DEPTH` (float build).
pub const MAX_ENCODING_DEPTH: i32 = 24;

/// `celt_isnan`: `((x)!=(x))`.
#[cfg(not(feature = "float-approx"))]
#[inline(always)]
#[must_use]
pub const fn celt_isnan(x: f32) -> bool {
    x.is_nan()
}

/// `celt_isnan` (`FLOAT_APPROX`): tests the IEEE 754 bits (all-ones exponent, non-zero
/// mantissa) so NaN is detected even under `-ffast-math` in C. Same result as `x != x` here.
#[cfg(feature = "float-approx")]
#[inline(always)]
#[must_use]
pub const fn celt_isnan(x: f32) -> bool {
    let i = x.to_bits();
    ((i >> 23) & 0xFF) == 0xFF && (i & 0x007F_FFFF) != 0
}

macro_rules! ident1 {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub const fn $name(x: f32) -> f32 { x }
    )+};
}
macro_rules! ident_shift {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub const fn $name(a: f32, _shift: i32) -> f32 { a }
    )+};
}
macro_rules! mul2 {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub const fn $name(a: f32, b: f32) -> f32 { a * b }
    )+};
}
macro_rules! mac3 {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$m])*
        #[inline(always)]
        #[must_use]
        pub const fn $name(c: f32, a: f32, b: f32) -> f32 { c + a * b }
    )+};
}

ident1!(
    /// `EXTRACT16`.
    extract16,
    /// `EXTEND32`.
    extend32,
    /// `SATURATE16`.
    saturate16,
    /// `COEF2VAL16`.
    coef2val16,
    /// `RES2FLOAT`.
    res2float,
    /// `FLOAT2RES`.
    float2res,
    /// `RES2VAL16`.
    res2val16,
);
ident_shift!(
    /// `SHR16`.
    shr16,
    /// `SHL16`.
    shl16,
    /// `SHR32`.
    shr32,
    /// `SHL32`.
    shl32,
    /// `PSHR32`.
    pshr32,
    /// `VSHR32`.
    vshr32,
    /// `SHR64`.
    shr64,
    /// `PSHR`.
    pshr,
    /// `SHR`.
    shr,
    /// `SHL`.
    shl,
    /// `SATURATE`.
    saturate,
    /// `ROUND16`.
    round16,
    /// `SROUND16`.
    sround16,
    /// `SHL32_ovflw`.
    shl32_ovflw,
    /// `PSHR32_ovflw`.
    pshr32_ovflw,
    /// `QCONST16`.
    qconst16,
    /// `QCONST32`.
    qconst32,
);
mul2!(
    /// `MULT16_16_16`.
    mult16_16_16,
    /// `MULT16_16`.
    mult16_16,
    /// `MULT16_32_Q15`.
    mult16_32_q15,
    /// `MULT16_32_Q16`.
    mult16_32_q16,
    /// `MULT32_32_Q16`.
    mult32_32_q16,
    /// `MULT32_32_Q31`.
    mult32_32_q31,
    /// `MULT32_32_P31`.
    mult32_32_p31,
    /// `MULT32_32_P31_ovflw`.
    mult32_32_p31_ovflw,
    /// `MULT16_16_Q11_32`.
    mult16_16_q11_32,
    /// `MULT16_16_Q11`.
    mult16_16_q11,
    /// `MULT16_16_Q13`.
    mult16_16_q13,
    /// `MULT16_16_Q14`.
    mult16_16_q14,
    /// `MULT16_16_Q15`.
    mult16_16_q15,
    /// `MULT16_16_P15`.
    mult16_16_p15,
    /// `MULT16_16_P13`.
    mult16_16_p13,
    /// `MULT16_16_P14`.
    mult16_16_p14,
    /// `MULT16_32_P16`.
    mult16_32_p16,
    /// `MULT_COEF_32`.
    mult_coef_32,
    /// `MULT_COEF`.
    mult_coef,
    /// `MULT_COEF_TAPS`.
    mult_coef_taps,
    /// `MULT16_RES_Q15`.
    mult16_res_q15,
);
mac3!(
    /// `MAC16_16`.
    mac16_16,
    /// `MAC16_32_Q15`.
    mac16_32_q15,
    /// `MAC16_32_Q16`.
    mac16_32_q16,
    /// `MAC_COEF_32_ARM`.
    mac_coef_32_arm,
);

/// `GCONST`.
#[inline(always)]
#[must_use]
pub const fn gconst(x: f32) -> f32 {
    x
}
/// `NEG16`.
#[inline(always)]
#[must_use]
pub const fn neg16(x: f32) -> f32 {
    -x
}
/// `NEG32`.
#[inline(always)]
#[must_use]
pub const fn neg32(x: f32) -> f32 {
    -x
}
/// `NEG32_ovflw`.
#[inline(always)]
#[must_use]
pub const fn neg32_ovflw(x: f32) -> f32 {
    -x
}
/// `ABS16` (C: `(float)fabs(x)`).
#[inline(always)]
#[must_use]
pub const fn abs16(x: f32) -> f32 {
    x.abs()
}
/// `ABS32`.
#[inline(always)]
#[must_use]
pub const fn abs32(x: f32) -> f32 {
    x.abs()
}
/// `HALF16` (C: `.5f*(x)`).
#[inline(always)]
#[must_use]
pub const fn half16(x: f32) -> f32 {
    0.5 * x
}
/// `HALF32`.
#[inline(always)]
#[must_use]
pub const fn half32(x: f32) -> f32 {
    0.5 * x
}
/// `ADD16`.
#[inline(always)]
#[must_use]
pub const fn add16(a: f32, b: f32) -> f32 {
    a + b
}
/// `SUB16`.
#[inline(always)]
#[must_use]
pub const fn sub16(a: f32, b: f32) -> f32 {
    a - b
}
/// `ADD32`.
#[inline(always)]
#[must_use]
pub const fn add32(a: f32, b: f32) -> f32 {
    a + b
}
/// `SUB32`.
#[inline(always)]
#[must_use]
pub const fn sub32(a: f32, b: f32) -> f32 {
    a - b
}
/// `ADD32_ovflw`.
#[inline(always)]
#[must_use]
pub const fn add32_ovflw(a: f32, b: f32) -> f32 {
    a + b
}
/// `SUB32_ovflw`.
#[inline(always)]
#[must_use]
pub const fn sub32_ovflw(a: f32, b: f32) -> f32 {
    a - b
}
/// `ADD_RES`.
#[inline(always)]
#[must_use]
pub const fn add_res(a: f32, b: f32) -> f32 {
    a + b
}
/// `DIV32_16`.
#[inline(always)]
#[must_use]
pub const fn div32_16(a: f32, b: f32) -> f32 {
    a / b
}
/// `DIV32`.
#[inline(always)]
#[must_use]
pub const fn div32(a: f32, b: f32) -> f32 {
    a / b
}
/// `SIG2RES` (C: `(1/CELT_SIG_SCALE)*(a)`).
#[inline(always)]
#[must_use]
pub const fn sig2res(a: f32) -> f32 {
    (1.0 / CELT_SIG_SCALE) * a
}
/// `INT16TORES` (C: `(a)*(1/CELT_SIG_SCALE)`).
#[inline(always)]
#[must_use]
pub const fn int16tores(a: i16) -> f32 {
    a as f32 * (1.0 / CELT_SIG_SCALE)
}
/// `INT24TORES` (C: `(1.f/32768.f/256.f)*(a)`).
#[inline(always)]
#[must_use]
pub const fn int24tores(a: i32) -> f32 {
    (1.0 / 32768.0 / 256.0) * a as f32
}
/// `RES2SIG` (C: `CELT_SIG_SCALE*(a)`).
#[inline(always)]
#[must_use]
pub const fn res2sig(a: f32) -> f32 {
    CELT_SIG_SCALE * a
}
/// `FLOAT2SIG` (C: `(a)*CELT_SIG_SCALE`).
#[inline(always)]
#[must_use]
pub const fn float2sig(a: f32) -> f32 {
    a * CELT_SIG_SCALE
}
/// `INT16TOSIG`.
#[inline(always)]
#[must_use]
pub const fn int16tosig(a: i16) -> f32 {
    a as f32
}
/// `INT24TOSIG` (C: `(float)(a)*(1.f/256.f)`).
#[inline(always)]
#[must_use]
pub const fn int24tosig(a: i32) -> f32 {
    a as f32 * (1.0 / 256.0)
}
/// `RES2INT16` (= `FLOAT2INT16`).
#[inline(always)]
#[must_use]
pub fn res2int16(a: f32) -> i16 {
    crate::celt::mathops::float2int16(a)
}
/// `RES2INT24` (C: `float2int(32768.f*256.f*(a))`).
#[inline(always)]
#[must_use]
pub fn res2int24(a: f32) -> i32 {
    crate::celt::mathops::float2int(32768.0 * 256.0 * a)
}

/// `MIN16`/`MIN32`/`FMIN`/`MING` (C: `(a) < (b) ? (a) : (b)` — note NaN semantics differ from
/// `f32::min`, so always use this).
#[inline(always)]
#[must_use]
pub const fn min16(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
/// `MAX16`/`MAX32`/`FMAX`/`MAXG` (C: `(a) > (b) ? (a) : (b)`).
#[inline(always)]
#[must_use]
pub const fn max16(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
/// `MIN32` (float).
#[inline(always)]
#[must_use]
pub const fn min32(a: f32, b: f32) -> f32 {
    min16(a, b)
}
/// `MAX32` (float).
#[inline(always)]
#[must_use]
pub const fn max32(a: f32, b: f32) -> f32 {
    max16(a, b)
}
/// `FMIN`.
#[inline(always)]
#[must_use]
pub const fn fmin(a: f32, b: f32) -> f32 {
    min16(a, b)
}
/// `FMAX`.
#[inline(always)]
#[must_use]
pub const fn fmax(a: f32, b: f32) -> f32 {
    max16(a, b)
}
/// `MING`.
#[inline(always)]
#[must_use]
pub const fn ming(a: f32, b: f32) -> f32 {
    min16(a, b)
}
/// `MAXG`.
#[inline(always)]
#[must_use]
pub const fn maxg(a: f32, b: f32) -> f32 {
    max16(a, b)
}

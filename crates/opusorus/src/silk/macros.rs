//! Port of SILK fixed-point primitives: `silk/macros.h`, the macro/inline part of
//! `silk/SigProc_FIX.h`, `silk/Inlines.h` and the limits in `silk/typedef.h`.
//!
//! Naming: `silk_SMULWB` → `silk_smulwb`. All take/return `i32` unless the C macro is
//! specifically 16- or 64-bit. The 64-bit (`OPUS_FAST_INT64`) forms are used; they are exactly
//! equivalent to the 32-bit fallbacks.
//!
//! Overflow policy: macros documented as wrapping in C (`*_ovflw`, `LSHIFT` via unsigned casts,
//! `silk_RAND`) use `wrapping_*`; all others use plain arithmetic so debug builds trap on bugs.

#![allow(
    clippy::cast_lossless,
    reason = "casts mirror C integer promotions one-to-one; keep them explicit"
)]

/// `silk_int64_MAX`.
pub const SILK_INT64_MAX: i64 = i64::MAX;
/// `silk_int64_MIN`.
pub const SILK_INT64_MIN: i64 = i64::MIN;
/// `silk_int32_MAX`.
pub const SILK_INT32_MAX: i32 = i32::MAX;
/// `silk_int32_MIN`.
pub const SILK_INT32_MIN: i32 = i32::MIN;
/// `silk_int16_MAX`.
pub const SILK_INT16_MAX: i32 = i16::MAX as i32;
/// `silk_int16_MIN`.
pub const SILK_INT16_MIN: i32 = i16::MIN as i32;
/// `silk_int8_MAX`.
pub const SILK_INT8_MAX: i32 = i8::MAX as i32;
/// `silk_int8_MIN`.
pub const SILK_INT8_MIN: i32 = i8::MIN as i32;
/// `silk_uint8_MAX`.
pub const SILK_UINT8_MAX: i32 = u8::MAX as i32;

/// `RAND_MULTIPLIER`.
pub const RAND_MULTIPLIER: i32 = 196_314_165;
/// `RAND_INCREMENT`.
pub const RAND_INCREMENT: i32 = 907_633_515;

// FIXED_DEBUG: `silk/MacroDebug.h` replaces the macros below that are
// `#[cfg(not(feature = "fixed-point-debug"))]` with checking versions.
#[cfg(feature = "fixed-point-debug")]
pub use super::macro_debug::*;

/// `SILK_MAX_ORDER_LPC`.
pub const SILK_MAX_ORDER_LPC: usize = 24;

// ---------------------------------------------------------------------------------------------
// macros.h
// ---------------------------------------------------------------------------------------------

#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMULWB`: `(a32 * (i16)b32) >> 16`.
#[inline(always)]
#[must_use]
pub const fn silk_smulwb(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * (b32 as i16) as i64) >> 16) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLAWB`: `a32 + ((b32 * (i16)c32) >> 16)`.
#[inline(always)]
#[must_use]
pub const fn silk_smlawb(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * (c32 as i16) as i64) >> 16)) as i32
}
/// Perf helper, no C counterpart: the `silk_SMLAWB` chain of the SILK short-term prediction
/// loops,
/// `acc = silk_SMLAWB(acc, x[n-1], c[0]); acc = silk_SMLAWB(acc, x[n-2], c[1]); ...
/// acc = silk_SMLAWB(acc, x[0], c[n-1])` with `n = x.len()`, over a history window `x` (oldest
/// sample first, so `x[n-1]` is the newest one).
///
/// Each `silk_SMLAWB` step is an `i64` sum truncated to 32 bits (the `OPUS_FAST_INT64` form),
/// so the chain equals `acc` plus the `silk_SMULWB` terms summed modulo 2^32, in any order:
/// the result is bit-identical to the C chain. The terms are added oldest first here, which
/// takes the newest sample (usually the output of the previous loop iteration) off the long
/// serial add chain of the recursive filters. Callers pass a window of constant length (the
/// branches on the LPC order) so that the loop is fully unrolled.
#[inline(always)]
#[must_use]
pub fn silk_smlawb_chain(acc: i32, x: &[i32], c: &[i16]) -> i32 {
    let n = x.len();
    let c = &c[..n];
    let mut acc = acc;
    // FIXED_DEBUG: the C chain of checking `silk_SMLAWB`s, newest sample first.
    #[cfg(feature = "fixed-point-debug")]
    for k in 0..n {
        acc = silk_smlawb(acc, x[n - 1 - k], c[k] as i32);
    }
    #[cfg(not(feature = "fixed-point-debug"))]
    for k in (0..n).rev() {
        acc = acc.wrapping_add(silk_smulwb(x[n - 1 - k], c[k] as i32));
    }
    acc
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMULWT`: `(a32 * (b32 >> 16)) >> 16`.
#[inline(always)]
#[must_use]
pub const fn silk_smulwt(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * (b32 >> 16) as i64) >> 16) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLAWT`: `a32 + ((b32 * (c32 >> 16)) >> 16)`.
#[inline(always)]
#[must_use]
pub const fn silk_smlawt(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * ((c32 as i64) >> 16)) >> 16)) as i32
}
/// `silk_SMULBB`: `(i16)a32 * (i16)b32`.
#[inline(always)]
#[must_use]
pub const fn silk_smulbb(a32: i32, b32: i32) -> i32 {
    (a32 as i16) as i32 * (b32 as i16) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLABB`: `a32 + (i16)b32 * (i16)c32`.
#[inline(always)]
#[must_use]
pub const fn silk_smlabb(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + (b32 as i16) as i32 * (c32 as i16) as i32
}
/// `silk_SMULBT`: `(i16)a32 * (b32 >> 16)`.
#[inline(always)]
#[must_use]
pub const fn silk_smulbt(a32: i32, b32: i32) -> i32 {
    (a32 as i16) as i32 * (b32 >> 16)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLABT`: `a32 + (i16)b32 * (c32 >> 16)`.
#[inline(always)]
#[must_use]
pub const fn silk_smlabt(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + (b32 as i16) as i32 * (c32 >> 16)
}
/// `silk_SMLAL`: `a64 + (i64)b32 * (i64)c32`.
#[inline(always)]
#[must_use]
pub const fn silk_smlal(a64: i64, b32: i32, c32: i32) -> i64 {
    a64 + b32 as i64 * c32 as i64
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMULWW`: `((i64)a32 * b32) >> 16`.
#[inline(always)]
#[must_use]
pub const fn silk_smulww(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * b32 as i64) >> 16) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLAWW`: `a32 + (((i64)b32 * c32) >> 16)`.
#[inline(always)]
#[must_use]
pub const fn silk_smlaww(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * c32 as i64) >> 16)) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_SAT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB_SAT32`.
#[inline(always)]
#[must_use]
pub const fn silk_sub_sat32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}
/// `silk_CLZ16`.
#[inline(always)]
#[must_use]
pub const fn silk_clz16(in16: i16) -> i32 {
    // C: 32 - EC_ILOG(in16<<16|0x8000) — the result is the count of leading zeros of in16.
    32 - crate::celt::entcode::ec_ilog((((in16 as i32) << 16) | 0x8000) as u32)
}
/// `silk_CLZ32`.
#[inline(always)]
#[must_use]
pub const fn silk_clz32(in32: i32) -> i32 {
    (in32 as u32).leading_zeros() as i32
}

// ---------------------------------------------------------------------------------------------
// SigProc_FIX.h
// ---------------------------------------------------------------------------------------------

/// `silk_ROR32`: rotate right.
#[inline(always)]
#[must_use]
pub const fn silk_ror32(a32: i32, rot: i32) -> i32 {
    let x = a32 as u32;
    if rot == 0 {
        a32
    } else if rot < 0 {
        x.rotate_left((-rot) as u32) as i32
    } else {
        x.rotate_right(rot as u32) as i32
    }
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_MUL`.
#[inline(always)]
#[must_use]
pub const fn silk_mul(a32: i32, b32: i32) -> i32 {
    a32 * b32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_MUL_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_mul_uint(a32: u32, b32: u32) -> u32 {
    a32 * b32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_MLA`.
#[inline(always)]
#[must_use]
pub const fn silk_mla(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + b32 * c32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_MLA_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_mla_uint(a32: u32, b32: u32, c32: u32) -> u32 {
    a32 + b32 * c32
}
/// `silk_SMULTT`.
#[inline(always)]
#[must_use]
pub const fn silk_smultt(a32: i32, b32: i32) -> i32 {
    (a32 >> 16) * (b32 >> 16)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMLATT`.
#[inline(always)]
#[must_use]
pub const fn silk_smlatt(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + (b32 >> 16) * (c32 >> 16)
}
/// `silk_SMLALBB`.
#[inline(always)]
#[must_use]
pub const fn silk_smlalbb(a64: i64, b16: i16, c16: i16) -> i64 {
    a64 + (b16 as i32 * c16 as i32) as i64
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SMULL`.
#[inline(always)]
#[must_use]
pub const fn silk_smull(a32: i32, b32: i32) -> i64 {
    a32 as i64 * b32 as i64
}
/// `silk_ADD32_ovflw`.
#[inline(always)]
#[must_use]
pub const fn silk_add32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}
/// `silk_SUB32_ovflw`.
#[inline(always)]
#[must_use]
pub const fn silk_sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}
/// `silk_MLA_ovflw`.
#[inline(always)]
#[must_use]
pub const fn silk_mla_ovflw(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add((b32 as u32).wrapping_mul(c32 as u32) as i32)
}
/// `silk_SMLABB_ovflw`.
#[inline(always)]
#[must_use]
pub const fn silk_smlabb_ovflw(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add((b32 as i16) as i32 * (c32 as i16) as i32)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_DIV32_16`.
#[inline(always)]
#[must_use]
pub const fn silk_div32_16(a32: i32, b16: i32) -> i32 {
    a32 / b16
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_DIV32`.
#[inline(always)]
#[must_use]
pub const fn silk_div32(a32: i32, b32: i32) -> i32 {
    a32 / b32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD16`.
#[inline(always)]
#[must_use]
pub const fn silk_add16(a: i16, b: i16) -> i16 {
    a + b
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD32`.
#[inline(always)]
#[must_use]
pub const fn silk_add32(a: i32, b: i32) -> i32 {
    a + b
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD64`.
#[inline(always)]
#[must_use]
pub const fn silk_add64(a: i64, b: i64) -> i64 {
    a + b
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB16`.
#[inline(always)]
#[must_use]
pub const fn silk_sub16(a: i16, b: i16) -> i16 {
    a - b
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB32`.
#[inline(always)]
#[must_use]
pub const fn silk_sub32(a: i32, b: i32) -> i32 {
    a - b
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB64`.
#[inline(always)]
#[must_use]
pub const fn silk_sub64(a: i64, b: i64) -> i64 {
    a - b
}
/// `silk_SAT8`.
#[inline(always)]
#[must_use]
pub const fn silk_sat8(a: i32) -> i32 {
    if a > SILK_INT8_MAX {
        SILK_INT8_MAX
    } else if a < SILK_INT8_MIN {
        SILK_INT8_MIN
    } else {
        a
    }
}
/// `silk_SAT16`.
#[inline(always)]
#[must_use]
pub const fn silk_sat16(a: i32) -> i32 {
    if a > SILK_INT16_MAX {
        SILK_INT16_MAX
    } else if a < SILK_INT16_MIN {
        SILK_INT16_MIN
    } else {
        a
    }
}
/// `silk_SAT32` (on a 64-bit input).
#[inline(always)]
#[must_use]
pub const fn silk_sat32(a: i64) -> i32 {
    if a > SILK_INT32_MAX as i64 {
        SILK_INT32_MAX
    } else if a < SILK_INT32_MIN as i64 {
        SILK_INT32_MIN
    } else {
        a as i32
    }
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_SAT16`.
#[inline(always)]
#[must_use]
pub const fn silk_add_sat16(a: i16, b: i32) -> i16 {
    silk_sat16(a as i32 + b) as i16
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB_SAT16`.
#[inline(always)]
#[must_use]
pub const fn silk_sub_sat16(a: i16, b: i32) -> i16 {
    silk_sat16(a as i32 - b) as i16
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_SAT64`.
#[inline(always)]
#[must_use]
pub const fn silk_add_sat64(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB_SAT64`.
#[inline(always)]
#[must_use]
pub const fn silk_sub_sat64(a: i64, b: i64) -> i64 {
    a.saturating_sub(b)
}
/// `silk_POS_SAT32` (on a 64-bit input).
#[inline(always)]
#[must_use]
pub const fn silk_pos_sat32(a: i64) -> i32 {
    if a > SILK_INT32_MAX as i64 {
        SILK_INT32_MAX
    } else {
        a as i32
    }
}
/// `silk_ADD_POS_SAT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_pos_sat32(a: i32, b: i32) -> i32 {
    if ((a as u32).wrapping_add(b as u32) & 0x8000_0000) != 0 {
        SILK_INT32_MAX
    } else {
        a + b
    }
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT8`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift8(a: i8, shift: u32) -> i8 {
    ((a as u8) << shift) as i8
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT16`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift16(a: i16, shift: u32) -> i16 {
    ((a as u16) << shift) as i16
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT32` / `silk_LSHIFT` (defined via unsigned shift: wraps).
#[inline(always)]
#[must_use]
pub const fn silk_lshift(a: i32, shift: i32) -> i32 {
    debug_assert!(shift >= 0 && shift < 32);
    ((a as u32) << shift) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift32(a: i32, shift: i32) -> i32 {
    silk_lshift(a, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT64`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift64(a: i64, shift: i32) -> i64 {
    debug_assert!(shift >= 0 && shift < 64);
    ((a as u64) << shift) as i64
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT32` / `silk_RSHIFT`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift(a: i32, shift: i32) -> i32 {
    debug_assert!(shift >= 0 && shift < 32);
    a >> shift
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift32(a: i32, shift: i32) -> i32 {
    silk_rshift(a, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT64`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift64(a: i64, shift: i32) -> i64 {
    debug_assert!(shift >= 0 && shift < 64);
    a >> shift
}
const_unless_fixed_debug! {
/// `silk_LSHIFT_SAT32`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift_sat32(a: i32, shift: i32) -> i32 {
    silk_lshift32(
        silk_limit(
            a,
            silk_rshift32(SILK_INT32_MIN, shift),
            silk_rshift32(SILK_INT32_MAX, shift),
        ),
        shift,
    )
}
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT_ovflw`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift_ovflw(a: i32, shift: i32) -> i32 {
    ((a as u32) << shift) as i32
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_LSHIFT_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_lshift_uint(a: u32, shift: i32) -> u32 {
    a << shift
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift_uint(a: u32, shift: i32) -> u32 {
    a >> shift
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_LSHIFT` / `silk_ADD_LSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_lshift(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_lshift(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_LSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_lshift32(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_LSHIFT_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_add_lshift_uint(a: u32, b: u32, shift: i32) -> u32 {
    a + (b << shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_RSHIFT` / `silk_ADD_RSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_rshift(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_rshift(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_RSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_rshift32(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_ADD_RSHIFT_uint`.
#[inline(always)]
#[must_use]
pub const fn silk_add_rshift_uint(a: u32, b: u32, shift: i32) -> u32 {
    a + (b >> shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB_LSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a - silk_lshift32(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_SUB_RSHIFT32`.
#[inline(always)]
#[must_use]
pub const fn silk_sub_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a - silk_rshift32(b, shift)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT_ROUND`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_RSHIFT_ROUND64`.
#[inline(always)]
#[must_use]
pub const fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}
/// `silk_NSHIFT_MUL_32_32`.
#[inline(always)]
#[must_use]
pub const fn silk_nshift_mul_32_32(a: i32, b: i32) -> i32 {
    -(31 - (32 - silk_clz32(silk_abs(a)) + (32 - silk_clz32(silk_abs(b)))))
}
/// `silk_NSHIFT_MUL_16_16`.
#[inline(always)]
#[must_use]
pub const fn silk_nshift_mul_16_16(a: i16, b: i16) -> i32 {
    -(15 - (16 - silk_clz16(silk_abs16(a)) + (16 - silk_clz16(silk_abs16(b)))))
}
/// `silk_min` / `silk_min_int` / `silk_min_32`.
#[inline(always)]
#[must_use]
pub const fn silk_min(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
/// `silk_max` / `silk_max_int` / `silk_max_32`.
#[inline(always)]
#[must_use]
pub const fn silk_max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
/// `silk_min_int`.
#[inline(always)]
#[must_use]
pub const fn silk_min_int(a: i32, b: i32) -> i32 {
    silk_min(a, b)
}
/// `silk_max_int`.
#[inline(always)]
#[must_use]
pub const fn silk_max_int(a: i32, b: i32) -> i32 {
    silk_max(a, b)
}
/// `silk_min_16`.
#[inline(always)]
#[must_use]
pub const fn silk_min_16(a: i16, b: i16) -> i16 {
    if a < b { a } else { b }
}
/// `silk_max_16`.
#[inline(always)]
#[must_use]
pub const fn silk_max_16(a: i16, b: i16) -> i16 {
    if a > b { a } else { b }
}
/// `silk_min_32`.
#[inline(always)]
#[must_use]
pub const fn silk_min_32(a: i32, b: i32) -> i32 {
    silk_min(a, b)
}
/// `silk_max_32`.
#[inline(always)]
#[must_use]
pub const fn silk_max_32(a: i32, b: i32) -> i32 {
    silk_max(a, b)
}
/// `silk_min_64`.
#[inline(always)]
#[must_use]
pub const fn silk_min_64(a: i64, b: i64) -> i64 {
    if a < b { a } else { b }
}
/// `silk_max_64`.
#[inline(always)]
#[must_use]
pub const fn silk_max_64(a: i64, b: i64) -> i64 {
    if a > b { a } else { b }
}
/// `SILK_FIX_CONST(C, Q)`: `(opus_int32)((C) * ((opus_int64)1 << (Q)) + 0.5)` (double math).
#[inline(always)]
#[must_use]
pub const fn silk_fix_const(c: f64, q: i32) -> i32 {
    (c * ((1i64 << q) as f64) + 0.5) as i32
}
/// `silk_LIMIT` / `silk_LIMIT_int` / `silk_LIMIT_32`.
#[inline(always)]
#[must_use]
pub const fn silk_limit(a: i32, limit1: i32, limit2: i32) -> i32 {
    if limit1 > limit2 {
        if a > limit1 {
            limit1
        } else if a < limit2 {
            limit2
        } else {
            a
        }
    } else if a > limit2 {
        limit2
    } else if a < limit1 {
        limit1
    } else {
        a
    }
}
/// `silk_LIMIT_int`.
#[inline(always)]
#[must_use]
pub const fn silk_limit_int(a: i32, limit1: i32, limit2: i32) -> i32 {
    silk_limit(a, limit1, limit2)
}
/// `silk_LIMIT_16`.
#[inline(always)]
#[must_use]
pub const fn silk_limit_16(a: i16, limit1: i16, limit2: i16) -> i16 {
    silk_limit(a as i32, limit1 as i32, limit2 as i32) as i16
}
/// `silk_LIMIT_32`.
#[inline(always)]
#[must_use]
pub const fn silk_limit_32(a: i32, limit1: i32, limit2: i32) -> i32 {
    silk_limit(a, limit1, limit2)
}
/// `silk_LIMIT` on floats (used by the FLP encoder with float operands).
#[inline(always)]
#[must_use]
pub const fn silk_limit_f32(a: f32, limit1: f32, limit2: f32) -> f32 {
    if limit1 > limit2 {
        if a > limit1 {
            limit1
        } else if a < limit2 {
            limit2
        } else {
            a
        }
    } else if a > limit2 {
        limit2
    } else if a < limit1 {
        limit1
    } else {
        a
    }
}
/// `silk_abs` (i32). Wraps for `i32::MIN` exactly like two's-complement C.
#[inline(always)]
#[must_use]
pub const fn silk_abs(a: i32) -> i32 {
    if a > 0 { a } else { a.wrapping_neg() }
}
/// `silk_abs` on a 16-bit value (result promoted like C int arithmetic, truncated back).
#[inline(always)]
#[must_use]
pub const fn silk_abs16(a: i16) -> i16 {
    if a > 0 { a } else { a.wrapping_neg() }
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_abs_int32`.
#[inline(always)]
#[must_use]
pub const fn silk_abs_int32(a: i32) -> i32 {
    (a ^ (a >> 31)).wrapping_sub(a >> 31)
}
#[cfg(not(feature = "fixed-point-debug"))]
/// `silk_abs_int64`.
#[inline(always)]
#[must_use]
pub const fn silk_abs_int64(a: i64) -> i64 {
    if a > 0 { a } else { a.wrapping_neg() }
}
/// `silk_CHECK_FIT8`: narrowing (checked with `fixed-point-debug`).
#[cfg(not(feature = "fixed-point-debug"))]
#[inline(always)]
#[must_use]
pub const fn silk_check_fit8(a: i64) -> i8 {
    a as i8
}
/// `silk_CHECK_FIT16`: narrowing (checked with `fixed-point-debug`).
#[cfg(not(feature = "fixed-point-debug"))]
#[inline(always)]
#[must_use]
pub const fn silk_check_fit16(a: i64) -> i16 {
    a as i16
}
/// `silk_CHECK_FIT32`: narrowing (checked with `fixed-point-debug`).
#[cfg(not(feature = "fixed-point-debug"))]
#[inline(always)]
#[must_use]
pub const fn silk_check_fit32(a: i64) -> i32 {
    a as i32
}
/// `silk_sign`.
#[inline(always)]
#[must_use]
pub const fn silk_sign(a: i32) -> i32 {
    if a > 0 {
        1
    } else if a < 0 {
        -1
    } else {
        0
    }
}
/// `silk_RAND`: linear congruential generator (wrapping).
#[inline(always)]
#[must_use]
pub const fn silk_rand(seed: i32) -> i32 {
    silk_mla_ovflw(RAND_INCREMENT, seed, RAND_MULTIPLIER)
}
const_unless_fixed_debug! {
/// `silk_SMMUL`: `(i32)(((i64)a32 * b32) >> 32)`.
#[inline(always)]
#[must_use]
pub const fn silk_smmul(a32: i32, b32: i32) -> i32 {
    silk_rshift64(silk_smull(a32, b32), 32) as i32
}
}

// ---------------------------------------------------------------------------------------------
// Inlines.h
// ---------------------------------------------------------------------------------------------

const_unless_fixed_debug! {
/// `silk_CLZ64`.
#[inline(always)]
#[must_use]
pub const fn silk_clz64(input: i64) -> i32 {
    let in_upper = silk_rshift64(input, 32) as i32;
    if in_upper == 0 {
        // Search in the lower 32 bits.
        32 + silk_clz32(input as i32)
    } else {
        // Search in the upper 32 bits.
        silk_clz32(in_upper)
    }
}
}

/// `silk_CLZ_FRAC`: returns `(leading zeros, 7 bits right after the leading one)`.
#[inline(always)]
#[must_use]
pub const fn silk_clz_frac(input: i32) -> (i32, i32) {
    let lzeros = silk_clz32(input);
    (lzeros, silk_ror32(input, 24 - lzeros) & 0x7f)
}

const_unless_fixed_debug! {
/// `silk_SQRT_APPROX`: approximation of square root.
#[inline]
#[must_use]
pub const fn silk_sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = silk_clz_frac(x);
    let mut y: i32 = if lz & 1 != 0 { 32768 } else { 46214 }; // 46214 = sqrt(2) * 32768
    // Get scaling right.
    y >>= silk_rshift(lz, 1);
    // Increment using fractional part of input.
    silk_smlawb(y, y, silk_smulbb(213, frac_q7))
}
}

const_unless_fixed_debug! {
/// `silk_DIV32_varQ`: a good approximation of `(a32 << Qres) / b32`.
#[inline]
#[must_use]
pub const fn silk_div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    silk_assert!(b32 != 0);
    silk_assert!(qres >= 0);
    // Compute number of bits head room and normalize inputs.
    let a_headrm = silk_clz32(silk_abs(a32)) - 1;
    let mut a32_nrm = silk_lshift(a32, a_headrm); // Q: a_headrm
    let b_headrm = silk_clz32(silk_abs(b32)) - 1;
    let b32_nrm = silk_lshift(b32, b_headrm); // Q: b_headrm
    // Inverse of b32, with 14 bits of precision.
    let b32_inv = silk_div32_16(SILK_INT32_MAX >> 2, silk_rshift(b32_nrm, 16)); // Q: 29 + 16 - b_headrm
    // First approximation.
    let mut result = silk_smulwb(a32_nrm, b32_inv); // Q: 29 + a_headrm - b_headrm
    // Compute residual by subtracting product of denominator and first approximation.
    // It's OK to overflow because the final value of a32_nrm should always be small.
    a32_nrm = silk_sub32_ovflw(a32_nrm, silk_lshift_ovflw(silk_smmul(b32_nrm, result), 3)); // Q: a_headrm
    // Refinement.
    result = silk_smlawb(result, a32_nrm, b32_inv); // Q: 29 + a_headrm - b_headrm
    // Convert to Qres domain.
    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        silk_rshift(result, lshift)
    } else {
        // Avoid undefined result.
        0
    }
}
}

const_unless_fixed_debug! {
/// `silk_INVERSE32_varQ`: a good approximation of `(1 << Qres) / b32`.
#[inline]
#[must_use]
pub const fn silk_inverse32_varq(b32: i32, qres: i32) -> i32 {
    silk_assert!(b32 != 0);
    silk_assert!(qres > 0);
    // Compute number of bits head room and normalize input.
    let b_headrm = silk_clz32(silk_abs(b32)) - 1;
    let b32_nrm = silk_lshift(b32, b_headrm); // Q: b_headrm
    // Inverse of b32, with 14 bits of precision.
    let b32_inv = silk_div32_16(SILK_INT32_MAX >> 2, silk_rshift(b32_nrm, 16)); // Q: 29 + 16 - b_headrm
    // First approximation.
    let mut result = silk_lshift(b32_inv, 16); // Q: 61 - b_headrm
    // Compute residual by subtracting product of denominator and first approximation from one.
    let err_q32 = silk_lshift((1i32 << 29) - silk_smulwb(b32_nrm, b32_inv), 3); // Q32
    // Refinement.
    result = silk_smlaww(result, err_q32, b32_inv); // Q: 61 - b_headrm
    // Convert to Qres domain.
    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        silk_rshift(result, lshift)
    } else {
        // Avoid undefined result.
        0
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 32-bit fallback forms from macros.h, used to confirm equivalence with the 64-bit forms.
    const fn smulwb_32(a32: i32, b32: i32) -> i32 {
        ((a32 >> 16) * (b32 as i16) as i32) + (((a32 & 0xFFFF) * (b32 as i16) as i32) >> 16)
    }

    #[test]
    fn smulwb_forms_agree() {
        let mut x: u32 = 0x9E37_79B9;
        for _ in 0..500_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let a = x as i32;
            let b = x.rotate_left(11) as i32;
            assert_eq!(silk_smulwb(a, b), smulwb_32(a, b));
        }
    }

    #[test]
    fn clz() {
        assert_eq!(silk_clz32(0), 32);
        assert_eq!(silk_clz32(1), 31);
        assert_eq!(silk_clz32(-1), 0);
        assert_eq!(silk_clz16(0), 16);
        assert_eq!(silk_clz16(1), 15);
        assert_eq!(silk_clz16(-1), 0);
        assert_eq!(silk_clz64(1), 63);
        assert_eq!(silk_clz64(0), 64);
    }

    #[test]
    fn rshift_round() {
        assert_eq!(silk_rshift_round(3, 1), 2);
        assert_eq!(silk_rshift_round(-3, 1), -1);
        assert_eq!(silk_rshift_round(6, 2), 2);
    }

    #[test]
    fn fix_const() {
        assert_eq!(silk_fix_const(0.5, 16), 32768);
        assert_eq!(silk_fix_const(1.0, 14), 16384);
    }

    #[test]
    fn ror() {
        assert_eq!(silk_ror32(1, 1), i32::MIN);
        assert_eq!(silk_ror32(1, -1), 2);
    }
}

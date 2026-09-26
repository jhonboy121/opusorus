//! Port of `celt/entcode.h`, `celt/entcode.c` and `celt/mfrngcod.h`: shared range-coder
//! definitions.
//!
//! The C code uses one `ec_ctx` for both directions. The port splits it into
//! [`EcEnc`](crate::celt::entenc::EcEnc) (owns `&mut [u8]`) and
//! [`EcDec`](crate::celt::entdec::EcDec) (borrows `&[u8]`), and provides [`EcCoder`] for code
//! paths such as `quant_all_bands` that take an `ec_ctx*` plus an `encode` flag.

use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;

/// `ec_window`.
pub type EcWindow = u32;
/// `EC_WINDOW_SIZE`.
pub const EC_WINDOW_SIZE: i32 = 32;
/// `EC_UINT_BITS`: number of bits to output at a time in `ec_enc_uint`/`ec_dec_uint`.
pub const EC_UINT_BITS: i32 = 8;
/// `BITRES`: resolution of fractional-precision bit usage measurements (1/8th bits).
pub const BITRES: i32 = 3;

/// `EC_SYM_BITS`.
pub const EC_SYM_BITS: i32 = 8;
/// `EC_CODE_BITS`.
pub const EC_CODE_BITS: i32 = 32;
/// `EC_SYM_MAX`.
pub const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1;
/// `EC_CODE_SHIFT`.
pub const EC_CODE_SHIFT: i32 = EC_CODE_BITS - EC_SYM_BITS - 1;
/// `EC_CODE_TOP`.
pub const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
/// `EC_CODE_BOT`.
pub const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
/// `EC_CODE_EXTRA`.
pub const EC_CODE_EXTRA: i32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;

/// `EC_ILOG` / `ec_ilog`: number of bits needed to represent `v` (0 for 0).
#[inline(always)]
#[must_use]
pub const fn ec_ilog(v: u32) -> i32 {
    32 - v.leading_zeros() as i32
}

/// `celt_udiv` (no small-div-table variant; plain division is what the non-ARM-asm C build uses).
#[inline(always)]
#[must_use]
pub const fn celt_udiv(n: u32, d: u32) -> u32 {
    debug_assert!(d > 0);
    n / d
}

/// `celt_sudiv`.
#[inline(always)]
#[must_use]
pub const fn celt_sudiv(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    n / d
}

/// Common range-coder state accessors shared by encoder and decoder (the parts of `ec_ctx`
/// that `ec_tell`/`ec_tell_frac` need).
pub trait EcState {
    /// `nbits_total`.
    fn nbits_total(&self) -> i32;
    /// `rng`.
    fn rng(&self) -> u32;

    /// Port of `ec_tell`: number of bits "used" so far, rounded up.
    #[inline(always)]
    fn tell(&self) -> i32 {
        self.nbits_total() - ec_ilog(self.rng())
    }

    /// Port of `ec_tell_frac`: bits used scaled by `2**BITRES`.
    #[inline]
    fn tell_frac(&self) -> u32 {
        ec_tell_frac(self.nbits_total(), self.rng())
    }
}

/// Port of `ec_tell_frac` (the table-driven `#if 1` variant).
#[inline]
#[must_use]
pub const fn ec_tell_frac(nbits_total: i32, rng: u32) -> u32 {
    const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
    let nbits: u32 = (nbits_total as u32) << BITRES;
    let mut l: i32 = ec_ilog(rng);
    let r: u32 = rng >> (l - 16);
    let mut b: u32 = (r >> 12) - 8;
    b += (r > CORRECTION[b as usize]) as u32;
    l = (l << 3) + b as i32;
    nbits.wrapping_sub(l as u32)
}

/// A range coder in either direction, for shared encode/decode code paths
/// (C functions taking `ec_ctx *ec` together with an `encode` flag).
#[derive(Debug)]
pub enum EcCoder<'r, 'a> {
    /// Encoding.
    Enc(&'r mut EcEnc<'a>),
    /// Decoding.
    Dec(&'r mut EcDec<'a>),
}

impl<'a> EcCoder<'_, 'a> {
    /// True if encoding (the C `encode` flag).
    #[inline(always)]
    #[must_use]
    pub const fn is_encoder(&self) -> bool {
        matches!(self, Self::Enc(_))
    }
    /// `ec_tell`.
    #[inline(always)]
    #[must_use]
    pub fn tell(&self) -> i32 {
        match self {
            Self::Enc(e) => e.tell(),
            Self::Dec(d) => d.tell(),
        }
    }
    /// `ec_tell_frac`.
    #[inline(always)]
    #[must_use]
    pub fn tell_frac(&self) -> u32 {
        match self {
            Self::Enc(e) => e.tell_frac(),
            Self::Dec(d) => d.tell_frac(),
        }
    }
    /// `ec->storage`.
    #[inline(always)]
    #[must_use]
    pub const fn storage(&self) -> u32 {
        match self {
            Self::Enc(e) => e.storage,
            Self::Dec(d) => d.storage,
        }
    }
    /// `ec->rng`.
    #[inline(always)]
    #[must_use]
    pub const fn rng(&self) -> u32 {
        match self {
            Self::Enc(e) => e.rng,
            Self::Dec(d) => d.rng,
        }
    }
    /// Reborrows with a shorter lifetime (like passing `ec` down in C).
    #[inline(always)]
    pub const fn reborrow(&mut self) -> EcCoder<'_, 'a> {
        match self {
            Self::Enc(e) => EcCoder::Enc(e),
            Self::Dec(d) => EcCoder::Dec(d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilog_matches_reference_loop() {
        // Reference: the portable branchless C fallback.
        const fn ec_ilog_ref(mut v: u32) -> i32 {
            let mut ret = (v != 0) as i32;
            let mut m = ((v & 0xFFFF_0000 != 0) as i32) << 4;
            v >>= m;
            ret |= m;
            m = ((v & 0xFF00 != 0) as i32) << 3;
            v >>= m;
            ret |= m;
            m = ((v & 0xF0 != 0) as i32) << 2;
            v >>= m;
            ret |= m;
            m = ((v & 0xC != 0) as i32) << 1;
            v >>= m;
            ret |= m;
            ret += (v & 0x2 != 0) as i32;
            ret
        }
        let mut x: u32 = 0x1234_5678;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            assert_eq!(ec_ilog(x), ec_ilog_ref(x));
            assert_eq!(ec_ilog(x >> (x & 31)), ec_ilog_ref(x >> (x & 31)));
        }
        assert_eq!(ec_ilog(0), 0);
        assert_eq!(ec_ilog(1), 1);
        assert_eq!(ec_ilog(u32::MAX), 32);
    }
}

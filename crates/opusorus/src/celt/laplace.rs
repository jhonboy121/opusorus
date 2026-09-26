//! Port of celt/laplace.c, celt/laplace.h: Laplace-distributed symbol coding for the coarse
//! energy deltas (and the geometric `_p0` variant used by the QEXT energy coder).

use crate::celt::arch::{imax, imin};
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;

/// The minimum probability of an energy delta (out of 32768), as a log2 (`LAPLACE_LOG_MINP`).
const LAPLACE_LOG_MINP: u32 = 0;
/// `LAPLACE_MINP`.
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
/// The minimum number of guaranteed representable energy deltas, in one direction
/// (`LAPLACE_NMIN`).
const LAPLACE_NMIN: u32 = 16;

/// Port of celt/laplace.c:ec_laplace_get_freq1.
///
/// When called, `decay` is positive and at most 11456.
#[inline]
#[must_use]
const fn ec_laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft: u32 = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    // C: `ft*(opus_int32)(16384-decay)>>15` — the product is computed in `unsigned`.
    ft.wrapping_mul((16384 - decay) as u32) >> 15
}

/// Port of celt/laplace.c:ec_laplace_encode.
///
/// Encodes `*value` with a Laplace-like distribution whose probability of 0 is `fs/32768` and
/// whose decay (probability of ±1, times 16384) is `decay`. If the value is not representable
/// it is clamped, and `*value` is updated to the value actually coded.
pub fn ec_laplace_encode(enc: &mut EcEnc<'_>, value: &mut i32, mut fs: u32, decay: i32) {
    let mut fl: u32 = 0;
    let mut val = *value;
    if val != 0 {
        let s: i32 = -i32::from(val < 0);
        val = (val + s) ^ s;
        fl = fs;
        fs = ec_laplace_get_freq1(fs, decay);
        // Search the decaying part of the PDF.
        let mut i: i32 = 1;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            // C: `(fs*(opus_int32)decay)>>15` — unsigned multiply.
            fs = fs.wrapping_mul(decay as u32) >> 15;
            i += 1;
        }
        // Everything beyond that has probability LAPLACE_MINP.
        if fs == 0 {
            let mut ndi_max: i32 =
                ((32768u32.wrapping_sub(fl) + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
            ndi_max = (ndi_max - s) >> 1;
            let di = imin(val - i, ndi_max - 1);
            fl = fl.wrapping_add(((2 * di + 1 + s) as u32).wrapping_mul(LAPLACE_MINP));
            // C: `IMIN(LAPLACE_MINP, 32768-fl)` — an unsigned comparison.
            fs = umin(LAPLACE_MINP, 32768u32.wrapping_sub(fl));
            *value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            fl += fs & !(s as u32);
        }
        celt_assert!(fl + fs <= 32768);
        celt_assert!(fs > 0);
    }
    enc.encode_bin(fl, fl + fs, 15);
}

/// Port of celt/laplace.c:ec_laplace_decode.
#[must_use]
pub fn ec_laplace_decode(dec: &mut EcDec<'_>, mut fs: u32, decay: i32) -> i32 {
    let mut val: i32 = 0;
    let fm: u32 = dec.decode_bin(15);
    let mut fl: u32 = 0;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = ec_laplace_get_freq1(fs, decay) + LAPLACE_MINP;
        // Search the decaying part of the PDF.
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (fs - 2 * LAPLACE_MINP).wrapping_mul(decay as u32) >> 15;
            fs += LAPLACE_MINP;
            val += 1;
        }
        // Everything beyond that has probability LAPLACE_MINP.
        if fs <= LAPLACE_MINP {
            let di: i32 = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
            val += di;
            fl += (2 * di) as u32 * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    celt_assert!(fl < 32768);
    celt_assert!(fs > 0);
    celt_assert!(fl <= fm);
    celt_assert!(fm < umin(fl + fs, 32768));
    dec.update(fl, umin(fl + fs, 32768), 32768);
    val
}

/// `IMIN` evaluated on `unsigned` operands (C's usual arithmetic conversions when one side is
/// `unsigned`).
#[inline(always)]
const fn umin(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

/// Builds the `_p0` magnitude ICDF (shared by encoder and decoder).
#[inline]
const fn laplace_p0_icdf(decay: u16) -> [u16; 8] {
    let mut icdf = [0u16; 8];
    icdf[0] = imax(7, decay as i32) as u16;
    let mut i = 1;
    while i < 7 {
        icdf[i] = imax(7 - i as i32, (icdf[i - 1] as i32 * decay as i32) >> 15) as u16;
        i += 1;
    }
    icdf[7] = 0;
    icdf
}

/// Builds the `_p0` sign ICDF.
#[inline]
const fn laplace_p0_sign_icdf(p0: u16) -> [u16; 3] {
    let s0 = (32768 - p0 as i32) as u16;
    [s0, s0 / 2, 0]
}

/// Port of celt/laplace.c:ec_laplace_encode_p0.
pub const fn ec_laplace_encode_p0(enc: &mut EcEnc<'_>, mut value: i32, p0: u16, decay: u16) {
    let sign_icdf = laplace_p0_sign_icdf(p0);
    let s: usize = if value == 0 {
        0
    } else if value > 0 {
        1
    } else {
        2
    };
    enc.enc_icdf16(s, &sign_icdf, 15);
    value = value.abs();
    if value != 0 {
        let icdf = laplace_p0_icdf(decay);
        value -= 1;
        loop {
            enc.enc_icdf16(imin(value, 7) as usize, &icdf, 15);
            value -= 7;
            if value < 0 {
                break;
            }
        }
    }
}

/// Port of celt/laplace.c:ec_laplace_decode_p0.
#[must_use]
pub const fn ec_laplace_decode_p0(dec: &mut EcDec<'_>, p0: u16, decay: u16) -> i32 {
    let sign_icdf = laplace_p0_sign_icdf(p0);
    let mut s = dec.dec_icdf16(&sign_icdf, 15) as i32;
    if s == 2 {
        s = -1;
    }
    if s != 0 {
        let icdf = laplace_p0_icdf(decay);
        let mut value: i32 = 1;
        loop {
            let v = dec.dec_icdf16(&icdf, 15) as i32;
            value += v;
            if v != 7 {
                break;
            }
        }
        s * value
    } else {
        0
    }
}

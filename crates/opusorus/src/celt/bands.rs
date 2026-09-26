//! Port of celt/bands.c, celt/bands.h: band energy computation and normalisation, the band
//! shape (PVQ) quantisation driver `quant_all_bands` with its recursive band splitting, stereo
//! coupling and spectral folding, anti-collapse and spreading decisions.
//!
//! Float build. Only the portable C paths are ported (no `MEASURE_NORM_MSE`, no `FUZZING`
//! random decisions, `DISABLE_UPDATE_DRAFT` is not defined upstream so the "update draft"
//! folding is the one ported, `RESYNTH` is not defined).
//!
//! # Buffer aliasing
//!
//! The C code passes raw pointers that sometimes overlap. The port reproduces the exact memory
//! effects:
//! * the decoder's `lowband_scratch` points into the tail of `X_` (from band
//!   `effEBands-1`); the port uses that tail as scratch whenever the current band lies before
//!   it, like C. When the band itself is that tail (band `effEBands-1` when it is not the last
//!   band, e.g. the QEXT bands of a 96 kHz stream decoded at 48 kHz, where the QEXT mode has
//!   `effEBands = 2 < qext_end`), C aliases scratch and band: quant_band copies the lowband
//!   over `X` and transforms that one buffer (`quant_band`'s `x_alias` reproduces it), and in
//!   dual stereo quant_band(`Y`) copies its lowband over the `X` band decoded just before (the
//!   write-back after the `Y` call);
//! * the hybrid folding band (`start+1`) may read a `lowband` that overlaps the `lowband_out`
//!   it writes; all `lowband` accesses happen before the `lowband_out` writes, so the port uses
//!   copies and writes them back in that order;
//! * for bands `i >= effEBands` (unreachable through the public API: `end <= effEBands` for
//!   every mode the codec uses) C quantises `norm` itself as `X` and `Y`. The port quantises a
//!   copy of it (written back to `norm` for mono); the bitstream and decoder side effects
//!   match, but the encoder input for those bands can differ from C.

#![allow(
    clippy::too_many_arguments,
    reason = "band quantisation functions mirror the C signatures"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
#![allow(
    clippy::precedence,
    reason = "shift/arithmetic expressions are copied verbatim from C, whose precedence rules \
              for `*`, `+` and `<<` are the same as Rust's"
)]

use alloc::vec::Vec;
#[cfg(not(feature = "qext"))]
use core::marker::PhantomData;

use crate::celt::arch::{
    CeltEner, CeltGlog, CeltNorm, CeltSig, EPSILON, NORM_SCALING, OpusVal16, OpusVal32, Q31ONE,
    add32, div32_16, extend32, half32, imax, imin, mac16_16, max32, maxg, min16, min32, ming,
    mult16_16, mult16_16_q15, mult16_32_q15, mult32_32_q31, qconst16, qconst32, shl32, shr32,
    sub32, vshr32,
};
use crate::celt::entcode::{BITRES, EcCoder, celt_sudiv, celt_udiv, ec_ilog};
use crate::celt::entenc::EcEncSnapshot;
#[cfg(feature = "qext")]
use crate::celt::mathops::celt_cos_norm2;
use crate::celt::mathops::{
    celt_exp2, celt_exp2_db, celt_rsqrt, celt_rsqrt_norm32, celt_sqrt, frac_mul16, isqrt32,
};
use crate::celt::pitch::celt_inner_prod;
use crate::celt::quant_bands::E_MEANS;
use crate::celt::rate::{
    QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, bits2pulses, get_pulses, pulses2bits,
};
use crate::celt::static_modes::CeltMode;
use crate::celt::vq::{
    MAX_BAND_SIZE, Scratch, alg_quant, alg_unquant, renormalise_vector, stereo_itheta,
};
#[cfg(feature = "qext")]
use crate::celt::vq::{cubic_quant, cubic_unquant};

/// `SPREAD_NONE`.
pub const SPREAD_NONE: i32 = 0;
/// `SPREAD_LIGHT`.
pub const SPREAD_LIGHT: i32 = 1;
/// `SPREAD_NORMAL`.
pub const SPREAD_NORMAL: i32 = 2;
/// `SPREAD_AGGRESSIVE`.
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// `MIN_STEREO_ENERGY` (float build).
const MIN_STEREO_ENERGY: f32 = 1e-10;

/// Port of celt/bands.c:hysteresis_decision.
#[must_use]
pub fn hysteresis_decision(
    val: OpusVal16,
    thresholds: &[OpusVal16],
    hysteresis: &[OpusVal16],
    n: i32,
    prev: i32,
) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        if val < thresholds[i as usize] {
            break;
        }
        i += 1;
    }
    let p = prev as usize;
    if i > prev && val < thresholds[p] + hysteresis[p] {
        i = prev;
    }
    if i < prev && val > thresholds[p - 1] - hysteresis[p - 1] {
        i = prev;
    }
    i
}

/// Port of celt/bands.c:celt_lcg_rand.
#[inline(always)]
#[must_use]
pub const fn celt_lcg_rand(seed: u32) -> u32 {
    1664525u32.wrapping_mul(seed).wrapping_add(1013904223)
}

/// Port of celt/bands.c:bitexact_cos: a cos() approximation designed to be bit-exact on any
/// platform. Bit exactness with this approximation is important because it has an impact on
/// the bit allocation.
#[must_use]
pub const fn bitexact_cos(x: i16) -> i16 {
    let tmp: i32 = (4096 + (x as i32) * (x as i32)) >> 13;
    debug_assert!(tmp <= 32767);
    let mut x2: i16 = tmp as i16;
    let x2i = x2 as i32;
    x2 = ((32767 - x2i) + frac_mul16(x2i, -7651 + frac_mul16(x2i, 8277 + frac_mul16(-626, x2i))))
        as i16;
    debug_assert!(x2 <= 32766);
    (1 + x2 as i32) as i16
}

/// Port of celt/bands.c:bitexact_log2tan.
#[must_use]
pub const fn bitexact_log2tan(mut isin: i32, mut icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Port of celt/bands.c:compute_band_energies (float build): the amplitude (sqrt energy) of
/// each band `0..end` of each channel.
pub fn compute_band_energies(
    m: &CeltMode,
    x: &[CeltSig],
    band_e: &mut [CeltEner],
    end: i32,
    c: i32,
    lm: i32,
) {
    let e_bands = &m.e_bands;
    let n = m.short_mdct_size << lm;
    if c == 2 {
        // Perf: stereo computes the two channels' sums of each band in one loop (two
        // independent dependency chains); each sum has the operations of
        // `celt_inner_prod`, in the same order.
        let nb = m.nb_ebands as usize;
        for i in 0..end as usize {
            let off = (i32::from(e_bands[i]) << lm) as usize;
            let len = ((i32::from(e_bands[i + 1]) - i32::from(e_bands[i])) << lm) as usize;
            let x0 = &x[off..off + len];
            let x1 = &x[n as usize + off..n as usize + off + len];
            let mut xy0: OpusVal32 = 0.0;
            let mut xy1: OpusVal32 = 0.0;
            for (&a, &b) in x0.iter().zip(x1) {
                xy0 = mac16_16(xy0, a, a);
                xy1 = mac16_16(xy1, b, b);
            }
            band_e[i] = celt_sqrt(1e-27f32 + xy0);
            band_e[i + nb] = celt_sqrt(1e-27f32 + xy1);
        }
        return;
    }
    for ch in 0..c {
        for i in 0..end {
            let iu = i as usize;
            let off = (ch * n + (i32::from(e_bands[iu]) << lm)) as usize;
            let len = ((i32::from(e_bands[iu + 1]) - i32::from(e_bands[iu])) << lm) as usize;
            let xs = &x[off..off + len];
            let sum: OpusVal32 = 1e-27f32 + celt_inner_prod(xs, xs, len);
            band_e[(i + ch * m.nb_ebands) as usize] = celt_sqrt(sum);
        }
    }
}

/// Port of celt/bands.c:normalise_bands (float build): normalises each band such that the
/// energy is one.
pub fn normalise_bands(
    m: &CeltMode,
    freq: &[CeltSig],
    x: &mut [CeltNorm],
    band_e: &[CeltEner],
    end: i32,
    c: i32,
    mm: i32,
) {
    let e_bands = &m.e_bands;
    let n = mm * m.short_mdct_size;
    for ch in 0..c {
        for i in 0..end {
            let iu = i as usize;
            let g: OpusVal16 = 1.0f32 / (1e-27f32 + band_e[(i + ch * m.nb_ebands) as usize]);
            let lo = (mm * i32::from(e_bands[iu]) + ch * n) as usize;
            let hi = (mm * i32::from(e_bands[iu + 1]) + ch * n) as usize;
            for j in lo..hi {
                x[j] = freq[j] * g;
            }
        }
    }
}

/// Port of celt/bands.c:denormalise_bands: de-normalises the energy to produce the synthesis
/// from the unit-energy bands. `freq` receives `M*shortMdctSize` samples.
pub fn denormalise_bands(
    m: &CeltMode,
    x: &[CeltNorm],
    freq: &mut [CeltSig],
    band_log_e: &[CeltGlog],
    mut start: i32,
    mut end: i32,
    mm: i32,
    downsample: i32,
    silence: bool,
) {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let n = mm * m.short_mdct_size;
    let mut bound = mm * eb(end);
    if downsample != 1 {
        bound = imin(bound, n / downsample);
    }
    if silence {
        bound = 0;
        start = 0;
        end = 0;
    }
    let mut f: usize = 0;
    let mut xi: usize = (mm * eb(start)) as usize;
    if start != 0 {
        for _ in 0..mm * eb(start) {
            freq[f] = 0.0;
            f += 1;
        }
    } else {
        f += (mm * eb(start)) as usize;
    }
    for i in start..end {
        let band_end = mm * eb(i + 1);
        let lg: CeltGlog = add32(band_log_e[i as usize], shl32(E_MEANS[i as usize], 24 - 4));
        let g: OpusVal32 = celt_exp2_db(min32(32.0, lg));
        // FIXED_POINT: integer/fractional split not ported (float build).
        let mut j = mm * eb(i);
        loop {
            freq[f] = mult32_32_q31(shl32(x[xi], 0), g);
            f += 1;
            xi += 1;
            j += 1;
            if j >= band_end {
                break;
            }
        }
    }
    debug_assert!(start <= end);
    freq[bound as usize..n as usize].fill(0.0);
}

/// Port of celt/bands.c:anti_collapse: prevents energy collapse for transients with multiple
/// short MDCTs by filling collapsed blocks with noise. `x_` holds `c` channels of `size`
/// samples.
pub fn anti_collapse(
    m: &CeltMode,
    x_: &mut [CeltNorm],
    collapse_masks: &[u8],
    lm: i32,
    c: i32,
    size: i32,
    start: i32,
    end: i32,
    log_e: &[CeltGlog],
    prev1log_e: &[CeltGlog],
    prev2log_e: &[CeltGlog],
    pulses: &[i32],
    mut seed: u32,
    encode: bool,
) {
    let nb = m.nb_ebands;
    for i in start..end {
        let iu = i as usize;
        let n0 = i32::from(m.e_bands[iu + 1]) - i32::from(m.e_bands[iu]);
        // depth in 1/8 bits
        debug_assert!(pulses[iu] >= 0);
        let depth: i32 = (celt_udiv((1 + pulses[iu]) as u32, n0 as u32) >> lm) as i32;

        // FIXED_POINT: not ported (float build).
        let thresh: OpusVal16 = 0.5f32 * celt_exp2(-0.125f32 * depth as f32);
        let sqrt_1: OpusVal16 = celt_rsqrt((n0 << lm) as f32);

        for ch in 0..c {
            let mut renormalize = false;
            let mut prev1: CeltGlog = prev1log_e[(ch * nb + i) as usize];
            let mut prev2: CeltGlog = prev2log_e[(ch * nb + i) as usize];
            if !encode && c == 1 {
                prev1 = maxg(prev1, prev1log_e[(nb + i) as usize]);
                prev2 = maxg(prev2, prev2log_e[(nb + i) as usize]);
            }
            let mut ediff: OpusVal32 = log_e[(ch * nb + i) as usize] - ming(prev1, prev2);
            ediff = max32(0.0, ediff);

            // r needs to be multiplied by 2 or 2*sqrt(2) depending on LM because short blocks
            // don't have the same energy as long
            let mut r: CeltNorm = 2.0f32 * celt_exp2_db(-ediff);
            if lm == 3 {
                r *= 1.41421356f32;
            }
            r = min16(thresh, r);
            r *= sqrt_1;
            let xoff = (ch * size + (i32::from(m.e_bands[iu]) << lm)) as usize;
            let x = &mut x_[xoff..];
            for k in 0..(1 << lm) {
                // Detect collapse
                if i32::from(collapse_masks[(i * c + ch) as usize]) & (1 << k) == 0 {
                    // Fill with noise
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x[((j << lm) + k) as usize] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            // We just added some energy, so we need to renormalise
            if renormalize {
                renormalise_vector(x, n0 << lm, Q31ONE);
            }
        }
    }
}

/// Port of celt/bands.c:compute_channel_weights: weights for optimizing normalized distortion
/// across channels (amplitude-weighted square distortion).
#[must_use]
pub fn compute_channel_weights(mut ex: CeltEner, mut ey: CeltEner) -> [OpusVal16; 2] {
    let min_e: CeltEner = min32(ex, ey);
    // Adjustment to make the weights a bit more conservative.
    ex = add32(ex, min_e / 3.0);
    ey = add32(ey, min_e / 3.0);
    // FIXED_POINT: shift not ported (float build).
    [vshr32(ex, 0), vshr32(ey, 0)]
}

/// Port of celt/bands.c:intensity_stereo: `X = a1*X + a2*Y` (side is not coded).
pub fn intensity_stereo(
    m: &CeltMode,
    x: &mut [CeltNorm],
    y: &[CeltNorm],
    band_e: &[CeltEner],
    band_id: i32,
    n: i32,
) {
    let i = band_id as usize;
    // FIXED_POINT: shift not ported (float build).
    let left: OpusVal16 = vshr32(band_e[i], 0);
    let right: OpusVal16 = vshr32(band_e[i + m.nb_ebands as usize], 0);
    let norm: OpusVal16 =
        EPSILON + celt_sqrt(EPSILON + mult16_16(left, left) + mult16_16(right, right));
    let a1: OpusVal16 = div32_16(shl32(extend32(left), 15), norm);
    let a2: OpusVal16 = div32_16(shl32(extend32(right), 15), norm);
    let n = n as usize;
    for j in 0..n {
        x[j] = add32(mult16_32_q15(a1, x[j]), mult16_32_q15(a2, y[j]));
        // Side is not encoded, no need to calculate
    }
}

/// Port of celt/bands.c:stereo_split: L/R to M/S.
pub fn stereo_split(x: &mut [CeltNorm], y: &mut [CeltNorm], n: i32) {
    let n = n as usize;
    for j in 0..n {
        let l: OpusVal32 = mult32_32_q31(qconst32(0.70710678, 31), x[j]);
        let r: OpusVal32 = mult32_32_q31(qconst32(0.70710678, 31), y[j]);
        x[j] = add32(l, r);
        y[j] = sub32(r, l);
    }
}

/// Port of celt/bands.c:stereo_merge: M/S to L/R with the mid gain `mid`.
pub fn stereo_merge(x: &mut [CeltNorm], y: &mut [CeltNorm], mid: OpusVal32, n: i32) {
    let nu = n as usize;
    // Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy)
    let mut xp: OpusVal32 = celt_inner_prod(y, x, nu);
    let side: OpusVal32 = celt_inner_prod(y, y, nu);
    // Compensating for the mid normalization
    xp = mult32_32_q31(mid, xp);
    // mid and side are in Q15, not Q14 like X and Y
    let el: OpusVal32 = shr32(mult32_32_q31(mid, mid), 3) + side - 2.0 * xp;
    let er: OpusVal32 = shr32(mult32_32_q31(mid, mid), 3) + side + 2.0 * xp;
    if er < qconst32(6e-4, 28) || el < qconst32(6e-4, 28) {
        y[..nu].copy_from_slice(&x[..nu]);
        return;
    }

    // FIXED_POINT: kl/kr not ported (float build).
    let t: OpusVal32 = vshr32(el, 0);
    let lgain: OpusVal32 = celt_rsqrt_norm32(t);
    let t: OpusVal32 = vshr32(er, 0);
    let rgain: OpusVal32 = celt_rsqrt_norm32(t);

    for j in 0..nu {
        // Apply mid scaling (side is already scaled)
        let l: CeltNorm = mult32_32_q31(mid, x[j]);
        let r: CeltNorm = y[j];
        x[j] = vshr32(mult32_32_q31(lgain, sub32(l, r)), 0);
        y[j] = vshr32(mult32_32_q31(rgain, add32(l, r)), 0);
    }
}

/// Port of celt/bands.c:spreading_decision: decides whether we should spread the pulses in the
/// current frame (and updates the tapset decision when `update_hf`).
pub fn spreading_decision(
    m: &CeltMode,
    x: &[CeltNorm],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: i32,
    c: i32,
    mm: i32,
    spread_weight: &[i32],
) -> i32 {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut sum: i32 = 0;
    let mut nb_bands: i32 = 0;
    let mut hf_sum: i32 = 0;

    debug_assert!(end > 0);

    let n0 = mm * m.short_mdct_size;

    if mm * (eb(end) - eb(end - 1)) <= 8 {
        return SPREAD_NONE;
    }
    for ch in 0..c {
        for i in 0..end {
            let mut tcount: [i32; 3] = [0, 0, 0];
            let off = (mm * eb(i) + ch * n0) as usize;
            let n = mm * (eb(i + 1) - eb(i));
            if n <= 8 {
                continue;
            }
            let xs = &x[off..off + n as usize];
            // Compute rough CDF of |x[j]|
            for &xj in xs {
                // Q13
                let x2n: OpusVal32 = mult16_16(mult16_16_q15(shr32(xj, 0), shr32(xj, 0)), n as f32);
                if x2n < qconst16(0.25, 13) {
                    tcount[0] += 1;
                }
                if x2n < qconst16(0.0625, 13) {
                    tcount[1] += 1;
                }
                if x2n < qconst16(0.015625, 13) {
                    tcount[2] += 1;
                }
            }

            // Only include four last bands (8 kHz and up)
            if i > m.nb_ebands - 4 {
                hf_sum += celt_udiv((32 * (tcount[1] + tcount[0])) as u32, n as u32) as i32;
            }
            let tmp = i32::from(2 * tcount[2] >= n)
                + i32::from(2 * tcount[1] >= n)
                + i32::from(2 * tcount[0] >= n);
            sum += tmp * spread_weight[i as usize];
            nb_bands += spread_weight[i as usize];
        }
    }

    if update_hf {
        if hf_sum != 0 {
            hf_sum = celt_udiv(hf_sum as u32, (c * (4 - m.nb_ebands + end)) as u32) as i32;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }
    debug_assert!(nb_bands > 0); // end has to be non-zero
    debug_assert!(sum >= 0);
    sum = celt_udiv((sum << 8) as u32, nb_bands as u32) as i32;
    // Recursive averaging
    sum = (sum + *average) >> 1;
    *average = sum;
    // Hysteresis
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/// `ordery_table`: indexing table for converting from natural Hadamard to ordery Hadamard. This
/// is essentially a bit-reversed Gray, on top of which we've added an inversion of the order
/// because we want the DC at the end rather than the beginning. The lines are for N=2, 4, 8,
/// 16.
#[rustfmt::skip]
static ORDERY_TABLE: [usize; 30] = [
    1, 0,
    3, 0, 2, 1,
    7, 0, 4, 3, 6, 1, 5, 2,
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

/// Port of celt/bands.c:deinterleave_hadamard.
pub fn deinterleave_hadamard(x: &mut [CeltNorm], n0: i32, stride: i32, hadamard: bool) {
    debug_assert!(stride > 0);
    let (n0, stride) = (n0 as usize, stride as usize);
    let n = n0 * stride;
    let mut tmp_buf = Scratch::<CeltNorm, MAX_BAND_SIZE>::new();
    let tmp = tmp_buf.get(n);
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

/// Port of celt/bands.c:interleave_hadamard.
pub fn interleave_hadamard(x: &mut [CeltNorm], n0: i32, stride: i32, hadamard: bool) {
    let (n0, stride) = (n0 as usize, stride as usize);
    let n = n0 * stride;
    let mut tmp_buf = Scratch::<CeltNorm, MAX_BAND_SIZE>::new();
    let tmp = tmp_buf.get(n);
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

/// Port of celt/bands.c:haar1.
pub fn haar1(x: &mut [CeltNorm], n0: i32, stride: i32) {
    let n0 = (n0 >> 1) as usize;
    let stride = stride as usize;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1: OpusVal32 = mult32_32_q31(qconst32(0.70710678, 31), x[a]);
            let tmp2: OpusVal32 = mult32_32_q31(qconst32(0.70710678, 31), x[b]);
            x[a] = add32(tmp1, tmp2);
            x[b] = sub32(tmp1, tmp2);
        }
    }
}

/// Port of celt/bands.c:compute_qn: the number of quantisation steps for the split angle.
#[must_use]
pub const fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    // The upper limit ensures that in a stereo split with itheta==16384, we'll always have
    // enough bits left over to code at least one pulse in the side; otherwise it would
    // collapse, since it doesn't get folded.
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = imin(b - pulse_cap - (4 << BITRES), qb);

    qb = imin(8 << BITRES, qb);

    let mut qn: i32;
    if qb < (1 << BITRES >> 1) {
        qn = 1;
    } else {
        qn = EXP2_TABLE8[(qb & 0x7) as usize] as i32 >> (14 - (qb >> BITRES));
        qn = (qn + 1) >> 1 << 1;
    }
    debug_assert!(qn <= 256);
    qn
}

/// `struct band_ctx`: state shared by the recursive band quantisation functions.
struct BandCtx<'c, 'a, 'e> {
    encode: bool,
    resynth: bool,
    m: &'c CeltMode,
    i: i32,
    intensity: i32,
    spread: i32,
    tf_change: i32,
    ec: EcCoder<'c, 'a>,
    remaining_bits: i32,
    band_e: &'c [CeltEner],
    seed: u32,
    theta_round: i32,
    disable_inv: bool,
    avoid_split_noise: bool,
    #[cfg(feature = "qext")]
    ext_ec: EcCoder<'c, 'e>,
    #[cfg(feature = "qext")]
    extra_bits: i32,
    #[cfg(feature = "qext")]
    ext_total_bits: i32,
    #[cfg(feature = "qext")]
    extra_bands: bool,
    #[cfg(not(feature = "qext"))]
    _ext: PhantomData<&'e ()>,
}

/// The scalar fields of a [`BandCtx`] (the C code saves/restores the whole struct by value).
#[derive(Clone, Copy)]
struct BandCtxSave {
    i: i32,
    intensity: i32,
    spread: i32,
    tf_change: i32,
    remaining_bits: i32,
    seed: u32,
    theta_round: i32,
    disable_inv: bool,
    avoid_split_noise: bool,
    #[cfg(feature = "qext")]
    extra_bits: i32,
    #[cfg(feature = "qext")]
    ext_total_bits: i32,
    #[cfg(feature = "qext")]
    extra_bands: bool,
}

impl BandCtx<'_, '_, '_> {
    /// `ctx_save = ctx`.
    const fn save(&self) -> BandCtxSave {
        BandCtxSave {
            i: self.i,
            intensity: self.intensity,
            spread: self.spread,
            tf_change: self.tf_change,
            remaining_bits: self.remaining_bits,
            seed: self.seed,
            theta_round: self.theta_round,
            disable_inv: self.disable_inv,
            avoid_split_noise: self.avoid_split_noise,
            #[cfg(feature = "qext")]
            extra_bits: self.extra_bits,
            #[cfg(feature = "qext")]
            ext_total_bits: self.ext_total_bits,
            #[cfg(feature = "qext")]
            extra_bands: self.extra_bands,
        }
    }

    /// `ctx = ctx_save`.
    const fn restore(&mut self, s: &BandCtxSave) {
        self.i = s.i;
        self.intensity = s.intensity;
        self.spread = s.spread;
        self.tf_change = s.tf_change;
        self.remaining_bits = s.remaining_bits;
        self.seed = s.seed;
        self.theta_round = s.theta_round;
        self.disable_inv = s.disable_inv;
        self.avoid_split_noise = s.avoid_split_noise;
        #[cfg(feature = "qext")]
        {
            self.extra_bits = s.extra_bits;
            self.ext_total_bits = s.ext_total_bits;
            self.extra_bands = s.extra_bands;
        }
    }
}

/// Encoder state + buffer bytes saved for the theta RDO (`ec_save = *ec` plus the bytes
/// `buf[offs..storage]`).
struct EcSave {
    snap: EcEncSnapshot,
    offs: usize,
    storage: usize,
}

/// Snapshots an encoder (`None` for a decoder, which never takes the RDO path).
const fn ec_snapshot(ec: &EcCoder<'_, '_>) -> Option<EcSave> {
    match ec {
        EcCoder::Enc(e) => Some(EcSave {
            snap: e.snapshot(),
            offs: e.offs as usize,
            storage: e.storage as usize,
        }),
        EcCoder::Dec(_) => None,
    }
}

/// Restores the numeric encoder state of a snapshot.
const fn ec_restore(ec: &mut EcCoder<'_, '_>, s: Option<&EcSave>) {
    if let (EcCoder::Enc(e), Some(s)) = (ec, s) {
        e.restore(&s.snap);
    }
}

/// Copies `buf[s.offs..s.storage]` of the encoder into `out` (resized).
fn ec_save_bytes(ec: &EcCoder<'_, '_>, s: Option<&EcSave>, out: &mut Vec<u8>) {
    out.clear();
    if let (EcCoder::Enc(e), Some(s)) = (ec, s)
        && s.storage > s.offs
    {
        out.extend_from_slice(&e.buffer()[s.offs..s.storage]);
    }
}

/// Copies `bytes` back into the encoder buffer at `s.offs`.
fn ec_restore_bytes(ec: &mut EcCoder<'_, '_>, s: Option<&EcSave>, bytes: &[u8]) {
    if let (EcCoder::Enc(e), Some(s)) = (ec, s) {
        e.buffer_mut()[s.offs..s.offs + bytes.len()].copy_from_slice(bytes);
    }
}

/// `struct split_ctx`.
#[derive(Clone, Copy)]
struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    #[cfg(feature = "qext")]
    itheta_q30: i32,
    qalloc: i32,
}

/// Port of celt/bands.c:compute_theta: decides (encoder) and codes the split angle `itheta`
/// between `x` and `y` (mid/side for `stereo`, the two halves of a band otherwise).
fn compute_theta(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    y: &mut [CeltNorm],
    n: i32,
    b: &mut i32,
    blk: i32,
    blk0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
    #[cfg(feature = "qext")] ext_b: &mut i32,
) -> SplitCtx {
    let mut itheta: i32 = 0;
    #[allow(unused_mut, reason = "only updated with qext")]
    let mut itheta_q30: i32 = 0;
    let mut inv = false;
    let encode = ctx.encode;
    let m = ctx.m;
    let i = ctx.i;
    let intensity = ctx.intensity;

    // Decide on the resolution to give to the split parameter theta
    let pulse_cap = i32::from(m.log_n[i as usize]) + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && i >= intensity {
        qn = 1;
    }
    if encode {
        // theta is the atan() of the ratio between the (normalized) side and mid. With just
        // that parameter, we can re-scale both mid and side because we know that 1) they have
        // unit norm and 2) they are orthogonal.
        itheta_q30 = stereo_itheta(x, y, stereo, n);
        itheta = itheta_q30 >> 16;
    }
    let tell: i32 = ctx.ec.tell_frac() as i32;
    if qn != 1 {
        if encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    // Check if the selected value of theta will cause the bit allocation to
                    // inject noise on one side. If so, make sure the energy of that side is
                    // zero.
                    let unquantized = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
                    let imid = i32::from(bitexact_cos(unquantized as i16));
                    let iside = i32::from(bitexact_cos((16384 - unquantized) as i16));
                    let delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                // Bias quantization towards itheta=0 and itheta=16384.
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = imin(qn - 1, imax(0, (itheta * qn + bias) >> 14));
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }
        // Entropy coding of the angle. We use a uniform pdf for the time split, a step for
        // stereo, and a triangular one for the rest.
        if stereo && n > 2 {
            let p0: i32 = 3;
            let mut xv = itheta;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            // Use a probability of p0 up to itheta=8192 and then use 1 after
            let fl_of = |xv: i32| {
                if xv <= x0 {
                    p0 * xv
                } else {
                    (xv - 1 - x0) + (x0 + 1) * p0
                }
            };
            let fh_of = |xv: i32| {
                if xv <= x0 {
                    p0 * (xv + 1)
                } else {
                    (xv - x0) + (x0 + 1) * p0
                }
            };
            match &mut ctx.ec {
                EcCoder::Enc(e) => {
                    e.encode(fl_of(xv) as u32, fh_of(xv) as u32, ft as u32);
                }
                EcCoder::Dec(d) => {
                    let fs = d.decode(ft as u32) as i32;
                    if fs < (x0 + 1) * p0 {
                        xv = fs / p0;
                    } else {
                        xv = x0 + 1 + (fs - (x0 + 1) * p0);
                    }
                    d.update(fl_of(xv) as u32, fh_of(xv) as u32, ft as u32);
                    itheta = xv;
                }
            }
        } else if blk0 > 1 || stereo {
            // Uniform pdf
            match &mut ctx.ec {
                EcCoder::Enc(e) => e.enc_uint(itheta as u32, (qn + 1) as u32),
                EcCoder::Dec(d) => itheta = d.dec_uint((qn + 1) as u32) as i32,
            }
        } else {
            let fs: i32;
            let fl: i32;
            let ft: i32 = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            match &mut ctx.ec {
                EcCoder::Enc(e) => {
                    fs = if itheta <= (qn >> 1) {
                        itheta + 1
                    } else {
                        qn + 1 - itheta
                    };
                    fl = if itheta <= (qn >> 1) {
                        itheta * (itheta + 1) >> 1
                    } else {
                        ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)
                    };
                    e.encode(fl as u32, (fl + fs) as u32, ft as u32);
                }
                EcCoder::Dec(d) => {
                    // Triangular pdf
                    let fm = d.decode(ft as u32) as i32;

                    if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                        itheta = (isqrt32(8 * fm as u32 + 1).wrapping_sub(1) >> 1) as i32;
                        fs = itheta + 1;
                        fl = itheta * (itheta + 1) >> 1;
                    } else {
                        itheta = (((2 * (qn + 1)) as u32)
                            .wrapping_sub(isqrt32(8 * (ft - fm - 1) as u32 + 1))
                            >> 1) as i32;
                        fs = qn + 1 - itheta;
                        fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                    }

                    d.update(fl as u32, (fl + fs) as u32, ft as u32);
                }
            }
        }
        debug_assert!(itheta >= 0);
        itheta = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
        #[cfg(feature = "qext")]
        {
            *ext_b = imin(*ext_b, ctx.ext_total_bits - ctx.ext_ec.tell_frac() as i32);
            if *ext_b >= 2 * n << BITRES
                && (ctx.ext_total_bits as u32)
                    .wrapping_sub(ctx.ext_ec.tell_frac())
                    .wrapping_sub(1)
                    > (2 << BITRES) as u32
            {
                let ext_tell = ctx.ext_ec.tell_frac() as i32;
                let extra_bits = imin(12, imax(2, celt_sudiv(*ext_b, (2 * n - 1) << BITRES)));
                let steps = (1i32 << extra_bits) - 1;
                match &mut ctx.ext_ec {
                    EcCoder::Enc(e) => {
                        itheta_q30 -= itheta << 16;
                        itheta_q30 = ((i64::from(itheta_q30) * i64::from(qn) * i64::from(steps)
                            + (1i64 << 29))
                            >> 30) as i32;
                        itheta_q30 += (1 << (extra_bits - 1)) - 1;
                        itheta_q30 = imax(0, imin((1 << extra_bits) - 2, itheta_q30));
                        e.enc_uint(itheta_q30 as u32, steps as u32);
                    }
                    EcCoder::Dec(d) => {
                        itheta_q30 = d.dec_uint(steps as u32) as i32;
                    }
                }
                itheta_q30 -= (1 << (extra_bits - 1)) - 1;
                itheta_q30 = (i64::from(itheta << 16)
                    + i64::from(itheta_q30) * i64::from(1i32 << 30) / i64::from(qn * steps))
                    as i32;
                // Hard bounds on itheta (can only trigger on corrupted bitstreams).
                itheta_q30 = imax(0, imin(itheta_q30, 1073741824));
                *ext_b = (*ext_b as u32)
                    .wrapping_sub(ctx.ext_ec.tell_frac().wrapping_sub(ext_tell as u32))
                    as i32;
            } else {
                itheta_q30 = itheta << 16;
            }
        }
        if encode && stereo {
            if itheta == 0 {
                intensity_stereo(m, x, y, ctx.band_e, i, n);
            } else {
                stereo_split(x, y, n);
            }
        }
        // NOTE: Renormalising X and Y *may* help fixed-point a bit at very high rate. Let's do
        // that at higher complexity
    } else if stereo {
        if encode {
            inv = itheta > 8192 && !ctx.disable_inv;
            if inv {
                for v in y[..n as usize].iter_mut() {
                    *v = -*v;
                }
            }
            intensity_stereo(m, x, y, ctx.band_e, i, n);
        }
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            match &mut ctx.ec {
                EcCoder::Enc(e) => e.enc_bit_logp(inv, 2),
                EcCoder::Dec(d) => inv = d.dec_bit_logp(2),
            }
        } else {
            inv = false;
        }
        // inv flag override to avoid problems with downmixing.
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
        itheta_q30 = 0;
    }
    let qalloc = ctx.ec.tell_frac().wrapping_sub(tell as u32) as i32;
    *b -= qalloc;

    let (imid, iside, delta): (i32, i32, i32) = if itheta == 0 {
        *fill &= (1 << blk) - 1;
        (32767, 0, -16384)
    } else if itheta == 16384 {
        *fill &= ((1 << blk) - 1) << blk;
        (0, 32767, 16384)
    } else {
        let imid = i32::from(bitexact_cos(itheta as i16));
        let iside = i32::from(bitexact_cos((16384 - itheta) as i16));
        // This is the mid vs side allocation that minimizes squared error in that band.
        (
            imid,
            iside,
            frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid)),
        )
    };

    #[cfg(not(feature = "qext"))]
    let _ = itheta_q30;
    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        #[cfg(feature = "qext")]
        itheta_q30,
        qalloc,
    }
}

/// Port of celt/bands.c:quant_band_n1: the special case for one-sample bands (just a sign per
/// channel).
fn quant_band_n1(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    y: Option<&mut [CeltNorm]>,
    lowband_out: Option<&mut [CeltNorm]>,
) -> u32 {
    fn code(ctx: &mut BandCtx<'_, '_, '_>, xs: &mut [CeltNorm]) {
        let mut sign = false;
        if ctx.remaining_bits >= 1 << BITRES {
            match &mut ctx.ec {
                EcCoder::Enc(e) => {
                    sign = xs[0] < 0.0;
                    e.enc_bits(u32::from(sign), 1);
                }
                EcCoder::Dec(d) => {
                    sign = d.dec_bits(1) != 0;
                }
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            xs[0] = if sign { -NORM_SCALING } else { NORM_SCALING };
        }
    }
    code(ctx, x);
    if let Some(y) = y {
        code(ctx, y);
    }
    if let Some(lo) = lowband_out {
        lo[0] = shr32(x[0], 4);
    }
    1
}

/// Port of celt/bands.c:quant_partition: encodes or decodes a mono partition, possibly
/// splitting it in two recursively (bands can end up being split in 8 parts).
fn quant_partition(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    mut n: i32,
    mut b: i32,
    mut blk: i32,
    lowband: Option<&[CeltNorm]>,
    mut lm: i32,
    gain: OpusVal32,
    mut fill: i32,
    #[cfg(feature = "qext")] mut ext_b: i32,
) -> u32 {
    let blk0 = blk;
    let mut cm: u32 = 0;
    let m = ctx.m;
    let i = ctx.i;
    let spread = ctx.spread;

    // If we need 1.5 more bit than we can produce, split the band in two.
    let cache = &m.cache.bits[m.cache.index[((lm + 1) * m.nb_ebands + i) as usize] as usize..];
    if lm != -1 && b > i32::from(cache[cache[0] as usize]) + 12 && n > 2 {
        n >>= 1;
        let nu = n as usize;
        let (xh, yh) = x[..2 * nu].split_at_mut(nu);
        lm -= 1;
        if blk == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blk = (blk + 1) >> 1;

        let sctx = compute_theta(
            ctx,
            xh,
            yh,
            n,
            &mut b,
            blk,
            blk0,
            lm,
            false,
            &mut fill,
            #[cfg(feature = "qext")]
            &mut ext_b,
        );
        let mut delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;
        #[cfg(not(feature = "qext"))]
        let (mid, side): (OpusVal32, OpusVal32) = (
            (1.0f32 / 32768.0) * sctx.imid as f32,
            (1.0f32 / 32768.0) * sctx.iside as f32,
        );
        #[cfg(feature = "qext")]
        let (mid, side): (OpusVal32, OpusVal32) = {
            let _ = (sctx.imid, sctx.iside);
            (
                celt_cos_norm2(sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32)),
                celt_cos_norm2(1.0f32 - sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32)),
            )
        };
        // FIXED_POINT: not ported (float build).

        // Give more bits to low-energy MDCTs than they would otherwise deserve
        if blk0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                // Rough approximation for pre-echo masking
                delta -= delta >> (4 - lm);
            } else {
                // Corresponds to a forward-masking slope of 1.5 dB per 10 ms
                delta = imin(0, delta + (n << BITRES >> (5 - lm)));
            }
        }
        let mut mbits = imax(0, imin(b, (b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let (lowband, next_lowband2) = match lowband {
            Some(l) => {
                let (a, bh) = l.split_at(nu);
                (Some(a), Some(bh)) // >32-bit split case
            }
            None => (None, None),
        };

        let mut rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_partition(
                ctx,
                xh,
                n,
                mbits,
                blk,
                lowband,
                lm,
                mult32_32_q31(gain, mid),
                fill,
                #[cfg(feature = "qext")]
                (ext_b / 2),
            );
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(
                ctx,
                yh,
                n,
                sbits,
                blk,
                next_lowband2,
                lm,
                mult32_32_q31(gain, side),
                fill >> blk,
                #[cfg(feature = "qext")]
                (ext_b / 2),
            ) << (blk0 >> 1);
        } else {
            cm = quant_partition(
                ctx,
                yh,
                n,
                sbits,
                blk,
                next_lowband2,
                lm,
                mult32_32_q31(gain, side),
                fill >> blk,
                #[cfg(feature = "qext")]
                (ext_b / 2),
            ) << (blk0 >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(
                ctx,
                xh,
                n,
                mbits,
                blk,
                lowband,
                lm,
                mult32_32_q31(gain, mid),
                fill,
                #[cfg(feature = "qext")]
                (ext_b / 2),
            );
        }
    } else {
        let nu = n as usize;
        let x = &mut x[..nu];
        #[cfg(feature = "qext")]
        let extra_bits: i32 = {
            let mut extra_bits = ext_b / (n - 1) >> BITRES;
            let ext_remaining_bits = ctx.ext_total_bits - ctx.ext_ec.tell_frac() as i32;
            if ext_remaining_bits < ((extra_bits + 1) * (n - 1) + n) << BITRES {
                extra_bits = (ext_remaining_bits - (n << BITRES)) / (n - 1) >> BITRES;
                extra_bits = imax(extra_bits - 1, 0);
            }
            imin(12, extra_bits)
        };
        // This is the basic no-split case
        let mut q = bits2pulses(m, i, lm, b);
        let mut curr_bits = pulses2bits(m, i, lm, q);
        ctx.remaining_bits -= curr_bits;

        // Ensures we can never bust the budget
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(m, i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        #[cfg(feature = "qext")]
        let cubic = q == 0 && ext_b > 2 * n << BITRES;
        #[cfg(not(feature = "qext"))]
        let cubic = false;

        if q != 0 {
            let k = get_pulses(q);

            // Finally do the actual quantization
            #[cfg(not(feature = "qext"))]
            {
                cm = match &mut ctx.ec {
                    EcCoder::Enc(e) => alg_quant(x, n, k, spread, blk, e, gain, ctx.resynth),
                    EcCoder::Dec(d) => alg_unquant(x, n, k, spread, blk, d, gain),
                };
            }
            #[cfg(feature = "qext")]
            {
                cm = match (&mut ctx.ec, &mut ctx.ext_ec) {
                    (EcCoder::Enc(e), EcCoder::Enc(xe)) => {
                        alg_quant(x, n, k, spread, blk, e, gain, ctx.resynth, xe, extra_bits)
                    }
                    (EcCoder::Dec(d), EcCoder::Dec(xd)) => {
                        alg_unquant(x, n, k, spread, blk, d, gain, xd, extra_bits)
                    }
                    _ => unreachable!("ec and ext_ec must code in the same direction"),
                };
            }
        } else if cubic {
            #[cfg(feature = "qext")]
            {
                let mut extra_bits = ext_b / (n - 1) >> BITRES;
                let ext_remaining_bits =
                    (ctx.ext_total_bits as u32).wrapping_sub(ctx.ext_ec.tell_frac()) as i32;
                if ext_remaining_bits < ((extra_bits + 1) * (n - 1) + n) << BITRES {
                    extra_bits = (ext_remaining_bits - (n << BITRES)) / (n - 1) >> BITRES;
                    extra_bits = imax(extra_bits - 1, 0);
                }
                extra_bits = imin(14, extra_bits);
                cm = match &mut ctx.ext_ec {
                    EcCoder::Enc(e) => cubic_quant(x, n, extra_bits, blk, e, gain, ctx.resynth),
                    EcCoder::Dec(d) => cubic_unquant(x, n, extra_bits, blk, d, gain),
                };
            }
        } else {
            // If there's no pulse, fill the band anyway
            if ctx.resynth {
                // B can be as large as 16, so this shift might overflow an int on a 16-bit
                // platform; use a long to get defined behavior.
                let cm_mask: u32 = ((1u64 << blk) as u32).wrapping_sub(1);
                fill = (fill as u32 & cm_mask) as i32;
                if fill == 0 {
                    x.fill(0.0);
                } else {
                    match lowband {
                        None => {
                            // Noise
                            for v in x.iter_mut() {
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                *v = shl32(((ctx.seed as i32) >> 20) as CeltNorm, 0);
                            }
                            cm = cm_mask;
                        }
                        Some(lowband) => {
                            // Folded spectrum
                            for (v, &l) in x.iter_mut().zip(lowband) {
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                // About 48 dB below the "normal" folding level
                                let mut tmp: OpusVal16 = qconst16(1.0f32 / 256.0, 10);
                                tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                                *v = l + tmp;
                            }
                            cm = fill as u32;
                        }
                    }
                    renormalise_vector(x, n, gain);
                }
            }
        }
    }

    cm
}

/// Port of celt/bands.c:cubic_quant_partition (QEXT): recursive cubic quantisation of the
/// extra bands.
#[cfg(feature = "qext")]
fn cubic_quant_partition(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    mut n: i32,
    mut b: i32,
    mut blk: i32,
    mut lm: i32,
    gain: OpusVal32,
    resynth: bool,
    encode: bool,
) -> u32 {
    debug_assert!(lm >= 0);
    let remaining = |ec: &EcCoder<'_, '_>| {
        ec.storage()
            .wrapping_mul(8 * 8)
            .wrapping_sub(ec.tell_frac()) as i32
    };
    ctx.remaining_bits = remaining(&ctx.ec);
    b = imin(b, ctx.remaining_bits);
    // As long as we have at least two bits of depth, split all the way to LM=0 (not -1 like
    // PVQ).
    if lm == 0 || b <= 2 * n << BITRES {
        b = imin(b + ((n - 1) << BITRES) / 2, ctx.remaining_bits);
        // Resolution left after taking into account coding the cube face.
        let mut res =
            (b - (1 << BITRES) - i32::from(ctx.m.log_n[ctx.i as usize]) - (lm << BITRES) - 1)
                / (n - 1)
                >> BITRES;
        res = imin(14, imax(0, res));
        let _ = encode;
        let ret = match &mut ctx.ec {
            EcCoder::Enc(e) => cubic_quant(x, n, res, blk, e, gain, resynth),
            EcCoder::Dec(d) => cubic_unquant(x, n, res, blk, d, gain),
        };
        ctx.remaining_bits = remaining(&ctx.ec);
        ret
    } else {
        let n0 = n;
        n >>= 1;
        let nu = n as usize;
        let (xh, yh) = x[..2 * nu].split_at_mut(nu);
        lm -= 1;
        blk = (blk + 1) >> 1;
        let theta_res = imin(16, (b >> BITRES) / (n0 - 1) + 1);
        let qtheta: i32 = match &mut ctx.ec {
            EcCoder::Enc(e) => {
                let itheta_q30 = stereo_itheta(xh, yh, false, n);
                let qtheta = (itheta_q30 + (1 << (29 - theta_res))) >> (30 - theta_res);
                e.enc_uint(qtheta as u32, ((1 << theta_res) + 1) as u32);
                qtheta
            }
            EcCoder::Dec(d) => d.dec_uint(((1 << theta_res) + 1) as u32) as i32,
        };
        let itheta_q30 = qtheta << (30 - theta_res);
        b -= theta_res << BITRES;
        let delta = (n0 - 1) * 23 * ((itheta_q30 >> 16) - 8192) >> (17 - BITRES);

        // FIXED_POINT: not ported (float build).
        let g1: OpusVal32 = celt_cos_norm2(itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        let g2: OpusVal32 =
            celt_cos_norm2(1.0f32 - itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        let (b1, b2) = if itheta_q30 == 0 {
            (b, 0)
        } else if itheta_q30 == 1073741824 {
            (0, b)
        } else {
            let b1 = imin(b, imax(0, (b - delta) / 2));
            (b1, b - b1)
        };
        let mut cm = cubic_quant_partition(
            ctx,
            xh,
            n,
            b1,
            blk,
            lm,
            mult32_32_q31(gain, g1),
            resynth,
            encode,
        );
        cm |= cubic_quant_partition(
            ctx,
            yh,
            n,
            b2,
            blk,
            lm,
            mult32_32_q31(gain, g2),
            resynth,
            encode,
        );
        cm
    }
}

/// `bit_interleave_table` of quant_band.
static BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
/// `bit_deinterleave_table` of quant_band.
static BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// Port of celt/bands.c:quant_band: encodes or decodes a band for the mono case, handling the
/// time-frequency resolution changes and the Hadamard reordering around `quant_partition`.
fn quant_band(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    n: i32,
    b: i32,
    mut blk: i32,
    lowband: Option<&mut [CeltNorm]>,
    lm: i32,
    lowband_out: Option<&mut [CeltNorm]>,
    gain: OpusVal32,
    lowband_scratch: Option<&mut [CeltNorm]>,
    x_alias: bool,
    mut fill: i32,
    #[cfg(feature = "qext")] ext_b: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let mut blk0 = blk;
    let mut time_divide = 0;
    let mut recombine = 0;
    let encode = ctx.encode;
    let mut tf_change = ctx.tf_change;

    let long_blocks = blk0 == 1;

    n_b = celt_udiv(n_b as u32, blk as u32) as i32;

    // Special case for one sample
    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }
    let nu = n as usize;
    let x = &mut x[..nu];

    if tf_change > 0 {
        recombine = tf_change;
    }
    // Band recombining to increase frequency resolution

    let mut lowband = lowband;
    // C: `lowband_scratch` may be `X` itself (`x_alias`, see quant_all_bands): the lowband is
    // then copied over `X` and every in-place transform below hits the same memory twice
    // (`X` when encoding, then `lowband`). `alias` tracks that `X` and `lowband` are one buffer
    // (`lowband` is kept equal to `x`).
    let mut alias = false;
    if let (Some(scratch), Some(lb)) = (lowband_scratch, lowband.as_deref())
        && (recombine != 0 || ((n_b & 1) == 0 && tf_change < 0) || blk0 > 1)
    {
        scratch[..nu].copy_from_slice(&lb[..nu]);
        if x_alias {
            x.copy_from_slice(&scratch[..nu]);
            alias = true;
        }
        lowband = Some(&mut scratch[..nu]);
    }

    for k in 0..recombine {
        if encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(lb) = lowband.as_deref_mut() {
            if alias {
                haar1(x, n >> k, 1 << k);
                lb.copy_from_slice(x);
            } else {
                haar1(lb, n >> k, 1 << k);
            }
        }
        fill = i32::from(BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize])
            | i32::from(BIT_INTERLEAVE_TABLE[(fill >> 4) as usize]) << 2;
    }
    blk >>= recombine;
    n_b <<= recombine;

    // Increasing the time resolution
    while (n_b & 1) == 0 && tf_change < 0 {
        if encode {
            haar1(x, n_b, blk);
        }
        if let Some(lb) = lowband.as_deref_mut() {
            if alias {
                haar1(x, n_b, blk);
                lb.copy_from_slice(x);
            } else {
                haar1(lb, n_b, blk);
            }
        }
        fill |= fill << blk;
        blk <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    blk0 = blk;
    let n_b0 = n_b;

    // Reorganize the samples in time order instead of frequency order
    if blk0 > 1 {
        if encode {
            deinterleave_hadamard(x, n_b >> recombine, blk0 << recombine, long_blocks);
        }
        if let Some(lb) = lowband.as_deref_mut() {
            if alias {
                deinterleave_hadamard(x, n_b >> recombine, blk0 << recombine, long_blocks);
                lb.copy_from_slice(x);
            } else {
                deinterleave_hadamard(lb, n_b >> recombine, blk0 << recombine, long_blocks);
            }
        }
    }

    let mut cm: u32;
    #[cfg(feature = "qext")]
    {
        if ctx.extra_bands
            && b > (3 * n << BITRES) + (i32::from(ctx.m.log_n[ctx.i as usize]) + 8 + 8 * lm)
        {
            let resynth = ctx.resynth;
            cm = cubic_quant_partition(ctx, x, n, b, blk, lm, gain, resynth, encode);
        } else {
            cm = quant_partition(ctx, x, n, b, blk, lowband.as_deref(), lm, gain, fill, ext_b);
        }
    }
    #[cfg(not(feature = "qext"))]
    {
        cm = quant_partition(ctx, x, n, b, blk, lowband.as_deref(), lm, gain, fill);
    }

    // This code is used by the decoder and by the resynthesis-enabled encoder
    if ctx.resynth {
        // Undo the sample reorganization going from time order to frequency order
        if blk0 > 1 {
            interleave_hadamard(x, n_b >> recombine, blk0 << recombine, long_blocks);
        }

        // Undo time-freq changes that we did earlier
        n_b = n_b0;
        blk = blk0;
        for _ in 0..time_divide {
            blk >>= 1;
            n_b <<= 1;
            cm |= cm >> blk;
            haar1(x, n_b, blk);
        }

        for k in 0..recombine {
            cm = u32::from(BIT_DEINTERLEAVE_TABLE[cm as usize]);
            haar1(x, n0 >> k, 1 << k);
        }
        blk <<= recombine;

        // Scale output for later folding
        if let Some(lo) = lowband_out {
            let nn: OpusVal16 = celt_sqrt(shl32(extend32(n0 as f32), 22));
            for (o, &v) in lo[..nu].iter_mut().zip(x.iter()) {
                *o = mult16_32_q15(nn, v);
            }
        }
        cm &= ((1i32 << blk) - 1) as u32;
    }
    cm
}

/// Port of celt/bands.c:quant_band_stereo: encodes or decodes a band for the stereo case.
fn quant_band_stereo(
    ctx: &mut BandCtx<'_, '_, '_>,
    x: &mut [CeltNorm],
    y: &mut [CeltNorm],
    n: i32,
    mut b: i32,
    blk: i32,
    lowband: Option<&mut [CeltNorm]>,
    lm: i32,
    lowband_out: Option<&mut [CeltNorm]>,
    lowband_scratch: Option<&mut [CeltNorm]>,
    x_alias: bool,
    mut fill: i32,
    #[cfg(feature = "qext")] mut ext_b: i32,
    #[cfg(feature = "qext")] cap: Option<&[i32]>,
) -> u32 {
    let mut cm: u32;
    let encode = ctx.encode;

    // Special case for one sample
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }
    let nu = n as usize;
    let x = &mut x[..nu];
    let y = &mut y[..nu];

    let orig_fill = fill;

    if encode {
        let nb = ctx.m.nb_ebands as usize;
        let iu = ctx.i as usize;
        if ctx.band_e[iu] < MIN_STEREO_ENERGY || ctx.band_e[nb + iu] < MIN_STEREO_ENERGY {
            if ctx.band_e[iu] > ctx.band_e[nb + iu] {
                y.copy_from_slice(x);
            } else {
                x.copy_from_slice(y);
            }
        }
    }
    let sctx = compute_theta(
        ctx,
        x,
        y,
        n,
        &mut b,
        blk,
        blk,
        lm,
        true,
        &mut fill,
        #[cfg(feature = "qext")]
        &mut ext_b,
    );
    let inv = sctx.inv;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;
    #[cfg(not(feature = "qext"))]
    let (mid, side): (OpusVal32, OpusVal32) = (
        (1.0f32 / 32768.0) * sctx.imid as f32,
        (1.0f32 / 32768.0) * sctx.iside as f32,
    );
    #[cfg(feature = "qext")]
    let (mid, side): (OpusVal32, OpusVal32) = {
        let _ = (sctx.imid, sctx.iside);
        (
            celt_cos_norm2(sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32)),
            celt_cos_norm2(1.0f32 - sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32)),
        )
    };
    // FIXED_POINT: not ported (float build).

    // This is a special case for N=2 that only works for stereo and takes advantage of the
    // fact that mid and side are orthogonal to encode the side with just one bit.
    if n == 2 {
        let mut sign: i32 = 0;
        let mut mbits = b;
        let mut sbits = 0;
        // Only need one bit for the side.
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;

        let (x2, y2): (&mut [CeltNorm], &mut [CeltNorm]) = if c {
            (&mut *y, &mut *x)
        } else {
            (&mut *x, &mut *y)
        };
        if sbits != 0 {
            match &mut ctx.ec {
                EcCoder::Enc(e) => {
                    // Here we only need to encode a sign for the side.
                    // FIXME: Need to increase fixed-point precision?
                    sign =
                        i32::from(mult32_32_q31(x2[0], y2[1]) - mult32_32_q31(x2[1], y2[0]) < 0.0);
                    e.enc_bits(sign as u32, 1);
                }
                EcCoder::Dec(d) => {
                    sign = d.dec_bits(1) as i32;
                }
            }
        }
        sign = 1 - 2 * sign;
        // We use orig_fill here because we want to fold the side, but if itheta==16384, we'll
        // have cleared the low bits of fill.
        cm = quant_band(
            ctx,
            x2,
            n,
            mbits,
            blk,
            lowband,
            lm,
            lowband_out,
            Q31ONE,
            lowband_scratch,
            // The scratch is `X`; it only aliases the band quantised here when that is `X`
            // (with `c`, C copies the lowband over `y2`, which N=2 then overwrites in the
            // decoder; the encoder case is not reproduced, see quant_all_bands).
            x_alias && !c,
            orig_fill,
            #[cfg(feature = "qext")]
            ext_b,
        );
        // We don't split N=2 bands, so cm is either 1 or 0 (for a fold-collapse), and there's
        // no need to worry about mixing with the other channel.
        y2[0] = (-sign) as f32 * x2[1];
        y2[1] = sign as f32 * x2[0];
        if ctx.resynth {
            x[0] = mult32_32_q31(mid, x[0]);
            x[1] = mult32_32_q31(mid, x[1]);
            y[0] = mult32_32_q31(side, y[0]);
            y[1] = mult32_32_q31(side, y[1]);
            let tmp = x[0];
            x[0] = sub32(tmp, y[0]);
            y[0] = add32(tmp, y[0]);
            let tmp = x[1];
            x[1] = sub32(tmp, y[1]);
            y[1] = add32(tmp, y[1]);
        }
    } else {
        // "Normal" split code
        let mut mbits = imax(0, imin(b, (b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let mut rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            #[cfg(feature = "qext")]
            let qext_extra: i32 = match cap {
                // Reallocate any mid bits that cannot be used to extra mid bits.
                Some(cap) if ext_b != 0 => {
                    imax(0, imin(ext_b / 2, mbits - cap[ctx.i as usize] / 2))
                }
                _ => 0,
            };
            // In stereo mode, we do not apply a scaling to the mid because we need the
            // normalized mid for folding later.
            cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                blk,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                x_alias,
                fill,
                #[cfg(feature = "qext")]
                (ext_b / 2 + qext_extra),
            );
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            // Guard against overflowing the EC with the angle if the cubic quant used too many
            // bits for the mid.
            #[cfg(feature = "qext")]
            if ctx.extra_bands {
                sbits = imin(sbits, ctx.remaining_bits);
            }
            // For a stereo split, the high bits of fill are always zero, so no folding will be
            // done to the side.
            cm |= quant_band(
                ctx,
                y,
                n,
                sbits,
                blk,
                None,
                lm,
                None,
                side,
                None,
                false,
                fill >> blk,
                #[cfg(feature = "qext")]
                (ext_b / 2 - qext_extra),
            );
        } else {
            #[cfg(feature = "qext")]
            let qext_extra: i32 = match cap {
                // Reallocate any side bits that cannot be used to extra side bits.
                Some(cap) if ext_b != 0 => {
                    imax(0, imin(ext_b / 2, sbits - cap[ctx.i as usize] / 2))
                }
                _ => 0,
            };
            // For a stereo split, the high bits of fill are always zero, so no folding will be
            // done to the side.
            cm = quant_band(
                ctx,
                y,
                n,
                sbits,
                blk,
                None,
                lm,
                None,
                side,
                None,
                false,
                fill >> blk,
                #[cfg(feature = "qext")]
                (ext_b / 2 + qext_extra),
            );
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            // Guard against overflowing the EC with the angle if the cubic quant used too many
            // bits for the side.
            #[cfg(feature = "qext")]
            if ctx.extra_bands {
                mbits = imin(mbits, ctx.remaining_bits);
            }
            // In stereo mode, we do not apply a scaling to the mid because we need the
            // normalized mid for folding later.
            cm |= quant_band(
                ctx,
                x,
                n,
                mbits,
                blk,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                x_alias,
                fill,
                #[cfg(feature = "qext")]
                (ext_b / 2 - qext_extra),
            );
        }
    }

    // This code is used by the decoder and by the resynthesis-enabled encoder
    if ctx.resynth {
        if n != 2 {
            stereo_merge(x, y, mid, n);
        }
        if inv {
            for v in y.iter_mut() {
                *v = -*v;
            }
        }
    }
    cm
}

/// Port of celt/bands.c:special_hybrid_folding: duplicates enough of the first band folding
/// data to be able to fold the second band. Copies no data for CELT-only mode.
pub fn special_hybrid_folding(
    m: &CeltMode,
    norm: &mut [CeltNorm],
    norm2: &mut [CeltNorm],
    start: i32,
    mm: i32,
    dual_stereo: bool,
) {
    let eb = |i: i32| i32::from(m.e_bands[i as usize]);
    let n1 = mm * (eb(start + 1) - eb(start));
    let n2 = mm * (eb(start + 2) - eb(start + 1));
    if n2 > n1 {
        let (src, dst, cnt) = ((2 * n1 - n2) as usize, n1 as usize, (n2 - n1) as usize);
        norm.copy_within(src..src + cnt, dst);
        if dual_stereo {
            norm2.copy_within(src..src + cnt, dst);
        }
    }
}

/// Scratch buffers of [`quant_all_bands`] (the C VLAs `_norm`, `_lowband_scratch`, `X_save`,
/// ...). Keep one per encoder/decoder so no allocation happens after the first frame.
#[derive(Debug, Clone, Default)]
pub struct BandsScratch {
    norm: Vec<CeltNorm>,
    lowband_scratch: Vec<CeltNorm>,
    x_save: Vec<CeltNorm>,
    y_save: Vec<CeltNorm>,
    x_save2: Vec<CeltNorm>,
    y_save2: Vec<CeltNorm>,
    norm_save2: Vec<CeltNorm>,
    lowband_tmp: Vec<CeltNorm>,
    lowband_out_tmp: Vec<CeltNorm>,
    degen_x: Vec<CeltNorm>,
    degen_y: Vec<CeltNorm>,
    bytes_save: Vec<u8>,
    #[cfg(feature = "qext")]
    ext_bytes_save: Vec<u8>,
}

impl BandsScratch {
    /// Creates empty scratch buffers (they grow on first use).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            norm: Vec::new(),
            lowband_scratch: Vec::new(),
            x_save: Vec::new(),
            y_save: Vec::new(),
            x_save2: Vec::new(),
            y_save2: Vec::new(),
            norm_save2: Vec::new(),
            lowband_tmp: Vec::new(),
            lowband_out_tmp: Vec::new(),
            degen_x: Vec::new(),
            degen_y: Vec::new(),
            bytes_save: Vec::new(),
            #[cfg(feature = "qext")]
            ext_bytes_save: Vec::new(),
        }
    }
}

/// Grows `v` to at least `n` elements.
fn ensure_len(v: &mut Vec<CeltNorm>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Calls `f(lowband, lowband_out)` with the windows `buf[lb..lb+n]` and `buf[out..out+n]`.
///
/// They only overlap for the hybrid folding band (`lb < out < lb+n`); C then uses aliased
/// pointers. Every access to `lowband` happens before the writes to `lowband_out`, so running
/// on copies and writing them back in that order reproduces the C memory effects.
fn with_lowband<R>(
    buf: &mut [CeltNorm],
    lb: Option<usize>,
    out: Option<usize>,
    n: usize,
    tmp_lb: &mut [CeltNorm],
    tmp_out: &mut [CeltNorm],
    f: impl FnOnce(Option<&mut [CeltNorm]>, Option<&mut [CeltNorm]>) -> R,
) -> R {
    match (lb, out) {
        (Some(l), Some(o)) if l + n > o => {
            let tl = &mut tmp_lb[..n];
            let to = &mut tmp_out[..n];
            tl.copy_from_slice(&buf[l..l + n]);
            to.copy_from_slice(&buf[o..o + n]);
            let r = f(Some(&mut *tl), Some(&mut *to));
            buf[l..l + n].copy_from_slice(tl);
            buf[o..o + n].copy_from_slice(to);
            r
        }
        (Some(l), Some(o)) => {
            let (a, b) = buf.split_at_mut(o);
            f(Some(&mut a[l..l + n]), Some(&mut b[..n]))
        }
        (Some(l), None) => f(Some(&mut buf[l..l + n]), None),
        (None, Some(o)) => f(None, Some(&mut buf[o..o + n])),
        (None, None) => f(None, None),
    }
}

/// Which buffer `lowband_scratch` points to in C.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScratchSel {
    /// `_lowband_scratch` (resynthesising encoder).
    Own,
    /// `X_+M*eBands[effEBands-1]` (decoder / plain encoder).
    XTail,
    /// `NULL`.
    Null,
}

/// Port of celt/bands.c:quant_all_bands: quantisation (encoder) or decoding of the residual
/// (normalised) spectrum of all bands `start..end`.
///
/// * `x_` / `y_`: the normalised spectrum of each channel (`M*shortMdctSize` samples; `y_` is
///   `None` for mono). Encoding modifies them; decoding writes the decoded bands.
/// * `collapse_masks`: `C*nbEBands` anti-collapse masks (written for `start..end`).
/// * `ec`: the range coder; its direction is the C `encode` flag.
/// * `seed`: the random generator seed, updated.
/// * QEXT: `ext_ec` (same direction as `ec`), `extra_pulses`, `ext_total_bits`, `cap`.
/// * `scratch`: reusable scratch buffers.
pub fn quant_all_bands(
    m: &CeltMode,
    start: i32,
    end: i32,
    x_: &mut [CeltNorm],
    mut y_: Option<&mut [CeltNorm]>,
    collapse_masks: &mut [u8],
    band_e: &[CeltEner],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    mut dual_stereo: bool,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    ec: &mut EcCoder<'_, '_>,
    lm: i32,
    coded_bands: i32,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
    #[cfg(feature = "qext")] ext_ec: &mut EcCoder<'_, '_>,
    #[cfg(feature = "qext")] extra_pulses: &[i32],
    #[cfg(feature = "qext")] ext_total_bits: i32,
    #[cfg(feature = "qext")] cap: Option<&[i32]>,
    scratch: &mut BandsScratch,
) {
    let encode = ec.is_encoder();
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let nb = m.nb_ebands;
    let stereo = y_.is_some();
    let c: i32 = if stereo { 2 } else { 1 };
    #[allow(unused_mut, reason = "only cleared with qext")]
    let mut theta_rdo = encode && stereo && !dual_stereo && complexity >= 8;
    // RESYNTH is not defined upstream.
    let resynth = !encode || theta_rdo;
    let mut update_lowband = true;
    let mut lowband_offset: i32 = 0;

    let mm = 1 << lm;
    let blk = if short_blocks { mm } else { 1 };
    let norm_offset = mm * eb(start);
    // No need to allocate norm for the last band because we don't need an output in that band.
    let norm_len = (mm * eb(nb - 1) - norm_offset) as usize;

    // Largest band handled, for the per-band scratch buffers (C sizes them by the last band).
    let mut max_n = (mm * (eb(nb) - eb(nb - 1))) as usize;
    for i in start..end {
        max_n = max_n.max((mm * (eb(i + 1) - eb(i))) as usize);
    }
    let BandsScratch {
        norm: norm_buf,
        lowband_scratch: own_scratch,
        x_save,
        y_save,
        x_save2,
        y_save2,
        norm_save2,
        lowband_tmp,
        lowband_out_tmp,
        degen_x,
        degen_y,
        bytes_save,
        #[cfg(feature = "qext")]
        ext_bytes_save,
    } = scratch;
    ensure_len(norm_buf, c as usize * norm_len);
    for v in [
        &mut *own_scratch,
        &mut *x_save,
        &mut *y_save,
        &mut *x_save2,
        &mut *y_save2,
        &mut *norm_save2,
        &mut *lowband_tmp,
        &mut *lowband_out_tmp,
        &mut *degen_x,
        &mut *degen_y,
    ] {
        ensure_len(v, max_n);
    }
    let (norm, norm2) = norm_buf[..c as usize * norm_len].split_at_mut(norm_len);

    // For decoding, we can use the last band as scratch space because we don't need that
    // scratch space for the last band and we don't care about the data there until we're
    // decoding the last band.
    let mut scratch_sel = if encode && resynth {
        ScratchSel::Own
    } else {
        ScratchSel::XTail
    };
    let tail_off = (mm * eb(m.eff_ebands - 1)) as usize;
    let (x_lo, x_hi) = x_.split_at_mut(tail_off.min(x_.len()));

    let mut ctx = BandCtx {
        encode,
        resynth,
        m,
        i: 0,
        intensity,
        spread,
        tf_change: 0,
        ec: ec.reborrow(),
        remaining_bits: 0,
        band_e,
        seed: *seed,
        theta_round: 0,
        disable_inv,
        // Avoid injecting noise in the first band on transients.
        avoid_split_noise: blk > 1,
        #[cfg(feature = "qext")]
        ext_ec: ext_ec.reborrow(),
        #[cfg(feature = "qext")]
        extra_bits: 0,
        #[cfg(feature = "qext")]
        ext_total_bits,
        #[cfg(feature = "qext")]
        extra_bands: end == crate::celt::modes::NB_QEXT_BANDS || end == 2,
        #[cfg(not(feature = "qext"))]
        _ext: PhantomData,
    };
    #[cfg(feature = "qext")]
    let mut ext_balance: i32 = 0;
    #[cfg(feature = "qext")]
    let mut ext_tell: i32 = 0;
    #[cfg(feature = "qext")]
    if ctx.extra_bands {
        theta_rdo = false;
    }

    for i in start..end {
        ctx.i = i;
        let last = i == end - 1;
        let x_off = (mm * eb(i)) as usize;
        let n = mm * eb(i + 1) - mm * eb(i);
        debug_assert!(n > 0);
        let nu = n as usize;
        let tell: i32 = ctx.ec.tell_frac() as i32;

        // Compute how many bits we want to allocate to this band
        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        #[cfg(feature = "qext")]
        let ext_b: i32 = {
            if i != start {
                ext_balance += extra_pulses[(i - 1) as usize] + ext_tell;
            }
            ext_tell = ctx.ext_ec.tell_frac() as i32;
            ctx.extra_bits = extra_pulses[i as usize];
            if i != start {
                ext_balance -= ext_tell;
            }
            // C: `i <= codedBands-1`.
            if i < coded_bands {
                let ext_curr_balance = celt_sudiv(ext_balance, imin(3, coded_bands - i));
                imax(
                    0,
                    imin(
                        16383,
                        imin(
                            ext_total_bits - ext_tell,
                            extra_pulses[i as usize] + ext_curr_balance,
                        ),
                    ),
                )
            } else {
                0
            }
        };
        // C: `i <= codedBands-1`.
        let b = if i < coded_bands {
            let curr_balance = celt_sudiv(balance, imin(3, coded_bands - i));
            imax(
                0,
                imin(
                    16383,
                    imin(remaining_bits + 1, pulses[i as usize] + curr_balance),
                ),
            )
        } else {
            0
        };

        if resynth
            && (mm * eb(i) - n >= mm * eb(start) || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == start + 1 {
            special_hybrid_folding(m, norm, norm2, start, mm, dual_stereo);
        }

        let tf_change = tf_res[i as usize];
        ctx.tf_change = tf_change;
        let degenerate = i >= m.eff_ebands;
        if degenerate {
            scratch_sel = ScratchSel::Null;
        }
        if last && !theta_rdo {
            scratch_sel = ScratchSel::Null;
        }

        // Get a conservative estimate of the collapse_mask's for the bands we're going to be
        // folding from.
        let mut effective_lowband: i32 = -1;
        let mut x_cm: u32;
        let mut y_cm: u32;
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || blk > 1 || tf_change < 0) {
            // This ensures we never repeat spectral content within one band
            effective_lowband = imax(0, mm * eb(lowband_offset) - norm_offset - n);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if mm * eb(fold_start) <= effective_lowband + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i && mm * eb(fold_end) < effective_lowband + norm_offset + n) {
                    break;
                }
            }
            x_cm = 0;
            y_cm = 0;
            let mut fold_i = fold_start;
            loop {
                x_cm |= u32::from(collapse_masks[(fold_i * c) as usize]);
                y_cm |= u32::from(collapse_masks[(fold_i * c + c - 1) as usize]);
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            // Otherwise, we'll be using the LCG to fold, so all blocks will (almost always) be
            // non-zero.
            x_cm = ((1i32 << blk) - 1) as u32;
            y_cm = x_cm;
        }

        if dual_stereo && i == intensity {
            // Switch off dual stereo to do intensity.
            dual_stereo = false;
            if resynth {
                let lim = (mm * eb(i) - norm_offset) as usize;
                for j in 0..lim {
                    norm[j] = half32(norm[j] + norm2[j]);
                }
            }
        }

        // The band buffers (C `X`, `Y`) and the lowband scratch.
        let lb = (effective_lowband != -1).then_some(effective_lowband as usize);
        let lb_out = (!last).then_some((mm * eb(i) - norm_offset) as usize);
        // C's `lowband_scratch` is this band's own `X` (band `effEBands-1` when it is not the
        // last one; e.g. the QEXT bands of a 48 kHz mode): see quant_band and the dual-stereo
        // write-back below.
        let mut alias = false;
        let (xb, mut yb, mut lsel): (
            &mut [CeltNorm],
            Option<&mut [CeltNorm]>,
            Option<&mut [CeltNorm]>,
        );
        if degenerate {
            // C: X = Y = norm (see the module documentation).
            degen_x[..nu].copy_from_slice(&norm[..nu]);
            degen_y[..nu].copy_from_slice(&norm[..nu]);
            xb = &mut degen_x[..nu];
            yb = if stereo {
                Some(&mut degen_y[..nu])
            } else {
                None
            };
            lsel = None;
        } else {
            let tail: Option<&mut [CeltNorm]>;
            alias = scratch_sel == ScratchSel::XTail && x_off + nu > tail_off;
            if x_off + nu <= tail_off {
                xb = &mut x_lo[x_off..x_off + nu];
                tail = Some(&mut *x_hi);
            } else {
                xb = &mut x_hi[x_off - tail_off..x_off - tail_off + nu];
                tail = None;
            }
            yb = y_.as_deref_mut().map(|y| &mut y[x_off..x_off + nu]);
            lsel = match scratch_sel {
                ScratchSel::Own => Some(&mut own_scratch[..]),
                ScratchSel::XTail => match tail {
                    Some(t) => Some(t),
                    // The band lies in the tail itself: C aliases them (see module docs).
                    None => Some(&mut own_scratch[..]),
                },
                ScratchSel::Null => None,
            };
        }

        if dual_stereo {
            x_cm = with_lowband(
                norm,
                lb,
                lb_out,
                nu,
                lowband_tmp,
                lowband_out_tmp,
                |lowband, lowband_out| {
                    quant_band(
                        &mut ctx,
                        xb,
                        n,
                        b / 2,
                        blk,
                        lowband,
                        lm,
                        lowband_out,
                        Q31ONE,
                        lsel.as_deref_mut(),
                        alias,
                        x_cm as i32,
                        #[cfg(feature = "qext")]
                        (ext_b / 2),
                    )
                },
            );
            // Dual stereo needs a second channel (C would dereference a NULL `Y`).
            debug_assert!(stereo);
            // C: when `lowband_scratch` is `X` (`alias`), quant_band(Y) copies Y's lowband
            // there (then transforms it in place), overwriting the band just decoded into `X`.
            let y_copies_lowband = alias
                && lb.is_some()
                && n != 1
                && (tf_change > 0
                    || ((celt_udiv(n as u32, blk as u32) & 1) == 0 && tf_change < 0)
                    || blk > 1);
            if let Some(ybd) = yb.as_deref_mut() {
                y_cm = with_lowband(
                    norm2,
                    lb,
                    lb_out,
                    nu,
                    lowband_tmp,
                    lowband_out_tmp,
                    |lowband, lowband_out| {
                        quant_band(
                            &mut ctx,
                            ybd,
                            n,
                            b / 2,
                            blk,
                            lowband,
                            lm,
                            lowband_out,
                            Q31ONE,
                            lsel.as_deref_mut(),
                            false,
                            y_cm as i32,
                            #[cfg(feature = "qext")]
                            (ext_b / 2),
                        )
                    },
                );
            }
            if y_copies_lowband && let Some(s) = lsel.as_deref() {
                xb.copy_from_slice(&s[..nu]);
            }
        } else {
            if let Some(yb) = yb {
                if theta_rdo && i < intensity {
                    let w = compute_channel_weights(band_e[i as usize], band_e[(i + nb) as usize]);
                    // Make a copy.
                    let cm = x_cm | y_cm;
                    let ec_save = ec_snapshot(&ctx.ec);
                    #[cfg(feature = "qext")]
                    let ext_ec_save = ec_snapshot(&ctx.ext_ec);
                    let ctx_save = ctx.save();
                    x_save[..nu].copy_from_slice(xb);
                    y_save[..nu].copy_from_slice(yb);
                    // Encode and round down.
                    ctx.theta_round = -1;
                    x_cm = with_lowband(
                        norm,
                        lb,
                        lb_out,
                        nu,
                        lowband_tmp,
                        lowband_out_tmp,
                        |lowband, lowband_out| {
                            quant_band_stereo(
                                &mut ctx,
                                xb,
                                yb,
                                n,
                                b,
                                blk,
                                lowband,
                                lm,
                                lowband_out,
                                lsel.as_deref_mut(),
                                false,
                                cm as i32,
                                #[cfg(feature = "qext")]
                                ext_b,
                                #[cfg(feature = "qext")]
                                cap,
                            )
                        },
                    );
                    let dist0: OpusVal32 = mult16_32_q15(w[0], celt_inner_prod(x_save, xb, nu))
                        + mult16_32_q15(w[1], celt_inner_prod(y_save, yb, nu));

                    // Save first result.
                    let cm2 = x_cm;
                    let ec_save2 = ec_snapshot(&ctx.ec);
                    #[cfg(feature = "qext")]
                    let ext_ec_save2 = ec_snapshot(&ctx.ext_ec);
                    let ctx_save2 = ctx.save();
                    x_save2[..nu].copy_from_slice(xb);
                    y_save2[..nu].copy_from_slice(yb);
                    if let Some(o) = lb_out {
                        norm_save2[..nu].copy_from_slice(&norm[o..o + nu]);
                    }
                    ec_save_bytes(&ctx.ec, ec_save.as_ref(), bytes_save);
                    #[cfg(feature = "qext")]
                    ec_save_bytes(&ctx.ext_ec, ext_ec_save.as_ref(), ext_bytes_save);
                    // Restore
                    ec_restore(&mut ctx.ec, ec_save.as_ref());
                    #[cfg(feature = "qext")]
                    ec_restore(&mut ctx.ext_ec, ext_ec_save.as_ref());
                    ctx.restore(&ctx_save);
                    xb.copy_from_slice(&x_save[..nu]);
                    yb.copy_from_slice(&y_save[..nu]);
                    if i == start + 1 {
                        special_hybrid_folding(m, norm, norm2, start, mm, dual_stereo);
                    }
                    // Encode and round up.
                    ctx.theta_round = 1;
                    x_cm = with_lowband(
                        norm,
                        lb,
                        lb_out,
                        nu,
                        lowband_tmp,
                        lowband_out_tmp,
                        |lowband, lowband_out| {
                            quant_band_stereo(
                                &mut ctx,
                                xb,
                                yb,
                                n,
                                b,
                                blk,
                                lowband,
                                lm,
                                lowband_out,
                                lsel.as_deref_mut(),
                                false,
                                cm as i32,
                                #[cfg(feature = "qext")]
                                ext_b,
                                #[cfg(feature = "qext")]
                                cap,
                            )
                        },
                    );
                    let dist1: OpusVal32 = mult16_32_q15(w[0], celt_inner_prod(x_save, xb, nu))
                        + mult16_32_q15(w[1], celt_inner_prod(y_save, yb, nu));
                    if dist0 >= dist1 {
                        x_cm = cm2;
                        ec_restore(&mut ctx.ec, ec_save2.as_ref());
                        #[cfg(feature = "qext")]
                        ec_restore(&mut ctx.ext_ec, ext_ec_save2.as_ref());
                        ctx.restore(&ctx_save2);
                        xb.copy_from_slice(&x_save2[..nu]);
                        yb.copy_from_slice(&y_save2[..nu]);
                        if let Some(o) = lb_out {
                            norm[o..o + nu].copy_from_slice(&norm_save2[..nu]);
                        }
                        ec_restore_bytes(&mut ctx.ec, ec_save.as_ref(), bytes_save);
                        #[cfg(feature = "qext")]
                        ec_restore_bytes(&mut ctx.ext_ec, ext_ec_save.as_ref(), ext_bytes_save);
                    }
                } else {
                    ctx.theta_round = 0;
                    x_cm = with_lowband(
                        norm,
                        lb,
                        lb_out,
                        nu,
                        lowband_tmp,
                        lowband_out_tmp,
                        |lowband, lowband_out| {
                            quant_band_stereo(
                                &mut ctx,
                                xb,
                                yb,
                                n,
                                b,
                                blk,
                                lowband,
                                lm,
                                lowband_out,
                                lsel.as_deref_mut(),
                                alias,
                                (x_cm | y_cm) as i32,
                                #[cfg(feature = "qext")]
                                ext_b,
                                #[cfg(feature = "qext")]
                                cap,
                            )
                        },
                    );
                }
            } else {
                x_cm = with_lowband(
                    norm,
                    lb,
                    lb_out,
                    nu,
                    lowband_tmp,
                    lowband_out_tmp,
                    |lowband, lowband_out| {
                        quant_band(
                            &mut ctx,
                            xb,
                            n,
                            b,
                            blk,
                            lowband,
                            lm,
                            lowband_out,
                            Q31ONE,
                            lsel,
                            alias,
                            (x_cm | y_cm) as i32,
                            #[cfg(feature = "qext")]
                            ext_b,
                        )
                    },
                );
            }
            y_cm = x_cm;
        }
        if degenerate && !stereo {
            // C quantised norm[0..N] in place.
            norm[..nu].copy_from_slice(&degen_x[..nu]);
        }
        collapse_masks[(i * c) as usize] = x_cm as u8;
        collapse_masks[(i * c + c - 1) as usize] = y_cm as u8;
        balance += pulses[i as usize] + tell;

        // Update the folding position only as long as we have 1 bit/sample depth.
        update_lowband = b > n << BITRES;
        // We only need to avoid noise on a split for the first band. After that, we have
        // folding.
        ctx.avoid_split_noise = false;
    }
    *seed = ctx.seed;
}

//! Port of celt/rate.c, celt/rate.h: pulse cache helpers and the CELT bit allocation.

#![allow(
    clippy::too_many_arguments,
    reason = "allocation functions mirror the C signatures"
)]
#![allow(
    clippy::precedence,
    reason = "shift/arithmetic expressions are copied verbatim from C, whose precedence rules \
              for `*`, `+` and `<<` are the same as Rust's"
)]

#[cfg(feature = "custom-modes")]
use alloc::{borrow::Cow, vec, vec::Vec};

#[cfg(feature = "qext")]
use crate::celt::arch::{CeltGlog, OpusVal16, OpusVal32, max16, max32, maxg, min16, min32};
use crate::celt::arch::{imax, imin};
use crate::celt::entcode::{BITRES, EcCoder, celt_udiv};
use crate::celt::static_modes::CeltMode;
#[cfg(feature = "custom-modes")]
use crate::celt::static_modes::PulseCache;

/// `MAX_PSEUDO`.
pub const MAX_PSEUDO: i32 = 40;
/// `LOG_MAX_PSEUDO`.
pub const LOG_MAX_PSEUDO: i32 = 6;
/// `CELT_MAX_PULSES`.
pub const CELT_MAX_PULSES: i32 = 128;
/// `MAX_FINE_BITS`.
pub const MAX_FINE_BITS: i32 = 8;
/// `FINE_OFFSET`.
pub const FINE_OFFSET: i32 = 21;
/// `QTHETA_OFFSET`.
pub const QTHETA_OFFSET: i32 = 4;
/// `QTHETA_OFFSET_TWOPHASE`.
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// `ALLOC_STEPS`.
const ALLOC_STEPS: i32 = 6;

/// Upper bound on `nbEBands` for the stack scratch arrays of [`clt_compute_allocation`]
/// (C uses VLAs of `nbEBands` entries). Standard modes use 21 bands, the QEXT mode 14, and
/// custom modes are limited to fewer than 32 by `compute_ebands` (shorts are at most 3.3 ms, so
/// the band resolution is at least 150 Hz).
const MAX_ALLOC_BANDS: usize = 32;

/// `LOG2_FRAC_TABLE`.
#[rustfmt::skip]
static LOG2_FRAC_TABLE: [u8; 24] = [
    0,
    8, 13,
    16, 19, 21, 23,
    24, 26, 27, 28, 29, 30, 31, 32,
    32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Port of celt/rate.h:get_pulses.
#[inline(always)]
#[must_use]
pub const fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Offset of the pulse-cache row for (`band`, `LM+1`).
#[inline(always)]
fn cache_row(m: &CeltMode, band: i32, lm: i32) -> &[u8] {
    let lm = lm + 1;
    let off = m.cache.index[(lm * m.nb_ebands + band) as usize];
    &m.cache.bits[off as usize..]
}

/// Port of celt/rate.h:bits2pulses.
#[inline]
#[must_use]
pub fn bits2pulses(m: &CeltMode, band: i32, lm: i32, mut bits: i32) -> i32 {
    let cache = cache_row(m, band, lm);
    let mut lo: i32 = 0;
    let mut hi: i32 = i32::from(cache[0]);
    bits -= 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        // OPT: Make sure this is implemented with a conditional move
        if i32::from(cache[mid as usize]) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 {
        -1
    } else {
        i32::from(cache[lo as usize])
    };
    if bits - lo_bits <= i32::from(cache[hi as usize]) - bits {
        lo
    } else {
        hi
    }
}

/// Port of celt/rate.h:pulses2bits.
#[inline]
#[must_use]
pub fn pulses2bits(m: &CeltMode, band: i32, lm: i32, pulses: i32) -> i32 {
    let cache = cache_row(m, band, lm);
    if pulses == 0 {
        0
    } else {
        i32::from(cache[pulses as usize]) + 1
    }
}

/// Port of celt/rate.c:fits_in32.
///
/// Determines if `V(N,K)` fits in a 32-bit unsigned integer. `N` and `K` are themselves limited
/// to 15 bits.
#[cfg(feature = "custom-modes")]
#[must_use]
pub const fn fits_in32(n: i32, k: i32) -> bool {
    const MAX_N: [i16; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [i16; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];
    if n >= 14 {
        if k >= 14 {
            false
        } else {
            n <= MAX_N[k as usize] as i32
        }
    } else {
        k <= MAX_K[n as usize] as i32
    }
}

/// Port of celt/rate.c:compute_pulse_cache.
///
/// Builds `m.cache` (index, bits and caps tables) for a custom mode with bands `m.e_bands`.
#[cfg(feature = "custom-modes")]
pub fn compute_pulse_cache(m: &mut CeltMode, lm: i32) {
    use crate::celt::cwrs::get_required_bits;

    let nb = m.nb_ebands;
    let e_bands = &m.e_bands;
    let mut curr: i32 = 0;
    let mut nb_entries: usize = 0;
    let mut entry_n = [0i32; 100];
    let mut entry_k = [0i32; 100];
    let mut entry_i = [0i32; 100];

    let mut cindex: Vec<i16> = vec![0; (nb * (lm + 2)) as usize];

    // Scan for all unique band sizes
    for i in 0..=lm + 1 {
        for j in 0..nb {
            let n_ =
                (i32::from(e_bands[(j + 1) as usize]) - i32::from(e_bands[j as usize])) << i >> 1;
            cindex[(i * nb + j) as usize] = -1;
            // Find other bands that have the same size
            for k in 0..=i {
                let mut n = 0;
                while n < nb && (k != i || n < j) {
                    if n_
                        == (i32::from(e_bands[(n + 1) as usize]) - i32::from(e_bands[n as usize]))
                            << k
                            >> 1
                    {
                        cindex[(i * nb + j) as usize] = cindex[(k * nb + n) as usize];
                        break;
                    }
                    n += 1;
                }
            }
            if cindex[(i * nb + j) as usize] == -1 && n_ != 0 {
                entry_n[nb_entries] = n_;
                let mut kk = 0;
                while fits_in32(n_, get_pulses(kk + 1)) && kk < MAX_PSEUDO {
                    kk += 1;
                }
                entry_k[nb_entries] = kk;
                cindex[(i * nb + j) as usize] = curr as i16;
                entry_i[nb_entries] = curr;

                curr += kk + 1;
                nb_entries += 1;
            }
        }
    }
    let mut bits: Vec<u8> = vec![0; curr as usize];
    // Compute the cache for all unique sizes
    for i in 0..nb_entries {
        let ptr = &mut bits[entry_i[i] as usize..];
        let mut tmp = [0i16; (CELT_MAX_PULSES + 1) as usize];
        get_required_bits(&mut tmp, entry_n[i], get_pulses(entry_k[i]), BITRES);
        for j in 1..=entry_k[i] {
            ptr[j as usize] = (tmp[get_pulses(j) as usize] - 1) as u8;
        }
        ptr[0] = entry_k[i] as u8;
    }

    // Compute the maximum rate for each band at which we'll reliably use as many bits as we ask
    // for.
    let mut caps: Vec<u8> = Vec::with_capacity(((lm + 1) * 2 * nb) as usize);
    for i in 0..=lm {
        for c in 1..=2 {
            for j in 0..nb {
                let mut n0 = i32::from(e_bands[(j + 1) as usize]) - i32::from(e_bands[j as usize]);
                let mut max_bits: i32;
                // N=1 bands only have a sign bit and fine bits.
                if n0 << i == 1 {
                    max_bits = c * (1 + MAX_FINE_BITS) << BITRES;
                } else {
                    let mut lm0: i32 = 0;
                    // Even-sized bands bigger than N=2 can be split one more time.
                    // As of commit 44203907 all bands >1 are even, including custom modes.
                    if n0 > 2 {
                        n0 >>= 1;
                        lm0 -= 1;
                    }
                    // N0=1 bands can't be split down to N<2.
                    else if n0 <= 1 {
                        lm0 = imin(i, 1);
                        n0 <<= lm0;
                    }
                    // Compute the cost for the lowest-level PVQ of a fully split band.
                    let pcache = &bits[cindex[((lm0 + 1) * nb + j) as usize] as usize..];
                    max_bits = i32::from(pcache[pcache[0] as usize]) + 1;
                    // Add in the cost of coding regular splits.
                    let mut n = n0;
                    for k in 0..i - lm0 {
                        max_bits <<= 1;
                        // Offset the number of qtheta bits by log2(N)/2 + QTHETA_OFFSET
                        // compared to their "fair share" of total/N
                        let offset = ((i32::from(m.log_n[j as usize])
                            + (((lm0 + k) as u32) << BITRES) as i32)
                            >> 1)
                            - QTHETA_OFFSET;
                        // The number of qtheta bits we'll allocate if the remainder is to be
                        // max_bits. The average measured cost for theta is 0.89701 times qb,
                        // approximated here as 459/512.
                        let num = 459 * ((2 * n - 1) * offset + max_bits);
                        let den = ((2 * n - 1) << 9) - 459;
                        let qb = imin((num + (den >> 1)) / den, 57);
                        celt_assert!(qb >= 0);
                        max_bits += qb;
                        n <<= 1;
                    }
                    // Add in the cost of a stereo split, if necessary.
                    if c == 2 {
                        max_bits <<= 1;
                        let offset = ((i32::from(m.log_n[j as usize]) + (i << BITRES)) >> 1)
                            - if n == 2 {
                                QTHETA_OFFSET_TWOPHASE
                            } else {
                                QTHETA_OFFSET
                            };
                        let ndof = 2 * n - 1 - i32::from(n == 2);
                        // The average measured cost for theta with the step PDF is 0.95164
                        // times qb, approximated here as 487/512.
                        let num = if n == 2 { 512 } else { 487 } * (max_bits + ndof * offset);
                        let den = (ndof << 9) - if n == 2 { 512 } else { 487 };
                        let qb = imin((num + (den >> 1)) / den, if n == 2 { 64 } else { 61 });
                        celt_assert!(qb >= 0);
                        max_bits += qb;
                    }
                    // Add the fine bits we'll use.
                    // Compensate for the extra DoF in stereo
                    let ndof = c * n + i32::from(c == 2 && n > 2);
                    // Offset the number of fine bits by log2(N)/2 + FINE_OFFSET compared to
                    // their "fair share" of total/N
                    let mut offset =
                        ((i32::from(m.log_n[j as usize]) + (i << BITRES)) >> 1) - FINE_OFFSET;
                    // N=2 is the only point that doesn't match the curve
                    if n == 2 {
                        offset += 1 << BITRES >> 2;
                    }
                    // The number of fine bits we'll allocate if the remainder is to be max_bits.
                    let num = max_bits + ndof * offset;
                    let den = (ndof - 1) << BITRES;
                    let qb = imin((num + (den >> 1)) / den, MAX_FINE_BITS);
                    celt_assert!(qb >= 0);
                    max_bits += c * qb << BITRES;
                }
                let width = i32::from(e_bands[(j + 1) as usize]) - i32::from(e_bands[j as usize]);
                max_bits = (4 * max_bits / (c * (width << i))) - 64;
                celt_assert!(max_bits >= 0);
                celt_assert!(max_bits < 256);
                caps.push(max_bits as u8);
            }
        }
    }
    m.cache = PulseCache {
        size: curr,
        index: Cow::Owned(cindex),
        bits: Cow::Owned(bits),
        caps: Cow::Owned(caps),
    };
}

/// Port of celt/rate.c:interp_bits2pulses.
///
/// `bits` is the C `bits` output (the caller's `pulses` array).
fn interp_bits2pulses(
    m: &CeltMode,
    start: i32,
    end: i32,
    skip_start: i32,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    balance_out: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EcCoder<'_, '_>,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let eb = |i: i32| i32::from(m.e_bands[i as usize]);
    let alloc_floor = c << BITRES;
    let stereo = i32::from(c > 1);

    let log_m = lm << BITRES;
    let mut lo: i32 = 0;
    let mut hi: i32 = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum: i32 = 0;
        let mut done = false;
        let mut j = end;
        while j > start {
            j -= 1;
            let ju = j as usize;
            let tmp = bits1[ju] + (mid * bits2[ju] >> ALLOC_STEPS);
            if tmp >= thresh[ju] || done {
                done = true;
                // Don't allocate more than we can actually use
                psum += imin(tmp, cap[ju]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum: i32 = 0;
    let mut done = false;
    let mut j = end;
    while j > start {
        j -= 1;
        let ju = j as usize;
        let mut tmp = bits1[ju] + (lo * bits2[ju] >> ALLOC_STEPS);
        if tmp < thresh[ju] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        // Don't allocate more than we can actually use
        tmp = imin(tmp, cap[ju]);
        bits[ju] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working backwards from the end.
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        let ju = j as usize;
        // Never skip the first band, nor a band that has been boosted by dynalloc.
        // In the first case, we'd be coding a bit to signal we're going to waste all the other
        // bits.
        // In the second case, we'd be coding a bit to redistribute all the bits we just
        // signaled should be concentrated in this band.
        if j <= skip_start {
            // Give the bit we reserved to end skipping back.
            total += skip_rsv;
            break;
        }
        // Figure out how many left-over bits we would be adding to this band.
        // This can include bits we've stolen back from higher, skipped bands.
        let mut left = total - psum;
        let percoeff = celt_udiv(left as u32, (eb(coded_bands) - eb(start)) as u32) as i32;
        left -= (eb(coded_bands) - eb(start)) * percoeff;
        let rem = imax(left - (eb(j) - eb(start)), 0);
        let band_width = eb(coded_bands) - eb(j);
        let mut band_bits = bits[ju] + percoeff * band_width + rem;
        // Only code a skip decision if we're above the threshold for this band.
        // Otherwise it is force-skipped.
        // This ensures that we have enough bits to code the skip flag.
        if band_bits >= imax(thresh[ju], alloc_floor + (1 << BITRES)) {
            match ec {
                EcCoder::Enc(enc) => {
                    // This if() block is the only part of the allocation function that is not a
                    // mandatory part of the bitstream: any bands we choose to skip here must be
                    // explicitly signaled.
                    // We choose a threshold with some hysteresis to keep bands from fluctuating
                    // in and out, but we try not to fold below a certain point.
                    let depth_threshold = if coded_bands > 17 {
                        if j < prev { 7 } else { 9 }
                    } else {
                        0
                    };
                    #[cfg(not(feature = "fuzzing"))]
                    let skip = coded_bands <= start + 2
                        || (band_bits > (depth_threshold * band_width << lm << BITRES) >> 4
                            && j <= signal_bandwidth);
                    // FUZZING: a random skip decision.
                    #[cfg(feature = "fuzzing")]
                    let skip = {
                        let _ = (signal_bandwidth, depth_threshold);
                        (crate::glibc_rand::rand() & 0x1) == 0
                    };
                    if skip {
                        enc.enc_bit_logp(true, 1);
                        break;
                    }
                    enc.enc_bit_logp(false, 1);
                }
                EcCoder::Dec(dec) => {
                    if dec.dec_bit_logp(1) {
                        break;
                    }
                }
            }
            // We used a bit to skip this band.
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // Reclaim the bits originally allocated to this band.
        psum -= bits[ju] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = i32::from(LOG2_FRAC_TABLE[(j - start) as usize]);
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            // If we have enough for a fine energy bit per channel, use it.
            psum += alloc_floor;
            bits[ju] = alloc_floor;
        } else {
            // Otherwise this band gets nothing at all.
            bits[ju] = 0;
        }
        coded_bands -= 1;
    }

    celt_assert!(coded_bands > start);
    // Code the intensity and dual stereo parameters.
    if intensity_rsv > 0 {
        match ec {
            EcCoder::Enc(enc) => {
                *intensity = imin(*intensity, coded_bands);
                enc.enc_uint(
                    (*intensity - start) as u32,
                    (coded_bands + 1 - start) as u32,
                );
            }
            EcCoder::Dec(dec) => {
                *intensity = start + dec.dec_uint((coded_bands + 1 - start) as u32) as i32;
            }
        }
    } else {
        *intensity = 0;
    }
    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        match ec {
            EcCoder::Enc(enc) => enc.enc_bit_logp(*dual_stereo != 0, 1),
            EcCoder::Dec(dec) => *dual_stereo = i32::from(dec.dec_bit_logp(1)),
        }
    } else {
        *dual_stereo = 0;
    }

    // Allocate the remaining bits
    let mut left = total - psum;
    let percoeff = celt_udiv(left as u32, (eb(coded_bands) - eb(start)) as u32) as i32;
    left -= (eb(coded_bands) - eb(start)) * percoeff;
    for j in start..coded_bands {
        bits[j as usize] += percoeff * (eb(j + 1) - eb(j));
    }
    for j in start..coded_bands {
        let tmp = imin(left, eb(j + 1) - eb(j));
        bits[j as usize] += tmp;
        left -= tmp;
    }

    let mut balance: i32 = 0;
    let mut j = start;
    while j < coded_bands {
        let ju = j as usize;
        celt_assert!(bits[ju] >= 0);
        let n0 = eb(j + 1) - eb(j);
        let n = n0 << lm;
        let bit = bits[ju] + balance;
        let mut excess: i32;

        if n > 1 {
            excess = imax(bit - cap[ju], 0);
            bits[ju] = bit - excess;

            // Compensate for the extra DoF in stereo
            let den = c * n + i32::from(c == 2 && n > 2 && *dual_stereo == 0 && j < *intensity);

            let nclogn = den * (i32::from(m.log_n[ju]) + log_m);

            // Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET compared to their
            // "fair share" of total/N
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;

            // N=2 is the only point that doesn't match the curve
            if n == 2 {
                offset += den << BITRES >> 2;
            }

            // Changing the offset for allocating the second and third fine energy bit
            if bits[ju] + offset < den * 2 << BITRES {
                offset += nclogn >> 2;
            } else if bits[ju] + offset < den * 3 << BITRES {
                offset += nclogn >> 3;
            }

            // Divide with rounding
            ebits[ju] = imax(0, bits[ju] + offset + (den << (BITRES - 1)));
            ebits[ju] = (celt_udiv(ebits[ju] as u32, den as u32) >> BITRES) as i32;

            // Make sure not to bust
            if c * ebits[ju] > (bits[ju] >> BITRES) {
                ebits[ju] = bits[ju] >> stereo >> BITRES;
            }

            // More than that is useless because that's about as far as PVQ can go
            ebits[ju] = imin(ebits[ju], MAX_FINE_BITS);

            // If we rounded down or capped this band, make it a candidate for the final fine
            // energy pass
            fine_priority[ju] = i32::from(ebits[ju] * (den << BITRES) >= bits[ju] + offset);

            // Remove the allocated fine bits; the rest are assigned to PVQ
            bits[ju] -= c * ebits[ju] << BITRES;
        } else {
            // For N=1, all bits go to fine energy except for a single sign bit
            excess = imax(0, bit - (c << BITRES));
            bits[ju] = bit - excess;
            ebits[ju] = 0;
            fine_priority[ju] = 1;
        }

        // Fine energy can't take advantage of the re-balancing in quant_all_bands().
        // Instead, do the re-balancing here.
        if excess > 0 {
            let extra_fine = imin(excess >> (stereo + BITRES), MAX_FINE_BITS - ebits[ju]);
            ebits[ju] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine_priority[ju] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;

        celt_assert!(bits[ju] >= 0);
        celt_assert!(ebits[ju] >= 0);
        j += 1;
    }
    // Save any remaining bits over the cap for the rebalancing in quant_all_bands().
    *balance_out = balance;

    // The skipped bands use all their bits for fine energy.
    while j < end {
        let ju = j as usize;
        ebits[ju] = bits[ju] >> stereo >> BITRES;
        celt_assert!(c * ebits[ju] << BITRES == bits[ju]);
        bits[ju] = 0;
        fine_priority[ju] = i32::from(ebits[ju] < 1);
        j += 1;
    }
    coded_bands
}

/// Port of celt/rate.c:clt_compute_allocation.
///
/// Computes the pulse (`pulses`), fine energy (`ebits`) and fine priority allocation for bands
/// `start..end`, coding (or decoding, depending on `ec`) the band skip, intensity and dual
/// stereo decisions. Returns the number of coded bands. The C `encode` flag is
/// `ec.is_encoder()`.
pub fn clt_compute_allocation(
    m: &CeltMode,
    start: i32,
    end: i32,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EcCoder<'_, '_>,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let eb = |i: i32| i32::from(m.e_bands[i as usize]);
    total = imax(total, 0);
    let len = m.nb_ebands;
    let mut skip_start = start;
    // Reserve a bit to signal the end of manually skipped bands.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    // Reserve bits for the intensity and dual stereo parameters.
    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if c == 2 {
        intensity_rsv = i32::from(LOG2_FRAC_TABLE[(end - start) as usize]);
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let mut bits1 = [0i32; MAX_ALLOC_BANDS];
    let mut bits2 = [0i32; MAX_ALLOC_BANDS];
    let mut thresh = [0i32; MAX_ALLOC_BANDS];
    let mut trim_offset = [0i32; MAX_ALLOC_BANDS];
    let bits1 = &mut bits1[..len as usize];
    let bits2 = &mut bits2[..len as usize];
    let thresh = &mut thresh[..len as usize];
    let trim_offset = &mut trim_offset[..len as usize];

    for j in start..end {
        let ju = j as usize;
        // Below this threshold, we're sure not to allocate any PVQ bits
        thresh[ju] = imax(c << BITRES, (3 * (eb(j + 1) - eb(j)) << lm << BITRES) >> 4);
        // Tilt of the allocation curve
        trim_offset[ju] =
            c * (eb(j + 1) - eb(j)) * (alloc_trim - 5 - lm) * (end - j - 1) * (1 << (lm + BITRES))
                >> 6;
        // Giving less resolution to single-coefficient bands because they get more benefit from
        // having one coarse value per coefficient
        if (eb(j + 1) - eb(j)) << lm == 1 {
            trim_offset[ju] -= c << BITRES;
        }
    }
    let mut lo: i32 = 1;
    let mut hi: i32 = m.nb_alloc_vectors - 1;
    loop {
        let mut done = false;
        let mut psum: i32 = 0;
        let mid = (lo + hi) >> 1;
        let mut j = end;
        while j > start {
            j -= 1;
            let ju = j as usize;
            let n = eb(j + 1) - eb(j);
            let mut bitsj = c * n * i32::from(m.alloc_vectors[(mid * len + j) as usize]) << lm >> 2;
            if bitsj > 0 {
                bitsj = imax(0, bitsj + trim_offset[ju]);
            }
            bitsj += offsets[ju];
            if bitsj >= thresh[ju] || done {
                done = true;
                // Don't allocate more than we can actually use
                psum += imin(bitsj, cap[ju]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;
    for j in start..end {
        let ju = j as usize;
        let n = eb(j + 1) - eb(j);
        let mut bits1j = c * n * i32::from(m.alloc_vectors[(lo * len + j) as usize]) << lm >> 2;
        let mut bits2j = if hi >= m.nb_alloc_vectors {
            cap[ju]
        } else {
            c * n * i32::from(m.alloc_vectors[(hi * len + j) as usize]) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = imax(0, bits1j + trim_offset[ju]);
        }
        if bits2j > 0 {
            bits2j = imax(0, bits2j + trim_offset[ju]);
        }
        if lo > 0 {
            bits1j += offsets[ju];
        }
        bits2j += offsets[ju];
        if offsets[ju] > 0 {
            skip_start = j;
        }
        bits2j = imax(0, bits2j - bits1j);
        bits1[ju] = bits1j;
        bits2[ju] = bits2j;
    }
    interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        bits1,
        bits2,
        thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        ec,
        prev,
        signal_bandwidth,
    )
}

/// `last_zero` ICDF of the QEXT depth coder.
#[cfg(feature = "qext")]
static LAST_ZERO: [u8; 3] = [64, 50, 0];
/// `last_cap` ICDF of the QEXT depth coder.
#[cfg(feature = "qext")]
static LAST_CAP: [u8; 3] = [110, 60, 0];
/// `last_other` ICDF of the QEXT depth coder.
#[cfg(feature = "qext")]
static LAST_OTHER: [u8; 4] = [120, 112, 70, 0];

// `eMeans` lives in celt/quant_bands.rs (its C home; Q4 `i8` in the fixed-point build).
#[cfg(feature = "qext")]
use crate::celt::quant_bands::E_MEANS;

/// `FUZZING` (celt/rate.c:clt_compute_extra_allocation): a random depth,
/// `(int)-depth_std*log(1e-8+(float)rand()/(float)RAND_MAX)`, i.e. `(int)(-depth_std)` times the
/// double logarithm, truncated.
#[cfg(all(feature = "fuzzing", feature = "qext"))]
fn fuzzing_depth(depth_std: f32) -> i32 {
    let u = crate::glibc_rand::rand() as f32 / crate::glibc_rand::RAND_MAX as f32;
    (f64::from((-depth_std) as i32) * crate::math::log(1e-8 + f64::from(u))) as i32
}

/// Port of celt/rate.c:ec_enc_depth.
#[cfg(feature = "qext")]
fn ec_enc_depth(enc: &mut crate::celt::entenc::EcEnc<'_>, depth: i32, cap: i32, last: &mut i32) {
    let mut sym: i32 = 3;
    if depth == *last {
        sym = 2;
    }
    if depth == cap {
        sym = 1;
    }
    if depth == 0 {
        sym = 0;
    }
    if *last == 0 {
        enc.enc_icdf(imin(sym, 2) as usize, &LAST_ZERO, 7);
    } else if *last == cap {
        enc.enc_icdf(imin(sym, 2) as usize, &LAST_CAP, 7);
    } else {
        enc.enc_icdf(sym as usize, &LAST_OTHER, 7);
    }
    // We accept some redundancy if depth==last (for last different from 0 and cap).
    if sym == 3 {
        enc.enc_uint((depth - 1) as u32, cap as u32);
    }
    *last = depth;
}

/// Port of celt/rate.c:ec_dec_depth.
#[cfg(feature = "qext")]
fn ec_dec_depth(dec: &mut crate::celt::entdec::EcDec<'_>, cap: i32, last: &mut i32) -> i32 {
    let mut sym: i32;
    if *last == 0 {
        sym = dec.dec_icdf(&LAST_ZERO, 7) as i32;
        if sym == 2 {
            sym = 3;
        }
    } else if *last == cap {
        sym = dec.dec_icdf(&LAST_CAP, 7) as i32;
        if sym == 2 {
            sym = 3;
        }
    } else {
        sym = dec.dec_icdf(&LAST_OTHER, 7) as i32;
    }
    let depth = match sym {
        0 => 0,
        1 => cap,
        2 => *last,
        _ => 1 + dec.dec_uint(cap as u32) as i32,
    };
    *last = depth;
    depth
}

/// Port of celt/rate.c:median_of_5_val16.
#[cfg(feature = "qext")]
#[must_use]
fn median_of_5_val16(x: &[OpusVal16]) -> OpusVal16 {
    let (mut t0, mut t1, t2, mut t3, mut t4);
    t2 = x[2];
    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    if x[3] > x[4] {
        t3 = x[4];
        t4 = x[3];
    } else {
        t3 = x[3];
        t4 = x[4];
    }
    if t0 > t3 {
        core::mem::swap(&mut t0, &mut t3);
        core::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 {
            min16(t2, t3)
        } else {
            min16(t4, t1)
        }
    } else if t2 < t3 {
        min16(t1, t3)
    } else {
        min16(t2, t4)
    }
}

/// Upper bound on `tot_bands` in [`clt_compute_extra_allocation`] (`nbEBands` +
/// `NB_QEXT_BANDS`).
#[cfg(feature = "qext")]
const MAX_EXTRA_BANDS: usize = MAX_ALLOC_BANDS + crate::celt::modes::NB_QEXT_BANDS as usize;

/// Port of celt/rate.c:clt_compute_extra_allocation (QEXT).
///
/// `qext_band_log_e` is only read when `qext_mode` is `Some`. The C `encode` flag is
/// `ec.is_encoder()`.
#[cfg(feature = "qext")]
pub fn clt_compute_extra_allocation(
    m: &CeltMode,
    qext_mode: Option<&CeltMode>,
    start: i32,
    end: i32,
    qext_end: i32,
    band_log_e: &[CeltGlog],
    qext_band_log_e: &[CeltGlog],
    mut total: i32,
    extra_pulses: &mut [i32],
    extra_equant: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EcCoder<'_, '_>,
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
) {
    use crate::celt::modes::NB_QEXT_BANDS;
    #[cfg(any(not(feature = "fixed-point"), feature = "fuzzing"))]
    use crate::math;

    let eb = |i: i32| i32::from(m.e_bands[i as usize]);
    let mut last: i32 = 0;
    let tot_bands: i32;
    let tot_samples: i32;
    // FUZZING: the standard deviation of the random depths, drawn on every call (the decoder's
    // too, which advances the shared generator like C).
    #[cfg(feature = "fuzzing")]
    let depth_std: f32 = {
        let d = (-10.0f32 as f64
            * math::log(
                1e-8 + f64::from(
                    crate::glibc_rand::rand() as f32 / crate::glibc_rand::RAND_MAX as f32,
                ),
            )) as f32;
        // FMAX(0, FMIN(48, depth_std))
        let d = if 48.0 < d { 48.0 } else { d };
        if 0.0 > d { 0.0 } else { d }
    };
    if let Some(q) = qext_mode {
        celt_assert!(end == m.nb_ebands);
        tot_bands = end + qext_end;
        tot_samples = i32::from(q.e_bands[qext_end as usize]) * c << lm;
    } else {
        tot_bands = end;
        tot_samples = (eb(end) - eb(start)) * c << lm;
    }
    let mut cap = [0i32; MAX_EXTRA_BANDS];
    let cap = &mut cap[..tot_bands as usize];
    for i in start..end {
        cap[i as usize] = 12;
    }
    if qext_mode.is_some() {
        for i in 0..qext_end {
            cap[(end + i) as usize] = 14;
        }
    }
    if total <= 0 {
        for i in start..m.nb_ebands + qext_end {
            extra_pulses[i as usize] = 0;
            extra_equant[i as usize] = 0;
        }
        return;
    }
    let mut depth = [0i32; MAX_EXTRA_BANDS];
    let depth = &mut depth[..tot_bands as usize];
    match ec {
        #[cfg(not(feature = "fixed-point"))]
        EcCoder::Enc(enc) => {
            let mut flat_e = [0.0 as OpusVal16; MAX_EXTRA_BANDS];
            let mut min = [0.0 as OpusVal16; MAX_EXTRA_BANDS];
            let mut ncoef = [0i32; MAX_EXTRA_BANDS];
            let mut follower = [0.0 as OpusVal16; MAX_EXTRA_BANDS];
            let flat_e = &mut flat_e[..tot_bands as usize];
            let min = &mut min[..tot_bands as usize];
            let ncoef = &mut ncoef[..tot_bands as usize];
            let follower = &mut follower[..tot_bands as usize];
            for i in start..end {
                ncoef[i as usize] = (eb(i + 1) - eb(i)) * c << lm;
            }
            // Remove the effect of band width, eMeans and pre-emphasis to compute the real
            // (flat) spectrum.
            let flat = |log_e: CeltGlog, log_n: i16, mean: OpusVal16, b: i32| -> OpusVal16 {
                log_e - 0.0625f32 * f32::from(log_n) + mean
                    - 0.0062f32 * (b + 5) as f32 * (b + 5) as f32
            };
            for i in start..end {
                let iu = i as usize;
                flat_e[iu] = flat(band_log_e[iu], m.log_n[iu], E_MEANS[iu], i);
                min[iu] = 0.0;
            }
            if c == 2 {
                for i in start..end {
                    let iu = i as usize;
                    flat_e[iu] = maxg(
                        flat_e[iu],
                        flat(
                            band_log_e[m.nb_ebands as usize + iu],
                            m.log_n[iu],
                            E_MEANS[iu],
                            i,
                        ),
                    );
                }
            }
            flat_e[(end - 1) as usize] += 2.0f32;
            if let Some(q) = qext_mode {
                let qeb = |i: i32| i32::from(q.e_bands[i as usize]);
                let mut min_depth: OpusVal16 = 0.0;
                // If we have enough bits, give at least 1 bit of depth to all higher bands.
                if total >= 3 * c * (qeb(qext_end) - qeb(start)) << lm << BITRES
                    && (toneishness < 0.98f32 || tone_freq > 1.33f32)
                {
                    min_depth = 1.0f32;
                }
                for i in 0..qext_end {
                    ncoef[(end + i) as usize] = (qeb(i + 1) - qeb(i)) * c << lm;
                    min[(end + i) as usize] = min_depth;
                }
                for i in 0..qext_end {
                    let iu = i as usize;
                    flat_e[(end + i) as usize] =
                        flat(qext_band_log_e[iu], q.log_n[iu], E_MEANS[iu], end + i);
                }
                if c == 2 {
                    for i in 0..qext_end {
                        let iu = i as usize;
                        flat_e[(end + i) as usize] = maxg(
                            flat_e[(end + i) as usize],
                            flat(
                                qext_band_log_e[NB_QEXT_BANDS as usize + iu],
                                q.log_n[iu],
                                E_MEANS[iu],
                                end + i,
                            ),
                        );
                    }
                }
            }
            for i in start + 2..tot_bands - 2 {
                let iu = i as usize;
                follower[iu] = median_of_5_val16(&flat_e[iu - 2..]);
            }
            let s = start as usize;
            follower[s] = follower[s + 2];
            follower[s + 1] = follower[s + 2];
            let t = tot_bands as usize;
            follower[t - 1] = follower[t - 3];
            follower[t - 2] = follower[t - 3];
            for i in start + 1..tot_bands {
                let iu = i as usize;
                follower[iu] = max16(follower[iu], follower[iu - 1] - 1.0f32);
            }
            let mut i = tot_bands - 2;
            while i >= start {
                let iu = i as usize;
                follower[iu] = max16(follower[iu], follower[iu + 1] - 1.0f32);
                i -= 1;
            }
            for i in start..tot_bands {
                let iu = i as usize;
                flat_e[iu] -= (1.0f32 - toneishness) * follower[iu];
            }
            if qext_mode.is_some() {
                for i in 0..qext_end {
                    let iu = (end + i) as usize;
                    flat_e[iu] = flat_e[iu] + 3.0f32 + 0.2f32 * i as f32;
                }
            }
            // Approximate fill level assuming all bands contribute fully.
            let mut sum: OpusVal32 = 0.0;
            for i in start..tot_bands {
                let iu = i as usize;
                sum += ncoef[iu] as f32 * flat_e[iu];
            }
            total >>= BITRES;
            let mut fill: OpusVal32 = (total as f32 + sum) / tot_samples as f32;
            // Iteratively refine the fill level considering the depth min and cap.
            for _ in 0..10 {
                sum = 0.0;
                for i in start..tot_bands {
                    let iu = i as usize;
                    sum +=
                        ncoef[iu] as f32 * min32(cap[iu] as f32, max32(min[iu], flat_e[iu] - fill));
                }
                fill -= (total as f32 - sum) / tot_samples as f32;
            }
            for i in start..tot_bands {
                let iu = i as usize;
                depth[iu] = math::floor(
                    0.5 + f64::from(
                        4.0f32 * min32(cap[iu] as f32, max32(min[iu], flat_e[iu] - fill)),
                    ),
                ) as i32;
                #[cfg(feature = "fuzzing")]
                {
                    depth[iu] = fuzzing_depth(depth_std);
                    depth[iu] = imax(0, imin(cap[iu] << 2, depth[iu]));
                }
                if enc.tell_frac() + 80 < enc.storage * 8 << BITRES {
                    ec_enc_depth(enc, depth[iu], 4 * cap[iu], &mut last);
                } else {
                    depth[iu] = 0;
                }
            }
        }
        #[cfg(feature = "fixed-point")]
        EcCoder::Enc(enc) => {
            use crate::celt::arch::{
                DB_SHIFT, Q15ONE, gconst, mult16_16, mult16_16_q15, pshr32, qconst16, qconst32,
                shl32,
            };

            let mut flat_e = [0 as OpusVal16; MAX_EXTRA_BANDS];
            let mut min = [0 as OpusVal16; MAX_EXTRA_BANDS];
            let mut ncoef = [0i32; MAX_EXTRA_BANDS];
            let mut follower = [0 as OpusVal16; MAX_EXTRA_BANDS];
            let flat_e = &mut flat_e[..tot_bands as usize];
            let min = &mut min[..tot_bands as usize];
            let ncoef = &mut ncoef[..tot_bands as usize];
            let follower = &mut follower[..tot_bands as usize];
            for i in start..end {
                ncoef[i as usize] = (eb(i + 1) - eb(i)) * c << lm;
            }
            // Remove the effect of band width, eMeans and pre-emphasis to compute the real
            // (flat) spectrum (Q`DB_SHIFT` → Q10; the `int` result is narrowed by the caller).
            let flat = |log_e: CeltGlog, log_n: i16, mean: i8, b: i32| -> i32 {
                pshr32(
                    log_e - gconst(0.0625f32 as f64) * i32::from(log_n) + shl32(mean, DB_SHIFT - 4)
                        - gconst(0.0062f32 as f64) * (b + 5) * (b + 5),
                    DB_SHIFT - 10,
                )
            };
            for i in start..end {
                let iu = i as usize;
                flat_e[iu] = flat(band_log_e[iu], m.log_n[iu], E_MEANS[iu], i) as i16;
                min[iu] = 0;
            }
            if c == 2 {
                for i in start..end {
                    let iu = i as usize;
                    flat_e[iu] = maxg(
                        i32::from(flat_e[iu]),
                        flat(
                            band_log_e[m.nb_ebands as usize + iu],
                            m.log_n[iu],
                            E_MEANS[iu],
                            i,
                        ),
                    ) as i16;
                }
            }
            let e1 = (end - 1) as usize;
            flat_e[e1] = (i32::from(flat_e[e1]) + i32::from(qconst16(2.0, 10))) as i16;
            if let Some(q) = qext_mode {
                let qeb = |i: i32| i32::from(q.e_bands[i as usize]);
                let mut min_depth: OpusVal16 = 0;
                // If we have enough bits, give at least 1 bit of depth to all higher bands.
                // (`tone_freq > 1.33f` compares the Q-format integer with a float, as in C.)
                if total >= 3 * c * (qeb(qext_end) - qeb(start)) << lm << BITRES
                    && (toneishness < qconst32(0.98f32 as f64, 29)
                        || f32::from(tone_freq) > 1.33f32)
                {
                    min_depth = qconst16(1.0, 10);
                }
                for i in 0..qext_end {
                    ncoef[(end + i) as usize] = (qeb(i + 1) - qeb(i)) * c << lm;
                    min[(end + i) as usize] = min_depth;
                }
                for i in 0..qext_end {
                    let iu = i as usize;
                    flat_e[(end + i) as usize] =
                        flat(qext_band_log_e[iu], q.log_n[iu], E_MEANS[iu], end + i) as i16;
                }
                if c == 2 {
                    for i in 0..qext_end {
                        let iu = i as usize;
                        flat_e[(end + i) as usize] = maxg(
                            i32::from(flat_e[(end + i) as usize]),
                            flat(
                                qext_band_log_e[NB_QEXT_BANDS as usize + iu],
                                q.log_n[iu],
                                E_MEANS[iu],
                                end + i,
                            ),
                        ) as i16;
                    }
                }
            }
            for i in start + 2..tot_bands - 2 {
                let iu = i as usize;
                follower[iu] = median_of_5_val16(&flat_e[iu - 2..]);
            }
            let s = start as usize;
            follower[s] = follower[s + 2];
            follower[s + 1] = follower[s + 2];
            let t = tot_bands as usize;
            follower[t - 1] = follower[t - 3];
            follower[t - 2] = follower[t - 3];
            let one_q10 = i32::from(qconst16(1.0, 10));
            for i in start + 1..tot_bands {
                let iu = i as usize;
                follower[iu] = max16(
                    i32::from(follower[iu]),
                    i32::from(follower[iu - 1]) - one_q10,
                ) as i16;
            }
            let mut i = tot_bands - 2;
            while i >= start {
                let iu = i as usize;
                follower[iu] = max16(
                    i32::from(follower[iu]),
                    i32::from(follower[iu + 1]) - one_q10,
                ) as i16;
                i -= 1;
            }
            for i in start..tot_bands {
                let iu = i as usize;
                flat_e[iu] = (i32::from(flat_e[iu])
                    - mult16_16_q15(i32::from(Q15ONE) - pshr32(toneishness, 14), follower[iu]))
                    as i16;
            }
            if qext_mode.is_some() {
                for i in 0..qext_end {
                    let iu = (end + i) as usize;
                    flat_e[iu] = (i32::from(flat_e[iu])
                        + i32::from(qconst16(3.0, 10))
                        + i32::from(qconst16(0.2f32 as f64, 10)) * i)
                        as i16;
                }
            }
            // Approximate fill level assuming all bands contribute fully.
            let mut sum: OpusVal32 = 0;
            for i in start..tot_bands {
                let iu = i as usize;
                sum += mult16_16(ncoef[iu], flat_e[iu]);
            }
            total >>= BITRES;
            let mut fill: OpusVal32 = (shl32(total, 10) + sum) / tot_samples;
            // Iteratively refine the fill level considering the depth min and cap.
            let level = |iu: usize, fill: OpusVal32| -> OpusVal32 {
                min32(
                    shl32(cap[iu], 10),
                    max32(i32::from(min[iu]), i32::from(flat_e[iu]) - fill),
                )
            };
            for _ in 0..10 {
                sum = 0;
                for i in start..tot_bands {
                    let iu = i as usize;
                    sum += ncoef[iu] * level(iu, fill);
                }
                fill -= (shl32(total, 10) - sum) / tot_samples;
            }
            for i in start..tot_bands {
                let iu = i as usize;
                depth[iu] = pshr32(level(iu, fill), 10 - 2);
                #[cfg(feature = "fuzzing")]
                {
                    depth[iu] = fuzzing_depth(depth_std);
                    depth[iu] = imax(0, imin(cap[iu] << 2, depth[iu]));
                }
                if enc.tell_frac() + 80 < enc.storage * 8 << BITRES {
                    ec_enc_depth(enc, depth[iu], 4 * cap[iu], &mut last);
                } else {
                    depth[iu] = 0;
                }
            }
        }
        EcCoder::Dec(dec) => {
            for i in start..tot_bands {
                let iu = i as usize;
                if dec.tell_frac() + 80 < dec.storage * 8 << BITRES {
                    depth[iu] = ec_dec_depth(dec, 4 * cap[iu], &mut last);
                } else {
                    depth[iu] = 0;
                }
            }
        }
    }
    for i in start..end {
        let iu = i as usize;
        extra_equant[iu] = (depth[iu] + 3) >> 2;
        extra_pulses[iu] =
            ((((eb(i + 1) - eb(i)) << lm) - 1) * c * depth[iu] * (1 << BITRES) + 2) >> 2;
    }
    if let Some(q) = qext_mode {
        let qeb = |i: i32| i32::from(q.e_bands[i as usize]);
        for i in 0..qext_end {
            let iu = (end + i) as usize;
            extra_equant[iu] = (depth[iu] + 3) >> 2;
            extra_pulses[iu] =
                ((((qeb(i + 1) - qeb(i)) << lm) - 1) * c * depth[iu] * (1 << BITRES) + 2) >> 2;
        }
    }
}

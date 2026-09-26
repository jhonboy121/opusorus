//! Fixed-point (`#ifdef FIXED_POINT`) versions of the celt/celt_encoder.c analysis helpers whose
//! arithmetic differs from the float build (feature `fixed-point`; `ENABLE_RES24` with
//! `fixed-res24`, `ENABLE_QEXT` with `qext`).
//!
//! Every function mirrors the C fixed-point path macro by macro (see `crate::celt::arch` for
//! the typing rules): Q15 gains, Q14 `tf_estimate`, Q13 `tone_freq`, Q29 `toneishness` and LPC
//! coefficients, Q24 (`DB_SHIFT`) log energies, Q24 (`NORM_SHIFT`) normalised bands and
//! `SIG_SHIFT` signals. Implicit C narrowings to `opus_val16` are written `as i16`.

use crate::celt::arch::{
    CeltGlog, CeltNorm, CeltSig, DB_SHIFT, EPSILON, NORM_SHIFT, OpusRes, OpusVal16, OpusVal32,
    SIG_SHIFT, abs16, abs32, add16, add32, celt_isnan, div32, div32_16, extend32, extract16,
    gconst, half16, half32, imax, imin, mac16_32_q15, max16, max32, maxg, min16, min32, ming,
    mult16_16, mult16_16_q14, mult16_16_q15, mult16_32_q15, mult32_32_q31, pshr32, qconst16,
    qconst32, res2sig, shl16, shl32, shr16, shr32, sround16, sub32,
};
#[cfg(not(feature = "disable-float-api"))]
use crate::celt::celt::LEAK_BANDS;
use crate::celt::celt::comb_filter;
use crate::celt::celt::{AnalysisInfo, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::entcode::BITRES;
use crate::celt::mathops::{
    celt_exp2_db, celt_ilog2, celt_log2, celt_maxabs16, celt_maxabs32, celt_sqrt, frac_div32_q29,
};
use crate::celt::pitch::{pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::E_MEANS;
use crate::celt::static_modes::CeltMode;
use crate::celt::vq::celt_inner_prod_norm_shift;

use super::{DynallocScratch, INV_TABLE, PrefilterOut, PrefilterState};

/// Port of celt/celt_encoder.c:transient_analysis (fixed-point build).
///
/// `input` holds `c` channels of `len` samples. `tmp` is scratch of at least `len` samples.
/// Returns `is_transient`; writes `tf_estimate` (Q14), `tf_chan` and `weak_transient`.
/// `tone_freq` is Q13 and `toneishness` Q29.
pub fn transient_analysis(
    input: &[OpusVal32],
    len: i32,
    c: i32,
    tf_estimate: &mut OpusVal16,
    tf_chan: &mut i32,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
    tmp: &mut [OpusVal16],
) -> bool {
    let mut mask_metric: i32 = 0;
    // Forward masking: 6.7 dB/ms.
    let mut forward_shift: i32 = 4;
    let lenu = len as usize;
    let cu = c as usize;
    let input = &input[..cu * lenu];
    let in_shift = imax(0, celt_ilog2(1 + celt_maxabs32(input)) - 14);
    let tmp = &mut tmp[..lenu];

    *weak_transient = false;
    // For lower bitrates, let's be more conservative and have a forward masking decay of
    // 3.3 dB/ms. This avoids having to code transients at very low bitrate (mostly for
    // hybrid), which can result in unstable energy and/or partial collapse.
    if allow_weak_transients {
        forward_shift = 5;
    }
    let len2 = len / 2;
    let len2u = len2 as usize;
    for ch in 0..cu {
        let mut unmask: i32 = 0;
        let mut mem0: OpusVal32 = 0;
        let mut mem1: OpusVal32 = 0;
        // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2)
        for i in 0..lenu {
            let x: OpusVal32 = shr32(input[i + ch * lenu], in_shift);
            let y: OpusVal32 = add32(mem0, x);
            mem0 = mem1 + y - shl32(x, 1);
            mem1 = x - shr32(y, 1);
            tmp[i] = sround16(y, 2);
        }
        // First few samples are bad because we don't propagate the memory
        tmp[..12].fill(0);

        // Normalize tmp to max range
        {
            let shift = 14 - celt_ilog2(max16(1, celt_maxabs16(tmp)));
            if shift != 0 {
                for v in tmp.iter_mut() {
                    *v = shl16(*v, shift);
                }
            }
        }

        let mut mean: OpusVal32 = 0;
        mem0 = 0;
        // Grouping by two to reduce complexity
        // Forward pass to compute the post-echo threshold
        for i in 0..len2u {
            let x2: OpusVal32 = pshr32(
                mult16_16(tmp[2 * i], tmp[2 * i]) + mult16_16(tmp[2 * i + 1], tmp[2 * i + 1]),
                4,
            );
            mean += pshr32(x2, 12);
            mem0 += pshr32(x2 - mem0, forward_shift);
            tmp[i] = pshr32(mem0, 12) as i16;
        }

        mem0 = 0;
        let mut max_e: OpusVal16 = 0;
        // Backward pass to compute the pre-echo threshold
        for i in (0..len2u).rev() {
            // Backward masking: 13.9 dB/ms.
            mem0 += pshr32(shl32(tmp[i], 4) - mem0, 3);
            tmp[i] = pshr32(mem0, 4) as i16;
            max_e = max16(max_e, tmp[i]);
        }

        // Compute the ratio of the "frame energy" over the harmonic mean of the energy. As a
        // compromise with the old transient detector, frame energy is the geometric mean of
        // the energy and half the max (costs two sqrt() to avoid overflows).
        mean = mult16_16(celt_sqrt(mean), celt_sqrt(mult16_16(max_e, len2 >> 1)));
        // Inverse of the mean energy in Q15+6
        let norm: OpusVal32 = shl32(extend32(len2), 6 + 14) / add32(EPSILON, shr32(mean, 1));
        debug_assert!(!celt_isnan(i32::from(tmp[0])));
        debug_assert!(!celt_isnan(norm));
        let mut i = 12usize;
        while (i as i32) < len2 - 5 {
            // Do not round to nearest
            let id = max32(
                0,
                min32(127, mult16_32_q15(i32::from(tmp[i]) + EPSILON, norm)),
            );
            unmask += i32::from(INV_TABLE[id as usize]);
            i += 4;
        }
        // Normalize, compensate for the 1/4th of the sample and the factor of 6 in the inverse
        // table
        unmask = 64 * unmask * 4 / (6 * (len2 - 17));
        if unmask > mask_metric {
            *tf_chan = ch as i32;
            mask_metric = unmask;
        }
    }
    let mut is_transient = mask_metric > 200;
    // Prevent the transient detector from confusing the partial cycle of a very low frequency
    // tone with a transient.
    if toneishness > qconst32(0.98f32 as f64, 29) && tone_freq < qconst16(0.026f32 as f64, 13) {
        is_transient = false;
        mask_metric = 0;
    }
    // For low bitrates, define "weak transients" that need to be handled differently to avoid
    // partial collapse.
    if allow_weak_transients && is_transient && mask_metric < 600 {
        is_transient = false;
        *weak_transient = true;
    }
    // Arbitrary metric for VBR boost
    let tf_max: OpusVal16 = max16(0, celt_sqrt(27 * mask_metric) - 42) as i16;
    // *tf_estimate = 1 + MIN16(1, sqrt(MAX16(0, tf_max-30))/20);
    *tf_estimate = celt_sqrt(max32(
        0,
        shl32(
            mult16_16(qconst16(0.0069, 14), min16(163, i32::from(tf_max))),
            14,
        ) - qconst32(0.139, 28),
    )) as i16;
    is_transient
}

/// Port of celt/celt_encoder.c:patch_transient_decision (fixed-point build): looks for sudden
/// increases of energy to decide whether we need to patch the transient decision.
#[must_use]
pub fn patch_transient_decision(
    new_e: &[CeltGlog],
    old_e: &[CeltGlog],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
) -> bool {
    // C: celt_glog spread_old[26] (sized larger here for custom modes).
    let mut spread_old: [CeltGlog; 64] = [0; 64];
    let nb = nb_ebands as usize;
    let (start, end) = (start as usize, end as usize);
    // Apply an aggressive (-6 dB/Bark) spreading function to the old frame to avoid false
    // detection caused by irrelevant bands
    if c == 1 {
        spread_old[start] = old_e[start];
        for i in start + 1..end {
            spread_old[i] = maxg(spread_old[i - 1] - gconst(1.0), old_e[i]);
        }
    } else {
        spread_old[start] = maxg(old_e[start], old_e[start + nb]);
        for i in start + 1..end {
            spread_old[i] = maxg(
                spread_old[i - 1] - gconst(1.0),
                maxg(old_e[i], old_e[i + nb]),
            );
        }
    }
    for i in (start..end.saturating_sub(1)).rev() {
        spread_old[i] = maxg(spread_old[i], spread_old[i + 1] - gconst(1.0));
    }
    // Compute mean increase
    let mut mean_diff: OpusVal32 = 0;
    let i0 = imax(2, start as i32);
    for ch in 0..c as usize {
        for i in i0..end as i32 - 1 {
            let iu = i as usize;
            // C stores the Q24 log energies in `opus_val16` variables (truncating).
            let x1: OpusVal16 = maxg(0, new_e[iu + ch * nb]) as i16;
            let x2: OpusVal16 = maxg(0, spread_old[iu]) as i16;
            mean_diff = add32(mean_diff, maxg(0, sub32(x1, x2)));
        }
    }
    mean_diff = div32(mean_diff, c * (end as i32 - 1 - i0));
    mean_diff > gconst(1.0)
}

/// Port of celt/celt_encoder.c:celt_preemphasis (fixed-point build).
///
/// `pcmp` starts at the channel's first sample (C `pcm+c`) and is read with stride `cc`;
/// `inp` receives `n` pre-emphasised samples. `coef` is `mode->preemph`.
pub fn celt_preemphasis(
    pcmp: &[OpusRes],
    inp: &mut [CeltSig],
    n: i32,
    cc: i32,
    upsample: i32,
    coef: &[OpusVal16; 4],
    mem: &mut CeltSig,
    clip: bool,
) {
    let coef0 = coef[0];
    let mut m = *mem;
    let nn = n as usize;
    let ccu = cc as usize;
    let inp = &mut inp[..nn];

    // Fast path for the normal 48kHz case and no clipping
    if coef[1] == 0 && upsample == 1 && !clip {
        for i in 0..nn {
            let x: CeltSig = res2sig(pcmp[ccu * i]);
            // Apply pre-emphasis
            inp[i] = x - m;
            m = mult16_32_q15(coef0, x);
        }
        *mem = m;
        return;
    }

    let nu = (n / upsample) as usize;
    let up = upsample as usize;
    if upsample != 1 {
        inp.fill(0);
    }
    for i in 0..nu {
        inp[i * up] = res2sig(pcmp[ccu * i]);
    }

    #[cfg(feature = "fixed-res24")]
    if clip {
        // Clip input to avoid encoding non-portable files
        for i in 0..nu {
            inp[i * up] = max32(
                -(65536 << SIG_SHIFT),
                min32(65536 << SIG_SHIFT, inp[i * up]),
            );
        }
    }
    #[cfg(any(feature = "custom-modes", feature = "qext"))]
    if coef[1] != 0 {
        let coef1 = coef[1];
        // If we need the extra precision, we use the fact that coef[3] is exact to do a
        // Newton-Raphson iteration and get us more precision on coef[2].
        #[cfg(feature = "qext")]
        let coef2_q30: OpusVal32 = shl32(coef[2], 18)
            + pshr32(
                mult16_16(qconst32(1.0, 25) - mult16_16(coef[3], coef[2]), coef[2]),
                7,
            );
        #[cfg(feature = "qext")]
        const {
            assert!(SIG_SHIFT == 12)
        };
        #[cfg(not(feature = "qext"))]
        let coef2 = coef[2];
        for i in 0..nn {
            let x: CeltSig = inp[i];
            // Apply pre-emphasis
            #[cfg(feature = "qext")]
            let tmp: CeltSig = shl32(mult32_32_q31(coef2_q30, x), 1);
            #[cfg(not(feature = "qext"))]
            let tmp: CeltSig = shl32(mult16_32_q15(coef2, x), 15 - SIG_SHIFT);
            inp[i] = tmp + m;
            m = mult16_32_q15(coef1, inp[i]) - mult16_32_q15(coef0, tmp);
        }
        *mem = m;
        return;
    }
    for i in 0..nn {
        let x: CeltSig = inp[i];
        // Apply pre-emphasis
        inp[i] = x - m;
        m = mult16_32_q15(coef0, x);
    }
    *mem = m;
}

/// Port of celt/celt_encoder.c:l1_metric (fixed-point build).
#[must_use]
pub fn l1_metric(tmp: &[CeltNorm], n: i32, lm: i32, bias: OpusVal16) -> OpusVal32 {
    let mut l1: OpusVal32 = 0;
    for &v in &tmp[..n as usize] {
        l1 += extend32(abs16(shr32(v, NORM_SHIFT - 14)));
    }
    // When in doubt, prefer good freq resolution
    mac16_32_q15(l1, lm * i32::from(bias), l1)
}

/// `bias` of celt/celt_encoder.c:tf_analysis (fixed-point build; `tf_estimate` is Q14).
#[must_use]
pub fn tf_bias(tf_estimate: OpusVal16) -> OpusVal16 {
    mult16_16_q14(
        qconst16(0.04f32 as f64, 15),
        max16(
            -i32::from(qconst16(0.25, 14)),
            i32::from(qconst16(0.5, 14)) - i32::from(tf_estimate),
        ),
    ) as i16
}

/// Port of celt/celt_encoder.c:alloc_trim_analysis (fixed-point build). Returns the allocation
/// trim index.
#[must_use]
pub fn alloc_trim_analysis(
    m: &CeltMode,
    x: &[CeltNorm],
    band_log_e: &[CeltGlog],
    end: i32,
    lm: i32,
    c: i32,
    n0: i32,
    analysis: &AnalysisInfo,
    stereo_saving: &mut OpusVal16,
    tf_estimate: OpusVal16,
    intensity: i32,
    surround_trim: CeltGlog,
    equiv_rate: i32,
) -> i32 {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut diff: OpusVal32 = 0;
    let mut trim: OpusVal16 = qconst16(5.0, 8);
    // At low bitrate, reducing the trim seems to help. At higher bitrates, it's less clear
    // what's best, so we're keeping it as it was before, at least for now.
    if equiv_rate < 64000 {
        trim = qconst16(4.0, 8);
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) >> 10;
        trim = (i32::from(qconst16(4.0, 8)) + i32::from(qconst16(1.0 / 16.0, 8)) * frac) as i16;
    }
    if c == 2 {
        let mut sum: OpusVal16 = 0; // Q10
        // Compute inter-channel correlation for low frequencies
        for i in 0..8 {
            let off = (eb(i) << lm) as usize;
            let partial = celt_inner_prod_norm_shift(
                &x[off..],
                &x[n0 as usize + off..],
                ((eb(i + 1) - eb(i)) << lm) as usize,
            );
            sum = add16(sum, extract16(shr32(partial, 18)));
        }
        sum = mult16_16_q15(qconst16(1.0 / 8.0, 15), sum) as i16;
        sum = min16(i32::from(qconst16(1.0, 10)), abs16(sum)) as i16;
        let mut min_xc: OpusVal16 = sum; // Q10
        for i in 8..intensity {
            let off = (eb(i) << lm) as usize;
            let partial = celt_inner_prod_norm_shift(
                &x[off..],
                &x[n0 as usize + off..],
                ((eb(i + 1) - eb(i)) << lm) as usize,
            );
            min_xc = min16(i32::from(min_xc), abs16(extract16(shr32(partial, 18)))) as i16;
        }
        min_xc = min16(i32::from(qconst16(1.0, 10)), abs16(min_xc)) as i16;
        // mid-side savings estimations based on the LF average
        let mut log_xc: OpusVal16 = celt_log2(qconst32(1.001f32 as f64, 20) - mult16_16(sum, sum));
        // mid-side savings estimations based on min correlation
        let mut log_xc2: OpusVal16 = max16(
            half16(log_xc),
            i32::from(celt_log2(
                qconst32(1.001f32 as f64, 20) - mult16_16(min_xc, min_xc),
            )),
        ) as i16;
        // Compensate for Q20 vs Q14 input and convert output to Q8
        log_xc = pshr32(i32::from(log_xc) - i32::from(qconst16(6.0, 10)), 10 - 8) as i16;
        log_xc2 = pshr32(i32::from(log_xc2) - i32::from(qconst16(6.0, 10)), 10 - 8) as i16;

        trim = (i32::from(trim)
            + max16(
                -i32::from(qconst16(4.0, 8)),
                mult16_16_q15(qconst16(0.75, 15), log_xc),
            )) as i16;
        *stereo_saving = min16(
            i32::from(*stereo_saving) + i32::from(qconst16(0.25, 8)),
            -half16(log_xc2),
        ) as i16;
    }

    // Estimate spectral tilt
    let nb = m.nb_ebands as usize;
    for ch in 0..c as usize {
        for i in 0..(end - 1) as usize {
            // C UB (signed overflow of the `opus_val32` sum): with up-sampled input and the end
            // band left above the input bandwidth (direct CELT use; the Opus encoder limits it),
            // the silent high bands have log energies near -14 and the weighted sum overflows
            // at high rates. It wraps as in C in practice (bit-exact with the oracle) instead of
            // panicking in debug builds.
            diff = diff.wrapping_add(
                shr32(band_log_e[i + ch * nb], 5).wrapping_mul(2 + 2 * i as i32 - end),
            );
        }
    }
    diff /= c * (end - 1);
    trim = (i32::from(trim)
        - max32(
            -i32::from(qconst16(2.0, 8)),
            min32(
                i32::from(qconst16(2.0, 8)),
                shr32(diff + qconst32(1.0, DB_SHIFT - 5), DB_SHIFT - 13) / 6,
            ),
        )) as i16;
    trim = (i32::from(trim) - shr16(surround_trim, DB_SHIFT - 8)) as i16;
    trim = (i32::from(trim) - 2 * shr16(tf_estimate, 14 - 8)) as i16;
    #[cfg(feature = "disable-float-api")]
    let _ = analysis;
    #[cfg(not(feature = "disable-float-api"))]
    if analysis.valid != 0 {
        // C: (opus_val16)(QCONST16(2.f, 8)*(analysis->tonality_slope+.05f)) is a float
        // product truncated to 16 bits.
        let t = (f32::from(qconst16(2.0, 8)) * (analysis.tonality_slope + 0.05f32)) as i16;
        trim = (i32::from(trim)
            - max16(
                -i32::from(qconst16(2.0, 8)),
                min16(i32::from(qconst16(2.0, 8)), i32::from(t)),
            )) as i16;
    }

    let trim_index = pshr32(trim, 8);
    imax(0, imin(10, trim_index))
}

/// Port of celt/celt_encoder.c:stereo_analysis (fixed-point build): whether dual (L/R) stereo
/// is cheaper than M/S.
#[must_use]
pub fn stereo_analysis(m: &CeltMode, x: &[CeltNorm], lm: i32, n0: i32) -> bool {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut sum_lr: OpusVal32 = EPSILON;
    let mut sum_ms: OpusVal32 = EPSILON;

    // Use the L1 norm to model the entropy of the L/R signal vs the M/S signal
    for i in 0..13 {
        for j in (eb(i) << lm) as usize..(eb(i + 1) << lm) as usize {
            // We cast to 32-bit first because of the -32768 case
            let l: OpusVal32 = shr32(x[j], NORM_SHIFT - 14);
            let r: OpusVal32 = shr32(x[n0 as usize + j], NORM_SHIFT - 14);
            let mm: OpusVal32 = add32(l, r);
            let s: OpusVal32 = sub32(l, r);
            sum_lr = add32(sum_lr, add32(abs32(l), abs32(r)));
            sum_ms = add32(sum_ms, add32(abs32(mm), abs32(s)));
        }
    }
    sum_ms = mult16_32_q15(qconst16(0.707107f32 as f64, 15), sum_ms);
    let mut thetas = 13;
    // We don't need thetas for lower bands with LM<=1
    if lm <= 1 {
        thetas -= 8;
    }
    mult16_32_q15((eb(13) << (lm + 1)) + thetas, sum_ms) > mult16_32_q15(eb(13) << (lm + 1), sum_lr)
}

/// Port of celt/celt_encoder.c:dynalloc_analysis (fixed-point build). Returns `maxDepth`;
/// writes `offsets` (`nbEBands`), `importance`, `spread_weight` and `tot_boost_`.
pub fn dynalloc_analysis(
    band_log_e: &[CeltGlog],
    band_log_e2: &[CeltGlog],
    old_band_e: &[CeltGlog],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    e_bands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost_: &mut i32,
    lfe: bool,
    surround_dynalloc: &[CeltGlog],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
    qext_scale: i32,
    scratch: &mut DynallocScratch,
) -> CeltGlog {
    let nb = nb_ebands as usize;
    let (startu, endu) = (start as usize, end as usize);
    let eb = |i: usize| i32::from(e_bands[i]);
    let mut tot_boost: i32 = 0;
    let DynallocScratch {
        follower,
        noise_floor,
        band_log_e3,
        mask,
        sig,
    } = scratch;
    offsets[..nb].fill(0);
    // Dynamic allocation code
    let mut max_depth: CeltGlog = -gconst(31.9f32 as f64);
    for i in 0..endu {
        // Noise floor must take into account eMeans, the depth, the width of the bands and the
        // preemphasis filter (approx. square of bark band ID)
        let i5 = i as i32 + 5;
        noise_floor[i] =
            gconst(0.0625) * i32::from(log_n[i]) + gconst(0.5) + shl32(9 - lsb_depth, DB_SHIFT)
                - shl32(E_MEANS[i], DB_SHIFT - 4)
                + gconst(0.0062f32 as f64) * i5 * i5;
    }
    for ch in 0..c as usize {
        for i in 0..endu {
            max_depth = maxg(max_depth, band_log_e[ch * nb + i] - noise_floor[i]);
        }
    }
    {
        // Compute a really simple masking model to avoid taking into account completely masked
        // bands when computing the spreading decision.
        for i in 0..endu {
            mask[i] = band_log_e[i] - noise_floor[i];
        }
        if c == 2 {
            for i in 0..endu {
                mask[i] = maxg(mask[i], band_log_e[nb + i] - noise_floor[i]);
            }
        }
        sig[..endu].copy_from_slice(&mask[..endu]);
        for i in 1..endu {
            mask[i] = maxg(mask[i], mask[i - 1] - gconst(2.0));
        }
        for i in (0..endu.saturating_sub(1)).rev() {
            mask[i] = maxg(mask[i], mask[i + 1] - gconst(3.0));
        }
        for i in 0..endu {
            // Compute SMR: Mask is never more than 72 dB below the peak and never below the
            // noise floor.
            let smr: CeltGlog = sig[i] - maxg(maxg(0, max_depth - gconst(12.0)), mask[i]);
            // Clamp SMR to make sure we're not shifting by something negative or too large.
            let shift = -pshr32(maxg(-gconst(5.0), ming(0, smr)), DB_SHIFT);
            spread_weight[i] = 32 >> shift;
        }
    }
    // Make sure that dynamic allocation can't make us bust the budget. We enable the feature
    // starting at 24 kb/s for 20-ms frames and 96 kb/s for 2.5 ms frames.
    if effective_bytes >= (30 + 5 * lm) && !lfe {
        let mut last: usize = 0;
        for ch in 0..c as usize {
            band_log_e3[..endu].copy_from_slice(&band_log_e2[ch * nb..ch * nb + endu]);
            if lm == 0 {
                // For 2.5 ms frames, the first 8 bands have just one bin, so the energy is
                // highly unreliable (high variance). For that reason, we take the max with the
                // previous energy so that at least 2 bins are getting used.
                for i in 0..endu.min(8) {
                    band_log_e3[i] = maxg(band_log_e2[ch * nb + i], old_band_e[ch * nb + i]);
                }
            }
            let f = &mut follower[ch * nb..ch * nb + nb];
            f[0] = band_log_e3[0];
            for i in 1..endu {
                // The last band to be at least 3 dB higher than the previous one is the last
                // we'll consider. Otherwise, we run into problems on bandlimited signals.
                if band_log_e3[i] > band_log_e3[i - 1] + gconst(0.5) {
                    last = i;
                }
                f[i] = ming(f[i - 1] + gconst(1.5), band_log_e3[i]);
            }
            for i in (0..last).rev() {
                f[i] = ming(f[i], ming(f[i + 1] + gconst(2.0), band_log_e3[i]));
            }

            // Combine with a median filter to avoid dynalloc triggering unnecessarily. The
            // "offset" value controls how conservative we are -- a higher offset reduces the
            // impact of the median filter and makes dynalloc use more bits.
            let offset: CeltGlog = gconst(1.0);
            for i in 2..endu.saturating_sub(2) {
                f[i] = maxg(f[i], super::median_of_5(&band_log_e3[i - 2..]) - offset);
            }
            let mut tmp = super::median_of_3(&band_log_e3[0..]) - offset;
            f[0] = maxg(f[0], tmp);
            f[1] = maxg(f[1], tmp);
            tmp = super::median_of_3(&band_log_e3[endu - 3..]) - offset;
            f[endu - 2] = maxg(f[endu - 2], tmp);
            f[endu - 1] = maxg(f[endu - 1], tmp);

            for i in 0..endu {
                f[i] = maxg(f[i], noise_floor[i]);
            }
        }
        if c == 2 {
            for i in startu..endu {
                // Consider 24 dB "cross-talk"
                follower[nb + i] = maxg(follower[nb + i], follower[i] - gconst(4.0));
                follower[i] = maxg(follower[i], follower[nb + i] - gconst(4.0));
                follower[i] = half32(
                    maxg(0, band_log_e[i] - follower[i])
                        + maxg(0, band_log_e[nb + i] - follower[nb + i]),
                );
            }
        } else {
            for i in startu..endu {
                follower[i] = maxg(0, band_log_e[i] - follower[i]);
            }
        }
        for i in startu..endu {
            follower[i] = maxg(follower[i], surround_dynalloc[i]);
        }
        for i in startu..endu {
            importance[i] = pshr32(13 * celt_exp2_db(ming(follower[i], gconst(4.0))), 16);
        }
        // For non-transient CBR/CVBR frames, halve the dynalloc contribution
        if (!vbr || constrained_vbr) && !is_transient {
            for i in startu..endu {
                follower[i] = half32(follower[i]);
            }
        }
        for i in startu..endu {
            if i < 8 {
                follower[i] *= 2;
            }
            if i >= 12 {
                follower[i] = half32(follower[i]);
            }
        }
        // Compensate for Opus' under-allocation on tones.
        if toneishness > qconst32(0.98f32 as f64, 29) {
            let freq_bin = pshr32(
                qext_scale * i32::from(tone_freq) * i32::from(qconst16(120.0 / C_M_PI, 9)),
                13 + 9,
            );
            for i in startu..endu {
                if freq_bin >= eb(i) && freq_bin <= eb(i + 1) {
                    follower[i] += gconst(2.0);
                }
                if freq_bin >= eb(i) - 1 && freq_bin <= eb(i + 1) + 1 {
                    follower[i] += gconst(1.0);
                }
                if freq_bin >= eb(i) - 2 && freq_bin <= eb(i + 1) + 2 {
                    follower[i] += gconst(1.0);
                }
                if freq_bin >= eb(i) - 3 && freq_bin <= eb(i + 1) + 3 {
                    follower[i] += gconst(0.5);
                }
            }
            if freq_bin >= eb(endu) {
                follower[endu - 1] += gconst(2.0);
                follower[endu - 2] += gconst(1.0);
            }
        }
        #[cfg(feature = "disable-float-api")]
        let _ = analysis;
        #[cfg(not(feature = "disable-float-api"))]
        if analysis.valid != 0 {
            for i in startu..LEAK_BANDS.min(endu) {
                follower[i] += gconst(1.0 / 64.0) * i32::from(analysis.leak_boost[i]);
            }
        }
        for i in startu..endu {
            follower[i] = ming(follower[i], gconst(4.0));
            follower[i] = shr32(follower[i], 8);

            let width = (c * (eb(i + 1) - eb(i))) << lm;
            let (boost, boost_bits) = if width < 6 {
                let boost = shr32(follower[i], DB_SHIFT - 8);
                (boost, (boost * width) << BITRES)
            } else if width > 48 {
                let boost = shr32(follower[i] * 8, DB_SHIFT - 8);
                (boost, ((boost * width) << BITRES) / 8)
            } else {
                let boost = shr32(follower[i] * width / 6, DB_SHIFT - 8);
                (boost, (boost * 6) << BITRES)
            };
            // For CBR and non-transient CVBR frames, limit dynalloc to 2/3 of the bits
            if (!vbr || (constrained_vbr && !is_transient))
                && ((tot_boost + boost_bits) >> BITRES >> 3) > 2 * effective_bytes / 3
            {
                let cap = (2 * effective_bytes / 3) << BITRES << 3;
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            }
            offsets[i] = boost;
            tot_boost += boost_bits;
        }
    } else {
        for i in startu..endu {
            importance[i] = 13;
        }
    }
    *tot_boost_ = tot_boost;
    max_depth
}

/// `M_PI` as seen by celt_encoder.c (`<math.h>` defines it; the file's fallback is only used
/// without it).
const C_M_PI: f64 = core::f64::consts::PI;

/// Port of celt/celt_encoder.c:normalize_tone_input (fixed-point build).
pub fn normalize_tone_input(x: &mut [OpusVal16]) {
    let len = x.len() as i32;
    let mut ac0: OpusVal32 = len;
    for &v in x.iter() {
        ac0 = add32(ac0, shr32(mult16_16(v, v), 10));
    }
    let shift = 5 - (28 - celt_ilog2(ac0)) / 2;
    if shift > 0 {
        for v in x.iter_mut() {
            *v = pshr32(*v, shift) as i16;
        }
    }
}

/// Port of celt/celt_encoder.c:acos_approx (fixed-point build): `acos` of a Q29 value (Q13
/// output).
#[must_use]
pub fn acos_approx(mut x: OpusVal32) -> i32 {
    let flip = x < 0;
    x = x.abs();
    let x14: OpusVal16 = (x >> 15) as i16;
    let mut tmp: OpusVal32 = ((762 * i32::from(x14)) >> 14) - 3308;
    tmp = ((tmp * i32::from(x14)) >> 14) + 25726;
    tmp = (tmp * celt_sqrt(imax(0, (1 << 30) - (x << 1)))) >> 16;
    if flip {
        tmp = 25736 - tmp;
    }
    tmp
}

/// Port of celt/celt_encoder.c:tone_lpc (fixed-point build): computes the LPC coefficients
/// (Q29) using a least-squares fit for both forward and backward prediction. Returns `true` on
/// failure (C returns 1).
pub fn tone_lpc(x: &[OpusVal16], len: i32, delay: i32, lpc: &mut [OpusVal32; 2]) -> bool {
    let (lenu, d) = (len as usize, delay as usize);
    let x = &x[..lenu];
    let mut r00: OpusVal32 = 0;
    let mut r01: OpusVal32 = 0;
    let mut r02: OpusVal32 = 0;
    debug_assert!(len > 2 * delay);
    // Compute correlations as if using the forward prediction covariance method.
    for i in 0..lenu - 2 * d {
        r00 += mult16_16(x[i], x[i]);
        r01 += mult16_16(x[i], x[i + d]);
        r02 += mult16_16(x[i], x[i + 2 * d]);
    }
    let mut edges: OpusVal32 = 0;
    for i in 0..d {
        edges += mult16_16(x[lenu + i - 2 * d], x[lenu + i - 2 * d]) - mult16_16(x[i], x[i]);
    }
    let r11 = r00 + edges;
    edges = 0;
    for i in 0..d {
        edges += mult16_16(x[lenu + i - d], x[lenu + i - d]) - mult16_16(x[i + d], x[i + d]);
    }
    let r22 = r11 + edges;
    edges = 0;
    for i in 0..d {
        edges += mult16_16(x[lenu + i - 2 * d], x[lenu + i - d]) - mult16_16(x[i], x[i + d]);
    }
    let r12 = r01 + edges;
    // Reverse and sum to get the backward contribution.
    let r00b = r00 + r22;
    let r01b = r01 + r12;
    let r11b = 2 * r11;
    let r02b = 2 * r02;
    let r12b = r12 + r01;
    let (r00, r01, r11, r02, r12) = (r00b, r01b, r11b, r02b, r12b);
    // Solve A*x=b, where A=[r00, r01; r01, r11] and b=[r02; r12].
    let den: OpusVal32 = mult32_32_q31(r00, r11) - mult32_32_q31(r01, r01);
    if den <= shr32(mult32_32_q31(r00, r11), 10) {
        return true;
    }
    let num1: OpusVal32 = mult32_32_q31(r02, r11) - mult32_32_q31(r01, r12);
    lpc[1] = if num1 >= den {
        qconst32(1.0, 29)
    } else if num1 <= -den {
        -qconst32(1.0, 29)
    } else {
        frac_div32_q29(num1, den)
    };
    let num0: OpusVal32 = mult32_32_q31(r00, r12) - mult32_32_q31(r02, r01);
    lpc[0] = if half32(num0) >= den {
        qconst32(1.999999f32 as f64, 29)
    } else if half32(num0) <= -den {
        -qconst32(1.999999f32 as f64, 29)
    } else {
        frac_div32_q29(num0, den)
    };
    false
}

/// Port of celt/celt_encoder.c:tone_detect (fixed-point build): detects pure or nearly pure
/// tones so we can prevent them from causing problems with the encoder. Returns the tone
/// frequency (Q13 radians/sample, -1 if none) and writes `toneishness` (Q29). `x` is scratch of
/// at least `n` samples.
pub fn tone_detect(
    input: &[CeltSig],
    cc: i32,
    n: i32,
    toneishness: &mut OpusVal32,
    fs: i32,
    x: &mut [OpusVal16],
) -> OpusVal16 {
    let nu = n as usize;
    let mut delay: i32 = 1;
    let mut lpc: [OpusVal32; 2] = [0; 2];
    let x = &mut x[..nu];
    // Shift by SIG_SHIFT+2 (+3 for stereo) to account for HF gain of the preemphasis filter.
    if cc == 2 {
        for i in 0..nu {
            x[i] = pshr32(
                add32(shr32(input[i], 1), shr32(input[i + nu], 1)),
                SIG_SHIFT + 2,
            ) as i16;
        }
    } else {
        for i in 0..nu {
            x[i] = pshr32(input[i], SIG_SHIFT + 2) as i16;
        }
    }
    normalize_tone_input(x);
    let mut fail = tone_lpc(x, n, delay, &mut lpc);
    // If our LPC filter resonates too close to DC, retry the analysis with down-sampling.
    while delay <= fs / 3000 && (fail || (lpc[0] > qconst32(1.0, 29) && lpc[1] < 0)) {
        delay *= 2;
        fail = tone_lpc(x, n, delay, &mut lpc);
    }
    // Check that our filter has complex roots.
    if !fail && mult32_32_q31(lpc[0], lpc[0]) + mult32_32_q31(qconst32(3.999999, 29), lpc[1]) < 0 {
        // Squared radius of the poles.
        *toneishness = -lpc[1];
        ((acos_approx(lpc[0] >> 1) + delay / 2) / delay) as i16
    } else {
        *toneishness = 0;
        -1
    }
}

/// Port of celt/celt_encoder.c:run_prefilter (fixed-point build): pitch pre-filter (comb
/// filter) of `input` (`CC*(N+overlap)`), updating `prefilter_mem`
/// (`CC*QEXT_SCALE(COMBFILTER_MAXPERIOD)`).
///
/// `pre` needs `CC*(N+max_period)` samples and `pitch_buf` `(max_period+N)>>1`. Gains are Q15,
/// `tf_estimate` Q14, `tone_freq` Q13 and `toneishness` Q29.
pub fn run_prefilter(
    mode: &CeltMode,
    st: PrefilterState<'_>,
    input: &mut [CeltSig],
    prefilter_mem: &mut [CeltSig],
    cc: i32,
    n: i32,
    prefilter_tapset: i32,
    enabled: bool,
    complexity: i32,
    tf_estimate: OpusVal16,
    nb_available_bytes: i32,
    analysis: &AnalysisInfo,
    mut tone_freq: OpusVal16,
    toneishness: OpusVal32,
    qext_scale: i32,
    pre: &mut [CeltSig],
    pitch_buf: &mut [OpusVal16],
) -> PrefilterOut {
    let PrefilterState {
        prefilter_period,
        prefilter_gain: st_gain,
        prefilter_tapset: st_tapset,
        loss_rate,
        in_mem,
    } = st;
    let mut before: [OpusVal32; 2] = [0; 2];
    let mut after: [OpusVal32; 2] = [0; 2];
    let mut cancel_pitch = false;
    let max_period = qext_scale * COMBFILTER_MAXPERIOD;
    let min_period = qext_scale * COMBFILTER_MINPERIOD;
    let overlap = mode.overlap as usize;
    let (nu, mp) = (n as usize, max_period as usize);
    let ccu = cc as usize;
    let stride = nu + mp;
    let pre = &mut pre[..ccu * stride];

    for c in 0..ccu {
        pre[c * stride..c * stride + mp].copy_from_slice(&prefilter_mem[c * mp..c * mp + mp]);
        let src = c * (nu + overlap) + overlap;
        pre[c * stride + mp..c * stride + mp + nu].copy_from_slice(&input[src..src + nu]);
    }

    let mut gain1: OpusVal16;
    let mut pitch_index: i32;
    // If we detect that the signal is dominated by a single tone, don't rely on the standard
    // pitch estimator, as it can become unreliable.
    if enabled && toneishness > qconst32(0.99f32 as f64, 29) {
        let mut multiple: i32 = 1;
        // Using aliased version of the postfilter above 24 kHz. First value is purposely
        // slightly above pi to avoid triggering for Fs=48kHz.
        if qext_scale * i32::from(tone_freq) >= i32::from(qconst16(3.1416f32 as f64, 13)) {
            tone_freq = (i32::from(qconst16(3.141593f32 as f64, 13)) - i32::from(tone_freq)) as i16;
        }
        // If the pitch is too high for our post-filter, apply pitch doubling until we can get
        // something that fits (not ideal, but better than nothing).
        while qext_scale * i32::from(tone_freq)
            >= multiple * i32::from(qconst16(0.39f32 as f64, 13))
        {
            multiple += 1;
        }
        let qtf = qext_scale * i32::from(tone_freq);
        if qtf > i32::from(qconst16(0.006148f32 as f64, 13)) {
            pitch_index = imin((51472 * multiple + qtf / 2) / qtf, COMBFILTER_MAXPERIOD - 2);
        } else {
            // If the pitch is too low, using a very high pitch will actually give us an
            // improvement due to the DC component of the filter that will be close to our
            // tone. Again, not ideal, but if we only have a single tone, it's better than
            // nothing.
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = qconst16(0.75, 15);
    } else if enabled && complexity >= 5 {
        let half = ((max_period + n) >> 1) as usize;
        let pitch_buf = &mut pitch_buf[..half];
        {
            let (p0, p1) = pre.split_at(stride);
            let chans: [&[CeltSig]; 2] = [p0, if ccu == 2 { p1 } else { p0 }];
            pitch_downsample(&chans[..ccu], pitch_buf, half, ccu, 2);
        }
        // Don't search for the fir last 1.5 octave of the range because there's too many
        // false-positives due to short-term correlation
        pitch_index = pitch_search(
            &pitch_buf[(max_period >> 1) as usize..],
            pitch_buf,
            nu,
            (max_period - 3 * min_period) as usize,
        );
        pitch_index = max_period - pitch_index;

        gain1 = remove_doubling(
            pitch_buf,
            max_period,
            min_period,
            n,
            &mut pitch_index,
            *prefilter_period,
            st_gain,
        );
        if pitch_index > max_period - 2 * qext_scale {
            pitch_index = max_period - 2 * qext_scale;
        }
        #[cfg(feature = "qext")]
        {
            pitch_index /= qext_scale;
        }
        gain1 = mult16_16_q15(qconst16(0.7f32 as f64, 15), gain1) as i16;
        if loss_rate > 2 {
            gain1 = half32(gain1) as i16;
        }
        if loss_rate > 4 {
            gain1 = half32(gain1) as i16;
        }
        if loss_rate > 8 {
            gain1 = 0;
        }
    } else {
        gain1 = 0;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    #[cfg(feature = "disable-float-api")]
    let _ = analysis;
    #[cfg(not(feature = "disable-float-api"))]
    if analysis.valid != 0 {
        // C: (opus_val16)(gain1 * analysis->max_pitch_ratio), a float product truncated.
        gain1 = (f32::from(gain1) * analysis.max_pitch_ratio) as i16;
    }
    // Gain threshold for enabling the prefilter/postfilter
    let mut pf_threshold: OpusVal16 = qconst16(0.2f32 as f64, 15);

    // Adjusting the threshold based on rate and continuity
    if (pitch_index - *prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += qconst16(0.2f32 as f64, 15);
        // Completely disable the prefilter on strong transients without continuity.
        if tf_estimate > qconst16(0.98f32 as f64, 14) {
            gain1 = 0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += qconst16(0.1f32 as f64, 15);
    }
    if nb_available_bytes < 35 {
        pf_threshold += qconst16(0.1f32 as f64, 15);
    }
    if st_gain > qconst16(0.4f32 as f64, 15) {
        pf_threshold -= qconst16(0.1f32 as f64, 15);
    }
    if st_gain > qconst16(0.55f32 as f64, 15) {
        pf_threshold -= qconst16(0.1f32 as f64, 15);
    }

    // Hard threshold at 0.2
    pf_threshold = max16(pf_threshold, qconst16(0.2f32 as f64, 15));
    let mut pf_on: bool;
    let mut qg: i32;
    if gain1 < pf_threshold {
        gain1 = 0;
        pf_on = false;
        qg = 0;
    } else {
        // This block is not gated by a total bits check only because of the nbAvailableBytes
        // check above.
        if abs16(i32::from(gain1) - i32::from(st_gain)) < i32::from(qconst16(0.1f32 as f64, 15)) {
            gain1 = st_gain;
        }
        qg = ((i32::from(gain1) + 1536) >> 10) / 3 - 1;
        qg = imax(0, imin(7, qg));
        gain1 = (i32::from(qconst16(0.09375, 15)) * (qg + 1)) as i16;
        pf_on = true;
    }

    let offset = mode.short_mdct_size - mode.overlap;
    let window = &mode.window[..];
    for c in 0..ccu {
        let base = c * (nu + overlap);
        *prefilter_period = imax(*prefilter_period, COMBFILTER_MINPERIOD);
        input[base..base + overlap].copy_from_slice(&in_mem[c * overlap..c * overlap + overlap]);
        for i in 0..nu {
            before[c] += abs32(shr32(input[base + overlap + i], 12));
        }
        let pre_c = &pre[c * stride..c * stride + stride];
        if offset != 0 {
            comb_filter(
                &mut input[base + overlap..],
                pre_c,
                mp,
                *prefilter_period,
                *prefilter_period,
                offset,
                -st_gain,
                -st_gain,
                st_tapset,
                st_tapset,
                &[],
                0,
            );
        }

        comb_filter(
            &mut input[base + overlap + offset as usize..],
            pre_c,
            mp + offset as usize,
            *prefilter_period,
            pitch_index,
            n - offset,
            -st_gain,
            -gain1,
            st_tapset,
            prefilter_tapset,
            window,
            mode.overlap,
        );
        for i in 0..nu {
            after[c] += abs32(shr32(input[base + overlap + i], 12));
        }
    }

    if cc == 2 {
        // C stores the thresholds in `opus_val16` variables (truncating).
        let thresh: [OpusVal16; 2] = [
            (mult16_32_q15(mult16_16_q15(qconst16(0.25, 15), gain1), before[0])
                + mult16_32_q15(qconst16(0.01f32 as f64, 15), before[1])) as i16,
            (mult16_32_q15(mult16_16_q15(qconst16(0.25, 15), gain1), before[1])
                + mult16_32_q15(qconst16(0.01f32 as f64, 15), before[0])) as i16,
        ];
        let th = [i32::from(thresh[0]), i32::from(thresh[1])];
        // Don't use the filter if one channel gets significantly worse.
        if after[0] - before[0] > th[0] || after[1] - before[1] > th[1] {
            cancel_pitch = true;
        }
        // Use the filter only if at least one channel gets significantly better.
        if before[0] - after[0] < th[0] && before[1] - after[1] < th[1] {
            cancel_pitch = true;
        }
    } else {
        // Check that the mono channel actually got better.
        if after[0] > before[0] {
            cancel_pitch = true;
        }
    }
    // If needed, revert to a gain of zero.
    if cancel_pitch {
        for c in 0..ccu {
            let base = c * (nu + overlap);
            let pre_c = &pre[c * stride..c * stride + stride];
            input[base + overlap..base + overlap + nu].copy_from_slice(&pre_c[mp..mp + nu]);
            comb_filter(
                &mut input[base + overlap + offset as usize..],
                pre_c,
                mp + offset as usize,
                *prefilter_period,
                pitch_index,
                mode.overlap,
                -st_gain,
                0,
                st_tapset,
                prefilter_tapset,
                window,
                mode.overlap,
            );
        }
        gain1 = 0;
        pf_on = false;
        qg = 0;
    }

    for c in 0..ccu {
        let base = c * (nu + overlap);
        in_mem[c * overlap..c * overlap + overlap]
            .copy_from_slice(&input[base + nu..base + nu + overlap]);
        let pre_c = &pre[c * stride..c * stride + stride];
        let pm = &mut prefilter_mem[c * mp..c * mp + mp];
        if nu > mp {
            pm.copy_from_slice(&pre_c[nu..nu + mp]);
        } else {
            pm.copy_within(nu..mp, 0);
            pm[mp - nu..].copy_from_slice(&pre_c[mp..mp + nu]);
        }
    }

    PrefilterOut {
        pitch: pitch_index,
        gain: gain1,
        qgain: qg,
        pf_on,
    }
}

/// Port of celt/celt_encoder.c:compute_vbr (fixed-point build): the VBR target (in 1/8 bits)
/// for this frame. `stereo_saving` is Q8, `tf_estimate` Q14 and the log energies Q24.
#[must_use]
pub fn compute_vbr(
    mode: &CeltMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: i32,
    intensity: i32,
    constrained_vbr: bool,
    mut stereo_saving: OpusVal16,
    tot_boost: i32,
    tf_estimate: OpusVal16,
    pitch_change: bool,
    max_depth: CeltGlog,
    lfe: bool,
    has_surround_mask: bool,
    surround_masking: CeltGlog,
    temporal_vbr: CeltGlog,
    #[cfg(feature = "qext")] enable_qext: bool,
) -> i32 {
    let nb_ebands = mode.nb_ebands;
    let e_bands = &mode.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);

    let coded_bands = if last_coded_bands != 0 {
        last_coded_bands
    } else {
        nb_ebands
    };
    let mut coded_bins = eb(coded_bands) << lm;
    if c == 2 {
        coded_bins += eb(imin(intensity, coded_bands)) << lm;
    }

    let mut target = base_target;

    #[cfg(not(feature = "disable-float-api"))]
    if analysis.valid != 0 && f64::from(analysis.activity) < 0.4 {
        target -= ((coded_bins << BITRES) as f32 * (0.4f32 - analysis.activity)) as i32;
    }
    // Stereo savings
    if c == 2 {
        let coded_stereo_bands = imin(intensity, coded_bands);
        let coded_stereo_dof = (eb(coded_stereo_bands) << lm) - coded_stereo_bands;
        // Maximum fraction of the bits we can save if the signal is mono.
        let max_frac: OpusVal16 = div32_16(
            mult16_16(qconst16(0.8f32 as f64, 15), coded_stereo_dof),
            coded_bins,
        );
        stereo_saving = min16(stereo_saving, qconst16(1.0, 8));
        target -= min32(
            mult16_32_q15(max_frac, target),
            shr32(
                mult16_16(
                    i32::from(stereo_saving) - i32::from(qconst16(0.1f32 as f64, 8)),
                    coded_stereo_dof << BITRES,
                ),
                8,
            ),
        );
    }
    // Boost the rate according to dynalloc (minus the dynalloc average for calibration).
    target += tot_boost - (19 << lm);
    // Apply transient boost, compensating for average boost.
    let tf_calibration: OpusVal16 = qconst16(0.044f32 as f64, 14);
    target += shl32(
        mult16_32_q15(i32::from(tf_estimate) - i32::from(tf_calibration), target),
        1,
    );

    // Apply tonality boost
    #[cfg(not(feature = "disable-float-api"))]
    if analysis.valid != 0 && !lfe {
        // Tonality boost (compensating for the average).
        let tonal: f32 = max16(0.0f32, analysis.tonality - 0.15f32) - 0.12f32;
        let mut tonal_target = target + ((coded_bins << BITRES) as f32 * 1.2f32 * tonal) as i32;
        if pitch_change {
            tonal_target += ((coded_bins << BITRES) as f32 * 0.8f32) as i32;
        }
        target = tonal_target;
    }
    #[cfg(feature = "disable-float-api")]
    let _ = (analysis, pitch_change);

    if has_surround_mask && !lfe {
        let surround_target = target
            + shr32(
                mult16_16(shr32(surround_masking, DB_SHIFT - 10), coded_bins << BITRES),
                10,
            );
        target = imax(target / 4, surround_target);
    }

    {
        #[allow(unused_mut, reason = "only reassigned with qext")]
        let mut bins = eb(nb_ebands - 2) << lm;
        #[cfg(feature = "qext")]
        if enable_qext {
            bins = mode.short_mdct_size << lm;
        }
        let mut floor_depth = shr32(
            mult16_32_q15((c * bins) << BITRES, max_depth),
            DB_SHIFT - 15,
        );
        floor_depth = imax(floor_depth, target >> 2);
        target = imin(target, floor_depth);
    }

    // Make VBR less aggressive for constrained VBR because we can't keep a higher bitrate for
    // long. Needs tuning.
    if (!has_surround_mask || lfe) && constrained_vbr {
        target = base_target + mult16_32_q15(qconst16(0.67f32 as f64, 15), target - base_target);
    }

    if !has_surround_mask && tf_estimate < qconst16(0.2f32 as f64, 14) {
        let amount: OpusVal16 = mult16_16_q15(
            qconst16(0.0000031f32 as f64, 30),
            imax(0, imin(32000, 96000 - bitrate)),
        ) as i16;
        let tvbr_factor: OpusVal16 =
            shr32(mult16_16(shr32(temporal_vbr, DB_SHIFT - 10), amount), 10) as i16;
        target += mult16_32_q15(tvbr_factor, target);
    }

    // Don't allow more than doubling the rate
    imin(2 * base_target, target)
}

/// The surround masking analysis of celt/celt_encoder.c:celt_encode_with_ec (fixed-point
/// build). Writes `surround_dynalloc[..mask_end]`; returns `(surround_masking,
/// surround_trim)`.
pub(super) fn surround_masking_analysis(
    energy_mask: &[CeltGlog],
    e_bands: &[i16],
    nb_ebands: i32,
    c: i32,
    last_coded_bands: i32,
    surround_dynalloc: &mut [CeltGlog],
) -> (CeltGlog, CeltGlog) {
    let nb = nb_ebands as usize;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut mask_avg: OpusVal32 = 0;
    let mut diff: OpusVal32 = 0;
    let mut count: i32 = 0;
    let mask_end = imax(2, last_coded_bands);
    for ch in 0..c as usize {
        for i in 0..mask_end {
            let mut mask: CeltGlog = maxg(
                ming(energy_mask[nb * ch + i as usize], gconst(0.25)),
                -gconst(2.0),
            );
            if mask > 0 {
                mask = half32(mask);
            }
            let mask16: OpusVal16 = shr32(mask, DB_SHIFT - 10) as i16;
            mask_avg += mult16_16(mask16, eb(i + 1) - eb(i));
            count += eb(i + 1) - eb(i);
            diff += mult16_16(mask16, 1 + 2 * i - mask_end);
        }
    }
    debug_assert!(count > 0);
    mask_avg = shl32(div32_16(mask_avg, count), DB_SHIFT - 10);
    mask_avg += gconst(0.2f32 as f64);
    diff = shl32(
        diff * 6 / (c * (mask_end - 1) * (mask_end + 1) * mask_end),
        DB_SHIFT - 10,
    );
    // Again, being conservative
    diff = half32(diff);
    diff = max32(
        min32(diff, gconst(0.031f32 as f64)),
        -gconst(0.031f32 as f64),
    );
    // Find the band that's in the middle of the coded spectrum
    let mut midband: i32 = 0;
    while eb(midband + 1) < eb(mask_end) / 2 {
        midband += 1;
    }
    let mut count_dynalloc = 0;
    for i in 0..mask_end {
        let iu = i as usize;
        let lin: OpusVal32 = mask_avg + diff * (i - midband);
        let mut unmask: CeltGlog = if c == 2 {
            maxg(energy_mask[iu], energy_mask[nb + iu])
        } else {
            energy_mask[iu]
        };
        unmask = ming(unmask, gconst(0.0));
        unmask -= lin;
        if unmask > gconst(0.25) {
            surround_dynalloc[iu] = unmask - gconst(0.25);
            count_dynalloc += 1;
        }
    }
    if count_dynalloc >= 3 {
        // If we need dynalloc in many bands, it's probably because our initial masking rate
        // was too low.
        mask_avg += gconst(0.25);
        if mask_avg > 0 {
            // Something went really wrong in the original calculations, disabling masking.
            mask_avg = 0;
            diff = 0;
            surround_dynalloc[..mask_end as usize].fill(0);
        } else {
            for v in &mut surround_dynalloc[..mask_end as usize] {
                *v = maxg(0, *v - gconst(0.25));
            }
        }
    }
    mask_avg += gconst(0.2f32 as f64);
    // Convert to 1/64th units used for the trim
    (mask_avg, 64 * diff)
}

/// The temporal VBR analysis of celt/celt_encoder.c:celt_encode_with_ec (fixed-point build).
/// Returns `temporal_vbr`, updating `spec_avg`.
pub(super) fn temporal_vbr_analysis(
    band_log_e: &[CeltGlog],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
    short_blocks: bool,
    lm: i32,
    spec_avg: &mut CeltGlog,
) -> CeltGlog {
    let nb = nb_ebands as usize;
    let mut follow: CeltGlog = -qconst32(10.0, DB_SHIFT - 5);
    let mut frame_avg: OpusVal32 = 0;
    let offset: CeltGlog = if short_blocks {
        half32(shl32(lm, DB_SHIFT - 5))
    } else {
        0
    };
    for i in start as usize..end as usize {
        follow = maxg(
            follow - qconst32(1.0, DB_SHIFT - 5),
            shr32(band_log_e[i], 5) - offset,
        );
        if c == 2 {
            follow = maxg(follow, shr32(band_log_e[i + nb], 5) - offset);
        }
        frame_avg += follow;
    }
    frame_avg /= end - start;
    let mut temporal_vbr = sub32(shl32(frame_avg, 5), *spec_avg);
    temporal_vbr = ming(gconst(3.0), maxg(-gconst(1.5), temporal_vbr));
    *spec_avg += mult16_32_q15(qconst16(0.02f32 as f64, 15), temporal_vbr);
    temporal_vbr
}

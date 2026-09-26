//! Port of the leaf DSP files of `silk/fixed/`: `apply_sine_window_FIX.c`, `autocorr_FIX.c`,
//! `burg_modified_FIX.c`, `k2a_FIX.c`, `k2a_Q16_FIX.c`, `schur_FIX.c`, `schur64_FIX.c`,
//! `vector_ops_FIX.c`, `corrMatrix_FIX.c`, `residual_energy_FIX.c`, `residual_energy16_FIX.c`,
//! `regularize_correlations_FIX.c`, `warped_autocorrelation_FIX.c` and
//! `LTP_analysis_filter_FIX.c`.
//!
//! `silk_inner_prod16_c` (also in `vector_ops_FIX.c`) is shared with the float build and lives
//! in [`crate::silk::sigproc::silk_inner_prod16`]. The ARM/MIPS/x86 overrides are not ported.

use crate::celt::celt_lpc::_celt_autocorr;
use crate::celt::pitch::{celt_inner_prod, celt_pitch_xcorr};
use crate::silk::define::{
    LTP_ORDER, MAX_LPC_ORDER, MAX_MATRIX_SIZE, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER,
};
use crate::silk::macros::{
    SILK_MAX_ORDER_LPC, silk_abs, silk_abs_int32, silk_add_lshift32, silk_add_rshift32, silk_add32,
    silk_clz32, silk_clz64, silk_div32, silk_div32_16, silk_div32_varq, silk_fix_const, silk_limit,
    silk_lshift, silk_lshift32, silk_lshift64, silk_max_32, silk_max_int, silk_min, silk_min_int,
    silk_mla, silk_mla_ovflw, silk_rshift, silk_rshift_round, silk_rshift32, silk_rshift64,
    silk_sat16, silk_smlabb, silk_smlabb_ovflw, silk_smlawb, silk_smlaww, silk_smmul, silk_smulbb,
    silk_smull, silk_smulwb, silk_sqrt_approx, silk_sub32,
};
use crate::silk::sigproc::{silk_inner_prod16, silk_lpc_analysis_filter, silk_sum_sqr_shift};
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const LTPO: usize = LTP_ORDER as usize;

// ---------------------------------------------------------------------------------------------
// vector_ops_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/vector_ops_FIX.c:silk_scale_copy_vector16` — copy and multiply a vector
/// by a constant (`gain_q16`).
pub fn silk_scale_copy_vector16(data_out: &mut [i16], data_in: &[i16], gain_q16: i32, n: usize) {
    for (o, &i) in data_out[..n].iter_mut().zip(&data_in[..n]) {
        let tmp32 = silk_smulwb(gain_q16, i as i32);
        // silk_CHECK_FIT16 is a no-op outside FIXED_DEBUG: plain narrowing
        *o = tmp32 as i16;
    }
}

/// Port of `silk/fixed/vector_ops_FIX.c:silk_scale_vector32_Q26_lshift_18` — multiply a vector
/// by a constant (Q0 in, Q18 out).
pub fn silk_scale_vector32_q26_lshift_18(data1: &mut [i32], gain_q26: i32, n: usize) {
    for d in &mut data1[..n] {
        *d = silk_rshift64(silk_smull(*d, gain_q26), 8) as i32; // OUTPUT: Q18
    }
}

/// Port of `silk/fixed/vector_ops_FIX.c:silk_inner_prod_aligned` (`FIXED_POINT` branch:
/// `celt_inner_prod`).
#[must_use]
#[inline]
pub fn silk_inner_prod_aligned(in_vec1: &[i16], in_vec2: &[i16], len: usize) -> i32 {
    celt_inner_prod(in_vec1, in_vec2, len)
}

// ---------------------------------------------------------------------------------------------
// apply_sine_window_FIX.c
// ---------------------------------------------------------------------------------------------

/// `freq_table_Q16` (apply_sine_window_FIX.c).
const FREQ_TABLE_Q16: [i16; 27] = [
    12111, 9804, 8235, 7100, 6239, 5565, 5022, 4575, 4202, 3885, 3612, 3375, 3167, 2984, 2820,
    2674, 2542, 2422, 2313, 2214, 2123, 2038, 1961, 1889, 1822, 1760, 1702,
];

/// Port of `silk/fixed/apply_sine_window_FIX.c:silk_apply_sine_window` — apply a sine window
/// (type 1: 0 to pi/2, type 2: pi/2 to pi) of `length` samples (16..=120, multiple of 4).
pub fn silk_apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
    celt_assert!(win_type == 1 || win_type == 2);
    // Length must be in a range from 16 to 120 and a multiple of 4
    celt_assert!((16..=120).contains(&length));
    celt_assert!(length & 3 == 0);
    let px_win = &mut px_win[..length];
    let px = &px[..length];

    // Frequency
    let k = (length >> 2) - 4;
    celt_assert!(k <= 26);
    let f_q16 = FREQ_TABLE_Q16[k] as i32;

    // Factor used for cosine approximation
    let c_q16 = silk_smulwb(f_q16, -f_q16);
    silk_assert!(c_q16 >= -32768);

    let length_i = length as i32;
    let (mut s0_q16, mut s1_q16);
    // initialize state
    if win_type == 1 {
        // start from 0
        s0_q16 = 0;
        // approximation of sin(f)
        s1_q16 = f_q16 + silk_rshift(length_i, 3);
    } else {
        // start from 1
        s0_q16 = 1i32 << 16;
        // approximation of cos(f)
        s1_q16 = (1i32 << 16) + silk_rshift(c_q16, 1) + silk_rshift(length_i, 4);
    }

    // Uses the recursive equation: sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)
    // 4 samples at a time
    let mut k = 0;
    while k < length {
        px_win[k] = silk_smulwb(silk_rshift(s0_q16 + s1_q16, 1), px[k] as i32) as i16;
        px_win[k + 1] = silk_smulwb(s1_q16, px[k + 1] as i32) as i16;
        s0_q16 = silk_smulwb(s1_q16, c_q16) + silk_lshift(s1_q16, 1) - s0_q16 + 1;
        s0_q16 = silk_min(s0_q16, 1i32 << 16);

        px_win[k + 2] = silk_smulwb(silk_rshift(s0_q16 + s1_q16, 1), px[k + 2] as i32) as i16;
        px_win[k + 3] = silk_smulwb(s0_q16, px[k + 3] as i32) as i16;
        s1_q16 = silk_smulwb(s0_q16, c_q16) + silk_lshift(s0_q16, 1) - s1_q16;
        s1_q16 = silk_min(s1_q16, 1i32 << 16);
        k += 4;
    }
}

// ---------------------------------------------------------------------------------------------
// autocorr_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/autocorr_FIX.c:silk_autocorr` — compute autocorrelation
/// (`results[..min(input_data_size, correlation_count)]`). Returns the C `*scale` output (the
/// scaling of the correlation vector).
pub fn silk_autocorr(
    results: &mut [i32],
    input_data: &[i16],
    input_data_size: usize,
    correlation_count: usize,
) -> i32 {
    let corr_count = input_data_size.min(correlation_count);
    _celt_autocorr(input_data, results, &[], 0, corr_count - 1, input_data_size)
}

// ---------------------------------------------------------------------------------------------
// burg_modified_FIX.c
// ---------------------------------------------------------------------------------------------

/// `MAX_FRAME_SIZE` (burg_modified_FIX.c): `subfr_length * nb_subfr = (0.005 * 16000 + 16) * 4`.
const BURG_MAX_FRAME_SIZE: usize = 384;
/// `QA` (burg_modified_FIX.c).
const QA: i32 = 25;
/// `N_BITS_HEAD_ROOM` (burg_modified_FIX.c).
const N_BITS_HEAD_ROOM: i32 = 3;
/// `MIN_RSHIFTS` (burg_modified_FIX.c).
const MIN_RSHIFTS: i32 = -16;
/// `MAX_RSHIFTS` (burg_modified_FIX.c).
const MAX_RSHIFTS: i32 = 32 - QA;

/// Port of `silk/fixed/burg_modified_FIX.c:silk_burg_modified_c` — compute reflection
/// coefficients from the input signal.
///
/// `x` holds `nb_subfr` subframes of `subfr_length` samples (each including `d` preceding
/// samples); writes `a_q16[..d]`. Returns `(res_nrg, res_nrg_Q)` (the C out-parameters).
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_burg_modified(
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> (i32, i32) {
    const SMO: usize = SILK_MAX_ORDER_LPC;
    let mut c_first_row = [0i32; SMO];
    let mut c_last_row = [0i32; SMO];
    let mut af_qa = [0i32; SMO];
    let mut caf = [0i32; SMO + 1];
    let mut cab = [0i32; SMO + 1];
    let mut xcorr = [0i32; SMO];

    celt_assert!(subfr_length * nb_subfr <= BURG_MAX_FRAME_SIZE);
    celt_assert!(d <= SMO);
    let x = &x[..subfr_length * nb_subfr];
    let cond_fac_q32 = silk_fix_const(FIND_LPC_COND_FAC as f64, 32);

    // Compute autocorrelations, added over subframes
    let c0_64 = silk_inner_prod16(x, x, subfr_length * nb_subfr);
    let lz = silk_clz64(c0_64);
    let mut rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    rshifts = rshifts.clamp(MIN_RSHIFTS, MAX_RSHIFTS);

    let mut c0 = if rshifts > 0 {
        silk_rshift64(c0_64, rshifts) as i32
    } else {
        silk_lshift32(c0_64 as i32, -rshifts)
    };

    let init = c0 + silk_smmul(cond_fac_q32, c0) + 1; // Q(-rshifts)
    caf[0] = init;
    cab[0] = init;
    if rshifts > 0 {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
            for n in 1..d + 1 {
                c_first_row[n - 1] += silk_rshift64(
                    silk_inner_prod16(x_ptr, &x_ptr[n..], subfr_length - n),
                    rshifts,
                ) as i32;
            }
        }
    } else {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            celt_pitch_xcorr(x_ptr, &x_ptr[1..], &mut xcorr, subfr_length - d, d);
            for n in 1..d + 1 {
                let mut dd: i32 = 0;
                for i in n + subfr_length - d..subfr_length {
                    dd += x_ptr[i] as i32 * x_ptr[i - n] as i32;
                }
                xcorr[n - 1] += dd;
            }
            for n in 1..d + 1 {
                c_first_row[n - 1] += silk_lshift32(xcorr[n - 1], -rshifts);
            }
        }
    }
    c_last_row.copy_from_slice(&c_first_row);

    // Initialize
    caf[0] = init;
    cab[0] = init;

    let mut inv_gain_q30: i32 = 1 << 30;
    let mut reached_max_gain = false;
    for n in 0..d {
        // Update first row of correlation matrix (without first element)
        // Update last row of correlation matrix (without last element, stored in reversed
        // order)
        // Update C * Af
        // Update C * flipud(Af) (stored in reversed order)
        if rshifts > -2 {
            for s in 0..nb_subfr {
                let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
                let x1 = -silk_lshift32(x_ptr[n] as i32, 16 - rshifts); // Q(16-rshifts)
                let x2 = -silk_lshift32(x_ptr[subfr_length - n - 1] as i32, 16 - rshifts); // Q(16-rshifts)
                let mut tmp1 = silk_lshift32(x_ptr[n] as i32, QA - 16); // Q(QA-16)
                let mut tmp2 = silk_lshift32(x_ptr[subfr_length - n - 1] as i32, QA - 16); // Q(QA-16)
                for k in 0..n {
                    c_first_row[k] = silk_smlawb(c_first_row[k], x1, x_ptr[n - k - 1] as i32); // Q( -rshifts )
                    c_last_row[k] =
                        silk_smlawb(c_last_row[k], x2, x_ptr[subfr_length - n + k] as i32); // Q( -rshifts )
                    let atmp_qa = af_qa[k];
                    tmp1 = silk_smlawb(tmp1, atmp_qa, x_ptr[n - k - 1] as i32); // Q(QA-16)
                    tmp2 = silk_smlawb(tmp2, atmp_qa, x_ptr[subfr_length - n + k] as i32); // Q(QA-16)
                }
                tmp1 = silk_lshift32(-tmp1, 32 - QA - rshifts); // Q(16-rshifts)
                tmp2 = silk_lshift32(-tmp2, 32 - QA - rshifts); // Q(16-rshifts)
                for k in 0..=n {
                    caf[k] = silk_smlawb(caf[k], tmp1, x_ptr[n - k] as i32); // Q( -rshift )
                    cab[k] = silk_smlawb(cab[k], tmp2, x_ptr[subfr_length - n + k - 1] as i32); // Q( -rshift )
                }
            }
        } else {
            for s in 0..nb_subfr {
                let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
                let x1 = -silk_lshift32(x_ptr[n] as i32, -rshifts); // Q( -rshifts )
                let x2 = -silk_lshift32(x_ptr[subfr_length - n - 1] as i32, -rshifts); // Q( -rshifts )
                let mut tmp1 = silk_lshift32(x_ptr[n] as i32, 17); // Q17
                let mut tmp2 = silk_lshift32(x_ptr[subfr_length - n - 1] as i32, 17); // Q17
                for k in 0..n {
                    c_first_row[k] = silk_mla(c_first_row[k], x1, x_ptr[n - k - 1] as i32); // Q( -rshifts )
                    c_last_row[k] = silk_mla(c_last_row[k], x2, x_ptr[subfr_length - n + k] as i32); // Q( -rshifts )
                    let atmp1 = silk_rshift_round(af_qa[k], QA - 17); // Q17
                    // We sometimes get overflows in the multiplications (even beyond +/- 2^32),
                    // but they cancel each other and the real result seems to always fit in a
                    // 32-bit signed integer. This was determined experimentally, not
                    // theoretically (unfortunately).
                    tmp1 = silk_mla_ovflw(tmp1, x_ptr[n - k - 1] as i32, atmp1); // Q17
                    tmp2 = silk_mla_ovflw(tmp2, x_ptr[subfr_length - n + k] as i32, atmp1); // Q17
                }
                tmp1 = -tmp1; // Q17
                tmp2 = -tmp2; // Q17
                for k in 0..=n {
                    caf[k] = silk_smlaww(
                        caf[k],
                        tmp1,
                        silk_lshift32(x_ptr[n - k] as i32, -rshifts - 1),
                    ); // Q( -rshift )
                    cab[k] = silk_smlaww(
                        cab[k],
                        tmp2,
                        silk_lshift32(x_ptr[subfr_length - n + k - 1] as i32, -rshifts - 1),
                    ); // Q( -rshift )
                }
            }
        }

        // Calculate nominator and denominator for the next order reflection (parcor)
        // coefficient
        let mut tmp1 = c_first_row[n]; // Q( -rshifts )
        let mut tmp2 = c_last_row[n]; // Q( -rshifts )
        let mut num: i32 = 0; // Q( -rshifts )
        let mut nrg = silk_add32(cab[0], caf[0]); // Q( 1-rshifts )
        for k in 0..n {
            let atmp_qa = af_qa[k];
            let mut lz = silk_clz32(silk_abs(atmp_qa)) - 1;
            lz = silk_min(32 - QA, lz);
            let atmp1 = silk_lshift32(atmp_qa, lz); // Q( QA + lz )

            tmp1 = silk_add_lshift32(tmp1, silk_smmul(c_last_row[n - k - 1], atmp1), 32 - QA - lz); // Q( -rshifts )
            tmp2 = silk_add_lshift32(
                tmp2,
                silk_smmul(c_first_row[n - k - 1], atmp1),
                32 - QA - lz,
            ); // Q( -rshifts )
            num = silk_add_lshift32(num, silk_smmul(cab[n - k], atmp1), 32 - QA - lz); // Q( -rshifts )
            nrg = silk_add_lshift32(
                nrg,
                silk_smmul(silk_add32(cab[k + 1], caf[k + 1]), atmp1),
                32 - QA - lz,
            ); // Q( 1-rshifts )
        }
        caf[n + 1] = tmp1; // Q( -rshifts )
        cab[n + 1] = tmp2; // Q( -rshifts )
        num = silk_add32(num, tmp2); // Q( -rshifts )
        num = silk_lshift32(-num, 1); // Q( 1-rshifts )

        // Calculate the next order reflection (parcor) coefficient
        let mut rc_q31 = if silk_abs(num) < nrg {
            silk_div32_varq(num, nrg, 31)
        } else if num > 0 {
            i32::MAX
        } else {
            i32::MIN
        };

        // Update inverse prediction gain
        let mut tmp1 = (1i32 << 30) - silk_smmul(rc_q31, rc_q31);
        tmp1 = silk_lshift(silk_smmul(inv_gain_q30, tmp1), 2);
        if tmp1 <= min_inv_gain_q30 {
            // Max prediction gain exceeded; set reflection coefficient such that max prediction
            // gain is exactly hit
            let tmp2 = (1i32 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30); // Q30
            rc_q31 = silk_sqrt_approx(tmp2); // Q15
            if rc_q31 > 0 {
                // Newton-Raphson iteration
                rc_q31 = silk_rshift32(rc_q31 + silk_div32(tmp2, rc_q31), 1); // Q15
                rc_q31 = silk_lshift32(rc_q31, 16); // Q31
                if num < 0 {
                    // Ensure adjusted reflection coefficients has the original sign
                    rc_q31 = -rc_q31;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
        } else {
            inv_gain_q30 = tmp1;
        }

        // Update the AR coefficients
        for k in 0..(n + 1) >> 1 {
            let tmp1 = af_qa[k]; // QA
            let tmp2 = af_qa[n - k - 1]; // QA
            af_qa[k] = silk_add_lshift32(tmp1, silk_smmul(tmp2, rc_q31), 1); // QA
            af_qa[n - k - 1] = silk_add_lshift32(tmp2, silk_smmul(tmp1, rc_q31), 1); // QA
        }
        af_qa[n] = silk_rshift32(rc_q31, 31 - QA); // QA

        if reached_max_gain {
            // Reached max prediction gain; set remaining coefficients to zero and exit loop
            for a in &mut af_qa[n + 1..d] {
                *a = 0;
            }
            break;
        }

        // Update C * Af and C * Ab
        for k in 0..=n + 1 {
            let tmp1 = caf[k]; // Q( -rshifts )
            let tmp2 = cab[n + 1 - k]; // Q( -rshifts )
            caf[k] = silk_add_lshift32(tmp1, silk_smmul(tmp2, rc_q31), 1); // Q( -rshifts )
            cab[n + 1 - k] = silk_add_lshift32(tmp2, silk_smmul(tmp1, rc_q31), 1); // Q( -rshifts )
        }
    }

    if reached_max_gain {
        for k in 0..d {
            // Scale coefficients
            a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
        }
        // Subtract energy of preceding samples from C0
        if rshifts > 0 {
            for s in 0..nb_subfr {
                let x_ptr = &x[s * subfr_length..];
                c0 -= silk_rshift64(silk_inner_prod16(x_ptr, x_ptr, d), rshifts) as i32;
            }
        } else {
            for s in 0..nb_subfr {
                let x_ptr = &x[s * subfr_length..];
                c0 -= silk_lshift32(silk_inner_prod_aligned(x_ptr, x_ptr, d), -rshifts);
            }
        }
        // Approximate residual energy
        (silk_lshift(silk_smmul(inv_gain_q30, c0), 2), -rshifts)
    } else {
        // Return residual energy
        let mut nrg = caf[0]; // Q( -rshifts )
        let mut tmp1: i32 = 1 << 16; // Q16
        for k in 0..d {
            let atmp1 = silk_rshift_round(af_qa[k], QA - 16); // Q16
            nrg = silk_smlaww(nrg, caf[k + 1], atmp1); // Q( -rshifts )
            tmp1 = silk_smlaww(tmp1, atmp1, atmp1); // Q16
            a_q16[k] = -atmp1;
        }
        (
            silk_smlaww(nrg, silk_smmul(cond_fac_q32, c0), -tmp1), // Q( -rshifts )
            -rshifts,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// k2a_FIX.c / k2a_Q16_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/k2a_FIX.c:silk_k2a` — step up function, converts reflection
/// coefficients (Q15) to prediction coefficients (Q24).
pub fn silk_k2a(a_q24: &mut [i32], rc_q15: &[i16], order: usize) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, silk_lshift(tmp2, 1), rc);
            a_q24[k - n - 1] = silk_smlawb(tmp2, silk_lshift(tmp1, 1), rc);
        }
        a_q24[k] = -silk_lshift(rc_q15[k] as i32, 9);
    }
}

/// Port of `silk/fixed/k2a_Q16_FIX.c:silk_k2a_Q16` — step up function, converts reflection
/// coefficients (Q16) to prediction coefficients (Q24).
pub fn silk_k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    for k in 0..order {
        let rc = rc_q16[k];
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlaww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smlaww(tmp2, tmp1, rc);
        }
        a_q24[k] = -silk_lshift(rc, 8);
    }
}

// ---------------------------------------------------------------------------------------------
// schur_FIX.c / schur64_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/schur_FIX.c:silk_schur` — faster than [`silk_schur64`], but much less
/// accurate. Writes the reflection coefficients `rc_q15[..order]` from the correlations
/// `c[..=order]`; returns the residual energy.
pub fn silk_schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    let mut cc = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];
    celt_assert!(order <= SILK_MAX_ORDER_LPC);

    // Get number of leading zeros
    let mut lz = silk_clz32(c[0]);

    // Copy correlations and adjust level to Q30
    if lz < 2 {
        // lz must be 1, so shift one to the right
        for k in 0..=order {
            let v = silk_rshift(c[k], 1);
            cc[k] = [v, v];
        }
    } else if lz > 2 {
        // Shift to the left
        lz -= 2;
        for k in 0..=order {
            let v = silk_lshift(c[k], lz);
            cc[k] = [v, v];
        }
    } else {
        // No need to shift
        for k in 0..=order {
            cc[k] = [c[k], c[k]];
        }
    }

    let mut k = 0;
    while k < order {
        // Check that we won't be getting an unstable rc, otherwise stop here.
        if silk_abs_int32(cc[k + 1][0]) >= cc[0][1] {
            if cc[k + 1][0] > 0 {
                rc_q15[k] = -(silk_fix_const(0.99f32 as f64, 15) as i16);
            } else {
                rc_q15[k] = silk_fix_const(0.99f32 as f64, 15) as i16;
            }
            k += 1;
            break;
        }

        // Get reflection coefficient
        let mut rc_tmp_q15 =
            -silk_div32_16(cc[k + 1][0], silk_max_32(silk_rshift(cc[0][1], 15), 1));

        // Clip (shouldn't happen for properly conditioned inputs)
        rc_tmp_q15 = silk_sat16(rc_tmp_q15);

        // Store
        rc_q15[k] = rc_tmp_q15 as i16;

        // Update correlations
        for n in 0..order - k {
            let ctmp1 = cc[n + k + 1][0];
            let ctmp2 = cc[n][1];
            cc[n + k + 1][0] = silk_smlawb(ctmp1, silk_lshift(ctmp2, 1), rc_tmp_q15);
            cc[n][1] = silk_smlawb(ctmp2, silk_lshift(ctmp1, 1), rc_tmp_q15);
        }
        k += 1;
    }

    while k < order {
        rc_q15[k] = 0;
        k += 1;
    }

    // return residual energy
    silk_max_32(1, cc[0][1])
}

/// Port of `silk/fixed/schur64_FIX.c:silk_schur64` — slower than [`silk_schur`], but more
/// accurate. Writes the reflection coefficients `rc_q16[..order]` from the correlations
/// `c[..=order]`; returns the residual energy.
pub fn silk_schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    let mut cc = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];
    celt_assert!(order <= SILK_MAX_ORDER_LPC);

    // Check for invalid input
    if c[0] <= 0 {
        rc_q16[..order].fill(0);
        return 0;
    }

    for k in 0..=order {
        cc[k] = [c[k], c[k]];
    }

    let mut k = 0;
    while k < order {
        // Check that we won't be getting an unstable rc, otherwise stop here.
        if silk_abs_int32(cc[k + 1][0]) >= cc[0][1] {
            if cc[k + 1][0] > 0 {
                rc_q16[k] = -silk_fix_const(0.99f32 as f64, 16);
            } else {
                rc_q16[k] = silk_fix_const(0.99f32 as f64, 16);
            }
            k += 1;
            break;
        }

        // Get reflection coefficient: divide two Q30 values and get result in Q31
        let rc_tmp_q31 = silk_div32_varq(-cc[k + 1][0], cc[0][1], 31);

        // Save the output
        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        // Update correlations
        for n in 0..order - k {
            let ctmp1_q30 = cc[n + k + 1][0];
            let ctmp2_q30 = cc[n][1];

            // Multiply and add the highest int32
            cc[n + k + 1][0] = ctmp1_q30 + silk_smmul(silk_lshift(ctmp2_q30, 1), rc_tmp_q31);
            cc[n][1] = ctmp2_q30 + silk_smmul(silk_lshift(ctmp1_q30, 1), rc_tmp_q31);
        }
        k += 1;
    }

    while k < order {
        rc_q16[k] = 0;
        k += 1;
    }

    silk_max_32(1, cc[0][1])
}

// ---------------------------------------------------------------------------------------------
// corrMatrix_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/corrMatrix_FIX.c:silk_corrVector_FIX` — calculates the correlation
/// vector `X'*t` (`xt[..order]`); `x` holds `l + order - 1` samples, `t` holds `l`.
pub fn silk_corr_vector_fix(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshifts: i32,
) {
    let t = &t[..l];
    // C: ptr1 = &x[ order - 1 ], points to first sample of column 0 of X: X[:,0]
    if rshifts > 0 {
        // Right shifting used
        for lag in 0..order {
            let ptr1 = &x[order - 1 - lag..order - 1 - lag + l];
            let mut inner_prod: i32 = 0;
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(ptr1[i] as i32, t[i] as i32),
                    rshifts,
                );
            }
            xt[lag] = inner_prod; // X[:,lag]'*t
        }
    } else {
        silk_assert!(rshifts == 0);
        for lag in 0..order {
            xt[lag] = silk_inner_prod_aligned(&x[order - 1 - lag..], t, l); // X[:,lag]'*t
        }
    }
}

/// Port of `silk/fixed/corrMatrix_FIX.c:silk_corrMatrix_FIX` — calculates the correlation
/// matrix `X'*X` (`xx[..order * order]`, row-major); `x` holds `l + order - 1` samples.
/// Returns `(nrg, rshifts)`: the energy of `x` and the right shifts of the correlations.
pub fn silk_corr_matrix_fix(x: &[i16], l: usize, order: usize, xx: &mut [i32]) -> (i32, i32) {
    let x = &x[..l + order - 1];
    let xx = &mut xx[..order * order];

    // Calculate energy to find shift used to fit in 32 bits
    let (nrg, rshifts) = silk_sum_sqr_shift(x, l + order - 1);
    let mut energy = nrg;

    // Calculate energy of first column (0) of X: X[:,0]'*X[:,0]
    // Remove contribution of first order - 1 samples
    for i in 0..order - 1 {
        energy -= silk_rshift32(silk_smulbb(x[i] as i32, x[i] as i32), rshifts);
    }

    // Calculate energy of remaining columns of X: X[:,j]'*X[:,j]
    // Fill out the diagonal of the correlation matrix
    xx[0] = energy;
    silk_assert!(energy >= 0);
    let p1 = order - 1; // First sample of column 0 of X
    for j in 1..order {
        energy = silk_sub32(
            energy,
            silk_rshift32(
                silk_smulbb(x[p1 + l - j] as i32, x[p1 + l - j] as i32),
                rshifts,
            ),
        );
        energy = silk_add32(
            energy,
            silk_rshift32(silk_smulbb(x[p1 - j] as i32, x[p1 - j] as i32), rshifts),
        );
        xx[j * order + j] = energy;
        silk_assert!(energy >= 0);
    }

    // C: ptr2 = &x[ order - 2 ] (first sample of column 1 of X), decremented after each lag:
    // `p2 = order - 1 - lag` below.
    // Calculate the remaining elements of the correlation matrix
    if rshifts > 0 {
        // Right shifting used
        for lag in 1..order {
            let p2 = order - 1 - lag;
            // Inner product of column 0 and column lag: X[:,0]'*X[:,lag]
            energy = 0;
            for i in 0..l {
                energy += silk_rshift32(silk_smulbb(x[p1 + i] as i32, x[p2 + i] as i32), rshifts);
            }
            // Calculate remaining off diagonal: X[:,j]'*X[:,j + lag]
            xx[lag * order] = energy;
            xx[lag] = energy;
            for j in 1..order - lag {
                energy = silk_sub32(
                    energy,
                    silk_rshift32(
                        silk_smulbb(x[p1 + l - j] as i32, x[p2 + l - j] as i32),
                        rshifts,
                    ),
                );
                energy = silk_add32(
                    energy,
                    silk_rshift32(silk_smulbb(x[p1 - j] as i32, x[p2 - j] as i32), rshifts),
                );
                xx[(lag + j) * order + j] = energy;
                xx[j * order + lag + j] = energy;
            }
        }
    } else {
        for lag in 1..order {
            let p2 = order - 1 - lag;
            // Inner product of column 0 and column lag: X[:,0]'*X[:,lag]
            energy = silk_inner_prod_aligned(&x[p1..], &x[p2..], l);
            xx[lag * order] = energy;
            xx[lag] = energy;
            // Calculate remaining off diagonal: X[:,j]'*X[:,j + lag]
            for j in 1..order - lag {
                energy = silk_sub32(
                    energy,
                    silk_smulbb(x[p1 + l - j] as i32, x[p2 + l - j] as i32),
                );
                energy = silk_smlabb(energy, x[p1 - j] as i32, x[p2 - j] as i32);
                xx[(lag + j) * order + j] = energy;
                xx[j * order + lag + j] = energy;
            }
        }
    }
    (nrg, rshifts)
}

// ---------------------------------------------------------------------------------------------
// residual_energy16_FIX.c / residual_energy_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/residual_energy16_FIX.c:silk_residual_energy16_covar_FIX` — residual
/// energy `nrg = wxx - 2 * wXx * c + c' * wXX * c` (`c` in Q`c_q`, `0 < c_q < 16`, `d <= 16`).
#[must_use]
pub fn silk_residual_energy16_covar_fix(
    c: &[i16],
    w_xx_mat: &[i32],
    w_xx_vec: &[i32],
    wxx: i32,
    d: usize,
    c_q: i32,
) -> i32 {
    const MMS: usize = MAX_MATRIX_SIZE as usize;
    let mut cn = [0i32; MMS];

    // Safety checks
    celt_assert!(d <= 16);
    celt_assert!(c_q > 0);
    celt_assert!(c_q < 16);

    let mut lshifts = 16 - c_q;
    let mut qxtra = lshifts;

    let mut c_max = 0;
    for &ci in &c[..d] {
        c_max = silk_max_32(c_max, silk_abs(ci as i32));
    }
    qxtra = silk_min_int(qxtra, silk_clz32(c_max) - 17);

    let w_max = silk_max_32(w_xx_mat[0], w_xx_mat[d * d - 1]);
    qxtra = silk_min_int(
        qxtra,
        silk_clz32((d as i32) * silk_rshift(silk_smulwb(w_max, c_max), 4)) - 5,
    );
    qxtra = silk_max_int(qxtra, 0);
    for i in 0..d {
        cn[i] = silk_lshift(c[i] as i32, qxtra);
        // Check that silk_SMLAWB can be used
        silk_assert!(silk_abs(cn[i]) <= i16::MAX as i32 + 1);
    }
    lshifts -= qxtra;

    // Compute wxx - 2 * wXx * c
    let mut tmp: i32 = 0;
    for i in 0..d {
        tmp = silk_smlawb(tmp, w_xx_vec[i], cn[i]);
    }
    let mut nrg = silk_rshift(wxx, 1 + lshifts) - tmp; // Q: -lshifts - 1

    // Add c' * wXX * c, assuming wXX is symmetric
    let mut tmp2: i32 = 0;
    for i in 0..d {
        let mut tmp: i32 = 0;
        let p_row = &w_xx_mat[i * d..(i + 1) * d];
        for j in i + 1..d {
            tmp = silk_smlawb(tmp, p_row[j], cn[j]);
        }
        tmp = silk_smlawb(tmp, silk_rshift(p_row[i], 1), cn[i]);
        tmp2 = silk_smlawb(tmp2, tmp, cn[i]);
    }
    nrg = silk_add_lshift32(nrg, tmp2, lshifts); // Q: -lshifts - 1

    // Keep one bit free always, because we add them for LSF interpolation
    if nrg < 1 {
        1
    } else if nrg > silk_rshift(i32::MAX, lshifts + 2) {
        i32::MAX >> 1
    } else {
        silk_lshift(nrg, lshifts + 1) // Q0
    }
}

/// Port of `silk/fixed/residual_energy_FIX.c:silk_residual_energy_FIX` — calculates residual
/// energies of input subframes where all subframes have `lpc_order` of preceding samples.
///
/// Writes `nrgs[..nb_subfr]` and their Q values `nrgs_q[..nb_subfr]`.
pub fn silk_residual_energy_fix(
    nrgs: &mut [i32; MNSF],
    nrgs_q: &mut [i32; MNSF],
    x: &[i16],
    a_q12: &[[i16; MLPC]; 2],
    gains: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) {
    const HALF: usize = MNSF >> 1;
    // C: ALLOC( LPC_res, ( MAX_NB_SUBFR >> 1 ) * offset ), offset <= MAX_LPC_ORDER + 80
    let mut lpc_res_buf = [0i16; HALF * (MLPC + 80)];
    let offset = lpc_order + subfr_length;
    let lpc_res = &mut lpc_res_buf[..HALF * offset];

    // Filter input to create the LPC residual for each frame half, and measure subframe
    // energies
    celt_assert!((nb_subfr >> 1) * HALF == nb_subfr);
    let mut x_ptr = 0;
    for i in 0..nb_subfr >> 1 {
        // Calculate half frame LPC residual signal including preceding samples
        silk_lpc_analysis_filter(lpc_res, &x[x_ptr..], &a_q12[i], HALF * offset, lpc_order);

        // Point to first subframe of the just calculated LPC residual signal
        let mut lpc_res_ptr = lpc_order;
        for j in 0..HALF {
            // Measure subframe energy
            let (nrg, rshift) = silk_sum_sqr_shift(&lpc_res[lpc_res_ptr..], subfr_length);
            nrgs[i * HALF + j] = nrg;

            // Set Q values for the measured energy
            nrgs_q[i * HALF + j] = -rshift;

            // Move to next subframe
            lpc_res_ptr += offset;
        }
        // Move to next frame half
        x_ptr += HALF * offset;
    }

    // Apply the squared subframe gains
    for i in 0..nb_subfr {
        // Fully upscale gains and energies
        let lz1 = silk_clz32(nrgs[i]) - 1;
        let lz2 = silk_clz32(gains[i]) - 1;

        let mut tmp32 = silk_lshift32(gains[i], lz2);

        // Find squared gains
        tmp32 = silk_smmul(tmp32, tmp32); // Q( 2 * lz2 - 32 )

        // Scale energies
        nrgs[i] = silk_smmul(tmp32, silk_lshift32(nrgs[i], lz1)); // Q( nrgsQ[ i ] + lz1 + 2 * lz2 - 32 - 32 )
        nrgs_q[i] += lz1 + 2 * lz2 - 32 - 32;
    }
}

// ---------------------------------------------------------------------------------------------
// regularize_correlations_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/regularize_correlations_FIX.c:silk_regularize_correlations_FIX` — add
/// noise to the matrix diagonal (`xx_mat` is `d x d`) and to `xx[0]`.
pub fn silk_regularize_correlations_fix(xx_mat: &mut [i32], xx: &mut [i32], noise: i32, d: usize) {
    for i in 0..d {
        xx_mat[i * d + i] = silk_add32(xx_mat[i * d + i], noise);
    }
    xx[0] += noise;
}

// ---------------------------------------------------------------------------------------------
// warped_autocorrelation_FIX.c
// ---------------------------------------------------------------------------------------------

/// `QC` (main_FIX.h).
const QC: i32 = 10;
/// `QS` (main_FIX.h).
const QS: i32 = 13;

/// Port of `silk/fixed/warped_autocorrelation_FIX.c:silk_warped_autocorrelation_FIX_c` —
/// autocorrelations for a warped frequency axis (`corr[..=order]`, `order` even). Returns the
/// C `*scale` output (the scaling of the correlation vector).
pub fn silk_warped_autocorrelation_fix(
    corr: &mut [i32],
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) -> i32 {
    const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;
    let mut state_qs = [0i32; MSO + 1];
    let mut corr_qc = [0i64; MSO + 1];

    // Order must be even
    celt_assert!(order & 1 == 0);
    celt_assert!(order <= MSO);

    // Loop over samples
    for &inp in &input[..length] {
        let mut tmp1_qs = silk_lshift32(inp as i32, QS);
        // Loop over allpass sections
        let mut i = 0;
        while i < order {
            // Output of allpass section
            let tmp2_qs = silk_smlawb(state_qs[i], state_qs[i + 1] - tmp1_qs, warping_q16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
            // Output of allpass section
            tmp1_qs = silk_smlawb(state_qs[i + 1], state_qs[i + 2] - tmp2_qs, warping_q16);
            state_qs[i + 1] = tmp2_qs;
            corr_qc[i + 1] += silk_rshift64(silk_smull(tmp2_qs, state_qs[0]), 2 * QS - QC);
            i += 2;
        }
        state_qs[order] = tmp1_qs;
        corr_qc[order] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
    }

    let mut lsh = silk_clz64(corr_qc[0]) - 35;
    lsh = silk_limit(lsh, -12 - QC, 30 - QC);
    let scale = -(QC + lsh);
    celt_assert!((-30..=12).contains(&scale));
    if lsh >= 0 {
        for i in 0..order + 1 {
            corr[i] = silk_lshift64(corr_qc[i], lsh) as i32;
        }
    } else {
        for i in 0..order + 1 {
            corr[i] = silk_rshift64(corr_qc[i], -lsh) as i32;
        }
    }
    silk_assert!(corr_qc[0] >= 0); // If breaking, decrease QC
    scale
}

// ---------------------------------------------------------------------------------------------
// LTP_analysis_filter_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/LTP_analysis_filter_FIX.c:silk_LTP_analysis_filter_FIX`.
///
/// The C input pointer `x` is `&x_buf[x_off]` (the filter reads up to `max(pitchL) + 2`
/// samples before it). Writes `nb_subfr * (pre_length + subfr_length)` samples of `ltp_res`.
pub fn silk_ltp_analysis_filter_fix(
    ltp_res: &mut [i16],
    x_buf: &[i16],
    x_off: usize,
    ltp_coef_q14: &[i16],
    pitch_l: &[i32],
    inv_gains_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut btmp_q14 = [0i16; LTPO];
    let mut x_ptr = x_off;
    let mut ltp_res_ptr = 0;
    let n = subfr_length + pre_length;
    for k in 0..nb_subfr {
        // C: x_lag_ptr = x_ptr - pitchL[ k ]; the taps read x_lag_ptr[ -2..=2 ]
        let x_lag_ptr = x_ptr - pitch_l[k] as usize;

        btmp_q14.copy_from_slice(&ltp_coef_q14[k * LTPO..(k + 1) * LTPO]);

        let x = &x_buf[x_ptr..x_ptr + n];
        let xl = &x_buf[x_lag_ptr - 2..x_lag_ptr + n + 2];
        let res = &mut ltp_res[ltp_res_ptr..ltp_res_ptr + n];

        // LTP analysis FIR filter
        for i in 0..n {
            // Long-term prediction (xl[ i + 2 ] is x_lag_ptr[ 0 ])
            let mut ltp_est = silk_smulbb(xl[i + 4] as i32, btmp_q14[0] as i32);
            ltp_est = silk_smlabb_ovflw(ltp_est, xl[i + 3] as i32, btmp_q14[1] as i32);
            ltp_est = silk_smlabb_ovflw(ltp_est, xl[i + 2] as i32, btmp_q14[2] as i32);
            ltp_est = silk_smlabb_ovflw(ltp_est, xl[i + 1] as i32, btmp_q14[3] as i32);
            ltp_est = silk_smlabb_ovflw(ltp_est, xl[i] as i32, btmp_q14[4] as i32);

            ltp_est = silk_rshift_round(ltp_est, 14); // round and -> Q0

            // Subtract long-term prediction
            let r = silk_sat16(x[i] as i32 - ltp_est) as i16;

            // Scale residual
            res[i] = silk_smulwb(inv_gains_q16[k], r as i32) as i16;
        }

        // Update pointers
        ltp_res_ptr += n;
        x_ptr += subfr_length;
    }
}

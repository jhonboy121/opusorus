//! Port of the SILK NLSF code: `silk/NLSF2A.c`, `silk/A2NLSF.c`, `silk/NLSF_decode.c`,
//! `silk/NLSF_unpack.c`, `silk/NLSF_stabilize.c`, `silk/NLSF_VQ_weights_laroia.c`,
//! `silk/NLSF_VQ.c`, `silk/NLSF_encode.c` and `silk/NLSF_del_dec_quant.c`.
//!
//! Conversion between prediction filter coefficients and LSFs uses a piecewise linear
//! approximation of LSF <-> cos(LSF); the result is not accurate LSFs, but the two functions are
//! accurate inverses of each other. Filter orders must be even.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::silk::define::{
    LSF_COS_TAB_SZ_FIX, MAX_LPC_ORDER, MAX_LPC_STABILIZE_ITERATIONS, NLSF_QUANT_DEL_DEC_STATES,
    NLSF_QUANT_DEL_DEC_STATES_LOG2, NLSF_QUANT_LEVEL_ADJ, NLSF_QUANT_MAX_AMPLITUDE,
    NLSF_QUANT_MAX_AMPLITUDE_EXT, NLSF_VQ_MAX_VECTORS, NLSF_W_Q,
};
use crate::silk::macros::{
    SILK_INT16_MAX, SILK_MAX_ORDER_LPC, silk_abs, silk_add_lshift32, silk_add_rshift,
    silk_add_sat16, silk_add32, silk_div32, silk_div32_16, silk_div32_varq, silk_fix_const,
    silk_limit, silk_limit_32, silk_lshift, silk_lshift16, silk_max_int, silk_min_32, silk_min_int,
    silk_mla, silk_mul, silk_rshift, silk_rshift_round, silk_rshift_round64, silk_smlabb,
    silk_smlawb, silk_smlaww, silk_smulbb, silk_smull, silk_sub_lshift32, silk_sub_rshift32,
};
use crate::silk::sigproc::{
    silk_bwexpander_32, silk_insertion_sort_increasing,
    silk_insertion_sort_increasing_all_values_int16, silk_lin2log, silk_lpc_fit,
    silk_lpc_inverse_pred_gain,
};
use crate::silk::structs::SilkNlsfCbStruct;
use crate::silk::tables::SILK_LSFCOSTAB_FIX_Q12;

const MLPC: usize = MAX_LPC_ORDER as usize;

// ---------------------------------------------------------------------------------------------
// NLSF2A.c
// ---------------------------------------------------------------------------------------------

/// `QA` (silk/NLSF2A.c).
const QA_NLSF2A: i32 = 16;

/// Port of silk/NLSF2A.c:silk_NLSF2A_find_poly (static helper): intermediate polynomial,
/// QA `[dd+1]`, from the interleaved `2*cos(LSFs)` vector `c_lsf` (read at even offsets).
fn silk_nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], dd: usize) {
    out[0] = silk_lshift(1, QA_NLSF2A);
    out[1] = -c_lsf[0];
    for k in 1..dd {
        let ftmp = c_lsf[2 * k]; // QA
        out[k + 1] = silk_lshift(out[k - 1], 1)
            - silk_rshift_round64(silk_smull(ftmp, out[k]), QA_NLSF2A) as i32;
        for n in (2..=k).rev() {
            out[n] +=
                out[n - 2] - silk_rshift_round64(silk_smull(ftmp, out[n - 1]), QA_NLSF2A) as i32;
        }
        out[1] -= ftmp;
    }
}

/// `ordering16` (silk/NLSF2A.c): maximizes numerical accuracy of the polynomial expansion.
const ORDERING16: [u8; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
/// `ordering10` (silk/NLSF2A.c).
const ORDERING10: [u8; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

/// Port of silk/NLSF2A.c:silk_NLSF2A — compute whitening filter coefficients (monic, Q12,
/// `[d]`) from normalized line spectral frequencies (Q15, `[d]`). `d` must be 10 or 16.
pub fn silk_nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    let mut cos_lsf_qa = [0i32; SILK_MAX_ORDER_LPC];
    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut a32_qa1 = [0i32; SILK_MAX_ORDER_LPC];

    const { assert!(LSF_COS_TAB_SZ_FIX == 128) };
    celt_assert!(d == 10 || d == 16);

    // convert LSFs to 2*cos(LSF), using piecewise linear curve from table
    let ordering: &[u8] = if d == 16 { &ORDERING16 } else { &ORDERING10 };
    for k in 0..d {
        silk_assert!(nlsf[k] >= 0);

        // f_int on a scale 0-127 (rounded down)
        let f_int = silk_rshift(nlsf[k] as i32, 15 - 7);

        // f_frac, range: 0..255
        let f_frac = nlsf[k] as i32 - silk_lshift(f_int, 15 - 7);

        silk_assert!(f_int >= 0);
        silk_assert!(f_int < LSF_COS_TAB_SZ_FIX);

        // Read start and end value from table
        let cos_val = SILK_LSFCOSTAB_FIX_Q12[f_int as usize] as i32; // Q12
        let delta = SILK_LSFCOSTAB_FIX_Q12[f_int as usize + 1] as i32 - cos_val; // Q12, with a range of 0..200

        // Linear interpolation
        cos_lsf_qa[ordering[k] as usize] = silk_rshift_round(
            silk_lshift(cos_val, 8) + silk_mul(delta, f_frac),
            20 - QA_NLSF2A,
        ); // QA
    }

    let dd = d >> 1;

    // generate even and odd polynomials using convolution
    silk_nlsf2a_find_poly(&mut p, &cos_lsf_qa[0..], dd);
    silk_nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    // convert even and odd polynomials to opus_int32 Q12 filter coefs
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];

        // the Ptmp and Qtmp values at this stage need to fit in int32
        a32_qa1[k] = -qtmp - ptmp; // QA+1
        a32_qa1[d - k - 1] = qtmp - ptmp; // QA+1
    }

    // Convert int32 coefficients to Q12 int16 coefs
    silk_lpc_fit(a_q12, &mut a32_qa1, 12, QA_NLSF2A + 1, d);

    let mut i = 0;
    while silk_lpc_inverse_pred_gain(a_q12, d) == 0 && i < MAX_LPC_STABILIZE_ITERATIONS {
        // Prediction coefficients are (too close to) unstable; apply bandwidth expansion
        // on the unscaled coefficients, convert to Q12 and measure again
        silk_bwexpander_32(&mut a32_qa1, d, 65536 - silk_lshift(2, i));
        for k in 0..d {
            a_q12[k] = silk_rshift_round(a32_qa1[k], QA_NLSF2A + 1 - 12) as i16; // QA+1 -> Q12
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------------------------
// A2NLSF.c
// ---------------------------------------------------------------------------------------------

/// `BIN_DIV_STEPS_A2NLSF_FIX`: number of binary divisions, when not in low complexity mode.
const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3; // must be no higher than 16 - log2( LSF_COS_TAB_SZ_FIX )
/// `MAX_ITERATIONS_A2NLSF_FIX`.
const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;

/// Polynomial storage for `silk_A2NLSF` (`P` and `Q`).
type A2nlsfPoly = [i32; SILK_MAX_ORDER_LPC / 2 + 1];

/// Port of silk/A2NLSF.c:silk_A2NLSF_trans_poly (static helper): transforms polynomials from
/// `cos(n*f)` to `cos(f)^n`.
fn silk_a2nlsf_trans_poly(p: &mut A2nlsfPoly, dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= silk_lshift(p[k], 1);
    }
}

/// Port of silk/A2NLSF.c:silk_A2NLSF_eval_poly (static helper): polynomial evaluation.
/// Returns the evaluation in Q16; `x` is in Q12.
///
/// C unrolls the `dd == 8` case; the unrolled code performs exactly the loop below.
fn silk_a2nlsf_eval_poly(p: &A2nlsfPoly, x: i32, dd: usize) -> i32 {
    let mut y32 = p[dd]; // Q16
    let x_q16 = silk_lshift(x, 4);
    for n in (0..dd).rev() {
        y32 = silk_smlaww(p[n], y32, x_q16); // Q16
    }
    y32
}

/// Port of silk/A2NLSF.c:silk_A2NLSF_init (static helper).
fn silk_a2nlsf_init(a_q16: &[i32], pq: &mut [A2nlsfPoly; 2], dd: usize) {
    let [p, q] = pq;

    // Convert filter coefs to even and odd polynomials
    p[dd] = silk_lshift(1, 16);
    q[dd] = silk_lshift(1, 16);
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k]; // Q16
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k]; // Q16
    }

    // Divide out zeros as we have that for even filter orders,
    // z =  1 is always a root in Q, and
    // z = -1 is always a root in P
    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }

    // Transform polynomials from cos(n*f) to cos(f)^n
    silk_a2nlsf_trans_poly(p, dd);
    silk_a2nlsf_trans_poly(q, dd);
}

/// Port of silk/A2NLSF.c:silk_A2NLSF — compute Normalized Line Spectral Frequencies (NLSFs,
/// Q15, `[d]`) from monic whitening filter coefficients (Q16, `[d]`). If not all roots are
/// found, the `a_q16` coefficients are bandwidth expanded until convergence.
pub fn silk_a2nlsf(nlsf: &mut [i16], a_q16: &mut [i32], d: usize) {
    // `PQ[ 0 ] = P`, `PQ[ 1 ] = Q`; `p` below indexes into it (C: pointer to polynomial).
    let mut pq: [A2nlsfPoly; 2] = [[0; SILK_MAX_ORDER_LPC / 2 + 1]; 2];

    let dd = d >> 1;

    silk_a2nlsf_init(a_q16, &mut pq, dd);

    // Find roots, alternating between P and Q
    let mut p = 0usize; // Pointer to polynomial

    let mut xlo = SILK_LSFCOSTAB_FIX_Q12[0] as i32; // Q12
    let mut ylo = silk_a2nlsf_eval_poly(&pq[p], xlo, dd);

    let mut root_ix: usize;
    if ylo < 0 {
        // Set the first NLSF to zero and move on to the next
        nlsf[0] = 0;
        p = 1; // Pointer to polynomial
        ylo = silk_a2nlsf_eval_poly(&pq[p], xlo, dd);
        root_ix = 1; // Index of current root
    } else {
        root_ix = 0; // Index of current root
    }
    let mut k: usize = 1; // Loop counter
    let mut i: i32 = 0; // Counter for bandwidth expansions applied
    let mut thr: i32 = 0;
    loop {
        // Evaluate polynomial
        let mut xhi = SILK_LSFCOSTAB_FIX_Q12[k] as i32; // Q12
        let mut yhi = silk_a2nlsf_eval_poly(&pq[p], xhi, dd);

        // Detect zero crossing
        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            if yhi == 0 {
                // If the root lies exactly at the end of the current
                // interval, look for the next root in the next interval
                thr = 1;
            } else {
                thr = 0;
            }
            // Binary division
            let mut ffrac: i32 = -256;
            for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                // Evaluate polynomial
                let xmid = silk_rshift_round(xlo + xhi, 1);
                let ymid = silk_a2nlsf_eval_poly(&pq[p], xmid, dd);

                // Detect zero crossing
                if (ylo <= 0 && ymid >= 0) || (ylo >= 0 && ymid <= 0) {
                    // Reduce frequency
                    xhi = xmid;
                    yhi = ymid;
                } else {
                    // Increase frequency
                    xlo = xmid;
                    ylo = ymid;
                    ffrac = silk_add_rshift(ffrac, 128, m);
                }
            }

            // Interpolate
            if silk_abs(ylo) < 65536 {
                // Avoid dividing by zero
                let den = ylo - yhi;
                let nom = silk_lshift(ylo, 8 - BIN_DIV_STEPS_A2NLSF_FIX) + silk_rshift(den, 1);
                if den != 0 {
                    ffrac += silk_div32(nom, den);
                }
            } else {
                // No risk of dividing by zero because abs(ylo - yhi) >= abs(ylo) >= 65536
                ffrac += silk_div32(ylo, silk_rshift(ylo - yhi, 8 - BIN_DIV_STEPS_A2NLSF_FIX));
            }
            nlsf[root_ix] = silk_min_32(silk_lshift(k as i32, 8) + ffrac, SILK_INT16_MAX) as i16;

            silk_assert!(nlsf[root_ix] >= 0);

            root_ix += 1; // Next root
            if root_ix >= d {
                // Found all roots
                break;
            }

            // Alternate pointer to polynomial
            p = root_ix & 1;

            // Evaluate polynomial
            xlo = SILK_LSFCOSTAB_FIX_Q12[k - 1] as i32; // Q12
            ylo = silk_lshift(1 - (root_ix as i32 & 2), 12);
        } else {
            // Increment loop counter
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX as usize {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF_FIX {
                    // Set NLSFs to white spectrum and exit
                    nlsf[0] = silk_div32_16(1 << 15, d as i32 + 1) as i16;
                    for k in 1..d {
                        nlsf[k] = (nlsf[k - 1] as i32 + nlsf[0] as i32) as i16;
                    }
                    return;
                }

                // Error: Apply progressively more bandwidth expansion and run again
                silk_bwexpander_32(a_q16, d, 65536 - silk_lshift(1, i));

                silk_a2nlsf_init(a_q16, &mut pq, dd);
                p = 0; // Pointer to polynomial
                xlo = SILK_LSFCOSTAB_FIX_Q12[0] as i32; // Q12
                ylo = silk_a2nlsf_eval_poly(&pq[p], xlo, dd);
                if ylo < 0 {
                    // Set the first NLSF to zero and move on to the next
                    nlsf[0] = 0;
                    p = 1; // Pointer to polynomial
                    ylo = silk_a2nlsf_eval_poly(&pq[p], xlo, dd);
                    root_ix = 1; // Index of current root
                } else {
                    root_ix = 0; // Index of current root
                }
                k = 1; // Reset loop counter
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// NLSF_unpack.c / NLSF_decode.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NLSF_unpack.c:silk_NLSF_unpack — unpack predictor values and indices for
/// entropy coding tables.
pub fn silk_nlsf_unpack(
    ec_ix: &mut [i16],
    pred_q8: &mut [u8],
    ps_nlsf_cb: &SilkNlsfCbStruct,
    cb1_index: i32,
) {
    let order = ps_nlsf_cb.order as usize;
    let ec_sel_ptr = &ps_nlsf_cb.ec_sel[(cb1_index * ps_nlsf_cb.order as i32 / 2) as usize..];
    for i in (0..order).step_by(2) {
        let entry = ec_sel_ptr[i / 2] as i32;
        ec_ix[i] = silk_smulbb(silk_rshift(entry, 1) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as i16;
        pred_q8[i] = ps_nlsf_cb.pred_q8[i + (entry & 1) as usize * (order - 1)];
        ec_ix[i + 1] =
            silk_smulbb(silk_rshift(entry, 5) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as i16;
        pred_q8[i + 1] =
            ps_nlsf_cb.pred_q8[i + (silk_rshift(entry, 4) & 1) as usize * (order - 1) + 1];
    }
}

/// Port of silk/NLSF_decode.c:silk_NLSF_residual_dequant (static helper): predictive
/// dequantizer for NLSF residuals.
fn silk_nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: i16,
) {
    let mut out_q10: i32 = 0;
    for i in (0..order as usize).rev() {
        let pred_q10 = silk_rshift(silk_smulbb(out_q10, pred_coef_q8[i] as i32), 8);
        out_q10 = silk_lshift(indices[i] as i32, 10);
        if out_q10 > 0 {
            out_q10 -= silk_fix_const(NLSF_QUANT_LEVEL_ADJ, 10);
        } else if out_q10 < 0 {
            out_q10 += silk_fix_const(NLSF_QUANT_LEVEL_ADJ, 10);
        }
        out_q10 = silk_smlawb(pred_q10, out_q10, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
}

/// Port of silk/NLSF_decode.c:silk_NLSF_decode — NLSF vector decoder.
///
/// `nlsf_indices` is the codebook path vector `[LPC_ORDER + 1]`; writes the quantized NLSF
/// vector `[LPC_ORDER]`.
pub fn silk_nlsf_decode(
    p_nlsf_q15: &mut [i16],
    nlsf_indices: &[i8],
    ps_nlsf_cb: &SilkNlsfCbStruct,
) {
    let mut pred_q8 = [0u8; MLPC];
    let mut ec_ix = [0i16; MLPC];
    let mut res_q10 = [0i16; MLPC];
    let order = ps_nlsf_cb.order as usize;

    // Unpack entropy table indices and predictor for current CB1 index
    silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, ps_nlsf_cb, nlsf_indices[0] as i32);

    // Predictive residual dequantizer
    silk_nlsf_residual_dequant(
        &mut res_q10,
        &nlsf_indices[1..],
        &pred_q8,
        ps_nlsf_cb.quant_step_size_q16 as i32,
        ps_nlsf_cb.order,
    );

    // Apply inverse square-rooted weights to first stage and add to output
    let base = nlsf_indices[0] as usize * order;
    let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[base..base + order];
    let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[base..base + order];
    for i in 0..order {
        let nlsf_q15_tmp = silk_add_lshift32(
            silk_div32_16(silk_lshift(res_q10[i] as i32, 14), p_cb_wght_q9[i] as i32),
            p_cb_element[i] as i32,
            7,
        );
        p_nlsf_q15[i] = silk_limit(nlsf_q15_tmp, 0, 32767) as i16;
    }

    // NLSF stabilization
    silk_nlsf_stabilize(p_nlsf_q15, ps_nlsf_cb.delta_min_q15, order);
}

// ---------------------------------------------------------------------------------------------
// NLSF_stabilize.c
// ---------------------------------------------------------------------------------------------

/// `MAX_LOOPS` (silk/NLSF_stabilize.c).
const MAX_LOOPS: i32 = 20;

/// Port of silk/NLSF_stabilize.c:silk_NLSF_stabilize — NLSF stabilizer, for a single input
/// data vector.
///
/// Moves NLSFs further apart if they are too close, and away from the borders; output is
/// sorted. `ndelta_min_q15` has `l + 1` entries and `ndelta_min_q15[l]` must be >= 1.
pub fn silk_nlsf_stabilize(nlsf_q15: &mut [i16], ndelta_min_q15: &[i16], l: usize) {
    // This is necessary to ensure an output within range of a opus_int16
    silk_assert!(ndelta_min_q15[l] >= 1);

    let mut loops = 0;
    while loops < MAX_LOOPS {
        // Find smallest distance
        // First element
        let mut min_diff_q15 = nlsf_q15[0] as i32 - ndelta_min_q15[0] as i32;
        let mut ii = 0usize;
        // Middle elements
        for i in 1..l {
            let diff_q15 = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + ndelta_min_q15[i] as i32);
            if diff_q15 < min_diff_q15 {
                min_diff_q15 = diff_q15;
                ii = i;
            }
        }
        // Last element
        let diff_q15 = (1 << 15) - (nlsf_q15[l - 1] as i32 + ndelta_min_q15[l] as i32);
        if diff_q15 < min_diff_q15 {
            min_diff_q15 = diff_q15;
            ii = l;
        }

        // Now check if the smallest distance non-negative
        if min_diff_q15 >= 0 {
            return;
        }

        if ii == 0 {
            // Move away from lower limit
            nlsf_q15[0] = ndelta_min_q15[0];
        } else if ii == l {
            // Move away from higher limit
            nlsf_q15[l - 1] = ((1 << 15) - ndelta_min_q15[l] as i32) as i16;
        } else {
            // Find the lower extreme for the location of the current center frequency
            let mut min_center_q15: i32 = 0;
            for k in 0..ii {
                min_center_q15 += ndelta_min_q15[k] as i32;
            }
            min_center_q15 += silk_rshift(ndelta_min_q15[ii] as i32, 1);

            // Find the upper extreme for the location of the current center frequency
            let mut max_center_q15: i32 = 1 << 15;
            for k in (ii + 1..=l).rev() {
                max_center_q15 -= ndelta_min_q15[k] as i32;
            }
            max_center_q15 -= silk_rshift(ndelta_min_q15[ii] as i32, 1);

            // Move apart, sorted by value, keeping the same center frequency
            let center_freq_q15 = silk_limit_32(
                silk_rshift_round(nlsf_q15[ii - 1] as i32 + nlsf_q15[ii] as i32, 1),
                min_center_q15,
                max_center_q15,
            ) as i16;
            nlsf_q15[ii - 1] =
                (center_freq_q15 as i32 - silk_rshift(ndelta_min_q15[ii] as i32, 1)) as i16;
            nlsf_q15[ii] = (nlsf_q15[ii - 1] as i32 + ndelta_min_q15[ii] as i32) as i16;
        }
        loops += 1;
    }

    // Safe and simple fall back method, which is less ideal than the above
    if loops == MAX_LOOPS {
        // Insertion sort (fast for already almost sorted arrays)
        silk_insertion_sort_increasing_all_values_int16(nlsf_q15, l);

        // First NLSF should be no less than NDeltaMin[0]
        nlsf_q15[0] = silk_max_int(nlsf_q15[0] as i32, ndelta_min_q15[0] as i32) as i16;

        // Keep delta_min distance between the NLSFs
        for i in 1..l {
            nlsf_q15[i] = silk_max_int(
                nlsf_q15[i] as i32,
                silk_add_sat16(nlsf_q15[i - 1], ndelta_min_q15[i] as i32) as i32,
            ) as i16;
        }

        // Last NLSF should be no higher than 1 - NDeltaMin[L]
        nlsf_q15[l - 1] =
            silk_min_int(nlsf_q15[l - 1] as i32, (1 << 15) - ndelta_min_q15[l] as i32) as i16;

        // Keep NDeltaMin distance between the NLSFs
        for i in (0..l - 1).rev() {
            nlsf_q15[i] = silk_min_int(
                nlsf_q15[i] as i32,
                nlsf_q15[i + 1] as i32 - ndelta_min_q15[i + 1] as i32,
            ) as i16;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// NLSF_VQ_weights_laroia.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NLSF_VQ_weights_laroia.c:silk_NLSF_VQ_weights_laroia — Laroia low complexity
/// NLSF weights (R. Laroia, N. Phamdo and N. Farvardin, ICASSP 1991).
pub fn silk_nlsf_vq_weights_laroia(p_nlsfw_q_out: &mut [i16], p_nlsf_q15: &[i16], d: usize) {
    celt_assert!(d > 0);
    celt_assert!((d & 1) == 0);

    // First value
    let mut tmp1_int = silk_max_int(p_nlsf_q15[0] as i32, 1);
    tmp1_int = silk_div32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
    let mut tmp2_int = silk_max_int(p_nlsf_q15[1] as i32 - p_nlsf_q15[0] as i32, 1);
    tmp2_int = silk_div32_16(1i32 << (15 + NLSF_W_Q), tmp2_int);
    p_nlsfw_q_out[0] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
    silk_assert!(p_nlsfw_q_out[0] > 0);

    // Main loop
    let mut k = 1;
    while k < d - 1 {
        tmp1_int = silk_max_int(p_nlsf_q15[k + 1] as i32 - p_nlsf_q15[k] as i32, 1);
        tmp1_int = silk_div32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
        p_nlsfw_q_out[k] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
        silk_assert!(p_nlsfw_q_out[k] > 0);

        tmp2_int = silk_max_int(p_nlsf_q15[k + 2] as i32 - p_nlsf_q15[k + 1] as i32, 1);
        tmp2_int = silk_div32_16(1i32 << (15 + NLSF_W_Q), tmp2_int);
        p_nlsfw_q_out[k + 1] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
        silk_assert!(p_nlsfw_q_out[k + 1] > 0);
        k += 2;
    }

    // Last value
    tmp1_int = silk_max_int((1 << 15) - p_nlsf_q15[d - 1] as i32, 1);
    tmp1_int = silk_div32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
    p_nlsfw_q_out[d - 1] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
    silk_assert!(p_nlsfw_q_out[d - 1] > 0);
}

// ---------------------------------------------------------------------------------------------
// NLSF_VQ.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NLSF_VQ.c:silk_NLSF_VQ — compute quantization errors (`err_q24[K]`) for an
/// `lpc_order` element input vector for a VQ codebook of `k` vectors.
pub fn silk_nlsf_vq(
    err_q24: &mut [i32],
    in_q15: &[i16],
    p_cb_q8: &[u8],
    p_wght_q9: &[i16],
    k: usize,
    lpc_order: usize,
) {
    celt_assert!((lpc_order & 1) == 0);

    // Loop over codebook
    for i in 0..k {
        let cb_q8_ptr = &p_cb_q8[i * lpc_order..(i + 1) * lpc_order];
        let w_q9_ptr = &p_wght_q9[i * lpc_order..(i + 1) * lpc_order];
        let mut sum_error_q24: i32 = 0;
        let mut pred_q24: i32 = 0;
        for m in (0..lpc_order).step_by(2).rev() {
            // Compute weighted absolute predictive quantization error for index m + 1
            let diff_q15 = silk_sub_lshift32(in_q15[m + 1] as i32, cb_q8_ptr[m + 1] as i32, 7); // range: [ -32767 : 32767 ]
            let diffw_q24 = silk_smulbb(diff_q15, w_q9_ptr[m + 1] as i32);
            sum_error_q24 = silk_add32(
                sum_error_q24,
                silk_abs(silk_sub_rshift32(diffw_q24, pred_q24, 1)),
            );
            pred_q24 = diffw_q24;

            // Compute weighted absolute predictive quantization error for index m
            let diff_q15 = silk_sub_lshift32(in_q15[m] as i32, cb_q8_ptr[m] as i32, 7); // range: [ -32767 : 32767 ]
            let diffw_q24 = silk_smulbb(diff_q15, w_q9_ptr[m] as i32);
            sum_error_q24 = silk_add32(
                sum_error_q24,
                silk_abs(silk_sub_rshift32(diffw_q24, pred_q24, 1)),
            );
            pred_q24 = diffw_q24;

            silk_assert!(sum_error_q24 >= 0);
        }
        err_q24[i] = sum_error_q24;
    }
}

// ---------------------------------------------------------------------------------------------
// NLSF_del_dec_quant.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NLSF_del_dec_quant.c:silk_NLSF_del_dec_quant — delayed-decision quantizer for
/// NLSF residuals. Writes `indices[order]` and returns the RD value in Q25.
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    const NS: usize = NLSF_QUANT_DEL_DEC_STATES as usize;
    const EXT: i32 = NLSF_QUANT_MAX_AMPLITUDE_EXT;
    const AMP: i32 = NLSF_QUANT_MAX_AMPLITUDE;
    let adj = silk_fix_const(NLSF_QUANT_LEVEL_ADJ, 10);

    let mut ind_sort = [0i32; NS];
    let mut ind = [[0i8; MLPC]; NS];
    let mut prev_out_q10 = [0i16; 2 * NS];
    let mut rd_q25 = [0i32; 2 * NS];
    let mut rd_min_q25 = [0i32; NS];
    let mut rd_max_q25 = [0i32; NS];
    let mut out0_q10_table = [0i32; 2 * EXT as usize];
    let mut out1_q10_table = [0i32; 2 * EXT as usize];

    for i in -EXT..=EXT - 1 {
        // `out0_Q10` / `out1_Q10` are `opus_int16` in C: every store truncates.
        let mut out0_q10 = silk_lshift(i, 10) as i16;
        let mut out1_q10 = (out0_q10 as i32 + 1024) as i16;
        if i > 0 {
            out0_q10 = (out0_q10 as i32 - adj) as i16;
            out1_q10 = (out1_q10 as i32 - adj) as i16;
        } else if i == 0 {
            out1_q10 = (out1_q10 as i32 - adj) as i16;
        } else if i == -1 {
            out0_q10 = (out0_q10 as i32 + adj) as i16;
        } else {
            out0_q10 = (out0_q10 as i32 + adj) as i16;
            out1_q10 = (out1_q10 as i32 + adj) as i16;
        }
        out0_q10_table[(i + EXT) as usize] =
            silk_rshift(silk_smulbb(out0_q10 as i32, quant_step_size_q16), 16);
        out1_q10_table[(i + EXT) as usize] =
            silk_rshift(silk_smulbb(out1_q10 as i32, quant_step_size_q16), 16);
    }

    const { assert!((NS & (NS - 1)) == 0) }; // must be power of two

    let mut n_states: usize = 1;
    rd_q25[0] = 0;
    prev_out_q10[0] = 0;
    for i in (0..order as usize).rev() {
        let rates_q5 = &ec_rates_q5[ec_ix[i] as usize..];
        let in_q10 = x_q10[i] as i32;
        for j in 0..n_states {
            let pred_q10 = silk_rshift(
                silk_smulbb(pred_coef_q8[i] as i32, prev_out_q10[j] as i32),
                8,
            );
            let res_q10 = in_q10 - pred_q10;
            let mut ind_tmp = silk_rshift(silk_smulbb(inv_quant_step_size_q6 as i32, res_q10), 16);
            ind_tmp = silk_limit(ind_tmp, -EXT, EXT - 1);
            ind[j][i] = ind_tmp as i8;

            // compute outputs for ind_tmp and ind_tmp + 1
            let mut out0_q10 = out0_q10_table[(ind_tmp + EXT) as usize] as i16;
            let mut out1_q10 = out1_q10_table[(ind_tmp + EXT) as usize] as i16;

            out0_q10 = (out0_q10 as i32 + pred_q10) as i16;
            out1_q10 = (out1_q10 as i32 + pred_q10) as i16;
            prev_out_q10[j] = out0_q10;
            prev_out_q10[j + n_states] = out1_q10;

            // compute RD for ind_tmp and ind_tmp + 1
            let (rate0_q5, rate1_q5);
            if ind_tmp + 1 >= AMP {
                if ind_tmp + 1 == AMP {
                    rate0_q5 = rates_q5[(ind_tmp + AMP) as usize] as i32;
                    rate1_q5 = 280;
                } else {
                    rate0_q5 = silk_smlabb(280 - 43 * AMP, 43, ind_tmp);
                    rate1_q5 = rate0_q5 + 43;
                }
            } else if ind_tmp <= -AMP {
                if ind_tmp == -AMP {
                    rate0_q5 = 280;
                    rate1_q5 = rates_q5[(ind_tmp + 1 + AMP) as usize] as i32;
                } else {
                    rate0_q5 = silk_smlabb(280 - 43 * AMP, -43, ind_tmp);
                    rate1_q5 = rate0_q5 - 43;
                }
            } else {
                rate0_q5 = rates_q5[(ind_tmp + AMP) as usize] as i32;
                rate1_q5 = rates_q5[(ind_tmp + 1 + AMP) as usize] as i32;
            }
            let rd_tmp_q25 = rd_q25[j];
            let diff_q10 = in_q10 - out0_q10 as i32;
            rd_q25[j] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                mu_q20,
                rate0_q5,
            );
            let diff_q10 = in_q10 - out1_q10 as i32;
            rd_q25[j + n_states] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                mu_q20,
                rate1_q5,
            );
        }

        if n_states <= NS / 2 {
            // double number of states and copy
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i] + 1;
            }
            n_states <<= 1;
            for j in n_states..NS {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            // sort lower and upper half of RD_Q25, pairwise
            for j in 0..NS {
                if rd_q25[j] > rd_q25[j + NS] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + NS];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + NS] = rd_max_q25[j];
                    // swap prev_out values
                    prev_out_q10.swap(j, j + NS);
                    ind_sort[j] = (j + NS) as i32;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + NS];
                    ind_sort[j] = j as i32;
                }
            }
            // compare the highest RD values of the winning half with the lowest one in the
            // losing half, and copy if necessary; afterwards ind_sort[] will contain the
            // indices of the NLSF_QUANT_DEL_DEC_STATES winning RD values
            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25 = 0;
                let mut ind_min_max = 0usize;
                let mut ind_max_min = 0usize;
                for j in 0..NS {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }
                // copy ind_min_max to ind_max_min
                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ NS as i32;
                rd_q25[ind_max_min] = rd_q25[ind_min_max + NS];
                prev_out_q10[ind_max_min] = prev_out_q10[ind_min_max + NS];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }
            // increment index if it comes from the upper half
            for j in 0..NS {
                ind[j][i] += silk_rshift(ind_sort[j], NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    // last sample: find winner, copy indices and return RD value
    let mut ind_tmp = 0usize;
    let mut min_q25 = i32::MAX;
    for j in 0..2 * NS {
        if min_q25 > rd_q25[j] {
            min_q25 = rd_q25[j];
            ind_tmp = j;
        }
    }
    for j in 0..order as usize {
        indices[j] = ind[ind_tmp & (NS - 1)][j];
        celt_assert!(indices[j] as i32 >= -EXT);
        celt_assert!(indices[j] as i32 <= EXT);
    }
    indices[0] += (ind_tmp >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
    celt_assert!(indices[0] as i32 <= EXT);
    silk_assert!(min_q25 >= 0);
    min_q25
}

// ---------------------------------------------------------------------------------------------
// NLSF_encode.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NLSF_encode.c:silk_NLSF_encode — NLSF vector encoder.
///
/// Writes the codebook path vector `nlsf_indices[LPC_ORDER + 1]`, replaces `p_nlsf_q15` by the
/// quantized NLSF vector and returns the RD value in Q25.
pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8],
    p_nlsf_q15: &mut [i16],
    ps_nlsf_cb: &SilkNlsfCbStruct,
    p_w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: usize,
    signal_type: i32,
) -> i32 {
    const NVM: usize = NLSF_VQ_MAX_VECTORS as usize;
    let mut res_q10 = [0i16; MLPC];
    let mut nlsf_tmp_q15 = [0i16; MLPC];
    let mut w_adj_q5 = [0i16; MLPC];
    let mut pred_q8 = [0u8; MLPC];
    let mut ec_ix = [0i16; MLPC];

    celt_assert!((0..=2).contains(&signal_type));
    celt_assert!((0..=32767).contains(&nlsf_mu_q20));

    let order = ps_nlsf_cb.order as usize;
    let n_vectors = ps_nlsf_cb.n_vectors as usize;

    // NLSF stabilization
    silk_nlsf_stabilize(p_nlsf_q15, ps_nlsf_cb.delta_min_q15, order);

    // First stage: VQ
    let mut err_q24 = [0i32; NVM];
    silk_nlsf_vq(
        &mut err_q24,
        p_nlsf_q15,
        ps_nlsf_cb.cb1_nlsf_q8,
        ps_nlsf_cb.cb1_wght_q9,
        n_vectors,
        order,
    );

    // Sort the quantization errors
    let mut temp_indices1 = [0i32; NVM];
    silk_insertion_sort_increasing(&mut err_q24, &mut temp_indices1, n_vectors, n_survivors);

    let mut rd_q25 = [0i32; NVM];
    let mut temp_indices2 = [[0i8; MLPC]; NVM];

    // Loop over survivors
    for s in 0..n_survivors {
        let ind1 = temp_indices1[s] as usize;

        // Residual after first stage
        let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[ind1 * order..(ind1 + 1) * order];
        let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[ind1 * order..(ind1 + 1) * order];
        for i in 0..order {
            nlsf_tmp_q15[i] = silk_lshift16(p_cb_element[i] as i16, 7);
            let w_tmp_q9 = p_cb_wght_q9[i] as i32;
            res_q10[i] = silk_rshift(
                silk_smulbb(p_nlsf_q15[i] as i32 - nlsf_tmp_q15[i] as i32, w_tmp_q9),
                14,
            ) as i16;
            w_adj_q5[i] =
                silk_div32_varq(p_w_q2[i] as i32, silk_smulbb(w_tmp_q9, w_tmp_q9), 21) as i16;
        }

        // Unpack entropy table indices and predictor for current CB1 index
        silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, ps_nlsf_cb, ind1 as i32);

        // Trellis quantizer
        rd_q25[s] = silk_nlsf_del_dec_quant(
            &mut temp_indices2[s],
            &res_q10,
            &w_adj_q5,
            &pred_q8,
            &ec_ix,
            ps_nlsf_cb.ec_rates_q5,
            ps_nlsf_cb.quant_step_size_q16 as i32,
            ps_nlsf_cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            ps_nlsf_cb.order,
        );

        // Add rate for first stage
        let icdf_ptr = &ps_nlsf_cb.cb1_icdf[(signal_type >> 1) as usize * n_vectors..];
        let prob_q8 = if ind1 == 0 {
            256 - icdf_ptr[ind1] as i32
        } else {
            icdf_ptr[ind1 - 1] as i32 - icdf_ptr[ind1] as i32
        };
        let bits_q7 = (8 << 7) - silk_lin2log(prob_q8);
        rd_q25[s] = silk_smlabb(rd_q25[s], bits_q7, silk_rshift(nlsf_mu_q20, 2));
    }

    // Find the lowest rate-distortion error
    let mut best_index = [0i32; 1];
    silk_insertion_sort_increasing(&mut rd_q25, &mut best_index, n_survivors, 1);

    let best = best_index[0] as usize;
    nlsf_indices[0] = temp_indices1[best] as i8;
    nlsf_indices[1..=order].copy_from_slice(&temp_indices2[best][..order]);

    // Decode
    silk_nlsf_decode(p_nlsf_q15, nlsf_indices, ps_nlsf_cb);

    rd_q25[0]
}

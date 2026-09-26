//! Port of the SILK signal-processing helpers: `silk/lin2log.c`, `silk/log2lin.c`,
//! `silk/sigm_Q15.c`, `silk/sort.c`, `silk/bwexpander.c`, `silk/bwexpander_32.c`,
//! `silk/inner_prod_aligned.c` (+ `silk_inner_prod16_c` from `silk/fixed/vector_ops_FIX.c`),
//! `silk/sum_sqr_shift.c`, `silk/interpolate.c`, `silk/biquad_alt.c`,
//! `silk/LP_variable_cutoff.c`, `silk/ana_filt_bank_1.c`, `silk/LPC_inv_pred_gain.c`,
//! `silk/LPC_fit.c` and `silk/LPC_analysis_filter.c`.
//!
//! Lengths / orders that C passes as `opus_int` are `usize` here; everything that takes part in
//! arithmetic stays `i32` like in C.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::silk::define::{
    MAX_PREDICTION_POWER_GAIN, TRANSITION_FRAMES, TRANSITION_INT_NUM, TRANSITION_NA, TRANSITION_NB,
};
use crate::silk::macros::{
    SILK_INT16_MAX, SILK_MAX_ORDER_LPC, silk_abs, silk_add_lshift32, silk_add_rshift,
    silk_add_rshift32, silk_add32, silk_clz_frac, silk_clz32, silk_div32, silk_fix_const,
    silk_inverse32_varq, silk_limit, silk_lshift, silk_lshift32, silk_max_32, silk_min, silk_mla,
    silk_mul, silk_rshift, silk_rshift_round, silk_rshift_round64, silk_rshift32, silk_sat16,
    silk_smlabb_ovflw, silk_smlalbb, silk_smlawb, silk_smmul, silk_smulbb, silk_smull, silk_smulwb,
    silk_smulww, silk_sub_sat32, silk_sub32, silk_sub32_ovflw,
};
use crate::silk::structs::SilkLpState;
use crate::silk::tables::{SILK_TRANSITION_LP_A_Q28, SILK_TRANSITION_LP_B_Q28};

// ---------------------------------------------------------------------------------------------
// lin2log.c / log2lin.c / sigm_Q15.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/lin2log.c:silk_lin2log — approximation of `128 * log2()` (very close inverse
/// of [`silk_log2lin`]).
#[must_use]
pub const fn silk_lin2log(in_lin: i32) -> i32 {
    let (lz, frac_q7) = silk_clz_frac(in_lin);
    // Piece-wise parabolic approximation
    silk_add_lshift32(
        silk_smlawb(frac_q7, silk_mul(frac_q7, 128 - frac_q7), 179),
        31 - lz,
        7,
    )
}

/// Port of silk/log2lin.c:silk_log2lin — approximation of `2^()` (very close inverse of
/// [`silk_lin2log`]).
#[must_use]
pub const fn silk_log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let mut out = silk_lshift(1, silk_rshift(in_log_q7, 7));
    let frac_q7 = in_log_q7 & 0x7F;
    if in_log_q7 < 2048 {
        // Piece-wise parabolic approximation
        out = silk_add_rshift32(
            out,
            silk_mul(
                out,
                silk_smlawb(frac_q7, silk_smulbb(frac_q7, 128 - frac_q7), -174),
            ),
            7,
        );
    } else {
        // Piece-wise parabolic approximation
        out = silk_mla(
            out,
            silk_rshift(out, 7),
            silk_smlawb(frac_q7, silk_smulbb(frac_q7, 128 - frac_q7), -174),
        );
    }
    out
}

/// `sigm_LUT_slope_Q10` (silk/sigm_Q15.c).
const SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];
/// `sigm_LUT_pos_Q15` (silk/sigm_Q15.c).
const SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];
/// `sigm_LUT_neg_Q15` (silk/sigm_Q15.c).
const SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];

/// Port of silk/sigm_Q15.c:silk_sigm_Q15 — approximate sigmoid function.
#[must_use]
pub const fn silk_sigm_q15(mut in_q5: i32) -> i32 {
    if in_q5 < 0 {
        // Negative input
        in_q5 = -in_q5;
        if in_q5 >= 6 * 32 {
            0 // Clip
        } else {
            // Linear interpolation of look up table
            let ind = silk_rshift(in_q5, 5) as usize;
            SIGM_LUT_NEG_Q15[ind] - silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5 & 0x1F)
        }
    } else if in_q5 >= 6 * 32 {
        // Positive input
        32767 // clip
    } else {
        // Linear interpolation of look up table
        let ind = silk_rshift(in_q5, 5) as usize;
        SIGM_LUT_POS_Q15[ind] + silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5 & 0x1F)
    }
}

// ---------------------------------------------------------------------------------------------
// sort.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/sort.c:silk_insertion_sort_increasing.
///
/// Sorts the first `k` of `l` values of `a` in increasing order and writes their original
/// indices to `idx[..k]`. Only the `k` first values are guaranteed correct.
pub fn silk_insertion_sort_increasing(a: &mut [i32], idx: &mut [i32], l: usize, k: usize) {
    // Safety checks
    debug_assert!(k > 0);
    debug_assert!(l > 0);
    debug_assert!(l >= k);

    // Write start indices in index vector
    for i in 0..k {
        idx[i] = i as i32;
    }

    // Sort vector elements by value, increasing order
    for i in 1..k {
        let value = a[i];
        // `j` is C's `j + 1`: the slot the value will be written to.
        let mut j = i;
        while j > 0 && value < a[j - 1] {
            a[j] = a[j - 1]; // Shift value
            idx[j] = idx[j - 1]; // Shift index
            j -= 1;
        }
        a[j] = value; // Write value
        idx[j] = i as i32; // Write index
    }

    // If less than L values are asked for, check the remaining values,
    // but only spend CPU to ensure that the K first values are correct
    for i in k..l {
        let value = a[i];
        if value < a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value < a[j - 1] {
                a[j] = a[j - 1]; // Shift value
                idx[j] = idx[j - 1]; // Shift index
                j -= 1;
            }
            a[j] = value; // Write value
            idx[j] = i as i32; // Write index
        }
    }
}

/// Port of silk/sort.c:silk_insertion_sort_decreasing_int16 (only used by the fixed-point
/// build: `silk/fixed/pitch_analysis_core_FIX.c`).
///
/// Sorts the first `k` of `l` values of `a` in decreasing order and writes their original
/// indices to `idx[..k]`. Only the `k` first values are guaranteed correct.
#[cfg(feature = "fixed-point")]
pub fn silk_insertion_sort_decreasing_int16(a: &mut [i16], idx: &mut [i32], l: usize, k: usize) {
    // Safety checks
    debug_assert!(k > 0);
    debug_assert!(l > 0);
    debug_assert!(l >= k);

    // Write start indices in index vector
    for i in 0..k {
        idx[i] = i as i32;
    }

    // Sort vector elements by value, decreasing order
    for i in 1..k {
        let value = a[i];
        // `j` is C's `j + 1`: the slot the value will be written to.
        let mut j = i;
        while j > 0 && value > a[j - 1] {
            a[j] = a[j - 1]; // Shift value
            idx[j] = idx[j - 1]; // Shift index
            j -= 1;
        }
        a[j] = value; // Write value
        idx[j] = i as i32; // Write index
    }

    // If less than L values are asked for, check the remaining values,
    // but only spend CPU to ensure that the K first values are correct
    for i in k..l {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value > a[j - 1] {
                a[j] = a[j - 1]; // Shift value
                idx[j] = idx[j - 1]; // Shift index
                j -= 1;
            }
            a[j] = value; // Write value
            idx[j] = i as i32; // Write index
        }
    }
}

/// Port of silk/sort.c:silk_insertion_sort_increasing_all_values_int16.
pub fn silk_insertion_sort_increasing_all_values_int16(a: &mut [i16], l: usize) {
    // Safety checks
    debug_assert!(l > 0);

    // Sort vector elements by value, increasing order
    for i in 1..l {
        let value = a[i];
        let mut j = i;
        while j > 0 && value < a[j - 1] {
            a[j] = a[j - 1]; // Shift value
            j -= 1;
        }
        a[j] = value; // Write value
    }
}

// ---------------------------------------------------------------------------------------------
// bwexpander.c / bwexpander_32.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/bwexpander.c:silk_bwexpander — chirp (bandwidth expand) LP AR filter.
pub fn silk_bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    // NB: Dont use silk_SMULWB, instead of silk_RSHIFT_ROUND( silk_MUL(), 16 ), below.
    // Bias in silk_SMULWB can lead to unstable filters
    for i in 0..d - 1 {
        ar[i] = silk_rshift_round(silk_mul(chirp_q16, ar[i] as i32), 16) as i16;
        chirp_q16 += silk_rshift_round(silk_mul(chirp_q16, chirp_minus_one_q16), 16);
    }
    ar[d - 1] = silk_rshift_round(silk_mul(chirp_q16, ar[d - 1] as i32), 16) as i16;
}

/// Port of silk/bwexpander_32.c:silk_bwexpander_32 — chirp (bandwidth expand) LP AR filter.
pub fn silk_bwexpander_32(ar: &mut [i32], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    for i in 0..d - 1 {
        ar[i] = silk_smulww(chirp_q16, ar[i]);
        chirp_q16 += silk_rshift_round(silk_mul(chirp_q16, chirp_minus_one_q16), 16);
    }
    ar[d - 1] = silk_smulww(chirp_q16, ar[d - 1]);
}

// ---------------------------------------------------------------------------------------------
// inner_prod_aligned.c / fixed/vector_ops_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/inner_prod_aligned.c:silk_inner_prod_aligned_scale.
#[must_use]
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: usize,
) -> i32 {
    let (in_vec1, in_vec2) = (&in_vec1[..len], &in_vec2[..len]);
    let mut sum: i32 = 0;
    for i in 0..len {
        sum = silk_add_rshift32(
            sum,
            silk_smulbb(in_vec1[i] as i32, in_vec2[i] as i32),
            scale,
        );
    }
    sum
}

/// Port of silk/fixed/vector_ops_FIX.c:silk_inner_prod16_c (the `silk_inner_prod16` macro).
#[must_use]
pub fn silk_inner_prod16(in_vec1: &[i16], in_vec2: &[i16], len: usize) -> i64 {
    let (in_vec1, in_vec2) = (&in_vec1[..len], &in_vec2[..len]);
    let mut sum: i64 = 0;
    for i in 0..len {
        sum = silk_smlalbb(sum, in_vec1[i], in_vec2[i]);
    }
    sum
}

// ---------------------------------------------------------------------------------------------
// sum_sqr_shift.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/sum_sqr_shift.c:silk_sum_sqr_shift.
///
/// Computes the number of bits to right shift the sum of squares of a vector of int16s to make
/// it fit in an int32. Returns `(energy, shift)` (the C out-parameters, in order).
#[must_use]
pub fn silk_sum_sqr_shift(x: &[i16], len: usize) -> (i32, i32) {
    let x = &x[..len];
    let len_i = len as i32;

    // `silk_ADD_RSHIFT_uint` on (int32, uint32): unsigned (wrapping) addition in C.
    let add_rshift_uint = |nrg: i32, nrg_tmp: u32, shft: i32| -> i32 {
        (nrg as u32).wrapping_add(nrg_tmp >> shft) as i32
    };

    // Do a first run with the maximum shift we could have.
    let mut shft = 31 - silk_clz32(len_i);
    // Let's be conservative with rounding and start with nrg=len.
    let mut nrg = len_i;
    let mut i = 0;
    while i + 1 < len {
        let mut nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg_tmp = silk_smlabb_ovflw(nrg_tmp as i32, x[i + 1] as i32, x[i + 1] as i32) as u32;
        nrg = add_rshift_uint(nrg, nrg_tmp, shft);
        i += 2;
    }
    if i < len {
        // One sample left to process
        let nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg = add_rshift_uint(nrg, nrg_tmp, shft);
    }
    debug_assert!(nrg >= 0);
    // Make sure the result will fit in a 32-bit signed integer with two bits of headroom.
    shft = silk_max_32(0, shft + 3 - silk_clz32(nrg));
    nrg = 0;
    let mut i = 0;
    while i + 1 < len {
        let mut nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg_tmp = silk_smlabb_ovflw(nrg_tmp as i32, x[i + 1] as i32, x[i + 1] as i32) as u32;
        nrg = add_rshift_uint(nrg, nrg_tmp, shft);
        i += 2;
    }
    if i < len {
        // One sample left to process
        let nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg = add_rshift_uint(nrg, nrg_tmp, shft);
    }
    debug_assert!(nrg >= 0);

    // Output arguments
    (nrg, shft)
}

// ---------------------------------------------------------------------------------------------
// interpolate.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/interpolate.c:silk_interpolate — interpolate two vectors.
pub fn silk_interpolate(xi: &mut [i16], x0: &[i16], x1: &[i16], ifact_q2: i32, d: usize) {
    debug_assert!(ifact_q2 >= 0);
    debug_assert!(ifact_q2 <= 4);

    for i in 0..d {
        xi[i] = silk_add_rshift(
            x0[i] as i32,
            silk_smulbb(x1[i] as i32 - x0[i] as i32, ifact_q2),
            2,
        ) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// biquad_alt.c
// ---------------------------------------------------------------------------------------------

/// Negated and split AR coefficients `(A0_L, A0_U, A1_L, A1_U)` shared by the biquads.
#[inline(always)]
const fn biquad_split_a(a_q28: &[i32]) -> (i32, i32, i32, i32) {
    (
        (-a_q28[0]) & 0x0000_3FFF,  // lower part
        silk_rshift(-a_q28[0], 14), // upper part
        (-a_q28[1]) & 0x0000_3FFF,  // lower part
        silk_rshift(-a_q28[1], 14), // upper part
    )
}

/// One sample of `silk_biquad_alt_stride1` (DIRECT FORM II TRANSPOSED); returns the output.
#[inline(always)]
const fn biquad_alt_step(
    inval: i32,
    b_q28: &[i32],
    (a0_l_q28, a0_u_q28, a1_l_q28, a1_u_q28): (i32, i32, i32, i32),
    s: &mut [i32],
) -> i16 {
    // S[ 0 ], S[ 1 ]: Q12
    let out32_q14 = silk_lshift(silk_smlawb(s[0], b_q28[0], inval), 2);

    s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
    s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
    s[0] = silk_smlawb(s[0], b_q28[1], inval);

    s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
    s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
    s[1] = silk_smlawb(s[1], b_q28[2], inval);

    // Scale back to Q0 and saturate
    silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16
}

/// Port of silk/biquad_alt.c:silk_biquad_alt_stride1 — second order ARMA filter, alternative
/// implementation. `s` is the state `[2]`.
pub fn silk_biquad_alt_stride1(
    input: &[i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    out: &mut [i16],
    len: usize,
) {
    let a = biquad_split_a(a_q28);
    let (input, out) = (&input[..len], &mut out[..len]);
    for k in 0..len {
        out[k] = biquad_alt_step(input[k] as i32, b_q28, a, s);
    }
}

/// In-place form of [`silk_biquad_alt_stride1`] (C calls it with `in == out`, which is safe
/// because each output sample is written after its input sample has been read).
pub fn silk_biquad_alt_stride1_inplace(
    buf: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    let a = biquad_split_a(a_q28);
    for v in &mut buf[..len] {
        *v = biquad_alt_step(*v as i32, b_q28, a, s);
    }
}

/// Port of silk/biquad_alt.c:silk_biquad_alt_stride2_c (the `silk_biquad_alt_stride2` macro):
/// the stride-1 biquad on two interleaved channels. `s` is the state `[4]`; `len` counts
/// sample pairs.
pub fn silk_biquad_alt_stride2(
    input: &[i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    out: &mut [i16],
    len: usize,
) {
    // DIRECT FORM II TRANSPOSED (uses 2 element state vector)
    let (a0_l_q28, a0_u_q28, a1_l_q28, a1_u_q28) = biquad_split_a(a_q28);
    let (input, out) = (&input[..2 * len], &mut out[..2 * len]);
    let mut out32_q14 = [0i32; 2];
    for k in 0..len {
        let in0 = input[2 * k] as i32;
        let in1 = input[2 * k + 1] as i32;
        // S[ 0 ], S[ 1 ], S[ 2 ], S[ 3 ]: Q12
        out32_q14[0] = silk_lshift(silk_smlawb(s[0], b_q28[0], in0), 2);
        out32_q14[1] = silk_lshift(silk_smlawb(s[2], b_q28[0], in1), 2);

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14[0], a0_l_q28), 14);
        s[2] = s[3] + silk_rshift_round(silk_smulwb(out32_q14[1], a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14[0], a0_u_q28);
        s[2] = silk_smlawb(s[2], out32_q14[1], a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], in0);
        s[2] = silk_smlawb(s[2], b_q28[1], in1);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14[0], a1_l_q28), 14);
        s[3] = silk_rshift_round(silk_smulwb(out32_q14[1], a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14[0], a1_u_q28);
        s[3] = silk_smlawb(s[3], out32_q14[1], a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], in0);
        s[3] = silk_smlawb(s[3], b_q28[2], in1);

        // Scale back to Q0 and saturate
        out[2 * k] = silk_sat16(silk_rshift(out32_q14[0] + (1 << 14) - 1, 14)) as i16;
        out[2 * k + 1] = silk_sat16(silk_rshift(out32_q14[1] + (1 << 14) - 1, 14)) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// LP_variable_cutoff.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/LP_variable_cutoff.c:silk_LP_interpolate_filter_taps (static helper):
/// interpolates the filter taps.
fn silk_lp_interpolate_filter_taps(
    b_q28: &mut [i32; TRANSITION_NB as usize],
    a_q28: &mut [i32; TRANSITION_NA as usize],
    ind: usize,
    fac_q16: i32,
) {
    let tb = &SILK_TRANSITION_LP_B_Q28;
    let ta = &SILK_TRANSITION_LP_A_Q28;
    if ind < TRANSITION_INT_NUM as usize - 1 {
        if fac_q16 > 0 {
            if fac_q16 < 32768 {
                // fac_Q16 is in range of a 16-bit int
                // Piece-wise linear interpolation of B and A
                for nb in 0..TRANSITION_NB as usize {
                    b_q28[nb] = silk_smlawb(tb[ind][nb], tb[ind + 1][nb] - tb[ind][nb], fac_q16);
                }
                for na in 0..TRANSITION_NA as usize {
                    a_q28[na] = silk_smlawb(ta[ind][na], ta[ind + 1][na] - ta[ind][na], fac_q16);
                }
            } else {
                // ( fac_Q16 - ( 1 << 16 ) ) is in range of a 16-bit int
                debug_assert!(fac_q16 - (1 << 16) == silk_sat16(fac_q16 - (1 << 16)));
                // Piece-wise linear interpolation of B and A
                for nb in 0..TRANSITION_NB as usize {
                    b_q28[nb] = silk_smlawb(
                        tb[ind + 1][nb],
                        tb[ind + 1][nb] - tb[ind][nb],
                        fac_q16 - (1i32 << 16),
                    );
                }
                for na in 0..TRANSITION_NA as usize {
                    a_q28[na] = silk_smlawb(
                        ta[ind + 1][na],
                        ta[ind + 1][na] - ta[ind][na],
                        fac_q16 - (1i32 << 16),
                    );
                }
            }
        } else {
            *b_q28 = tb[ind];
            *a_q28 = ta[ind];
        }
    } else {
        *b_q28 = tb[TRANSITION_INT_NUM as usize - 1];
        *a_q28 = ta[TRANSITION_INT_NUM as usize - 1];
    }
}

/// Port of silk/LP_variable_cutoff.c:silk_LP_variable_cutoff.
///
/// Low-pass filter with variable cutoff frequency based on piece-wise linear interpolation
/// between elliptic filters. Start by setting `ps_lp.mode <> 0`; deactivate by setting
/// `ps_lp.mode = 0`. Filters `frame[..frame_length]` in place.
pub fn silk_lp_variable_cutoff(ps_lp: &mut SilkLpState, frame: &mut [i16], frame_length: usize) {
    let mut b_q28 = [0i32; TRANSITION_NB as usize];
    let mut a_q28 = [0i32; TRANSITION_NA as usize];

    debug_assert!(ps_lp.transition_frame_no >= 0 && ps_lp.transition_frame_no <= TRANSITION_FRAMES);

    // Run filter if needed
    if ps_lp.mode != 0 {
        // Calculate index and interpolation factor for interpolation
        // (TRANSITION_INT_STEPS == 64, so C uses the shift form.)
        let mut fac_q16 = silk_lshift(TRANSITION_FRAMES - ps_lp.transition_frame_no, 16 - 6);
        let ind = silk_rshift(fac_q16, 16);
        fac_q16 -= silk_lshift(ind, 16);

        debug_assert!(ind >= 0);
        debug_assert!(ind < TRANSITION_INT_NUM);

        // Interpolate filter coefficients
        silk_lp_interpolate_filter_taps(&mut b_q28, &mut a_q28, ind as usize, fac_q16);

        // Update transition frame number for next frame
        ps_lp.transition_frame_no =
            silk_limit(ps_lp.transition_frame_no + ps_lp.mode, 0, TRANSITION_FRAMES);

        // ARMA low-pass filtering
        silk_biquad_alt_stride1_inplace(
            frame,
            &b_q28,
            &a_q28,
            &mut ps_lp.in_lp_state,
            frame_length,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// ana_filt_bank_1.c
// ---------------------------------------------------------------------------------------------

/// `A_fb1_20` (silk/ana_filt_bank_1.c): coefficient for the 2-band first-order allpass bank.
const A_FB1_20: i32 = (5394 << 1) as i16 as i32;
/// `A_fb1_21` (silk/ana_filt_bank_1.c): `(opus_int16)(20623 << 1)`.
const A_FB1_21: i32 = -24290;

/// Port of silk/ana_filt_bank_1.c:silk_ana_filt_bank_1 — split signal into two decimated
/// bands using first-order allpass filters. `s` is the state `[2]`.
pub fn silk_ana_filt_bank_1(
    input: &[i16],
    s: &mut [i32],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: usize,
) {
    let n2 = n >> 1;

    // Internal variables and state are in Q10 format
    for k in 0..n2 {
        // Convert to Q10
        let mut in32 = silk_lshift(input[2 * k] as i32, 10);

        // All-pass section for even input sample
        let mut y = silk_sub32(in32, s[0]);
        let mut x = silk_smlawb(y, y, A_FB1_21);
        let out_1 = silk_add32(s[0], x);
        s[0] = silk_add32(in32, x);

        // Convert to Q10
        in32 = silk_lshift(input[2 * k + 1] as i32, 10);

        // All-pass section for odd input sample, and add to output of previous section
        y = silk_sub32(in32, s[1]);
        x = silk_smulwb(y, A_FB1_20);
        let out_2 = silk_add32(s[1], x);
        s[1] = silk_add32(in32, x);

        // Add/subtract, convert back to int16 and store to output
        out_l[k] = silk_sat16(silk_rshift_round(silk_add32(out_2, out_1), 11)) as i16;
        out_h[k] = silk_sat16(silk_rshift_round(silk_sub32(out_2, out_1), 11)) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// LPC_inv_pred_gain.c
// ---------------------------------------------------------------------------------------------

/// `QA` (silk/LPC_inv_pred_gain.c).
const QA: i32 = 24;
/// `A_LIMIT` (silk/LPC_inv_pred_gain.c).
const A_LIMIT: i32 = silk_fix_const(0.99975, QA);
/// `SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 )`: the constant is a *float*
/// multiplied by `(opus_int64)1 << 30` (converted to float), then `+ 0.5` in double.
const INV_MAX_PRED_GAIN_Q30: i32 =
    (((1.0f32 / MAX_PREDICTION_POWER_GAIN) * (1i64 << 30) as f32) as f64 + 0.5) as i32;

/// `MUL32_FRAC_Q` (silk/LPC_inv_pred_gain.c).
#[inline(always)]
const fn mul32_frac_q(a32: i32, b32: i32, q: i32) -> i32 {
    silk_rshift_round64(silk_smull(a32, b32), q) as i32
}

/// Port of silk/LPC_inv_pred_gain.c:LPC_inverse_pred_gain_QA_c (static helper).
///
/// Computes the inverse of the LPC prediction gain and tests if the LPC coefficients are stable
/// (all poles within unit circle). Returns the inverse prediction gain in energy domain, Q30.
fn lpc_inverse_pred_gain_qa_c(a_qa: &mut [i32; SILK_MAX_ORDER_LPC], order: usize) -> i32 {
    let mut inv_gain_q30 = silk_fix_const(1.0, 30);
    let mut k = order - 1;
    while k > 0 {
        // Check for stability
        if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
            return 0;
        }

        // Set RC equal to negated AR coef
        let rc_q31 = -silk_lshift(a_qa[k], 31 - QA);

        // rc_mult1_Q30 range: [ 1 : 2^30 ]
        let rc_mult1_q30 = silk_sub32(silk_fix_const(1.0, 30), silk_smmul(rc_q31, rc_q31));
        debug_assert!(rc_mult1_q30 > (1 << 15)); // reduce A_LIMIT if fails
        debug_assert!(rc_mult1_q30 <= (1 << 30));

        // Update inverse gain
        // invGain_Q30 range: [ 0 : 2^30 ]
        inv_gain_q30 = silk_lshift(silk_smmul(inv_gain_q30, rc_mult1_q30), 2);
        debug_assert!(inv_gain_q30 >= 0);
        debug_assert!(inv_gain_q30 <= (1 << 30));
        if inv_gain_q30 < INV_MAX_PRED_GAIN_Q30 {
            return 0;
        }

        // rc_mult2 range: [ 2^30 : silk_int32_MAX ]
        let mult2q = 32 - silk_clz32(silk_abs(rc_mult1_q30));
        let rc_mult2 = silk_inverse32_varq(rc_mult1_q30, mult2q + 30);

        // Update AR coefficient
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];
            let tmp64 = silk_rshift_round64(
                silk_smull(
                    silk_sub_sat32(tmp1, mul32_frac_q(tmp2, rc_q31, 31)),
                    rc_mult2,
                ),
                mult2q,
            );
            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            a_qa[n] = tmp64 as i32;
            let tmp64 = silk_rshift_round64(
                silk_smull(
                    silk_sub_sat32(tmp2, mul32_frac_q(tmp1, rc_q31, 31)),
                    rc_mult2,
                ),
                mult2q,
            );
            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            a_qa[k - n - 1] = tmp64 as i32;
        }
        k -= 1;
    }

    // Check for stability
    if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
        return 0;
    }

    // Set RC equal to negated AR coef
    let rc_q31 = -silk_lshift(a_qa[0], 31 - QA);

    // Range: [ 1 : 2^30 ]
    let rc_mult1_q30 = silk_sub32(silk_fix_const(1.0, 30), silk_smmul(rc_q31, rc_q31));

    // Update inverse gain
    // Range: [ 0 : 2^30 ]
    inv_gain_q30 = silk_lshift(silk_smmul(inv_gain_q30, rc_mult1_q30), 2);
    debug_assert!(inv_gain_q30 >= 0);
    debug_assert!(inv_gain_q30 <= (1 << 30));
    if inv_gain_q30 < INV_MAX_PRED_GAIN_Q30 {
        return 0;
    }

    inv_gain_q30
}

/// Port of silk/LPC_inv_pred_gain.c:silk_LPC_inverse_pred_gain_c (the
/// `silk_LPC_inverse_pred_gain` macro), for input in Q12 domain.
///
/// Returns the inverse prediction gain in energy domain, Q30 (0 if unstable).
#[must_use]
pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    let mut atmp_qa = [0i32; SILK_MAX_ORDER_LPC];
    let mut dc_resp: i32 = 0;

    // Increase Q domain of the AR coefficients
    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        atmp_qa[k] = silk_lshift32(a_q12[k] as i32, QA - 12);
    }
    // If the DC is unstable, we don't even need to do the full calculations
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa_c(&mut atmp_qa, order)
}

// ---------------------------------------------------------------------------------------------
// LPC_fit.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/LPC_fit.c:silk_LPC_fit — convert int32 coefficients to int16 coefs and make
/// sure there's no wrap-around.
pub fn silk_lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: i32, qin: i32, d: usize) {
    let mut idx = 0usize;

    // Limit the maximum absolute value of the prediction coefficients, so that they'll fit in
    // int16
    let mut i = 0;
    while i < 10 {
        // Find maximum absolute value and its index
        let mut maxabs = 0;
        for k in 0..d {
            let absval = silk_abs(a_qin[k]);
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = silk_rshift_round(maxabs, qin - qout);

        if maxabs > SILK_INT16_MAX {
            // Reduce magnitude of prediction coefficients
            maxabs = silk_min(maxabs, 163838); // ( silk_int32_MAX >> 14 ) + silk_int16_MAX = 163838
            let chirp_q16 = silk_fix_const(0.999, 16)
                - silk_div32(
                    silk_lshift(maxabs - SILK_INT16_MAX, 14),
                    silk_rshift32(silk_mul(maxabs, idx as i32 + 1), 2),
                );
            silk_bwexpander_32(a_qin, d, chirp_q16);
        } else {
            break;
        }
        i += 1;
    }

    if i == 10 {
        // Reached the last iteration, clip the coefficients
        for k in 0..d {
            a_qout[k] = silk_sat16(silk_rshift_round(a_qin[k], qin - qout)) as i16;
            a_qin[k] = silk_lshift(a_qout[k] as i32, qin - qout);
        }
    } else {
        for k in 0..d {
            a_qout[k] = silk_rshift_round(a_qin[k], qin - qout) as i16;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LPC_analysis_filter.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/LPC_analysis_filter.c:silk_LPC_analysis_filter — variable order MA prediction
/// error filter. The filter always starts with zero state; the first `d` output samples are
/// set to zero.
pub fn silk_lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16], len: usize, d: usize) {
    debug_assert!(d >= 6);
    debug_assert!((d & 1) == 0);
    debug_assert!(d <= len);
    // (FIXED_POINT && USE_CELT_FIR) branch not ported: USE_CELT_FIR is 0.

    let (out, input, b) = (&mut out[..len], &input[..len], &b[..d]);

    // Perf: the C loop computes, for each output `ix`, the prediction
    // `sum_j in[ ix - 1 - j ] * B[ j ]` with a chain of `silk_SMLABB_ovflw` (wrapping) steps.
    // Wrapping i32 addition is associative, so the terms are summed here tap by tap over a
    // block of outputs instead (tap-outer, sample-inner), which vectorizes; the result is
    // bit-identical. `acc[k]` holds the prediction of output `start + k`.
    const BLOCK: usize = 64;
    let mut start = d;
    while start < len {
        let n = (len - start).min(BLOCK);
        let mut acc = [0i32; BLOCK];
        let acc = &mut acc[..n];
        for (j, &bj) in b.iter().enumerate() {
            // in_ptr[ -j ] for the outputs start..start + n
            let x = &input[start - 1 - j..start - 1 - j + n];
            for (a, &xv) in acc.iter_mut().zip(x) {
                *a = a.wrapping_add(silk_smulbb(xv as i32, bj as i32));
            }
        }
        for ((o, &pred), &xv) in out[start..start + n]
            .iter_mut()
            .zip(acc.iter())
            .zip(&input[start..start + n])
        {
            // Subtract prediction
            let out32_q12 = silk_sub32_ovflw(silk_lshift(xv as i32, 12), pred);

            // Scale to Q0
            let out32 = silk_rshift_round(out32_q12, 12);

            // Saturate output
            *o = silk_sat16(out32) as i16;
        }
        start += n;
    }

    // Set first d output samples to zero
    out[..d].fill(0);
}

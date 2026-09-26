//! Port of `silk/fixed/pitch_analysis_core_FIX.c`: the fixed-point pitch analyser.

use crate::celt::pitch::celt_pitch_xcorr;
use crate::silk::fixed::sigproc::silk_inner_prod_aligned;
use crate::silk::macros::{
    silk_add_lshift32, silk_add_sat16, silk_add_sat32, silk_add32, silk_clz32, silk_div32,
    silk_div32_16, silk_div32_varq, silk_fix_const, silk_limit, silk_limit_int, silk_lshift,
    silk_max_int, silk_min_int, silk_mul, silk_rshift, silk_smlawb, silk_smulbb, silk_smulwb,
};
use crate::silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use crate::silk::sigproc::{
    silk_insertion_sort_decreasing_int16, silk_lin2log, silk_sum_sqr_shift,
};
use crate::silk::tables::{
    PE_D_SRCH_LENGTH, PE_FLATCONTOUR_BIAS, PE_LTP_MEM_LENGTH_MS, PE_MAX_FRAME_LENGTH_MS,
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2, PE_NB_CBKS_STAGE2_10MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX, PE_NB_STAGE3_LAGS,
    PE_PREVLAG_BIAS, PE_SHORTLAG_BIAS, PE_SUBFR_LENGTH_MS, SILK_CB_LAGS_STAGE2,
    SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS,
    SILK_LAG_RANGE_STAGE3, SILK_LAG_RANGE_STAGE3_10_MS, SILK_NB_CBK_SEARCHS_STAGE3,
    SILK_PE_MAX_COMPLEX, SILK_PE_MIN_COMPLEX,
};

const SCRATCH_SIZE: usize = 22;
const SF_LENGTH_4KHZ: i32 = PE_SUBFR_LENGTH_MS * 4;
const SF_LENGTH_8KHZ: i32 = PE_SUBFR_LENGTH_MS * 8;
const MIN_LAG_4KHZ: i32 = PE_MIN_LAG_MS * 4;
const MIN_LAG_8KHZ: i32 = PE_MIN_LAG_MS * 8;
const MAX_LAG_4KHZ: i32 = PE_MAX_LAG_MS * 4;
const MAX_LAG_8KHZ: i32 = PE_MAX_LAG_MS * 8 - 1;
const CSTRIDE_4KHZ: i32 = MAX_LAG_4KHZ + 1 - MIN_LAG_4KHZ;
const CSTRIDE_8KHZ: i32 = MAX_LAG_8KHZ + 3 - (MIN_LAG_8KHZ - 2);
const D_COMP_MIN: i32 = MIN_LAG_8KHZ - 3;
const D_COMP_MAX: i32 = MAX_LAG_8KHZ + 4;
const D_COMP_STRIDE: i32 = D_COMP_MAX - D_COMP_MIN;

const PMNS: usize = PE_MAX_NB_SUBFR as usize;
const NB_CBKS3_MAX: usize = PE_NB_CBKS_STAGE3_MAX as usize;
const NB_ST3_LAGS: usize = PE_NB_STAGE3_LAGS as usize;

/// `silk_pe_stage3_vals`.
pub type PeStage3Vals = [i32; NB_ST3_LAGS];

/// Storage for the C `ALLOC( energies_st3 / cross_corr_st3, nb_subfr * nb_cbk_search,
/// silk_pe_stage3_vals )` arrays, indexed `k * nb_cbk_search + j`
/// (`matrix_ptr( x, k, j, nb_cbk_search )`).
pub type Stage3Array = [PeStage3Vals; PMNS * NB_CBKS3_MAX];

/// Row-major view of a C lag codebook (`matrix_ptr( Lag_CB_ptr, row, col, cbk_size )`).
#[derive(Clone, Copy)]
struct LagCb<'a> {
    data: &'a [i8],
    cbk_size: usize,
}

impl LagCb<'_> {
    #[inline(always)]
    const fn at(&self, row: usize, col: usize) -> i32 {
        self.data[row * self.cbk_size + col] as i32
    }
}

/// Port of `silk/fixed/pitch_analysis_core_FIX.c:silk_pitch_analysis_core` — fixed-point core
/// pitch analysis.
///
/// Returns the voicing estimate: 0 voiced, 1 unvoiced. `frame_unscaled` holds
/// `PE_FRAME_LENGTH_MS * fs_khz` samples; `pitch_out` has `nb_subfr` entries; `ltp_corr_q15`
/// is the normalized correlation (input: value from the previous frame).
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_pitch_analysis_core(
    frame_unscaled: &[i16],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr_q15: &mut i32,
    mut prev_lag: i32,
    search_thres1_q16: i32,
    search_thres2_q13: i32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
) -> i32 {
    const FL_MAX: usize = (PE_MAX_FRAME_LENGTH_MS * 16) as usize;
    const FL8_MAX: usize = (PE_MAX_FRAME_LENGTH_MS * 8) as usize;
    const FL4_MAX: usize = (PE_MAX_FRAME_LENGTH_MS * 4) as usize;
    let mut filt_state = [0i32; 6];
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH as usize];
    let mut cc = [0i32; PE_NB_CBKS_STAGE2_EXT as usize];

    // Check for valid sampling frequency
    debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);

    // Check for valid complexity setting
    debug_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    debug_assert!(complexity <= SILK_PE_MAX_COMPLEX);

    debug_assert!((0..=(1 << 16)).contains(&search_thres1_q16));
    debug_assert!((0..=(1 << 13)).contains(&search_thres2_q13));

    // Set up frame lengths max / min lag for the sampling frequency
    let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz;
    let frame_length_4khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let frame_length_8khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz;
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let max_lag = PE_MAX_LAG_MS * fs_khz - 1;
    let fl = frame_length as usize;
    let fl4 = frame_length_4khz as usize;
    let fl8 = frame_length_8khz as usize;

    // Downscale input if necessary
    let (energy, mut shift) = silk_sum_sqr_shift(frame_unscaled, fl);
    shift += 3 - silk_clz32(energy); // at least two bits headroom
    let mut frame_scaled = [0i16; FL_MAX];
    let frame: &[i16] = if shift > 0 {
        shift = silk_rshift(shift + 1, 1);
        for i in 0..fl {
            frame_scaled[i] = silk_rshift(frame_unscaled[i] as i32, shift) as i16;
        }
        &frame_scaled[..fl]
    } else {
        &frame_unscaled[..fl]
    };

    // Resample from input sampled at Fs_kHz to 8 kHz
    let mut frame_8khz_buf = [0i16; FL8_MAX];
    let frame_8khz: &[i16] = if fs_khz == 16 {
        filt_state[..2].fill(0);
        silk_resampler_down2(&mut filt_state, &mut frame_8khz_buf, frame, frame_length);
        &frame_8khz_buf[..fl8]
    } else if fs_khz == 12 {
        filt_state.fill(0);
        silk_resampler_down2_3(&mut filt_state, &mut frame_8khz_buf, frame, frame_length);
        &frame_8khz_buf[..fl8]
    } else {
        debug_assert!(fs_khz == 8);
        frame
    };

    // Decimate again to 4 kHz
    filt_state[..2].fill(0); // Set state to zero
    let mut frame_4khz = [0i16; FL4_MAX];
    silk_resampler_down2(
        &mut filt_state,
        &mut frame_4khz,
        frame_8khz,
        frame_length_8khz,
    );
    let frame_4khz = &mut frame_4khz[..fl4];

    // Low-pass filter
    for i in (1..fl4).rev() {
        frame_4khz[i] = silk_add_sat16(frame_4khz[i], frame_4khz[i - 1] as i32);
    }

    // *****************************************************************************
    // FIRST STAGE, operating in 4 khz
    // *****************************************************************************
    let mut c = [0i16; PMNS * CSTRIDE_8KHZ as usize];
    let mut xcorr32 = [0i32; (MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1) as usize];
    let cs4 = CSTRIDE_4KHZ as usize;
    c[..(nb_subfr >> 1) as usize * cs4].fill(0);
    let mut target_ptr = silk_lshift(SF_LENGTH_4KHZ, 2) as usize;
    for k in 0..(nb_subfr >> 1) as usize {
        // Check that we are within range of the array
        debug_assert!(target_ptr + SF_LENGTH_8KHZ as usize <= fl4);

        let mut basis_ptr = target_ptr - MIN_LAG_4KHZ as usize;

        let target = &frame_4khz[target_ptr..];
        celt_pitch_xcorr(
            target,
            &frame_4khz[target_ptr - MAX_LAG_4KHZ as usize..],
            &mut xcorr32,
            SF_LENGTH_8KHZ as usize,
            (MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1) as usize,
        );

        // Calculate first vector products before loop
        let mut cross_corr = xcorr32[(MAX_LAG_4KHZ - MIN_LAG_4KHZ) as usize];
        let mut normalizer = silk_inner_prod_aligned(target, target, SF_LENGTH_8KHZ as usize);
        normalizer = silk_add32(
            normalizer,
            silk_inner_prod_aligned(
                &frame_4khz[basis_ptr..],
                &frame_4khz[basis_ptr..],
                SF_LENGTH_8KHZ as usize,
            ),
        );
        normalizer = silk_add32(normalizer, silk_smulbb(SF_LENGTH_8KHZ, 4000));

        c[k * cs4] = silk_div32_varq(cross_corr, normalizer, 13 + 1) as i16; // Q13

        // From now on normalizer is computed recursively
        for d in MIN_LAG_4KHZ + 1..=MAX_LAG_4KHZ {
            basis_ptr -= 1;

            cross_corr = xcorr32[(MAX_LAG_4KHZ - d) as usize];

            // Add contribution of new sample and remove contribution from oldest sample
            let b0 = frame_4khz[basis_ptr] as i32;
            let bl = frame_4khz[basis_ptr + SF_LENGTH_8KHZ as usize] as i32;
            normalizer = silk_add32(normalizer, silk_smulbb(b0, b0) - silk_smulbb(bl, bl));

            c[k * cs4 + (d - MIN_LAG_4KHZ) as usize] =
                silk_div32_varq(cross_corr, normalizer, 13 + 1) as i16; // Q13
        }
        // Update target pointer
        target_ptr += SF_LENGTH_8KHZ as usize;
    }

    // Combine two subframes into single correlation measure and apply short-lag bias
    if nb_subfr == PE_MAX_NB_SUBFR {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let ix = (i - MIN_LAG_4KHZ) as usize;
            let mut sum = c[ix] as i32 + c[cs4 + ix] as i32; // Q14
            sum = silk_smlawb(sum, sum, silk_lshift(-i, 4)); // Q14
            c[ix] = sum as i16; // Q14
        }
    } else {
        // Only short-lag bias
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let ix = (i - MIN_LAG_4KHZ) as usize;
            let mut sum = silk_lshift(c[ix] as i32, 1); // Q14
            sum = silk_smlawb(sum, sum, silk_lshift(-i, 4)); // Q14
            c[ix] = sum as i16; // Q14
        }
    }

    // Sort
    let mut length_d_srch = silk_add_lshift32(4, complexity, 1);
    debug_assert!(3 * length_d_srch <= PE_D_SRCH_LENGTH);
    silk_insertion_sort_decreasing_int16(&mut c, &mut d_srch, cs4, length_d_srch as usize);

    // Escape if correlation is very low already here
    let cmax = c[0] as i32; // Q14
    if cmax < silk_fix_const(0.2, 14) {
        pitch_out[..nb_subfr as usize].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = silk_smulwb(search_thres1_q16, cmax);
    for i in 0..length_d_srch as usize {
        // Convert to 8 kHz indices for the sorted correlation that exceeds the threshold
        if c[i] as i32 > threshold {
            d_srch[i] = silk_lshift(d_srch[i] + MIN_LAG_4KHZ, 1);
        } else {
            length_d_srch = i as i32;
            break;
        }
    }
    debug_assert!(length_d_srch > 0);

    let mut d_comp = [0i16; D_COMP_STRIDE as usize];
    let dc = |i: i32| (i - D_COMP_MIN) as usize;
    for i in D_COMP_MIN..D_COMP_MAX {
        d_comp[dc(i)] = 0;
    }
    for i in 0..length_d_srch as usize {
        d_comp[dc(d_srch[i])] = 1;
    }

    // Convolution
    for i in (MIN_LAG_8KHZ..D_COMP_MAX).rev() {
        d_comp[dc(i)] += d_comp[dc(i - 1)] + d_comp[dc(i - 2)];
    }

    length_d_srch = 0;
    for i in MIN_LAG_8KHZ..MAX_LAG_8KHZ + 1 {
        if d_comp[dc(i + 1)] > 0 {
            d_srch[length_d_srch as usize] = i;
            length_d_srch += 1;
        }
    }

    // Convolution
    for i in (MIN_LAG_8KHZ..D_COMP_MAX).rev() {
        d_comp[dc(i)] += d_comp[dc(i - 1)] + d_comp[dc(i - 2)] + d_comp[dc(i - 3)];
    }

    let mut length_d_comp = 0usize;
    for i in MIN_LAG_8KHZ..D_COMP_MAX {
        if d_comp[dc(i)] > 0 {
            d_comp[length_d_comp] = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    // *********************************************************************************
    // SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    // *********************************************************************************

    // *********************************************************************************
    // Find energy of each subframe projected onto its history, for a range of delays
    // *********************************************************************************
    let cs8 = CSTRIDE_8KHZ as usize;
    c[..nb_subfr as usize * cs8].fill(0);

    let mut target_ptr = (PE_LTP_MEM_LENGTH_MS * 8) as usize;
    let sf8 = SF_LENGTH_8KHZ as usize;
    for k in 0..nb_subfr as usize {
        // Check that we are within range of the array
        debug_assert!(target_ptr + sf8 <= fl8);

        let target = &frame_8khz[target_ptr..target_ptr + sf8];
        let energy_target = silk_add32(silk_inner_prod_aligned(target, target, sf8), 1);
        for j in 0..length_d_comp {
            let d = d_comp[j] as i32;
            let basis = &frame_8khz[target_ptr - d as usize..];

            let cross_corr = silk_inner_prod_aligned(target, basis, sf8);
            let ix = k * cs8 + (d - (MIN_LAG_8KHZ - 2)) as usize;
            if cross_corr > 0 {
                let energy_basis = silk_inner_prod_aligned(basis, basis, sf8);
                c[ix] = silk_div32_varq(cross_corr, silk_add32(energy_target, energy_basis), 13 + 1)
                    as i16; // Q13
            } else {
                c[ix] = 0;
            }
        }
        target_ptr += sf8;
    }

    // search over lag range and lags codebook
    // scale factor for lag codebook, as a function of center lag

    let mut ccmax = i32::MIN;
    let mut ccmax_b = i32::MIN;

    let mut cbimax = 0usize; // To avoid returning undefined lag values
    let mut lag = -1i32; // To check if lag with strong enough correlation has been found

    let prev_lag_log2_q7 = if prev_lag > 0 {
        if fs_khz == 12 {
            prev_lag = silk_div32_16(silk_lshift(prev_lag, 1), 3);
        } else if fs_khz == 16 {
            prev_lag = silk_rshift(prev_lag, 1);
        }
        silk_lin2log(prev_lag)
    } else {
        0
    };
    debug_assert!(search_thres2_q13 == search_thres2_q13 as i16 as i32);

    // Set up stage 2 codebook based on number of subframes
    let (lag_cb, nb_cbk_search) = if nb_subfr == PE_MAX_NB_SUBFR {
        let cb = LagCb {
            data: SILK_CB_LAGS_STAGE2.as_flattened(),
            cbk_size: PE_NB_CBKS_STAGE2_EXT as usize,
        };
        if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            // If input is 8 khz use a larger codebook here because it is last stage
            (cb, PE_NB_CBKS_STAGE2_EXT as usize)
        } else {
            (cb, PE_NB_CBKS_STAGE2 as usize)
        }
    } else {
        (
            LagCb {
                data: SILK_CB_LAGS_STAGE2_10_MS.as_flattened(),
                cbk_size: PE_NB_CBKS_STAGE2_10MS as usize,
            },
            PE_NB_CBKS_STAGE2_10MS as usize,
        )
    };
    let shortlag_bias_q13 = nb_subfr * silk_fix_const(PE_SHORTLAG_BIAS as f64, 13);
    let prevlag_bias_q13 = nb_subfr * silk_fix_const(PE_PREVLAG_BIAS as f64, 13);

    for k in 0..length_d_srch as usize {
        let d = d_srch[k];
        for j in 0..nb_cbk_search {
            cc[j] = 0;
            for i in 0..nb_subfr as usize {
                // Try all codebooks
                let d_subfr = d + lag_cb.at(i, j);
                cc[j] += c[i * cs8 + (d_subfr - (MIN_LAG_8KHZ - 2)) as usize] as i32;
            }
        }
        // Find best codebook
        let mut ccmax_new = i32::MIN;
        let mut cbimax_new = 0usize;
        for i in 0..nb_cbk_search {
            if cc[i] > ccmax_new {
                ccmax_new = cc[i];
                cbimax_new = i;
            }
        }

        // Bias towards shorter lags
        let lag_log2_q7 = silk_lin2log(d); // Q7
        debug_assert!(lag_log2_q7 == lag_log2_q7 as i16 as i32);
        debug_assert!(shortlag_bias_q13 == shortlag_bias_q13 as i16 as i32);
        let mut ccmax_new_b =
            ccmax_new - silk_rshift(silk_smulbb(shortlag_bias_q13, lag_log2_q7), 7); // Q13

        // Bias towards previous lag
        debug_assert!(prevlag_bias_q13 == prevlag_bias_q13 as i16 as i32);
        if prev_lag > 0 {
            let mut delta_lag_log2_sqr_q7 = lag_log2_q7 - prev_lag_log2_q7;
            debug_assert!(delta_lag_log2_sqr_q7 == delta_lag_log2_sqr_q7 as i16 as i32);
            delta_lag_log2_sqr_q7 =
                silk_rshift(silk_smulbb(delta_lag_log2_sqr_q7, delta_lag_log2_sqr_q7), 7);
            let mut prev_lag_bias_q13 =
                silk_rshift(silk_smulbb(prevlag_bias_q13, *ltp_corr_q15), 15); // Q13
            prev_lag_bias_q13 = silk_div32(
                silk_mul(prev_lag_bias_q13, delta_lag_log2_sqr_q7),
                delta_lag_log2_sqr_q7 + silk_fix_const(0.5, 7),
            );
            ccmax_new_b -= prev_lag_bias_q13; // Q13
        }

        if ccmax_new_b > ccmax_b                                   // Find maximum biased correlation
            && ccmax_new > silk_smulbb(nb_subfr, search_thres2_q13) // Correlation needs to be high enough to be voiced
            && SILK_CB_LAGS_STAGE2[0][cbimax_new] as i32 <= MIN_LAG_8KHZ
        // Lag must be in range
        {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d;
            cbimax = cbimax_new;
        }
    }

    if lag == -1 {
        // No suitable candidate found
        pitch_out[..nb_subfr as usize].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    // Output normalized correlation
    *ltp_corr_q15 = silk_lshift(silk_div32_16(ccmax, nb_subfr), 2);
    debug_assert!(*ltp_corr_q15 >= 0);

    if fs_khz > 8 {
        // Search in original signal

        let cbimax_old = cbimax;
        // Compensate for decimation
        debug_assert!(lag == lag as i16 as i32);
        if fs_khz == 12 {
            lag = silk_rshift(silk_smulbb(lag, 3), 1);
        } else if fs_khz == 16 {
            lag = silk_lshift(lag, 1);
        } else {
            lag = silk_smulbb(lag, 3);
        }

        lag = silk_limit_int(lag, min_lag, max_lag);
        let start_lag = silk_max_int(lag - 2, min_lag);
        let end_lag = silk_min_int(lag + 2, max_lag);
        let mut lag_new = lag; // to avoid undefined lag
        cbimax = 0; // to avoid undefined lag

        ccmax = i32::MIN;
        // pitch lags according to second stage
        for k in 0..nb_subfr as usize {
            pitch_out[k] = lag + 2 * SILK_CB_LAGS_STAGE2[k][cbimax_old] as i32;
        }

        // Set up codebook parameters according to complexity setting and frame length
        let (nb_cbk_search, lag_cb) = if nb_subfr == PE_MAX_NB_SUBFR {
            (
                SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as usize,
                LagCb {
                    data: SILK_CB_LAGS_STAGE3.as_flattened(),
                    cbk_size: PE_NB_CBKS_STAGE3_MAX as usize,
                },
            )
        } else {
            (
                PE_NB_CBKS_STAGE3_10MS as usize,
                LagCb {
                    data: SILK_CB_LAGS_STAGE3_10_MS.as_flattened(),
                    cbk_size: PE_NB_CBKS_STAGE3_10MS as usize,
                },
            )
        };

        // Calculate the correlations and energies needed in stage 3
        let mut energies_st3: Stage3Array = [[0; NB_ST3_LAGS]; PMNS * NB_CBKS3_MAX];
        let mut cross_corr_st3: Stage3Array = [[0; NB_ST3_LAGS]; PMNS * NB_CBKS3_MAX];
        silk_p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
        );
        silk_p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
        );

        debug_assert!(lag == lag as i16 as i32);
        let contour_bias_q15 = silk_div32_16(silk_fix_const(PE_FLATCONTOUR_BIAS as f64, 15), lag);

        let target_ptr = (PE_LTP_MEM_LENGTH_MS * fs_khz) as usize;
        let target = &frame[target_ptr..];
        let energy_target = silk_add32(
            silk_inner_prod_aligned(target, target, (nb_subfr * sf_length) as usize),
            1,
        );
        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search {
                let mut cross_corr = 0i32;
                let mut energy = energy_target;
                for k in 0..nb_subfr as usize {
                    cross_corr = silk_add32(
                        cross_corr,
                        cross_corr_st3[k * nb_cbk_search + j][lag_counter],
                    );
                    energy = silk_add32(energy, energies_st3[k * nb_cbk_search + j][lag_counter]);
                    debug_assert!(energy >= 0);
                }
                let ccmax_new = if cross_corr > 0 {
                    let v = silk_div32_varq(cross_corr, energy, 13 + 1); // Q13
                    // Reduce depending on flatness of contour
                    let diff = i16::MAX as i32 - silk_mul(contour_bias_q15, j as i32); // Q15
                    debug_assert!(diff == diff as i16 as i32);
                    silk_smulwb(v, diff) // Q14
                } else {
                    0
                };

                if ccmax_new > ccmax && (d + SILK_CB_LAGS_STAGE3[0][j] as i32) <= max_lag {
                    ccmax = ccmax_new;
                    lag_new = d;
                    cbimax = j;
                }
            }
        }

        for k in 0..nb_subfr as usize {
            pitch_out[k] = lag_new + lag_cb.at(k, cbimax);
            pitch_out[k] = silk_limit(pitch_out[k], min_lag, PE_MAX_LAG_MS * fs_khz);
        }
        *lag_index = (lag_new - min_lag) as i16;
        *contour_index = cbimax as i8;
    } else {
        // Fs_kHz == 8
        // Save Lags
        for k in 0..nb_subfr as usize {
            pitch_out[k] = lag + lag_cb.at(k, cbimax);
            pitch_out[k] = silk_limit(pitch_out[k], MIN_LAG_8KHZ, PE_MAX_LAG_MS * 8);
        }
        *lag_index = (lag - MIN_LAG_8KHZ) as i16;
        *contour_index = cbimax as i8;
    }
    debug_assert!(*lag_index >= 0);
    // return as voiced
    0
}

/// Lag range table and codebook of stage 3 (`Lag_range_ptr`, `Lag_CB_ptr`, `nb_cbk_search`).
fn stage3_tables(nb_subfr: i32, complexity: i32) -> (&'static [[i8; 2]], LagCb<'static>, usize) {
    debug_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    debug_assert!(complexity <= SILK_PE_MAX_COMPLEX);
    if nb_subfr == PE_MAX_NB_SUBFR {
        (
            &SILK_LAG_RANGE_STAGE3[complexity as usize],
            LagCb {
                data: SILK_CB_LAGS_STAGE3.as_flattened(),
                cbk_size: PE_NB_CBKS_STAGE3_MAX as usize,
            },
            SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as usize,
        )
    } else {
        debug_assert!(nb_subfr == PE_MAX_NB_SUBFR >> 1);
        (
            &SILK_LAG_RANGE_STAGE3_10_MS,
            LagCb {
                data: SILK_CB_LAGS_STAGE3_10_MS.as_flattened(),
                cbk_size: PE_NB_CBKS_STAGE3_10MS as usize,
            },
            PE_NB_CBKS_STAGE3_10MS as usize,
        )
    }
}

/// Port of the static `pitch_analysis_core_FIX.c:silk_P_Ana_calc_corr_st3` — calculates the
/// correlations used in the stage 3 search. In order to cover the whole lag codebook for all
/// the searched offset lags (lag +- 2), the following correlations are needed in each sub
/// frame:
///
/// sf1: lag range [-8,...,7] total 16 correlations
/// sf2: lag range [-4,...,4] total 9 correlations
/// sf3: lag range [-3,....4] total 8 correltions
/// sf4: lag range [-6,....8] total 15 correlations
///
/// In total 48 correlations. The direct implementation computed in worst case 4*12*5 = 240
/// correlations, but more likely around 120.
pub fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut Stage3Array,
    frame: &[i16],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut scratch_mem = [0i32; SCRATCH_SIZE];
    let mut xcorr32 = [0i32; SCRATCH_SIZE];
    let (lag_range, lag_cb, nb_cbk_search) = stage3_tables(nb_subfr, complexity);

    let mut target_ptr = silk_lshift(sf_length, 2) as usize; // Pointer to middle of frame
    for k in 0..nb_subfr as usize {
        let mut lag_counter = 0usize;

        // Calculate the correlations for each subframe
        let lag_low = lag_range[k][0] as i32;
        let lag_high = lag_range[k][1] as i32;
        debug_assert!(lag_high - lag_low < SCRATCH_SIZE as i32);
        celt_pitch_xcorr(
            &frame[target_ptr..],
            &frame[target_ptr - (start_lag + lag_high) as usize..],
            &mut xcorr32,
            sf_length as usize,
            (lag_high - lag_low + 1) as usize,
        );
        for j in lag_low..=lag_high {
            debug_assert!(lag_counter < SCRATCH_SIZE);
            scratch_mem[lag_counter] = xcorr32[(lag_high - j) as usize];
            lag_counter += 1;
        }

        let delta = lag_range[k][0] as i32;
        for i in 0..nb_cbk_search {
            // Fill out the 3 dim array that stores the correlations for each code_book vector
            // for each start lag
            let idx = (lag_cb.at(k, i) - delta) as usize;
            for j in 0..NB_ST3_LAGS {
                debug_assert!(idx + j < SCRATCH_SIZE);
                debug_assert!(idx + j < lag_counter);
                cross_corr_st3[k * nb_cbk_search + i][j] = scratch_mem[idx + j];
            }
        }
        target_ptr += sf_length as usize;
    }
}

/// Port of the static `pitch_analysis_core_FIX.c:silk_P_Ana_calc_energy_st3` — calculate the
/// energies for first two subframes. The energies are calculated recursively.
pub fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut Stage3Array,
    frame: &[i16],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut scratch_mem = [0i32; SCRATCH_SIZE];
    let (lag_range, lag_cb, nb_cbk_search) = stage3_tables(nb_subfr, complexity);
    let sfl = sf_length as usize;

    let mut target_ptr = silk_lshift(sf_length, 2) as usize;
    for k in 0..nb_subfr as usize {
        let mut lag_counter = 0usize;

        // Calculate the energy for first lag
        let basis_ptr = target_ptr - (start_lag + lag_range[k][0] as i32) as usize;
        let basis = &frame[basis_ptr..basis_ptr + sfl];
        let mut energy = silk_inner_prod_aligned(basis, basis, sfl);
        debug_assert!(energy >= 0);
        scratch_mem[lag_counter] = energy;
        lag_counter += 1;

        let lag_diff = (lag_range[k][1] - lag_range[k][0] + 1) as usize;
        for i in 1..lag_diff {
            // remove part outside new window
            let r = frame[basis_ptr + sfl - i] as i32;
            energy -= silk_smulbb(r, r);
            debug_assert!(energy >= 0);

            // add part that comes into window
            let a = frame[basis_ptr - i] as i32;
            energy = silk_add_sat32(energy, silk_smulbb(a, a));
            debug_assert!(energy >= 0);
            debug_assert!(lag_counter < SCRATCH_SIZE);
            scratch_mem[lag_counter] = energy;
            lag_counter += 1;
        }

        let delta = lag_range[k][0] as i32;
        for i in 0..nb_cbk_search {
            // Fill out the 3 dim array that stores the correlations for each code_book vector
            // for each start lag
            let idx = (lag_cb.at(k, i) - delta) as usize;
            for j in 0..NB_ST3_LAGS {
                debug_assert!(idx + j < SCRATCH_SIZE);
                debug_assert!(idx + j < lag_counter);
                energies_st3[k * nb_cbk_search + i][j] = scratch_mem[idx + j];
                debug_assert!(energies_st3[k * nb_cbk_search + i][j] >= 0);
            }
        }
        target_ptr += sf_length as usize;
    }
}

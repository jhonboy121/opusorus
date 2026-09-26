//! Port of `silk/float/pitch_analysis_core_FLP.c`: the floating-point pitch analyser.

use crate::celt::pitch::celt_pitch_xcorr;
use crate::silk::float::sigproc::{
    silk_energy_flp, silk_float2short_array, silk_inner_product_flp,
    silk_insertion_sort_decreasing_flp, silk_log2, silk_short2float_array,
};
use crate::silk::macros::{
    silk_limit, silk_limit_int, silk_lshift, silk_max_int, silk_min_int, silk_rshift,
    silk_rshift_round, silk_sat16, silk_smulbb,
};
use crate::silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use crate::silk::tables::{
    PE_D_SRCH_LENGTH, PE_FLATCONTOUR_BIAS, PE_LTP_MEM_LENGTH_MS, PE_MAX_FRAME_LENGTH_MS,
    PE_MAX_LAG, PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2,
    PE_NB_CBKS_STAGE2_10MS, PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX,
    PE_NB_STAGE3_LAGS, PE_PREVLAG_BIAS, PE_SHORTLAG_BIAS, PE_SUBFR_LENGTH_MS, SILK_CB_LAGS_STAGE2,
    SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS,
    SILK_LAG_RANGE_STAGE3, SILK_LAG_RANGE_STAGE3_10_MS, SILK_NB_CBK_SEARCHS_STAGE3,
    SILK_PE_MAX_COMPLEX, SILK_PE_MIN_COMPLEX,
};

const SCRATCH_SIZE: usize = 22;
const PMNS: usize = PE_MAX_NB_SUBFR as usize;
const NB_CBKS3_MAX: usize = PE_NB_CBKS_STAGE3_MAX as usize;
const NB_ST3_LAGS: usize = PE_NB_STAGE3_LAGS as usize;
/// Width of the `C` correlation rows: `(PE_MAX_LAG >> 1) + 5`.
const C_WIDTH: usize = ((PE_MAX_LAG >> 1) + 5) as usize;

/// The 3-D stage-3 arrays `[PE_MAX_NB_SUBFR][PE_NB_CBKS_STAGE3_MAX][PE_NB_STAGE3_LAGS]`.
pub type Stage3Array = [[[f32; NB_ST3_LAGS]; NB_CBKS3_MAX]; PMNS];

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

/// Port of `silk/float/pitch_analysis_core_FLP.c:silk_pitch_analysis_core_FLP`.
///
/// Returns the voicing estimate: 0 voiced, 1 unvoiced. `frame` holds
/// `PE_FRAME_LENGTH_MS * fs_khz` samples; `pitch_out` has `PE_MAX_NB_SUBFR` entries.
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "function signature mirrors the C source one-to-one"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn silk_pitch_analysis_core_flp(
    frame: &[f32],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr: &mut f32,
    mut prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
) -> i32 {
    const FL8_MAX: usize = (PE_MAX_FRAME_LENGTH_MS * 8) as usize;
    const FL4_MAX: usize = (PE_MAX_FRAME_LENGTH_MS * 4) as usize;
    let mut frame_8khz = [0f32; FL8_MAX];
    let mut frame_4khz = [0f32; FL4_MAX];
    let mut frame_8_fix = [0i16; FL8_MAX];
    let mut frame_4_fix = [0i16; FL4_MAX];
    let mut filt_state = [0i32; 6];
    let mut c = [[0f32; C_WIDTH]; PMNS];
    let mut xcorr = [0f32; (PE_MAX_LAG_MS * 4 - PE_MIN_LAG_MS * 4 + 1) as usize];
    let mut cc = [0f32; PE_NB_CBKS_STAGE2_EXT as usize];
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH as usize];
    let mut d_comp = [0i16; C_WIDTH];
    let mut energies_st3: Stage3Array = [[[0.0; NB_ST3_LAGS]; NB_CBKS3_MAX]; PMNS];
    let mut cross_corr_st3: Stage3Array = [[[0.0; NB_ST3_LAGS]; NB_CBKS3_MAX]; PMNS];

    // Check for valid sampling frequency
    celt_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
    // Check for valid complexity setting
    celt_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    celt_assert!(complexity <= SILK_PE_MAX_COMPLEX);

    silk_assert!((0.0f32..=1.0f32).contains(&search_thres1));

    let nsf = nb_subfr as usize;

    // Set up frame lengths max / min lag for the sampling frequency
    let frame_length = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz) as usize;
    let frame_length_4khz = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4) as usize;
    let frame_length_8khz = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8) as usize;
    let sf_length = (PE_SUBFR_LENGTH_MS * fs_khz) as usize;
    let sf_length_4khz = (PE_SUBFR_LENGTH_MS * 4) as usize;
    let sf_length_8khz = (PE_SUBFR_LENGTH_MS * 8) as usize;
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let min_lag_4khz = PE_MIN_LAG_MS * 4;
    let min_lag_8khz = PE_MIN_LAG_MS * 8;
    let max_lag = PE_MAX_LAG_MS * fs_khz - 1;
    let max_lag_4khz = PE_MAX_LAG_MS * 4;
    let max_lag_8khz = PE_MAX_LAG_MS * 8 - 1;

    // Resample from input sampled at Fs_kHz to 8 kHz
    if fs_khz == 16 {
        // Resample to 16 -> 8 khz
        let mut frame_16_fix = [0i16; (16 * PE_MAX_FRAME_LENGTH_MS) as usize];
        silk_float2short_array(&mut frame_16_fix, frame, frame_length);
        filt_state[..2].fill(0);
        silk_resampler_down2(
            &mut filt_state,
            &mut frame_8_fix,
            &frame_16_fix,
            frame_length as i32,
        );
        silk_short2float_array(&mut frame_8khz, &frame_8_fix, frame_length_8khz);
    } else if fs_khz == 12 {
        // Resample to 12 -> 8 khz
        let mut frame_12_fix = [0i16; (12 * PE_MAX_FRAME_LENGTH_MS) as usize];
        silk_float2short_array(&mut frame_12_fix, frame, frame_length);
        filt_state.fill(0);
        silk_resampler_down2_3(
            &mut filt_state,
            &mut frame_8_fix,
            &frame_12_fix,
            frame_length as i32,
        );
        silk_short2float_array(&mut frame_8khz, &frame_8_fix, frame_length_8khz);
    } else {
        celt_assert!(fs_khz == 8);
        silk_float2short_array(&mut frame_8_fix, frame, frame_length_8khz);
    }

    // Decimate again to 4 kHz
    filt_state[..2].fill(0);
    silk_resampler_down2(
        &mut filt_state,
        &mut frame_4_fix,
        &frame_8_fix,
        frame_length_8khz as i32,
    );
    silk_short2float_array(&mut frame_4khz, &frame_4_fix, frame_length_4khz);

    // Low-pass filter
    for i in (1..frame_length_4khz).rev() {
        // silk_ADD_SAT16( frame_4kHz[ i ], frame_4kHz[ i - 1 ] ) on floats:
        // (opus_int16)silk_SAT16( (opus_int32)a + b ), the sum computed in float
        let s = (frame_4khz[i] as i32) as f32 + frame_4khz[i - 1];
        let s = s.clamp(-32768.0, 32767.0);
        frame_4khz[i] = (s as i16) as f32;
    }

    // ------------------------------------------------------------------------------------
    // FIRST STAGE, operating in 4 khz
    // ------------------------------------------------------------------------------------
    let mut target = sf_length_4khz << 2;
    for _k in 0..nsf >> 1 {
        let mut basis = target - min_lag_4khz as usize;

        celt_pitch_xcorr(
            &frame_4khz[target..],
            &frame_4khz[target - max_lag_4khz as usize..],
            &mut xcorr,
            sf_length_8khz,
            (max_lag_4khz - min_lag_4khz + 1) as usize,
        );

        // Calculate first vector products before loop
        let mut cross_corr = xcorr[(max_lag_4khz - min_lag_4khz) as usize] as f64;
        let mut normalizer = silk_energy_flp(&frame_4khz[target..], sf_length_8khz)
            + silk_energy_flp(&frame_4khz[basis..], sf_length_8khz)
            + (sf_length_8khz as f32 * 4000.0f32) as f64;

        c[0][min_lag_4khz as usize] += (2.0 * cross_corr / normalizer) as f32;

        // From now on normalizer is computed recursively
        for d in min_lag_4khz + 1..=max_lag_4khz {
            basis -= 1;

            cross_corr = xcorr[(max_lag_4khz - d) as usize] as f64;

            // Add contribution of new sample and remove contribution from oldest sample
            let b0 = frame_4khz[basis] as f64;
            let bl = frame_4khz[basis + sf_length_8khz] as f64;
            normalizer += b0 * b0 - bl * bl;
            c[0][d as usize] += (2.0 * cross_corr / normalizer) as f32;
        }
        // Update target pointer
        target += sf_length_8khz;
    }

    // Apply short-lag bias
    for i in (min_lag_4khz..=max_lag_4khz).rev() {
        let v = c[0][i as usize];
        c[0][i as usize] -= v * i as f32 / 4096.0f32;
    }

    // Sort
    let mut length_d_srch = (4 + 2 * complexity) as usize;
    celt_assert!(3 * length_d_srch <= PE_D_SRCH_LENGTH as usize);
    silk_insertion_sort_decreasing_flp(
        &mut c[0][min_lag_4khz as usize..],
        &mut d_srch,
        (max_lag_4khz - min_lag_4khz + 1) as usize,
        length_d_srch,
    );

    // Escape if correlation is very low already here
    let cmax = c[0][min_lag_4khz as usize];
    if cmax < 0.2f32 {
        pitch_out[..nsf].fill(0);
        *ltp_corr = 0.0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = search_thres1 * cmax;
    for i in 0..length_d_srch {
        // Convert to 8 kHz indices for the sorted correlation that exceeds the threshold
        if c[0][min_lag_4khz as usize + i] > threshold {
            d_srch[i] = silk_lshift(d_srch[i] + min_lag_4khz, 1);
        } else {
            length_d_srch = i;
            break;
        }
    }
    celt_assert!(length_d_srch > 0);

    for i in (min_lag_8khz - 5) as usize..(max_lag_8khz + 5) as usize {
        d_comp[i] = 0;
    }
    for i in 0..length_d_srch {
        d_comp[d_srch[i] as usize] = 1;
    }

    // Convolution
    for i in (min_lag_8khz as usize..=(max_lag_8khz + 3) as usize).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2];
    }

    length_d_srch = 0;
    for i in min_lag_8khz..max_lag_8khz + 1 {
        if d_comp[(i + 1) as usize] > 0 {
            d_srch[length_d_srch] = i;
            length_d_srch += 1;
        }
    }

    // Convolution
    for i in (min_lag_8khz as usize..=(max_lag_8khz + 3) as usize).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2] + d_comp[i - 3];
    }

    let mut length_d_comp = 0usize;
    for i in min_lag_8khz..max_lag_8khz + 4 {
        if d_comp[i as usize] > 0 {
            d_comp[length_d_comp] = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    // ------------------------------------------------------------------------------------
    // SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    // ------------------------------------------------------------------------------------
    // Find energy of each subframe projected onto its history, for a range of delays
    c = [[0f32; C_WIDTH]; PMNS];

    let src8: &[f32] = if fs_khz == 8 { frame } else { &frame_8khz };
    let mut target = (PE_LTP_MEM_LENGTH_MS * 8) as usize;
    for k in 0..nsf {
        let energy_tmp = silk_energy_flp(&src8[target..], sf_length_8khz) + 1.0;
        for j in 0..length_d_comp {
            let d = d_comp[j] as usize;
            let basis = target - d;
            let cross_corr =
                silk_inner_product_flp(&src8[basis..], &src8[target..], sf_length_8khz);
            if cross_corr > 0.0 {
                let energy = silk_energy_flp(&src8[basis..], sf_length_8khz);
                c[k][d] = (2.0 * cross_corr / (energy + energy_tmp)) as f32;
            } else {
                c[k][d] = 0.0;
            }
        }
        target += sf_length_8khz;
    }

    // search over lag range and lags codebook
    // scale factor for lag codebook, as a function of center lag

    let mut ccmax = 0.0f32; // This value doesn't matter
    let mut ccmax_b = -1000.0f32;

    let mut cbimax = 0i32; // To avoid returning undefined lag values
    let mut lag = -1i32; // To check if lag with strong enough correlation has been found

    let prev_lag_log2 = if prev_lag > 0 {
        if fs_khz == 12 {
            prev_lag = silk_lshift(prev_lag, 1) / 3;
        } else if fs_khz == 16 {
            prev_lag = silk_rshift(prev_lag, 1);
        }
        silk_log2(prev_lag as f32 as f64)
    } else {
        0.0
    };

    // Set up stage 2 codebook based on number of subframes
    let lag_cb;
    let nb_cbk_search;
    if nb_subfr == PE_MAX_NB_SUBFR {
        lag_cb = LagCb {
            data: SILK_CB_LAGS_STAGE2.as_flattened(),
            cbk_size: PE_NB_CBKS_STAGE2_EXT as usize,
        };
        if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            // If input is 8 khz use a larger codebook here because it is last stage
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT as usize;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2 as usize;
        }
    } else {
        lag_cb = LagCb {
            data: SILK_CB_LAGS_STAGE2_10_MS.as_flattened(),
            cbk_size: PE_NB_CBKS_STAGE2_10MS as usize,
        };
        nb_cbk_search = PE_NB_CBKS_STAGE2_10MS as usize;
    }

    for k in 0..length_d_srch {
        let d = d_srch[k];
        for j in 0..nb_cbk_search {
            cc[j] = 0.0;
            for i in 0..nsf {
                // Try all codebooks
                cc[j] += c[i][(d + lag_cb.at(i, j)) as usize];
            }
        }
        // Find best codebook
        let mut ccmax_new = -1000.0f32;
        let mut cbimax_new = 0i32;
        for i in 0..nb_cbk_search {
            if cc[i] > ccmax_new {
                ccmax_new = cc[i];
                cbimax_new = i as i32;
            }
        }

        // Bias towards shorter lags
        let lag_log2 = silk_log2(d as f32 as f64);
        let mut ccmax_new_b = ccmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;

        // Bias towards previous lag
        if prev_lag > 0 {
            let mut delta_lag_log2_sqr = lag_log2 - prev_lag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            ccmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * (*ltp_corr) * delta_lag_log2_sqr
                / (delta_lag_log2_sqr + 0.5f32);
        }

        if ccmax_new_b > ccmax_b // Find maximum biased correlation
            && ccmax_new > nb_subfr as f32 * search_thres2
        // Correlation needs to be high enough to be voiced
        {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d;
            cbimax = cbimax_new;
        }
    }

    if lag == -1 {
        // No suitable candidate found
        pitch_out[..PMNS].fill(0);
        *ltp_corr = 0.0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    // Output normalized correlation
    *ltp_corr = ccmax / nb_subfr as f32;

    if fs_khz > 8 {
        // Search in original signal

        // Compensate for decimation
        silk_assert!(lag == silk_sat16(lag));
        if fs_khz == 12 {
            lag = silk_rshift_round(silk_smulbb(lag, 3), 1);
        } else {
            // Fs_kHz == 16
            lag = silk_lshift(lag, 1);
        }

        lag = silk_limit_int(lag, min_lag, max_lag);
        let start_lag = silk_max_int(lag - 2, min_lag);
        let end_lag = silk_min_int(lag + 2, max_lag);
        let mut lag_new = lag; // to avoid undefined lag
        cbimax = 0; // to avoid undefined lag

        ccmax = -1000.0f32;

        // Calculate the correlations and energies needed in stage 3
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

        silk_assert!(lag == silk_sat16(lag));
        let contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;

        // Set up cbk parameters according to complexity setting and frame length

        let (nb_cbk_search, lag_cb3) = if nb_subfr == PE_MAX_NB_SUBFR {
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

        let target = (PE_LTP_MEM_LENGTH_MS * fs_khz) as usize;
        let energy_tmp = silk_energy_flp(&frame[target..], nsf * sf_length) + 1.0;
        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search {
                let mut cross_corr = 0.0f64;
                let mut energy = energy_tmp;
                for k in 0..nsf {
                    cross_corr += cross_corr_st3[k][j][lag_counter] as f64;
                    energy += energies_st3[k][j][lag_counter] as f64;
                }
                let mut ccmax_new;
                if cross_corr > 0.0 {
                    ccmax_new = (2.0 * cross_corr / energy) as f32;
                    // Reduce depending on flatness of contour
                    ccmax_new *= 1.0f32 - contour_bias * j as f32;
                } else {
                    ccmax_new = 0.0f32;
                }

                // Note: C always checks the 20 ms stage-3 codebook row 0 here.
                if ccmax_new > ccmax && (d + SILK_CB_LAGS_STAGE3[0][j] as i32) <= max_lag {
                    ccmax = ccmax_new;
                    lag_new = d;
                    cbimax = j as i32;
                }
            }
        }

        for k in 0..nsf {
            pitch_out[k] = lag_new + lag_cb3.at(k, cbimax as usize);
            pitch_out[k] = silk_limit(pitch_out[k], min_lag, PE_MAX_LAG_MS * fs_khz);
        }
        *lag_index = (lag_new - min_lag) as i16;
        *contour_index = cbimax as i8;
    } else {
        // Fs_kHz == 8
        // Save Lags
        for k in 0..nsf {
            pitch_out[k] = lag + lag_cb.at(k, cbimax as usize);
            pitch_out[k] = silk_limit(pitch_out[k], min_lag_8khz, PE_MAX_LAG_MS * 8);
        }
        *lag_index = (lag - min_lag_8khz) as i16;
        *contour_index = cbimax as i8;
    }
    celt_assert!(*lag_index >= 0);
    // return as voiced
    0
}

/// Stage-3 lag range and codebook for the given complexity / frame length.
fn stage3_tables(nb_subfr: i32, complexity: i32) -> (LagCb<'static>, &'static [i8], usize) {
    if nb_subfr == PE_MAX_NB_SUBFR {
        (
            LagCb {
                data: SILK_CB_LAGS_STAGE3.as_flattened(),
                cbk_size: PE_NB_CBKS_STAGE3_MAX as usize,
            },
            SILK_LAG_RANGE_STAGE3[complexity as usize].as_flattened(),
            SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as usize,
        )
    } else {
        celt_assert!(nb_subfr == PE_MAX_NB_SUBFR >> 1);
        (
            LagCb {
                data: SILK_CB_LAGS_STAGE3_10_MS.as_flattened(),
                cbk_size: PE_NB_CBKS_STAGE3_10MS as usize,
            },
            SILK_LAG_RANGE_STAGE3_10_MS.as_flattened(),
            PE_NB_CBKS_STAGE3_10MS as usize,
        )
    }
}

/// Port of the static `pitch_analysis_core_FLP.c:silk_P_Ana_calc_corr_st3` — correlations
/// used in the stage-3 search, for every codebook vector and start lag.
pub fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut Stage3Array,
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut scratch_mem = [0f32; SCRATCH_SIZE];
    let mut xcorr = [0f32; SCRATCH_SIZE];

    celt_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    celt_assert!(complexity <= SILK_PE_MAX_COMPLEX);

    let (lag_cb, lag_range, nb_cbk_search) = stage3_tables(nb_subfr, complexity);

    let mut target = sf_length << 2; // Pointer to middle of frame
    for k in 0..nb_subfr as usize {
        // Calculate the correlations for each subframe
        let lag_low = lag_range[k * 2] as i32;
        let lag_high = lag_range[k * 2 + 1] as i32;
        silk_assert!((lag_high - lag_low + 1) as usize <= SCRATCH_SIZE);
        celt_pitch_xcorr(
            &frame[target..],
            &frame[target - (start_lag + lag_high) as usize..],
            &mut xcorr,
            sf_length,
            (lag_high - lag_low + 1) as usize,
        );
        for j in lag_low..=lag_high {
            scratch_mem[(j - lag_low) as usize] = xcorr[(lag_high - j) as usize];
        }

        let delta = lag_range[k * 2] as i32;
        for i in 0..nb_cbk_search {
            // Fill out the 3 dim array that stores the correlations for each code_book vector
            // for each start lag
            let idx = (lag_cb.at(k, i) - delta) as usize;
            cross_corr_st3[k][i].copy_from_slice(&scratch_mem[idx..idx + NB_ST3_LAGS]);
        }
        target += sf_length;
    }
}

/// Port of the static `pitch_analysis_core_FLP.c:silk_P_Ana_calc_energy_st3` — energies used
/// in the stage-3 search (computed recursively).
pub fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut Stage3Array,
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut scratch_mem = [0f32; SCRATCH_SIZE];

    celt_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    celt_assert!(complexity <= SILK_PE_MAX_COMPLEX);

    let (lag_cb, lag_range, nb_cbk_search) = stage3_tables(nb_subfr, complexity);

    let mut target = sf_length << 2;
    for k in 0..nb_subfr as usize {
        let mut lag_counter = 0usize;

        // Calculate the energy for first lag
        let basis = target - (start_lag + lag_range[k * 2] as i32) as usize;
        let mut energy = silk_energy_flp(&frame[basis..], sf_length) + 1e-3;
        scratch_mem[lag_counter] = energy as f32;
        lag_counter += 1;

        let lag_diff = (lag_range[k * 2 + 1] as i32 - lag_range[k * 2] as i32 + 1) as usize;
        for i in 1..lag_diff {
            // remove part outside new window
            let out = frame[basis + sf_length - i] as f64;
            energy -= out * out;

            // add part that comes into window
            let inp = frame[basis - i] as f64;
            energy += inp * inp;
            scratch_mem[lag_counter] = energy as f32;
            lag_counter += 1;
        }

        let delta = lag_range[k * 2] as i32;
        for i in 0..nb_cbk_search {
            // Fill out the 3 dim array that stores the correlations for each code_book vector
            // for each start lag
            let idx = (lag_cb.at(k, i) - delta) as usize;
            energies_st3[k][i].copy_from_slice(&scratch_mem[idx..idx + NB_ST3_LAGS]);
            for &e in &energies_st3[k][i] {
                silk_assert!(e >= 0.0f32);
            }
        }
        target += sf_length;
    }
}

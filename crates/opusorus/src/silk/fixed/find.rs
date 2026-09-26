//! Port of the prediction analysis files of `silk/fixed/`: `find_LPC_FIX.c`, `find_LTP_FIX.c`,
//! `find_pitch_lags_FIX.c`, `find_pred_coefs_FIX.c` and `LTP_scale_ctrl_FIX.c`.
//!
//! The C functions take `silk_encoder_state_FIX *psEnc` together with pointers into
//! `psEnc->x_buf` (the `x` argument, `x_frame = x_buf + ltp_mem_length` in C). Rust cannot alias
//! the two, so the state's parts are passed separately: `s_cmn` (`psEnc->sCmn`), `ltp_corr_q15`
//! (`psEnc->LTPCorr_Q15`), and the input buffer as `(x_buf, x_off)` with `x_off` the index of
//! the C pointer (the functions read samples before it).

use crate::silk::define::{
    CODE_INDEPENDENTLY, FIND_PITCH_LPC_WIN_MAX, LTP_ORDER, MAX_FIND_PITCH_LPC_ORDER,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_PREDICTION_POWER_GAIN,
    MAX_PREDICTION_POWER_GAIN_AFTER_RESET, TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, TYPE_VOICED,
};
use crate::silk::encoder_common::{silk_process_nlsfs, silk_quant_ltp_gains};
use crate::silk::fixed::pitch_analysis_core::silk_pitch_analysis_core;
use crate::silk::fixed::sigproc::{
    silk_apply_sine_window, silk_autocorr, silk_burg_modified, silk_corr_matrix_fix,
    silk_corr_vector_fix, silk_k2a, silk_ltp_analysis_filter_fix, silk_residual_energy_fix,
    silk_scale_copy_vector16, silk_schur,
};
use crate::silk::fixed::structs::SilkEncoderControlFix;
use crate::silk::macros::{
    silk_add32, silk_div32, silk_div32_varq, silk_fix_const, silk_lshift, silk_lshift64, silk_max,
    silk_max_int, silk_min, silk_rshift, silk_rshift32, silk_sat16, silk_smlabb, silk_smlawb,
    silk_smulbb, silk_smulww,
};
use crate::silk::nlsf::{silk_a2nlsf, silk_nlsf2a};
use crate::silk::sigproc::{
    silk_bwexpander, silk_interpolate, silk_log2lin, silk_lpc_analysis_filter, silk_sum_sqr_shift,
};
use crate::silk::structs::SilkEncoderState;
use crate::silk::tables::SILK_LTPSCALES_TABLE_Q14;
use crate::silk::tuning_parameters::{
    FIND_PITCH_BANDWIDTH_EXPANSION, FIND_PITCH_WHITE_NOISE_FRACTION, LTP_CORR_INV_MAX,
};

const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const LTPO: usize = LTP_ORDER as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

// ---------------------------------------------------------------------------------------------
// find_LPC_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/find_LPC_FIX.c:silk_find_LPC_FIX` — finds the LPC vector from
/// correlations and converts it to NLSFs. `x` is `LPC_in_pre` (`nb_subfr` subframes, each
/// preceded by `predictLPCOrder` samples).
pub fn silk_find_lpc_fix(
    ps_enc_c: &mut SilkEncoderState,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
) {
    let mut a_q16 = [0i32; MLPC];
    // Used only for LSF interpolation
    let mut a_tmp_q16 = [0i32; MLPC];
    let mut a_tmp_q12 = [0i16; MLPC];
    let mut nlsf0_q15 = [0i16; MLPC];
    // C: ALLOC( LPC_res, 2 * subfr_length )
    let mut lpc_res = [0i16; 2 * (MFL / MNSF + MLPC)];

    let order = ps_enc_c.predict_lpc_order as usize;
    let subfr_length = ps_enc_c.subfr_length as usize + order;

    // Default: no interpolation
    ps_enc_c.indices.nlsf_interp_coef_q2 = 4;

    // Burg AR analysis for the full frame
    let (mut res_nrg, mut res_nrg_q) = silk_burg_modified(
        &mut a_q16,
        x,
        min_inv_gain_q30,
        subfr_length,
        ps_enc_c.nb_subfr as usize,
        order,
    );

    if ps_enc_c.use_interpolated_nlsfs != 0
        && ps_enc_c.first_frame_after_reset == 0
        && ps_enc_c.nb_subfr == MAX_NB_SUBFR
    {
        // Optimal solution for last 10 ms
        let (res_tmp_nrg, res_tmp_nrg_q) = silk_burg_modified(
            &mut a_tmp_q16,
            &x[2 * subfr_length..],
            min_inv_gain_q30,
            subfr_length,
            2,
            order,
        );

        // subtract residual energy here, as that's easier than adding it to the residual
        // energy of the first 10 ms in each iteration of the search below
        let shift = res_tmp_nrg_q - res_nrg_q;
        if shift >= 0 {
            if shift < 32 {
                res_nrg -= silk_rshift(res_tmp_nrg, shift);
            }
        } else {
            silk_assert!(shift > -32);
            res_nrg = silk_rshift(res_nrg, -shift) - res_tmp_nrg;
            res_nrg_q = res_tmp_nrg_q;
        }

        // Convert to NLSFs
        silk_a2nlsf(nlsf_q15, &mut a_tmp_q16, order);

        let lpc_res = &mut lpc_res[..2 * subfr_length];

        // Search over interpolation indices to find the one with lowest residual energy
        for k in (0..=3).rev() {
            // Interpolate NLSFs for first half
            silk_interpolate(&mut nlsf0_q15, &ps_enc_c.prev_nlsfq_q15, nlsf_q15, k, order);

            // Convert to LPC for residual energy evaluation
            silk_nlsf2a(&mut a_tmp_q12, &nlsf0_q15, order);

            // Calculate residual energy with NLSF interpolation
            silk_lpc_analysis_filter(lpc_res, x, &a_tmp_q12, 2 * subfr_length, order);

            let (mut res_nrg0, rshift0) =
                silk_sum_sqr_shift(&lpc_res[order..], subfr_length - order);
            let (mut res_nrg1, rshift1) =
                silk_sum_sqr_shift(&lpc_res[order + subfr_length..], subfr_length - order);

            // Add subframe energies from first half frame
            let shift = rshift0 - rshift1;
            let res_nrg_interp_q;
            if shift >= 0 {
                res_nrg1 = silk_rshift(res_nrg1, shift);
                res_nrg_interp_q = -rshift0;
            } else {
                res_nrg0 = silk_rshift(res_nrg0, -shift);
                res_nrg_interp_q = -rshift1;
            }
            let res_nrg_interp = silk_add32(res_nrg0, res_nrg1);

            // Compare with first half energy without NLSF interpolation, or best interpolated
            // value so far
            let shift = res_nrg_interp_q - res_nrg_q;
            let is_interp_lower = if shift >= 0 {
                silk_rshift(res_nrg_interp, shift) < res_nrg
            } else if -shift < 32 {
                res_nrg_interp < silk_rshift(res_nrg, -shift)
            } else {
                false
            };

            // Determine whether current interpolated NLSFs are best so far
            if is_interp_lower {
                // Interpolation has lower residual energy
                res_nrg = res_nrg_interp;
                res_nrg_q = res_nrg_interp_q;
                ps_enc_c.indices.nlsf_interp_coef_q2 = k as i8;
            }
        }
    }

    if ps_enc_c.indices.nlsf_interp_coef_q2 == 4 {
        // NLSF interpolation is currently inactive, calculate NLSFs from full frame AR
        // coefficients
        silk_a2nlsf(nlsf_q15, &mut a_q16, order);
    }

    celt_assert!(
        ps_enc_c.indices.nlsf_interp_coef_q2 == 4
            || (ps_enc_c.use_interpolated_nlsfs != 0
                && ps_enc_c.first_frame_after_reset == 0
                && ps_enc_c.nb_subfr == MAX_NB_SUBFR)
    );
}

// ---------------------------------------------------------------------------------------------
// find_LTP_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/find_LTP_FIX.c:silk_find_LTP_FIX` — LTP analysis.
///
/// The C residual pointer `r_ptr` is `&r_buf[r_off]` (the function reads up to
/// `lag + LTP_ORDER / 2` samples before it). `xx_ltp_q17` receives `nb_subfr` 5x5 matrices and
/// `x_x_ltp_q17` `nb_subfr` 5-vectors.
pub fn silk_find_ltp_fix(
    xx_ltp_q17: &mut [i32],
    x_x_ltp_q17: &mut [i32],
    r_buf: &[i16],
    r_off: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
) {
    let mut r_ptr = r_off;
    for k in 0..nb_subfr {
        let xx_ptr = &mut xx_ltp_q17[k * LTPO * LTPO..(k + 1) * LTPO * LTPO];
        let xv_ptr = &mut x_x_ltp_q17[k * LTPO..(k + 1) * LTPO];
        let lag_ptr = r_ptr - (lag[k] as usize + LTPO / 2);

        let (mut xx, xx_shifts) = silk_sum_sqr_shift(&r_buf[r_ptr..], subfr_length + LTPO); // xx in Q( -xx_shifts )
        let (mut nrg, xx_mat_shifts) =
            silk_corr_matrix_fix(&r_buf[lag_ptr..], subfr_length, LTPO, xx_ptr); // XXLTP_Q17_ptr and nrg in Q( -XX_shifts )
        let extra_shifts = xx_shifts - xx_mat_shifts;
        let x_x_shifts;
        if extra_shifts > 0 {
            // Shift XX
            x_x_shifts = xx_shifts;
            for v in xx_ptr.iter_mut() {
                *v = silk_rshift32(*v, extra_shifts); // Q( -xX_shifts )
            }
            nrg = silk_rshift32(nrg, extra_shifts); // Q( -xX_shifts )
        } else if extra_shifts < 0 {
            // Shift xx
            x_x_shifts = xx_mat_shifts;
            xx = silk_rshift32(xx, -extra_shifts); // Q( -xX_shifts )
        } else {
            x_x_shifts = xx_shifts;
        }
        silk_corr_vector_fix(
            &r_buf[lag_ptr..],
            &r_buf[r_ptr..],
            subfr_length,
            LTPO,
            xv_ptr,
            x_x_shifts,
        ); // xXLTP_Q17_ptr in Q( -xX_shifts )

        // At this point all correlations are in Q(-xX_shifts)
        let mut temp = silk_smlawb(1, nrg, silk_fix_const(LTP_CORR_INV_MAX as f64, 16));
        temp = silk_max(temp, xx);
        for v in xx_ptr.iter_mut() {
            *v = (silk_lshift64(*v as i64, 17) / temp as i64) as i32;
        }
        for v in xv_ptr.iter_mut() {
            *v = (silk_lshift64(*v as i64, 17) / temp as i64) as i32;
        }
        r_ptr += subfr_length;
    }
}

// ---------------------------------------------------------------------------------------------
// find_pitch_lags_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/find_pitch_lags_FIX.c:silk_find_pitch_lags_FIX` — find pitch lags.
///
/// `res` receives the pitch-analysis LPC residual of the whole buffer
/// (`la_pitch + frame_length + ltp_mem_length` samples); the input is `x_buf[x_start..]` (the C
/// `x`, which is `x_frame - ltp_mem_length`).
pub fn silk_find_pitch_lags_fix(
    s_cmn: &mut SilkEncoderState,
    ltp_corr_q15: &mut i32,
    ps_enc_ctrl: &mut SilkEncoderControlFix,
    res: &mut [i16],
    x_buf: &[i16],
    x_start: usize,
) {
    const MFPO: usize = MAX_FIND_PITCH_LPC_ORDER as usize;
    let mut auto_corr = [0i32; MFPO + 1];
    let mut rc_q15 = [0i16; MFPO];
    let mut a_q24 = [0i32; MFPO];
    let mut a_q12 = [0i16; MFPO];
    let mut wsig = [0i16; FIND_PITCH_LPC_WIN_MAX as usize];

    // Set up buffer lengths etc based on Fs
    let la_pitch = s_cmn.la_pitch as usize;
    let win_len = s_cmn.pitch_lpc_win_length as usize;
    let buf_len = la_pitch + s_cmn.frame_length as usize + s_cmn.ltp_mem_length as usize;

    // Safety check
    celt_assert!(buf_len >= win_len);

    let x = &x_buf[x_start..x_start + buf_len];

    // Estimate LPC AR coefficients

    // Calculate windowed signal

    // First LA_LTP samples
    let mut xp = buf_len - win_len;
    let mut wp = 0usize;
    silk_apply_sine_window(&mut wsig[wp..], &x[xp..], 1, la_pitch);

    // Middle un - windowed samples
    wp += la_pitch;
    xp += la_pitch;
    let mid = win_len - silk_lshift(la_pitch as i32, 1) as usize;
    wsig[wp..wp + mid].copy_from_slice(&x[xp..xp + mid]);

    // Last LA_LTP samples
    wp += mid;
    xp += mid;
    silk_apply_sine_window(&mut wsig[wp..], &x[xp..], 2, la_pitch);

    // Calculate autocorrelation sequence
    let order = s_cmn.pitch_estimation_lpc_order as usize;
    let _scale = silk_autocorr(&mut auto_corr, &wsig, win_len, order + 1);

    // Add white noise, as fraction of energy
    auto_corr[0] = silk_smlawb(
        auto_corr[0],
        auto_corr[0],
        silk_fix_const(FIND_PITCH_WHITE_NOISE_FRACTION as f64, 16),
    ) + 1;

    // Calculate the reflection coefficients using schur
    let res_nrg = silk_schur(&mut rc_q15, &auto_corr, order);

    // Prediction gain
    ps_enc_ctrl.pred_gain_q16 = silk_div32_varq(auto_corr[0], silk_max_int(res_nrg, 1), 16);

    // Convert reflection coefficients to prediction coefficients
    silk_k2a(&mut a_q24, &rc_q15, order);

    // Convert From 32 bit Q24 to 16 bit Q12 coefs
    for i in 0..order {
        a_q12[i] = silk_sat16(silk_rshift(a_q24[i], 12)) as i16;
    }

    // Do BWE
    silk_bwexpander(
        &mut a_q12,
        order,
        silk_fix_const(FIND_PITCH_BANDWIDTH_EXPANSION as f64, 16),
    );

    // LPC analysis filtering
    silk_lpc_analysis_filter(res, x, &a_q12, buf_len, order);

    if s_cmn.indices.signal_type as i32 != TYPE_NO_VOICE_ACTIVITY
        && s_cmn.first_frame_after_reset == 0
    {
        // Threshold for pitch estimator
        let mut thrhld_q13 = silk_fix_const(0.6, 13);
        thrhld_q13 = silk_smlabb(
            thrhld_q13,
            silk_fix_const(-0.004, 13),
            s_cmn.pitch_estimation_lpc_order,
        );
        thrhld_q13 = silk_smlawb(
            thrhld_q13,
            silk_fix_const(-0.1, 21),
            s_cmn.speech_activity_q8,
        );
        thrhld_q13 = silk_smlabb(
            thrhld_q13,
            silk_fix_const(-0.15, 13),
            silk_rshift(s_cmn.prev_signal_type as i32, 1),
        );
        thrhld_q13 = silk_smlawb(thrhld_q13, silk_fix_const(-0.1, 14), s_cmn.input_tilt_q15);
        thrhld_q13 = silk_sat16(thrhld_q13);

        // Call pitch estimator
        if silk_pitch_analysis_core(
            res,
            &mut ps_enc_ctrl.pitch_l,
            &mut s_cmn.indices.lag_index,
            &mut s_cmn.indices.contour_index,
            ltp_corr_q15,
            s_cmn.prev_lag,
            s_cmn.pitch_estimation_threshold_q16,
            thrhld_q13,
            s_cmn.fs_khz,
            s_cmn.pitch_estimation_complexity,
            s_cmn.nb_subfr,
        ) == 0
        {
            s_cmn.indices.signal_type = TYPE_VOICED as i8;
        } else {
            s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        }
    } else {
        ps_enc_ctrl.pitch_l = [0; MNSF];
        s_cmn.indices.lag_index = 0;
        s_cmn.indices.contour_index = 0;
        *ltp_corr_q15 = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// LTP_scale_ctrl_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/LTP_scale_ctrl_FIX.c:silk_LTP_scale_ctrl_FIX` — calculation of LTP state
/// scaling.
pub fn silk_ltp_scale_ctrl_fix(
    s_cmn: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControlFix,
    cond_coding: i32,
) {
    if cond_coding == CODE_INDEPENDENTLY {
        // Only scale if first frame in packet
        let mut round_loss = s_cmn.packet_loss_perc * s_cmn.n_frames_per_packet;
        if s_cmn.lbrr_flag != 0 {
            // LBRR reduces the effective loss. In practice, it does not square the loss because
            // losses aren't independent, but that still seems to work best. We also never go
            // below 2%.
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        let prod = silk_smulbb(ps_enc_ctrl.lt_pred_cod_gain_q7, round_loss);
        let mut idx = (prod > silk_log2lin(128 * 7 + 2900 - s_cmn.snr_db_q7)) as i8;
        idx += (prod > silk_log2lin(128 * 7 + 3900 - s_cmn.snr_db_q7)) as i8;
        s_cmn.indices.ltp_scale_index = idx;
    } else {
        // Default is minimum scaling
        s_cmn.indices.ltp_scale_index = 0;
    }
    ps_enc_ctrl.ltp_scale_q14 =
        SILK_LTPSCALES_TABLE_Q14[s_cmn.indices.ltp_scale_index as usize] as i32;
}

// ---------------------------------------------------------------------------------------------
// find_pred_coefs_FIX.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/fixed/find_pred_coefs_FIX.c:silk_find_pred_coefs_FIX` — find LPC and LTP
/// coefficients.
///
/// `res_pitch` is `(res_buf, res_off)` (C `res_pitch = &res_buf[res_off]`, the pitch residual
/// frame); the speech signal is `(x_buf, x_off)`.
pub fn silk_find_pred_coefs_fix(
    s_cmn: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControlFix,
    res_buf: &[i16],
    res_off: usize,
    x_buf: &[i16],
    x_off: usize,
    cond_coding: i32,
) {
    let mut inv_gains_q16 = [0i32; MNSF];
    let mut local_gains = [0i32; MNSF];
    // Set to NLSF_Q15 to zero so we don't copy junk to the state.
    let mut nlsf_q15 = [0i16; MLPC];
    let mut lpc_in_pre = [0i16; MNSF * MLPC + MFL];

    let nb_subfr = s_cmn.nb_subfr as usize;
    let subfr_length = s_cmn.subfr_length as usize;
    let order = s_cmn.predict_lpc_order as usize;

    // weighting for weighted least squares
    let mut min_gain_q16 = i32::MAX >> 6;
    for i in 0..nb_subfr {
        min_gain_q16 = silk_min(min_gain_q16, ps_enc_ctrl.gains_q16[i]);
    }
    for i in 0..nb_subfr {
        // Divide to Q16
        silk_assert!(ps_enc_ctrl.gains_q16[i] > 0);
        // Invert and normalize gains, and ensure that maximum invGains_Q16 is within range of
        // a 16 bit int
        inv_gains_q16[i] = silk_div32_varq(min_gain_q16, ps_enc_ctrl.gains_q16[i], 16 - 2);

        // Limit inverse
        inv_gains_q16[i] = silk_max(inv_gains_q16[i], 100);

        // Square the inverted gains
        silk_assert!(inv_gains_q16[i] == silk_sat16(inv_gains_q16[i]));

        // Invert the inverted and normalized gains
        local_gains[i] = silk_div32(1i32 << 16, inv_gains_q16[i]);
    }

    let lpc_in_pre = &mut lpc_in_pre[..nb_subfr * order + s_cmn.frame_length as usize];
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        let mut x_x_ltp_q17 = [0i32; MNSF * LTPO];
        let mut xx_ltp_q17 = [0i32; MNSF * LTPO * LTPO];

        // VOICED
        celt_assert!(
            s_cmn.ltp_mem_length - s_cmn.predict_lpc_order
                >= ps_enc_ctrl.pitch_l[0] + LTP_ORDER / 2
        );

        // LTP analysis
        silk_find_ltp_fix(
            &mut xx_ltp_q17,
            &mut x_x_ltp_q17,
            res_buf,
            res_off,
            &ps_enc_ctrl.pitch_l,
            subfr_length,
            nb_subfr,
        );

        // Quantize LTP gain parameters
        silk_quant_ltp_gains(
            &mut ps_enc_ctrl.ltp_coef_q14,
            &mut s_cmn.indices.ltp_index,
            &mut s_cmn.indices.per_index,
            &mut s_cmn.sum_log_gain_q7,
            &mut ps_enc_ctrl.lt_pred_cod_gain_q7,
            &xx_ltp_q17,
            &x_x_ltp_q17,
            s_cmn.subfr_length,
            nb_subfr,
        );

        // Control LTP scaling
        silk_ltp_scale_ctrl_fix(s_cmn, ps_enc_ctrl, cond_coding);

        // Create LTP residual
        silk_ltp_analysis_filter_fix(
            lpc_in_pre,
            x_buf,
            x_off - order,
            &ps_enc_ctrl.ltp_coef_q14,
            &ps_enc_ctrl.pitch_l,
            &inv_gains_q16,
            subfr_length,
            nb_subfr,
            order,
        );
    } else {
        // UNVOICED
        // Create signal with prepended subframes, scaled by inverse gains
        let mut x_ptr = x_off - order;
        let mut x_pre_ptr = 0usize;
        for i in 0..nb_subfr {
            silk_scale_copy_vector16(
                &mut lpc_in_pre[x_pre_ptr..],
                &x_buf[x_ptr..],
                inv_gains_q16[i],
                subfr_length + order,
            );
            x_pre_ptr += subfr_length + order;
            x_ptr += subfr_length;
        }

        ps_enc_ctrl.ltp_coef_q14[..nb_subfr * LTPO].fill(0);
        ps_enc_ctrl.lt_pred_cod_gain_q7 = 0;
        s_cmn.sum_log_gain_q7 = 0;
        ps_enc_ctrl.ltp_scale_q14 = 0;
    }

    // Limit on total predictive coding gain
    let min_inv_gain_q30 = if s_cmn.first_frame_after_reset != 0 {
        // C: SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN_AFTER_RESET, 30 ), a float
        // division
        silk_fix_const((1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET) as f64, 30)
    } else {
        let v = silk_log2lin(silk_smlawb(
            16 << 7,
            ps_enc_ctrl.lt_pred_cod_gain_q7,
            silk_fix_const(1.0 / 3.0, 16),
        )); // Q16
        silk_div32_varq(
            v,
            silk_smulww(
                silk_fix_const(MAX_PREDICTION_POWER_GAIN as f64, 0),
                silk_smlawb(
                    silk_fix_const(0.25, 18),
                    silk_fix_const(0.75, 18),
                    ps_enc_ctrl.coding_quality_q14,
                ),
            ),
            14,
        )
    };

    // LPC_in_pre contains the LTP-filtered input for voiced, and the unfiltered input for
    // unvoiced
    silk_find_lpc_fix(s_cmn, &mut nlsf_q15, lpc_in_pre, min_inv_gain_q30);

    // Quantize LSFs
    let prev_nlsfq_q15 = s_cmn.prev_nlsfq_q15;
    silk_process_nlsfs(
        s_cmn,
        &mut ps_enc_ctrl.pred_coef_q12,
        &mut nlsf_q15,
        &prev_nlsfq_q15,
    );

    // Calculate residual energy using quantized LPC coefficients
    silk_residual_energy_fix(
        &mut ps_enc_ctrl.res_nrg,
        &mut ps_enc_ctrl.res_nrg_q,
        lpc_in_pre,
        &ps_enc_ctrl.pred_coef_q12,
        &local_gains,
        subfr_length,
        nb_subfr,
        order,
    );

    // Copy to prediction struct for use in next frame for interpolation
    s_cmn.prev_nlsfq_q15 = nlsf_q15;
}

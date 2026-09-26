//! Port of the prediction analysis files of `silk/float/`: `find_LPC_FLP.c`, `find_LTP_FLP.c`,
//! `find_pitch_lags_FLP.c`, `find_pred_coefs_FLP.c` and `LTP_scale_ctrl_FLP.c`.
//!
//! The C functions take `silk_encoder_state_FLP *psEnc` together with pointers into
//! `psEnc->x_buf` (the `x` argument, always `x_frame = x_buf + ltp_mem_length` in C). Rust cannot
//! alias the two, so the state's parts are passed separately: `s_cmn` (`psEnc->sCmn`),
//! `ltp_corr` (`psEnc->LTPCorr`), and the input buffer as `(x_buf, x_off)` with `x_off` the index
//! of the C pointer (the functions read samples before it).

use crate::math;
use crate::silk::define::{
    CODE_INDEPENDENTLY, FIND_PITCH_LPC_WIN_MAX, LTP_ORDER, MAX_FIND_PITCH_LPC_ORDER,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_PREDICTION_POWER_GAIN,
    MAX_PREDICTION_POWER_GAIN_AFTER_RESET, TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, TYPE_VOICED,
};
use crate::silk::float::pitch_analysis_core::silk_pitch_analysis_core_flp;
use crate::silk::float::sigproc::{
    silk_apply_sine_window_flp, silk_autocorrelation_flp, silk_burg_modified_flp,
    silk_bwexpander_flp, silk_corr_matrix_flp, silk_corr_vector_flp, silk_energy_flp, silk_k2a_flp,
    silk_lpc_analysis_filter_flp, silk_ltp_analysis_filter_flp, silk_max_float,
    silk_residual_energy_flp, silk_scale_copy_vector_flp, silk_scale_vector_flp, silk_schur_flp,
};
use crate::silk::float::structs::SilkEncoderControlFlp;
use crate::silk::float::wrappers::{
    silk_a2nlsf_flp, silk_nlsf2a_flp, silk_process_nlsfs_flp, silk_quant_ltp_gains_flp,
};
use crate::silk::macros::silk_smulbb;
use crate::silk::sigproc::{silk_interpolate, silk_log2lin};
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
// find_LPC_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/find_LPC_FLP.c:silk_find_LPC_FLP` — LPC analysis (with optional NLSF
/// interpolation search). `x` is `LPC_in_pre` (`nb_subfr` subframes, each preceded by
/// `predictLPCOrder` samples).
pub fn silk_find_lpc_flp(
    ps_enc_c: &mut SilkEncoderState,
    nlsf_q15: &mut [i16],
    x: &[f32],
    min_inv_gain: f32,
) {
    let mut a = [0f32; MLPC];
    // Used only for NLSF interpolation
    let mut nlsf0_q15 = [0i16; MLPC];
    let mut a_tmp = [0f32; MLPC];
    let mut lpc_res = [0f32; MFL + MNSF * MLPC];

    let order = ps_enc_c.predict_lpc_order as usize;
    let subfr_length = ps_enc_c.subfr_length as usize + order;

    // Default: No interpolation
    ps_enc_c.indices.nlsf_interp_coef_q2 = 4;

    // Burg AR analysis for the full frame
    let mut res_nrg = silk_burg_modified_flp(
        &mut a,
        x,
        min_inv_gain,
        subfr_length,
        ps_enc_c.nb_subfr as usize,
        order,
    );

    if ps_enc_c.use_interpolated_nlsfs != 0
        && ps_enc_c.first_frame_after_reset == 0
        && ps_enc_c.nb_subfr == MAX_NB_SUBFR
    {
        // Optimal solution for last 10 ms; subtract residual energy here, as that's easier than
        // adding it to the residual energy of the first 10 ms in each iteration of the search
        // below
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp,
            &x[(MNSF / 2) * subfr_length..],
            min_inv_gain,
            subfr_length,
            MNSF / 2,
            order,
        );

        // Convert to NLSFs
        silk_a2nlsf_flp(nlsf_q15, &a_tmp, order);

        // Search over interpolation indices to find the one with lowest residual energy
        let mut res_nrg_2nd = f32::MAX;
        for k in (0..=3).rev() {
            // Interpolate NLSFs for first half
            silk_interpolate(&mut nlsf0_q15, &ps_enc_c.prev_nlsfq_q15, nlsf_q15, k, order);

            // Convert to LPC for residual energy evaluation
            silk_nlsf2a_flp(&mut a_tmp, &nlsf0_q15, order);

            // Calculate residual energy with LSF interpolation
            silk_lpc_analysis_filter_flp(&mut lpc_res, &a_tmp, x, 2 * subfr_length, order);
            let res_nrg_interp = (silk_energy_flp(&lpc_res[order..], subfr_length - order)
                + silk_energy_flp(&lpc_res[order + subfr_length..], subfr_length - order))
                as f32;

            // Determine whether current interpolated NLSFs are best so far
            if res_nrg_interp < res_nrg {
                // Interpolation has lower residual energy
                res_nrg = res_nrg_interp;
                ps_enc_c.indices.nlsf_interp_coef_q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                // No reason to continue iterating - residual energies will continue to climb
                break;
            }
            res_nrg_2nd = res_nrg_interp;
        }
    }

    if ps_enc_c.indices.nlsf_interp_coef_q2 == 4 {
        // NLSF interpolation is currently inactive, calculate NLSFs from full frame AR
        // coefficients
        silk_a2nlsf_flp(nlsf_q15, &a, order);
    }

    celt_assert!(
        ps_enc_c.indices.nlsf_interp_coef_q2 == 4
            || (ps_enc_c.use_interpolated_nlsfs != 0
                && ps_enc_c.first_frame_after_reset == 0
                && ps_enc_c.nb_subfr == MAX_NB_SUBFR)
    );
}

// ---------------------------------------------------------------------------------------------
// find_LTP_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/find_LTP_FLP.c:silk_find_LTP_FLP` — LTP analysis.
///
/// The C residual pointer `r_ptr` is `&r_buf[r_off]` (the function reads up to
/// `lag + LTP_ORDER / 2` samples before it). `xx_mat` receives `nb_subfr` 5x5 matrices and
/// `xx_vec` `nb_subfr` 5-vectors.
pub fn silk_find_ltp_flp(
    xx_mat: &mut [f32],
    xx_vec: &mut [f32],
    r_buf: &[f32],
    r_off: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
) {
    let mut r_ptr = r_off;
    for k in 0..nb_subfr {
        let xx_ptr = &mut xx_mat[k * LTPO * LTPO..(k + 1) * LTPO * LTPO];
        let xv_ptr = &mut xx_vec[k * LTPO..(k + 1) * LTPO];
        let lag_ptr = r_ptr - (lag[k] as usize + LTPO / 2);
        silk_corr_matrix_flp(&r_buf[lag_ptr..], subfr_length, LTPO, xx_ptr);
        silk_corr_vector_flp(
            &r_buf[lag_ptr..],
            &r_buf[r_ptr..],
            subfr_length,
            LTPO,
            xv_ptr,
        );
        let xx = silk_energy_flp(&r_buf[r_ptr..], subfr_length + LTPO) as f32;
        // silk_max( xx, ... ): C ternary on floats
        let alt = LTP_CORR_INV_MAX * 0.5f32 * (xx_ptr[0] + xx_ptr[24]) + 1.0f32;
        let temp = 1.0f32 / if xx > alt { xx } else { alt };
        silk_scale_vector_flp(xx_ptr, temp, LTPO * LTPO);
        silk_scale_vector_flp(xv_ptr, temp, LTPO);

        r_ptr += subfr_length;
    }
}

// ---------------------------------------------------------------------------------------------
// find_pitch_lags_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/find_pitch_lags_FLP.c:silk_find_pitch_lags_FLP` — find pitch lags.
///
/// `res` receives the pitch-analysis LPC residual of the whole buffer
/// (`la_pitch + frame_length + ltp_mem_length` samples); the input is
/// `x_buf[x_off - ltp_mem_length..]` (C `x - ltp_mem_length` with `x = &x_buf[x_off]`).
pub fn silk_find_pitch_lags_flp(
    s_cmn: &mut SilkEncoderState,
    ltp_corr: &mut f32,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
    res: &mut [f32],
    x_buf: &[f32],
    x_off: usize,
) {
    const MFPO: usize = MAX_FIND_PITCH_LPC_ORDER as usize;
    let mut auto_corr = [0f32; MFPO + 1];
    let mut a = [0f32; MFPO];
    let mut refl_coef = [0f32; MFPO];
    let mut wsig = [0f32; FIND_PITCH_LPC_WIN_MAX as usize];

    // Set up buffer lengths etc based on Fs
    let la_pitch = s_cmn.la_pitch as usize;
    let win_len = s_cmn.pitch_lpc_win_length as usize;
    let buf_len = la_pitch + s_cmn.frame_length as usize + s_cmn.ltp_mem_length as usize;

    // Safety check
    celt_assert!(buf_len >= win_len);

    let x_buf_start = x_off - s_cmn.ltp_mem_length as usize;
    let x_buf = &x_buf[x_buf_start..x_buf_start + buf_len];

    // Estimate LPC AR coefficients

    // Calculate windowed signal

    // First LA_LTP samples
    let mut xp = buf_len - win_len;
    let mut wp = 0usize;
    silk_apply_sine_window_flp(&mut wsig[wp..], &x_buf[xp..], 1, la_pitch);

    // Middle non-windowed samples
    wp += la_pitch;
    xp += la_pitch;
    let mid = win_len - (la_pitch << 1);
    wsig[wp..wp + mid].copy_from_slice(&x_buf[xp..xp + mid]);

    // Last LA_LTP samples
    wp += mid;
    xp += mid;
    silk_apply_sine_window_flp(&mut wsig[wp..], &x_buf[xp..], 2, la_pitch);

    // Calculate autocorrelation sequence
    let order = s_cmn.pitch_estimation_lpc_order as usize;
    silk_autocorrelation_flp(&mut auto_corr, &wsig, win_len, order + 1);

    // Add white noise, as a fraction of the energy
    auto_corr[0] += auto_corr[0] * FIND_PITCH_WHITE_NOISE_FRACTION + 1.0f32;

    // Calculate the reflection coefficients using Schur
    let res_nrg = silk_schur_flp(&mut refl_coef, &auto_corr, order);

    // Prediction gain
    ps_enc_ctrl.pred_gain = auto_corr[0] / silk_max_float(res_nrg, 1.0f32);

    // Convert reflection coefficients to prediction coefficients
    silk_k2a_flp(&mut a, &refl_coef, order);

    // Bandwidth expansion
    silk_bwexpander_flp(&mut a, order, FIND_PITCH_BANDWIDTH_EXPANSION);

    // LPC analysis filtering
    silk_lpc_analysis_filter_flp(res, &a, x_buf, buf_len, order);

    if s_cmn.indices.signal_type as i32 != TYPE_NO_VOICE_ACTIVITY
        && s_cmn.first_frame_after_reset == 0
    {
        // Threshold for pitch estimator
        let mut thrhld = 0.6f32;
        thrhld -= 0.004f32 * s_cmn.pitch_estimation_lpc_order as f32;
        thrhld -= 0.1f32 * s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);
        thrhld -= 0.15f32 * (s_cmn.prev_signal_type as i32 >> 1) as f32;
        thrhld -= 0.1f32 * s_cmn.input_tilt_q15 as f32 * (1.0f32 / 32768.0f32);

        // Call Pitch estimator
        if silk_pitch_analysis_core_flp(
            res,
            &mut ps_enc_ctrl.pitch_l,
            &mut s_cmn.indices.lag_index,
            &mut s_cmn.indices.contour_index,
            ltp_corr,
            s_cmn.prev_lag,
            s_cmn.pitch_estimation_threshold_q16 as f32 / 65536.0f32,
            thrhld,
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
        *ltp_corr = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// LTP_scale_ctrl_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/LTP_scale_ctrl_FLP.c:silk_LTP_scale_ctrl_FLP` — calculation of LTP state
/// scaling.
pub fn silk_ltp_scale_ctrl_flp(
    s_cmn: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
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
        // silk_SMULBB( float, int ): the float is converted to opus_int16 (truncation)
        let g = (ps_enc_ctrl.lt_pred_cod_gain as i32) as i16 as i32;
        let prod = silk_smulbb(g, round_loss);
        let mut idx = (prod > silk_log2lin(2900 - s_cmn.snr_db_q7)) as i8;
        idx += (prod > silk_log2lin(3900 - s_cmn.snr_db_q7)) as i8;
        s_cmn.indices.ltp_scale_index = idx;
    } else {
        // Default is minimum scaling
        s_cmn.indices.ltp_scale_index = 0;
    }

    ps_enc_ctrl.ltp_scale =
        SILK_LTPSCALES_TABLE_Q14[s_cmn.indices.ltp_scale_index as usize] as f32 / 16384.0f32;
}

// ---------------------------------------------------------------------------------------------
// find_pred_coefs_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/find_pred_coefs_FLP.c:silk_find_pred_coefs_FLP` — find LPC and LTP
/// coefficients.
///
/// `res_pitch` is `(res_buf, res_off)` (C `res_pitch = &res_buf[res_off]`, the pitch residual
/// frame); the speech signal is `(x_buf, x_off)`.
pub fn silk_find_pred_coefs_flp(
    s_cmn: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
    res_buf: &[f32],
    res_off: usize,
    x_buf: &[f32],
    x_off: usize,
    cond_coding: i32,
) {
    let mut xx_ltp = [0f32; MNSF * LTPO * LTPO];
    let mut xx_ltp_vec = [0f32; MNSF * LTPO];
    let mut inv_gains = [0f32; MNSF];
    // Set to NLSF_Q15 to zero so we don't copy junk to the state.
    let mut nlsf_q15 = [0i16; MLPC];
    let mut lpc_in_pre = [0f32; MNSF * MLPC + MFL];

    let nb_subfr = s_cmn.nb_subfr as usize;
    let subfr_length = s_cmn.subfr_length as usize;
    let order = s_cmn.predict_lpc_order as usize;

    // Weighting for weighted least squares
    for i in 0..nb_subfr {
        inv_gains[i] = 1.0f32 / ps_enc_ctrl.gains[i];
    }

    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // VOICED
        celt_assert!(
            s_cmn.ltp_mem_length - s_cmn.predict_lpc_order
                >= ps_enc_ctrl.pitch_l[0] + LTP_ORDER / 2
        );

        // LTP analysis
        silk_find_ltp_flp(
            &mut xx_ltp,
            &mut xx_ltp_vec,
            res_buf,
            res_off,
            &ps_enc_ctrl.pitch_l,
            subfr_length,
            nb_subfr,
        );

        // Quantize LTP gain parameters
        silk_quant_ltp_gains_flp(
            &mut ps_enc_ctrl.ltp_coef,
            &mut s_cmn.indices.ltp_index,
            &mut s_cmn.indices.per_index,
            &mut s_cmn.sum_log_gain_q7,
            &mut ps_enc_ctrl.lt_pred_cod_gain,
            &xx_ltp,
            &xx_ltp_vec,
            s_cmn.subfr_length,
            nb_subfr,
        );

        // Control LTP scaling
        silk_ltp_scale_ctrl_flp(s_cmn, ps_enc_ctrl, cond_coding);

        // Create LTP residual
        silk_ltp_analysis_filter_flp(
            &mut lpc_in_pre,
            x_buf,
            x_off - order,
            &ps_enc_ctrl.ltp_coef,
            &ps_enc_ctrl.pitch_l,
            &inv_gains,
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
            silk_scale_copy_vector_flp(
                &mut lpc_in_pre[x_pre_ptr..],
                &x_buf[x_ptr..],
                inv_gains[i],
                subfr_length + order,
            );
            x_pre_ptr += subfr_length + order;
            x_ptr += subfr_length;
        }
        ps_enc_ctrl.ltp_coef[..nb_subfr * LTPO].fill(0.0);
        ps_enc_ctrl.lt_pred_cod_gain = 0.0;
        s_cmn.sum_log_gain_q7 = 0;
    }

    // Limit on total predictive coding gain
    let mut min_inv_gain;
    if s_cmn.first_frame_after_reset != 0 {
        min_inv_gain = 1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET;
    } else {
        min_inv_gain = math::pow(2.0, (ps_enc_ctrl.lt_pred_cod_gain / 3.0f32) as f64) as f32
            / MAX_PREDICTION_POWER_GAIN;
        min_inv_gain /= 0.25f32 + 0.75f32 * ps_enc_ctrl.coding_quality;
    }

    // LPC_in_pre contains the LTP-filtered input for voiced, and the unfiltered input for
    // unvoiced
    silk_find_lpc_flp(s_cmn, &mut nlsf_q15, &lpc_in_pre, min_inv_gain);

    // Quantize LSFs
    let prev_nlsfq_q15 = s_cmn.prev_nlsfq_q15;
    silk_process_nlsfs_flp(
        s_cmn,
        &mut ps_enc_ctrl.pred_coef,
        &mut nlsf_q15,
        &prev_nlsfq_q15,
    );

    // Calculate residual energy using quantized LPC coefficients
    silk_residual_energy_flp(
        &mut ps_enc_ctrl.res_nrg,
        &lpc_in_pre,
        &ps_enc_ctrl.pred_coef,
        &ps_enc_ctrl.gains,
        subfr_length,
        nb_subfr,
        order,
    );

    // Copy to prediction struct for use in next frame for interpolation
    s_cmn.prev_nlsfq_q15 = nlsf_q15;
}

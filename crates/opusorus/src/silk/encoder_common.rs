//! Port of silk/{encode_indices,process_NLSFs,quant_LTP_gains,VQ_WMat_EC,stereo_LR_to_MS,stereo_find_predictor,stereo_quant_pred,HP_variable_cutoff,control_SNR,control_audio_bandwidth,check_control_input}.c.

#![allow(
    clippy::too_many_arguments,
    reason = "function signatures mirror the C sources one-to-one"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::celt::entenc::EcEnc;
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, ENCODER_NUM_CHANNELS, LA_SHAPE_MS, LTP_ORDER,
    MAX_DELTA_GAIN_QUANT, MAX_LPC_ORDER, MAX_NB_SUBFR, MIN_DELTA_GAIN_QUANT, N_LEVELS_QGAIN,
    NLSF_QUANT_MAX_AMPLITUDE, STEREO_INTERP_LEN_MS, STEREO_QUANT_SUB_STEPS, STEREO_QUANT_TAB_SIZE,
    STEREO_RATIO_SMOOTH_COEF, TRANSITION_FRAMES, TYPE_VOICED,
};
use crate::silk::errors::{
    SILK_ENC_FS_NOT_SUPPORTED, SILK_ENC_INVALID_CBR_SETTING, SILK_ENC_INVALID_COMPLEXITY_SETTING,
    SILK_ENC_INVALID_DTX_SETTING, SILK_ENC_INVALID_INBAND_FEC_SETTING, SILK_ENC_INVALID_LOSS_RATE,
    SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR, SILK_ENC_PACKET_SIZE_NOT_SUPPORTED, SILK_NO_ERROR,
};
use crate::silk::macros::{
    SILK_INT32_MAX, silk_abs, silk_add_lshift32, silk_add_pos_sat32, silk_add_rshift,
    silk_div32_16, silk_div32_varq, silk_fix_const, silk_limit, silk_limit_32, silk_lshift,
    silk_lshift32, silk_max, silk_max_int, silk_min, silk_mla, silk_mul, silk_rshift,
    silk_rshift_round, silk_rshift32, silk_sat16, silk_smlabb, silk_smlawb, silk_smulbb,
    silk_smulwb, silk_sqrt_approx, silk_sub_lshift32, silk_sub32,
};
use crate::silk::nlsf::{
    silk_nlsf_encode, silk_nlsf_unpack, silk_nlsf_vq_weights_laroia, silk_nlsf2a,
};
use crate::silk::sigproc::{
    silk_inner_prod_aligned_scale, silk_interpolate, silk_lin2log, silk_log2lin, silk_sum_sqr_shift,
};
use crate::silk::structs::{SilkEncControlStruct, SilkEncoderState, StereoEncState};
use crate::silk::tables::{
    SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF, SILK_LTP_GAIN_BITS_Q5_PTRS, SILK_LTP_GAIN_ICDF_PTRS,
    SILK_LTP_PER_INDEX_ICDF, SILK_LTP_VQ_GAIN_PTRS_Q7, SILK_LTP_VQ_PTRS_Q7, SILK_LTP_VQ_SIZES,
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF, SILK_STEREO_PRED_QUANT_Q13,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tuning_parameters::{
    MAX_SUM_LOG_GAIN_DB, VARIABLE_HP_MAX_CUTOFF_HZ, VARIABLE_HP_MAX_DELTA_FREQ,
    VARIABLE_HP_MIN_CUTOFF_HZ, VARIABLE_HP_SMTH_COEF1,
};

const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const LTPO: usize = LTP_ORDER as usize;

/// `SILK_FIX_CONST( C, Q )` for a `float` constant `C` (the tuning parameters are `...f`
/// literals): `C * (opus_int64)1 << Q` is a float multiply, `+ 0.5` a double add.
const fn silk_fix_const_f32(c: f32, q: i32) -> i32 {
    ((c * (1i64 << q) as f32) as f64 + 0.5) as i32
}

// ---------------------------------------------------------------------------------------------
// encode_indices.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/encode_indices.c:silk_encode_indices — encode side-information parameters to
/// payload.
///
/// Encodes `ps_enc_c.indices_lbrr[ frame_index ]` if `encode_lbrr != 0`, else
/// `ps_enc_c.indices`; updates `ec_prev_lag_index` / `ec_prev_signal_type`.
pub fn silk_encode_indices(
    ps_enc_c: &mut SilkEncoderState,
    ps_range_enc: &mut EcEnc<'_>,
    frame_index: usize,
    encode_lbrr: i32,
    cond_coding: i32,
) {
    let mut ec_ix = [0i16; MLPC];
    let mut pred_q8 = [0u8; MLPC];

    let ps_indices = if encode_lbrr != 0 {
        ps_enc_c.indices_lbrr[frame_index]
    } else {
        ps_enc_c.indices
    };
    let cb = ps_enc_c.ps_nlsf_cb;

    // Encode signal type and quantizer offset
    let type_offset = 2 * ps_indices.signal_type as i32 + ps_indices.quant_offset_type as i32;
    debug_assert!((0..6).contains(&type_offset));
    debug_assert!(encode_lbrr == 0 || type_offset >= 2);
    if encode_lbrr != 0 || type_offset >= 2 {
        ps_range_enc.enc_icdf((type_offset - 2) as usize, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        ps_range_enc.enc_icdf(type_offset as usize, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }

    // Encode gains
    // first subframe
    if cond_coding == CODE_CONDITIONALLY {
        // conditional coding
        debug_assert!(
            ps_indices.gains_indices[0] >= 0
                && (ps_indices.gains_indices[0] as i32)
                    < MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1
        );
        ps_range_enc.enc_icdf(
            ps_indices.gains_indices[0] as usize,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    } else {
        // independent coding, in two stages: MSB bits followed by 3 LSBs
        debug_assert!(
            ps_indices.gains_indices[0] >= 0
                && (ps_indices.gains_indices[0] as i32) < N_LEVELS_QGAIN
        );
        ps_range_enc.enc_icdf(
            silk_rshift(ps_indices.gains_indices[0] as i32, 3) as usize,
            &SILK_GAIN_ICDF[ps_indices.signal_type as usize],
            8,
        );
        ps_range_enc.enc_icdf(
            (ps_indices.gains_indices[0] & 7) as usize,
            &SILK_UNIFORM8_ICDF,
            8,
        );
    }

    // remaining subframes
    for i in 1..ps_enc_c.nb_subfr as usize {
        debug_assert!(
            ps_indices.gains_indices[i] >= 0
                && (ps_indices.gains_indices[i] as i32)
                    < MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1
        );
        ps_range_enc.enc_icdf(
            ps_indices.gains_indices[i] as usize,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    }

    // Encode NLSFs
    ps_range_enc.enc_icdf(
        ps_indices.nlsf_indices[0] as usize,
        &cb.cb1_icdf[(ps_indices.signal_type as usize >> 1) * cb.n_vectors as usize..],
        8,
    );
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        cb,
        ps_indices.nlsf_indices[0] as i32,
    );
    debug_assert!(cb.order as i32 == ps_enc_c.predict_lpc_order);
    for i in 0..cb.order as usize {
        let idx = ps_indices.nlsf_indices[i + 1] as i32;
        let icdf = &cb.ec_icdf[ec_ix[i] as usize..];
        if idx >= NLSF_QUANT_MAX_AMPLITUDE {
            ps_range_enc.enc_icdf((2 * NLSF_QUANT_MAX_AMPLITUDE) as usize, icdf, 8);
            ps_range_enc.enc_icdf(
                (idx - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if idx <= -NLSF_QUANT_MAX_AMPLITUDE {
            ps_range_enc.enc_icdf(0, icdf, 8);
            ps_range_enc.enc_icdf(
                (-idx - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            ps_range_enc.enc_icdf((idx + NLSF_QUANT_MAX_AMPLITUDE) as usize, icdf, 8);
        }
    }

    // Encode NLSF interpolation factor
    if ps_enc_c.nb_subfr == MAX_NB_SUBFR {
        debug_assert!(ps_indices.nlsf_interp_coef_q2 >= 0 && ps_indices.nlsf_interp_coef_q2 < 5);
        ps_range_enc.enc_icdf(
            ps_indices.nlsf_interp_coef_q2 as usize,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }

    if ps_indices.signal_type as i32 == TYPE_VOICED {
        // Encode pitch lags
        // lag index
        let mut encode_absolute_lag_index = true;
        if cond_coding == CODE_CONDITIONALLY && ps_enc_c.ec_prev_signal_type == TYPE_VOICED {
            // Delta Encoding
            let mut delta_lag_index =
                ps_indices.lag_index as i32 - ps_enc_c.ec_prev_lag_index as i32;
            if !(-8..=11).contains(&delta_lag_index) {
                delta_lag_index = 0;
            } else {
                delta_lag_index += 9;
                encode_absolute_lag_index = false; // Only use delta
            }
            debug_assert!((0..21).contains(&delta_lag_index));
            ps_range_enc.enc_icdf(delta_lag_index as usize, &SILK_PITCH_DELTA_ICDF, 8);
        }
        if encode_absolute_lag_index {
            // Absolute encoding
            let pitch_high_bits =
                silk_div32_16(ps_indices.lag_index as i32, silk_rshift(ps_enc_c.fs_khz, 1));
            let pitch_low_bits = ps_indices.lag_index as i32
                - silk_smulbb(pitch_high_bits, silk_rshift(ps_enc_c.fs_khz, 1));
            debug_assert!(pitch_low_bits < ps_enc_c.fs_khz / 2);
            debug_assert!(pitch_high_bits < 32);
            ps_range_enc.enc_icdf(pitch_high_bits as usize, &SILK_PITCH_LAG_ICDF, 8);
            ps_range_enc.enc_icdf(pitch_low_bits as usize, ps_enc_c.pitch_lag_low_bits_icdf, 8);
        }
        ps_enc_c.ec_prev_lag_index = ps_indices.lag_index;

        // Contour index
        debug_assert!(ps_indices.contour_index >= 0);
        debug_assert!(
            (ps_indices.contour_index < 34 && ps_enc_c.fs_khz > 8 && ps_enc_c.nb_subfr == 4)
                || (ps_indices.contour_index < 11
                    && ps_enc_c.fs_khz == 8
                    && ps_enc_c.nb_subfr == 4)
                || (ps_indices.contour_index < 12 && ps_enc_c.fs_khz > 8 && ps_enc_c.nb_subfr == 2)
                || (ps_indices.contour_index < 3 && ps_enc_c.fs_khz == 8 && ps_enc_c.nb_subfr == 2)
        );
        ps_range_enc.enc_icdf(
            ps_indices.contour_index as usize,
            ps_enc_c.pitch_contour_icdf,
            8,
        );

        // Encode LTP gains
        // PERIndex value
        debug_assert!(ps_indices.per_index >= 0 && ps_indices.per_index < 3);
        ps_range_enc.enc_icdf(ps_indices.per_index as usize, &SILK_LTP_PER_INDEX_ICDF, 8);

        // Codebook Indices
        for k in 0..ps_enc_c.nb_subfr as usize {
            debug_assert!(
                ps_indices.ltp_index[k] >= 0
                    && (ps_indices.ltp_index[k] as i32) < (8 << ps_indices.per_index)
            );
            ps_range_enc.enc_icdf(
                ps_indices.ltp_index[k] as usize,
                SILK_LTP_GAIN_ICDF_PTRS[ps_indices.per_index as usize],
                8,
            );
        }

        // Encode LTP scaling
        if cond_coding == CODE_INDEPENDENTLY {
            debug_assert!(ps_indices.ltp_scale_index >= 0 && ps_indices.ltp_scale_index < 3);
            ps_range_enc.enc_icdf(ps_indices.ltp_scale_index as usize, &SILK_LTPSCALE_ICDF, 8);
        }
        debug_assert!(cond_coding == 0 || ps_indices.ltp_scale_index == 0);
    }

    ps_enc_c.ec_prev_signal_type = ps_indices.signal_type as i32;

    // Encode seed
    debug_assert!(ps_indices.seed >= 0 && ps_indices.seed < 4);
    ps_range_enc.enc_icdf(ps_indices.seed as usize, &SILK_UNIFORM4_ICDF, 8);
}

// ---------------------------------------------------------------------------------------------
// process_NLSFs.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/process_NLSFs.c:silk_process_NLSFs — limit, stabilize, convert and quantize
/// NLSFs.
///
/// Writes `ps_enc_c.indices.nlsf_indices`. C passes `prev_NLSFq_Q15 =
/// psEnc->sCmn.prev_NLSFq_Q15`, which aliases the encoder state; Rust callers pass a copy.
pub fn silk_process_nlsfs(
    ps_enc_c: &mut SilkEncoderState,
    pred_coef_q12: &mut [[i16; MLPC]; 2],
    p_nlsf_q15: &mut [i16],
    prev_nlsfq_q15: &[i16],
) {
    let mut p_nlsf0_temp_q15 = [0i16; MLPC];
    let mut p_nlsfw_qw = [0i16; MLPC];
    let mut p_nlsfw0_temp_qw = [0i16; MLPC];
    let order = ps_enc_c.predict_lpc_order as usize;

    debug_assert!(ps_enc_c.speech_activity_q8 >= 0);
    debug_assert!(ps_enc_c.speech_activity_q8 <= silk_fix_const(1.0, 8));
    debug_assert!(
        ps_enc_c.use_interpolated_nlsfs == 1 || ps_enc_c.indices.nlsf_interp_coef_q2 == (1 << 2)
    );

    // Calculate mu values
    // NLSF_mu  = 0.003 - 0.0015 * psEnc->speech_activity;
    let mut nlsf_mu_q20 = silk_smlawb(
        silk_fix_const(0.003, 20),
        silk_fix_const(-0.001, 28),
        ps_enc_c.speech_activity_q8,
    );
    if ps_enc_c.nb_subfr == 2 {
        // Multiply by 1.5 for 10 ms packets
        nlsf_mu_q20 = silk_add_rshift(nlsf_mu_q20, nlsf_mu_q20, 1);
    }

    debug_assert!(nlsf_mu_q20 > 0);
    debug_assert!(nlsf_mu_q20 <= silk_fix_const(0.005, 20));

    // Calculate NLSF weights
    silk_nlsf_vq_weights_laroia(&mut p_nlsfw_qw, p_nlsf_q15, order);

    // Update NLSF weights for interpolated NLSFs
    let interp_coef_q2 = ps_enc_c.indices.nlsf_interp_coef_q2 as i32;
    let do_interpolate = ps_enc_c.use_interpolated_nlsfs == 1 && interp_coef_q2 < 4;
    if do_interpolate {
        // Calculate the interpolated NLSF vector for the first half
        silk_interpolate(
            &mut p_nlsf0_temp_q15,
            prev_nlsfq_q15,
            p_nlsf_q15,
            interp_coef_q2,
            order,
        );

        // Calculate first half NLSF weights for the interpolated NLSFs
        silk_nlsf_vq_weights_laroia(&mut p_nlsfw0_temp_qw, &p_nlsf0_temp_q15, order);

        // Update NLSF weights with contribution from first half
        let i_sqr_q15 = silk_lshift(silk_smulbb(interp_coef_q2, interp_coef_q2), 11) as i16;
        for i in 0..order {
            // C: silk_ADD16( ... ) stored to opus_int16
            p_nlsfw_qw[i] = (silk_rshift(p_nlsfw_qw[i] as i32, 1)
                + silk_rshift(
                    silk_smulbb(p_nlsfw0_temp_qw[i] as i32, i_sqr_q15 as i32),
                    16,
                )) as i16;
            debug_assert!(p_nlsfw_qw[i] >= 1);
        }
    }

    let signal_type = ps_enc_c.indices.signal_type as i32;
    silk_nlsf_encode(
        &mut ps_enc_c.indices.nlsf_indices,
        p_nlsf_q15,
        ps_enc_c.ps_nlsf_cb,
        &p_nlsfw_qw,
        nlsf_mu_q20,
        ps_enc_c.nlsf_msvq_survivors as usize,
        signal_type,
    );

    // Convert quantized NLSFs back to LPC coefficients
    silk_nlsf2a(&mut pred_coef_q12[1], p_nlsf_q15, order);

    if do_interpolate {
        // Calculate the interpolated, quantized LSF vector for the first half
        silk_interpolate(
            &mut p_nlsf0_temp_q15,
            prev_nlsfq_q15,
            p_nlsf_q15,
            interp_coef_q2,
            order,
        );

        // Convert back to LPC coefficients
        silk_nlsf2a(&mut pred_coef_q12[0], &p_nlsf0_temp_q15, order);
    } else {
        // Copy LPC coefficients for first half from second half
        debug_assert!(order <= MLPC);
        let (first, second) = pred_coef_q12.split_at_mut(1);
        first[0][..order].copy_from_slice(&second[0][..order]);
    }
}

// ---------------------------------------------------------------------------------------------
// quant_LTP_gains.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/quant_LTP_gains.c:silk_quant_LTP_gains.
///
/// * `b_q14`: `[ MAX_NB_SUBFR * LTP_ORDER ]` quantized LTP gains (out)
/// * `cbk_index`: `[ MAX_NB_SUBFR ]` codebook indices (out)
/// * `xx_q17`: `[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ]` correlation matrix
/// * `x_x_q17`: `[ MAX_NB_SUBFR * LTP_ORDER ]` correlation vector
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    subfr_len: i32,
    nb_subfr: usize,
) {
    let mut temp_idx = [0i8; MNSF];
    let mut res_nrg_q15: i32 = 0;
    // C leaves `gain_Q7` uninitialized; it is always written by silk_VQ_WMat_EC_c before being
    // read unless no codebook vector has a non-negative quantization error.
    let mut gain_q7: i32 = 0;

    // iterate over different codebooks with different rates/distortions, and choose best
    let mut min_rate_dist_q7 = SILK_INT32_MAX;
    let mut best_sum_log_gain_q7 = 0;
    for k in 0..3 {
        // Safety margin for pitch gain control, to take into account factors
        // such as state rescaling/rewhitening.
        let gain_safety = silk_fix_const(0.4, 7);

        let cl_ptr_q5 = SILK_LTP_GAIN_BITS_Q5_PTRS[k];
        let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[k];
        let cbk_gain_ptr_q7 = SILK_LTP_VQ_GAIN_PTRS_Q7[k];
        let cbk_size = SILK_LTP_VQ_SIZES[k] as i32;

        res_nrg_q15 = 0;
        let mut rate_dist_q7: i32 = 0;
        let mut sum_log_gain_tmp_q7 = *sum_log_gain_q7;
        for j in 0..nb_subfr {
            let max_gain_q7 = silk_log2lin(
                (silk_fix_const(MAX_SUM_LOG_GAIN_DB as f64 / 6.0, 7) - sum_log_gain_tmp_q7)
                    + silk_fix_const(7.0, 7),
            ) - gain_safety;
            let mut res_nrg_q15_subfr = 0;
            let mut rate_dist_q7_subfr = 0;
            silk_vq_wmat_ec_c(
                &mut temp_idx[j],
                &mut res_nrg_q15_subfr,
                &mut rate_dist_q7_subfr,
                &mut gain_q7,
                &xx_q17[j * LTPO * LTPO..(j + 1) * LTPO * LTPO],
                &x_x_q17[j * LTPO..(j + 1) * LTPO],
                cbk_ptr_q7,
                cbk_gain_ptr_q7,
                cl_ptr_q5,
                subfr_len,
                max_gain_q7,
                cbk_size,
            );

            res_nrg_q15 = silk_add_pos_sat32(res_nrg_q15, res_nrg_q15_subfr);
            rate_dist_q7 = silk_add_pos_sat32(rate_dist_q7, rate_dist_q7_subfr);
            sum_log_gain_tmp_q7 = silk_max(
                0,
                sum_log_gain_tmp_q7 + silk_lin2log(gain_safety + gain_q7) - silk_fix_const(7.0, 7),
            );
        }

        if rate_dist_q7 <= min_rate_dist_q7 {
            min_rate_dist_q7 = rate_dist_q7;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr].copy_from_slice(&temp_idx[..nb_subfr]);
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
        }
    }

    let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[*periodicity_index as usize];
    for j in 0..nb_subfr {
        for k in 0..LTPO {
            b_q14[j * LTPO + k] =
                silk_lshift(cbk_ptr_q7[cbk_index[j] as usize][k] as i32, 7) as i16;
        }
    }

    if nb_subfr == 2 {
        res_nrg_q15 = silk_rshift32(res_nrg_q15, 1);
    } else {
        res_nrg_q15 = silk_rshift32(res_nrg_q15, 2);
    }

    *sum_log_gain_q7 = best_sum_log_gain_q7;
    *pred_gain_db_q7 = silk_smulbb(-3, silk_lin2log(res_nrg_q15) - (15 << 7));
}

// ---------------------------------------------------------------------------------------------
// VQ_WMat_EC.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/VQ_WMat_EC.c:silk_VQ_WMat_EC_c — entropy constrained matrix-weighted VQ,
/// hard-coded to 5-element vectors, for a single input data vector.
///
/// `gain_q7` is only written when a codebook vector is selected (as in C).
pub fn silk_vq_wmat_ec_c(
    ind: &mut i8,
    res_nrg_q15: &mut i32,
    rate_dist_q8: &mut i32,
    gain_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    cb_q7: &[[i8; 5]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
    l: i32,
) {
    let xx = &xx_q17[..25];
    let mut neg_x_x_q24 = [0i32; 5];

    // Negate and convert to new Q domain
    neg_x_x_q24[0] = -silk_lshift32(x_x_q17[0], 7);
    neg_x_x_q24[1] = -silk_lshift32(x_x_q17[1], 7);
    neg_x_x_q24[2] = -silk_lshift32(x_x_q17[2], 7);
    neg_x_x_q24[3] = -silk_lshift32(x_x_q17[3], 7);
    neg_x_x_q24[4] = -silk_lshift32(x_x_q17[4], 7);

    // Loop over codebook
    *rate_dist_q8 = SILK_INT32_MAX;
    *res_nrg_q15 = SILK_INT32_MAX;
    // If things go really bad, at least *ind is set to something safe.
    *ind = 0;
    for k in 0..l as usize {
        let cb_row_q7 = &cb_q7[k];
        let c = |n: usize| cb_row_q7[n] as i32;
        let gain_tmp_q7 = cb_gain_q7[k] as i32;
        // Weighted rate
        // Quantization error: 1 - 2 * xX * cb + cb' * XX * cb
        let mut sum1_q15 = silk_fix_const(1.001, 15);

        // Penalty for too large gain
        let penalty = silk_lshift32(silk_max(silk_sub32(gain_tmp_q7, max_gain_q7), 0), 11);

        // first row of XX_Q17
        let mut sum2_q24 = silk_mla(neg_x_x_q24[0], xx[1], c(1));
        sum2_q24 = silk_mla(sum2_q24, xx[2], c(2));
        sum2_q24 = silk_mla(sum2_q24, xx[3], c(3));
        sum2_q24 = silk_mla(sum2_q24, xx[4], c(4));
        sum2_q24 = silk_lshift32(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[0], c(0));
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, c(0));

        // second row of XX_Q17
        sum2_q24 = silk_mla(neg_x_x_q24[1], xx[7], c(2));
        sum2_q24 = silk_mla(sum2_q24, xx[8], c(3));
        sum2_q24 = silk_mla(sum2_q24, xx[9], c(4));
        sum2_q24 = silk_lshift32(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[6], c(1));
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, c(1));

        // third row of XX_Q17
        sum2_q24 = silk_mla(neg_x_x_q24[2], xx[13], c(3));
        sum2_q24 = silk_mla(sum2_q24, xx[14], c(4));
        sum2_q24 = silk_lshift32(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[12], c(2));
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, c(2));

        // fourth row of XX_Q17
        sum2_q24 = silk_mla(neg_x_x_q24[3], xx[19], c(4));
        sum2_q24 = silk_lshift32(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[18], c(3));
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, c(3));

        // last row of XX_Q17
        sum2_q24 = silk_lshift32(neg_x_x_q24[4], 1);
        sum2_q24 = silk_mla(sum2_q24, xx[24], c(4));
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, c(4));

        // find best
        if sum1_q15 >= 0 {
            // Translate residual energy to bits using high-rate assumption (6 dB ==> 1 bit/sample)
            let bits_res_q8 = silk_smulbb(subfr_len, silk_lin2log(sum1_q15 + penalty) - (15 << 7));
            // In the following line we reduce the codelength component by half ("-1"); seems to
            // slightly improve quality
            let bits_tot_q8 = silk_add_lshift32(bits_res_q8, cl_q5[k] as i32, 3 - 1);
            if bits_tot_q8 <= *rate_dist_q8 {
                *rate_dist_q8 = bits_tot_q8;
                *res_nrg_q15 = sum1_q15 + penalty;
                *ind = k as i8;
                *gain_q7 = gain_tmp_q7;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// stereo_LR_to_MS.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/stereo_LR_to_MS.c:silk_stereo_LR_to_MS — convert Left/Right stereo signal to
/// adaptive Mid/Side representation.
///
/// `x1` / `x2` start **two samples before** the C pointers `x1` / `x2` (C reads and writes
/// `x1[ -2 ]`, `x1[ -1 ]`, `x2[ -1 ]`; the C call passes `&inputBuf[ 2 ]`, so Rust passes
/// `&mut inputBuf[..]`) and hold `frame_length + 2` samples.
///
/// C passes `ix = psEnc->sStereo.predIx[ n ]` and `mid_only_flag =
/// &psEnc->sStereo.mid_only_flags[ n ]`, which alias `state`; neither is read from `state` here,
/// so Rust callers pass separate outputs and store them into `state` afterwards.
pub fn silk_stereo_lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16],
    x2: &mut [i16],
    ix: &mut [[i8; 3]; 2],
    mid_only_flag: &mut i8,
    mid_side_rates_bps: &mut [i32],
    mut total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: i32,
    fs_khz: i32,
    frame_length: i32,
) {
    const MFL: usize = crate::silk::define::MAX_FRAME_LENGTH as usize;
    let fl = frame_length as usize;
    let mut pred_q13 = [0i32; 2];
    let mut lp_ratio_q14 = 0;
    let mut hp_ratio_q14 = 0;

    // C: mid = &x1[ -2 ] (Rust x1 already starts there)
    let mid = &mut x1[..fl + 2];
    let x2 = &mut x2[..fl + 2];
    // C: ALLOC( side, frame_length + 2 )
    let mut side_buf = [0i16; MFL + 2];
    let side = &mut side_buf[..fl + 2];
    // Convert to basic mid/side signals
    for n in 0..fl + 2 {
        let sum = mid[n] as i32 + x2[n] as i32;
        let diff = mid[n] as i32 - x2[n] as i32;
        mid[n] = silk_rshift_round(sum, 1) as i16;
        side[n] = silk_sat16(silk_rshift_round(diff, 1)) as i16;
    }

    // Buffering
    mid[..2].copy_from_slice(&state.s_mid);
    side[..2].copy_from_slice(&state.s_side);
    state.s_mid.copy_from_slice(&mid[fl..fl + 2]);
    state.s_side.copy_from_slice(&side[fl..fl + 2]);

    // LP and HP filter mid signal
    let mut lp_mid_buf = [0i16; MFL];
    let mut hp_mid_buf = [0i16; MFL];
    let lp_mid = &mut lp_mid_buf[..fl];
    let hp_mid = &mut hp_mid_buf[..fl];
    for n in 0..fl {
        let sum = silk_rshift_round(
            silk_add_lshift32(mid[n] as i32 + mid[n + 2] as i32, mid[n + 1] as i32, 1),
            2,
        );
        lp_mid[n] = sum as i16;
        hp_mid[n] = (mid[n + 1] as i32 - sum) as i16;
    }

    // LP and HP filter side signal
    let mut lp_side_buf = [0i16; MFL];
    let mut hp_side_buf = [0i16; MFL];
    let lp_side = &mut lp_side_buf[..fl];
    let hp_side = &mut hp_side_buf[..fl];
    for n in 0..fl {
        let sum = silk_rshift_round(
            silk_add_lshift32(side[n] as i32 + side[n + 2] as i32, side[n + 1] as i32, 1),
            2,
        );
        lp_side[n] = sum as i16;
        hp_side[n] = (side[n + 1] as i32 - sum) as i16;
    }

    // Find energies and predictors
    let is10ms_frame = frame_length == 10 * fs_khz;
    let mut smooth_coef_q16 = if is10ms_frame {
        silk_fix_const(STEREO_RATIO_SMOOTH_COEF / 2.0, 16)
    } else {
        silk_fix_const(STEREO_RATIO_SMOOTH_COEF, 16)
    };
    smooth_coef_q16 = silk_smulwb(
        silk_smulbb(prev_speech_act_q8, prev_speech_act_q8),
        smooth_coef_q16,
    );

    pred_q13[0] = silk_stereo_find_predictor(
        &mut lp_ratio_q14,
        lp_mid,
        lp_side,
        &mut state.mid_side_amp_q0[0..2],
        fl,
        smooth_coef_q16,
    );
    pred_q13[1] = silk_stereo_find_predictor(
        &mut hp_ratio_q14,
        hp_mid,
        hp_side,
        &mut state.mid_side_amp_q0[2..4],
        fl,
        smooth_coef_q16,
    );
    // Ratio of the norms of residual and mid signals
    let mut frac_q16 = silk_smlabb(hp_ratio_q14, lp_ratio_q14, 3);
    frac_q16 = silk_min(frac_q16, silk_fix_const(1.0, 16));

    // Determine bitrate distribution between mid and side, and possibly reduce stereo width
    total_rate_bps -= if is10ms_frame { 1200 } else { 600 }; // Subtract approximate bitrate for coding stereo parameters
    if total_rate_bps < 1 {
        total_rate_bps = 1;
    }
    let min_mid_rate_bps = silk_smlabb(2000, fs_khz, 600);
    debug_assert!(min_mid_rate_bps < 32767);
    // Default bitrate distribution: 8 parts for Mid and (5+3*frac) parts for Side. so:
    // mid_rate = ( 8 / ( 13 + 3 * frac ) ) * total_ rate
    let frac_3_q16 = silk_mul(3, frac_q16);
    mid_side_rates_bps[0] = silk_div32_varq(
        total_rate_bps,
        silk_fix_const(8.0 + 5.0, 16) + frac_3_q16,
        16 + 3,
    );
    let mut width_q14;
    // If Mid bitrate below minimum, reduce stereo width
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        // width = 4 * ( 2 * side_rate - min_rate ) / ( ( 1 + 3 * frac ) * min_rate )
        width_q14 = silk_div32_varq(
            silk_lshift(mid_side_rates_bps[1], 1) - min_mid_rate_bps,
            silk_smulwb(silk_fix_const(1.0, 16) + frac_3_q16, min_mid_rate_bps),
            14 + 2,
        );
        width_q14 = silk_limit(width_q14, 0, silk_fix_const(1.0, 14));
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = silk_fix_const(1.0, 14);
    }

    // Smoother
    state.smth_width_q14 = silk_smlawb(
        state.smth_width_q14 as i32,
        width_q14 - state.smth_width_q14 as i32,
        smooth_coef_q16,
    ) as i16;

    // At very low bitrates or for inputs that are nearly amplitude panned, switch to panned-mono
    // coding
    *mid_only_flag = 0;
    let smth_width_q14 = state.smth_width_q14 as i32;
    if to_mono != 0 {
        // Last frame before stereo->mono transition; collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
        silk_stereo_quant_pred(&mut pred_q13, ix);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || silk_smulwb(frac_q16, smth_width_q14) < silk_fix_const(0.05, 14))
    {
        // Code as panned-mono; previous frame already had zero width
        // Scale down and quantize predictors
        pred_q13[0] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[0]), 14);
        pred_q13[1] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[1]), 14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        // Collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        *mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || silk_smulwb(frac_q16, smth_width_q14) < silk_fix_const(0.02, 14))
    {
        // Transition to zero-width stereo
        // Scale down and quantize predictors
        pred_q13[0] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[0]), 14);
        pred_q13[1] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[1]), 14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        // Collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
    } else if smth_width_q14 > silk_fix_const(0.95, 14) {
        // Full-width stereo coding
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = silk_fix_const(1.0, 14);
    } else {
        // Reduced-width stereo coding; scale down and quantize predictors
        pred_q13[0] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[0]), 14);
        pred_q13[1] = silk_rshift(silk_smulbb(smth_width_q14, pred_q13[1]), 14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = smth_width_q14;
    }

    // Make sure to keep on encoding until the tapered output has been transmitted
    if *mid_only_flag == 1 {
        // C: opus_int16 += int (truncated on store)
        state.silent_side_len =
            (state.silent_side_len as i32 + frame_length - STEREO_INTERP_LEN_MS * fs_khz) as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS * fs_khz {
            *mid_only_flag = 0;
        } else {
            // Limit to avoid wrapping around
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }

    if *mid_only_flag == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = silk_max_int(1, total_rate_bps - mid_side_rates_bps[1]);
    }

    // Interpolate predictors and subtract prediction from side channel
    let mut pred0_q13 = -(state.pred_prev_q13[0] as i32);
    let mut pred1_q13 = -(state.pred_prev_q13[1] as i32);
    let mut w_q24 = silk_lshift(state.width_prev_q14 as i32, 10);
    let denom_q16 = silk_div32_16(1i32 << 16, STEREO_INTERP_LEN_MS * fs_khz);
    let delta0_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );
    let deltaw_q24 = silk_lshift(
        silk_smulwb(width_q14 - state.width_prev_q14 as i32, denom_q16),
        10,
    );
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        w_q24 += deltaw_q24;
        let mut sum = silk_lshift(
            silk_add_lshift32(mid[n] as i32 + mid[n + 2] as i32, mid[n + 1] as i32, 1),
            9,
        ); // Q11
        sum = silk_smlawb(silk_smulwb(w_q24, side[n + 1] as i32), sum, pred0_q13); // Q8
        sum = silk_smlawb(sum, silk_lshift(mid[n + 1] as i32, 11), pred1_q13); // Q8
        // C: x2[ n - 1 ]
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }

    pred0_q13 = -pred_q13[0];
    pred1_q13 = -pred_q13[1];
    w_q24 = silk_lshift(width_q14, 10);
    for n in interp_len..fl {
        let mut sum = silk_lshift(
            silk_add_lshift32(mid[n] as i32 + mid[n + 2] as i32, mid[n + 1] as i32, 1),
            9,
        ); // Q11
        sum = silk_smlawb(silk_smulwb(w_q24, side[n + 1] as i32), sum, pred0_q13); // Q8
        sum = silk_smlawb(sum, silk_lshift(mid[n + 1] as i32, 11), pred1_q13); // Q8
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;
    state.width_prev_q14 = width_q14 as i16;
}

// ---------------------------------------------------------------------------------------------
// stereo_find_predictor.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/stereo_find_predictor.c:silk_stereo_find_predictor — find least-squares
/// prediction gain for one signal based on another and quantize it. Returns the predictor in
/// Q13.
///
/// `mid_res_amp_q0`: `[ 2 ]` smoothed mid and residual norms (I/O).
pub fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32],
    length: usize,
    mut smooth_coef_q16: i32,
) -> i32 {
    // Find  predictor
    let (mut nrgx, scale1) = silk_sum_sqr_shift(x, length);
    let (mut nrgy, scale2) = silk_sum_sqr_shift(y, length);
    let mut scale = silk_max_int(scale1, scale2);
    scale += scale & 1; // make even
    nrgy = silk_rshift32(nrgy, scale - scale2);
    nrgx = silk_rshift32(nrgx, scale - scale1);
    nrgx = silk_max_int(nrgx, 1);
    let corr = silk_inner_prod_aligned_scale(x, y, scale, length);
    let mut pred_q13 = silk_div32_varq(corr, nrgx, 13);
    pred_q13 = silk_limit(pred_q13, -(1 << 14), 1 << 14);
    let pred2_q10 = silk_smulwb(pred_q13, pred_q13);

    // Faster update for signals with large prediction parameters
    smooth_coef_q16 = silk_max_int(smooth_coef_q16, silk_abs(pred2_q10));

    // Smoothed mid and residual norms
    debug_assert!(smooth_coef_q16 < 32768);
    scale = silk_rshift(scale, 1);
    mid_res_amp_q0[0] = silk_smlawb(
        mid_res_amp_q0[0],
        silk_lshift(silk_sqrt_approx(nrgx), scale) - mid_res_amp_q0[0],
        smooth_coef_q16,
    );
    // Residual energy = nrgy - 2 * pred * corr + pred^2 * nrgx
    nrgy = silk_sub_lshift32(nrgy, silk_smulwb(corr, pred_q13), 3 + 1);
    nrgy = silk_add_lshift32(nrgy, silk_smulwb(nrgx, pred2_q10), 6);
    mid_res_amp_q0[1] = silk_smlawb(
        mid_res_amp_q0[1],
        silk_lshift(silk_sqrt_approx(nrgy), scale) - mid_res_amp_q0[1],
        smooth_coef_q16,
    );

    // Ratio of smoothed residual and mid norms
    *ratio_q14 = silk_div32_varq(mid_res_amp_q0[1], silk_max(mid_res_amp_q0[0], 1), 14);
    *ratio_q14 = silk_limit(*ratio_q14, 0, 32767);

    pred_q13
}

// ---------------------------------------------------------------------------------------------
// stereo_quant_pred.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/stereo_quant_pred.c:silk_stereo_quant_pred — quantize mid/side predictors.
pub fn silk_stereo_quant_pred(pred_q13: &mut [i32; 2], ix: &mut [[i8; 3]; 2]) {
    let mut quant_pred_q13: i32 = 0;

    // Quantize
    for n in 0..2 {
        // Brute-force search over quantization levels
        let mut err_min_q13 = SILK_INT32_MAX;
        'done: for i in 0..(STEREO_QUANT_TAB_SIZE - 1) as usize {
            let low_q13 = SILK_STEREO_PRED_QUANT_Q13[i] as i32;
            let step_q13 = silk_smulwb(
                SILK_STEREO_PRED_QUANT_Q13[i + 1] as i32 - low_q13,
                silk_fix_const(0.5 / STEREO_QUANT_SUB_STEPS as f64, 16),
            );
            for j in 0..STEREO_QUANT_SUB_STEPS {
                let lvl_q13 = silk_smlabb(low_q13, step_q13, 2 * j + 1);
                let err_q13 = silk_abs(pred_q13[n] - lvl_q13);
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // Error increasing, so we're past the optimum
                    break 'done;
                }
            }
        }
        ix[n][2] = silk_div32_16(ix[n][0] as i32, 3) as i8;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }

    // Subtract second from first predictor (helps when actually applying these)
    pred_q13[0] -= pred_q13[1];
}

// ---------------------------------------------------------------------------------------------
// HP_variable_cutoff.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/HP_variable_cutoff.c:silk_HP_variable_cutoff — high-pass filter with cutoff
/// frequency adaptation based on pitch lag statistics.
///
/// C takes the `silk_encoder_state_Fxx state_Fxx[]` array and only touches
/// `state_Fxx[ 0 ].sCmn`; Rust callers pass `&mut state_fxx[ 0 ].s_cmn`.
pub const fn silk_hp_variable_cutoff(ps_enc_c1: &mut SilkEncoderState) {
    // Adaptive cutoff frequency: estimate low end of pitch frequency range
    if ps_enc_c1.prev_signal_type as i32 == TYPE_VOICED {
        // difference, in log domain
        let pitch_freq_hz_q16 = silk_div32_16(
            silk_lshift(silk_mul(ps_enc_c1.fs_khz, 1000), 16),
            ps_enc_c1.prev_lag,
        );
        let mut pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - (16 << 7);

        // adjustment based on quality
        let quality_q15 = ps_enc_c1.input_quality_bands_q15[0];
        pitch_freq_log_q7 = silk_smlawb(
            pitch_freq_log_q7,
            silk_smulwb(silk_lshift(-quality_q15, 2), quality_q15),
            pitch_freq_log_q7
                - (silk_lin2log(silk_fix_const(VARIABLE_HP_MIN_CUTOFF_HZ as f64, 16)) - (16 << 7)),
        );

        // delta_freq = pitch_freq_log - psEnc->variable_HP_smth1;
        let mut delta_freq_q7 = pitch_freq_log_q7 - silk_rshift(ps_enc_c1.variable_hp_smth1_q15, 8);
        if delta_freq_q7 < 0 {
            // less smoothing for decreasing pitch frequency, to track something close to the
            // minimum
            delta_freq_q7 = silk_mul(delta_freq_q7, 3);
        }

        // limit delta, to reduce impact of outliers in pitch estimation
        delta_freq_q7 = silk_limit_32(
            delta_freq_q7,
            -silk_fix_const_f32(VARIABLE_HP_MAX_DELTA_FREQ, 7),
            silk_fix_const_f32(VARIABLE_HP_MAX_DELTA_FREQ, 7),
        );

        // update smoother
        ps_enc_c1.variable_hp_smth1_q15 = silk_smlawb(
            ps_enc_c1.variable_hp_smth1_q15,
            silk_smulbb(ps_enc_c1.speech_activity_q8, delta_freq_q7),
            silk_fix_const_f32(VARIABLE_HP_SMTH_COEF1, 16),
        );

        // limit frequency range
        ps_enc_c1.variable_hp_smth1_q15 = silk_limit_32(
            ps_enc_c1.variable_hp_smth1_q15,
            silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8),
            silk_lshift(silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ), 8),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// control_SNR.c
// ---------------------------------------------------------------------------------------------

// These tables hold SNR values divided by 21 (so they fit in 8 bits)
// for different target bitrates spaced at 400 bps interval. The first
// 10 values are omitted (0-4 kb/s) because they're all zeros.
// These tables were obtained by running different SNRs through the
// encoder and measuring the active bitrate.

/// `silk_TargetRate_NB_21`.
pub static SILK_TARGET_RATE_NB_21: [u8; 117 - 10] = [
    0, 15, 39, 52, 61, 68, 74, 79, 84, 88, 92, 95, 99, 102, 105, 108, 111, 114, 117, 119, 122, 124,
    126, 129, 131, 133, 135, 137, 139, 142, 143, 145, 147, 149, 151, 153, 155, 157, 158, 160, 162,
    163, 165, 167, 168, 170, 171, 173, 174, 176, 177, 179, 180, 182, 183, 185, 186, 187, 189, 190,
    192, 193, 194, 196, 197, 199, 200, 201, 203, 204, 205, 207, 208, 209, 211, 212, 213, 215, 216,
    217, 219, 220, 221, 223, 224, 225, 227, 228, 230, 231, 232, 234, 235, 236, 238, 239, 241, 242,
    243, 245, 246, 248, 249, 250, 252, 253, 255,
];

/// `silk_TargetRate_MB_21`.
pub static SILK_TARGET_RATE_MB_21: [u8; 165 - 10] = [
    0, 0, 28, 43, 52, 59, 65, 70, 74, 78, 81, 85, 87, 90, 93, 95, 98, 100, 102, 105, 107, 109, 111,
    113, 115, 116, 118, 120, 122, 123, 125, 127, 128, 130, 131, 133, 134, 136, 137, 138, 140, 141,
    143, 144, 145, 147, 148, 149, 151, 152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165,
    166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
    186, 187, 188, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 236, 237,
    238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
];

/// `silk_TargetRate_WB_21`.
pub static SILK_TARGET_RATE_WB_21: [u8; 201 - 10] = [
    0, 0, 0, 8, 29, 41, 49, 56, 62, 66, 70, 74, 77, 80, 83, 86, 88, 91, 93, 95, 97, 99, 101, 103,
    105, 107, 108, 110, 112, 113, 115, 116, 118, 119, 121, 122, 123, 125, 126, 127, 129, 130, 131,
    132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 156, 157, 158, 159, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
    171, 172, 173, 174, 175, 176, 177, 177, 178, 179, 180, 181, 181, 182, 183, 184, 185, 185, 186,
    187, 188, 189, 189, 190, 191, 192, 192, 193, 194, 195, 195, 196, 197, 198, 198, 199, 200, 200,
    201, 202, 203, 203, 204, 205, 206, 206, 207, 208, 209, 209, 210, 211, 211, 212, 213, 214, 214,
    215, 216, 216, 217, 218, 219, 219, 220, 221, 221, 222, 223, 224, 224, 225, 226, 226, 227, 228,
    229, 229, 230, 231, 232, 232, 233, 234, 234, 235, 236, 237, 237, 238, 239, 240, 240, 241, 242,
    243, 243, 244, 245, 246, 246, 247, 248, 249, 249, 250, 251, 252, 253, 255,
];

/// Port of silk/control_SNR.c:silk_control_SNR — control SNR of residual quantizer. Returns
/// `SILK_NO_ERROR`.
pub fn silk_control_snr(ps_enc_c: &mut SilkEncoderState, mut target_rate_bps: i32) -> i32 {
    ps_enc_c.target_rate_bps = target_rate_bps;
    if ps_enc_c.nb_subfr == 2 {
        target_rate_bps -= 2000 + ps_enc_c.fs_khz / 16;
    }
    let snr_table: &[u8] = if ps_enc_c.fs_khz == 8 {
        &SILK_TARGET_RATE_NB_21
    } else if ps_enc_c.fs_khz == 12 {
        &SILK_TARGET_RATE_MB_21
    } else {
        &SILK_TARGET_RATE_WB_21
    };
    let bound = snr_table.len() as i32;
    let mut id = (target_rate_bps + 200) / 400;
    id = silk_min(id - 10, bound - 1);
    if id <= 0 {
        ps_enc_c.snr_db_q7 = 0;
    } else {
        ps_enc_c.snr_db_q7 = snr_table[id as usize] as i32 * 21;
    }
    SILK_NO_ERROR
}

// ---------------------------------------------------------------------------------------------
// control_audio_bandwidth.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/control_audio_bandwidth.c:silk_control_audio_bandwidth — control internal
/// sampling rate. Returns the internal sampling rate in kHz.
pub const fn silk_control_audio_bandwidth(
    ps_enc_c: &mut SilkEncoderState,
    enc_control: &mut SilkEncControlStruct,
) -> i32 {
    let mut orig_khz = ps_enc_c.fs_khz;
    // Handle a bandwidth-switching reset where we need to be aware what the last sampling rate
    // was.
    if orig_khz == 0 {
        orig_khz = ps_enc_c.s_lp.saved_fs_khz;
    }
    let mut fs_khz = orig_khz;
    let mut fs_hz = silk_smulbb(fs_khz, 1000);
    if fs_hz == 0 {
        // Encoder has just been initialized
        fs_hz = silk_min(ps_enc_c.desired_internal_fs_hz, ps_enc_c.api_fs_hz);
        fs_khz = silk_div32_16(fs_hz, 1000);
    } else if fs_hz > ps_enc_c.api_fs_hz
        || fs_hz > ps_enc_c.max_internal_fs_hz
        || fs_hz < ps_enc_c.min_internal_fs_hz
    {
        // Make sure internal rate is not higher than external rate or maximum allowed, or lower
        // than minimum allowed
        fs_hz = ps_enc_c.api_fs_hz;
        fs_hz = silk_min(fs_hz, ps_enc_c.max_internal_fs_hz);
        fs_hz = silk_max(fs_hz, ps_enc_c.min_internal_fs_hz);
        fs_khz = silk_div32_16(fs_hz, 1000);
    } else {
        // State machine for the internal sampling rate switching
        if ps_enc_c.s_lp.transition_frame_no >= TRANSITION_FRAMES {
            // Stop transition phase
            ps_enc_c.s_lp.mode = 0;
        }
        if ps_enc_c.allow_bandwidth_switch != 0 || enc_control.opus_can_switch != 0 {
            // Check if we should switch down
            if silk_smulbb(orig_khz, 1000) > ps_enc_c.desired_internal_fs_hz {
                // Switch down
                if ps_enc_c.s_lp.mode == 0 {
                    // New transition
                    ps_enc_c.s_lp.transition_frame_no = TRANSITION_FRAMES;

                    // Reset transition filter state
                    ps_enc_c.s_lp.in_lp_state = [0; 2];
                }
                if enc_control.opus_can_switch != 0 {
                    // Stop transition phase
                    ps_enc_c.s_lp.mode = 0;

                    // Switch to a lower sample frequency
                    fs_khz = if orig_khz == 16 { 12 } else { 8 };
                } else if ps_enc_c.s_lp.transition_frame_no <= 0 {
                    enc_control.switch_ready = 1;
                    // Make room for redundancy
                    enc_control.max_bits -=
                        enc_control.max_bits * 5 / (enc_control.payload_size_ms + 5);
                } else {
                    // Direction: down (at double speed)
                    ps_enc_c.s_lp.mode = -2;
                }
            } else if silk_smulbb(orig_khz, 1000) < ps_enc_c.desired_internal_fs_hz {
                // Check if we should switch up
                // Switch up
                if enc_control.opus_can_switch != 0 {
                    // Switch to a higher sample frequency
                    fs_khz = if orig_khz == 8 { 12 } else { 16 };

                    // New transition
                    ps_enc_c.s_lp.transition_frame_no = 0;

                    // Reset transition filter state
                    ps_enc_c.s_lp.in_lp_state = [0; 2];

                    // Direction: up
                    ps_enc_c.s_lp.mode = 1;
                } else if ps_enc_c.s_lp.mode == 0 {
                    enc_control.switch_ready = 1;
                    // Make room for redundancy
                    enc_control.max_bits -=
                        enc_control.max_bits * 5 / (enc_control.payload_size_ms + 5);
                } else {
                    // Direction: up
                    ps_enc_c.s_lp.mode = 1;
                }
            } else if ps_enc_c.s_lp.mode < 0 {
                ps_enc_c.s_lp.mode = 1;
            }
        }
    }

    fs_khz
}

// ---------------------------------------------------------------------------------------------
// check_control_input.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/check_control_input.c:check_control_input — check encoder control struct.
///
/// Returns `SILK_NO_ERROR` or the first error code found. The C code executes
/// `celt_assert( 0 )` before each error return; that assertion is not reproduced so invalid
/// user settings are reported as errors (what a C build without assertions does).
#[must_use]
pub const fn check_control_input(enc_control: &SilkEncControlStruct) -> i32 {
    let api_ok = matches!(
        enc_control.api_sample_rate,
        8000 | 12000 | 16000 | 24000 | 32000 | 44100 | 48000
    ) || api_rate_qext_ok(enc_control.api_sample_rate);
    if !api_ok
        || !internal_ok(enc_control.desired_internal_sample_rate)
        || !internal_ok(enc_control.max_internal_sample_rate)
        || !internal_ok(enc_control.min_internal_sample_rate)
        || enc_control.min_internal_sample_rate > enc_control.desired_internal_sample_rate
        || enc_control.max_internal_sample_rate < enc_control.desired_internal_sample_rate
        || enc_control.min_internal_sample_rate > enc_control.max_internal_sample_rate
    {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if enc_control.payload_size_ms != 10
        && enc_control.payload_size_ms != 20
        && enc_control.payload_size_ms != 40
        && enc_control.payload_size_ms != 60
    {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if enc_control.packet_loss_percentage < 0 || enc_control.packet_loss_percentage > 100 {
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if enc_control.use_dtx < 0 || enc_control.use_dtx > 1 {
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if enc_control.use_cbr < 0 || enc_control.use_cbr > 1 {
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if enc_control.use_in_band_fec < 0 || enc_control.use_in_band_fec > 1 {
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if enc_control.n_channels_api < 1 || enc_control.n_channels_api > ENCODER_NUM_CHANNELS {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal < 1 || enc_control.n_channels_internal > ENCODER_NUM_CHANNELS
    {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal > enc_control.n_channels_api {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.complexity < 0 || enc_control.complexity > 10 {
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    SILK_NO_ERROR
}

/// Valid internal sample rate for `check_control_input`.
const fn internal_ok(rate: i32) -> bool {
    matches!(rate, 8000 | 12000 | 16000)
}

/// The `ENABLE_QEXT` API sample rate accepted by `check_control_input` (96 kHz).
#[cfg(feature = "qext")]
const fn api_rate_qext_ok(rate: i32) -> bool {
    rate == 96000
}

/// Without `ENABLE_QEXT` no extra API sample rate is accepted.
#[cfg(not(feature = "qext"))]
const fn api_rate_qext_ok(_rate: i32) -> bool {
    false
}

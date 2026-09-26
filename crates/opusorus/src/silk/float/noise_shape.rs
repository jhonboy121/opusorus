//! Port of `silk/float/noise_shape_analysis_FLP.c` and `silk/float/process_gains_FLP.c`.

use crate::math;
use crate::silk::coding::silk_gains_quant;
use crate::silk::define::{
    CODE_CONDITIONALLY, MAX_SHAPE_LPC_ORDER, MIN_QGAIN_DB, SHAPE_LPC_WIN_MAX, SUB_FRAME_LENGTH_MS,
    TYPE_VOICED, USE_HARM_SHAPING,
};
use crate::silk::float::sigproc::{
    silk_abs_float, silk_apply_sine_window_flp, silk_autocorrelation_flp, silk_bwexpander_flp,
    silk_energy_flp, silk_k2a_flp, silk_log2, silk_min_float, silk_schur_flp, silk_sigmoid,
    silk_warped_autocorrelation_flp,
};
use crate::silk::float::structs::{SilkEncoderControlFlp, SilkShapeStateFlp};
use crate::silk::macros::silk_smulbb;
use crate::silk::structs::SilkEncoderState;
use crate::silk::tables::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::tuning_parameters::{
    BANDWIDTH_EXPANSION, BG_SNR_DECR_dB, ENERGY_VARIATION_THRESHOLD_QNT_OFFSET,
    FIND_PITCH_WHITE_NOISE_FRACTION, HARM_HP_NOISE_COEF, HARM_SNR_INCR_dB, HARMONIC_SHAPING,
    HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, HP_NOISE_COEF, LAMBDA_CODING_QUALITY,
    LAMBDA_DELAYED_DECISIONS, LAMBDA_INPUT_QUALITY, LAMBDA_OFFSET, LAMBDA_QUANT_OFFSET,
    LAMBDA_SPEECH_ACT, LOW_FREQ_SHAPING, LOW_QUALITY_LOW_FREQ_SHAPING_DECR,
    SHAPE_WHITE_NOISE_FRACTION, SUBFR_SMTH_COEF,
};

const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;

// ---------------------------------------------------------------------------------------------
// noise_shape_analysis_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of the static `noise_shape_analysis_FLP.c:warped_gain` — gain that makes warped filter
/// coefficients have a zero mean log frequency response on a non-warped frequency scale.
#[must_use]
pub fn warped_gain(coefs: &[f32], mut lambda: f32, order: usize) -> f32 {
    lambda = -lambda;
    let mut gain = coefs[order - 1];
    for i in (0..order - 1).rev() {
        gain = lambda * gain + coefs[i];
    }
    1.0f32 / (1.0f32 - lambda * gain)
}

/// Port of the static `noise_shape_analysis_FLP.c:warped_true2monic_coefs` — convert warped
/// filter coefficients to monic pseudo-warped coefficients and limit the maximum amplitude of
/// the monic warped coefficients by bandwidth expansion on the true coefficients.
pub fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: usize) {
    let coefs = &mut coefs[..order];
    let mut ind = 0usize;

    // Convert to monic coefficients
    for i in (1..order).rev() {
        coefs[i - 1] -= lambda * coefs[i];
    }
    let mut gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs[0]);
    for c in coefs.iter_mut() {
        *c *= gain;
    }

    // Limit
    for iter in 0..10 {
        // Find maximum absolute value
        let mut maxabs = -1.0f32;
        for (i, &c) in coefs.iter().enumerate() {
            let tmp = silk_abs_float(c);
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            // Coefficients are within range - done
            return;
        }

        // Convert back to true warped coefficients
        for i in 1..order {
            coefs[i - 1] += lambda * coefs[i];
        }
        gain = 1.0f32 / gain;
        for c in coefs.iter_mut() {
            *c *= gain;
        }

        // Apply bandwidth expansion
        let chirp = 0.99f32
            - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);

        // Convert to monic warped coefficients
        for i in (1..order).rev() {
            coefs[i - 1] -= lambda * coefs[i];
        }
        gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs[0]);
        for c in coefs.iter_mut() {
            *c *= gain;
        }
    }
    // C: silk_assert( 0 ): not reached for encoder input; the unit tests drive it.
    assertion_failure!("0");
}

/// Port of the static `noise_shape_analysis_FLP.c:limit_coefs` — limit the maximum absolute
/// coefficient value by bandwidth expansion.
pub fn limit_coefs(coefs: &mut [f32], limit: f32, order: usize) {
    let coefs = &mut coefs[..order];
    let mut ind = 0usize;
    for iter in 0..10 {
        // Find maximum absolute value
        let mut maxabs = -1.0f32;
        for (i, &c) in coefs.iter().enumerate() {
            let tmp = silk_abs_float(c);
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            // Coefficients are within range - done
            return;
        }

        // Apply bandwidth expansion
        let chirp = 0.99f32
            - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);
    }
    // C: silk_assert( 0 ): not reached for encoder input; the unit tests drive it.
    assertion_failure!("0");
}

/// Port of `silk/float/noise_shape_analysis_FLP.c:silk_noise_shape_analysis_FLP` — compute noise
/// shaping coefficients and initial gain values.
///
/// `pitch_res` is the LPC residual from the pitch analysis (frame part); the input signal is
/// `x_buf[x_off..]` (C `x`; the analysis starts `la_shape` samples before it). `ltp_corr` is
/// `psEnc->LTPCorr`.
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_noise_shape_analysis_flp(
    s_cmn: &mut SilkEncoderState,
    ps_shape_st: &mut SilkShapeStateFlp,
    ltp_corr: f32,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
    pitch_res: &[f32],
    x_buf: &[f32],
    x_off: usize,
) {
    let mut x_windowed = [0f32; SHAPE_LPC_WIN_MAX as usize];
    let mut auto_corr = [0f32; MSO + 1];
    let mut rc = [0f32; MSO + 1];

    let nb_subfr = s_cmn.nb_subfr as usize;

    // Point to start of first LPC analysis block
    let mut x_ptr = x_off - s_cmn.la_shape as usize;

    // GAIN CONTROL
    let mut snr_adj_db = s_cmn.snr_db_q7 as f32 * (1.0f32 / 128.0f32);

    // Input quality is the average of the quality in the lowest two VAD bands
    ps_enc_ctrl.input_quality = 0.5f32
        * (s_cmn.input_quality_bands_q15[0] + s_cmn.input_quality_bands_q15[1]) as f32
        * (1.0f32 / 32768.0f32);

    // Coding quality level, between 0.0 and 1.0
    ps_enc_ctrl.coding_quality = silk_sigmoid(0.25f32 * (snr_adj_db - 20.0f32));

    if s_cmn.use_cbr == 0 {
        // Reduce coding SNR during low speech activity
        let b = 1.0f32 - s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);
        snr_adj_db -= BG_SNR_DECR_dB
            * ps_enc_ctrl.coding_quality
            * (0.5f32 + 0.5f32 * ps_enc_ctrl.input_quality)
            * b
            * b;
    }

    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Reduce gains for periodic signals
        snr_adj_db += HARM_SNR_INCR_dB * ltp_corr;
    } else {
        // For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB
        // setting
        snr_adj_db += (-0.4f32 * s_cmn.snr_db_q7 as f32 * (1.0f32 / 128.0f32) + 6.0f32)
            * (1.0f32 - ps_enc_ctrl.input_quality);
    }

    // SPARSENESS PROCESSING
    // Set quantizer offset
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Initially set to 0; may be overruled in process_gains(..)
        s_cmn.indices.quant_offset_type = 0;
    } else {
        // Sparseness measure, based on relative fluctuations of energy per 2 milliseconds
        let n_samples = (2 * s_cmn.fs_khz) as usize;
        let mut energy_variation = 0.0f32;
        let mut log_energy_prev = 0.0f32;
        let mut pitch_res_ptr = 0usize;
        let n_segs = silk_smulbb(SUB_FRAME_LENGTH_MS, s_cmn.nb_subfr) / 2;
        for k in 0..n_segs {
            let nrg =
                n_samples as f32 + silk_energy_flp(&pitch_res[pitch_res_ptr..], n_samples) as f32;
            let log_energy = silk_log2(nrg as f64);
            if k > 0 {
                energy_variation += silk_abs_float(log_energy - log_energy_prev);
            }
            log_energy_prev = log_energy;
            pitch_res_ptr += n_samples;
        }

        // Set quantization offset depending on sparseness measure
        if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs - 1) as f32 {
            s_cmn.indices.quant_offset_type = 0;
        } else {
            s_cmn.indices.quant_offset_type = 1;
        }
    }

    // Control bandwidth expansion
    // More BWE for signals with high prediction gain
    let mut strength = FIND_PITCH_WHITE_NOISE_FRACTION * ps_enc_ctrl.pred_gain; // between 0.0 and 1.0
    let bw_exp = BANDWIDTH_EXPANSION / (1.0f32 + strength * strength);

    // Slightly more warping in analysis will move quantization noise up in frequency, where
    // it's better masked
    let warping = s_cmn.warping_q16 as f32 / 65536.0f32 + 0.01f32 * ps_enc_ctrl.coding_quality;

    // Compute noise shaping AR coefs and gains
    let shape_win_length = s_cmn.shape_win_length as usize;
    let shaping_order = s_cmn.shaping_lpc_order as usize;
    for k in 0..nb_subfr {
        // Apply window: sine slope followed by flat part followed by cosine slope
        let flat_part = (s_cmn.fs_khz * 3) as usize;
        let slope_part = (shape_win_length - flat_part) / 2;

        silk_apply_sine_window_flp(&mut x_windowed, &x_buf[x_ptr..], 1, slope_part);
        let mut shift = slope_part;
        x_windowed[shift..shift + flat_part]
            .copy_from_slice(&x_buf[x_ptr + shift..x_ptr + shift + flat_part]);
        shift += flat_part;
        silk_apply_sine_window_flp(
            &mut x_windowed[shift..],
            &x_buf[x_ptr + shift..],
            2,
            slope_part,
        );

        // Update pointer: next LPC analysis block
        x_ptr += s_cmn.subfr_length as usize;

        if s_cmn.warping_q16 > 0 {
            // Calculate warped auto correlation
            silk_warped_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed,
                warping,
                shape_win_length,
                shaping_order,
            );
        } else {
            // Calculate regular auto correlation
            silk_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed,
                shape_win_length,
                shaping_order + 1,
            );
        }

        // Add white noise, as a fraction of energy
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0f32;

        // Convert correlations to prediction coefficients, and compute residual energy
        let nrg = silk_schur_flp(&mut rc, &auto_corr, shaping_order);
        let ar = &mut ps_enc_ctrl.ar[k * MSO..(k + 1) * MSO];
        silk_k2a_flp(ar, &rc, shaping_order);
        ps_enc_ctrl.gains[k] = math::sqrt(nrg as f64) as f32;

        if s_cmn.warping_q16 > 0 {
            // Adjust gain for warping
            ps_enc_ctrl.gains[k] *= warped_gain(ar, warping, shaping_order);
        }

        // Bandwidth expansion for synthesis filter shaping
        silk_bwexpander_flp(ar, shaping_order, bw_exp);

        if s_cmn.warping_q16 > 0 {
            // Convert to monic warped prediction coefficients and limit absolute values
            warped_true2monic_coefs(ar, warping, 3.999f32, shaping_order);
        } else {
            // Limit absolute values
            limit_coefs(ar, 3.999f32, shaping_order);
        }
    }

    // Gain tweaking
    // Increase gains during low speech activity
    let gain_mult = math::pow(2.0, (-0.16f32 * snr_adj_db) as f64) as f32;
    let gain_add = math::pow(2.0, (0.16f32 * MIN_QGAIN_DB as f32) as f64) as f32;
    for k in 0..nb_subfr {
        ps_enc_ctrl.gains[k] *= gain_mult;
        ps_enc_ctrl.gains[k] += gain_add;
    }

    // Control low-frequency shaping and noise tilt
    // Less low frequency shaping for noisy inputs
    strength = LOW_FREQ_SHAPING
        * (1.0f32
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR
                * (s_cmn.input_quality_bands_q15[0] as f32 * (1.0f32 / 32768.0f32) - 1.0f32));
    strength *= s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);

    let tilt = if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Reduce low frequencies quantization noise for periodic signals, depending on pitch lag
        for k in 0..nb_subfr {
            let b = 0.2f32 / s_cmn.fs_khz as f32 + 3.0f32 / ps_enc_ctrl.pitch_l[k] as f32;
            ps_enc_ctrl.lf_ma_shp[k] = -1.0f32 + b;
            ps_enc_ctrl.lf_ar_shp[k] = 1.0f32 - b - b * strength;
        }
        -HP_NOISE_COEF
            - (1.0f32 - HP_NOISE_COEF)
                * HARM_HP_NOISE_COEF
                * s_cmn.speech_activity_q8 as f32
                * (1.0f32 / 256.0f32)
    } else {
        let b = 1.3f32 / s_cmn.fs_khz as f32;
        ps_enc_ctrl.lf_ma_shp[0] = -1.0f32 + b;
        ps_enc_ctrl.lf_ar_shp[0] = 1.0f32 - b - b * strength * 0.6f32;
        for k in 1..nb_subfr {
            ps_enc_ctrl.lf_ma_shp[k] = ps_enc_ctrl.lf_ma_shp[0];
            ps_enc_ctrl.lf_ar_shp[k] = ps_enc_ctrl.lf_ar_shp[0];
        }
        -HP_NOISE_COEF
    };

    // HARMONIC SHAPING CONTROL
    let mut harm_shape_gain;
    if USE_HARM_SHAPING != 0 && s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Harmonic noise shaping
        harm_shape_gain = HARMONIC_SHAPING;

        // More harmonic noise shaping for high bitrates or noisy input
        harm_shape_gain += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            * (1.0f32 - (1.0f32 - ps_enc_ctrl.coding_quality) * ps_enc_ctrl.input_quality);

        // Less harmonic noise shaping for less periodic signals
        harm_shape_gain *= math::sqrt(ltp_corr as f64) as f32;
    } else {
        harm_shape_gain = 0.0f32;
    }

    // Smooth over subframes
    for k in 0..nb_subfr {
        ps_shape_st.harm_shape_gain_smth +=
            SUBFR_SMTH_COEF * (harm_shape_gain - ps_shape_st.harm_shape_gain_smth);
        ps_enc_ctrl.harm_shape_gain[k] = ps_shape_st.harm_shape_gain_smth;
        ps_shape_st.tilt_smth += SUBFR_SMTH_COEF * (tilt - ps_shape_st.tilt_smth);
        ps_enc_ctrl.tilt[k] = ps_shape_st.tilt_smth;
    }
}

// ---------------------------------------------------------------------------------------------
// process_gains_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/process_gains_FLP.c:silk_process_gains_FLP` — processing of gains.
pub fn silk_process_gains_flp(
    s_cmn: &mut SilkEncoderState,
    ps_shape_st: &mut SilkShapeStateFlp,
    ps_enc_ctrl: &mut SilkEncoderControlFlp,
    cond_coding: i32,
) {
    let nb_subfr = s_cmn.nb_subfr as usize;
    let mut p_gains_q16 = [0i32; 4];

    // Gain reduction when LTP coding gain is high
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        let s = 1.0f32 - 0.5f32 * silk_sigmoid(0.25f32 * (ps_enc_ctrl.lt_pred_cod_gain - 12.0f32));
        for k in 0..nb_subfr {
            ps_enc_ctrl.gains[k] *= s;
        }
    }

    // Limit the quantized signal
    let inv_max_sqr_val = (math::pow(
        2.0,
        (0.33f32 * (21.0f32 - s_cmn.snr_db_q7 as f32 * (1.0f32 / 128.0f32))) as f64,
    ) / s_cmn.subfr_length as f64) as f32;

    for k in 0..nb_subfr {
        // Soft limit on ratio residual energy and squared gains
        let mut gain = ps_enc_ctrl.gains[k];
        gain = math::sqrt((gain * gain + ps_enc_ctrl.res_nrg[k] * inv_max_sqr_val) as f64) as f32;
        ps_enc_ctrl.gains[k] = silk_min_float(gain, 32767.0f32);
    }

    // Prepare gains for noise shaping quantization
    for k in 0..nb_subfr {
        p_gains_q16[k] = (ps_enc_ctrl.gains[k] * 65536.0f32) as i32;
    }

    // Save unquantized gains and gain Index
    ps_enc_ctrl.gains_unq_q16[..nb_subfr].copy_from_slice(&p_gains_q16[..nb_subfr]);
    ps_enc_ctrl.last_gain_index_prev = ps_shape_st.last_gain_index;

    // Quantize gains
    silk_gains_quant(
        &mut s_cmn.indices.gains_indices,
        &mut p_gains_q16,
        &mut ps_shape_st.last_gain_index,
        (cond_coding == CODE_CONDITIONALLY) as i32,
        nb_subfr,
    );

    // Overwrite unquantized gains with quantized gains and convert back to Q0 from Q16
    for k in 0..nb_subfr {
        ps_enc_ctrl.gains[k] = p_gains_q16[k] as f32 / 65536.0f32;
    }

    // Set quantizer offset for voiced signals. Larger offset when LTP coding gain is low or
    // tilt is high (ie low-pass)
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        if ps_enc_ctrl.lt_pred_cod_gain + s_cmn.input_tilt_q15 as f32 * (1.0f32 / 32768.0f32)
            > 1.0f32
        {
            s_cmn.indices.quant_offset_type = 0;
        } else {
            s_cmn.indices.quant_offset_type = 1;
        }
    }

    // Quantizer boundary adjustment
    let quant_offset = SILK_QUANTIZATION_OFFSETS_Q10[(s_cmn.indices.signal_type >> 1) as usize]
        [s_cmn.indices.quant_offset_type as usize] as f32
        / 1024.0f32;
    ps_enc_ctrl.lambda = LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * s_cmn.n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32)
        + LAMBDA_INPUT_QUALITY * ps_enc_ctrl.input_quality
        + LAMBDA_CODING_QUALITY * ps_enc_ctrl.coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset;
}

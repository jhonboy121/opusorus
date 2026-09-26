//! Port of `silk/fixed/noise_shape_analysis_FIX.c` and `silk/fixed/process_gains_FIX.c`: noise
//! shaping analysis and gain processing.

use crate::silk::coding::silk_gains_quant;
use crate::silk::define::{
    CODE_CONDITIONALLY, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER, MIN_QGAIN_DB, SHAPE_LPC_WIN_MAX,
    SUB_FRAME_LENGTH_MS, TYPE_VOICED, USE_HARM_SHAPING,
};
use crate::silk::fixed::sigproc::{
    silk_apply_sine_window, silk_autocorr, silk_k2a_q16, silk_schur64,
    silk_warped_autocorrelation_fix,
};
use crate::silk::fixed::structs::{SilkEncoderControlFix, SilkShapeStateFix};
use crate::silk::macros::{
    silk_abs, silk_abs_int32, silk_add_pos_sat32, silk_add_sat32, silk_add32, silk_div32_16,
    silk_div32_varq, silk_fix_const, silk_inverse32_varq, silk_lshift, silk_lshift_sat32,
    silk_lshift32, silk_max_32, silk_min, silk_mul, silk_rshift, silk_rshift_round, silk_sat16,
    silk_smlabb, silk_smlawb, silk_smlaww, silk_smmul, silk_smulbb, silk_smulwb, silk_smulww,
    silk_sqrt_approx,
};
use crate::silk::sigproc::{
    silk_bwexpander_32, silk_lin2log, silk_log2lin, silk_lpc_fit, silk_sigm_q15, silk_sum_sqr_shift,
};
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

const MNSF: usize = MAX_NB_SUBFR as usize;
const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;

/// `SILK_FIX_CONST( C, Q )` of a C `float` tuning constant.
#[inline(always)]
const fn fc(c: f32, q: i32) -> i32 {
    silk_fix_const(c as f64, q)
}

/// Port of the static `noise_shape_analysis_FIX.c:warped_gain` — compute gain (Q16) to make
/// warped filter coefficients have a zero mean log frequency response on a non-warped
/// frequency scale (so that it can be implemented with a minimum-phase monic filter).
///
/// Note: a monic filter is one with the first coefficient equal to 1.0. In Silk we omit the
/// first coefficient in an array of coefficients, for monic filters.
#[must_use]
pub fn warped_gain(coefs_q24: &[i32], mut lambda_q16: i32, order: usize) -> i32 {
    lambda_q16 = -lambda_q16;
    let mut gain_q24 = coefs_q24[order - 1];
    for i in (0..order - 1).rev() {
        gain_q24 = silk_smlawb(coefs_q24[i], gain_q24, lambda_q16);
    }
    gain_q24 = silk_smlawb(silk_fix_const(1.0, 24), gain_q24, -lambda_q16);
    silk_inverse32_varq(gain_q24, 40)
}

/// Port of the static `noise_shape_analysis_FIX.c:limit_warped_coefs` — convert warped filter
/// coefficients to monic pseudo-warped coefficients and limit the maximum amplitude of monic
/// warped coefficients by using bandwidth expansion on the true coefficients.
pub fn limit_warped_coefs(
    coefs_q24: &mut [i32],
    mut lambda_q16: i32,
    limit_q24: i32,
    order: usize,
) {
    let mut ind = 0usize;

    // Convert to monic coefficients
    lambda_q16 = -lambda_q16;
    for i in (1..order).rev() {
        coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
    }
    lambda_q16 = -lambda_q16;
    let mut nom_q16 = silk_smlawb(silk_fix_const(1.0, 16), -lambda_q16, lambda_q16);
    let mut den_q24 = silk_smlawb(silk_fix_const(1.0, 24), coefs_q24[0], lambda_q16);
    let mut gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
    for c in &mut coefs_q24[..order] {
        *c = silk_smulww(gain_q16, *c);
    }
    let limit_q20 = silk_rshift(limit_q24, 4);
    for iter in 0..10 {
        // Find maximum absolute value
        let mut maxabs_q24 = -1;
        for i in 0..order {
            let tmp = silk_abs_int32(coefs_q24[i]);
            if tmp > maxabs_q24 {
                maxabs_q24 = tmp;
                ind = i;
            }
        }
        // Use Q20 to avoid any overflow when multiplying by (ind + 1) later.
        let maxabs_q20 = silk_rshift(maxabs_q24, 4);
        if maxabs_q20 <= limit_q20 {
            // Coefficients are within range - done
            return;
        }

        // Convert back to true warped coefficients
        for i in 1..order {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
        }
        gain_q16 = silk_inverse32_varq(gain_q16, 32);
        for c in &mut coefs_q24[..order] {
            *c = silk_smulww(gain_q16, *c);
        }

        // Apply bandwidth expansion
        let chirp_q16 = silk_fix_const(0.99, 16)
            - silk_div32_varq(
                silk_smulwb(
                    maxabs_q20 - limit_q20,
                    silk_smlabb(silk_fix_const(0.8, 10), silk_fix_const(0.1, 10), iter),
                ),
                silk_mul(maxabs_q20, ind as i32 + 1),
                22,
            );
        silk_bwexpander_32(coefs_q24, order, chirp_q16);

        // Convert to monic warped coefficients
        lambda_q16 = -lambda_q16;
        for i in (1..order).rev() {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
        }
        lambda_q16 = -lambda_q16;
        nom_q16 = silk_smlawb(silk_fix_const(1.0, 16), -lambda_q16, lambda_q16);
        den_q24 = silk_smlawb(silk_fix_const(1.0, 24), coefs_q24[0], lambda_q16);
        gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
        for c in &mut coefs_q24[..order] {
            *c = silk_smulww(gain_q16, *c);
        }
    }
    // C: silk_assert( 0 ) (debug builds only)
}

/// Port of `silk/fixed/noise_shape_analysis_FIX.c:silk_noise_shape_analysis_FIX` — compute
/// noise shaping coefficients and initial gain values.
///
/// `pitch_res` is the LPC residual from pitch analysis (from the frame start); the input signal
/// is `(x_buf, x_off)` (C `x = &x_buf[x_off]`, read from `x - la_shape`).
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_noise_shape_analysis_fix(
    s_cmn: &mut SilkEncoderState,
    ps_shape_st: &mut SilkShapeStateFix,
    ltp_corr_q15: i32,
    ps_enc_ctrl: &mut SilkEncoderControlFix,
    pitch_res: &[i16],
    x_buf: &[i16],
    x_off: usize,
) {
    let mut auto_corr = [0i32; MSO + 1];
    let mut refl_coef_q16 = [0i32; MSO];
    let mut ar_q24 = [0i32; MSO];
    let mut x_windowed = [0i16; SHAPE_LPC_WIN_MAX as usize];

    // Point to start of first LPC analysis block
    let mut x_ptr = x_off - s_cmn.la_shape as usize;

    // ****************
    // GAIN CONTROL
    // ****************
    let mut snr_adj_db_q7 = s_cmn.snr_db_q7;

    // Input quality is the average of the quality in the lowest two VAD bands
    ps_enc_ctrl.input_quality_q14 = silk_rshift(
        s_cmn.input_quality_bands_q15[0] + s_cmn.input_quality_bands_q15[1],
        2,
    );

    // Coding quality level, between 0.0_Q0 and 1.0_Q0, but in Q14
    ps_enc_ctrl.coding_quality_q14 = silk_rshift(
        silk_sigm_q15(silk_rshift_round(
            snr_adj_db_q7 - silk_fix_const(20.0, 7),
            4,
        )),
        1,
    );

    // Reduce coding SNR during low speech activity
    if s_cmn.use_cbr == 0 {
        let mut b_q8 = silk_fix_const(1.0, 8) - s_cmn.speech_activity_q8;
        b_q8 = silk_smulwb(silk_lshift(b_q8, 8), b_q8);
        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smulbb(fc(-BG_SNR_DECR_dB, 7) >> (4 + 1), b_q8), // Q11
            silk_smulwb(
                silk_fix_const(1.0, 14) + ps_enc_ctrl.input_quality_q14,
                ps_enc_ctrl.coding_quality_q14,
            ), // Q12
        );
    }

    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Reduce gains for periodic signals
        snr_adj_db_q7 = silk_smlawb(snr_adj_db_q7, fc(HARM_SNR_INCR_dB, 8), ltp_corr_q15);
    } else {
        // For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB
        // setting
        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smlawb(
                silk_fix_const(6.0, 9),
                -silk_fix_const(0.4, 18),
                s_cmn.snr_db_q7,
            ),
            silk_fix_const(1.0, 14) - ps_enc_ctrl.input_quality_q14,
        );
    }

    // ***********************
    // SPARSENESS PROCESSING
    // ***********************
    // Set quantizer offset
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Initially set to 0; may be overruled in process_gains(..)
        s_cmn.indices.quant_offset_type = 0;
    } else {
        // Sparseness measure, based on relative fluctuations of energy per 2 milliseconds
        let n_samples = silk_lshift(s_cmn.fs_khz, 1);
        let mut energy_variation_q7 = 0i32;
        let mut log_energy_prev_q7 = 0i32;
        let mut pitch_res_ptr = 0usize;
        let n_segs = silk_smulbb(SUB_FRAME_LENGTH_MS, s_cmn.nb_subfr) / 2;
        for k in 0..n_segs {
            let (mut nrg, scale) =
                silk_sum_sqr_shift(&pitch_res[pitch_res_ptr..], n_samples as usize);
            nrg += silk_rshift(n_samples, scale); // Q(-scale)

            let log_energy_q7 = silk_lin2log(nrg);
            if k > 0 {
                energy_variation_q7 += silk_abs(log_energy_q7 - log_energy_prev_q7);
            }
            log_energy_prev_q7 = log_energy_q7;
            pitch_res_ptr += n_samples as usize;
        }

        // Set quantization offset depending on sparseness measure
        if energy_variation_q7 > fc(ENERGY_VARIATION_THRESHOLD_QNT_OFFSET, 7) * (n_segs - 1) {
            s_cmn.indices.quant_offset_type = 0;
        } else {
            s_cmn.indices.quant_offset_type = 1;
        }
    }

    // *****************************
    // Control bandwidth expansion
    // *****************************
    // More BWE for signals with high prediction gain
    let mut strength_q16 = silk_smulwb(
        ps_enc_ctrl.pred_gain_q16,
        fc(FIND_PITCH_WHITE_NOISE_FRACTION, 16),
    );
    let bw_exp_q16 = silk_div32_varq(
        fc(BANDWIDTH_EXPANSION, 16),
        silk_smlaww(silk_fix_const(1.0, 16), strength_q16, strength_q16),
        16,
    );

    let warping_q16 = if s_cmn.warping_q16 > 0 {
        // Slightly more warping in analysis will move quantization noise up in frequency,
        // where it's better masked
        silk_smlawb(
            s_cmn.warping_q16,
            ps_enc_ctrl.coding_quality_q14,
            silk_fix_const(0.01, 18),
        )
    } else {
        0
    };

    // ******************************************
    // Compute noise shaping AR coefs and gains
    // ******************************************
    let shape_win_length = s_cmn.shape_win_length as usize;
    let shaping_lpc_order = s_cmn.shaping_lpc_order as usize;
    for k in 0..s_cmn.nb_subfr as usize {
        // Apply window: sine slope followed by flat part followed by cosine slope
        let flat_part = (s_cmn.fs_khz * 3) as usize;
        let slope_part = silk_rshift(s_cmn.shape_win_length - flat_part as i32, 1) as usize;

        silk_apply_sine_window(&mut x_windowed, &x_buf[x_ptr..], 1, slope_part);
        let mut shift = slope_part;
        x_windowed[shift..shift + flat_part]
            .copy_from_slice(&x_buf[x_ptr + shift..x_ptr + shift + flat_part]);
        shift += flat_part;
        silk_apply_sine_window(
            &mut x_windowed[shift..],
            &x_buf[x_ptr + shift..],
            2,
            slope_part,
        );

        // Update pointer: next LPC analysis block
        x_ptr += s_cmn.subfr_length as usize;

        let scale = if s_cmn.warping_q16 > 0 {
            // Calculate warped auto correlation
            silk_warped_autocorrelation_fix(
                &mut auto_corr,
                &x_windowed,
                warping_q16,
                shape_win_length,
                shaping_lpc_order,
            )
        } else {
            // Calculate regular auto correlation
            silk_autocorr(
                &mut auto_corr,
                &x_windowed,
                shape_win_length,
                shaping_lpc_order + 1,
            )
        };

        // Add white noise, as a fraction of energy
        auto_corr[0] = silk_add32(
            auto_corr[0],
            silk_max_32(
                silk_smulwb(
                    silk_rshift(auto_corr[0], 4),
                    fc(SHAPE_WHITE_NOISE_FRACTION, 20),
                ),
                1,
            ),
        );

        // Calculate the reflection coefficients using schur
        let mut nrg = silk_schur64(&mut refl_coef_q16, &auto_corr, shaping_lpc_order);
        debug_assert!(nrg >= 0);

        // Convert reflection coefficients to prediction coefficients
        silk_k2a_q16(&mut ar_q24, &refl_coef_q16, shaping_lpc_order);

        let mut qnrg = -scale; // range: -12...30
        debug_assert!(qnrg >= -12);
        debug_assert!(qnrg <= 30);

        // Make sure that Qnrg is an even number
        if qnrg & 1 != 0 {
            qnrg -= 1;
            nrg >>= 1;
        }

        let tmp32 = silk_sqrt_approx(nrg);
        qnrg >>= 1; // range: -6...15

        ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(tmp32, 16 - qnrg);

        if s_cmn.warping_q16 > 0 {
            // Adjust gain for warping
            let gain_mult_q16 = warped_gain(&ar_q24, warping_q16, shaping_lpc_order);
            debug_assert!(ps_enc_ctrl.gains_q16[k] > 0);
            if ps_enc_ctrl.gains_q16[k] < silk_fix_const(0.25, 16) {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16);
            } else {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(
                    silk_rshift_round(ps_enc_ctrl.gains_q16[k], 1),
                    gain_mult_q16,
                );
                if ps_enc_ctrl.gains_q16[k] >= (i32::MAX >> 1) {
                    ps_enc_ctrl.gains_q16[k] = i32::MAX;
                } else {
                    ps_enc_ctrl.gains_q16[k] = silk_lshift32(ps_enc_ctrl.gains_q16[k], 1);
                }
            }
            debug_assert!(ps_enc_ctrl.gains_q16[k] > 0);
        }

        // Bandwidth expansion
        silk_bwexpander_32(&mut ar_q24, shaping_lpc_order, bw_exp_q16);

        let ar_q13 = &mut ps_enc_ctrl.ar_q13[k * MSO..(k + 1) * MSO];
        if s_cmn.warping_q16 > 0 {
            // Convert to monic warped prediction coefficients and limit absolute values
            limit_warped_coefs(
                &mut ar_q24,
                warping_q16,
                silk_fix_const(3.999, 24),
                shaping_lpc_order,
            );

            // Convert from Q24 to Q13 and store in int16
            for i in 0..shaping_lpc_order {
                ar_q13[i] = silk_sat16(silk_rshift_round(ar_q24[i], 11)) as i16;
            }
        } else {
            silk_lpc_fit(ar_q13, &mut ar_q24, 13, 24, shaping_lpc_order);
        }
    }

    // ***************
    // Gain tweaking
    // ***************
    // Increase gains during low speech activity and put lower limit on gains
    let gain_mult_q16 = silk_log2lin(-silk_smlawb(
        -silk_fix_const(16.0, 7),
        snr_adj_db_q7,
        silk_fix_const(0.16, 16),
    ));
    let gain_add_q16 = silk_log2lin(silk_smlawb(
        silk_fix_const(16.0, 7),
        silk_fix_const(MIN_QGAIN_DB as f64, 7),
        silk_fix_const(0.16, 16),
    ));
    debug_assert!(gain_mult_q16 > 0);
    for k in 0..s_cmn.nb_subfr as usize {
        ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16);
        debug_assert!(ps_enc_ctrl.gains_q16[k] >= 0);
        ps_enc_ctrl.gains_q16[k] = silk_add_pos_sat32(ps_enc_ctrl.gains_q16[k], gain_add_q16);
    }

    // **********************************************
    // Control low-frequency shaping and noise tilt
    // **********************************************
    // Less low frequency shaping for noisy inputs
    strength_q16 = silk_mul(
        fc(LOW_FREQ_SHAPING, 4),
        silk_smlawb(
            silk_fix_const(1.0, 12),
            fc(LOW_QUALITY_LOW_FREQ_SHAPING_DECR, 13),
            s_cmn.input_quality_bands_q15[0] - silk_fix_const(1.0, 15),
        ),
    );
    strength_q16 = silk_rshift(silk_mul(strength_q16, s_cmn.speech_activity_q8), 8);
    let tilt_q16 = if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Reduce low frequencies quantization noise for periodic signals, depending on pitch
        // lag
        // f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])
        let fs_khz_inv = silk_div32_16(silk_fix_const(0.2, 14), s_cmn.fs_khz);
        for k in 0..s_cmn.nb_subfr as usize {
            let b_q14 = fs_khz_inv + silk_div32_16(silk_fix_const(3.0, 14), ps_enc_ctrl.pitch_l[k]);
            // Pack two coefficients in one int32
            ps_enc_ctrl.lf_shp_q14[k] = silk_lshift(
                silk_fix_const(1.0, 14) - b_q14 - silk_smulwb(strength_q16, b_q14),
                16,
            );
            ps_enc_ctrl.lf_shp_q14[k] |= (b_q14 - silk_fix_const(1.0, 14)) as u16 as i32;
        }
        // Guarantees that second argument to SMULWB() is within range of an opus_int16
        debug_assert!(fc(HARM_HP_NOISE_COEF, 24) < silk_fix_const(0.5, 24));
        -fc(HP_NOISE_COEF, 16)
            - silk_smulwb(
                silk_fix_const(1.0, 16) - fc(HP_NOISE_COEF, 16),
                silk_smulwb(fc(HARM_HP_NOISE_COEF, 24), s_cmn.speech_activity_q8),
            )
    } else {
        let b_q14 = silk_div32_16(21299, s_cmn.fs_khz); // 1.3_Q0 = 21299_Q14
        // Pack two coefficients in one int32
        ps_enc_ctrl.lf_shp_q14[0] = silk_lshift(
            silk_fix_const(1.0, 14)
                - b_q14
                - silk_smulwb(strength_q16, silk_smulwb(silk_fix_const(0.6, 16), b_q14)),
            16,
        );
        ps_enc_ctrl.lf_shp_q14[0] |= (b_q14 - silk_fix_const(1.0, 14)) as u16 as i32;
        for k in 1..s_cmn.nb_subfr as usize {
            ps_enc_ctrl.lf_shp_q14[k] = ps_enc_ctrl.lf_shp_q14[0];
        }
        -fc(HP_NOISE_COEF, 16)
    };

    // **************************
    // HARMONIC SHAPING CONTROL
    // **************************
    let harm_shape_gain_q16 =
        if USE_HARM_SHAPING != 0 && s_cmn.indices.signal_type as i32 == TYPE_VOICED {
            // More harmonic noise shaping for high bitrates or noisy input
            let mut h = silk_smlawb(
                fc(HARMONIC_SHAPING, 16),
                silk_fix_const(1.0, 16)
                    - silk_smulwb(
                        silk_fix_const(1.0, 18) - silk_lshift(ps_enc_ctrl.coding_quality_q14, 4),
                        ps_enc_ctrl.input_quality_q14,
                    ),
                fc(HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, 16),
            );

            // Less harmonic noise shaping for less periodic signals
            h = silk_smulwb(
                silk_lshift(h, 1),
                silk_sqrt_approx(silk_lshift(ltp_corr_q15, 15)),
            );
            h
        } else {
            0
        };

    // ***********************
    // Smooth over subframes
    // ***********************
    for k in 0..MNSF {
        ps_shape_st.harm_shape_gain_smth_q16 = silk_smlawb(
            ps_shape_st.harm_shape_gain_smth_q16,
            harm_shape_gain_q16 - ps_shape_st.harm_shape_gain_smth_q16,
            fc(SUBFR_SMTH_COEF, 16),
        );
        ps_shape_st.tilt_smth_q16 = silk_smlawb(
            ps_shape_st.tilt_smth_q16,
            tilt_q16 - ps_shape_st.tilt_smth_q16,
            fc(SUBFR_SMTH_COEF, 16),
        );

        ps_enc_ctrl.harm_shape_gain_q14[k] =
            silk_rshift_round(ps_shape_st.harm_shape_gain_smth_q16, 2);
        ps_enc_ctrl.tilt_q14[k] = silk_rshift_round(ps_shape_st.tilt_smth_q16, 2);
    }
}

/// Port of `silk/fixed/process_gains_FIX.c:silk_process_gains_FIX` — processing of gains.
pub fn silk_process_gains_fix(
    s_cmn: &mut SilkEncoderState,
    ps_shape_st: &mut SilkShapeStateFix,
    ps_enc_ctrl: &mut SilkEncoderControlFix,
    cond_coding: i32,
) {
    let nb_subfr = s_cmn.nb_subfr as usize;

    // Gain reduction when LTP coding gain is high
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // s = -0.5f * silk_sigmoid( 0.25f * ( psEncCtrl->LTPredCodGain - 12.0f ) );
        let s_q16 = -silk_sigm_q15(silk_rshift_round(
            ps_enc_ctrl.lt_pred_cod_gain_q7 - silk_fix_const(12.0, 7),
            4,
        ));
        for k in 0..nb_subfr {
            ps_enc_ctrl.gains_q16[k] =
                silk_smlawb(ps_enc_ctrl.gains_q16[k], ps_enc_ctrl.gains_q16[k], s_q16);
        }
    }

    // Limit the quantized signal
    // InvMaxSqrVal = pow( 2.0f, 0.33f * ( 21.0f - SNR_dB ) ) / subfr_length;
    let inv_max_sqr_val_q16 = silk_div32_16(
        silk_log2lin(silk_smulwb(
            silk_fix_const(21.0 + 16.0 / 0.33, 7) - s_cmn.snr_db_q7,
            silk_fix_const(0.33, 16),
        )),
        s_cmn.subfr_length,
    );

    for k in 0..nb_subfr {
        // Soft limit on ratio residual energy and squared gains
        let res_nrg = ps_enc_ctrl.res_nrg[k];
        let mut res_nrg_part = silk_smulww(res_nrg, inv_max_sqr_val_q16);
        if ps_enc_ctrl.res_nrg_q[k] > 0 {
            res_nrg_part = silk_rshift_round(res_nrg_part, ps_enc_ctrl.res_nrg_q[k]);
        } else if res_nrg_part >= silk_rshift(i32::MAX, -ps_enc_ctrl.res_nrg_q[k]) {
            res_nrg_part = i32::MAX;
        } else {
            res_nrg_part = silk_lshift(res_nrg_part, -ps_enc_ctrl.res_nrg_q[k]);
        }
        let mut gain = ps_enc_ctrl.gains_q16[k];
        let mut gain_squared = silk_add_sat32(res_nrg_part, silk_smmul(gain, gain));
        if gain_squared < i16::MAX as i32 {
            // recalculate with higher precision
            gain_squared = silk_smlaww(silk_lshift(res_nrg_part, 16), gain, gain);
            debug_assert!(gain_squared > 0);
            gain = silk_sqrt_approx(gain_squared); // Q8
            gain = silk_min(gain, i32::MAX >> 8);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain, 8); // Q16
        } else {
            gain = silk_sqrt_approx(gain_squared); // Q0
            gain = silk_min(gain, i32::MAX >> 16);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain, 16); // Q16
        }
    }

    // Save unquantized gains and gain Index
    ps_enc_ctrl.gains_unq_q16[..nb_subfr].copy_from_slice(&ps_enc_ctrl.gains_q16[..nb_subfr]);
    ps_enc_ctrl.last_gain_index_prev = ps_shape_st.last_gain_index;

    // Quantize gains
    silk_gains_quant(
        &mut s_cmn.indices.gains_indices,
        &mut ps_enc_ctrl.gains_q16,
        &mut ps_shape_st.last_gain_index,
        (cond_coding == CODE_CONDITIONALLY) as i32,
        nb_subfr,
    );

    // Set quantizer offset for voiced signals. Larger offset when LTP coding gain is low or
    // tilt is high (ie low-pass)
    if s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        if ps_enc_ctrl.lt_pred_cod_gain_q7 + silk_rshift(s_cmn.input_tilt_q15, 8)
            > silk_fix_const(1.0, 7)
        {
            s_cmn.indices.quant_offset_type = 0;
        } else {
            s_cmn.indices.quant_offset_type = 1;
        }
    }

    // Quantizer boundary adjustment
    let quant_offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(s_cmn.indices.signal_type >> 1) as usize]
        [s_cmn.indices.quant_offset_type as usize] as i32;
    ps_enc_ctrl.lambda_q10 = fc(LAMBDA_OFFSET, 10)
        + silk_smulbb(
            fc(LAMBDA_DELAYED_DECISIONS, 10),
            s_cmn.n_states_delayed_decision,
        )
        + silk_smulwb(fc(LAMBDA_SPEECH_ACT, 18), s_cmn.speech_activity_q8)
        + silk_smulwb(fc(LAMBDA_INPUT_QUALITY, 12), ps_enc_ctrl.input_quality_q14)
        + silk_smulwb(
            fc(LAMBDA_CODING_QUALITY, 12),
            ps_enc_ctrl.coding_quality_q14,
        )
        + silk_smulwb(fc(LAMBDA_QUANT_OFFSET, 16), quant_offset_q10);

    debug_assert!(ps_enc_ctrl.lambda_q10 > 0);
    debug_assert!(ps_enc_ctrl.lambda_q10 < silk_fix_const(2.0, 10));
}

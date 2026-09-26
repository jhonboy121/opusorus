//! Port of silk/NSQ.c, silk/NSQ.h: noise shaping quantization (single state).
//!
//! The C functions take `const silk_encoder_state *psEncC` together with pointers into that same
//! state (`&psEnc->sCmn.sNSQ`, `&psEnc->sCmn.indices`, `psEnc->sCmn.pulses`). Rust cannot borrow
//! the encoder state immutably and its fields mutably at the same time, so the handful of
//! encoder-state fields the quantizers read are passed as a [`NsqEncParams`] value instead
//! (build it with [`NsqEncParams::from_enc`] before borrowing the fields mutably).

#![allow(
    clippy::too_many_arguments,
    reason = "function signatures mirror the C sources one-to-one"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::silk::define::{
    HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER,
    MAX_SUB_FRAME_LENGTH, NSQ_LPC_BUF_LENGTH, QUANT_LEVEL_ADJUST_Q10, TYPE_VOICED,
};
use crate::silk::macros::{
    silk_add_lshift32, silk_add_sat32, silk_add32, silk_add32_ovflw, silk_div32_varq,
    silk_inverse32_varq, silk_limit_32, silk_lshift, silk_lshift32, silk_max, silk_rand,
    silk_rshift, silk_rshift_round, silk_sat16, silk_smlabb, silk_smlawb, silk_smlawb_chain,
    silk_smlawt, silk_smulbb, silk_smulwb, silk_smulww, silk_sub32, silk_sub32_ovflw,
};
use crate::silk::sigproc::silk_lpc_analysis_filter;
use crate::silk::structs::{SideInfoIndices, SilkEncoderState, SilkNsqState};
use crate::silk::tables::SILK_QUANTIZATION_OFFSETS_Q10;

const MFL: usize = MAX_FRAME_LENGTH as usize;
const MSFL: usize = MAX_SUB_FRAME_LENGTH as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MSHP: usize = MAX_SHAPE_LPC_ORDER as usize;
const LTPO: usize = LTP_ORDER as usize;
const NLBL: usize = NSQ_LPC_BUF_LENGTH as usize;

/// The fields of `silk_encoder_state` read by `silk_NSQ_c` / `silk_NSQ_del_dec_c` (C passes the
/// whole `const silk_encoder_state *psEncC`; see the module docs for why this is a value).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NsqEncParams {
    /// `psEncC->ltp_mem_length`.
    pub ltp_mem_length: i32,
    /// `psEncC->frame_length`.
    pub frame_length: i32,
    /// `psEncC->subfr_length`.
    pub subfr_length: i32,
    /// `psEncC->nb_subfr`.
    pub nb_subfr: i32,
    /// `psEncC->predictLPCOrder`.
    pub predict_lpc_order: i32,
    /// `psEncC->shapingLPCOrder`.
    pub shaping_lpc_order: i32,
    /// `psEncC->nStatesDelayedDecision`.
    pub n_states_delayed_decision: i32,
    /// `psEncC->warping_Q16`.
    pub warping_q16: i32,
}

impl NsqEncParams {
    /// Copies the NSQ-relevant fields out of an encoder state.
    #[must_use]
    pub const fn from_enc(ps_enc_c: &SilkEncoderState) -> Self {
        Self {
            ltp_mem_length: ps_enc_c.ltp_mem_length,
            frame_length: ps_enc_c.frame_length,
            subfr_length: ps_enc_c.subfr_length,
            nb_subfr: ps_enc_c.nb_subfr,
            predict_lpc_order: ps_enc_c.predict_lpc_order,
            shaping_lpc_order: ps_enc_c.shaping_lpc_order,
            n_states_delayed_decision: ps_enc_c.n_states_delayed_decision,
            warping_q16: ps_enc_c.warping_q16,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// NSQ.h
// ---------------------------------------------------------------------------------------------

/// Port of silk/NSQ.h:silk_noise_shape_quantizer_short_prediction_c.
///
/// `buf32` ends at the C pointer `buf32` (i.e. C `buf32[ -j ]` is `buf32[ buf32.len() - 1 - j ]`);
/// it must hold at least `order` elements.
#[inline(always)]
#[must_use]
pub fn silk_noise_shape_quantizer_short_prediction_c(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    debug_assert!(order == 10 || order == 16);
    let n = buf32.len();

    // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf.
    // Perf: `silk_smlawb_chain` is the C chain `out = silk_SMLAWB( out, buf32[ -j ],
    // coef16[ j ] )` for j = 0..order (bit-identical, see there); the window is
    // `buf32[ -order + 1 ..= 0 ]`.
    if order == 16 {
        silk_smlawb_chain(8, &buf32[n - 16..], coef16)
    } else {
        silk_smlawb_chain(5, &buf32[n - 10..], coef16)
    }
}

/// Port of silk/NSQ.h:silk_NSQ_noise_shape_feedback_loop_c.
///
/// `data0` is C `data0[ 0 ]` (the only element read).
#[inline]
pub fn silk_nsq_noise_shape_feedback_loop_c(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
) -> i32 {
    let order = order as usize;
    let (data1, coef) = (&mut data1[..order], &coef[..order]);

    let mut tmp2 = data0;
    let mut tmp1 = data1[0];
    data1[0] = tmp2;

    let mut out = silk_rshift(order as i32, 1);
    out = silk_smlawb(out, tmp2, coef[0] as i32);

    let mut j = 2;
    while j < order {
        tmp2 = data1[j - 1];
        data1[j - 1] = tmp1;
        out = silk_smlawb(out, tmp1, coef[j - 1] as i32);
        tmp1 = data1[j];
        data1[j] = tmp2;
        out = silk_smlawb(out, tmp2, coef[j] as i32);
        j += 2;
    }
    data1[order - 1] = tmp1;
    out = silk_smlawb(out, tmp1, coef[order - 1] as i32);
    // Q11 -> Q12
    silk_lshift32(out, 1)
}

// ---------------------------------------------------------------------------------------------
// NSQ.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/NSQ.c:silk_NSQ_c.
///
/// * `pred_coef_q12`: `[ 2 * MAX_LPC_ORDER ]` (C `PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ]` flattened)
/// * `ltp_coef_q14`: `[ LTP_ORDER * MAX_NB_SUBFR ]`
/// * `ar_q13`: `[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ]`
/// * `harm_shape_gain_q14`, `tilt_q14`, `lf_shp_q14`, `gains_q16`, `pitch_l`: `[ MAX_NB_SUBFR ]`
/// * `x16`, `pulses`: `[ frame_length ]`
///
/// `ps_indices` is only read here (`Seed`, `signalType`, `quantOffsetType`, `NLSFInterpCoef_Q2`).
pub fn silk_nsq_c(
    ps_enc_c: &NsqEncParams,
    nsq: &mut SilkNsqState,
    ps_indices: &SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    let ltp_mem_length = ps_enc_c.ltp_mem_length as usize;
    let frame_length = ps_enc_c.frame_length as usize;
    let subfr_length = ps_enc_c.subfr_length as usize;
    let signal_type = ps_indices.signal_type as i32;

    nsq.rand_seed = ps_indices.seed as i32;

    // Set unvoiced lag to the previous one, overwrite later for voiced
    let mut lag = nsq.lag_prev;

    debug_assert!(nsq.prev_gain_q16 != 0);

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;

    let lsf_interpolation_flag: i32 = if ps_indices.nlsf_interp_coef_q2 == 4 {
        0
    } else {
        1
    };

    // C: ALLOC( sLTP_Q15, ltp_mem_length + frame_length ), ALLOC( sLTP, ... ),
    // ALLOC( x_sc_Q10, subfr_length ).
    let mut s_ltp_q15 = [0i32; 2 * MFL];
    let mut s_ltp = [0i16; 2 * MFL];
    let mut x_sc_q10 = [0i32; MSFL];
    let s_ltp_q15 = &mut s_ltp_q15[..ltp_mem_length + frame_length];
    let s_ltp = &mut s_ltp[..ltp_mem_length + frame_length];
    let x_sc_q10 = &mut x_sc_q10[..subfr_length];

    // Set up pointers to start of sub frame
    nsq.s_ltp_shp_buf_idx = ltp_mem_length as i32;
    nsq.s_ltp_buf_idx = ltp_mem_length as i32;
    let mut pxq = ltp_mem_length; // index into nsq.xq
    for k in 0..ps_enc_c.nb_subfr as usize {
        let a_off = (((k as i32 >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER) as usize;
        let a_q12 = &pred_coef_q12[a_off..a_off + MLPC];
        let b_q14 = &ltp_coef_q14[k * LTPO..(k + 1) * LTPO];
        let ar_shp_q13 = &ar_q13[k * MSHP..(k + 1) * MSHP];

        // Noise shape parameters
        debug_assert!(harm_shape_gain_q14[k] >= 0);
        let mut harm_shape_fir_packed_q14 = silk_rshift(harm_shape_gain_q14[k], 2);
        harm_shape_fir_packed_q14 |= silk_lshift(silk_rshift(harm_shape_gain_q14[k], 1), 16);

        nsq.rewhite_flag = 0;
        if signal_type == TYPE_VOICED {
            // Voiced
            lag = pitch_l[k];

            // Re-whitening
            if (k as i32 & (3 - silk_lshift(lsf_interpolation_flag, 1))) == 0 {
                // Rewhiten with new A coefs
                let start_idx =
                    ps_enc_c.ltp_mem_length - lag - ps_enc_c.predict_lpc_order - LTP_ORDER / 2;
                debug_assert!(start_idx > 0);
                let start_idx = start_idx as usize;

                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &nsq.xq[start_idx + k * subfr_length..],
                    a_q12,
                    ltp_mem_length - start_idx,
                    ps_enc_c.predict_lpc_order as usize,
                );

                nsq.rewhite_flag = 1;
                nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length;
            }
        }

        silk_nsq_scale_states(
            ps_enc_c,
            nsq,
            &x16[k * subfr_length..],
            x_sc_q10,
            s_ltp,
            s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
        );

        silk_noise_shape_quantizer(
            nsq,
            signal_type,
            x_sc_q10,
            &mut pulses[k * subfr_length..],
            pxq,
            s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_length,
            ps_enc_c.shaping_lpc_order,
            ps_enc_c.predict_lpc_order,
        );

        pxq += subfr_length;
    }

    // Update lagPrev for next frame
    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr as usize - 1];

    // Save quantized speech and noise shaping signals
    nsq.xq
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
}

/// Port of silk/NSQ.c:silk_noise_shape_quantizer (static).
///
/// `xq_off` is the index in `nsq.xq` of the C output pointer `xq`.
fn silk_noise_shape_quantizer(
    nsq: &mut SilkNsqState,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: usize,
    shaping_lpc_order: i32,
    predict_lpc_order: i32,
) {
    // C pointers `shp_lag_ptr`, `pred_lag_ptr` as indices (only dereferenced when valid).
    let mut shp_lag_ptr = nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2;
    let mut pred_lag_ptr = nsq.s_ltp_buf_idx - lag + LTP_ORDER / 2;
    let gain_q10 = silk_rshift(gain_q16, 6);

    // Set up short term AR state: psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 ]
    let mut ps_lpc_q14 = NLBL - 1;

    for i in 0..length {
        // Generate dither
        nsq.rand_seed = silk_rand(nsq.rand_seed);

        // Short-term prediction
        let lpc_pred_q10 = silk_noise_shape_quantizer_short_prediction_c(
            &nsq.s_lpc_q14[..=ps_lpc_q14],
            a_q12,
            predict_lpc_order,
        );

        // Long-term prediction
        let ltp_pred_q13 = if signal_type == TYPE_VOICED {
            // Unrolled loop
            // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
            let p = pred_lag_ptr as usize;
            let mut v = 2;
            v = silk_smlawb(v, s_ltp_q15[p], b_q14[0] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 1], b_q14[1] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 2], b_q14[2] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 3], b_q14[3] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 4], b_q14[4] as i32);
            pred_lag_ptr += 1;
            v
        } else {
            0
        };

        // Noise shape feedback
        debug_assert!((shaping_lpc_order & 1) == 0); // check that order is even
        let mut n_ar_q12 = silk_nsq_noise_shape_feedback_loop_c(
            nsq.s_diff_shp_q14,
            &mut nsq.s_ar2_q14,
            ar_shp_q13,
            shaping_lpc_order,
        );

        n_ar_q12 = silk_smlawb(n_ar_q12, nsq.s_lf_ar_shp_q14, tilt_q14);

        let mut n_lf_q12 = silk_smulwb(
            nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize - 1],
            lf_shp_q14,
        );
        n_lf_q12 = silk_smlawt(n_lf_q12, nsq.s_lf_ar_shp_q14, lf_shp_q14);

        debug_assert!(lag > 0 || signal_type != TYPE_VOICED);

        // Combine prediction and noise shaping signals
        let mut tmp1 = silk_sub32_ovflw(silk_lshift32(lpc_pred_q10, 2), n_ar_q12); // Q12
        tmp1 = silk_sub32_ovflw(tmp1, n_lf_q12); // Q12
        if lag > 0 {
            // Symmetric, packed FIR coefficients
            let s = shp_lag_ptr as usize;
            let mut n_ltp_q13 = silk_smulwb(
                silk_add_sat32(nsq.s_ltp_shp_q14[s], nsq.s_ltp_shp_q14[s - 2]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = silk_smlawt(
                n_ltp_q13,
                nsq.s_ltp_shp_q14[s - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = silk_lshift(n_ltp_q13, 1);
            shp_lag_ptr += 1;

            let tmp2 = silk_sub32(ltp_pred_q13, n_ltp_q13); // Q13
            tmp1 = silk_add32_ovflw(tmp2, silk_lshift32(tmp1, 1)); // Q13
            tmp1 = silk_rshift_round(tmp1, 3); // Q10
        } else {
            tmp1 = silk_rshift_round(tmp1, 2); // Q10
        }

        let mut r_q10 = silk_sub32(x_sc_q10[i], tmp1); // residual error Q10

        // Flip sign depending on dither
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = silk_limit_32(r_q10, -(31 << 10), 30 << 10);

        // Find two quantization level candidates and measure their rate-distortion
        let mut q1_q10 = silk_sub32(r_q10, offset_q10);
        let mut q1_q0 = silk_rshift(q1_q10, 10);
        if lambda_q10 > 2048 {
            // For aggressive RDO, the bias becomes more than one pulse.
            let rdo_offset = lambda_q10 / 2 - 512;
            if q1_q10 > rdo_offset {
                q1_q0 = silk_rshift(q1_q10 - rdo_offset, 10);
            } else if q1_q10 < -rdo_offset {
                q1_q0 = silk_rshift(q1_q10 + rdo_offset, 10);
            } else if q1_q10 < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }
        let q2_q10;
        let mut rd1_q20;
        let mut rd2_q20;
        if q1_q0 > 0 {
            q1_q10 = silk_sub32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10);
            q1_q10 = silk_add32(q1_q10, offset_q10);
            q2_q10 = silk_add32(q1_q10, 1024);
            rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == 0 {
            q1_q10 = offset_q10;
            q2_q10 = silk_add32(q1_q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
            rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == -1 {
            q2_q10 = offset_q10;
            q1_q10 = silk_sub32(q2_q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
            rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else {
            // Q1_Q0 < -1
            q1_q10 = silk_add32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10);
            q1_q10 = silk_add32(q1_q10, offset_q10);
            q2_q10 = silk_add32(q1_q10, 1024);
            rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(-q2_q10, lambda_q10);
        }
        let mut rr_q10 = silk_sub32(r_q10, q1_q10);
        rd1_q20 = silk_smlabb(rd1_q20, rr_q10, rr_q10);
        rr_q10 = silk_sub32(r_q10, q2_q10);
        rd2_q20 = silk_smlabb(rd2_q20, rr_q10, rr_q10);

        if rd2_q20 < rd1_q20 {
            q1_q10 = q2_q10;
        }

        pulses[i] = silk_rshift_round(q1_q10, 10) as i8;

        // Excitation
        let mut exc_q14 = silk_lshift(q1_q10, 4);
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        // Add predictions
        let lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
        let xq_q14 = silk_add32_ovflw(lpc_exc_q14, silk_lshift32(lpc_pred_q10, 4));

        // Scale XQ back to normal level before saving
        nsq.xq[xq_off + i] = silk_sat16(silk_rshift_round(silk_smulww(xq_q14, gain_q10), 8)) as i16;

        // Update states
        ps_lpc_q14 += 1;
        nsq.s_lpc_q14[ps_lpc_q14] = xq_q14;
        nsq.s_diff_shp_q14 = silk_sub32_ovflw(xq_q14, silk_lshift32(x_sc_q10[i], 4));
        let s_lf_ar_shp_q14 = silk_sub32_ovflw(nsq.s_diff_shp_q14, silk_lshift32(n_ar_q12, 2));
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        let shp_idx = nsq.s_ltp_shp_buf_idx as usize;
        nsq.s_ltp_shp_q14[shp_idx] = silk_sub32_ovflw(s_lf_ar_shp_q14, silk_lshift32(n_lf_q12, 2));
        s_ltp_q15[nsq.s_ltp_buf_idx as usize] = silk_lshift(lpc_exc_q14, 1);
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Make dither dependent on quantized signal
        nsq.rand_seed = silk_add32_ovflw(nsq.rand_seed, pulses[i] as i32);
    }

    // Update LPC synth buffer
    nsq.s_lpc_q14.copy_within(length..length + NLBL, 0);
}

/// Port of silk/NSQ.c:silk_nsq_scale_states (static).
fn silk_nsq_scale_states(
    ps_enc_c: &NsqEncParams,
    nsq: &mut SilkNsqState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
) {
    let lag = pitch_l[subfr];
    let mut inv_gain_q31 = silk_inverse32_varq(silk_max(gains_q16[subfr], 1), 47);
    debug_assert!(inv_gain_q31 != 0);

    // Scale input
    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..ps_enc_c.subfr_length as usize {
        x_sc_q10[i] = silk_smulww(x16[i] as i32, inv_gain_q26);
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16
    if nsq.rewhite_flag != 0 {
        if subfr == 0 {
            // Do LTP downscaling
            inv_gain_q31 = silk_lshift(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2);
        }
        for i in (nsq.s_ltp_buf_idx - lag - LTP_ORDER / 2) as usize..nsq.s_ltp_buf_idx as usize {
            debug_assert!(i < MFL);
            s_ltp_q15[i] = silk_smulwb(inv_gain_q31, s_ltp[i] as i32);
        }
    }

    // Adjust for changing gain
    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        // Scale long-term shaping state
        for i in (nsq.s_ltp_shp_buf_idx - ps_enc_c.ltp_mem_length) as usize
            ..nsq.s_ltp_shp_buf_idx as usize
        {
            nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            for i in (nsq.s_ltp_buf_idx - lag - LTP_ORDER / 2) as usize..nsq.s_ltp_buf_idx as usize
            {
                s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }

        nsq.s_lf_ar_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_lf_ar_shp_q14);
        nsq.s_diff_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_diff_shp_q14);

        // Scale short-term prediction and shaping states
        for i in 0..NLBL {
            nsq.s_lpc_q14[i] = silk_smulww(gain_adj_q16, nsq.s_lpc_q14[i]);
        }
        for i in 0..MSHP {
            nsq.s_ar2_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ar2_q14[i]);
        }

        // Save inverse gain
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

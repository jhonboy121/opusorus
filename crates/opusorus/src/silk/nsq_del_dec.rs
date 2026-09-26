//! Port of silk/NSQ_del_dec.c: noise shaping quantization with delayed decision.
//!
//! See [`crate::silk::nsq`] for why the encoder-state parameters come in as [`NsqEncParams`].

#![allow(
    clippy::too_many_arguments,
    reason = "function signatures mirror the C sources one-to-one"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_DEL_DEC_STATES, MAX_FRAME_LENGTH,
    MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER, MAX_SUB_FRAME_LENGTH, NSQ_LPC_BUF_LENGTH,
    QUANT_LEVEL_ADJUST_Q10, TYPE_VOICED,
};
use crate::silk::macros::{
    SILK_INT32_MAX, silk_add_sat32, silk_add32, silk_add32_ovflw, silk_div32_varq,
    silk_inverse32_varq, silk_limit_32, silk_lshift, silk_lshift32, silk_max, silk_min_int,
    silk_rand, silk_rshift, silk_rshift_round, silk_rshift32, silk_sat16, silk_smlabb, silk_smlawb,
    silk_smlawt, silk_smulbb, silk_smulwb, silk_smulww, silk_sub_lshift32, silk_sub_sat32,
    silk_sub32, silk_sub32_ovflw,
};
use crate::silk::nsq::{NsqEncParams, silk_noise_shape_quantizer_short_prediction_c};
use crate::silk::sigproc::silk_lpc_analysis_filter;
use crate::silk::structs::{SideInfoIndices, SilkNsqState};
use crate::silk::tables::SILK_QUANTIZATION_OFFSETS_Q10;

const MFL: usize = MAX_FRAME_LENGTH as usize;
const MSFL: usize = MAX_SUB_FRAME_LENGTH as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MSHP: usize = MAX_SHAPE_LPC_ORDER as usize;
const LTPO: usize = LTP_ORDER as usize;
const NLBL: usize = NSQ_LPC_BUF_LENGTH as usize;
const DD: usize = DECISION_DELAY as usize;
const MDDS: usize = MAX_DEL_DEC_STATES as usize;

/// `NSQ_del_dec_struct`.
#[derive(Debug, Clone, Copy)]
struct NsqDelDecStruct {
    s_lpc_q14: [i32; MSFL + NLBL],
    rand_state: [i32; DD],
    q_q10: [i32; DD],
    xq_q14: [i32; DD],
    pred_q15: [i32; DD],
    shape_q14: [i32; DD],
    s_ar2_q14: [i32; MSHP],
    lf_ar_q14: i32,
    diff_q14: i32,
    seed: i32,
    seed_init: i32,
    rd_q10: i32,
}

impl NsqDelDecStruct {
    /// All-zero state (C `silk_memset( psDelDec, 0, ... )`).
    const ZERO: Self = Self {
        s_lpc_q14: [0; MSFL + NLBL],
        rand_state: [0; DD],
        q_q10: [0; DD],
        xq_q14: [0; DD],
        pred_q15: [0; DD],
        shape_q14: [0; DD],
        s_ar2_q14: [0; MSHP],
        lf_ar_q14: 0,
        diff_q14: 0,
        seed: 0,
        seed_init: 0,
        rd_q10: 0,
    };

    /// C `silk_memcpy( ( (opus_int32 *)dst ) + i, ( (opus_int32 *)src ) + i,
    /// sizeof( NSQ_del_dec_struct ) - i * sizeof( opus_int32 ) )`: copies every field of `src`
    /// except the first `i` words, which (as `i < MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH`)
    /// all lie in `sLPC_Q14`.
    ///
    /// Perf: copies the fields directly (one pass over the data, like the C `memcpy`) instead
    /// of copying the whole struct and restoring the kept prefix.
    fn copy_from_word(&mut self, src: &Self, i: usize) {
        debug_assert!(i < MSFL + NLBL);
        self.s_lpc_q14[i..].copy_from_slice(&src.s_lpc_q14[i..]);
        self.rand_state = src.rand_state;
        self.q_q10 = src.q_q10;
        self.xq_q14 = src.xq_q14;
        self.pred_q15 = src.pred_q15;
        self.shape_q14 = src.shape_q14;
        self.s_ar2_q14 = src.s_ar2_q14;
        self.lf_ar_q14 = src.lf_ar_q14;
        self.diff_q14 = src.diff_q14;
        self.seed = src.seed;
        self.seed_init = src.seed_init;
        self.rd_q10 = src.rd_q10;
    }
}

/// `NSQ_sample_struct`.
#[derive(Debug, Clone, Copy, Default)]
struct NsqSampleStruct {
    q_q10: i32,
    rd_q10: i32,
    xq_q14: i32,
    lf_ar_q14: i32,
    diff_q14: i32,
    s_ltp_shp_q14: i32,
    lpc_exc_q14: i32,
}

/// `NSQ_sample_pair`.
type NsqSamplePair = [NsqSampleStruct; 2];

/// Port of silk/NSQ_del_dec.c:silk_NSQ_del_dec_c.
///
/// Arguments as for [`crate::silk::nsq::silk_nsq_c`]; `ps_indices.seed` is updated with the
/// winning seed.
pub fn silk_nsq_del_dec_c(
    ps_enc_c: &NsqEncParams,
    nsq: &mut SilkNsqState,
    ps_indices: &mut SideInfoIndices,
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
    let n_states = ps_enc_c.n_states_delayed_decision as usize;
    let signal_type = ps_indices.signal_type as i32;

    // Set unvoiced lag to the previous one, overwrite later for voiced
    let mut lag = nsq.lag_prev;

    silk_assert!(nsq.prev_gain_q16 != 0);

    // Initialize delayed decision states
    debug_assert!(n_states > 0 && n_states <= MDDS);
    let mut ps_del_dec_buf = [NsqDelDecStruct::ZERO; MDDS];
    let ps_del_dec = &mut ps_del_dec_buf[..n_states];
    for (k, ps_dd) in ps_del_dec.iter_mut().enumerate() {
        ps_dd.seed = (k as i32 + ps_indices.seed as i32) & 3;
        ps_dd.seed_init = ps_dd.seed;
        ps_dd.rd_q10 = 0;
        ps_dd.lf_ar_q14 = nsq.s_lf_ar_shp_q14;
        ps_dd.diff_q14 = nsq.s_diff_shp_q14;
        ps_dd.shape_q14[0] = nsq.s_ltp_shp_q14[ltp_mem_length - 1];
        ps_dd.s_lpc_q14[..NLBL].copy_from_slice(&nsq.s_lpc_q14[..NLBL]);
        ps_dd.s_ar2_q14 = nsq.s_ar2_q14;
    }

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;
    let mut smpl_buf_idx: i32 = 0; // index of oldest samples

    let mut decision_delay = silk_min_int(DECISION_DELAY, ps_enc_c.subfr_length);

    // For voiced frames limit the decision delay to lower than the pitch lag
    if signal_type == TYPE_VOICED {
        for k in 0..ps_enc_c.nb_subfr as usize {
            decision_delay = silk_min_int(decision_delay, pitch_l[k] - LTP_ORDER / 2 - 1);
        }
    } else if lag > 0 {
        decision_delay = silk_min_int(decision_delay, lag - LTP_ORDER / 2 - 1);
    }
    let dd = decision_delay as usize;

    let lsf_interpolation_flag: i32 = if ps_indices.nlsf_interp_coef_q2 == 4 {
        0
    } else {
        1
    };

    // C: ALLOC( sLTP_Q15 ), ALLOC( sLTP ), ALLOC( x_sc_Q10 ), ALLOC( delayedGain_Q10 ).
    let mut s_ltp_q15 = [0i32; 2 * MFL];
    let mut s_ltp = [0i16; 2 * MFL];
    let mut x_sc_q10 = [0i32; MSFL];
    let mut delayed_gain_q10 = [0i32; DD];
    let s_ltp_q15 = &mut s_ltp_q15[..ltp_mem_length + frame_length];
    let s_ltp = &mut s_ltp[..ltp_mem_length + frame_length];
    let x_sc_q10 = &mut x_sc_q10[..subfr_length];

    // Set up pointers to start of sub frame
    let mut pxq = ltp_mem_length; // index into nsq.xq
    nsq.s_ltp_shp_buf_idx = ltp_mem_length as i32;
    nsq.s_ltp_buf_idx = ltp_mem_length as i32;
    let mut subfr: i32 = 0;
    for k in 0..ps_enc_c.nb_subfr as usize {
        let a_off = (((k as i32 >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER) as usize;
        let a_q12 = &pred_coef_q12[a_off..a_off + MLPC];
        let b_q14 = &ltp_coef_q14[k * LTPO..(k + 1) * LTPO];
        let ar_shp_q13 = &ar_q13[k * MSHP..(k + 1) * MSHP];
        let pulses_off = k * subfr_length; // C `pulses` pointer

        // Noise shape parameters
        silk_assert!(harm_shape_gain_q14[k] >= 0);
        let mut harm_shape_fir_packed_q14 = silk_rshift(harm_shape_gain_q14[k], 2);
        harm_shape_fir_packed_q14 |= silk_lshift(silk_rshift(harm_shape_gain_q14[k], 1), 16);

        nsq.rewhite_flag = 0;
        if signal_type == TYPE_VOICED {
            // Voiced
            lag = pitch_l[k];

            // Re-whitening
            if (k as i32 & (3 - silk_lshift(lsf_interpolation_flag, 1))) == 0 {
                if k == 2 {
                    // RESET DELAYED DECISIONS
                    // Find winner
                    let mut rdmin_q10 = ps_del_dec[0].rd_q10;
                    let mut winner_ind = 0;
                    for i in 1..n_states {
                        if ps_del_dec[i].rd_q10 < rdmin_q10 {
                            rdmin_q10 = ps_del_dec[i].rd_q10;
                            winner_ind = i;
                        }
                    }
                    for i in 0..n_states {
                        if i != winner_ind {
                            ps_del_dec[i].rd_q10 += SILK_INT32_MAX >> 4;
                            silk_assert!(ps_del_dec[i].rd_q10 >= 0);
                        }
                    }

                    // Copy final part of signals from winner state to output and long-term
                    // filter states
                    let ps_dd = &ps_del_dec[winner_ind];
                    let mut last_smple_idx = smpl_buf_idx + decision_delay;
                    for i in 0..dd {
                        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
                        if last_smple_idx < 0 {
                            last_smple_idx += DECISION_DELAY;
                        }
                        let ls = last_smple_idx as usize;
                        pulses[pulses_off + i - dd] = silk_rshift_round(ps_dd.q_q10[ls], 10) as i8;
                        nsq.xq[pxq + i - dd] = silk_sat16(silk_rshift_round(
                            silk_smulww(ps_dd.xq_q14[ls], gains_q16[1]),
                            14,
                        )) as i16;
                        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize - dd + i] =
                            ps_dd.shape_q14[ls];
                    }

                    subfr = 0;
                }

                // Rewhiten with new A coefs
                let start_idx =
                    ps_enc_c.ltp_mem_length - lag - ps_enc_c.predict_lpc_order - LTP_ORDER / 2;
                celt_assert!(start_idx > 0);
                let start_idx = start_idx as usize;

                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &nsq.xq[start_idx + k * subfr_length..],
                    a_q12,
                    ltp_mem_length - start_idx,
                    ps_enc_c.predict_lpc_order as usize,
                );

                nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length;
                nsq.rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states(
            ps_enc_c,
            nsq,
            ps_del_dec,
            &x16[k * subfr_length..],
            x_sc_q10,
            s_ltp,
            s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
            decision_delay,
        );

        silk_noise_shape_quantizer_del_dec(
            nsq,
            ps_del_dec,
            signal_type,
            x_sc_q10,
            pulses,
            pulses_off,
            pxq,
            s_ltp_q15,
            &mut delayed_gain_q10,
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
            subfr,
            ps_enc_c.shaping_lpc_order,
            ps_enc_c.predict_lpc_order,
            ps_enc_c.warping_q16,
            &mut smpl_buf_idx,
            decision_delay,
        );
        subfr += 1;

        pxq += subfr_length;
    }

    // Find winner
    let mut rdmin_q10 = ps_del_dec[0].rd_q10;
    let mut winner_ind = 0;
    for k in 1..n_states {
        if ps_del_dec[k].rd_q10 < rdmin_q10 {
            rdmin_q10 = ps_del_dec[k].rd_q10;
            winner_ind = k;
        }
    }

    // Copy final part of signals from winner state to output and long-term filter states
    let ps_dd = &ps_del_dec[winner_ind];
    ps_indices.seed = ps_dd.seed_init as i8;
    let mut last_smple_idx = smpl_buf_idx + decision_delay;
    let gain_q10 = silk_rshift32(gains_q16[ps_enc_c.nb_subfr as usize - 1], 6);
    let pulses_off = frame_length; // C `pulses` pointer after the last subframe
    for i in 0..dd {
        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
        if last_smple_idx < 0 {
            last_smple_idx += DECISION_DELAY;
        }
        let ls = last_smple_idx as usize;

        pulses[pulses_off + i - dd] = silk_rshift_round(ps_dd.q_q10[ls], 10) as i8;
        nsq.xq[pxq + i - dd] = silk_sat16(silk_rshift_round(
            silk_smulww(ps_dd.xq_q14[ls], gain_q10),
            8,
        )) as i16;
        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize - dd + i] = ps_dd.shape_q14[ls];
    }
    nsq.s_lpc_q14[..NLBL].copy_from_slice(&ps_dd.s_lpc_q14[subfr_length..subfr_length + NLBL]);
    nsq.s_ar2_q14 = ps_dd.s_ar2_q14;

    // Update states
    nsq.s_lf_ar_shp_q14 = ps_dd.lf_ar_q14;
    nsq.s_diff_shp_q14 = ps_dd.diff_q14;
    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr as usize - 1];

    // Save quantized speech signal
    nsq.xq
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
}

/// Port of silk/NSQ_del_dec.c:silk_noise_shape_quantizer_del_dec (static): noise shape
/// quantizer for one subframe.
///
/// `pulses_off` / `xq_off` are the indices in `pulses` / `nsq.xq` of the C pointers `pulses` /
/// `xq` (C writes `pulses[ i - decisionDelay ]`, i.e. before the pointer).
fn silk_noise_shape_quantizer_del_dec(
    nsq: &mut SilkNsqState,
    ps_del_dec: &mut [NsqDelDecStruct],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32; DD],
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
    subfr: i32,
    shaping_lpc_order: i32,
    predict_lpc_order: i32,
    warping_q16: i32,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
) {
    let n_states = ps_del_dec.len();
    celt_assert!(n_states > 0);
    let dd = decision_delay as usize;
    let shp_order = shaping_lpc_order as usize;
    let mut ps_sample_state_buf = [NsqSamplePair::default(); MDDS];
    let ps_sample_state = &mut ps_sample_state_buf[..n_states];

    // C pointers `shp_lag_ptr`, `pred_lag_ptr` as indices (only dereferenced when valid).
    let mut shp_lag_ptr = nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2;
    let mut pred_lag_ptr = nsq.s_ltp_buf_idx - lag + LTP_ORDER / 2;
    let gain_q10 = silk_rshift(gain_q16, 6);

    for i in 0..length {
        // Perform common calculations used in all states

        // Long-term prediction
        let ltp_pred_q14 = if signal_type == TYPE_VOICED {
            // Unrolled loop
            // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
            let p = pred_lag_ptr as usize;
            let mut v = 2;
            v = silk_smlawb(v, s_ltp_q15[p], b_q14[0] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 1], b_q14[1] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 2], b_q14[2] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 3], b_q14[3] as i32);
            v = silk_smlawb(v, s_ltp_q15[p - 4], b_q14[4] as i32);
            v = silk_lshift(v, 1); // Q13 -> Q14
            pred_lag_ptr += 1;
            v
        } else {
            0
        };

        // Long-term shaping
        let n_ltp_q14 = if lag > 0 {
            // Symmetric, packed FIR coefficients
            let s = shp_lag_ptr as usize;
            let mut v = silk_smulwb(
                silk_add_sat32(nsq.s_ltp_shp_q14[s], nsq.s_ltp_shp_q14[s - 2]),
                harm_shape_fir_packed_q14,
            );
            v = silk_smlawt(v, nsq.s_ltp_shp_q14[s - 1], harm_shape_fir_packed_q14);
            v = silk_sub_lshift32(ltp_pred_q14, v, 2); // Q12 -> Q14
            shp_lag_ptr += 1;
            v
        } else {
            0
        };

        // Perf: the delayed-decision states are independent. The short-term prediction and
        // the noise shape feedback (a long serial chain of allpass sections per state) are
        // first computed for all states, with the allpass sections of the states interleaved
        // so that their dependency chains overlap instead of running back to back; the rest
        // of the per-state work follows in a second loop. For each state the operations and
        // their order are exactly those of C.
        let mut lpc_pred = [0i32; MDDS];
        let mut t1 = [0i32; MDDS];
        let mut n_ar = [0i32; MDDS];
        celt_assert!((shaping_lpc_order & 1) == 0); // check that order is even
        for (((ps_dd, lpc_pred_q14), tmp1), n_ar_q14) in ps_del_dec
            .iter_mut()
            .zip(&mut lpc_pred)
            .zip(&mut t1)
            .zip(&mut n_ar)
        {
            // Generate dither
            ps_dd.seed = silk_rand(ps_dd.seed);

            // Pointer used in short term prediction and shaping:
            // psLPC_Q14 = &psDD->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 + i ]
            let ps_lpc_q14 = NLBL - 1 + i;
            // Short-term prediction
            *lpc_pred_q14 = silk_noise_shape_quantizer_short_prediction_c(
                &ps_dd.s_lpc_q14[..=ps_lpc_q14],
                a_q12,
                predict_lpc_order,
            );
            *lpc_pred_q14 = silk_lshift(*lpc_pred_q14, 4); // Q10 -> Q14

            // Noise shape feedback
            // Output of lowpass section
            let tmp2 = silk_smlawb(ps_dd.diff_q14, ps_dd.s_ar2_q14[0], warping_q16);
            // Output of allpass section
            *tmp1 = silk_smlawb(
                ps_dd.s_ar2_q14[0],
                silk_sub32_ovflw(ps_dd.s_ar2_q14[1], tmp2),
                warping_q16,
            );
            ps_dd.s_ar2_q14[0] = tmp2;
            *n_ar_q14 = silk_rshift(shaping_lpc_order, 1);
            *n_ar_q14 = silk_smlawb(*n_ar_q14, tmp2, ar_shp_q13[0] as i32);
        }
        // Loop over allpass sections
        let mut j = 2;
        while j < shp_order {
            for ((ps_dd, tmp1), n_ar_q14) in ps_del_dec.iter_mut().zip(&mut t1).zip(&mut n_ar) {
                let s_ar2_q14 = &mut ps_dd.s_ar2_q14;
                // Output of allpass section
                let tmp2 = silk_smlawb(
                    s_ar2_q14[j - 1],
                    silk_sub32_ovflw(s_ar2_q14[j], *tmp1),
                    warping_q16,
                );
                s_ar2_q14[j - 1] = *tmp1;
                *n_ar_q14 = silk_smlawb(*n_ar_q14, *tmp1, ar_shp_q13[j - 1] as i32);
                // Output of allpass section
                *tmp1 = silk_smlawb(
                    s_ar2_q14[j],
                    silk_sub32_ovflw(s_ar2_q14[j + 1], tmp2),
                    warping_q16,
                );
                s_ar2_q14[j] = tmp2;
                *n_ar_q14 = silk_smlawb(*n_ar_q14, tmp2, ar_shp_q13[j] as i32);
            }
            j += 2;
        }

        for k in 0..n_states {
            // Delayed decision state
            let ps_dd = &mut ps_del_dec[k];

            // Sample state
            let ps_ss = &mut ps_sample_state[k];

            let lpc_pred_q14 = lpc_pred[k];
            let mut tmp1 = t1[k];
            let mut n_ar_q14 = n_ar[k];
            ps_dd.s_ar2_q14[shp_order - 1] = tmp1;
            n_ar_q14 = silk_smlawb(n_ar_q14, tmp1, ar_shp_q13[shp_order - 1] as i32);

            n_ar_q14 = silk_lshift(n_ar_q14, 1); // Q11 -> Q12
            n_ar_q14 = silk_smlawb(n_ar_q14, ps_dd.lf_ar_q14, tilt_q14); // Q12
            n_ar_q14 = silk_lshift(n_ar_q14, 2); // Q12 -> Q14

            let mut n_lf_q14 = silk_smulwb(ps_dd.shape_q14[*smpl_buf_idx as usize], lf_shp_q14); // Q12
            n_lf_q14 = silk_smlawt(n_lf_q14, ps_dd.lf_ar_q14, lf_shp_q14); // Q12
            n_lf_q14 = silk_lshift(n_lf_q14, 2); // Q12 -> Q14

            // Input minus prediction plus noise feedback
            // r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP
            tmp1 = silk_add_sat32(n_ar_q14, n_lf_q14); // Q14
            let tmp2 = silk_add32_ovflw(n_ltp_q14, lpc_pred_q14); // Q13
            tmp1 = silk_sub_sat32(tmp2, tmp1); // Q13
            tmp1 = silk_rshift_round(tmp1, 4); // Q10

            let mut r_q10 = silk_sub32(x_q10[i], tmp1); // residual error Q10

            // Flip sign depending on dither
            if ps_dd.seed < 0 {
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
            let mut rd1_q10;
            let mut rd2_q10;
            if q1_q0 > 0 {
                q1_q10 = silk_sub32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10);
                q1_q10 = silk_add32(q1_q10, offset_q10);
                q2_q10 = silk_add32(q1_q10, 1024);
                rd1_q10 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = silk_add32(q1_q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q10 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = silk_sub32(q2_q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q10 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else {
                // q1_Q0 < -1
                q1_q10 = silk_add32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10);
                q1_q10 = silk_add32(q1_q10, offset_q10);
                q2_q10 = silk_add32(q1_q10, 1024);
                rd1_q10 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(-q2_q10, lambda_q10);
            }
            let mut rr_q10 = silk_sub32(r_q10, q1_q10);
            rd1_q10 = silk_rshift(silk_smlabb(rd1_q10, rr_q10, rr_q10), 10);
            rr_q10 = silk_sub32(r_q10, q2_q10);
            rd2_q10 = silk_rshift(silk_smlabb(rd2_q10, rr_q10, rr_q10), 10);

            if rd1_q10 < rd2_q10 {
                ps_ss[0].rd_q10 = silk_add32(ps_dd.rd_q10, rd1_q10);
                ps_ss[1].rd_q10 = silk_add32(ps_dd.rd_q10, rd2_q10);
                ps_ss[0].q_q10 = q1_q10;
                ps_ss[1].q_q10 = q2_q10;
            } else {
                ps_ss[0].rd_q10 = silk_add32(ps_dd.rd_q10, rd2_q10);
                ps_ss[1].rd_q10 = silk_add32(ps_dd.rd_q10, rd1_q10);
                ps_ss[0].q_q10 = q2_q10;
                ps_ss[1].q_q10 = q1_q10;
            }

            // Update states for best quantization

            // Quantized excitation
            let mut exc_q14 = silk_lshift32(ps_ss[0].q_q10, 4);
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }

            // Add predictions
            let mut lpc_exc_q14 = silk_add32(exc_q14, ltp_pred_q14);
            let mut xq_q14 = silk_add32_ovflw(lpc_exc_q14, lpc_pred_q14);

            // Update states
            ps_ss[0].diff_q14 = silk_sub32_ovflw(xq_q14, silk_lshift32(x_q10[i], 4));
            let mut s_lf_ar_shp_q14 = silk_sub32_ovflw(ps_ss[0].diff_q14, n_ar_q14);
            ps_ss[0].s_ltp_shp_q14 = silk_sub_sat32(s_lf_ar_shp_q14, n_lf_q14);
            ps_ss[0].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_ss[0].lpc_exc_q14 = lpc_exc_q14;
            ps_ss[0].xq_q14 = xq_q14;

            // Update states for second best quantization

            // Quantized excitation
            exc_q14 = silk_lshift32(ps_ss[1].q_q10, 4);
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }

            // Add predictions
            lpc_exc_q14 = silk_add32(exc_q14, ltp_pred_q14);
            xq_q14 = silk_add32_ovflw(lpc_exc_q14, lpc_pred_q14);

            // Update states
            ps_ss[1].diff_q14 = silk_sub32_ovflw(xq_q14, silk_lshift32(x_q10[i], 4));
            s_lf_ar_shp_q14 = silk_sub32_ovflw(ps_ss[1].diff_q14, n_ar_q14);
            ps_ss[1].s_ltp_shp_q14 = silk_sub_sat32(s_lf_ar_shp_q14, n_lf_q14);
            ps_ss[1].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_ss[1].lpc_exc_q14 = lpc_exc_q14;
            ps_ss[1].xq_q14 = xq_q14;
        }

        *smpl_buf_idx = (*smpl_buf_idx - 1) % DECISION_DELAY;
        if *smpl_buf_idx < 0 {
            *smpl_buf_idx += DECISION_DELAY;
        }
        let last_smple_idx = ((*smpl_buf_idx + decision_delay) % DECISION_DELAY) as usize;
        let sbi = *smpl_buf_idx as usize;

        // Find winner
        let mut rdmin_q10 = ps_sample_state[0][0].rd_q10;
        let mut winner_ind = 0;
        for k in 1..n_states {
            if ps_sample_state[k][0].rd_q10 < rdmin_q10 {
                rdmin_q10 = ps_sample_state[k][0].rd_q10;
                winner_ind = k;
            }
        }

        // Increase RD values of expired states
        let winner_rand_state = ps_del_dec[winner_ind].rand_state[last_smple_idx];
        for k in 0..n_states {
            if ps_del_dec[k].rand_state[last_smple_idx] != winner_rand_state {
                ps_sample_state[k][0].rd_q10 =
                    silk_add32(ps_sample_state[k][0].rd_q10, SILK_INT32_MAX >> 4);
                ps_sample_state[k][1].rd_q10 =
                    silk_add32(ps_sample_state[k][1].rd_q10, SILK_INT32_MAX >> 4);
                silk_assert!(ps_sample_state[k][0].rd_q10 >= 0);
            }
        }

        // Find worst in first set and best in second set
        let mut rdmax_q10 = ps_sample_state[0][0].rd_q10;
        let mut rdmin_q10 = ps_sample_state[0][1].rd_q10;
        let mut rdmax_ind = 0;
        let mut rdmin_ind = 0;
        for k in 1..n_states {
            // find worst in first set
            if ps_sample_state[k][0].rd_q10 > rdmax_q10 {
                rdmax_q10 = ps_sample_state[k][0].rd_q10;
                rdmax_ind = k;
            }
            // find best in second set
            if ps_sample_state[k][1].rd_q10 < rdmin_q10 {
                rdmin_q10 = ps_sample_state[k][1].rd_q10;
                rdmin_ind = k;
            }
        }

        // Replace a state if best from second set outperforms worst in first set
        if rdmin_q10 < rdmax_q10 {
            // rdmin_ind != rdmax_ind: ps_sample_state[k][0].rd_q10 < [k][1].rd_q10 for every
            // state, so the worst first-set state cannot also be the best second-set state
            // when rdmin_q10 < rdmax_q10.
            let (dst, src) = if rdmax_ind < rdmin_ind {
                let (a, b) = ps_del_dec.split_at_mut(rdmin_ind);
                (&mut a[rdmax_ind], &b[0])
            } else {
                let (a, b) = ps_del_dec.split_at_mut(rdmax_ind);
                (&mut b[0], &a[rdmin_ind])
            };
            dst.copy_from_word(src, i);
            ps_sample_state[rdmax_ind][0] = ps_sample_state[rdmin_ind][1];
        }

        // Write samples from winner to output and long-term filter states
        let ps_dd = &ps_del_dec[winner_ind];
        if subfr > 0 || i >= dd {
            pulses[pulses_off + i - dd] = silk_rshift_round(ps_dd.q_q10[last_smple_idx], 10) as i8;
            nsq.xq[xq_off + i - dd] = silk_sat16(silk_rshift_round(
                silk_smulww(
                    ps_dd.xq_q14[last_smple_idx],
                    delayed_gain_q10[last_smple_idx],
                ),
                8,
            )) as i16;
            nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize - dd] =
                ps_dd.shape_q14[last_smple_idx];
            s_ltp_q15[nsq.s_ltp_buf_idx as usize - dd] = ps_dd.pred_q15[last_smple_idx];
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Update states
        for k in 0..n_states {
            let ps_dd = &mut ps_del_dec[k];
            let ps_ss = &ps_sample_state[k][0];
            ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
            ps_dd.diff_q14 = ps_ss.diff_q14;
            ps_dd.s_lpc_q14[NLBL + i] = ps_ss.xq_q14;
            ps_dd.xq_q14[sbi] = ps_ss.xq_q14;
            ps_dd.q_q10[sbi] = ps_ss.q_q10;
            ps_dd.pred_q15[sbi] = silk_lshift32(ps_ss.lpc_exc_q14, 1);
            ps_dd.shape_q14[sbi] = ps_ss.s_ltp_shp_q14;
            ps_dd.seed = silk_add32_ovflw(ps_dd.seed, silk_rshift_round(ps_ss.q_q10, 10));
            ps_dd.rand_state[sbi] = ps_dd.seed;
            ps_dd.rd_q10 = ps_ss.rd_q10;
        }
        delayed_gain_q10[sbi] = gain_q10;
    }
    // Update LPC states
    for ps_dd in ps_del_dec.iter_mut() {
        ps_dd.s_lpc_q14.copy_within(length..length + NLBL, 0);
    }
}

/// Port of silk/NSQ_del_dec.c:silk_nsq_del_dec_scale_states (static).
fn silk_nsq_del_dec_scale_states(
    ps_enc_c: &NsqEncParams,
    nsq: &mut SilkNsqState,
    ps_del_dec: &mut [NsqDelDecStruct],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decision_delay: i32,
) {
    let lag = pitch_l[subfr];
    let mut inv_gain_q31 = silk_inverse32_varq(silk_max(gains_q16[subfr], 1), 47);
    silk_assert!(inv_gain_q31 != 0);

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
            celt_assert!(i < MFL);
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
            for i in (nsq.s_ltp_buf_idx - lag - LTP_ORDER / 2) as usize
                ..(nsq.s_ltp_buf_idx - decision_delay) as usize
            {
                s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }

        for ps_dd in ps_del_dec.iter_mut() {
            // Scale scalar states
            ps_dd.lf_ar_q14 = silk_smulww(gain_adj_q16, ps_dd.lf_ar_q14);
            ps_dd.diff_q14 = silk_smulww(gain_adj_q16, ps_dd.diff_q14);

            // Scale short-term prediction and shaping states
            for i in 0..NLBL {
                ps_dd.s_lpc_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_lpc_q14[i]);
            }
            for i in 0..MSHP {
                ps_dd.s_ar2_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_ar2_q14[i]);
            }
            for i in 0..DD {
                ps_dd.pred_q15[i] = silk_smulww(gain_adj_q16, ps_dd.pred_q15[i]);
                ps_dd.shape_q14[i] = silk_smulww(gain_adj_q16, ps_dd.shape_q14[i]);
            }
        }

        // Save inverse gain
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

//! Port of silk/CNG.c: comfort noise generation for the SILK decoder.

use crate::silk::define::{
    CNG_BUF_MASK_MAX, CNG_GAIN_SMTH_Q16, CNG_GAIN_SMTH_THRESHOLD_Q16, CNG_NLSF_SMTH_Q16,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::macros::{
    SILK_INT16_MAX, silk_add_sat16, silk_add_sat32, silk_div32_16, silk_lshift_sat32,
    silk_lshift32, silk_rand, silk_rshift, silk_rshift_round, silk_sat16, silk_smlawb_chain,
    silk_smultt, silk_smulwb, silk_smulww, silk_sqrt_approx, silk_sub_lshift32,
};
use crate::silk::nlsf::silk_nlsf2a;
use crate::silk::structs::{SilkDecoderControl, SilkDecoderState};

const MLPC: usize = MAX_LPC_ORDER as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

/// Port of silk/CNG.c:silk_CNG_exc (static): generates excitation for CNG LPC synthesis.
///
/// Writes `length` samples to `exc_q14`, picking random entries of `exc_buf_q14`.
fn silk_cng_exc(exc_q14: &mut [i32], exc_buf_q14: &[i32], length: usize, rand_seed: &mut i32) {
    let mut exc_mask = CNG_BUF_MASK_MAX;
    while exc_mask > length as i32 {
        exc_mask = silk_rshift(exc_mask, 1);
    }

    let mut seed = *rand_seed;
    for e in &mut exc_q14[..length] {
        seed = silk_rand(seed);
        let idx = (silk_rshift(seed, 24) & exc_mask) as usize;
        silk_assert!(idx <= CNG_BUF_MASK_MAX as usize);
        *e = exc_buf_q14[idx];
    }
    *rand_seed = seed;
}

/// Port of silk/CNG.c:silk_CNG_Reset: resets the CNG state (smoothed NLSFs evenly spread,
/// zero gain, fixed seed).
pub fn silk_cng_reset(ps_dec: &mut SilkDecoderState) {
    let lpc_order = ps_dec.lpc_order;
    let nlsf_step_q15 = silk_div32_16(SILK_INT16_MAX, lpc_order + 1);
    let mut nlsf_acc_q15 = 0;
    for v in &mut ps_dec.s_cng.cng_smth_nlsf_q15[..lpc_order as usize] {
        nlsf_acc_q15 += nlsf_step_q15;
        *v = nlsf_acc_q15 as i16;
    }
    ps_dec.s_cng.cng_smth_gain_q16 = 0;
    ps_dec.s_cng.rand_seed = 3176576;
}

/// Port of silk/CNG.c:silk_CNG: updates the CNG estimate, and applies the CNG when the packet
/// was lost (or during DTX). `frame` holds `length` samples and is modified in place.
pub fn silk_cng(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &SilkDecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    if ps_dec.fs_khz != ps_dec.s_cng.fs_khz {
        // Reset state
        silk_cng_reset(ps_dec);

        ps_dec.s_cng.fs_khz = ps_dec.fs_khz;
    }
    let lpc_order = ps_dec.lpc_order as usize;
    let nb_subfr = ps_dec.nb_subfr as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let ps_cng = &mut ps_dec.s_cng;

    if ps_dec.loss_cnt == 0 && ps_dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Update CNG parameters

        // Smoothing of LSF's
        for i in 0..lpc_order {
            let smth = ps_cng.cng_smth_nlsf_q15[i] as i32;
            ps_cng.cng_smth_nlsf_q15[i] = (smth
                + silk_smulwb(ps_dec.prev_nlsf_q15[i] as i32 - smth, CNG_NLSF_SMTH_Q16))
                as i16;
        }
        // Find the subframe with the highest gain
        let mut max_gain_q16 = 0;
        let mut subfr = 0;
        for i in 0..nb_subfr {
            if ps_dec_ctrl.gains_q16[i] > max_gain_q16 {
                max_gain_q16 = ps_dec_ctrl.gains_q16[i];
                subfr = i;
            }
        }
        // Update CNG excitation buffer with excitation from this subframe
        ps_cng
            .cng_exc_buf_q14
            .copy_within(0..(nb_subfr - 1) * subfr_length, subfr_length);
        ps_cng.cng_exc_buf_q14[..subfr_length]
            .copy_from_slice(&ps_dec.exc_q14[subfr * subfr_length..(subfr + 1) * subfr_length]);

        // Smooth gains
        for i in 0..nb_subfr {
            ps_cng.cng_smth_gain_q16 += silk_smulwb(
                ps_dec_ctrl.gains_q16[i] - ps_cng.cng_smth_gain_q16,
                CNG_GAIN_SMTH_Q16,
            );
            // If the smoothed gain is 3 dB greater than this subframe's gain, use this
            // subframe's gain to adapt faster.
            if silk_smulww(ps_cng.cng_smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16)
                > ps_dec_ctrl.gains_q16[i]
            {
                ps_cng.cng_smth_gain_q16 = ps_dec_ctrl.gains_q16[i];
            }
        }
    }

    // Add CNG when packet is lost or during DTX
    if ps_dec.loss_cnt != 0 {
        let mut cng_sig_q14 = [0i32; MFL + MLPC];
        let mut a_q12 = [0i16; MLPC];

        // Generate CNG excitation
        let mut gain_q16 = silk_smulww(
            ps_dec.s_plc.rand_scale_q14 as i32,
            ps_dec.s_plc.prev_gain_q16[1],
        );
        if gain_q16 >= (1 << 21) || ps_cng.cng_smth_gain_q16 > (1 << 23) {
            gain_q16 = silk_smultt(gain_q16, gain_q16);
            gain_q16 = silk_sub_lshift32(
                silk_smultt(ps_cng.cng_smth_gain_q16, ps_cng.cng_smth_gain_q16),
                gain_q16,
                5,
            );
            gain_q16 = silk_lshift32(silk_sqrt_approx(gain_q16), 16);
        } else {
            gain_q16 = silk_smulww(gain_q16, gain_q16);
            gain_q16 = silk_sub_lshift32(
                silk_smulww(ps_cng.cng_smth_gain_q16, ps_cng.cng_smth_gain_q16),
                gain_q16,
                5,
            );
            gain_q16 = silk_lshift32(silk_sqrt_approx(gain_q16), 8);
        }
        let gain_q10 = silk_rshift(gain_q16, 6);

        silk_cng_exc(
            &mut cng_sig_q14[MLPC..],
            &ps_cng.cng_exc_buf_q14,
            length,
            &mut ps_cng.rand_seed,
        );

        // Convert CNG NLSF to filter representation
        silk_nlsf2a(&mut a_q12, &ps_cng.cng_smth_nlsf_q15, lpc_order);

        // Generate CNG signal, by synthesis filtering
        cng_sig_q14[..MLPC].copy_from_slice(&ps_cng.cng_synth_state);
        celt_assert!(lpc_order == 10 || lpc_order == 16);
        for i in 0..length {
            // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf.
            // Perf: `silk_smlawb_chain` is the C chain `silk_SMLAWB( lpc_pred_Q10,
            // CNG_sig_Q14[ MAX_LPC_ORDER + i - j - 1 ], A_Q12[ j ] )` for j = 0..LPC_order
            // (bit-identical, see there).
            let acc = silk_rshift(lpc_order as i32, 1);
            let lpc_pred_q10 = if lpc_order == 16 {
                silk_smlawb_chain(acc, &cng_sig_q14[i + MLPC - 16..i + MLPC], &a_q12)
            } else {
                silk_smlawb_chain(acc, &cng_sig_q14[i + MLPC - 10..i + MLPC], &a_q12)
            };

            // Update states
            let v = silk_add_sat32(cng_sig_q14[MLPC + i], silk_lshift_sat32(lpc_pred_q10, 4));
            cng_sig_q14[MLPC + i] = v;

            // Scale with Gain and add to input signal
            frame[i] = silk_add_sat16(
                frame[i],
                silk_sat16(silk_rshift_round(silk_smulww(v, gain_q10), 8)),
            );
        }
        ps_cng
            .cng_synth_state
            .copy_from_slice(&cng_sig_q14[length..length + MLPC]);
    } else {
        ps_cng.cng_synth_state[..lpc_order].fill(0);
    }
}

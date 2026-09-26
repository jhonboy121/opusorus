//! Port of silk/PLC.c and silk/PLC.h: packet loss concealment for the SILK decoder.
//!
//! DNN: with the `deep-plc` feature (C `ENABLE_DEEP_PLC`), [`silk_plc`] takes the Opus
//! decoder's LPCNet PLC state (`None` = C `NULL`): good 16 kHz frames update it and lost 16 kHz
//! frames are concealed by it when it has a model and the deep PLC is enabled (complexity >= 5)
//! or DRED features are queued. [`silk_plc_glue_frames`] then skips the energy fade at 16 kHz
//! (whether or not a model is loaded, as in C).

use crate::silk::define::{LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_VOICED};
use crate::silk::macros::{
    SILK_INT32_MAX, silk_add_sat32, silk_clz32, silk_div32, silk_div32_16, silk_fix_const,
    silk_inverse32_varq, silk_lshift, silk_lshift_sat32, silk_lshift32, silk_max, silk_max_16,
    silk_max_32, silk_max_int, silk_min, silk_min_32, silk_min_int, silk_rand, silk_rshift,
    silk_rshift_round, silk_sat16, silk_smlawb, silk_smlawb_chain, silk_smulbb, silk_smulwb,
    silk_smulww, silk_sqrt_approx,
};
use crate::silk::sigproc::{
    silk_bwexpander, silk_lpc_analysis_filter, silk_lpc_inverse_pred_gain, silk_sum_sqr_shift,
};
use crate::silk::structs::{SilkDecoderControl, SilkDecoderState};

#[cfg(feature = "deep-plc")]
use crate::dnn::lpcnet_plc::{LpcnetPlcState, lpcnet_plc_conceal, lpcnet_plc_update};

// ---------------------------------------------------------------------------------------------
// PLC.h
// ---------------------------------------------------------------------------------------------

/// `BWE_COEF`.
pub const BWE_COEF: f64 = 0.99;
/// `V_PITCH_GAIN_START_MIN_Q14`: 0.7 in Q14.
pub const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
/// `V_PITCH_GAIN_START_MAX_Q14`: 0.95 in Q14.
pub const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
/// `MAX_PITCH_LAG_MS`.
pub const MAX_PITCH_LAG_MS: i32 = 18;
/// `RAND_BUF_SIZE`.
pub const RAND_BUF_SIZE: i32 = 128;
/// `RAND_BUF_MASK`.
pub const RAND_BUF_MASK: i32 = RAND_BUF_SIZE - 1;
/// `LOG2_INV_LPC_GAIN_HIGH_THRES`: 2^3 = 8 dB LPC gain.
pub const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3;
/// `LOG2_INV_LPC_GAIN_LOW_THRES`: 2^8 = 24 dB LPC gain.
pub const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8;
/// `PITCH_DRIFT_FAC_Q16`: 0.01 in Q16.
pub const PITCH_DRIFT_FAC_Q16: i32 = 655;

// ---------------------------------------------------------------------------------------------
// PLC.c
// ---------------------------------------------------------------------------------------------

/// `NB_ATT`.
const NB_ATT: usize = 2;
/// `HARM_ATT_Q15`: 0.99, 0.95.
const HARM_ATT_Q15: [i16; NB_ATT] = [32440, 31130];
/// `PLC_RAND_ATTENUATE_V_Q15`: 0.95, 0.8.
const PLC_RAND_ATTENUATE_V_Q15: [i16; NB_ATT] = [31130, 26214];
/// `PLC_RAND_ATTENUATE_UV_Q15`: 0.99, 0.9.
const PLC_RAND_ATTENUATE_UV_Q15: [i16; NB_ATT] = [32440, 29491];

const LTPO: usize = LTP_ORDER as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

/// Port of silk/PLC.c:silk_PLC_Reset.
pub const fn silk_plc_reset(ps_dec: &mut SilkDecoderState) {
    ps_dec.s_plc.pitch_l_q8 = silk_lshift(ps_dec.frame_length, 8 - 1);
    ps_dec.s_plc.prev_gain_q16[0] = silk_fix_const(1.0, 16);
    ps_dec.s_plc.prev_gain_q16[1] = silk_fix_const(1.0, 16);
    ps_dec.s_plc.subfr_length = 20;
    ps_dec.s_plc.nb_subfr = 2;
}

/// Port of silk/PLC.c:silk_PLC: PLC control function. With `lost != 0`, generates a
/// concealment frame into `frame` (`frame_length` samples); otherwise updates the PLC state
/// from the just-decoded frame.
pub fn silk_plc(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    frame: &mut [i16],
    lost: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LpcnetPlcState>,
) {
    // PLC control function
    if ps_dec.fs_khz != ps_dec.s_plc.fs_khz {
        silk_plc_reset(ps_dec);
        ps_dec.s_plc.fs_khz = ps_dec.fs_khz;
    }

    if lost != 0 {
        // Generate Signal
        silk_plc_conceal(
            ps_dec,
            ps_dec_ctrl,
            frame,
            #[cfg(feature = "deep-plc")]
            lpcnet,
        );

        ps_dec.loss_cnt += 1;
    } else {
        // Update state
        silk_plc_update(ps_dec, ps_dec_ctrl);
        #[cfg(feature = "deep-plc")]
        if let Some(lpcnet) = lpcnet
            && ps_dec.s_plc.fs_khz == 16
        {
            let sl = ps_dec.subfr_length as usize;
            for k in (0..ps_dec.nb_subfr as usize).step_by(2) {
                lpcnet_plc_update(lpcnet, &frame[k * sl..]);
            }
        }
    }
}

/// Port of silk/PLC.c:silk_PLC_update (static): update state of PLC.
fn silk_plc_update(ps_dec: &mut SilkDecoderState, ps_dec_ctrl: &SilkDecoderControl) {
    let nb_subfr = ps_dec.nb_subfr;
    let subfr_length = ps_dec.subfr_length;
    let ps_plc = &mut ps_dec.s_plc;

    // Update parameters used in case of packet loss
    ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
    let mut ltp_gain_q14: i32 = 0;
    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Find the parameters for the last subframe which contains a pitch pulse
        let mut j: i32 = 0;
        while j * subfr_length < ps_dec_ctrl.pitch_l[(nb_subfr - 1) as usize] {
            if j == nb_subfr {
                break;
            }
            let base = ((nb_subfr - 1 - j) as usize) * LTPO;
            let mut temp_ltp_gain_q14: i32 = 0;
            for i in 0..LTPO {
                temp_ltp_gain_q14 += ps_dec_ctrl.ltp_coef_q14[base + i] as i32;
            }
            if temp_ltp_gain_q14 > ltp_gain_q14 {
                ltp_gain_q14 = temp_ltp_gain_q14;
                let off = silk_smulbb(nb_subfr - 1 - j, LTP_ORDER) as usize;
                ps_plc
                    .ltp_coef_q14
                    .copy_from_slice(&ps_dec_ctrl.ltp_coef_q14[off..off + LTPO]);

                ps_plc.pitch_l_q8 =
                    silk_lshift(ps_dec_ctrl.pitch_l[(nb_subfr - 1 - j) as usize], 8);
            }
            j += 1;
        }

        ps_plc.ltp_coef_q14 = [0; LTPO];
        ps_plc.ltp_coef_q14[LTPO / 2] = ltp_gain_q14 as i16;

        // Limit LT coefs
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let tmp = silk_lshift(V_PITCH_GAIN_START_MIN_Q14, 10);
            let scale_q10 = silk_div32(tmp, silk_max(ltp_gain_q14, 1));
            for c in &mut ps_plc.ltp_coef_q14 {
                *c = silk_rshift(silk_smulbb(*c as i32, scale_q10), 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let tmp = silk_lshift(V_PITCH_GAIN_START_MAX_Q14, 14);
            let scale_q14 = silk_div32(tmp, silk_max(ltp_gain_q14, 1));
            for c in &mut ps_plc.ltp_coef_q14 {
                *c = silk_rshift(silk_smulbb(*c as i32, scale_q14), 14) as i16;
            }
        }
    } else {
        ps_plc.pitch_l_q8 = silk_lshift(silk_smulbb(ps_dec.fs_khz, 18), 8);
        ps_plc.ltp_coef_q14 = [0; LTPO];
    }

    // Save LPC coefficients
    let lpc_order = ps_dec.lpc_order as usize;
    ps_plc.prev_lpc_q12[..lpc_order].copy_from_slice(&ps_dec_ctrl.pred_coef_q12[1][..lpc_order]);
    ps_plc.prev_ltp_scale_q14 = ps_dec_ctrl.ltp_scale_q14 as i16;

    // Save last two gains
    let nb = nb_subfr as usize;
    ps_plc
        .prev_gain_q16
        .copy_from_slice(&ps_dec_ctrl.gains_q16[nb - 2..nb]);

    ps_plc.subfr_length = subfr_length;
    ps_plc.nb_subfr = nb_subfr;
}

/// Port of silk/PLC.c:silk_PLC_energy (static): energies of the last two subframes of the
/// previous excitation, scaled by the previous gains. Returns `(energy1, shift1, energy2,
/// shift2)`.
fn silk_plc_energy(
    exc_q14: &[i32],
    prev_gain_q10: &[i32; 2],
    subfr_length: usize,
    nb_subfr: usize,
) -> (i32, i32, i32, i32) {
    let mut exc_buf = [0i16; 2 * MFL / 4];
    let exc_buf = &mut exc_buf[..2 * subfr_length];
    // Find random noise component
    // Scale previous excitation signal
    for k in 0..2 {
        for i in 0..subfr_length {
            exc_buf[k * subfr_length + i] = silk_sat16(silk_rshift(
                silk_smulww(
                    exc_q14[i + (k + nb_subfr - 2) * subfr_length],
                    prev_gain_q10[k],
                ),
                8,
            )) as i16;
        }
    }
    // Find the subframe with lowest energy of the last two and use that as random noise
    // generator
    let (energy1, shift1) = silk_sum_sqr_shift(exc_buf, subfr_length);
    let (energy2, shift2) = silk_sum_sqr_shift(&exc_buf[subfr_length..], subfr_length);
    (energy1, shift1, energy2, shift2)
}

/// Port of silk/PLC.c:silk_PLC_conceal (static): generates a concealment frame.
fn silk_plc_conceal(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    frame: &mut [i16],
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LpcnetPlcState>,
) {
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;
    let frame_length = ps_dec.frame_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;
    let mut s_ltp_q14 = [0i32; 2 * MFL];
    let mut s_ltp = [0i16; MFL];
    let mut a_q12 = [0i16; MLPC];

    let ps_plc = &mut ps_dec.s_plc;

    let prev_gain_q10 = [
        silk_rshift(ps_plc.prev_gain_q16[0], 6),
        silk_rshift(ps_plc.prev_gain_q16[1], 6),
    ];

    if ps_dec.first_frame_after_reset != 0 {
        ps_plc.prev_lpc_q12 = [0; MLPC];
    }

    let (energy1, shift1, energy2, shift2) = silk_plc_energy(
        &ps_dec.exc_q14,
        &prev_gain_q10,
        ps_dec.subfr_length as usize,
        ps_dec.nb_subfr as usize,
    );

    let rand_off = if silk_rshift(energy1, shift2) < silk_rshift(energy2, shift1) {
        // First sub-frame has lowest energy
        silk_max_int(
            0,
            (ps_plc.nb_subfr - 1) * ps_plc.subfr_length - RAND_BUF_SIZE,
        )
    } else {
        // Second sub-frame has lowest energy
        silk_max_int(0, ps_plc.nb_subfr * ps_plc.subfr_length - RAND_BUF_SIZE)
    } as usize;
    let rand_ptr = &ps_dec.exc_q14[rand_off..];

    // Set up Gain to random noise component
    // (B_Q14 aliases psPLC->LTPCoef_Q14 in C: it is modified in place below.)
    let mut rand_scale_q14 = ps_plc.rand_scale_q14;

    // Set up attenuation gains
    let att = silk_min_int(NB_ATT as i32 - 1, ps_dec.loss_cnt) as usize;
    let harm_gain_q15 = HARM_ATT_Q15[att] as i32;
    let mut rand_gain_q15 = if ps_dec.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[att] as i32
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[att] as i32
    };

    // LPC concealment. Apply BWE to previous LPC
    silk_bwexpander(
        &mut ps_plc.prev_lpc_q12,
        lpc_order,
        silk_fix_const(BWE_COEF, 16),
    );

    // Preload LPC coefficients to array on stack. Gives small performance gain
    a_q12[..lpc_order].copy_from_slice(&ps_plc.prev_lpc_q12[..lpc_order]);

    // First Lost frame
    if ps_dec.loss_cnt == 0 {
        rand_scale_q14 = 1 << 14;

        // Reduce random noise Gain for voiced frames
        if ps_dec.prev_signal_type == TYPE_VOICED {
            for i in 0..LTPO {
                rand_scale_q14 = (rand_scale_q14 as i32 - ps_plc.ltp_coef_q14[i] as i32) as i16;
            }
            rand_scale_q14 = silk_max_16(3277, rand_scale_q14); // 0.2
            rand_scale_q14 = silk_rshift(
                silk_smulbb(rand_scale_q14 as i32, ps_plc.prev_ltp_scale_q14 as i32),
                14,
            ) as i16;
        } else {
            // Reduce random noise for unvoiced frames with high LPC gain
            let inv_gain_q30 = silk_lpc_inverse_pred_gain(&ps_plc.prev_lpc_q12, lpc_order);

            let mut down_scale_q30 = silk_min_32(
                silk_rshift(1i32 << 30, LOG2_INV_LPC_GAIN_HIGH_THRES),
                inv_gain_q30,
            );
            down_scale_q30 = silk_max_32(
                silk_rshift(1i32 << 30, LOG2_INV_LPC_GAIN_LOW_THRES),
                down_scale_q30,
            );
            down_scale_q30 = silk_lshift(down_scale_q30, LOG2_INV_LPC_GAIN_HIGH_THRES);

            rand_gain_q15 = silk_rshift(silk_smulwb(down_scale_q30, rand_gain_q15), 14);
        }
    }

    let mut rand_seed = ps_plc.rand_seed;
    let mut lag = silk_rshift_round(ps_plc.pitch_l_q8, 8);
    let mut s_ltp_buf_idx = ltp_mem_length;

    // Rewhiten LTP state
    let idx = ps_dec.ltp_mem_length - lag - ps_dec.lpc_order - LTP_ORDER / 2;
    debug_assert!(idx > 0);
    let idx = idx as usize;
    silk_lpc_analysis_filter(
        &mut s_ltp[idx..],
        &ps_dec.out_buf[idx..],
        &a_q12,
        ltp_mem_length - idx,
        lpc_order,
    );
    // Scale LTP state
    let mut inv_gain_q30 = silk_inverse32_varq(ps_plc.prev_gain_q16[1], 46);
    inv_gain_q30 = silk_min(inv_gain_q30, SILK_INT32_MAX >> 1);
    for i in idx + lpc_order..ltp_mem_length {
        s_ltp_q14[i] = silk_smulwb(inv_gain_q30, s_ltp[i] as i32);
    }

    // LTP synthesis filtering
    for _k in 0..ps_dec.nb_subfr {
        // Set up pointer
        let p0 = s_ltp_buf_idx - lag as usize + LTPO / 2;
        let b_q14 = &mut ps_plc.ltp_coef_q14;
        for i in 0..ps_dec.subfr_length as usize {
            let p = p0 + i;
            // Unrolled loop
            // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
            let mut ltp_pred_q12: i32 = 2;
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[p], b_q14[0] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[p - 1], b_q14[1] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[p - 2], b_q14[2] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[p - 3], b_q14[3] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[p - 4], b_q14[4] as i32);

            // Generate LPC excitation
            rand_seed = silk_rand(rand_seed);
            let ridx = (silk_rshift(rand_seed, 25) & RAND_BUF_MASK) as usize;
            s_ltp_q14[s_ltp_buf_idx] = silk_lshift32(
                silk_smlawb(ltp_pred_q12, rand_ptr[ridx], rand_scale_q14 as i32),
                2,
            );
            s_ltp_buf_idx += 1;
        }

        // Gradually reduce LTP gain
        for b in b_q14.iter_mut() {
            *b = silk_rshift(silk_smulbb(harm_gain_q15, *b as i32), 15) as i16;
        }
        // Gradually reduce excitation gain
        rand_scale_q14 = silk_rshift(silk_smulbb(rand_scale_q14 as i32, rand_gain_q15), 15) as i16;

        // Slowly increase pitch lag
        ps_plc.pitch_l_q8 = silk_smlawb(ps_plc.pitch_l_q8, ps_plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
        ps_plc.pitch_l_q8 = silk_min_32(
            ps_plc.pitch_l_q8,
            silk_lshift(silk_smulbb(MAX_PITCH_LAG_MS, ps_dec.fs_khz), 8),
        );
        lag = silk_rshift_round(ps_plc.pitch_l_q8, 8);
    }

    // LPC synthesis filtering
    let base = ltp_mem_length - MLPC; // sLPC_Q14_ptr = &sLTP_Q14[ltp_mem_length - MAX_LPC_ORDER]
    let s_lpc_q14 = &mut s_ltp_q14[base..base + MLPC + frame_length];

    // Copy LPC state
    s_lpc_q14[..MLPC].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    debug_assert!(lpc_order >= 10); // check that unrolling works
    for i in 0..frame_length {
        // partly unrolled
        // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
        // Perf: `silk_smlawb_chain` is the C chain `silk_SMLAWB( lpc_pred_Q10,
        // sLPC_Q14_ptr[ MAX_LPC_ORDER + i - j - 1 ], A_Q12[ j ] )` for j = 0..LPC_order
        // (bit-identical, see there); the branches give the window a constant length.
        let acc = silk_rshift(lpc_order as i32, 1);
        let lpc_pred_q10 = match lpc_order {
            16 => silk_smlawb_chain(acc, &s_lpc_q14[i + MLPC - 16..i + MLPC], &a_q12),
            10 => silk_smlawb_chain(acc, &s_lpc_q14[i + MLPC - 10..i + MLPC], &a_q12),
            _ => silk_smlawb_chain(acc, &s_lpc_q14[i + MLPC - lpc_order..i + MLPC], &a_q12),
        };

        // Add prediction to LPC excitation
        let v = silk_add_sat32(s_lpc_q14[MLPC + i], silk_lshift_sat32(lpc_pred_q10, 4));
        s_lpc_q14[MLPC + i] = v;

        // Scale with Gain
        frame[i] = silk_sat16(silk_sat16(silk_rshift_round(
            silk_smulww(v, prev_gain_q10[1]),
            8,
        ))) as i16;
    }
    #[cfg(feature = "deep-plc")]
    if let Some(lpcnet) = lpcnet
        && lpcnet.loaded
        && ps_plc.fs_khz == 16
    {
        let run_deep_plc = ps_plc.enable_deep_plc != 0 || lpcnet.fec_fill_pos != 0;
        let sl = ps_dec.subfr_length as usize;
        if run_deep_plc {
            for k in (0..ps_dec.nb_subfr as usize).step_by(2) {
                lpcnet_plc_conceal(lpcnet, &mut frame[k * sl..]);
            }
            // We *should* be able to copy only from psDec->frame_length-MAX_LPC_ORDER, i.e.
            // the last MAX_LPC_ORDER samples.
            for i in 0..frame_length {
                // C: `(int)floor(.5 + frame[i]*(float)(1 << 24)/prevGain_Q10[1])` (the float
                // quotient is promoted to double by the `.5 +`).
                let q = f32::from(frame[i]) * 16_777_216.0f32 / prev_gain_q10[1] as f32;
                s_lpc_q14[MLPC + i] = crate::math::floor(0.5f64 + f64::from(q)) as i32;
            }
        } else {
            for k in (0..ps_dec.nb_subfr as usize).step_by(2) {
                lpcnet_plc_update(lpcnet, &frame[k * sl..]);
            }
        }
    }

    // Save LPC state
    ps_dec
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[frame_length..frame_length + MLPC]);

    // Update states
    ps_plc.rand_seed = rand_seed;
    ps_plc.rand_scale_q14 = rand_scale_q14;
    for i in 0..MAX_NB_SUBFR as usize {
        ps_dec_ctrl.pitch_l[i] = lag;
    }
}

/// Port of silk/PLC.c:silk_PLC_glue_frames: glues concealed frames with new good received
/// frames (fades in the energy difference). `frame` holds `length` samples.
pub fn silk_plc_glue_frames(ps_dec: &mut SilkDecoderState, frame: &mut [i16], length: usize) {
    let ps_plc = &mut ps_dec.s_plc;

    if ps_dec.loss_cnt != 0 {
        // Calculate energy in concealed residual
        let (e, s) = silk_sum_sqr_shift(frame, length);
        ps_plc.conc_energy = e;
        ps_plc.conc_energy_shift = s;

        ps_plc.last_frame_lost = 1;
    } else {
        if ps_plc.last_frame_lost != 0 {
            // Calculate residual in decoded signal if last frame was lost
            let (mut energy, energy_shift) = silk_sum_sqr_shift(frame, length);

            // Normalize energies
            if energy_shift > ps_plc.conc_energy_shift {
                ps_plc.conc_energy =
                    silk_rshift(ps_plc.conc_energy, energy_shift - ps_plc.conc_energy_shift);
            } else if energy_shift < ps_plc.conc_energy_shift {
                energy = silk_rshift(energy, ps_plc.conc_energy_shift - energy_shift);
            }

            // Fade in the energy difference
            if energy > ps_plc.conc_energy {
                let mut lz = silk_clz32(ps_plc.conc_energy);
                lz -= 1;
                ps_plc.conc_energy = silk_lshift(ps_plc.conc_energy, lz);
                energy = silk_rshift(energy, silk_max_32(24 - lz, 0));

                let frac_q24 = silk_div32(ps_plc.conc_energy, silk_max(energy, 1));

                let mut gain_q16 = silk_lshift(silk_sqrt_approx(frac_q24), 4);
                let mut slope_q16 = silk_div32_16((1i32 << 16) - gain_q16, length as i32);
                // Make slope 4x steeper to avoid missing onsets after DTX
                slope_q16 = silk_lshift(slope_q16, 2);
                // C: `#ifdef ENABLE_DEEP_PLC if ( psDec->sPLC.fs_kHz != 16 ) #endif`.
                if !cfg!(feature = "deep-plc") || ps_plc.fs_khz != 16 {
                    for f in &mut frame[..length] {
                        *f = silk_smulwb(gain_q16, *f as i32) as i16;
                        gain_q16 += slope_q16;
                        if gain_q16 > 1i32 << 16 {
                            break;
                        }
                    }
                }
            }
        }
        ps_plc.last_frame_lost = 0;
    }
}

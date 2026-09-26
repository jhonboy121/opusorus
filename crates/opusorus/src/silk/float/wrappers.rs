//! Port of `silk/float/wrappers_FLP.c`: float wrappers around the fixed-point NLSF, NSQ and LTP
//! quantization code.

use crate::silk::define::MAX_SHAPE_LPC_ORDER;
use crate::silk::define::{LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_VOICED};
use crate::silk::encoder_common::{silk_process_nlsfs, silk_quant_ltp_gains};
use crate::silk::float::sigproc::silk_float2int;
use crate::silk::float::structs::SilkEncoderControlFlp;
use crate::silk::macros::silk_lshift32;
use crate::silk::nlsf::{silk_a2nlsf, silk_nlsf2a};
use crate::silk::nsq::{NsqEncParams, silk_nsq_c};
use crate::silk::nsq_del_dec::silk_nsq_del_dec_c;
use crate::silk::structs::{SideInfoIndices, SilkEncoderState, SilkNsqState};
use crate::silk::tables::SILK_LTPSCALES_TABLE_Q14;

const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const LTPO: usize = LTP_ORDER as usize;
const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

/// Port of `silk/float/wrappers_FLP.c:silk_A2NLSF_FLP` — convert AR filter coefficients to NLSF
/// parameters.
pub fn silk_a2nlsf_flp(nlsf_q15: &mut [i16], p_ar: &[f32], lpc_order: usize) {
    let mut a_fix_q16 = [0i32; MLPC];
    for i in 0..lpc_order {
        a_fix_q16[i] = silk_float2int(p_ar[i] * 65536.0f32);
    }
    silk_a2nlsf(nlsf_q15, &mut a_fix_q16, lpc_order);
}

/// Port of `silk/float/wrappers_FLP.c:silk_NLSF2A_FLP` — convert NLSF parameters to AR
/// prediction filter coefficients.
pub fn silk_nlsf2a_flp(p_ar: &mut [f32], nlsf_q15: &[i16], lpc_order: usize) {
    let mut a_fix_q12 = [0i16; MLPC];
    silk_nlsf2a(&mut a_fix_q12, nlsf_q15, lpc_order);
    for i in 0..lpc_order {
        p_ar[i] = a_fix_q12[i] as f32 * (1.0f32 / 4096.0f32);
    }
}

/// Port of `silk/float/wrappers_FLP.c:silk_process_NLSFs_FLP` — floating-point NLSF processing
/// wrapper.
///
/// C passes `prev_NLSF_Q15 = psEnc->sCmn.prev_NLSFq_Q15`, which aliases the state; Rust callers
/// pass a copy.
pub fn silk_process_nlsfs_flp(
    ps_enc_c: &mut SilkEncoderState,
    pred_coef: &mut [[f32; MLPC]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
) {
    let mut pred_coef_q12 = [[0i16; MLPC]; 2];
    silk_process_nlsfs(ps_enc_c, &mut pred_coef_q12, nlsf_q15, prev_nlsf_q15);
    for j in 0..2 {
        for i in 0..ps_enc_c.predict_lpc_order as usize {
            pred_coef[j][i] = pred_coef_q12[j][i] as f32 * (1.0f32 / 4096.0f32);
        }
    }
}

/// Port of `silk/float/wrappers_FLP.c:silk_NSQ_wrapper_FLP` — floating-point Silk NSQ wrapper.
///
/// C takes the whole `silk_encoder_state_FLP` (plus pointers into it for `psIndices`, `psNSQ`,
/// `pulses` and `x`); the fields it reads are passed as [`NsqEncParams`] so the other
/// arguments can borrow the state mutably. `x` is the (pre-filtered) input frame.
pub fn silk_nsq_wrapper_flp(
    ps_enc: &NsqEncParams,
    ps_enc_ctrl: &SilkEncoderControlFlp,
    ps_indices: &mut SideInfoIndices,
    ps_nsq: &mut SilkNsqState,
    pulses: &mut [i8],
    x: &[f32],
) {
    let mut x16 = [0i16; MFL];
    let mut gains_q16 = [0i32; MNSF];
    let mut pred_coef_q12 = [0i16; 2 * MLPC];
    let mut ltp_coef_q14 = [0i16; LTPO * MNSF];

    // Noise shaping parameters
    let mut ar_q13 = [0i16; MNSF * MSO];
    let mut lf_shp_q14 = [0i32; MNSF]; // Packs two int16 coefficients per int32 value
    let mut tilt_q14 = [0i32; MNSF];
    let mut harm_shape_gain_q14 = [0i32; MNSF];

    let nb_subfr = ps_enc.nb_subfr as usize;

    // Convert control struct to fix control struct
    // Noise shape parameters
    for i in 0..nb_subfr {
        for j in 0..ps_enc.shaping_lpc_order as usize {
            ar_q13[i * MSO + j] = silk_float2int(ps_enc_ctrl.ar[i * MSO + j] * 8192.0f32) as i16;
        }
    }

    for i in 0..nb_subfr {
        lf_shp_q14[i] = silk_lshift32(silk_float2int(ps_enc_ctrl.lf_ar_shp[i] * 16384.0f32), 16)
            | (silk_float2int(ps_enc_ctrl.lf_ma_shp[i] * 16384.0f32) as u16 as i32);
        tilt_q14[i] = silk_float2int(ps_enc_ctrl.tilt[i] * 16384.0f32);
        harm_shape_gain_q14[i] = silk_float2int(ps_enc_ctrl.harm_shape_gain[i] * 16384.0f32);
    }
    let lambda_q10 = silk_float2int(ps_enc_ctrl.lambda * 1024.0f32);

    // prediction and coding parameters
    for i in 0..nb_subfr * LTPO {
        ltp_coef_q14[i] = silk_float2int(ps_enc_ctrl.ltp_coef[i] * 16384.0f32) as i16;
    }

    for j in 0..2 {
        for i in 0..ps_enc.predict_lpc_order as usize {
            pred_coef_q12[j * MLPC + i] =
                silk_float2int(ps_enc_ctrl.pred_coef[j][i] * 4096.0f32) as i16;
        }
    }

    for i in 0..nb_subfr {
        gains_q16[i] = silk_float2int(ps_enc_ctrl.gains[i] * 65536.0f32);
    }

    let ltp_scale_q14 = if ps_indices.signal_type as i32 == TYPE_VOICED {
        SILK_LTPSCALES_TABLE_Q14[ps_indices.ltp_scale_index as usize] as i32
    } else {
        0
    };

    // Convert input to fix
    let frame_length = ps_enc.frame_length as usize;
    for i in 0..frame_length {
        x16[i] = silk_float2int(x[i]) as i16;
    }

    // Call NSQ
    if ps_enc.n_states_delayed_decision > 1 || ps_enc.warping_q16 > 0 {
        silk_nsq_del_dec_c(
            ps_enc,
            ps_nsq,
            ps_indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ps_enc_ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    } else {
        silk_nsq_c(
            ps_enc,
            ps_nsq,
            ps_indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ps_enc_ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    }
}

/// Port of `silk/float/wrappers_FLP.c:silk_quant_LTP_gains_FLP` — floating-point LTP
/// quantization wrapper.
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db: &mut f32,
    xx_mat: &[f32],
    xx_vec: &[f32],
    subfr_len: i32,
    nb_subfr: usize,
) {
    let mut b_q14 = [0i16; MNSF * LTPO];
    let mut xx_q17 = [0i32; MNSF * LTPO * LTPO];
    let mut x_x_q17 = [0i32; MNSF * LTPO];
    let mut pred_gain_db_q7 = 0i32;

    // C: do { ... } while ( ++i < n ) (runs at least once)
    for i in 0..(nb_subfr * LTPO * LTPO).max(1) {
        xx_q17[i] = silk_float2int(xx_mat[i] * 131072.0f32);
    }
    for i in 0..(nb_subfr * LTPO).max(1) {
        x_x_q17[i] = silk_float2int(xx_vec[i] * 131072.0f32);
    }

    silk_quant_ltp_gains(
        &mut b_q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_db_q7,
        &xx_q17,
        &x_x_q17,
        subfr_len,
        nb_subfr,
    );

    for i in 0..nb_subfr * LTPO {
        b[i] = b_q14[i] as f32 * (1.0f32 / 16384.0f32);
    }

    *pred_gain_db = pred_gain_db_q7 as f32 * (1.0f32 / 128.0f32);
}

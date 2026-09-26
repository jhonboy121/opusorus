//! Port of `silk/fixed/structs_FIX.h`: the fixed-point encoder state and control structs.
//!
//! The encoder super-struct `silk_encoder` (also declared in `structs_FIX.h`) lives in
//! [`crate::silk::encoder`] as `SilkEncoder`, next to the `enc_API.c` functions that use it.

use crate::silk::define::{
    LA_SHAPE_MAX, LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER,
};
use crate::silk::structs::SilkEncoderState;

const MFL: usize = MAX_FRAME_LENGTH as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;
const LTPO: usize = LTP_ORDER as usize;
const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;

/// Length of [`SilkEncoderStateFix::x_buf`]: `2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX`.
pub const X_BUF_LEN: usize = 2 * MFL + LA_SHAPE_MAX as usize;

/// `silk_shape_state_FIX`: noise shaping analysis state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SilkShapeStateFix {
    pub last_gain_index: i8,
    /// Unused by the encoder (C keeps the field; it stays zero).
    pub harm_boost_smth_q16: i32,
    pub harm_shape_gain_smth_q16: i32,
    pub tilt_smth_q16: i32,
}

/// `silk_encoder_state_FIX`: encoder state FIX.
#[derive(Debug, Clone)]
pub struct SilkEncoderStateFix {
    /// Common struct, shared with floating-point code.
    pub s_cmn: SilkEncoderState,
    /// Shape state.
    pub s_shape: SilkShapeStateFix,
    /// Buffer for find pitch and noise shape analysis.
    pub x_buf: [i16; X_BUF_LEN],
    /// Normalized correlation from pitch lag estimator.
    pub ltp_corr_q15: i32,
    /// Unused by the encoder (C keeps the field; it stays zero).
    pub res_nrg_smth: i32,
}

impl SilkEncoderStateFix {
    /// All-zero state (C `silk_memset( psEnc, 0, sizeof( silk_encoder_state_FIX ) )`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            s_cmn: SilkEncoderState::new(),
            s_shape: SilkShapeStateFix::default(),
            x_buf: [0; X_BUF_LEN],
            ltp_corr_q15: 0,
            res_nrg_smth: 0,
        }
    }
}

impl Default for SilkEncoderStateFix {
    fn default() -> Self {
        Self::new()
    }
}

/// `silk_encoder_control_FIX`: encoder control FIX (per-frame analysis results).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkEncoderControlFix {
    // Prediction and coding parameters
    pub gains_q16: [i32; MNSF],
    pub pred_coef_q12: [[i16; MLPC]; 2],
    pub ltp_coef_q14: [i16; LTPO * MNSF],
    pub ltp_scale_q14: i32,
    pub pitch_l: [i32; MNSF],

    // Noise shaping parameters
    pub ar_q13: [i16; MNSF * MSO],
    /// Packs two int16 coefficients per int32 value.
    pub lf_shp_q14: [i32; MNSF],
    pub tilt_q14: [i32; MNSF],
    pub harm_shape_gain_q14: [i32; MNSF],
    pub lambda_q10: i32,
    pub input_quality_q14: i32,
    pub coding_quality_q14: i32,

    // Measures
    pub pred_gain_q16: i32,
    pub lt_pred_cod_gain_q7: i32,
    /// Residual energy per subframe.
    pub res_nrg: [i32; MNSF],
    /// Q domain for the residual energy > 0.
    pub res_nrg_q: [i32; MNSF],

    // Parameters for CBR mode
    pub gains_unq_q16: [i32; MNSF],
    pub last_gain_index_prev: i8,
}

impl SilkEncoderControlFix {
    /// All-zero control struct. C leaves the local `sEncCtrl` uninitialized; every field is
    /// written before it is read.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gains_q16: [0; MNSF],
            pred_coef_q12: [[0; MLPC]; 2],
            ltp_coef_q14: [0; LTPO * MNSF],
            ltp_scale_q14: 0,
            pitch_l: [0; MNSF],
            ar_q13: [0; MNSF * MSO],
            lf_shp_q14: [0; MNSF],
            tilt_q14: [0; MNSF],
            harm_shape_gain_q14: [0; MNSF],
            lambda_q10: 0,
            input_quality_q14: 0,
            coding_quality_q14: 0,
            pred_gain_q16: 0,
            lt_pred_cod_gain_q7: 0,
            res_nrg: [0; MNSF],
            res_nrg_q: [0; MNSF],
            gains_unq_q16: [0; MNSF],
            last_gain_index_prev: 0,
        }
    }
}

impl Default for SilkEncoderControlFix {
    fn default() -> Self {
        Self::new()
    }
}

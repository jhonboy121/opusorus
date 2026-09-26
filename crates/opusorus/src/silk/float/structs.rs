//! Port of `silk/float/structs_FLP.h`: the floating-point encoder state and control structs.
//!
//! The encoder super-struct `silk_encoder` (also declared in `structs_FLP.h`) lives in
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

/// Length of [`SilkEncoderStateFlp::x_buf`]: `2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX`.
pub const X_BUF_LEN: usize = 2 * MFL + LA_SHAPE_MAX as usize;

/// `silk_shape_state_FLP`: noise shaping analysis state.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SilkShapeStateFlp {
    pub last_gain_index: i8,
    pub harm_shape_gain_smth: f32,
    pub tilt_smth: f32,
}

/// `silk_encoder_state_FLP`: encoder state FLP.
#[derive(Debug, Clone)]
pub struct SilkEncoderStateFlp {
    /// Common struct, shared with fixed-point code.
    pub s_cmn: SilkEncoderState,
    /// Noise shaping state.
    pub s_shape: SilkShapeStateFlp,
    /// Buffer for find pitch and noise shape analysis.
    pub x_buf: [f32; X_BUF_LEN],
    /// Normalized correlation from pitch lag estimator.
    pub ltp_corr: f32,
}

impl SilkEncoderStateFlp {
    /// All-zero state (C `silk_memset( psEnc, 0, sizeof( silk_encoder_state_FLP ) )`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            s_cmn: SilkEncoderState::new(),
            s_shape: SilkShapeStateFlp::default(),
            x_buf: [0.0; X_BUF_LEN],
            ltp_corr: 0.0,
        }
    }
}

impl Default for SilkEncoderStateFlp {
    fn default() -> Self {
        Self::new()
    }
}

/// `silk_encoder_control_FLP`: encoder control FLP (per-frame analysis results).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilkEncoderControlFlp {
    // Prediction and coding parameters
    pub gains: [f32; MNSF],
    /// Holds interpolated and final coefficients.
    pub pred_coef: [[f32; MLPC]; 2],
    pub ltp_coef: [f32; LTPO * MNSF],
    pub ltp_scale: f32,
    pub pitch_l: [i32; MNSF],

    // Noise shaping parameters
    pub ar: [f32; MNSF * MSO],
    pub lf_ma_shp: [f32; MNSF],
    pub lf_ar_shp: [f32; MNSF],
    pub tilt: [f32; MNSF],
    pub harm_shape_gain: [f32; MNSF],
    pub lambda: f32,
    pub input_quality: f32,
    pub coding_quality: f32,

    // Measures
    pub pred_gain: f32,
    pub lt_pred_cod_gain: f32,
    /// Residual energy per subframe.
    pub res_nrg: [f32; MNSF],

    // Parameters for CBR mode
    pub gains_unq_q16: [i32; MNSF],
    pub last_gain_index_prev: i8,
}

impl SilkEncoderControlFlp {
    /// All-zero control struct. C leaves the local `sEncCtrl` uninitialized; every field is
    /// written before it is read.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gains: [0.0; MNSF],
            pred_coef: [[0.0; MLPC]; 2],
            ltp_coef: [0.0; LTPO * MNSF],
            ltp_scale: 0.0,
            pitch_l: [0; MNSF],
            ar: [0.0; MNSF * MSO],
            lf_ma_shp: [0.0; MNSF],
            lf_ar_shp: [0.0; MNSF],
            tilt: [0.0; MNSF],
            harm_shape_gain: [0.0; MNSF],
            lambda: 0.0,
            input_quality: 0.0,
            coding_quality: 0.0,
            pred_gain: 0.0,
            lt_pred_cod_gain: 0.0,
            res_nrg: [0.0; MNSF],
            gains_unq_q16: [0; MNSF],
            last_gain_index_prev: 0,
        }
    }
}

impl Default for SilkEncoderControlFlp {
    fn default() -> Self {
        Self::new()
    }
}

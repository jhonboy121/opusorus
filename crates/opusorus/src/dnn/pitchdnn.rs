//! Port of `dnn/pitchdnn.c` / `dnn/pitchdnn.h` plus the model binding (`init_pitchdnn`) and
//! layer sizes of the generated `dnn/pitchdnn_data.c` / `pitchdnn_data.h` (libopus 1.6.1
//! model, checkpoint `pitch_vsmallconv1.pth`).
//!
//! Upstream compiles the model arrays in (`pitchdnn_init`); the port always loads them from a
//! weight blob ([`PitchDnnState::load_model`], upstream `pitchdnn_load_model`), per PLAN D-015.

use alloc::vec;
use alloc::vec::Vec;

use crate::Result;
use crate::math;

use super::nnet::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, Conv2dLayer, LinearLayer, compute_conv2d,
    compute_generic_dense, compute_generic_gru,
};
use super::parse_lpcnet_weights::{WeightArray, conv2d_init, linear_init, parse_weights};

/// `PITCH_MIN_PERIOD`.
pub const PITCH_MIN_PERIOD: usize = 32;
/// `PITCH_MAX_PERIOD`.
pub const PITCH_MAX_PERIOD: usize = 256;
/// `NB_XCORR_FEATURES`.
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

/// `DENSE_IF_UPSAMPLER_1_OUT_SIZE` (pitchdnn_data.h).
pub const DENSE_IF_UPSAMPLER_1_OUT_SIZE: usize = 64;
/// `DENSE_IF_UPSAMPLER_2_OUT_SIZE`.
pub const DENSE_IF_UPSAMPLER_2_OUT_SIZE: usize = 64;
/// `DENSE_DOWNSAMPLER_OUT_SIZE`.
pub const DENSE_DOWNSAMPLER_OUT_SIZE: usize = 64;
/// `DENSE_FINAL_UPSAMPLER_OUT_SIZE`.
pub const DENSE_FINAL_UPSAMPLER_OUT_SIZE: usize = 192;
/// `GRU_1_OUT_SIZE`.
pub const GRU_1_OUT_SIZE: usize = 64;
/// `GRU_1_STATE_SIZE`.
pub const GRU_1_STATE_SIZE: usize = 64;
/// `PITCH_DNN_MAX_RNN_UNITS`.
pub const PITCH_DNN_MAX_RNN_UNITS: usize = 64;

/// Size of `xcorr_mem1`.
pub const XCORR_MEM1_SIZE: usize = (NB_XCORR_FEATURES + 2) * 2;
/// Size of `xcorr_mem2` / `xcorr_mem3`.
pub const XCORR_MEM2_SIZE: usize = (NB_XCORR_FEATURES + 2) * 2 * 8;
/// Scratch needed by the two `compute_conv2d` calls (`ktime*in_channels*(height+2)` of
/// `conv2d_2`: 3*4*226); stands in for the C stack `in_buf[MAX_CONV2D_INPUTS]`.
pub const CONV2D_SCRATCH_SIZE: usize = 3 * 4 * (NB_XCORR_FEATURES + 2);

/// `PitchDNN` (pitchdnn_data.h): the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PitchDnn {
    pub dense_if_upsampler_1: LinearLayer,
    pub dense_if_upsampler_2: LinearLayer,
    pub dense_downsampler: LinearLayer,
    pub dense_final_upsampler: LinearLayer,
    pub conv2d_1: Conv2dLayer,
    pub conv2d_2: Conv2dLayer,
    pub gru_1_input: LinearLayer,
    pub gru_1_recurrent: LinearLayer,
}

/// Port of the generated dnn/pitchdnn_data.c:init_pitchdnn (fails with `BadArg` where C
/// returns 1).
pub fn init_pitchdnn(arrays: &[WeightArray<'_>]) -> Result<PitchDnn> {
    Ok(PitchDnn {
        dense_if_upsampler_1: linear_init(
            arrays,
            Some("dense_if_upsampler_1_bias"),
            Some("dense_if_upsampler_1_subias"),
            Some("dense_if_upsampler_1_weights_int8"),
            Some("dense_if_upsampler_1_weights_float"),
            None,
            None,
            Some("dense_if_upsampler_1_scale"),
            88,
            64,
        )?,
        dense_if_upsampler_2: linear_init(
            arrays,
            Some("dense_if_upsampler_2_bias"),
            Some("dense_if_upsampler_2_subias"),
            Some("dense_if_upsampler_2_weights_int8"),
            Some("dense_if_upsampler_2_weights_float"),
            None,
            None,
            Some("dense_if_upsampler_2_scale"),
            64,
            64,
        )?,
        dense_downsampler: linear_init(
            arrays,
            Some("dense_downsampler_bias"),
            Some("dense_downsampler_subias"),
            Some("dense_downsampler_weights_int8"),
            Some("dense_downsampler_weights_float"),
            None,
            None,
            Some("dense_downsampler_scale"),
            288,
            64,
        )?,
        dense_final_upsampler: linear_init(
            arrays,
            Some("dense_final_upsampler_bias"),
            Some("dense_final_upsampler_subias"),
            Some("dense_final_upsampler_weights_int8"),
            Some("dense_final_upsampler_weights_float"),
            None,
            None,
            Some("dense_final_upsampler_scale"),
            64,
            192,
        )?,
        conv2d_1: conv2d_init(
            arrays,
            Some("conv2d_1_bias"),
            Some("conv2d_1_weight_float"),
            1,
            4,
            3,
            3,
        )?,
        conv2d_2: conv2d_init(
            arrays,
            Some("conv2d_2_bias"),
            Some("conv2d_2_weight_float"),
            4,
            1,
            3,
            3,
        )?,
        gru_1_input: linear_init(
            arrays,
            Some("gru_1_input_bias"),
            Some("gru_1_input_subias"),
            Some("gru_1_input_weights_int8"),
            Some("gru_1_input_weights_float"),
            None,
            None,
            Some("gru_1_input_scale"),
            64,
            192,
        )?,
        gru_1_recurrent: linear_init(
            arrays,
            Some("gru_1_recurrent_bias"),
            Some("gru_1_recurrent_subias"),
            Some("gru_1_recurrent_weights_int8"),
            Some("gru_1_recurrent_weights_float"),
            None,
            None,
            Some("gru_1_recurrent_scale"),
            64,
            192,
        )?,
    })
}

/// `PitchDNNState`.
#[derive(Debug, Clone, PartialEq)]
pub struct PitchDnnState {
    pub model: PitchDnn,
    pub gru_state: [f32; GRU_1_STATE_SIZE],
    pub xcorr_mem1: [f32; XCORR_MEM1_SIZE],
    pub xcorr_mem2: Vec<f32>,
    /// Unused upstream (kept for layout fidelity).
    pub xcorr_mem3: Vec<f32>,
    /// Scratch for [`compute_conv2d`] (not part of the C state).
    conv_scratch: Vec<f32>,
}

impl Default for PitchDnnState {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchDnnState {
    /// Port of dnn/pitchdnn.c:pitchdnn_init: a cleared state. Upstream also binds the
    /// compiled-in model; here the model stays empty until [`Self::load_model`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: PitchDnn::default(),
            gru_state: [0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: [0.0; XCORR_MEM1_SIZE],
            xcorr_mem2: vec![0.0; XCORR_MEM2_SIZE],
            xcorr_mem3: vec![0.0; XCORR_MEM2_SIZE],
            conv_scratch: vec![0.0; CONV2D_SCRATCH_SIZE],
        }
    }

    /// Port of dnn/pitchdnn.c:pitchdnn_init (clears the state, keeps the loaded model).
    pub fn pitchdnn_init(&mut self) {
        self.gru_state.fill(0.0);
        self.xcorr_mem1.fill(0.0);
        self.xcorr_mem2.fill(0.0);
        self.xcorr_mem3.fill(0.0);
    }

    /// Port of dnn/pitchdnn.c:pitchdnn_load_model: parses a weight blob and binds the model.
    /// On failure (C: -1) the previous model is kept.
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let list = parse_weights(data)?;
        self.model = init_pitchdnn(&list)?;
        Ok(())
    }
}

/// Port of dnn/pitchdnn.c:compute_pitchdnn. `if_features` holds `PITCH_IF_FEATURES` (88) and
/// `xcorr_features` `NB_XCORR_FEATURES` values; returns the pitch in the model's log domain.
pub fn compute_pitchdnn(
    st: &mut PitchDnnState,
    if_features: &[f32],
    xcorr_features: &[f32],
) -> f32 {
    let mut if1_out = [0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
    let mut downsampler_in = [0f32; NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE];
    let mut downsampler_out = [0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
    let mut conv1_tmp1 = [0f32; (NB_XCORR_FEATURES + 2) * 8];
    let mut conv1_tmp2 = [0f32; (NB_XCORR_FEATURES + 2) * 8];
    let mut output = [0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
    let mut pos = 0usize;
    let mut maxval = -1f32;
    let mut sum = 0f32;
    let mut count = 0f32;
    let model = &st.model;
    // IF
    compute_generic_dense(
        &model.dense_if_upsampler_1,
        &mut if1_out,
        if_features,
        ACTIVATION_TANH,
    );
    compute_generic_dense(
        &model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES..],
        &if1_out,
        ACTIVATION_TANH,
    );
    // xcorr
    conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(&xcorr_features[..NB_XCORR_FEATURES]);
    compute_conv2d(
        &model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut st.xcorr_mem1,
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES + 2,
        ACTIVATION_TANH,
        &mut st.conv_scratch,
    );
    compute_conv2d(
        &model.conv2d_2,
        &mut downsampler_in,
        &mut st.xcorr_mem2,
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        ACTIVATION_TANH,
        &mut st.conv_scratch,
    );

    compute_generic_dense(
        &model.dense_downsampler,
        &mut downsampler_out,
        &downsampler_in,
        ACTIVATION_TANH,
    );
    compute_generic_gru(
        &model.gru_1_input,
        &model.gru_1_recurrent,
        &mut st.gru_state,
        &downsampler_out,
    );
    compute_generic_dense(
        &model.dense_final_upsampler,
        &mut output,
        &st.gru_state,
        ACTIVATION_LINEAR,
    );
    for (i, &o) in output[..180].iter().enumerate() {
        if o > maxval {
            pos = i;
            maxval = o;
        }
    }
    let lo = pos.saturating_sub(2);
    let hi = (pos + 2).min(179);
    for (i, &o) in output.iter().enumerate().take(hi + 1).skip(lo) {
        let p = math::exp(o as f64) as f32;
        sum += p * i as f32;
        count += p;
    }
    // C: (1.f/60.f)*(sum/count) - 1.5 (double subtraction).
    (((1.0f32 / 60.0f32) * (sum / count)) as f64 - 1.5) as f32
}

//! Layer sizes and model binding of the generated `dnn/lace_data.h` /
//! `dnn/lace_data.c` (libopus 1.6.1 model, checkpoint `lace_v2.pth`,
//! sha1 `41eaab33c6cbdb192d14f43c9f292856cab789e9`).
//!
//! Generated from the upstream files by a script; the weights themselves are loaded from a
//! weight blob (PLAN D-015).

use crate::Result;
use crate::dnn::nnet::LinearLayer;
use crate::dnn::parse_lpcnet_weights::{WeightArray, linear_init};

/// `LACE_PREEMPH`.
pub const LACE_PREEMPH: f32 = 0.85_f32;
/// `LACE_FRAME_SIZE`.
pub const LACE_FRAME_SIZE: usize = 80;
/// `LACE_OVERLAP_SIZE`.
pub const LACE_OVERLAP_SIZE: usize = 40;
/// `LACE_NUM_FEATURES`.
pub const LACE_NUM_FEATURES: usize = 93;
/// `LACE_PITCH_MAX`.
pub const LACE_PITCH_MAX: usize = 300;
/// `LACE_PITCH_EMBEDDING_DIM`.
pub const LACE_PITCH_EMBEDDING_DIM: usize = 64;
/// `LACE_NUMBITS_RANGE_LOW`.
pub const LACE_NUMBITS_RANGE_LOW: usize = 50;
/// `LACE_NUMBITS_RANGE_HIGH`.
pub const LACE_NUMBITS_RANGE_HIGH: usize = 650;
/// `LACE_NUMBITS_EMBEDDING_DIM`.
pub const LACE_NUMBITS_EMBEDDING_DIM: usize = 8;
/// `LACE_COND_DIM`.
pub const LACE_COND_DIM: usize = 128;
/// `LACE_HIDDEN_FEATURE_DIM`.
pub const LACE_HIDDEN_FEATURE_DIM: usize = 96;
/// `LACE_NUMBITS_SCALE_0`.
pub const LACE_NUMBITS_SCALE_0: f32 = 1.0983514785766602_f32;
/// `LACE_NUMBITS_SCALE_1`.
pub const LACE_NUMBITS_SCALE_1: f32 = 2.0509142875671387_f32;
/// `LACE_NUMBITS_SCALE_2`.
pub const LACE_NUMBITS_SCALE_2: f32 = 3.5729939937591553_f32;
/// `LACE_NUMBITS_SCALE_3`.
pub const LACE_NUMBITS_SCALE_3: f32 = 4.478035926818848_f32;
/// `LACE_NUMBITS_SCALE_4`.
pub const LACE_NUMBITS_SCALE_4: f32 = 5.926519393920898_f32;
/// `LACE_NUMBITS_SCALE_5`.
pub const LACE_NUMBITS_SCALE_5: f32 = 7.152282238006592_f32;
/// `LACE_NUMBITS_SCALE_6`.
pub const LACE_NUMBITS_SCALE_6: f32 = 8.277412414550781_f32;
/// `LACE_NUMBITS_SCALE_7`.
pub const LACE_NUMBITS_SCALE_7: f32 = 8.926830291748047_f32;
/// `LACE_PITCH_EMBEDDING_OUT_SIZE`.
pub const LACE_PITCH_EMBEDDING_OUT_SIZE: usize = 64;
/// `LACE_FNET_CONV1_OUT_SIZE`.
pub const LACE_FNET_CONV1_OUT_SIZE: usize = 96;
/// `LACE_FNET_CONV1_IN_SIZE`.
pub const LACE_FNET_CONV1_IN_SIZE: usize = 173;
/// `LACE_FNET_CONV1_STATE_SIZE` (`(173 * (0))`).
pub const LACE_FNET_CONV1_STATE_SIZE: usize = 0;
/// `LACE_FNET_CONV1_DELAY`.
pub const LACE_FNET_CONV1_DELAY: usize = 0;
/// `LACE_FNET_CONV2_OUT_SIZE`.
pub const LACE_FNET_CONV2_OUT_SIZE: usize = 128;
/// `LACE_FNET_CONV2_IN_SIZE`.
pub const LACE_FNET_CONV2_IN_SIZE: usize = 384;
/// `LACE_FNET_CONV2_STATE_SIZE` (`(384 * (1))`).
pub const LACE_FNET_CONV2_STATE_SIZE: usize = 384;
/// `LACE_FNET_CONV2_DELAY`.
pub const LACE_FNET_CONV2_DELAY: usize = 0;
/// `LACE_FNET_TCONV_KERNEL_SIZE`.
pub const LACE_FNET_TCONV_KERNEL_SIZE: usize = 4;
/// `LACE_FNET_TCONV_STRIDE`.
pub const LACE_FNET_TCONV_STRIDE: usize = 4;
/// `LACE_FNET_TCONV_IN_CHANNELS`.
pub const LACE_FNET_TCONV_IN_CHANNELS: usize = 128;
/// `LACE_FNET_TCONV_OUT_CHANNELS`.
pub const LACE_FNET_TCONV_OUT_CHANNELS: usize = 128;
/// `LACE_FNET_GRU_OUT_SIZE`.
pub const LACE_FNET_GRU_OUT_SIZE: usize = 128;
/// `LACE_FNET_GRU_STATE_SIZE`.
pub const LACE_FNET_GRU_STATE_SIZE: usize = 128;
/// `LACE_CF1_FILTER_GAIN_A`.
pub const LACE_CF1_FILTER_GAIN_A: f32 = 0.690776_f32;
/// `LACE_CF1_FILTER_GAIN_B`.
pub const LACE_CF1_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `LACE_CF1_LOG_GAIN_LIMIT`.
pub const LACE_CF1_LOG_GAIN_LIMIT: f32 = 1.151293_f32;
/// `LACE_CF1_KERNEL_SIZE`.
pub const LACE_CF1_KERNEL_SIZE: usize = 16;
/// `LACE_CF1_LEFT_PADDING`.
pub const LACE_CF1_LEFT_PADDING: usize = 8;
/// `LACE_CF1_FRAME_SIZE`.
pub const LACE_CF1_FRAME_SIZE: usize = 80;
/// `LACE_CF1_OVERLAP_SIZE`.
pub const LACE_CF1_OVERLAP_SIZE: usize = 40;
/// `LACE_CF1_IN_CHANNELS`.
pub const LACE_CF1_IN_CHANNELS: usize = 1;
/// `LACE_CF1_OUT_CHANNELS`.
pub const LACE_CF1_OUT_CHANNELS: usize = 1;
/// `LACE_CF1_NORM_P`.
pub const LACE_CF1_NORM_P: usize = 2;
/// `LACE_CF1_FEATURE_DIM`.
pub const LACE_CF1_FEATURE_DIM: usize = 128;
/// `LACE_CF1_MAX_LAG`.
pub const LACE_CF1_MAX_LAG: usize = 301;
/// `LACE_CF1_KERNEL_OUT_SIZE`.
pub const LACE_CF1_KERNEL_OUT_SIZE: usize = 16;
/// `LACE_CF1_GAIN_OUT_SIZE`.
pub const LACE_CF1_GAIN_OUT_SIZE: usize = 1;
/// `LACE_CF1_GLOBAL_GAIN_OUT_SIZE`.
pub const LACE_CF1_GLOBAL_GAIN_OUT_SIZE: usize = 1;
/// `LACE_CF2_FILTER_GAIN_A`.
pub const LACE_CF2_FILTER_GAIN_A: f32 = 0.690776_f32;
/// `LACE_CF2_FILTER_GAIN_B`.
pub const LACE_CF2_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `LACE_CF2_LOG_GAIN_LIMIT`.
pub const LACE_CF2_LOG_GAIN_LIMIT: f32 = 1.151293_f32;
/// `LACE_CF2_KERNEL_SIZE`.
pub const LACE_CF2_KERNEL_SIZE: usize = 16;
/// `LACE_CF2_LEFT_PADDING`.
pub const LACE_CF2_LEFT_PADDING: usize = 8;
/// `LACE_CF2_FRAME_SIZE`.
pub const LACE_CF2_FRAME_SIZE: usize = 80;
/// `LACE_CF2_OVERLAP_SIZE`.
pub const LACE_CF2_OVERLAP_SIZE: usize = 40;
/// `LACE_CF2_IN_CHANNELS`.
pub const LACE_CF2_IN_CHANNELS: usize = 1;
/// `LACE_CF2_OUT_CHANNELS`.
pub const LACE_CF2_OUT_CHANNELS: usize = 1;
/// `LACE_CF2_NORM_P`.
pub const LACE_CF2_NORM_P: usize = 2;
/// `LACE_CF2_FEATURE_DIM`.
pub const LACE_CF2_FEATURE_DIM: usize = 128;
/// `LACE_CF2_MAX_LAG`.
pub const LACE_CF2_MAX_LAG: usize = 301;
/// `LACE_CF2_KERNEL_OUT_SIZE`.
pub const LACE_CF2_KERNEL_OUT_SIZE: usize = 16;
/// `LACE_CF2_GAIN_OUT_SIZE`.
pub const LACE_CF2_GAIN_OUT_SIZE: usize = 1;
/// `LACE_CF2_GLOBAL_GAIN_OUT_SIZE`.
pub const LACE_CF2_GLOBAL_GAIN_OUT_SIZE: usize = 1;
/// `LACE_AF1_FILTER_GAIN_A`.
pub const LACE_AF1_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `LACE_AF1_FILTER_GAIN_B`.
pub const LACE_AF1_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `LACE_AF1_SHAPE_GAIN`.
pub const LACE_AF1_SHAPE_GAIN: f32 = 1.000000_f32;
/// `LACE_AF1_KERNEL_SIZE`.
pub const LACE_AF1_KERNEL_SIZE: usize = 16;
/// `LACE_AF1_FRAME_SIZE`.
pub const LACE_AF1_FRAME_SIZE: usize = 80;
/// `LACE_AF1_LEFT_PADDING`.
pub const LACE_AF1_LEFT_PADDING: usize = 15;
/// `LACE_AF1_OVERLAP_SIZE`.
pub const LACE_AF1_OVERLAP_SIZE: usize = 40;
/// `LACE_AF1_IN_CHANNELS`.
pub const LACE_AF1_IN_CHANNELS: usize = 1;
/// `LACE_AF1_OUT_CHANNELS`.
pub const LACE_AF1_OUT_CHANNELS: usize = 1;
/// `LACE_AF1_NORM_P`.
pub const LACE_AF1_NORM_P: usize = 2;
/// `LACE_AF1_FEATURE_DIM`.
pub const LACE_AF1_FEATURE_DIM: usize = 128;
/// `LACE_AF1_KERNEL_OUT_SIZE`.
pub const LACE_AF1_KERNEL_OUT_SIZE: usize = 16;
/// `LACE_AF1_GAIN_OUT_SIZE`.
pub const LACE_AF1_GAIN_OUT_SIZE: usize = 1;

/// `LACELayers`: the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaceLayers {
    pub lace_pitch_embedding: LinearLayer,
    pub lace_fnet_conv1: LinearLayer,
    pub lace_fnet_conv2: LinearLayer,
    pub lace_fnet_tconv: LinearLayer,
    pub lace_fnet_gru_input: LinearLayer,
    pub lace_fnet_gru_recurrent: LinearLayer,
    pub lace_cf1_kernel: LinearLayer,
    pub lace_cf1_gain: LinearLayer,
    pub lace_cf1_global_gain: LinearLayer,
    pub lace_cf2_kernel: LinearLayer,
    pub lace_cf2_gain: LinearLayer,
    pub lace_cf2_global_gain: LinearLayer,
    pub lace_af1_kernel: LinearLayer,
    pub lace_af1_gain: LinearLayer,
}

/// Port of the generated dnn/lace_data.c:init_lacelayers (fails with `BadArg` where C returns 1).
pub fn init_lacelayers(arrays: &[WeightArray<'_>]) -> Result<LaceLayers> {
    Ok(LaceLayers {
        lace_pitch_embedding: linear_init(
            arrays,
            Some("lace_pitch_embedding_bias"),
            None,
            None,
            Some("lace_pitch_embedding_weights_float"),
            None,
            None,
            None,
            301,
            64,
        )?,
        lace_fnet_conv1: linear_init(
            arrays,
            Some("lace_fnet_conv1_bias"),
            None,
            None,
            Some("lace_fnet_conv1_weights_float"),
            None,
            None,
            None,
            173,
            96,
        )?,
        lace_fnet_conv2: linear_init(
            arrays,
            Some("lace_fnet_conv2_bias"),
            Some("lace_fnet_conv2_subias"),
            Some("lace_fnet_conv2_weights_int8"),
            Some("lace_fnet_conv2_weights_float"),
            None,
            None,
            Some("lace_fnet_conv2_scale"),
            768,
            128,
        )?,
        lace_fnet_tconv: linear_init(
            arrays,
            Some("lace_fnet_tconv_bias"),
            Some("lace_fnet_tconv_subias"),
            Some("lace_fnet_tconv_weights_int8"),
            Some("lace_fnet_tconv_weights_float"),
            None,
            None,
            Some("lace_fnet_tconv_scale"),
            128,
            512,
        )?,
        lace_fnet_gru_input: linear_init(
            arrays,
            Some("lace_fnet_gru_input_bias"),
            Some("lace_fnet_gru_input_subias"),
            Some("lace_fnet_gru_input_weights_int8"),
            Some("lace_fnet_gru_input_weights_float"),
            None,
            None,
            Some("lace_fnet_gru_input_scale"),
            128,
            384,
        )?,
        lace_fnet_gru_recurrent: linear_init(
            arrays,
            Some("lace_fnet_gru_recurrent_bias"),
            Some("lace_fnet_gru_recurrent_subias"),
            Some("lace_fnet_gru_recurrent_weights_int8"),
            Some("lace_fnet_gru_recurrent_weights_float"),
            None,
            None,
            Some("lace_fnet_gru_recurrent_scale"),
            128,
            384,
        )?,
        lace_cf1_kernel: linear_init(
            arrays,
            Some("lace_cf1_kernel_bias"),
            Some("lace_cf1_kernel_subias"),
            Some("lace_cf1_kernel_weights_int8"),
            Some("lace_cf1_kernel_weights_float"),
            None,
            None,
            Some("lace_cf1_kernel_scale"),
            128,
            16,
        )?,
        lace_cf1_gain: linear_init(
            arrays,
            Some("lace_cf1_gain_bias"),
            None,
            None,
            Some("lace_cf1_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
        lace_cf1_global_gain: linear_init(
            arrays,
            Some("lace_cf1_global_gain_bias"),
            None,
            None,
            Some("lace_cf1_global_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
        lace_cf2_kernel: linear_init(
            arrays,
            Some("lace_cf2_kernel_bias"),
            Some("lace_cf2_kernel_subias"),
            Some("lace_cf2_kernel_weights_int8"),
            Some("lace_cf2_kernel_weights_float"),
            None,
            None,
            Some("lace_cf2_kernel_scale"),
            128,
            16,
        )?,
        lace_cf2_gain: linear_init(
            arrays,
            Some("lace_cf2_gain_bias"),
            None,
            None,
            Some("lace_cf2_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
        lace_cf2_global_gain: linear_init(
            arrays,
            Some("lace_cf2_global_gain_bias"),
            None,
            None,
            Some("lace_cf2_global_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
        lace_af1_kernel: linear_init(
            arrays,
            Some("lace_af1_kernel_bias"),
            Some("lace_af1_kernel_subias"),
            Some("lace_af1_kernel_weights_int8"),
            Some("lace_af1_kernel_weights_float"),
            None,
            None,
            Some("lace_af1_kernel_scale"),
            128,
            16,
        )?,
        lace_af1_gain: linear_init(
            arrays,
            Some("lace_af1_gain_bias"),
            None,
            None,
            Some("lace_af1_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
    })
}

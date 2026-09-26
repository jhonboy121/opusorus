//! Layer sizes and model binding of the generated `dnn/bbwenet_data.h` /
//! `dnn/bbwenet_data.c` (libopus 1.6.1 model, checkpoint `bbwenet_v2.pth`,
//! sha1 `d0cf1f15f0b755a519b90a9e2f3972f7e0a5ac30`).
//!
//! Generated from the upstream files by a script; the weights themselves are loaded from a
//! weight blob (PLAN D-015).

use crate::Result;
use crate::dnn::nnet::LinearLayer;
use crate::dnn::parse_lpcnet_weights::{WeightArray, linear_init};

/// `BBWENET_FEATURE_DIM`.
pub const BBWENET_FEATURE_DIM: usize = 114;
/// `BBWENET_FRAME_SIZE16`.
pub const BBWENET_FRAME_SIZE16: usize = 80;
/// `BBWENET_COND_DIM`.
pub const BBWENET_COND_DIM: usize = 128;
/// `BBWENET_FNET_CONV1_OUT_SIZE`.
pub const BBWENET_FNET_CONV1_OUT_SIZE: usize = 128;
/// `BBWENET_FNET_CONV1_IN_SIZE`.
pub const BBWENET_FNET_CONV1_IN_SIZE: usize = 114;
/// `BBWENET_FNET_CONV1_STATE_SIZE` (`(114 * (2))`).
pub const BBWENET_FNET_CONV1_STATE_SIZE: usize = 228;
/// `BBWENET_FNET_CONV2_OUT_SIZE`.
pub const BBWENET_FNET_CONV2_OUT_SIZE: usize = 128;
/// `BBWENET_FNET_CONV2_IN_SIZE`.
pub const BBWENET_FNET_CONV2_IN_SIZE: usize = 128;
/// `BBWENET_FNET_CONV2_STATE_SIZE` (`(128 * (2))`).
pub const BBWENET_FNET_CONV2_STATE_SIZE: usize = 256;
/// `BBWENET_FNET_GRU_OUT_SIZE`.
pub const BBWENET_FNET_GRU_OUT_SIZE: usize = 128;
/// `BBWENET_FNET_GRU_STATE_SIZE`.
pub const BBWENET_FNET_GRU_STATE_SIZE: usize = 128;
/// `BBWENET_FNET_TCONV_KERNEL_SIZE`.
pub const BBWENET_FNET_TCONV_KERNEL_SIZE: usize = 2;
/// `BBWENET_FNET_TCONV_STRIDE`.
pub const BBWENET_FNET_TCONV_STRIDE: usize = 2;
/// `BBWENET_FNET_TCONV_IN_CHANNELS`.
pub const BBWENET_FNET_TCONV_IN_CHANNELS: usize = 128;
/// `BBWENET_FNET_TCONV_OUT_CHANNELS`.
pub const BBWENET_FNET_TCONV_OUT_CHANNELS: usize = 128;
/// `BBWENET_TDSHAPE1_FEATURE_DIM`.
pub const BBWENET_TDSHAPE1_FEATURE_DIM: usize = 128;
/// `BBWENET_TDSHAPE1_FRAME_SIZE`.
pub const BBWENET_TDSHAPE1_FRAME_SIZE: usize = 160;
/// `BBWENET_TDSHAPE1_AVG_POOL_K`.
pub const BBWENET_TDSHAPE1_AVG_POOL_K: usize = 8;
/// `BBWENET_TDSHAPE1_INNOVATE`.
pub const BBWENET_TDSHAPE1_INNOVATE: usize = 0;
/// `BBWENET_TDSHAPE1_POOL_AFTER`.
pub const BBWENET_TDSHAPE1_POOL_AFTER: usize = 0;
/// `BBWENET_TDSHAPE1_INTERPOLATE_K`.
pub const BBWENET_TDSHAPE1_INTERPOLATE_K: usize = 2;
/// `BBWENET_TDSHAPE1_ALPHA1_F_OUT_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA1_F_OUT_SIZE: usize = 80;
/// `BBWENET_TDSHAPE1_ALPHA1_F_IN_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA1_F_IN_SIZE: usize = 128;
/// `BBWENET_TDSHAPE1_ALPHA1_F_STATE_SIZE` (`(128 * (1))`).
pub const BBWENET_TDSHAPE1_ALPHA1_F_STATE_SIZE: usize = 128;
/// `BBWENET_TDSHAPE1_ALPHA1_T_OUT_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA1_T_OUT_SIZE: usize = 80;
/// `BBWENET_TDSHAPE1_ALPHA1_T_IN_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA1_T_IN_SIZE: usize = 21;
/// `BBWENET_TDSHAPE1_ALPHA1_T_STATE_SIZE` (`(21 * (1))`).
pub const BBWENET_TDSHAPE1_ALPHA1_T_STATE_SIZE: usize = 21;
/// `BBWENET_TDSHAPE1_ALPHA2_OUT_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA2_OUT_SIZE: usize = 80;
/// `BBWENET_TDSHAPE1_ALPHA2_IN_SIZE`.
pub const BBWENET_TDSHAPE1_ALPHA2_IN_SIZE: usize = 80;
/// `BBWENET_TDSHAPE1_ALPHA2_STATE_SIZE` (`(80 * (1))`).
pub const BBWENET_TDSHAPE1_ALPHA2_STATE_SIZE: usize = 80;
/// `BBWENET_TDSHAPE2_FEATURE_DIM`.
pub const BBWENET_TDSHAPE2_FEATURE_DIM: usize = 128;
/// `BBWENET_TDSHAPE2_FRAME_SIZE`.
pub const BBWENET_TDSHAPE2_FRAME_SIZE: usize = 240;
/// `BBWENET_TDSHAPE2_AVG_POOL_K`.
pub const BBWENET_TDSHAPE2_AVG_POOL_K: usize = 12;
/// `BBWENET_TDSHAPE2_INNOVATE`.
pub const BBWENET_TDSHAPE2_INNOVATE: usize = 0;
/// `BBWENET_TDSHAPE2_POOL_AFTER`.
pub const BBWENET_TDSHAPE2_POOL_AFTER: usize = 0;
/// `BBWENET_TDSHAPE2_INTERPOLATE_K`.
pub const BBWENET_TDSHAPE2_INTERPOLATE_K: usize = 2;
/// `BBWENET_TDSHAPE2_ALPHA1_F_OUT_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA1_F_OUT_SIZE: usize = 120;
/// `BBWENET_TDSHAPE2_ALPHA1_F_IN_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA1_F_IN_SIZE: usize = 128;
/// `BBWENET_TDSHAPE2_ALPHA1_F_STATE_SIZE` (`(128 * (1))`).
pub const BBWENET_TDSHAPE2_ALPHA1_F_STATE_SIZE: usize = 128;
/// `BBWENET_TDSHAPE2_ALPHA1_T_OUT_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA1_T_OUT_SIZE: usize = 120;
/// `BBWENET_TDSHAPE2_ALPHA1_T_IN_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA1_T_IN_SIZE: usize = 21;
/// `BBWENET_TDSHAPE2_ALPHA1_T_STATE_SIZE` (`(21 * (1))`).
pub const BBWENET_TDSHAPE2_ALPHA1_T_STATE_SIZE: usize = 21;
/// `BBWENET_TDSHAPE2_ALPHA2_OUT_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA2_OUT_SIZE: usize = 120;
/// `BBWENET_TDSHAPE2_ALPHA2_IN_SIZE`.
pub const BBWENET_TDSHAPE2_ALPHA2_IN_SIZE: usize = 120;
/// `BBWENET_TDSHAPE2_ALPHA2_STATE_SIZE` (`(120 * (1))`).
pub const BBWENET_TDSHAPE2_ALPHA2_STATE_SIZE: usize = 120;
/// `BBWENET_AF1_FILTER_GAIN_A`.
pub const BBWENET_AF1_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `BBWENET_AF1_FILTER_GAIN_B`.
pub const BBWENET_AF1_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `BBWENET_AF1_SHAPE_GAIN`.
pub const BBWENET_AF1_SHAPE_GAIN: f32 = 1.000000_f32;
/// `BBWENET_AF1_KERNEL_SIZE`.
pub const BBWENET_AF1_KERNEL_SIZE: usize = 16;
/// `BBWENET_AF1_FRAME_SIZE`.
pub const BBWENET_AF1_FRAME_SIZE: usize = 80;
/// `BBWENET_AF1_LEFT_PADDING`.
pub const BBWENET_AF1_LEFT_PADDING: usize = 15;
/// `BBWENET_AF1_OVERLAP_SIZE`.
pub const BBWENET_AF1_OVERLAP_SIZE: usize = 40;
/// `BBWENET_AF1_IN_CHANNELS`.
pub const BBWENET_AF1_IN_CHANNELS: usize = 1;
/// `BBWENET_AF1_OUT_CHANNELS`.
pub const BBWENET_AF1_OUT_CHANNELS: usize = 3;
/// `BBWENET_AF1_NORM_P`.
pub const BBWENET_AF1_NORM_P: usize = 2;
/// `BBWENET_AF1_FEATURE_DIM`.
pub const BBWENET_AF1_FEATURE_DIM: usize = 128;
/// `BBWENET_AF1_KERNEL_OUT_SIZE`.
pub const BBWENET_AF1_KERNEL_OUT_SIZE: usize = 48;
/// `BBWENET_AF1_GAIN_OUT_SIZE`.
pub const BBWENET_AF1_GAIN_OUT_SIZE: usize = 3;
/// `BBWENET_AF2_FILTER_GAIN_A`.
pub const BBWENET_AF2_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `BBWENET_AF2_FILTER_GAIN_B`.
pub const BBWENET_AF2_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `BBWENET_AF2_SHAPE_GAIN`.
pub const BBWENET_AF2_SHAPE_GAIN: f32 = 1.000000_f32;
/// `BBWENET_AF2_KERNEL_SIZE`.
pub const BBWENET_AF2_KERNEL_SIZE: usize = 32;
/// `BBWENET_AF2_FRAME_SIZE`.
pub const BBWENET_AF2_FRAME_SIZE: usize = 160;
/// `BBWENET_AF2_LEFT_PADDING`.
pub const BBWENET_AF2_LEFT_PADDING: usize = 31;
/// `BBWENET_AF2_OVERLAP_SIZE`.
pub const BBWENET_AF2_OVERLAP_SIZE: usize = 80;
/// `BBWENET_AF2_IN_CHANNELS`.
pub const BBWENET_AF2_IN_CHANNELS: usize = 3;
/// `BBWENET_AF2_OUT_CHANNELS`.
pub const BBWENET_AF2_OUT_CHANNELS: usize = 3;
/// `BBWENET_AF2_NORM_P`.
pub const BBWENET_AF2_NORM_P: usize = 2;
/// `BBWENET_AF2_FEATURE_DIM`.
pub const BBWENET_AF2_FEATURE_DIM: usize = 128;
/// `BBWENET_AF2_KERNEL_OUT_SIZE`.
pub const BBWENET_AF2_KERNEL_OUT_SIZE: usize = 288;
/// `BBWENET_AF2_GAIN_OUT_SIZE`.
pub const BBWENET_AF2_GAIN_OUT_SIZE: usize = 3;
/// `BBWENET_AF3_FILTER_GAIN_A`.
pub const BBWENET_AF3_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `BBWENET_AF3_FILTER_GAIN_B`.
pub const BBWENET_AF3_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `BBWENET_AF3_SHAPE_GAIN`.
pub const BBWENET_AF3_SHAPE_GAIN: f32 = 1.000000_f32;
/// `BBWENET_AF3_KERNEL_SIZE`.
pub const BBWENET_AF3_KERNEL_SIZE: usize = 16;
/// `BBWENET_AF3_FRAME_SIZE`.
pub const BBWENET_AF3_FRAME_SIZE: usize = 240;
/// `BBWENET_AF3_LEFT_PADDING`.
pub const BBWENET_AF3_LEFT_PADDING: usize = 15;
/// `BBWENET_AF3_OVERLAP_SIZE`.
pub const BBWENET_AF3_OVERLAP_SIZE: usize = 120;
/// `BBWENET_AF3_IN_CHANNELS`.
pub const BBWENET_AF3_IN_CHANNELS: usize = 3;
/// `BBWENET_AF3_OUT_CHANNELS`.
pub const BBWENET_AF3_OUT_CHANNELS: usize = 1;
/// `BBWENET_AF3_NORM_P`.
pub const BBWENET_AF3_NORM_P: usize = 2;
/// `BBWENET_AF3_FEATURE_DIM`.
pub const BBWENET_AF3_FEATURE_DIM: usize = 128;
/// `BBWENET_AF3_KERNEL_OUT_SIZE`.
pub const BBWENET_AF3_KERNEL_OUT_SIZE: usize = 48;
/// `BBWENET_AF3_GAIN_OUT_SIZE`.
pub const BBWENET_AF3_GAIN_OUT_SIZE: usize = 1;

/// `BBWENETLayers`: the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BbwenetLayers {
    pub bbwenet_fnet_conv1: LinearLayer,
    pub bbwenet_fnet_conv2: LinearLayer,
    pub bbwenet_fnet_gru_input: LinearLayer,
    pub bbwenet_fnet_gru_recurrent: LinearLayer,
    pub bbwenet_fnet_tconv: LinearLayer,
    pub bbwenet_tdshape1_alpha1_f: LinearLayer,
    pub bbwenet_tdshape1_alpha1_t: LinearLayer,
    pub bbwenet_tdshape1_alpha2: LinearLayer,
    pub bbwenet_tdshape2_alpha1_f: LinearLayer,
    pub bbwenet_tdshape2_alpha1_t: LinearLayer,
    pub bbwenet_tdshape2_alpha2: LinearLayer,
    pub bbwenet_af1_kernel: LinearLayer,
    pub bbwenet_af1_gain: LinearLayer,
    pub bbwenet_af2_kernel: LinearLayer,
    pub bbwenet_af2_gain: LinearLayer,
    pub bbwenet_af3_kernel: LinearLayer,
    pub bbwenet_af3_gain: LinearLayer,
}

/// Port of the generated dnn/bbwenet_data.c:init_bbwenetlayers (fails with `BadArg` where C returns 1).
pub fn init_bbwenetlayers(arrays: &[WeightArray<'_>]) -> Result<BbwenetLayers> {
    Ok(BbwenetLayers {
        bbwenet_fnet_conv1: linear_init(
            arrays,
            Some("bbwenet_fnet_conv1_bias"),
            None,
            None,
            Some("bbwenet_fnet_conv1_weights_float"),
            None,
            None,
            None,
            342,
            128,
        )?,
        bbwenet_fnet_conv2: linear_init(
            arrays,
            Some("bbwenet_fnet_conv2_bias"),
            Some("bbwenet_fnet_conv2_subias"),
            Some("bbwenet_fnet_conv2_weights_int8"),
            Some("bbwenet_fnet_conv2_weights_float"),
            None,
            None,
            Some("bbwenet_fnet_conv2_scale"),
            384,
            128,
        )?,
        bbwenet_fnet_gru_input: linear_init(
            arrays,
            Some("bbwenet_fnet_gru_input_bias"),
            Some("bbwenet_fnet_gru_input_subias"),
            Some("bbwenet_fnet_gru_input_weights_int8"),
            Some("bbwenet_fnet_gru_input_weights_float"),
            None,
            None,
            Some("bbwenet_fnet_gru_input_scale"),
            128,
            384,
        )?,
        bbwenet_fnet_gru_recurrent: linear_init(
            arrays,
            Some("bbwenet_fnet_gru_recurrent_bias"),
            Some("bbwenet_fnet_gru_recurrent_subias"),
            Some("bbwenet_fnet_gru_recurrent_weights_int8"),
            Some("bbwenet_fnet_gru_recurrent_weights_float"),
            None,
            None,
            Some("bbwenet_fnet_gru_recurrent_scale"),
            128,
            384,
        )?,
        bbwenet_fnet_tconv: linear_init(
            arrays,
            Some("bbwenet_fnet_tconv_bias"),
            Some("bbwenet_fnet_tconv_subias"),
            Some("bbwenet_fnet_tconv_weights_int8"),
            Some("bbwenet_fnet_tconv_weights_float"),
            None,
            None,
            Some("bbwenet_fnet_tconv_scale"),
            128,
            256,
        )?,
        bbwenet_tdshape1_alpha1_f: linear_init(
            arrays,
            Some("bbwenet_tdshape1_alpha1_f_bias"),
            Some("bbwenet_tdshape1_alpha1_f_subias"),
            Some("bbwenet_tdshape1_alpha1_f_weights_int8"),
            Some("bbwenet_tdshape1_alpha1_f_weights_float"),
            None,
            None,
            Some("bbwenet_tdshape1_alpha1_f_scale"),
            256,
            80,
        )?,
        bbwenet_tdshape1_alpha1_t: linear_init(
            arrays,
            Some("bbwenet_tdshape1_alpha1_t_bias"),
            None,
            None,
            Some("bbwenet_tdshape1_alpha1_t_weights_float"),
            None,
            None,
            None,
            42,
            80,
        )?,
        bbwenet_tdshape1_alpha2: linear_init(
            arrays,
            Some("bbwenet_tdshape1_alpha2_bias"),
            None,
            None,
            Some("bbwenet_tdshape1_alpha2_weights_float"),
            None,
            None,
            None,
            160,
            80,
        )?,
        bbwenet_tdshape2_alpha1_f: linear_init(
            arrays,
            Some("bbwenet_tdshape2_alpha1_f_bias"),
            Some("bbwenet_tdshape2_alpha1_f_subias"),
            Some("bbwenet_tdshape2_alpha1_f_weights_int8"),
            Some("bbwenet_tdshape2_alpha1_f_weights_float"),
            None,
            None,
            Some("bbwenet_tdshape2_alpha1_f_scale"),
            256,
            120,
        )?,
        bbwenet_tdshape2_alpha1_t: linear_init(
            arrays,
            Some("bbwenet_tdshape2_alpha1_t_bias"),
            None,
            None,
            Some("bbwenet_tdshape2_alpha1_t_weights_float"),
            None,
            None,
            None,
            42,
            120,
        )?,
        bbwenet_tdshape2_alpha2: linear_init(
            arrays,
            Some("bbwenet_tdshape2_alpha2_bias"),
            None,
            None,
            Some("bbwenet_tdshape2_alpha2_weights_float"),
            None,
            None,
            None,
            240,
            120,
        )?,
        bbwenet_af1_kernel: linear_init(
            arrays,
            Some("bbwenet_af1_kernel_bias"),
            Some("bbwenet_af1_kernel_subias"),
            Some("bbwenet_af1_kernel_weights_int8"),
            Some("bbwenet_af1_kernel_weights_float"),
            None,
            None,
            Some("bbwenet_af1_kernel_scale"),
            128,
            48,
        )?,
        bbwenet_af1_gain: linear_init(
            arrays,
            Some("bbwenet_af1_gain_bias"),
            None,
            None,
            Some("bbwenet_af1_gain_weights_float"),
            None,
            None,
            None,
            128,
            3,
        )?,
        bbwenet_af2_kernel: linear_init(
            arrays,
            Some("bbwenet_af2_kernel_bias"),
            Some("bbwenet_af2_kernel_subias"),
            Some("bbwenet_af2_kernel_weights_int8"),
            Some("bbwenet_af2_kernel_weights_float"),
            None,
            None,
            Some("bbwenet_af2_kernel_scale"),
            128,
            288,
        )?,
        bbwenet_af2_gain: linear_init(
            arrays,
            Some("bbwenet_af2_gain_bias"),
            None,
            None,
            Some("bbwenet_af2_gain_weights_float"),
            None,
            None,
            None,
            128,
            3,
        )?,
        bbwenet_af3_kernel: linear_init(
            arrays,
            Some("bbwenet_af3_kernel_bias"),
            Some("bbwenet_af3_kernel_subias"),
            Some("bbwenet_af3_kernel_weights_int8"),
            Some("bbwenet_af3_kernel_weights_float"),
            None,
            None,
            Some("bbwenet_af3_kernel_scale"),
            128,
            48,
        )?,
        bbwenet_af3_gain: linear_init(
            arrays,
            Some("bbwenet_af3_gain_bias"),
            None,
            None,
            Some("bbwenet_af3_gain_weights_float"),
            None,
            None,
            None,
            128,
            1,
        )?,
    })
}

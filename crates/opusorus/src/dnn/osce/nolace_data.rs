//! Layer sizes and model binding of the generated `dnn/nolace_data.h` /
//! `dnn/nolace_data.c` (libopus 1.6.1 model, checkpoint `nolace_small.pth`,
//! sha1 `953bf5854e1a33e8892da48a29b19aff3a272902`).
//!
//! Generated from the upstream files by a script; the weights themselves are loaded from a
//! weight blob (PLAN D-015).

use crate::Result;
use crate::dnn::nnet::LinearLayer;
use crate::dnn::parse_lpcnet_weights::{WeightArray, linear_init};

/// `NOLACE_PREEMPH`.
pub const NOLACE_PREEMPH: f32 = 0.85_f32;
/// `NOLACE_FRAME_SIZE`.
pub const NOLACE_FRAME_SIZE: usize = 80;
/// `NOLACE_OVERLAP_SIZE`.
pub const NOLACE_OVERLAP_SIZE: usize = 40;
/// `NOLACE_NUM_FEATURES`.
pub const NOLACE_NUM_FEATURES: usize = 93;
/// `NOLACE_PITCH_MAX`.
pub const NOLACE_PITCH_MAX: usize = 300;
/// `NOLACE_PITCH_EMBEDDING_DIM`.
pub const NOLACE_PITCH_EMBEDDING_DIM: usize = 64;
/// `NOLACE_NUMBITS_RANGE_LOW`.
pub const NOLACE_NUMBITS_RANGE_LOW: usize = 50;
/// `NOLACE_NUMBITS_RANGE_HIGH`.
pub const NOLACE_NUMBITS_RANGE_HIGH: usize = 650;
/// `NOLACE_NUMBITS_EMBEDDING_DIM`.
pub const NOLACE_NUMBITS_EMBEDDING_DIM: usize = 8;
/// `NOLACE_COND_DIM`.
pub const NOLACE_COND_DIM: usize = 160;
/// `NOLACE_HIDDEN_FEATURE_DIM`.
pub const NOLACE_HIDDEN_FEATURE_DIM: usize = 96;
/// `NOLACE_NUMBITS_SCALE_0`.
pub const NOLACE_NUMBITS_SCALE_0: f32 = 1.0357311964035034_f32;
/// `NOLACE_NUMBITS_SCALE_1`.
pub const NOLACE_NUMBITS_SCALE_1: f32 = 1.735559105873108_f32;
/// `NOLACE_NUMBITS_SCALE_2`.
pub const NOLACE_NUMBITS_SCALE_2: f32 = 3.6004557609558105_f32;
/// `NOLACE_NUMBITS_SCALE_3`.
pub const NOLACE_NUMBITS_SCALE_3: f32 = 4.552478313446045_f32;
/// `NOLACE_NUMBITS_SCALE_4`.
pub const NOLACE_NUMBITS_SCALE_4: f32 = 5.932559490203857_f32;
/// `NOLACE_NUMBITS_SCALE_5`.
pub const NOLACE_NUMBITS_SCALE_5: f32 = 7.176970481872559_f32;
/// `NOLACE_NUMBITS_SCALE_6`.
pub const NOLACE_NUMBITS_SCALE_6: f32 = 8.114998817443848_f32;
/// `NOLACE_NUMBITS_SCALE_7`.
pub const NOLACE_NUMBITS_SCALE_7: f32 = 8.77063274383545_f32;
/// `NOLACE_PITCH_EMBEDDING_OUT_SIZE`.
pub const NOLACE_PITCH_EMBEDDING_OUT_SIZE: usize = 64;
/// `NOLACE_FNET_CONV1_OUT_SIZE`.
pub const NOLACE_FNET_CONV1_OUT_SIZE: usize = 96;
/// `NOLACE_FNET_CONV1_IN_SIZE`.
pub const NOLACE_FNET_CONV1_IN_SIZE: usize = 173;
/// `NOLACE_FNET_CONV1_STATE_SIZE` (`(173 * (0))`).
pub const NOLACE_FNET_CONV1_STATE_SIZE: usize = 0;
/// `NOLACE_FNET_CONV1_DELAY`.
pub const NOLACE_FNET_CONV1_DELAY: usize = 0;
/// `NOLACE_FNET_CONV2_OUT_SIZE`.
pub const NOLACE_FNET_CONV2_OUT_SIZE: usize = 160;
/// `NOLACE_FNET_CONV2_IN_SIZE`.
pub const NOLACE_FNET_CONV2_IN_SIZE: usize = 384;
/// `NOLACE_FNET_CONV2_STATE_SIZE` (`(384 * (1))`).
pub const NOLACE_FNET_CONV2_STATE_SIZE: usize = 384;
/// `NOLACE_FNET_CONV2_DELAY`.
pub const NOLACE_FNET_CONV2_DELAY: usize = 0;
/// `NOLACE_FNET_TCONV_KERNEL_SIZE`.
pub const NOLACE_FNET_TCONV_KERNEL_SIZE: usize = 4;
/// `NOLACE_FNET_TCONV_STRIDE`.
pub const NOLACE_FNET_TCONV_STRIDE: usize = 4;
/// `NOLACE_FNET_TCONV_IN_CHANNELS`.
pub const NOLACE_FNET_TCONV_IN_CHANNELS: usize = 160;
/// `NOLACE_FNET_TCONV_OUT_CHANNELS`.
pub const NOLACE_FNET_TCONV_OUT_CHANNELS: usize = 160;
/// `NOLACE_FNET_GRU_OUT_SIZE`.
pub const NOLACE_FNET_GRU_OUT_SIZE: usize = 160;
/// `NOLACE_FNET_GRU_STATE_SIZE`.
pub const NOLACE_FNET_GRU_STATE_SIZE: usize = 160;
/// `NOLACE_CF1_FILTER_GAIN_A`.
pub const NOLACE_CF1_FILTER_GAIN_A: f32 = 0.690776_f32;
/// `NOLACE_CF1_FILTER_GAIN_B`.
pub const NOLACE_CF1_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_CF1_LOG_GAIN_LIMIT`.
pub const NOLACE_CF1_LOG_GAIN_LIMIT: f32 = 1.151293_f32;
/// `NOLACE_CF1_KERNEL_SIZE`.
pub const NOLACE_CF1_KERNEL_SIZE: usize = 16;
/// `NOLACE_CF1_LEFT_PADDING`.
pub const NOLACE_CF1_LEFT_PADDING: usize = 8;
/// `NOLACE_CF1_FRAME_SIZE`.
pub const NOLACE_CF1_FRAME_SIZE: usize = 80;
/// `NOLACE_CF1_OVERLAP_SIZE`.
pub const NOLACE_CF1_OVERLAP_SIZE: usize = 40;
/// `NOLACE_CF1_IN_CHANNELS`.
pub const NOLACE_CF1_IN_CHANNELS: usize = 1;
/// `NOLACE_CF1_OUT_CHANNELS`.
pub const NOLACE_CF1_OUT_CHANNELS: usize = 1;
/// `NOLACE_CF1_NORM_P`.
pub const NOLACE_CF1_NORM_P: usize = 2;
/// `NOLACE_CF1_FEATURE_DIM`.
pub const NOLACE_CF1_FEATURE_DIM: usize = 160;
/// `NOLACE_CF1_MAX_LAG`.
pub const NOLACE_CF1_MAX_LAG: usize = 301;
/// `NOLACE_CF1_KERNEL_OUT_SIZE`.
pub const NOLACE_CF1_KERNEL_OUT_SIZE: usize = 16;
/// `NOLACE_CF1_GAIN_OUT_SIZE`.
pub const NOLACE_CF1_GAIN_OUT_SIZE: usize = 1;
/// `NOLACE_CF1_GLOBAL_GAIN_OUT_SIZE`.
pub const NOLACE_CF1_GLOBAL_GAIN_OUT_SIZE: usize = 1;
/// `NOLACE_CF2_FILTER_GAIN_A`.
pub const NOLACE_CF2_FILTER_GAIN_A: f32 = 0.690776_f32;
/// `NOLACE_CF2_FILTER_GAIN_B`.
pub const NOLACE_CF2_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_CF2_LOG_GAIN_LIMIT`.
pub const NOLACE_CF2_LOG_GAIN_LIMIT: f32 = 1.151293_f32;
/// `NOLACE_CF2_KERNEL_SIZE`.
pub const NOLACE_CF2_KERNEL_SIZE: usize = 16;
/// `NOLACE_CF2_LEFT_PADDING`.
pub const NOLACE_CF2_LEFT_PADDING: usize = 8;
/// `NOLACE_CF2_FRAME_SIZE`.
pub const NOLACE_CF2_FRAME_SIZE: usize = 80;
/// `NOLACE_CF2_OVERLAP_SIZE`.
pub const NOLACE_CF2_OVERLAP_SIZE: usize = 40;
/// `NOLACE_CF2_IN_CHANNELS`.
pub const NOLACE_CF2_IN_CHANNELS: usize = 1;
/// `NOLACE_CF2_OUT_CHANNELS`.
pub const NOLACE_CF2_OUT_CHANNELS: usize = 1;
/// `NOLACE_CF2_NORM_P`.
pub const NOLACE_CF2_NORM_P: usize = 2;
/// `NOLACE_CF2_FEATURE_DIM`.
pub const NOLACE_CF2_FEATURE_DIM: usize = 160;
/// `NOLACE_CF2_MAX_LAG`.
pub const NOLACE_CF2_MAX_LAG: usize = 301;
/// `NOLACE_CF2_KERNEL_OUT_SIZE`.
pub const NOLACE_CF2_KERNEL_OUT_SIZE: usize = 16;
/// `NOLACE_CF2_GAIN_OUT_SIZE`.
pub const NOLACE_CF2_GAIN_OUT_SIZE: usize = 1;
/// `NOLACE_CF2_GLOBAL_GAIN_OUT_SIZE`.
pub const NOLACE_CF2_GLOBAL_GAIN_OUT_SIZE: usize = 1;
/// `NOLACE_AF1_FILTER_GAIN_A`.
pub const NOLACE_AF1_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `NOLACE_AF1_FILTER_GAIN_B`.
pub const NOLACE_AF1_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_AF1_SHAPE_GAIN`.
pub const NOLACE_AF1_SHAPE_GAIN: f32 = 1.000000_f32;
/// `NOLACE_AF1_KERNEL_SIZE`.
pub const NOLACE_AF1_KERNEL_SIZE: usize = 16;
/// `NOLACE_AF1_FRAME_SIZE`.
pub const NOLACE_AF1_FRAME_SIZE: usize = 80;
/// `NOLACE_AF1_LEFT_PADDING`.
pub const NOLACE_AF1_LEFT_PADDING: usize = 15;
/// `NOLACE_AF1_OVERLAP_SIZE`.
pub const NOLACE_AF1_OVERLAP_SIZE: usize = 40;
/// `NOLACE_AF1_IN_CHANNELS`.
pub const NOLACE_AF1_IN_CHANNELS: usize = 1;
/// `NOLACE_AF1_OUT_CHANNELS`.
pub const NOLACE_AF1_OUT_CHANNELS: usize = 2;
/// `NOLACE_AF1_NORM_P`.
pub const NOLACE_AF1_NORM_P: usize = 2;
/// `NOLACE_AF1_FEATURE_DIM`.
pub const NOLACE_AF1_FEATURE_DIM: usize = 160;
/// `NOLACE_AF1_KERNEL_OUT_SIZE`.
pub const NOLACE_AF1_KERNEL_OUT_SIZE: usize = 32;
/// `NOLACE_AF1_GAIN_OUT_SIZE`.
pub const NOLACE_AF1_GAIN_OUT_SIZE: usize = 2;
/// `NOLACE_TDSHAPE1_FEATURE_DIM`.
pub const NOLACE_TDSHAPE1_FEATURE_DIM: usize = 160;
/// `NOLACE_TDSHAPE1_FRAME_SIZE`.
pub const NOLACE_TDSHAPE1_FRAME_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_AVG_POOL_K`.
pub const NOLACE_TDSHAPE1_AVG_POOL_K: usize = 4;
/// `NOLACE_TDSHAPE1_INNOVATE`.
pub const NOLACE_TDSHAPE1_INNOVATE: usize = 0;
/// `NOLACE_TDSHAPE1_POOL_AFTER`.
pub const NOLACE_TDSHAPE1_POOL_AFTER: usize = 0;
/// `NOLACE_TDSHAPE1_ALPHA1_F_OUT_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA1_F_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_ALPHA1_F_IN_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA1_F_IN_SIZE: usize = 160;
/// `NOLACE_TDSHAPE1_ALPHA1_F_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_TDSHAPE1_ALPHA1_F_STATE_SIZE: usize = 160;
/// `NOLACE_TDSHAPE1_ALPHA1_F_DELAY`.
pub const NOLACE_TDSHAPE1_ALPHA1_F_DELAY: usize = 0;
/// `NOLACE_TDSHAPE1_ALPHA1_T_OUT_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA1_T_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_ALPHA1_T_IN_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA1_T_IN_SIZE: usize = 21;
/// `NOLACE_TDSHAPE1_ALPHA1_T_STATE_SIZE` (`(21 * (1))`).
pub const NOLACE_TDSHAPE1_ALPHA1_T_STATE_SIZE: usize = 21;
/// `NOLACE_TDSHAPE1_ALPHA1_T_DELAY`.
pub const NOLACE_TDSHAPE1_ALPHA1_T_DELAY: usize = 0;
/// `NOLACE_TDSHAPE1_ALPHA2_OUT_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA2_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_ALPHA2_IN_SIZE`.
pub const NOLACE_TDSHAPE1_ALPHA2_IN_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_ALPHA2_STATE_SIZE` (`(80 * (1))`).
pub const NOLACE_TDSHAPE1_ALPHA2_STATE_SIZE: usize = 80;
/// `NOLACE_TDSHAPE1_ALPHA2_DELAY`.
pub const NOLACE_TDSHAPE1_ALPHA2_DELAY: usize = 0;
/// `NOLACE_TDSHAPE2_FEATURE_DIM`.
pub const NOLACE_TDSHAPE2_FEATURE_DIM: usize = 160;
/// `NOLACE_TDSHAPE2_FRAME_SIZE`.
pub const NOLACE_TDSHAPE2_FRAME_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_AVG_POOL_K`.
pub const NOLACE_TDSHAPE2_AVG_POOL_K: usize = 4;
/// `NOLACE_TDSHAPE2_INNOVATE`.
pub const NOLACE_TDSHAPE2_INNOVATE: usize = 0;
/// `NOLACE_TDSHAPE2_POOL_AFTER`.
pub const NOLACE_TDSHAPE2_POOL_AFTER: usize = 0;
/// `NOLACE_TDSHAPE2_ALPHA1_F_OUT_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA1_F_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_ALPHA1_F_IN_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA1_F_IN_SIZE: usize = 160;
/// `NOLACE_TDSHAPE2_ALPHA1_F_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_TDSHAPE2_ALPHA1_F_STATE_SIZE: usize = 160;
/// `NOLACE_TDSHAPE2_ALPHA1_F_DELAY`.
pub const NOLACE_TDSHAPE2_ALPHA1_F_DELAY: usize = 0;
/// `NOLACE_TDSHAPE2_ALPHA1_T_OUT_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA1_T_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_ALPHA1_T_IN_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA1_T_IN_SIZE: usize = 21;
/// `NOLACE_TDSHAPE2_ALPHA1_T_STATE_SIZE` (`(21 * (1))`).
pub const NOLACE_TDSHAPE2_ALPHA1_T_STATE_SIZE: usize = 21;
/// `NOLACE_TDSHAPE2_ALPHA1_T_DELAY`.
pub const NOLACE_TDSHAPE2_ALPHA1_T_DELAY: usize = 0;
/// `NOLACE_TDSHAPE2_ALPHA2_OUT_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA2_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_ALPHA2_IN_SIZE`.
pub const NOLACE_TDSHAPE2_ALPHA2_IN_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_ALPHA2_STATE_SIZE` (`(80 * (1))`).
pub const NOLACE_TDSHAPE2_ALPHA2_STATE_SIZE: usize = 80;
/// `NOLACE_TDSHAPE2_ALPHA2_DELAY`.
pub const NOLACE_TDSHAPE2_ALPHA2_DELAY: usize = 0;
/// `NOLACE_TDSHAPE3_FEATURE_DIM`.
pub const NOLACE_TDSHAPE3_FEATURE_DIM: usize = 160;
/// `NOLACE_TDSHAPE3_FRAME_SIZE`.
pub const NOLACE_TDSHAPE3_FRAME_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_AVG_POOL_K`.
pub const NOLACE_TDSHAPE3_AVG_POOL_K: usize = 4;
/// `NOLACE_TDSHAPE3_INNOVATE`.
pub const NOLACE_TDSHAPE3_INNOVATE: usize = 0;
/// `NOLACE_TDSHAPE3_POOL_AFTER`.
pub const NOLACE_TDSHAPE3_POOL_AFTER: usize = 0;
/// `NOLACE_TDSHAPE3_ALPHA1_F_OUT_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA1_F_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_ALPHA1_F_IN_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA1_F_IN_SIZE: usize = 160;
/// `NOLACE_TDSHAPE3_ALPHA1_F_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_TDSHAPE3_ALPHA1_F_STATE_SIZE: usize = 160;
/// `NOLACE_TDSHAPE3_ALPHA1_F_DELAY`.
pub const NOLACE_TDSHAPE3_ALPHA1_F_DELAY: usize = 0;
/// `NOLACE_TDSHAPE3_ALPHA1_T_OUT_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA1_T_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_ALPHA1_T_IN_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA1_T_IN_SIZE: usize = 21;
/// `NOLACE_TDSHAPE3_ALPHA1_T_STATE_SIZE` (`(21 * (1))`).
pub const NOLACE_TDSHAPE3_ALPHA1_T_STATE_SIZE: usize = 21;
/// `NOLACE_TDSHAPE3_ALPHA1_T_DELAY`.
pub const NOLACE_TDSHAPE3_ALPHA1_T_DELAY: usize = 0;
/// `NOLACE_TDSHAPE3_ALPHA2_OUT_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA2_OUT_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_ALPHA2_IN_SIZE`.
pub const NOLACE_TDSHAPE3_ALPHA2_IN_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_ALPHA2_STATE_SIZE` (`(80 * (1))`).
pub const NOLACE_TDSHAPE3_ALPHA2_STATE_SIZE: usize = 80;
/// `NOLACE_TDSHAPE3_ALPHA2_DELAY`.
pub const NOLACE_TDSHAPE3_ALPHA2_DELAY: usize = 0;
/// `NOLACE_AF2_FILTER_GAIN_A`.
pub const NOLACE_AF2_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `NOLACE_AF2_FILTER_GAIN_B`.
pub const NOLACE_AF2_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_AF2_SHAPE_GAIN`.
pub const NOLACE_AF2_SHAPE_GAIN: f32 = 1.000000_f32;
/// `NOLACE_AF2_KERNEL_SIZE`.
pub const NOLACE_AF2_KERNEL_SIZE: usize = 16;
/// `NOLACE_AF2_FRAME_SIZE`.
pub const NOLACE_AF2_FRAME_SIZE: usize = 80;
/// `NOLACE_AF2_LEFT_PADDING`.
pub const NOLACE_AF2_LEFT_PADDING: usize = 15;
/// `NOLACE_AF2_OVERLAP_SIZE`.
pub const NOLACE_AF2_OVERLAP_SIZE: usize = 40;
/// `NOLACE_AF2_IN_CHANNELS`.
pub const NOLACE_AF2_IN_CHANNELS: usize = 2;
/// `NOLACE_AF2_OUT_CHANNELS`.
pub const NOLACE_AF2_OUT_CHANNELS: usize = 2;
/// `NOLACE_AF2_NORM_P`.
pub const NOLACE_AF2_NORM_P: usize = 2;
/// `NOLACE_AF2_FEATURE_DIM`.
pub const NOLACE_AF2_FEATURE_DIM: usize = 160;
/// `NOLACE_AF2_KERNEL_OUT_SIZE`.
pub const NOLACE_AF2_KERNEL_OUT_SIZE: usize = 64;
/// `NOLACE_AF2_GAIN_OUT_SIZE`.
pub const NOLACE_AF2_GAIN_OUT_SIZE: usize = 2;
/// `NOLACE_AF3_FILTER_GAIN_A`.
pub const NOLACE_AF3_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `NOLACE_AF3_FILTER_GAIN_B`.
pub const NOLACE_AF3_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_AF3_SHAPE_GAIN`.
pub const NOLACE_AF3_SHAPE_GAIN: f32 = 1.000000_f32;
/// `NOLACE_AF3_KERNEL_SIZE`.
pub const NOLACE_AF3_KERNEL_SIZE: usize = 16;
/// `NOLACE_AF3_FRAME_SIZE`.
pub const NOLACE_AF3_FRAME_SIZE: usize = 80;
/// `NOLACE_AF3_LEFT_PADDING`.
pub const NOLACE_AF3_LEFT_PADDING: usize = 15;
/// `NOLACE_AF3_OVERLAP_SIZE`.
pub const NOLACE_AF3_OVERLAP_SIZE: usize = 40;
/// `NOLACE_AF3_IN_CHANNELS`.
pub const NOLACE_AF3_IN_CHANNELS: usize = 2;
/// `NOLACE_AF3_OUT_CHANNELS`.
pub const NOLACE_AF3_OUT_CHANNELS: usize = 2;
/// `NOLACE_AF3_NORM_P`.
pub const NOLACE_AF3_NORM_P: usize = 2;
/// `NOLACE_AF3_FEATURE_DIM`.
pub const NOLACE_AF3_FEATURE_DIM: usize = 160;
/// `NOLACE_AF3_KERNEL_OUT_SIZE`.
pub const NOLACE_AF3_KERNEL_OUT_SIZE: usize = 64;
/// `NOLACE_AF3_GAIN_OUT_SIZE`.
pub const NOLACE_AF3_GAIN_OUT_SIZE: usize = 2;
/// `NOLACE_AF4_FILTER_GAIN_A`.
pub const NOLACE_AF4_FILTER_GAIN_A: f32 = 1.381551_f32;
/// `NOLACE_AF4_FILTER_GAIN_B`.
pub const NOLACE_AF4_FILTER_GAIN_B: f32 = 0.000000_f32;
/// `NOLACE_AF4_SHAPE_GAIN`.
pub const NOLACE_AF4_SHAPE_GAIN: f32 = 1.000000_f32;
/// `NOLACE_AF4_KERNEL_SIZE`.
pub const NOLACE_AF4_KERNEL_SIZE: usize = 16;
/// `NOLACE_AF4_FRAME_SIZE`.
pub const NOLACE_AF4_FRAME_SIZE: usize = 80;
/// `NOLACE_AF4_LEFT_PADDING`.
pub const NOLACE_AF4_LEFT_PADDING: usize = 15;
/// `NOLACE_AF4_OVERLAP_SIZE`.
pub const NOLACE_AF4_OVERLAP_SIZE: usize = 40;
/// `NOLACE_AF4_IN_CHANNELS`.
pub const NOLACE_AF4_IN_CHANNELS: usize = 2;
/// `NOLACE_AF4_OUT_CHANNELS`.
pub const NOLACE_AF4_OUT_CHANNELS: usize = 1;
/// `NOLACE_AF4_NORM_P`.
pub const NOLACE_AF4_NORM_P: usize = 2;
/// `NOLACE_AF4_FEATURE_DIM`.
pub const NOLACE_AF4_FEATURE_DIM: usize = 160;
/// `NOLACE_AF4_KERNEL_OUT_SIZE`.
pub const NOLACE_AF4_KERNEL_OUT_SIZE: usize = 32;
/// `NOLACE_AF4_GAIN_OUT_SIZE`.
pub const NOLACE_AF4_GAIN_OUT_SIZE: usize = 1;
/// `NOLACE_POST_CF1_OUT_SIZE`.
pub const NOLACE_POST_CF1_OUT_SIZE: usize = 160;
/// `NOLACE_POST_CF1_IN_SIZE`.
pub const NOLACE_POST_CF1_IN_SIZE: usize = 160;
/// `NOLACE_POST_CF1_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_POST_CF1_STATE_SIZE: usize = 160;
/// `NOLACE_POST_CF1_DELAY`.
pub const NOLACE_POST_CF1_DELAY: usize = 0;
/// `NOLACE_POST_CF2_OUT_SIZE`.
pub const NOLACE_POST_CF2_OUT_SIZE: usize = 160;
/// `NOLACE_POST_CF2_IN_SIZE`.
pub const NOLACE_POST_CF2_IN_SIZE: usize = 160;
/// `NOLACE_POST_CF2_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_POST_CF2_STATE_SIZE: usize = 160;
/// `NOLACE_POST_CF2_DELAY`.
pub const NOLACE_POST_CF2_DELAY: usize = 0;
/// `NOLACE_POST_AF1_OUT_SIZE`.
pub const NOLACE_POST_AF1_OUT_SIZE: usize = 160;
/// `NOLACE_POST_AF1_IN_SIZE`.
pub const NOLACE_POST_AF1_IN_SIZE: usize = 160;
/// `NOLACE_POST_AF1_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_POST_AF1_STATE_SIZE: usize = 160;
/// `NOLACE_POST_AF1_DELAY`.
pub const NOLACE_POST_AF1_DELAY: usize = 0;
/// `NOLACE_POST_AF2_OUT_SIZE`.
pub const NOLACE_POST_AF2_OUT_SIZE: usize = 160;
/// `NOLACE_POST_AF2_IN_SIZE`.
pub const NOLACE_POST_AF2_IN_SIZE: usize = 160;
/// `NOLACE_POST_AF2_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_POST_AF2_STATE_SIZE: usize = 160;
/// `NOLACE_POST_AF2_DELAY`.
pub const NOLACE_POST_AF2_DELAY: usize = 0;
/// `NOLACE_POST_AF3_OUT_SIZE`.
pub const NOLACE_POST_AF3_OUT_SIZE: usize = 160;
/// `NOLACE_POST_AF3_IN_SIZE`.
pub const NOLACE_POST_AF3_IN_SIZE: usize = 160;
/// `NOLACE_POST_AF3_STATE_SIZE` (`(160 * (1))`).
pub const NOLACE_POST_AF3_STATE_SIZE: usize = 160;
/// `NOLACE_POST_AF3_DELAY`.
pub const NOLACE_POST_AF3_DELAY: usize = 0;

/// `NOLACELayers`: the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoLaceLayers {
    pub nolace_pitch_embedding: LinearLayer,
    pub nolace_fnet_conv1: LinearLayer,
    pub nolace_fnet_conv2: LinearLayer,
    pub nolace_fnet_tconv: LinearLayer,
    pub nolace_fnet_gru_input: LinearLayer,
    pub nolace_fnet_gru_recurrent: LinearLayer,
    pub nolace_cf1_kernel: LinearLayer,
    pub nolace_cf1_gain: LinearLayer,
    pub nolace_cf1_global_gain: LinearLayer,
    pub nolace_cf2_kernel: LinearLayer,
    pub nolace_cf2_gain: LinearLayer,
    pub nolace_cf2_global_gain: LinearLayer,
    pub nolace_af1_kernel: LinearLayer,
    pub nolace_af1_gain: LinearLayer,
    pub nolace_tdshape1_alpha1_f: LinearLayer,
    pub nolace_tdshape1_alpha1_t: LinearLayer,
    pub nolace_tdshape1_alpha2: LinearLayer,
    pub nolace_tdshape2_alpha1_f: LinearLayer,
    pub nolace_tdshape2_alpha1_t: LinearLayer,
    pub nolace_tdshape2_alpha2: LinearLayer,
    pub nolace_tdshape3_alpha1_f: LinearLayer,
    pub nolace_tdshape3_alpha1_t: LinearLayer,
    pub nolace_tdshape3_alpha2: LinearLayer,
    pub nolace_af2_kernel: LinearLayer,
    pub nolace_af2_gain: LinearLayer,
    pub nolace_af3_kernel: LinearLayer,
    pub nolace_af3_gain: LinearLayer,
    pub nolace_af4_kernel: LinearLayer,
    pub nolace_af4_gain: LinearLayer,
    pub nolace_post_cf1: LinearLayer,
    pub nolace_post_cf2: LinearLayer,
    pub nolace_post_af1: LinearLayer,
    pub nolace_post_af2: LinearLayer,
    pub nolace_post_af3: LinearLayer,
}

/// Port of the generated dnn/nolace_data.c:init_nolacelayers (fails with `BadArg` where C returns 1).
pub fn init_nolacelayers(arrays: &[WeightArray<'_>]) -> Result<NoLaceLayers> {
    Ok(NoLaceLayers {
        nolace_pitch_embedding: linear_init(
            arrays,
            Some("nolace_pitch_embedding_bias"),
            None,
            None,
            Some("nolace_pitch_embedding_weights_float"),
            None,
            None,
            None,
            301,
            64,
        )?,
        nolace_fnet_conv1: linear_init(
            arrays,
            Some("nolace_fnet_conv1_bias"),
            None,
            None,
            Some("nolace_fnet_conv1_weights_float"),
            None,
            None,
            None,
            173,
            96,
        )?,
        nolace_fnet_conv2: linear_init(
            arrays,
            Some("nolace_fnet_conv2_bias"),
            Some("nolace_fnet_conv2_subias"),
            Some("nolace_fnet_conv2_weights_int8"),
            Some("nolace_fnet_conv2_weights_float"),
            None,
            None,
            Some("nolace_fnet_conv2_scale"),
            768,
            160,
        )?,
        nolace_fnet_tconv: linear_init(
            arrays,
            Some("nolace_fnet_tconv_bias"),
            Some("nolace_fnet_tconv_subias"),
            Some("nolace_fnet_tconv_weights_int8"),
            Some("nolace_fnet_tconv_weights_float"),
            None,
            None,
            Some("nolace_fnet_tconv_scale"),
            160,
            640,
        )?,
        nolace_fnet_gru_input: linear_init(
            arrays,
            Some("nolace_fnet_gru_input_bias"),
            Some("nolace_fnet_gru_input_subias"),
            Some("nolace_fnet_gru_input_weights_int8"),
            Some("nolace_fnet_gru_input_weights_float"),
            None,
            None,
            Some("nolace_fnet_gru_input_scale"),
            160,
            480,
        )?,
        nolace_fnet_gru_recurrent: linear_init(
            arrays,
            Some("nolace_fnet_gru_recurrent_bias"),
            Some("nolace_fnet_gru_recurrent_subias"),
            Some("nolace_fnet_gru_recurrent_weights_int8"),
            Some("nolace_fnet_gru_recurrent_weights_float"),
            None,
            None,
            Some("nolace_fnet_gru_recurrent_scale"),
            160,
            480,
        )?,
        nolace_cf1_kernel: linear_init(
            arrays,
            Some("nolace_cf1_kernel_bias"),
            Some("nolace_cf1_kernel_subias"),
            Some("nolace_cf1_kernel_weights_int8"),
            Some("nolace_cf1_kernel_weights_float"),
            None,
            None,
            Some("nolace_cf1_kernel_scale"),
            160,
            16,
        )?,
        nolace_cf1_gain: linear_init(
            arrays,
            Some("nolace_cf1_gain_bias"),
            None,
            None,
            Some("nolace_cf1_gain_weights_float"),
            None,
            None,
            None,
            160,
            1,
        )?,
        nolace_cf1_global_gain: linear_init(
            arrays,
            Some("nolace_cf1_global_gain_bias"),
            None,
            None,
            Some("nolace_cf1_global_gain_weights_float"),
            None,
            None,
            None,
            160,
            1,
        )?,
        nolace_cf2_kernel: linear_init(
            arrays,
            Some("nolace_cf2_kernel_bias"),
            Some("nolace_cf2_kernel_subias"),
            Some("nolace_cf2_kernel_weights_int8"),
            Some("nolace_cf2_kernel_weights_float"),
            None,
            None,
            Some("nolace_cf2_kernel_scale"),
            160,
            16,
        )?,
        nolace_cf2_gain: linear_init(
            arrays,
            Some("nolace_cf2_gain_bias"),
            None,
            None,
            Some("nolace_cf2_gain_weights_float"),
            None,
            None,
            None,
            160,
            1,
        )?,
        nolace_cf2_global_gain: linear_init(
            arrays,
            Some("nolace_cf2_global_gain_bias"),
            None,
            None,
            Some("nolace_cf2_global_gain_weights_float"),
            None,
            None,
            None,
            160,
            1,
        )?,
        nolace_af1_kernel: linear_init(
            arrays,
            Some("nolace_af1_kernel_bias"),
            Some("nolace_af1_kernel_subias"),
            Some("nolace_af1_kernel_weights_int8"),
            Some("nolace_af1_kernel_weights_float"),
            None,
            None,
            Some("nolace_af1_kernel_scale"),
            160,
            32,
        )?,
        nolace_af1_gain: linear_init(
            arrays,
            Some("nolace_af1_gain_bias"),
            None,
            None,
            Some("nolace_af1_gain_weights_float"),
            None,
            None,
            None,
            160,
            2,
        )?,
        nolace_tdshape1_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha1_f_bias"),
            Some("nolace_tdshape1_alpha1_f_subias"),
            Some("nolace_tdshape1_alpha1_f_weights_int8"),
            Some("nolace_tdshape1_alpha1_f_weights_float"),
            None,
            None,
            Some("nolace_tdshape1_alpha1_f_scale"),
            320,
            80,
        )?,
        nolace_tdshape1_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha1_t_bias"),
            None,
            None,
            Some("nolace_tdshape1_alpha1_t_weights_float"),
            None,
            None,
            None,
            42,
            80,
        )?,
        nolace_tdshape1_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha2_bias"),
            None,
            None,
            Some("nolace_tdshape1_alpha2_weights_float"),
            None,
            None,
            None,
            160,
            80,
        )?,
        nolace_tdshape2_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha1_f_bias"),
            Some("nolace_tdshape2_alpha1_f_subias"),
            Some("nolace_tdshape2_alpha1_f_weights_int8"),
            Some("nolace_tdshape2_alpha1_f_weights_float"),
            None,
            None,
            Some("nolace_tdshape2_alpha1_f_scale"),
            320,
            80,
        )?,
        nolace_tdshape2_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha1_t_bias"),
            None,
            None,
            Some("nolace_tdshape2_alpha1_t_weights_float"),
            None,
            None,
            None,
            42,
            80,
        )?,
        nolace_tdshape2_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha2_bias"),
            None,
            None,
            Some("nolace_tdshape2_alpha2_weights_float"),
            None,
            None,
            None,
            160,
            80,
        )?,
        nolace_tdshape3_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha1_f_bias"),
            Some("nolace_tdshape3_alpha1_f_subias"),
            Some("nolace_tdshape3_alpha1_f_weights_int8"),
            Some("nolace_tdshape3_alpha1_f_weights_float"),
            None,
            None,
            Some("nolace_tdshape3_alpha1_f_scale"),
            320,
            80,
        )?,
        nolace_tdshape3_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha1_t_bias"),
            None,
            None,
            Some("nolace_tdshape3_alpha1_t_weights_float"),
            None,
            None,
            None,
            42,
            80,
        )?,
        nolace_tdshape3_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha2_bias"),
            None,
            None,
            Some("nolace_tdshape3_alpha2_weights_float"),
            None,
            None,
            None,
            160,
            80,
        )?,
        nolace_af2_kernel: linear_init(
            arrays,
            Some("nolace_af2_kernel_bias"),
            Some("nolace_af2_kernel_subias"),
            Some("nolace_af2_kernel_weights_int8"),
            Some("nolace_af2_kernel_weights_float"),
            None,
            None,
            Some("nolace_af2_kernel_scale"),
            160,
            64,
        )?,
        nolace_af2_gain: linear_init(
            arrays,
            Some("nolace_af2_gain_bias"),
            None,
            None,
            Some("nolace_af2_gain_weights_float"),
            None,
            None,
            None,
            160,
            2,
        )?,
        nolace_af3_kernel: linear_init(
            arrays,
            Some("nolace_af3_kernel_bias"),
            Some("nolace_af3_kernel_subias"),
            Some("nolace_af3_kernel_weights_int8"),
            Some("nolace_af3_kernel_weights_float"),
            None,
            None,
            Some("nolace_af3_kernel_scale"),
            160,
            64,
        )?,
        nolace_af3_gain: linear_init(
            arrays,
            Some("nolace_af3_gain_bias"),
            None,
            None,
            Some("nolace_af3_gain_weights_float"),
            None,
            None,
            None,
            160,
            2,
        )?,
        nolace_af4_kernel: linear_init(
            arrays,
            Some("nolace_af4_kernel_bias"),
            Some("nolace_af4_kernel_subias"),
            Some("nolace_af4_kernel_weights_int8"),
            Some("nolace_af4_kernel_weights_float"),
            None,
            None,
            Some("nolace_af4_kernel_scale"),
            160,
            32,
        )?,
        nolace_af4_gain: linear_init(
            arrays,
            Some("nolace_af4_gain_bias"),
            None,
            None,
            Some("nolace_af4_gain_weights_float"),
            None,
            None,
            None,
            160,
            1,
        )?,
        nolace_post_cf1: linear_init(
            arrays,
            Some("nolace_post_cf1_bias"),
            Some("nolace_post_cf1_subias"),
            Some("nolace_post_cf1_weights_int8"),
            Some("nolace_post_cf1_weights_float"),
            None,
            None,
            Some("nolace_post_cf1_scale"),
            320,
            160,
        )?,
        nolace_post_cf2: linear_init(
            arrays,
            Some("nolace_post_cf2_bias"),
            Some("nolace_post_cf2_subias"),
            Some("nolace_post_cf2_weights_int8"),
            Some("nolace_post_cf2_weights_float"),
            None,
            None,
            Some("nolace_post_cf2_scale"),
            320,
            160,
        )?,
        nolace_post_af1: linear_init(
            arrays,
            Some("nolace_post_af1_bias"),
            Some("nolace_post_af1_subias"),
            Some("nolace_post_af1_weights_int8"),
            Some("nolace_post_af1_weights_float"),
            None,
            None,
            Some("nolace_post_af1_scale"),
            320,
            160,
        )?,
        nolace_post_af2: linear_init(
            arrays,
            Some("nolace_post_af2_bias"),
            Some("nolace_post_af2_subias"),
            Some("nolace_post_af2_weights_int8"),
            Some("nolace_post_af2_weights_float"),
            None,
            None,
            Some("nolace_post_af2_scale"),
            320,
            160,
        )?,
        nolace_post_af3: linear_init(
            arrays,
            Some("nolace_post_af3_bias"),
            Some("nolace_post_af3_subias"),
            Some("nolace_post_af3_weights_int8"),
            Some("nolace_post_af3_weights_float"),
            None,
            None,
            Some("nolace_post_af3_scale"),
            320,
            160,
        )?,
    })
}

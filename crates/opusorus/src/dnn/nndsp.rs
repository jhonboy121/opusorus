//! Port of `dnn/nndsp.c` / `dnn/nndsp.h`: the adaptive convolution / comb filter / temporal
//! shaping layers used by OSCE (LACE, NoLACE).
//!
//! C calls `adaconv_process_frame` / `adacomb_process_frame` / `adashape_process_frame` in place
//! (`x_out == x_in`); here `x_in: None` means "read the input from `x_out`". `DEBUG_NNDSP` is
//! not defined upstream (`print_float_vector` is only declared then), so it is not ported.

use crate::celt::mathops::celt_log;
use crate::celt::pitch::celt_pitch_xcorr;
use crate::math;

use super::nnet::{
    ACTIVATION_EXP, ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_TANH, LinearLayer,
    compute_activation_inplace, compute_generic_conv1d, compute_generic_dense,
};

/// `ADACONV_MAX_KERNEL_SIZE`.
pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
/// `ADACONV_MAX_INPUT_CHANNELS`.
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
/// `ADACONV_MAX_OUTPUT_CHANNELS`.
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
/// `ADACONV_MAX_FRAME_SIZE`.
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
/// `ADACONV_MAX_OVERLAP_SIZE`.
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;

/// `ADACOMB_MAX_LAG`.
pub const ADACOMB_MAX_LAG: usize = 300;
/// `ADACOMB_MAX_KERNEL_SIZE`.
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
/// `ADACOMB_MAX_FRAME_SIZE`.
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;
/// `ADACOMB_MAX_OVERLAP_SIZE`.
pub const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;

/// `ADASHAPE_MAX_INPUT_DIM`.
pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
/// `ADASHAPE_MAX_FRAME_SIZE`.
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

/// `M_PI` as seen by nndsp.c (glibc `<math.h>` defines it as a double constant).
const M_PI: f64 = 3.14159265358979323846;

/// `AdaConvState`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaConvState {
    pub history: [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
    pub last_kernel:
        [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS],
    pub last_gain: f32,
}

impl Default for AdaConvState {
    fn default() -> Self {
        Self {
            history: [0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
            last_kernel: [0.0; ADACONV_MAX_KERNEL_SIZE
                * ADACONV_MAX_INPUT_CHANNELS
                * ADACONV_MAX_OUTPUT_CHANNELS],
            last_gain: 0.0,
        }
    }
}

/// `AdaCombState`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaCombState {
    pub history: [f32; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
    pub last_kernel: [f32; ADACOMB_MAX_KERNEL_SIZE],
    pub last_global_gain: f32,
    pub last_pitch_lag: i32,
}

impl Default for AdaCombState {
    fn default() -> Self {
        Self {
            history: [0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: [0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

/// `AdaShapeState`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaShapeState {
    pub conv_alpha1f_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha1t_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha2_state: [f32; ADASHAPE_MAX_FRAME_SIZE],
    pub interpolate_state: [f32; 1],
}

impl Default for AdaShapeState {
    fn default() -> Self {
        Self {
            conv_alpha1f_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha1t_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha2_state: [0.0; ADASHAPE_MAX_FRAME_SIZE],
            interpolate_state: [0.0; 1],
        }
    }
}

/// Port of dnn/nndsp.c:init_adaconv_state.
pub fn init_adaconv_state(h: &mut AdaConvState) {
    *h = AdaConvState::default();
}

/// Port of dnn/nndsp.c:init_adacomb_state.
pub fn init_adacomb_state(h: &mut AdaCombState) {
    *h = AdaCombState::default();
}

/// Port of dnn/nndsp.c:init_adashape_state.
pub fn init_adashape_state(h: &mut AdaShapeState) {
    *h = AdaShapeState::default();
}

/// Port of dnn/nndsp.c:compute_overlap_window.
pub fn compute_overlap_window(window: &mut [f32], overlap_size: usize) {
    for (i_sample, w) in window[..overlap_size].iter_mut().enumerate() {
        // C: 0.5f + 0.5f * cos(M_PI * (i_sample + 0.5f) / overlap_size) — all in double.
        *w = (0.5f32 as f64
            + 0.5f32 as f64
                * math::cos(M_PI * (i_sample as f32 + 0.5f32) as f64 / overlap_size as f64))
            as f32;
    }
}

/// `KERNEL_INDEX(i_out_channels, i_in_channels, i_kernel)`.
#[inline(always)]
const fn kernel_index(
    i_out_channels: usize,
    i_in_channels: usize,
    i_kernel: usize,
    in_channels: usize,
    kernel_size: usize,
) -> usize {
    ((i_out_channels * in_channels) + i_in_channels) * kernel_size + i_kernel
}

/// Port of dnn/nndsp.c:scale_kernel (static): normalizes (p-norm) the kernel over the input
/// channel and kernel dimensions, then applies `gain[i_out_channels]`.
pub fn scale_kernel(
    kernel: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    gain: &[f32],
) {
    for i_out in 0..out_channels {
        let mut norm = 0f32;
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                let v = kernel[kernel_index(i_out, i_in, i_k, in_channels, kernel_size)];
                norm += v * v;
            }
        }
        // C: norm = 1.f / (1e-6f + sqrt(norm)) (double).
        norm = (1.0f64 / (1e-6f32 as f64 + math::sqrt(norm as f64))) as f32;
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                kernel[kernel_index(i_out, i_in, i_k, in_channels, kernel_size)] *=
                    norm * gain[i_out];
            }
        }
    }
}

/// Port of dnn/nndsp.c:transform_gains (static): `gains[i] = exp(a*gains[i] + b)`.
pub fn transform_gains(
    gains: &mut [f32],
    num_gains: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
) {
    for g in &mut gains[..num_gains] {
        *g = math::exp((filter_gain_a * *g + filter_gain_b) as f64) as f32;
    }
}

/// Port of dnn/nndsp.c:adaconv_process_frame. `x_in` (`None`: in place from `x_out`) holds
/// `in_channels*frame_size` samples, `x_out` receives `out_channels*frame_size`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of adaconv_process_frame"
)]
pub fn adaconv_process_frame(
    h: &mut AdaConvState,
    x_out: &mut [f32],
    x_in: Option<&[f32]>,
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    shape_gain: f32,
    window: &[f32],
) {
    let mut output_buffer = [0f32; ADACONV_MAX_FRAME_SIZE * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut kernel_buffer =
        [0f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut input_buffer =
        [0f32; ADACONV_MAX_INPUT_CHANNELS * (ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE)];
    let mut kernel0 = [0f32; ADACONV_MAX_KERNEL_SIZE];
    let mut kernel1 = [0f32; ADACONV_MAX_KERNEL_SIZE];
    let mut channel_buffer0 = [0f32; ADACONV_MAX_OVERLAP_SIZE];
    let mut channel_buffer1 = [0f32; ADACONV_MAX_FRAME_SIZE];
    let mut gain_buffer = [0f32; ADACONV_MAX_OUTPUT_CHANNELS];

    debug_assert!(shape_gain == 1.0);
    // Currently only supports the causal version.
    debug_assert!(left_padding + 1 == kernel_size);
    debug_assert!(kernel_size < frame_size);

    // Prepare input.
    {
        let x_in: &[f32] = match x_in {
            Some(x) => x,
            None => x_out,
        };
        for i_in in 0..in_channels {
            let base = i_in * (kernel_size + frame_size);
            input_buffer[base..base + kernel_size]
                .copy_from_slice(&h.history[i_in * kernel_size..(i_in + 1) * kernel_size]);
            input_buffer[base + kernel_size..base + kernel_size + frame_size]
                .copy_from_slice(&x_in[frame_size * i_in..frame_size * (i_in + 1)]);
        }
    }
    // p_input = input_buffer + kernel_size.
    let p_input = kernel_size;

    // Calculate new kernel and new gain.
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer,
        features,
        ACTIVATION_LINEAR,
    );
    compute_generic_dense(gain_layer, &mut gain_buffer, features, ACTIVATION_TANH);
    transform_gains(&mut gain_buffer, out_channels, filter_gain_a, filter_gain_b);
    scale_kernel(
        &mut kernel_buffer,
        in_channels,
        out_channels,
        kernel_size,
        &gain_buffer,
    );

    // Calculate overlapping part using kernel from last frame.
    for i_out in 0..out_channels {
        for i_in in 0..in_channels {
            kernel0.fill(0.0);
            kernel1.fill(0.0);
            let k0 = kernel_index(i_out, i_in, 0, in_channels, kernel_size);
            kernel0[..kernel_size].copy_from_slice(&h.last_kernel[k0..k0 + kernel_size]);
            kernel1[..kernel_size].copy_from_slice(&kernel_buffer[k0..k0 + kernel_size]);
            let start = p_input + i_in * (frame_size + kernel_size) - left_padding;
            celt_pitch_xcorr(
                &kernel0,
                &input_buffer[start..],
                &mut channel_buffer0,
                ADACONV_MAX_KERNEL_SIZE,
                overlap_size,
            );
            celt_pitch_xcorr(
                &kernel1,
                &input_buffer[start..],
                &mut channel_buffer1,
                ADACONV_MAX_KERNEL_SIZE,
                frame_size,
            );
            let out = &mut output_buffer[i_out * frame_size..(i_out + 1) * frame_size];
            for i_sample in 0..overlap_size {
                out[i_sample] += window[i_sample] * channel_buffer0[i_sample];
                out[i_sample] += (1.0f32 - window[i_sample]) * channel_buffer1[i_sample];
            }
            for i_sample in overlap_size..frame_size {
                out[i_sample] += channel_buffer1[i_sample];
            }
        }
    }

    x_out[..out_channels * frame_size].copy_from_slice(&output_buffer[..out_channels * frame_size]);

    // Buffer update.
    for i_in in 0..in_channels {
        let src = p_input + i_in * (frame_size + kernel_size) + frame_size - kernel_size;
        h.history[i_in * kernel_size..(i_in + 1) * kernel_size]
            .copy_from_slice(&input_buffer[src..src + kernel_size]);
    }
    let n = kernel_size * in_channels * out_channels;
    h.last_kernel[..n].copy_from_slice(&kernel_buffer[..n]);
}

/// Port of dnn/nndsp.c:adacomb_process_frame. `x_in` (`None`: in place from `x_out`) and
/// `x_out` hold `frame_size` samples.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of adacomb_process_frame"
)]
pub fn adacomb_process_frame(
    h: &mut AdaCombState,
    x_out: &mut [f32],
    x_in: Option<&[f32]>,
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    global_gain_layer: &LinearLayer,
    pitch_lag: i32,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    log_gain_limit: f32,
    window: &[f32],
) {
    let mut output_buffer = [0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut output_buffer_last = [0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut kernel_buffer = [0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut input_buffer =
        [0f32; ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];
    let mut gain = [0f32; 1];
    let mut global_gain = [0f32; 1];
    let mut kernel = [0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut last_kernel = [0f32; ADACOMB_MAX_KERNEL_SIZE];

    input_buffer[..kernel_size + ADACOMB_MAX_LAG]
        .copy_from_slice(&h.history[..kernel_size + ADACOMB_MAX_LAG]);
    {
        let x_in: &[f32] = match x_in {
            Some(x) => x,
            None => x_out,
        };
        input_buffer[kernel_size + ADACOMB_MAX_LAG..kernel_size + ADACOMB_MAX_LAG + frame_size]
            .copy_from_slice(&x_in[..frame_size]);
    }
    let p_input = kernel_size + ADACOMB_MAX_LAG;

    // Calculate new kernel and new gain.
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer,
        features,
        ACTIVATION_LINEAR,
    );
    compute_generic_dense(gain_layer, &mut gain, features, ACTIVATION_RELU);
    compute_generic_dense(
        global_gain_layer,
        &mut global_gain,
        features,
        ACTIVATION_TANH,
    );
    gain[0] = math::exp((log_gain_limit - gain[0]) as f64) as f32;
    let global_gain = math::exp((filter_gain_a * global_gain[0] + filter_gain_b) as f64) as f32;
    scale_kernel(&mut kernel_buffer, 1, 1, kernel_size, &gain);

    kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    last_kernel[..kernel_size].copy_from_slice(&h.last_kernel[..kernel_size]);

    // C: &p_input[- left_padding - last_pitch_lag] / &p_input[- left_padding - pitch_lag].
    let last_start = (p_input as i32 - left_padding as i32 - h.last_pitch_lag) as usize;
    celt_pitch_xcorr(
        &last_kernel,
        &input_buffer[last_start..],
        &mut output_buffer_last,
        ADACOMB_MAX_KERNEL_SIZE,
        overlap_size,
    );
    let start = (p_input as i32 - left_padding as i32 - pitch_lag) as usize;
    celt_pitch_xcorr(
        &kernel,
        &input_buffer[start..],
        &mut output_buffer,
        ADACOMB_MAX_KERNEL_SIZE,
        frame_size,
    );
    let x = &input_buffer[p_input..p_input + frame_size];
    for i in 0..overlap_size {
        output_buffer[i] = h.last_global_gain * window[i] * output_buffer_last[i]
            + global_gain * (1.0f32 - window[i]) * output_buffer[i];
    }
    for i in 0..overlap_size {
        output_buffer[i] +=
            (window[i] * h.last_global_gain + (1.0f32 - window[i]) * global_gain) * x[i];
    }
    for i in overlap_size..frame_size {
        output_buffer[i] = global_gain * (output_buffer[i] + x[i]);
    }
    x_out[..frame_size].copy_from_slice(&output_buffer[..frame_size]);

    // Buffer update.
    h.last_kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    let src = p_input + frame_size - kernel_size - ADACOMB_MAX_LAG;
    h.history[..kernel_size + ADACOMB_MAX_LAG]
        .copy_from_slice(&input_buffer[src..src + kernel_size + ADACOMB_MAX_LAG]);
    h.last_pitch_lag = pitch_lag;
    h.last_global_gain = global_gain;
}

/// Port of dnn/nndsp.c:adashape_process_frame. `x_in` (`None`: in place from `x_out`) and
/// `x_out` hold `frame_size` samples.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of adashape_process_frame"
)]
pub fn adashape_process_frame(
    h: &mut AdaShapeState,
    x_out: &mut [f32],
    x_in: Option<&[f32]>,
    features: &[f32],
    alpha1f: &LinearLayer,
    alpha1t: &LinearLayer,
    alpha2: &LinearLayer,
    feature_dim: usize,
    frame_size: usize,
    avg_pool_k: usize,
    interpolate_k: usize,
) {
    let mut in_buffer = [0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut out_buffer = [0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = [0f32; ADASHAPE_MAX_FRAME_SIZE];
    let hidden_dim = frame_size / interpolate_k;
    let f = 1.0f32 / avg_pool_k as f32;

    debug_assert!(frame_size.is_multiple_of(avg_pool_k));
    debug_assert!(frame_size.is_multiple_of(interpolate_k));
    debug_assert!(feature_dim + frame_size / avg_pool_k + 1 < ADASHAPE_MAX_INPUT_DIM);

    let tenv_size = frame_size / avg_pool_k;
    // tenv = in_buffer + feature_dim
    let tenv = feature_dim;
    in_buffer[tenv..tenv + tenv_size + 1].fill(0.0);

    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);

    // Calculate temporal envelope.
    let mut mean = 0f32;
    {
        let x_in: &[f32] = match x_in {
            Some(x) => x,
            None => x_out,
        };
        for i in 0..tenv_size {
            for k in 0..avg_pool_k {
                // C: tenv[i] += fabs(x_in[...]) (double addition, stored to float).
                in_buffer[tenv + i] = (in_buffer[tenv + i] as f64
                    + math::fabs(x_in[i * avg_pool_k + k] as f64))
                    as f32;
            }
            in_buffer[tenv + i] = celt_log(in_buffer[tenv + i] * f + 1.52587890625e-05f32);
            mean += in_buffer[tenv + i];
        }
    }
    mean /= tenv_size as f32;
    for v in &mut in_buffer[tenv..tenv + tenv_size] {
        *v -= mean;
    }
    in_buffer[tenv + tenv_size] = mean;

    // Calculate temporal weights.
    compute_generic_conv1d(
        alpha1f,
        &mut out_buffer,
        &mut h.conv_alpha1f_state,
        &in_buffer,
        feature_dim,
        ACTIVATION_LINEAR,
    );
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer,
        &mut h.conv_alpha1t_state,
        &in_buffer[tenv..],
        tenv_size + 1,
        ACTIVATION_LINEAR,
    );
    // Compute leaky ReLU by hand.
    for i in 0..hidden_dim {
        let tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if tmp >= 0.0 {
            tmp
        } else {
            (0.2 * tmp as f64) as f32
        };
    }
    compute_generic_conv1d(
        alpha2,
        &mut tmp_buffer,
        &mut h.conv_alpha2_state,
        &in_buffer,
        hidden_dim,
        ACTIVATION_LINEAR,
    );

    // Upsampling by linear interpolation.
    for i in 0..hidden_dim {
        for k in 0..interpolate_k {
            let alpha = (k + 1) as f32 / interpolate_k as f32;
            out_buffer[i * interpolate_k + k] =
                alpha * tmp_buffer[i] + (1.0f32 - alpha) * h.interpolate_state[0];
        }
        h.interpolate_state[0] = tmp_buffer[i];
    }

    compute_activation_inplace(&mut out_buffer, frame_size, ACTIVATION_EXP);
    // Shape signal.
    match x_in {
        Some(x_in) => {
            for i in 0..frame_size {
                x_out[i] = out_buffer[i] * x_in[i];
            }
        }
        None =>
        {
            #[expect(
                clippy::assign_op_pattern,
                reason = "C operand order (out_buffer[i] * x_in[i]) is kept for NaN payloads"
            )]
            for i in 0..frame_size {
                x_out[i] = out_buffer[i] * x_out[i];
            }
        }
    }
}

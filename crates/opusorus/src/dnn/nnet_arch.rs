//! Port of `dnn/nnet_arch.h` as instantiated by `dnn/nnet_default.c` (`RTCD_ARCH c`): the
//! generic `compute_activation_c`, `compute_linear_c` and `compute_conv2d_c`.
//!
//! The `arch` arguments and the RTCD dispatch (`compute_linear(..., arch)` macros in `nnet.h`)
//! are dropped: the port always runs the generic C path.

use super::nnet::{
    ACTIVATION_EXP, ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX,
    ACTIVATION_SWISH, ACTIVATION_TANH, Conv2dLayer, LinearLayer,
};
use super::vec::{
    cgemv8x4, sgemv, sigmoid_approx, softmax, softmax_inplace, sparse_cgemv8x4, sparse_sgemv8x4,
    vec_sigmoid, vec_sigmoid_inplace, vec_tanh, vec_tanh_inplace,
};

/// `MAX_ACTIVATIONS`.
pub const MAX_ACTIVATIONS: usize = 4096;

/// `MAX_CONV2D_INPUTS`: size of the C `in_buf` of [`compute_conv2d`] (the scratch passed in by
/// the caller must hold `ktime*in_channels*(height+kheight-1)` floats).
pub const MAX_CONV2D_INPUTS: usize = 8192;

/// Port of dnn/nnet_arch.h:vec_swish.
///
/// C first computes `tmp = sigmoid(x)` for the whole vector and then `y = x*tmp`; both are
/// element-wise, so computing them per element gives identical results without the
/// `MAX_ACTIVATIONS` temporary.
pub fn vec_swish(y: &mut [f32], x: &[f32], n: usize) {
    debug_assert!(n <= MAX_ACTIVATIONS);
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = xi * sigmoid_approx(xi);
    }
}

/// In-place [`vec_swish`].
pub fn vec_swish_inplace(y: &mut [f32], n: usize) {
    debug_assert!(n <= MAX_ACTIVATIONS);
    for v in &mut y[..n] {
        *v *= sigmoid_approx(*v);
    }
}

/// Port of dnn/nnet_arch.h:relu.
#[inline(always)]
#[must_use]
pub const fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

/// Normalization step of the non-`SOFTMAX_HACK` softmax branch of compute_activation_c.
fn softmax_normalize(output: &mut [f32]) {
    let mut sum = 0f32;
    for &o in output.iter() {
        sum += o;
    }
    // C: sum = 1.f/(sum+1e-30) (double).
    let sum = (1.0f64 / (sum as f64 + 1e-30)) as f32;
    #[expect(
        clippy::assign_op_pattern,
        reason = "C operand order (sum*output[i]) is kept for NaN payloads"
    )]
    for o in output.iter_mut() {
        *o = sum * *o;
    }
}

/// Port of dnn/nnet_arch.h:compute_activation_c (`output != input`).
///
/// `HIGH_ACCURACY` is not defined upstream. `SOFTMAX_HACK` is defined in nnet.c, but this
/// function is compiled in nnet_default.c where it is *not* defined, so `ACTIVATION_SOFTMAX` is
/// the normalized `lpcnet_exp` softmax.
pub fn compute_activation(output: &mut [f32], input: &[f32], n: usize, activation: i32) {
    let output = &mut output[..n];
    let input = &input[..n];
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid(output, input, n),
        ACTIVATION_TANH => vec_tanh(output, input, n),
        ACTIVATION_SWISH => vec_swish(output, input, n),
        ACTIVATION_RELU => {
            for (o, &i) in output.iter_mut().zip(input) {
                *o = relu(i);
            }
        }
        ACTIVATION_SOFTMAX => {
            softmax(output, input, n);
            softmax_normalize(output);
        }
        ACTIVATION_EXP => softmax(output, input, n),
        _ => {
            // C: celt_assert(activation == ACTIVATION_LINEAR), then copies.
            debug_assert!(activation == ACTIVATION_LINEAR);
            output.copy_from_slice(input);
        }
    }
}

/// Port of dnn/nnet_arch.h:compute_activation_c for the in-place calls (`output == input`).
pub fn compute_activation_inplace(x: &mut [f32], n: usize, activation: i32) {
    let x = &mut x[..n];
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid_inplace(x, n),
        ACTIVATION_TANH => vec_tanh_inplace(x, n),
        ACTIVATION_SWISH => vec_swish_inplace(x, n),
        ACTIVATION_RELU => {
            for v in x.iter_mut() {
                *v = relu(*v);
            }
        }
        ACTIVATION_SOFTMAX => {
            softmax_inplace(x, n);
            softmax_normalize(x);
        }
        ACTIVATION_EXP => softmax_inplace(x, n),
        // LINEAR with `input == output`: nothing to do.
        _ => debug_assert!(activation == ACTIVATION_LINEAR),
    }
}

/// Port of dnn/nnet_arch.h:compute_linear_c. `out` needs `nb_outputs`, `input` `nb_inputs`
/// elements (C asserts `in != out`, which the borrow rules guarantee here).
pub fn compute_linear(linear: &LinearLayer, out: &mut [f32], input: &[f32]) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;
    let out = &mut out[..n];
    let bias = linear.bias.as_deref();
    if let Some(fw) = linear.float_weights.as_deref() {
        if let Some(idx) = linear.weights_idx.as_deref() {
            sparse_sgemv8x4(out, fw, idx, n, input);
        } else {
            sgemv(out, fw, n, m, n, input);
        }
    } else if let Some(w) = linear.weights.as_deref() {
        // C dereferences `scale` unconditionally here; `linear_init` always binds it together
        // with `weights`, so a missing scale is a construction bug (C would crash).
        #[expect(
            clippy::expect_used,
            reason = "linear_init binds `scale` whenever `weights` is bound"
        )]
        let scale = linear
            .scale
            .as_deref()
            .expect("int8 LinearLayer without scale");
        if let Some(idx) = linear.weights_idx.as_deref() {
            sparse_cgemv8x4(out, w, idx, scale, n, m, input);
        } else {
            cgemv8x4(out, w, scale, n, m, input);
        }
        // USE_SU_BIAS: not ported (generic path) — `bias = linear->subias` on SU archs.
    } else {
        out.fill(0.0);
    }
    if let Some(bias) = bias {
        for (o, &b) in out.iter_mut().zip(&bias[..n]) {
            *o += b;
        }
    }
    if let Some(diag) = linear.diag.as_deref() {
        // Diag is only used for GRU recurrent weights.
        debug_assert!(3 * m == n);
        let input = &input[..m];
        let diag = &diag[..3 * m];
        for i in 0..m {
            out[i] += diag[i] * input[i];
            out[i + m] += diag[i + m] * input[i];
            out[i + 2 * m] += diag[i + 2 * m] * input[i];
        }
    }
}

/// Port of dnn/nnet_arch.h:conv2d_float. Computes the non-padded convolution for input
/// `[ktime x in_channels x (height+kheight-1)]` and kernel
/// `[out_channels x in_channels x ktime x kheight]`, storing `[out_channels x height]` with row
/// stride `hstride`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of conv2d_float"
)]
pub fn conv2d_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let o = &mut out[i * hstride..i * hstride + height];
        o.fill(0.0);
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    let w = weights
                        [i * in_channels * ktime * kheight + m * ktime * kheight + t * kheight + h];
                    let inp = &input[t * in_channels * in_stride + m * in_stride + h..][..height];
                    for (oj, &ij) in o.iter_mut().zip(inp) {
                        *oj += w * ij;
                    }
                }
            }
        }
    }
}

/// Port of dnn/nnet_arch.h:conv2d_3x3_float (the unrolled 3x3 kernel: the nine products are
/// summed left to right before being added to the output).
pub fn conv2d_3x3_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    const KHEIGHT: usize = 3;
    const KTIME: usize = 3;
    let in_stride = height + KHEIGHT - 1;
    for i in 0..out_channels {
        let o = &mut out[i * hstride..i * hstride + height];
        o.fill(0.0);
        for m in 0..in_channels {
            let wb = i * in_channels * KTIME * KHEIGHT + m * KTIME * KHEIGHT;
            let w = &weights[wb..wb + 9];
            let in0 = &input[m * in_stride..][..height + 2];
            let in1 = &input[in_channels * in_stride + m * in_stride..][..height + 2];
            let in2 = &input[2 * in_channels * in_stride + m * in_stride..][..height + 2];
            for j in 0..height {
                o[j] += w[0] * in0[j]
                    + w[1] * in0[j + 1]
                    + w[2] * in0[j + 2]
                    + w[3] * in1[j]
                    + w[4] * in1[j + 1]
                    + w[5] * in1[j + 2]
                    + w[6] * in2[j]
                    + w[7] * in2[j + 1]
                    + w[8] * in2[j + 2];
            }
        }
    }
}

/// Port of dnn/nnet_arch.h:compute_conv2d_c.
///
/// `input` holds one time step (`in_channels*(height+kheight-1)` floats), `mem` the previous
/// `ktime-1` steps (updated). The C stack buffer `in_buf[MAX_CONV2D_INPUTS]` is the
/// caller-provided `in_buf` scratch, which needs `ktime*in_channels*(height+kheight-1)` floats.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature plus the scratch buffer"
)]
pub fn compute_conv2d(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
    in_buf: &mut [f32],
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    let hist = (conv.ktime - 1) * time_stride;
    debug_assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);
    let in_buf = &mut in_buf[..hist + time_stride];
    in_buf[..hist].copy_from_slice(&mem[..hist]);
    in_buf[hist..].copy_from_slice(&input[..time_stride]);
    mem[..hist].copy_from_slice(&in_buf[time_stride..]);
    // C dereferences `float_weights` unconditionally (all upstream conv2d layers have them).
    #[expect(
        clippy::expect_used,
        reason = "a Conv2dLayer without weights is a construction bug (NULL dereference in C)"
    )]
    let weights = conv
        .float_weights
        .as_deref()
        .expect("Conv2dLayer without float weights");
    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            in_buf,
            height,
            hstride,
        );
    } else {
        conv2d_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            conv.ktime,
            conv.kheight,
            in_buf,
            height,
            hstride,
        );
    }
    if let Some(bias) = conv.bias.as_deref() {
        for (i, &b) in bias[..conv.out_channels].iter().enumerate() {
            for o in &mut out[i * hstride..i * hstride + height] {
                *o += b;
            }
        }
    }
    for i in 0..conv.out_channels {
        compute_activation_inplace(&mut out[i * hstride..], height, activation);
    }
}

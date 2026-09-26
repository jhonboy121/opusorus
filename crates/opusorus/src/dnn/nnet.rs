//! Port of `dnn/nnet.h` and `dnn/nnet.c`: layer types, activation ids and the generic
//! dense / GRU / GLU / conv1d layers built on [`compute_linear`] and [`compute_activation`].
//!
//! The weight-blob format and the layer binding helpers of `dnn/parse_lpcnet_weights.c` live in
//! [`super::parse_lpcnet_weights`].

use alloc::vec::Vec;

#[cfg(feature = "deep-plc")]
pub use super::nnet_arch::compute_conv2d;
pub use super::nnet_arch::{compute_activation_inplace, compute_linear};
use super::vec::MAX_INPUTS;

/// `ACTIVATION_LINEAR`.
pub const ACTIVATION_LINEAR: i32 = 0;
/// `ACTIVATION_SIGMOID`.
pub const ACTIVATION_SIGMOID: i32 = 1;
/// `ACTIVATION_TANH`.
pub const ACTIVATION_TANH: i32 = 2;
/// `ACTIVATION_RELU`.
pub const ACTIVATION_RELU: i32 = 3;
/// `ACTIVATION_SOFTMAX`.
pub const ACTIVATION_SOFTMAX: i32 = 4;
/// `ACTIVATION_SWISH`.
pub const ACTIVATION_SWISH: i32 = 5;
/// `ACTIVATION_EXP`.
pub const ACTIVATION_EXP: i32 = 6;

/// `WEIGHT_BLOB_VERSION`.
pub const WEIGHT_BLOB_VERSION: i32 = 0;
/// `WEIGHT_BLOCK_SIZE`: size of a record header, and the alignment of record payloads.
pub const WEIGHT_BLOCK_SIZE: usize = 64;

/// `WEIGHT_TYPE_float`.
pub const WEIGHT_TYPE_FLOAT: i32 = 0;
/// `WEIGHT_TYPE_int`.
pub const WEIGHT_TYPE_INT: i32 = 1;
/// `WEIGHT_TYPE_qweight`.
pub const WEIGHT_TYPE_QWEIGHT: i32 = 2;
/// `WEIGHT_TYPE_int8`.
pub const WEIGHT_TYPE_INT8: i32 = 3;

/// `MAX_RNN_NEURONS_ALL` (nnet.c) for the 1.6.1 models:
/// `max(FARGAN_MAX_RNN_NEURONS=160, PLC_MAX_RNN_UNITS=192, DRED_MAX_RNN_NEURONS=64,
/// OSCE_MAX_RNN_NEURONS=160, OSCE_BWE_MAX_RNN_NEURONS=128)`.
pub const MAX_RNN_NEURONS_ALL: usize = 192;

/// `MAX_CONV_INPUTS_ALL` (nnet.c): `IMAX(DRED_MAX_CONV_INPUTS=128, 1024)`.
pub const MAX_CONV_INPUTS_ALL: usize = 1024;

/// `LinearLayer`: generic (optionally sparse, optionally int8-quantized) affine transform.
///
/// C keeps pointers into the weight blob / compiled-in arrays; the port owns copies (blob
/// payloads are not aligned for `f32`/`i32`). `None` is C `NULL`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearLayer {
    pub bias: Option<Vec<f32>>,
    pub subias: Option<Vec<f32>>,
    pub weights: Option<Vec<i8>>,
    pub float_weights: Option<Vec<f32>>,
    pub weights_idx: Option<Vec<i32>>,
    pub diag: Option<Vec<f32>>,
    pub scale: Option<Vec<f32>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

/// `Conv2dLayer`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conv2dLayer {
    pub bias: Option<Vec<f32>>,
    pub float_weights: Option<Vec<f32>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

/// Port of dnn/nnet.c:compute_generic_dense.
pub fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
) {
    compute_linear(layer, output, input);
    compute_activation_inplace(output, layer.nb_outputs, activation);
}

/// Port of dnn/nnet.c:compute_generic_gru. `state` holds `recurrent_weights.nb_inputs` floats
/// and is updated; `input` holds `input_weights.nb_inputs` floats.
pub fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
) {
    let mut zrh = [0f32; 3 * MAX_RNN_NEURONS_ALL];
    let mut recur = [0f32; 3 * MAX_RNN_NEURONS_ALL];
    debug_assert!(3 * recurrent_weights.nb_inputs == recurrent_weights.nb_outputs);
    debug_assert!(input_weights.nb_outputs == recurrent_weights.nb_outputs);
    let n = recurrent_weights.nb_inputs;
    debug_assert!(recurrent_weights.nb_outputs <= 3 * MAX_RNN_NEURONS_ALL);
    let zrh = &mut zrh[..3 * n];
    let recur = &mut recur[..3 * n];
    let state = &mut state[..n];
    compute_linear(input_weights, zrh, input);
    compute_linear(recurrent_weights, recur, state);
    for (a, &b) in zrh[..2 * n].iter_mut().zip(&recur[..2 * n]) {
        *a += b;
    }
    compute_activation_inplace(zrh, 2 * n, ACTIVATION_SIGMOID);
    let (zr, h) = zrh.split_at_mut(2 * n);
    let (z, r) = zr.split_at(n);
    let recur_h = &recur[2 * n..];
    for i in 0..n {
        h[i] += recur_h[i] * r[i];
    }
    compute_activation_inplace(h, n, ACTIVATION_TANH);
    for i in 0..n {
        h[i] = z[i] * state[i] + (1.0 - z[i]) * h[i];
    }
    state.copy_from_slice(h);
}

/// Port of dnn/nnet.c:compute_glu (`output != input`).
pub fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32]) {
    let mut act2 = [0f32; MAX_INPUTS];
    debug_assert!(layer.nb_inputs == layer.nb_outputs);
    let n = layer.nb_outputs;
    let act2 = &mut act2[..n];
    compute_linear(layer, act2, input);
    compute_activation_inplace(act2, n, ACTIVATION_SIGMOID);
    for ((o, &i), &a) in output[..n].iter_mut().zip(&input[..n]).zip(act2.iter()) {
        *o = i * a;
    }
}

/// Port of dnn/nnet.c:compute_glu for the in-place calls (`output == input`).
pub fn compute_glu_inplace(layer: &LinearLayer, x: &mut [f32]) {
    let mut act2 = [0f32; MAX_INPUTS];
    debug_assert!(layer.nb_inputs == layer.nb_outputs);
    let n = layer.nb_outputs;
    let act2 = &mut act2[..n];
    compute_linear(layer, act2, x);
    compute_activation_inplace(act2, n, ACTIVATION_SIGMOID);
    for (o, &a) in x[..n].iter_mut().zip(act2.iter()) {
        *o *= a;
    }
}

// `compute_gated_activation` is declared in nnet.h but has no definition in libopus 1.6.1 (only
// the unbuilt fwgan.c calls it), so there is nothing to port.

/// Port of dnn/nnet.c:compute_generic_conv1d. `mem` holds `nb_inputs-input_size` floats of
/// history (updated); `input` holds `input_size` floats.
pub fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: i32,
) {
    let mut tmp = [0f32; MAX_CONV_INPUTS_ALL];
    debug_assert!(layer.nb_inputs <= MAX_CONV_INPUTS_ALL);
    let nb_inputs = layer.nb_inputs;
    let hist = nb_inputs - input_size;
    let tmp = &mut tmp[..nb_inputs];
    if nb_inputs != input_size {
        tmp[..hist].copy_from_slice(&mem[..hist]);
    }
    tmp[hist..].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, tmp);
    compute_activation_inplace(output, layer.nb_outputs, activation);
    if nb_inputs != input_size {
        mem[..hist].copy_from_slice(&tmp[input_size..]);
    }
}

/// Port of dnn/nnet.c:compute_generic_conv1d_dilation. `mem` holds
/// `input_size*dilation*(ksize-1)` floats of history (updated) for `dilation > 1`, where
/// `ksize = nb_inputs/input_size`.
pub fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: i32,
) {
    let mut tmp = [0f32; MAX_CONV_INPUTS_ALL];
    let nb_inputs = layer.nb_inputs;
    let ksize = nb_inputs / input_size;
    debug_assert!(nb_inputs <= MAX_CONV_INPUTS_ALL);
    let tmp = &mut tmp[..nb_inputs];
    let hist = nb_inputs - input_size;
    if dilation == 1 {
        tmp[..hist].copy_from_slice(&mem[..hist]);
    } else {
        for i in 0..ksize - 1 {
            tmp[i * input_size..(i + 1) * input_size]
                .copy_from_slice(&mem[i * input_size * dilation..][..input_size]);
        }
    }
    tmp[hist..].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, tmp);
    compute_activation_inplace(output, layer.nb_outputs, activation);
    if dilation == 1 {
        mem[..hist].copy_from_slice(&tmp[input_size..]);
    } else {
        let len = input_size * dilation * (ksize - 1) - input_size;
        // C: OPUS_COPY(mem, &mem[input_size], len) on overlapping ranges (dst < src); the
        // intended (and observed) result is a memmove.
        mem.copy_within(input_size..input_size + len, 0);
        mem[len..len + input_size].copy_from_slice(&input[..input_size]);
    }
}

//! Port of `dnn/vec.h`: dense/sparse matrix-vector products and the activation approximations.
//!
//! Only the generic C path is ported (`vec.h` without `vec_avx.h` / `vec_neon.h`); the oracle is
//! built with that path forced (`DISABLE_NEON`, `__SSE2__`/`__AVX__` undefined). In that path
//! `USE_SU_BIAS` is never defined, so the unsigned-input variants of `cgemv8x4` /
//! `sparse_cgemv8x4` are not ported (`// USE_SU_BIAS: not ported (generic path)`).

use crate::celt::arch::{max32, min32};
use crate::math;

/// `MAX_INPUTS`: maximum number of inputs of a quantized (int8) matrix product.
pub const MAX_INPUTS: usize = 2048;

/// `SCALE` (`128.f*127.f`).
pub const SCALE: f32 = 128.0 * 127.0;
/// `SCALE_1` (`1.f/128.f/127.f`).
pub const SCALE_1: f32 = 1.0 / 128.0 / 127.0;

/// Port of dnn/vec.h:sgemv16x1. `weights` is column-major with stride `col_stride`.
pub fn sgemv16x1(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let out = &mut out[..rows];
    let x = &x[..cols];
    out.fill(0.0);
    // SIMD (bit-identical: same multiply and add per lane), else the scalar loop below.
    if super::vec_simd::sgemv_blocks::<4>(out, weights, rows, cols, col_stride, x) {
        return;
    }
    let mut i = 0;
    while i < rows {
        let y = &mut out[i..i + 16];
        for (j, &xj) in x.iter().enumerate() {
            let w = &weights[j * col_stride + i..][..16];
            for k in 0..16 {
                y[k] += w[k] * xj;
            }
        }
        i += 16;
    }
}

/// Port of dnn/vec.h:sgemv8x1.
pub fn sgemv8x1(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let out = &mut out[..rows];
    let x = &x[..cols];
    out.fill(0.0);
    // SIMD (bit-identical: same multiply and add per lane), else the scalar loop below.
    if super::vec_simd::sgemv_blocks::<2>(out, weights, rows, cols, col_stride, x) {
        return;
    }
    let mut i = 0;
    while i < rows {
        let y = &mut out[i..i + 8];
        for (j, &xj) in x.iter().enumerate() {
            let w = &weights[j * col_stride + i..][..8];
            for k in 0..8 {
                y[k] += w[k] * xj;
            }
        }
        i += 8;
    }
}

/// Port of dnn/vec.h:sgemv.
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    if (rows & 0xf) == 0 {
        sgemv16x1(out, weights, rows, cols, col_stride, x);
    } else if (rows & 0x7) == 0 {
        sgemv8x1(out, weights, rows, cols, col_stride, x);
    } else {
        let x = &x[..cols];
        for (i, o) in out[..rows].iter_mut().enumerate() {
            *o = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                *o += weights[j * col_stride + i] * xj;
            }
        }
    }
}

/// Port of dnn/vec.h:sparse_sgemv8x4. `idx` holds, per block of 8 rows, the number of 4-column
/// blocks followed by their starting columns; `w` holds 32 weights per block.
pub fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    let out = &mut out[..rows];
    out.fill(0.0);
    let mut ip = 0usize;
    let mut wp = 0usize;
    let mut i = 0;
    while i < rows {
        let cols = idx[ip];
        ip += 1;
        let y = &mut out[i..i + 8];
        for _ in 0..cols {
            let pos = idx[ip] as usize;
            ip += 1;
            let xj = &x[pos..pos + 4];
            let (xj0, xj1, xj2, xj3) = (xj[0], xj[1], xj[2], xj[3]);
            let wb = &w[wp..wp + 32];
            for k in 0..8 {
                y[k] += wb[k] * xj0;
            }
            for k in 0..8 {
                y[k] += wb[8 + k] * xj1;
            }
            for k in 0..8 {
                y[k] += wb[16 + k] * xj2;
            }
            for k in 0..8 {
                y[k] += wb[24 + k] * xj3;
            }
            wp += 32;
        }
        i += 8;
    }
}

/// Input quantization shared by the int8 products: `x[i] = (int)floor(.5+127*_x[i])` stored in
/// an `opus_int8` (the int → int8 conversion wraps, as gcc does).
#[inline(always)]
fn quantize_input(xq: &mut [i8], x: &[f32]) {
    for (q, &v) in xq.iter_mut().zip(x) {
        *q = math::floor(0.5 + (127.0f32 * v) as f64) as i32 as i8;
    }
}

// USE_SU_BIAS: not ported (generic path) — unsigned-input sparse_cgemv8x4 / cgemv8x4.

/// Port of dnn/vec.h:sparse_cgemv8x4 (signed-input variant). The per-block dot products are
/// computed in `int` before being added to the float output.
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    let mut xq = [0i8; MAX_INPUTS];
    let xq = &mut xq[..cols];
    let out = &mut out[..rows];
    out.fill(0.0);
    quantize_input(xq, &x[..cols]);
    // SIMD (bit-identical, see `vec_simd`), else the scalar loop below.
    if !super::vec_simd::sparse_cgemv8x4(out, w, idx, rows, xq) {
        sparse_cgemv8x4_blocks(out, w, idx, rows, xq);
    }
    for (o, &s) in out.iter_mut().zip(&scale[..rows]) {
        *o *= s;
    }
}

/// The block loop of [`sparse_cgemv8x4`] (scalar).
fn sparse_cgemv8x4_blocks(out: &mut [f32], w: &[i8], idx: &[i32], rows: usize, xq: &[i8]) {
    let mut ip = 0usize;
    let mut wp = 0usize;
    let mut i = 0;
    while i < rows {
        let colblocks = idx[ip];
        ip += 1;
        let y = &mut out[i..i + 8];
        for _ in 0..colblocks {
            let pos = idx[ip] as usize;
            ip += 1;
            let xj = &xq[pos..pos + 4];
            let (xj0, xj1, xj2, xj3) = (
                i32::from(xj[0]),
                i32::from(xj[1]),
                i32::from(xj[2]),
                i32::from(xj[3]),
            );
            let wb = &w[wp..wp + 32];
            for k in 0..8 {
                let wk = &wb[4 * k..4 * k + 4];
                y[k] += (i32::from(wk[0]) * xj0
                    + i32::from(wk[1]) * xj1
                    + i32::from(wk[2]) * xj2
                    + i32::from(wk[3]) * xj3) as f32;
            }
            wp += 32;
        }
        i += 8;
    }
}

/// Port of dnn/vec.h:cgemv8x4 (signed-input variant). Here the quantized inputs are converted
/// back to float and the per-block dot products are float sums.
pub fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
    let mut xq = [0i8; MAX_INPUTS];
    let xq = &mut xq[..cols];
    let out = &mut out[..rows];
    out.fill(0.0);
    quantize_input(xq, &x[..cols]);
    // SIMD (bit-identical, see `vec_simd`), else the scalar loop below.
    if !super::vec_simd::cgemv8x4(out, w, rows, cols, xq) {
        cgemv8x4_blocks(out, w, rows, cols, xq);
    }
    for (o, &s) in out.iter_mut().zip(&scale[..rows]) {
        *o *= s;
    }
}

/// The block loop of [`cgemv8x4`] (scalar).
fn cgemv8x4_blocks(out: &mut [f32], w: &[i8], rows: usize, cols: usize, xq: &[i8]) {
    let mut wp = 0usize;
    let mut i = 0;
    while i < rows {
        let y = &mut out[i..i + 8];
        let mut j = 0;
        while j < cols {
            let xj = &xq[j..j + 4];
            let (xj0, xj1, xj2, xj3) = (
                f32::from(xj[0]),
                f32::from(xj[1]),
                f32::from(xj[2]),
                f32::from(xj[3]),
            );
            let wb = &w[wp..wp + 32];
            for k in 0..8 {
                let wk = &wb[4 * k..4 * k + 4];
                y[k] += f32::from(wk[0]) * xj0
                    + f32::from(wk[1]) * xj1
                    + f32::from(wk[2]) * xj2
                    + f32::from(wk[3]) * xj3;
            }
            wp += 32;
            j += 4;
        }
        i += 8;
    }
}

/// Port of dnn/vec.h:lpcnet_exp2 (generic path).
#[inline(always)]
#[must_use]
pub fn lpcnet_exp2(x: f32) -> f32 {
    // C: integer = floor(x) (double floor, truncating int conversion).
    let integer = math::floor(x as f64) as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    // K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2
    let res =
        0.99992522f32 + frac * (0.69583354f32 + frac * (0.22606716f32 + 0.078024523f32 * frac));
    // C: res.i = (res.i + (integer<<23)) & 0x7fffffff (unsigned wrap).
    f32::from_bits(res.to_bits().wrapping_add((integer << 23) as u32) & 0x7fff_ffff)
}

/// Port of dnn/vec.h:lpcnet_exp (`lpcnet_exp2((x)*1.44269504f)`).
#[inline(always)]
#[must_use]
pub fn lpcnet_exp(x: f32) -> f32 {
    lpcnet_exp2(x * 1.44269504f32)
}

/// Port of dnn/vec.h:tanh_approx (generic path; `fmadd(a,b,c)` is `a*b+c`, never fused).
#[inline(always)]
#[must_use]
pub fn tanh_approx(x: f32) -> f32 {
    const N0: f32 = 952.52801514;
    const N1: f32 = 96.39235687;
    const N2: f32 = 0.60863042;
    const D0: f32 = 952.72399902;
    const D1: f32 = 413.36801147;
    const D2: f32 = 11.88600922;
    let x2 = x * x;
    let mut num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    num = num * x / den;
    max32(-1.0, min32(1.0, num))
}

/// Port of dnn/vec.h:sigmoid_approx.
#[inline(always)]
#[must_use]
pub fn sigmoid_approx(x: f32) -> f32 {
    0.5f32 + 0.5f32 * tanh_approx(0.5f32 * x)
}

/// Port of dnn/vec.h:softmax (no normalization: `y[i] = lpcnet_exp(x[i])`).
pub fn softmax(y: &mut [f32], x: &[f32], n: usize) {
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = lpcnet_exp(xi);
    }
}

/// In-place [`softmax`] (C calls it with `y == x`).
pub fn softmax_inplace(y: &mut [f32], n: usize) {
    for v in &mut y[..n] {
        *v = lpcnet_exp(*v);
    }
}

/// Port of dnn/vec.h:vec_tanh.
pub fn vec_tanh(y: &mut [f32], x: &[f32], n: usize) {
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = tanh_approx(xi);
    }
}

/// In-place [`vec_tanh`].
pub fn vec_tanh_inplace(y: &mut [f32], n: usize) {
    for v in &mut y[..n] {
        *v = tanh_approx(*v);
    }
}

/// Port of dnn/vec.h:vec_sigmoid.
pub fn vec_sigmoid(y: &mut [f32], x: &[f32], n: usize) {
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = sigmoid_approx(xi);
    }
}

/// In-place [`vec_sigmoid`].
pub fn vec_sigmoid_inplace(y: &mut [f32], n: usize) {
    for v in &mut y[..n] {
        *v = sigmoid_approx(*v);
    }
}

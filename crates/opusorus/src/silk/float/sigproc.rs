//! Port of `silk/float/SigProc_FLP.h` (inline helpers) and the leaf floating-point signal
//! processing files of `silk/float/`: `apply_sine_window_FLP.c`, `autocorrelation_FLP.c`,
//! `burg_modified_FLP.c`, `bwexpander_FLP.c`, `corrMatrix_FLP.c`, `energy_FLP.c`,
//! `inner_product_FLP.c`, `k2a_FLP.c`, `LPC_analysis_filter_FLP.c`, `LPC_inv_pred_gain_FLP.c`,
//! `LTP_analysis_filter_FLP.c`, `regularize_correlations_FLP.c`, `residual_energy_FLP.c`,
//! `scale_copy_vector_FLP.c`, `scale_vector_FLP.c`, `schur_FLP.c`, `sort_FLP.c` and
//! `warped_autocorrelation_FLP.c`.
//!
//! C float/double promotions are reproduced exactly: `float * float` stays in `f32`, anything
//! touching a `double` (or a `double` libm call) is computed in `f64`.

use crate::celt::mathops::float2int;
use crate::math;
use crate::silk::define::{LTP_ORDER, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER};
use crate::silk::macros::{SILK_MAX_ORDER_LPC, silk_sat16};
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

const LTPO: usize = LTP_ORDER as usize;
const MLPC: usize = MAX_LPC_ORDER as usize;
const MNSF: usize = MAX_NB_SUBFR as usize;

// ---------------------------------------------------------------------------------------------
// SigProc_FLP.h
// ---------------------------------------------------------------------------------------------

/// `PI` from `SigProc_FLP.h` (a `float` constant, `3.1415926536f`).
pub const PI_FLP: f32 = 3.1415926536;

/// `silk_min_float` (C ternary semantics).
#[inline(always)]
#[must_use]
pub const fn silk_min_float(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `silk_max_float` (C ternary semantics).
#[inline(always)]
#[must_use]
pub const fn silk_max_float(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `silk_abs_float`: `(silk_float)fabs(a)`.
#[inline(always)]
#[must_use]
pub const fn silk_abs_float(a: f32) -> f32 {
    a.abs()
}

/// Port of `SigProc_FLP.h:silk_sigmoid`: `(float)(1.0 / (1.0 + exp(-x)))` in double.
#[inline]
#[must_use]
pub fn silk_sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + math::exp(-x as f64))) as f32
}

/// Port of `SigProc_FLP.h:silk_float2int` (round to nearest, ties to even).
#[inline(always)]
#[must_use]
pub fn silk_float2int(x: f32) -> i32 {
    float2int(x)
}

/// Port of `SigProc_FLP.h:silk_float2short_array` (rounding + saturation to 16 bits).
pub fn silk_float2short_array(out: &mut [i16], input: &[f32], length: usize) {
    let out = &mut out[..length];
    let input = &input[..length];
    for k in (0..length).rev() {
        out[k] = silk_sat16(float2int(input[k])) as i16;
    }
}

/// Port of `SigProc_FLP.h:silk_short2float_array`.
pub fn silk_short2float_array(out: &mut [f32], input: &[i16], length: usize) {
    let out = &mut out[..length];
    let input = &input[..length];
    for k in (0..length).rev() {
        out[k] = input[k] as f32;
    }
}

/// Port of `SigProc_FLP.h:silk_log2`: `(float)(3.32192809488736 * log10(x))`.
#[inline]
#[must_use]
pub fn silk_log2(x: f64) -> f32 {
    (3.32192809488736 * math::log10(x)) as f32
}

// ---------------------------------------------------------------------------------------------
// inner_product_FLP.c / energy_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/inner_product_FLP.c:silk_inner_product_FLP_c` — inner product of two
/// `float` arrays, accumulated in `double` (4x unrolled as in C, which fixes the summation
/// order).
#[must_use]
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32], data_size: usize) -> f64 {
    let d1 = &data1[..data_size];
    let d2 = &data2[..data_size];
    let mut result = 0.0f64;
    let mut i = 0usize;
    // C: for( i = 0; i < dataSize - 3; i += 4 )
    while i + 3 < data_size {
        result += d1[i] as f64 * d2[i] as f64
            + d1[i + 1] as f64 * d2[i + 1] as f64
            + d1[i + 2] as f64 * d2[i + 2] as f64
            + d1[i + 3] as f64 * d2[i + 3] as f64;
        i += 4;
    }
    while i < data_size {
        result += d1[i] as f64 * d2[i] as f64;
        i += 1;
    }
    result
}

/// Port of `silk/float/energy_FLP.c:silk_energy_FLP` — sum of squares of a `float` array,
/// accumulated in `double`.
#[must_use]
pub fn silk_energy_flp(data: &[f32], data_size: usize) -> f64 {
    let d = &data[..data_size];
    let mut result = 0.0f64;
    let mut i = 0usize;
    while i + 3 < data_size {
        result += d[i] as f64 * d[i] as f64
            + d[i + 1] as f64 * d[i + 1] as f64
            + d[i + 2] as f64 * d[i + 2] as f64
            + d[i + 3] as f64 * d[i + 3] as f64;
        i += 4;
    }
    while i < data_size {
        result += d[i] as f64 * d[i] as f64;
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------------------------
// autocorrelation_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/autocorrelation_FLP.c:silk_autocorrelation_FLP`.
pub fn silk_autocorrelation_flp(
    results: &mut [f32],
    input_data: &[f32],
    input_data_size: usize,
    mut correlation_count: usize,
) {
    if correlation_count > input_data_size {
        correlation_count = input_data_size;
    }
    for i in 0..correlation_count {
        results[i] =
            silk_inner_product_flp(input_data, &input_data[i..], input_data_size - i) as f32;
    }
}

// ---------------------------------------------------------------------------------------------
// apply_sine_window_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/apply_sine_window_FLP.c:silk_apply_sine_window_FLP`.
///
/// Window types: 1 → sine window from 0 to pi/2, 2 → sine window from pi/2 to pi. `length`
/// must be a multiple of 4.
pub fn silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    celt_assert!(win_type == 1 || win_type == 2);
    celt_assert!(length & 3 == 0);
    let px_win = &mut px_win[..length];
    let px = &px[..length];

    let freq = PI_FLP / (length as i32 + 1) as f32;
    // Approximation of 2 * cos(f)
    let c = 2.0f32 - freq * freq;

    let (mut s0, mut s1);
    if win_type < 2 {
        // Start from 0
        s0 = 0.0f32;
        // Approximation of sin(f)
        s1 = freq;
    } else {
        // Start from 1
        s0 = 1.0f32;
        // Approximation of cos(f)
        s1 = 0.5f32 * c;
    }

    // Uses the recursive equation: sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)
    // 4 samples at a time
    let mut k = 0usize;
    while k < length {
        px_win[k] = px[k] * 0.5f32 * (s0 + s1);
        px_win[k + 1] = px[k + 1] * s1;
        s0 = c * s1 - s0;
        px_win[k + 2] = px[k + 2] * 0.5f32 * (s1 + s0);
        px_win[k + 3] = px[k + 3] * s0;
        s1 = c * s0 - s1;
        k += 4;
    }
}

// ---------------------------------------------------------------------------------------------
// burg_modified_FLP.c
// ---------------------------------------------------------------------------------------------

/// `MAX_FRAME_SIZE` in `burg_modified_FLP.c`: `subfr_length * nb_subfr = (0.005 * 16000 + 16) * 4`.
const BURG_MAX_FRAME_SIZE: usize = 384;

/// Port of `silk/float/burg_modified_FLP.c:silk_burg_modified_FLP` — compute reflection
/// coefficients from input signal. Returns the residual energy.
///
/// `x` holds `nb_subfr` stacked subframes of `subfr_length` samples (each including `d`
/// preceding samples); `a` receives `d` prediction coefficients.
#[allow(
    clippy::too_many_lines,
    reason = "one C function; splitting it would obscure the correspondence"
)]
pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    const SMO: usize = SILK_MAX_ORDER_LPC;
    let mut c_first_row = [0f64; SMO];
    let mut caf = [0f64; SMO + 1];
    let mut cab = [0f64; SMO + 1];
    let mut af = [0f64; SMO];
    let sl = subfr_length;

    celt_assert!(subfr_length * nb_subfr <= BURG_MAX_FRAME_SIZE);
    celt_assert!(d <= SMO);
    let x = &x[..nb_subfr * sl];

    // Compute autocorrelations, added over subframes
    let mut c0 = silk_energy_flp(x, nb_subfr * sl);
    for s in 0..nb_subfr {
        let x_ptr = &x[s * sl..(s + 1) * sl];
        for n in 1..d + 1 {
            c_first_row[n - 1] += silk_inner_product_flp(x_ptr, &x_ptr[n..], sl - n);
        }
    }
    let mut c_last_row = c_first_row;

    // Initialize
    caf[0] = c0 + FIND_LPC_COND_FAC as f64 * c0 + 1e-9f32 as f64;
    cab[0] = caf[0];
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;
    let mut tmp1: f64;
    let mut tmp2: f64;
    let mut n = 0usize;
    while n < d {
        // Update first row of correlation matrix (without first element)
        // Update last row of correlation matrix (without last element, stored in reversed order)
        // Update C * Af
        // Update C * flipud(Af) (stored in reversed order)
        // Perf: the loops over `k` walk `x_ptr[n - k - 1]` / `x_ptr[n - k]` backwards and
        // `x_ptr[sl - n + k]` / `x_ptr[sl - n + k - 1]` forwards with iterators (no bounds
        // checks); same operations in the same order.
        for s in 0..nb_subfr {
            let x_ptr = &x[s * sl..(s + 1) * sl];
            let x_n = x_ptr[n];
            let x_e = x_ptr[sl - n - 1];
            tmp1 = x_n as f64;
            tmp2 = x_e as f64;
            for ((((cf, cl), &atmp), &xa), &xb) in c_first_row[..n]
                .iter_mut()
                .zip(&mut c_last_row[..n])
                .zip(&af[..n])
                .zip(x_ptr[..n].iter().rev())
                .zip(&x_ptr[sl - n..])
            {
                // xa = x_ptr[n - k - 1], xb = x_ptr[sl - n + k]
                *cf -= (x_n * xa) as f64;
                *cl -= (x_e * xb) as f64;
                tmp1 += xa as f64 * atmp;
                tmp2 += xb as f64 * atmp;
            }
            for (((cf, cb), &xa), &xb) in caf[..=n]
                .iter_mut()
                .zip(&mut cab[..=n])
                .zip(x_ptr[..=n].iter().rev())
                .zip(&x_ptr[sl - n - 1..])
            {
                // xa = x_ptr[n - k], xb = x_ptr[sl - n + k - 1]
                *cf -= tmp1 * xa as f64;
                *cb -= tmp2 * xb as f64;
            }
        }
        tmp1 = c_first_row[n];
        tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        // Calculate nominator and denominator for the next order reflection (parcor) coefficient
        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        // Calculate the next order reflection (parcor) coefficient
        let mut rc = -2.0 * num / (nrg_f + nrg_b);

        // Update inverse prediction gain
        tmp1 = inv_gain * (1.0 - rc * rc);
        if tmp1 <= min_inv_gain as f64 {
            // Max prediction gain exceeded; set reflection coefficient such that max prediction
            // gain is exactly hit
            rc = math::sqrt(1.0 - min_inv_gain as f64 / inv_gain);
            if num > 0.0 {
                // Ensure adjusted reflection coefficients has the original sign
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1;
        }

        // Update the AR coefficients
        for k in 0..(n + 1) >> 1 {
            tmp1 = af[k];
            tmp2 = af[n - k - 1];
            af[k] = tmp1 + rc * tmp2;
            af[n - k - 1] = tmp2 + rc * tmp1;
        }
        af[n] = rc;

        if reached_max_gain {
            // Reached max prediction gain; set remaining coefficients to zero and exit loop
            for v in &mut af[n + 1..d] {
                *v = 0.0;
            }
            break;
        }

        // Update C * Af and C * Ab
        for k in 0..=n + 1 {
            tmp1 = caf[k];
            caf[k] += rc * cab[n + 1 - k];
            cab[n + 1 - k] += rc * tmp1;
        }
        n += 1;
    }

    let nrg_f = if reached_max_gain {
        // Convert to silk_float
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        // Subtract energy of preceding samples from C0
        for s in 0..nb_subfr {
            c0 -= silk_energy_flp(&x[s * sl..], d);
        }
        // Approximate residual energy
        c0 * inv_gain
    } else {
        // Compute residual energy and store coefficients as silk_float
        let mut nf = caf[0];
        tmp1 = 1.0;
        for k in 0..d {
            let atmp = af[k];
            nf += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        nf -= FIND_LPC_COND_FAC as f64 * c0 * tmp1;
        nf
    };

    // Return residual energy
    nrg_f as f32
}

// ---------------------------------------------------------------------------------------------
// bwexpander_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/bwexpander_FLP.c:silk_bwexpander_FLP` — chirp (bandwidth expand) an LP
/// AR filter (without leading 1).
pub fn silk_bwexpander_flp(ar: &mut [f32], d: usize, chirp: f32) {
    let ar = &mut ar[..d];
    let mut cfac = chirp;
    for v in &mut ar[..d - 1] {
        *v *= cfac;
        cfac *= chirp;
    }
    ar[d - 1] *= cfac;
}

// ---------------------------------------------------------------------------------------------
// corrMatrix_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/corrMatrix_FLP.c:silk_corrVector_FLP` — correlation vector `X'*t`.
///
/// `x` is the `[L + order - 1]` vector used to create `X`; `t` the `[L]` target.
pub fn silk_corr_vector_flp(x: &[f32], t: &[f32], l: usize, order: usize, xt: &mut [f32]) {
    // ptr1 points to the first sample of column 0 of X: X[:,0]
    for lag in 0..order {
        // Calculate X[:,lag]'*t
        xt[lag] = silk_inner_product_flp(&x[order - 1 - lag..], t, l) as f32;
    }
}

/// Port of `silk/float/corrMatrix_FLP.c:silk_corrMatrix_FLP` — correlation matrix `X'*X`
/// (`[order x order]`, row-major).
pub fn silk_corr_matrix_flp(x: &[f32], l: usize, order: usize, xx: &mut [f32]) {
    let p1 = order - 1; // First sample of column 0 of X
    let x = &x[..l + order - 1];
    let mut energy = silk_energy_flp(&x[p1..], l); // X[:,0]'*X[:,0]
    xx[0] = energy as f32;
    for j in 1..order {
        // Calculate X[:,j]'*X[:,j]
        energy += (x[p1 - j] * x[p1 - j] - x[p1 + l - j] * x[p1 + l - j]) as f64;
        xx[j * order + j] = energy as f32;
    }

    for lag in 1..order {
        // ptr2 = &x[ Order - 2 ] (first sample of column 1 of X), moved back once per lag
        let p2 = p1 - lag;
        // Calculate X[:,0]'*X[:,lag]
        energy = silk_inner_product_flp(&x[p1..], &x[p2..], l);
        xx[lag * order] = energy as f32;
        xx[lag] = energy as f32;
        // Calculate X[:,j]'*X[:,j + lag]
        for j in 1..(order - lag) {
            energy += (x[p1 - j] * x[p2 - j] - x[p1 + l - j] * x[p2 + l - j]) as f64;
            xx[(lag + j) * order + j] = energy as f32;
            xx[j * order + lag + j] = energy as f32;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// k2a_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/k2a_FLP.c:silk_k2a_FLP` — step up function, converts reflection
/// coefficients to prediction coefficients.
pub fn silk_k2a_flp(a: &mut [f32], rc: &[f32], order: usize) {
    let a = &mut a[..order];
    for k in 0..order {
        let rck = rc[k];
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

// ---------------------------------------------------------------------------------------------
// LPC_analysis_filter_FLP.c
// ---------------------------------------------------------------------------------------------

/// Ports of the static `silk_LPC_analysis_filter{6,8,10,12,16}_FLP` helpers: an order-`N` LPC
/// analysis filter that does not write the first `N` samples. The C versions spell the
/// prediction sum out term by term (left to right); the loop below adds in the same order.
#[inline(always)]
fn lpc_analysis_filter_n<const N: usize>(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
) {
    let pred_coef = &pred_coef[..N];
    let s = &s[..length];
    let r_lpc = &mut r_lpc[..length];
    for ix in N..length {
        let s_ptr = &s[ix - N..ix];
        // short-term prediction: s_ptr[0] * PredCoef[0] + s_ptr[-1] * PredCoef[1] + ...
        let mut lpc_pred = s_ptr[N - 1] * pred_coef[0];
        for j in 1..N {
            lpc_pred += s_ptr[N - 1 - j] * pred_coef[j];
        }
        // prediction error
        r_lpc[ix] = s[ix] - lpc_pred;
    }
}

/// Port of `silk/float/LPC_analysis_filter_FLP.c:silk_LPC_analysis_filter_FLP` — LPC analysis
/// filter (orders 6, 8, 10, 12, 16). The first `order` output samples are set to zero.
pub fn silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
    order: usize,
) {
    celt_assert!(order <= length);
    match order {
        6 => lpc_analysis_filter_n::<6>(r_lpc, pred_coef, s, length),
        8 => lpc_analysis_filter_n::<8>(r_lpc, pred_coef, s, length),
        10 => lpc_analysis_filter_n::<10>(r_lpc, pred_coef, s, length),
        12 => lpc_analysis_filter_n::<12>(r_lpc, pred_coef, s, length),
        16 => lpc_analysis_filter_n::<16>(r_lpc, pred_coef, s, length),
        // C: celt_assert( 0 ) (only the memset below runs in a build without assertions).
        _ => celt_assert!(false, "unsupported LPC order {order}"),
    }
    // Set first Order output samples to zero
    for v in &mut r_lpc[..order] {
        *v = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// LPC_inv_pred_gain_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/LPC_inv_pred_gain_FLP.c:silk_LPC_inverse_pred_gain_FLP` — compute the
/// inverse of the LPC prediction gain (energy domain), and test if the LPC coefficients are
/// stable (returns 0 if not).
#[must_use]
pub fn silk_lpc_inverse_pred_gain_flp(a: &[f32], order: usize) -> f32 {
    use crate::silk::define::MAX_PREDICTION_POWER_GAIN;
    let mut atmp = [0f32; SILK_MAX_ORDER_LPC];
    atmp[..order].copy_from_slice(&a[..order]);

    let mut inv_gain = 1.0f64;
    let mut k = order - 1;
    while k > 0 {
        let rc = -atmp[k] as f64;
        let rc_mult1 = 1.0 - rc * rc;
        inv_gain *= rc_mult1;
        if inv_gain * (MAX_PREDICTION_POWER_GAIN as f64) < 1.0 {
            return 0.0;
        }
        let rc_mult2 = 1.0 / rc_mult1;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = atmp[n] as f64;
            let tmp2 = atmp[k - n - 1] as f64;
            atmp[n] = ((tmp1 - tmp2 * rc) * rc_mult2) as f32;
            atmp[k - n - 1] = ((tmp2 - tmp1 * rc) * rc_mult2) as f32;
        }
        k -= 1;
    }
    let rc = -atmp[0] as f64;
    let rc_mult1 = 1.0 - rc * rc;
    inv_gain *= rc_mult1;
    if inv_gain * (MAX_PREDICTION_POWER_GAIN as f64) < 1.0 {
        return 0.0;
    }
    inv_gain as f32
}

// ---------------------------------------------------------------------------------------------
// LTP_analysis_filter_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/LTP_analysis_filter_FLP.c:silk_LTP_analysis_filter_FLP`.
///
/// The C input pointer `x` is `&x_buf[x_off]`; the filter reads up to `pitchL[k] + 2` samples
/// before it, so the whole buffer is passed. `ltp_res` receives
/// `nb_subfr * (pre_length + subfr_length)` samples.
pub fn silk_ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x_buf: &[f32],
    x_off: usize,
    b: &[f32],
    pitch_l: &[i32],
    inv_gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut btmp = [0f32; LTPO];
    let len = subfr_length + pre_length;
    for k in 0..nb_subfr {
        let x_ptr = x_off + k * subfr_length;
        // x_lag_ptr = x_ptr - pitchL[ k ]; C reads x_lag_ptr[ LTP_ORDER / 2 - j ]
        let x_lag = x_ptr - pitch_l[k] as usize;
        let inv_gain = inv_gains[k];
        btmp.copy_from_slice(&b[k * LTPO..(k + 1) * LTPO]);
        let res = &mut ltp_res[k * len..(k + 1) * len];
        let xs = &x_buf[x_ptr..x_ptr + len];
        let xl = &x_buf[x_lag - LTPO / 2..x_lag + len + LTPO / 2];

        // LTP analysis FIR filter
        for i in 0..len {
            let mut r = xs[i];
            // Subtract long-term prediction
            for j in 0..LTPO {
                // x_lag_ptr[ LTP_ORDER / 2 - j ] with x_lag_ptr advanced by i
                r -= btmp[j] * xl[i + LTPO - 1 - j];
            }
            r *= inv_gain;
            res[i] = r;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// regularize_correlations_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/regularize_correlations_FLP.c:silk_regularize_correlations_FLP` — add
/// noise to the matrix diagonal.
pub fn silk_regularize_correlations_flp(xx_mat: &mut [f32], xx: &mut [f32], noise: f32, d: usize) {
    for i in 0..d {
        xx_mat[i * d + i] += noise;
    }
    xx[0] += noise;
}

// ---------------------------------------------------------------------------------------------
// residual_energy_FLP.c
// ---------------------------------------------------------------------------------------------

const MAX_ITERATIONS_RESIDUAL_NRG: i32 = 10;
const REGULARIZATION_FACTOR: f32 = 1e-8;

/// Port of `silk/float/residual_energy_FLP.c:silk_residual_energy_covar_FLP` — residual energy
/// `nrg = wxx - 2 * wXx * c + c' * wXX * c` (`wXX` is regularized in place if needed).
#[must_use]
pub fn silk_residual_energy_covar_flp(
    c: &[f32],
    w_xx_mat: &mut [f32],
    w_xx_vec: &[f32],
    wxx: f32,
    d: usize,
) -> f32 {
    let mut nrg = 0.0f32;
    let mut regularization = REGULARIZATION_FACTOR * (w_xx_mat[0] + w_xx_mat[d * d - 1]);
    let mut k = 0;
    while k < MAX_ITERATIONS_RESIDUAL_NRG {
        nrg = wxx;

        let mut tmp = 0.0f32;
        for i in 0..d {
            tmp += w_xx_vec[i] * c[i];
        }
        nrg -= 2.0f32 * tmp;

        // compute c' * wXX * c, assuming wXX is symmetric
        for i in 0..d {
            tmp = 0.0f32;
            for j in i + 1..d {
                tmp += w_xx_mat[i * d + j] * c[j];
            }
            nrg += c[i] * (2.0f32 * tmp + w_xx_mat[i * d + i] * c[i]);
        }
        if nrg > 0.0 {
            break;
        }
        // Add white noise
        for i in 0..d {
            w_xx_mat[i * d + i] += regularization;
        }
        // Increase noise for next run
        regularization *= 2.0f32;
        k += 1;
    }
    if k == MAX_ITERATIONS_RESIDUAL_NRG {
        // C: silk_assert( nrg == 0 ): a hard check only with the feature `assertions` (the unit
        // tests drive non-converging inputs).
        if nrg != 0.0 {
            assertion_failure!("nrg == 0");
        }
        nrg = 1.0f32;
    }
    nrg
}

/// Port of `silk/float/residual_energy_FLP.c:silk_residual_energy_FLP` — residual energies of
/// input subframes where all subframes have `lpc_order` preceding samples.
pub fn silk_residual_energy_flp(
    nrgs: &mut [f32; MNSF],
    x: &[f32],
    a: &[[f32; MLPC]; 2],
    gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) {
    let mut lpc_res = [0f32; (MAX_FRAME_LENGTH_US + MNSF * MLPC) / 2];
    let lpc_res_ptr = lpc_order;
    let shift = lpc_order + subfr_length;

    // Filter input to create the LPC residual for each frame half, and measure subframe
    // energies
    silk_lpc_analysis_filter_flp(&mut lpc_res, &a[0], x, 2 * shift, lpc_order);
    nrgs[0] = ((gains[0] * gains[0]) as f64
        * silk_energy_flp(&lpc_res[lpc_res_ptr..], subfr_length)) as f32;
    nrgs[1] = ((gains[1] * gains[1]) as f64
        * silk_energy_flp(&lpc_res[lpc_res_ptr + shift..], subfr_length)) as f32;

    if nb_subfr == MNSF {
        silk_lpc_analysis_filter_flp(&mut lpc_res, &a[1], &x[2 * shift..], 2 * shift, lpc_order);
        nrgs[2] = ((gains[2] * gains[2]) as f64
            * silk_energy_flp(&lpc_res[lpc_res_ptr..], subfr_length)) as f32;
        nrgs[3] = ((gains[3] * gains[3]) as f64
            * silk_energy_flp(&lpc_res[lpc_res_ptr + shift..], subfr_length))
            as f32;
    }
}

const MAX_FRAME_LENGTH_US: usize = crate::silk::define::MAX_FRAME_LENGTH as usize;

// ---------------------------------------------------------------------------------------------
// scale_copy_vector_FLP.c / scale_vector_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/scale_copy_vector_FLP.c:silk_scale_copy_vector_FLP` — copy and multiply
/// a vector by a constant.
pub fn silk_scale_copy_vector_flp(data_out: &mut [f32], data_in: &[f32], gain: f32, size: usize) {
    for (o, &i) in data_out[..size].iter_mut().zip(&data_in[..size]) {
        *o = gain * i;
    }
}

/// Port of `silk/float/scale_vector_FLP.c:silk_scale_vector_FLP` — multiply a vector by a
/// constant.
pub fn silk_scale_vector_flp(data1: &mut [f32], gain: f32, size: usize) {
    for v in &mut data1[..size] {
        *v *= gain;
    }
}

// ---------------------------------------------------------------------------------------------
// schur_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/schur_FLP.c:silk_schur_FLP`. Returns the residual energy.
pub fn silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    let mut c = [[0f64; 2]; SILK_MAX_ORDER_LPC + 1];
    celt_assert!(order <= SILK_MAX_ORDER_LPC);

    // Copy correlations
    for k in 0..=order {
        c[k][0] = auto_corr[k] as f64;
        c[k][1] = auto_corr[k] as f64;
    }

    for k in 0..order {
        // Get reflection coefficient: silk_max_float( C[ 0 ][ 1 ], 1e-9f ) evaluates in double
        let den = if c[0][1] > 1e-9f32 as f64 {
            c[0][1]
        } else {
            1e-9f32 as f64
        };
        let rc_tmp = -c[k + 1][0] / den;

        // Save the output
        refl_coef[k] = rc_tmp as f32;

        // Update correlations
        for n in 0..order - k {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }

    // Return residual energy
    c[0][1] as f32
}

// ---------------------------------------------------------------------------------------------
// sort_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/sort_FLP.c:silk_insertion_sort_decreasing_FLP` — partial insertion
/// sort: the first `k` of `l` values end up sorted in decreasing order, with their original
/// indices in `idx`.
pub fn silk_insertion_sort_decreasing_flp(a: &mut [f32], idx: &mut [i32], l: usize, k: usize) {
    celt_assert!(k > 0);
    celt_assert!(l > 0);
    celt_assert!(l >= k);
    let a = &mut a[..l];
    let idx = &mut idx[..k];

    // Write start indices in index vector
    for (i, v) in idx.iter_mut().enumerate() {
        *v = i as i32;
    }

    // Sort vector elements by value, decreasing order
    for i in 1..k {
        let value = a[i];
        let mut j = i as isize - 1;
        while j >= 0 && value > a[j as usize] {
            a[j as usize + 1] = a[j as usize]; // Shift value
            idx[j as usize + 1] = idx[j as usize]; // Shift index
            j -= 1;
        }
        a[(j + 1) as usize] = value; // Write value
        idx[(j + 1) as usize] = i as i32; // Write index
    }

    // If less than L values are asked check the remaining values, but only spend CPU to ensure
    // that the K first values are correct
    for i in k..l {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k as isize - 2;
            while j >= 0 && value > a[j as usize] {
                a[j as usize + 1] = a[j as usize]; // Shift value
                idx[j as usize + 1] = idx[j as usize]; // Shift index
                j -= 1;
            }
            a[(j + 1) as usize] = value; // Write value
            idx[(j + 1) as usize] = i as i32; // Write index
        }
    }
}

// ---------------------------------------------------------------------------------------------
// warped_autocorrelation_FLP.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/float/warped_autocorrelation_FLP.c:silk_warped_autocorrelation_FLP` —
/// autocorrelations for a warped frequency axis (`order` must be even).
pub fn silk_warped_autocorrelation_flp(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: usize,
    order: usize,
) {
    const MSO: usize = MAX_SHAPE_LPC_ORDER as usize;
    /// Most samples in flight at once in the wavefront below (`order / 2 + 1`).
    const MAX_IN_FLIGHT: usize = MSO / 2 + 1;
    let mut state = [0f64; MSO + 1];
    let mut c = [0f64; MSO + 1];
    let w = warping as f64;

    // Order must be even
    celt_assert!(order & 1 == 0);
    celt_assert!(order <= MSO);
    let input = &input[..length];

    // Perf: the C loop runs the allpass sections of one sample after the other, a serial
    // dependency chain of `order` sections per sample. Sample `n` only needs, at section pair
    // `p` (sections 2p, 2p+1), the states written by sample `n - 1` up to section pair
    // `p + 1`. The samples are therefore processed as a wavefront: at time step `t` every
    // sample `n` in flight advances by one step `p = t - n` (in increasing `n`), which keeps
    // the chains of up to `order / 2 + 1` samples overlapping. Every state value and every
    // `C[ i ]` accumulation sees exactly the operations of C in the same order (the
    // accumulations into each `C[ i ]` still happen in sample order), so the result is
    // bit-identical. `state[ 0 ]` equals the current input sample after the first section, so
    // the `C[ i ] += state[ 0 ] * tmp` products use the sample value `x` directly.
    let pairs = order / 2;
    // carry[p]: `tmp1` entering step `p` of the sample that runs step `p` next.
    let mut carry = [0f64; MAX_IN_FLIGHT];
    let last = length.saturating_sub(1);
    let steps = if length > 0 { length + pairs } else { 0 };
    for t in 0..steps {
        if t < length {
            carry[0] = input[t] as f64;
        }
        // Last step of sample t - pairs
        if t >= pairs {
            let x = input[t - pairs] as f64;
            let tmp1 = carry[pairs];
            state[order] = tmp1;
            c[order] += x * tmp1;
        }
        // Section pair p of sample n = t - p, for decreasing p (increasing n)
        if pairs > 0 {
            let p_hi = t.min(pairs - 1);
            let p_lo = t.saturating_sub(last);
            for p in (p_lo..=p_hi).rev() {
                let x = input[t - p] as f64;
                let tmp1 = carry[p];
                let i = 2 * p;
                // Output of allpass section
                let tmp2 = state[i] + w * state[i + 1] - w * tmp1;
                state[i] = tmp1;
                c[i] += x * tmp1;
                // Output of allpass section
                let tmp1 = state[i + 1] + w * state[i + 2] - w * tmp2;
                state[i + 1] = tmp2;
                c[i + 1] += x * tmp2;
                carry[p + 1] = tmp1;
            }
        }
    }

    // Copy correlations in silk_float output format
    for i in 0..order + 1 {
        corr[i] = c[i] as f32;
    }
}

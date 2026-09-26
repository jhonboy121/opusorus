//! Port of `dnn/burg.c` / `dnn/burg.h`: Burg LPC analysis (a double-precision copy of the SILK
//! float `silk_burg_modified_FLP`, used by the LPCNet feature extraction).

use crate::celt::arch::max32;
use crate::math;

/// `MAX_FRAME_SIZE` (burg.c): `subfr_length * nb_subfr` bound.
pub const MAX_FRAME_SIZE: usize = 384;
/// `SILK_MAX_ORDER_LPC` (burg.c).
pub const SILK_MAX_ORDER_LPC: usize = 16;
/// `FIND_LPC_COND_FAC` (burg.c).
pub const FIND_LPC_COND_FAC: f32 = 1e-5;

/// Port of dnn/burg.c:silk_energy_FLP (static): sum of squares in double, 4x unrolled.
#[must_use]
pub fn silk_energy_flp(data: &[f32], data_size: usize) -> f64 {
    let data = &data[..data_size];
    let mut result = 0.0f64;
    let mut i = 0;
    while i + 3 < data_size {
        let (d0, d1, d2, d3) = (
            data[i] as f64,
            data[i + 1] as f64,
            data[i + 2] as f64,
            data[i + 3] as f64,
        );
        result += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < data_size {
        let d = data[i] as f64;
        result += d * d;
        i += 1;
    }
    debug_assert!(result >= 0.0);
    result
}

/// Port of dnn/burg.c:silk_inner_product_FLP (static): inner product in double, 4x unrolled.
#[must_use]
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32], data_size: usize) -> f64 {
    let data1 = &data1[..data_size];
    let data2 = &data2[..data_size];
    let mut result = 0.0f64;
    let mut i = 0;
    while i + 3 < data_size {
        result += data1[i] as f64 * data2[i] as f64
            + data1[i + 1] as f64 * data2[i + 1] as f64
            + data1[i + 2] as f64 * data2[i + 2] as f64
            + data1[i + 3] as f64 * data2[i + 3] as f64;
        i += 4;
    }
    while i < data_size {
        result += data1[i] as f64 * data2[i] as f64;
        i += 1;
    }
    result
}

/// Port of dnn/burg.c:silk_burg_analysis. Computes `d` prediction coefficients into `a` from
/// `nb_subfr` stacked subframes of `subfr_length` samples in `x`; returns the residual energy.
#[must_use]
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    let mut c_first_row = [0f64; SILK_MAX_ORDER_LPC];
    let mut c_last_row = [0f64; SILK_MAX_ORDER_LPC];
    let mut caf = [0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0f64; SILK_MAX_ORDER_LPC + 1];
    let mut af = [0f64; SILK_MAX_ORDER_LPC];
    let min_inv_gain_d = min_inv_gain as f64;
    let cond_fac = FIND_LPC_COND_FAC as f64;

    debug_assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE);
    debug_assert!(d <= SILK_MAX_ORDER_LPC);
    let x = &x[..nb_subfr * subfr_length];

    // Compute autocorrelations, added over subframes.
    let mut c0 = silk_energy_flp(x, nb_subfr * subfr_length);
    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..d + 1 {
            c_first_row[n - 1] += silk_inner_product_flp(x_ptr, &x_ptr[n..], subfr_length - n);
        }
    }
    c_last_row.copy_from_slice(&c_first_row);

    // Initialize.
    let init = c0 + cond_fac * c0 + 1e-9f32 as f64;
    caf[0] = init;
    cab[0] = init;
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;
    for n in 0..d {
        // Update first row of correlation matrix (without first element), last row (without
        // last element, stored reversed), C * Af and C * flipud(Af) (stored reversed).
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                c_first_row[k] -= (x_ptr[n] * x_ptr[n - k - 1]) as f64;
                c_last_row[k] -= (x_ptr[subfr_length - n - 1] * x_ptr[subfr_length - n + k]) as f64;
                let atmp = af[k];
                tmp1 += x_ptr[n - k - 1] as f64 * atmp;
                tmp2 += x_ptr[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * x_ptr[n - k] as f64;
                cab[k] -= tmp2 * x_ptr[subfr_length - n + k - 1] as f64;
            }
        }
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        // Nominator and denominator for the next order reflection (parcor) coefficient.
        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }
        debug_assert!(nrg_f > 0.0);
        debug_assert!(nrg_b > 0.0);

        // Next order reflection (parcor) coefficient.
        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        debug_assert!(rc > -1.0 && rc < 1.0);

        // Update inverse prediction gain.
        let t = inv_gain * (1.0 - rc * rc);
        if t <= min_inv_gain_d {
            // Max prediction gain exceeded; set reflection coefficient such that max
            // prediction gain is exactly hit.
            rc = math::sqrt(1.0 - min_inv_gain_d / inv_gain);
            if num > 0.0 {
                // Ensure adjusted reflection coefficients has the original sign.
                rc = -rc;
            }
            inv_gain = min_inv_gain_d;
            reached_max_gain = true;
        } else {
            inv_gain = t;
        }

        // Update the AR coefficients.
        for k in 0..(n + 1) >> 1 {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            // Reached max prediction gain; set remaining coefficients to zero and exit loop.
            for v in &mut af[n + 1..d] {
                *v = 0.0;
            }
            break;
        }

        // Update C * Af and C * Ab.
        for k in 0..=n + 1 {
            let t1 = caf[k];
            caf[k] += rc * cab[n + 1 - k];
            cab[n + 1 - k] += rc * t1;
        }
    }

    let nrg_f = if reached_max_gain {
        // Convert to float.
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        // Subtract energy of preceding samples from C0.
        for s in 0..nb_subfr {
            c0 -= silk_energy_flp(&x[s * subfr_length..], d);
        }
        // Approximate residual energy.
        c0 * inv_gain
    } else {
        // Compute residual energy and store coefficients as float.
        let mut e = caf[0];
        let mut t1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            e += caf[k + 1] * atmp;
            t1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        e -= cond_fac * c0 * t1;
        e
    };

    // Return residual energy.
    max32(0.0, nrg_f as f32)
}

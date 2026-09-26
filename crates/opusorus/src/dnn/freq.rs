//! Port of `dnn/freq.c` / `dnn/freq.h`: the LPCNet 20 ms analysis helpers (band energies,
//! DCT, Burg cepstrum, LPC from cepstrum, analysis window and the 320-point transforms).

use crate::celt::arch::max16;
use crate::celt::kiss_fft::opus_fft;
use crate::celt::static_modes::KissFftCpx;
use crate::math;

use super::burg::silk_burg_analysis;
use super::lpcnet_tables::{DCT_TABLE, HALF_WINDOW, KFFT};

/// `LPC_ORDER`.
pub const LPC_ORDER: usize = 16;
/// `PREEMPHASIS` (`0.85f`).
pub const PREEMPHASIS: f32 = 0.85;
/// `FRAME_SIZE_5MS`.
pub const FRAME_SIZE_5MS: usize = 2;
/// `OVERLAP_SIZE_5MS`.
pub const OVERLAP_SIZE_5MS: usize = 2;
/// `TRAINING_OFFSET_5MS`.
pub const TRAINING_OFFSET_5MS: usize = 1;
/// `WINDOW_SIZE_5MS`.
pub const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;
/// `FRAME_SIZE` (160 samples at 16 kHz).
pub const FRAME_SIZE: usize = 80 * FRAME_SIZE_5MS;
/// `OVERLAP_SIZE`.
pub const OVERLAP_SIZE: usize = 80 * OVERLAP_SIZE_5MS;
/// `TRAINING_OFFSET`.
pub const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;
/// `WINDOW_SIZE`.
pub const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
/// `FREQ_SIZE`.
pub const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;
/// `NB_BANDS`.
pub const NB_BANDS: usize = 18;
/// `NB_BANDS_1`.
pub const NB_BANDS_1: usize = NB_BANDS - 1;

/// `eband5ms` (freq.c).
pub static EBAND5MS: [i16; NB_BANDS] = [
    // 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

/// `compensation` (freq.c).
pub static COMPENSATION: [f32; NB_BANDS] = [
    0.8, 1., 1., 1., 1., 1., 1., 1., 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667,
    0.173913,
];

/// Band `i` start bin and size (in bins) of the triangular band layout.
#[inline(always)]
fn band(i: usize) -> (usize, usize) {
    let start = EBAND5MS[i] as usize * WINDOW_SIZE_5MS;
    let size = (EBAND5MS[i + 1] - EBAND5MS[i]) as usize * WINDOW_SIZE_5MS;
    (start, size)
}

/// Port of dnn/freq.c:compute_band_energy_inverse (static).
pub fn compute_band_energy_inverse(band_e: &mut [f32], x: &[KissFftCpx]) {
    let mut sum = [0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let v = x[start + j];
            let mut tmp = v.r * v.r;
            tmp += v.i * v.i;
            tmp = (1.0f64 / (tmp as f64 + 1e-9)) as f32;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_e[..NB_BANDS].copy_from_slice(&sum);
}

/// Port of dnn/freq.c:lpcn_lpc (static; float build: the Q31 macros are plain products).
pub fn lpcn_lpc(lpc: &mut [f32], rc: &mut [f32], ac: &[f32], p: usize) -> f32 {
    let lpc = &mut lpc[..p];
    let rc = &mut rc[..p];
    let mut error = ac[0];
    lpc.fill(0.0);
    rc.fill(0.0);
    if ac[0] != 0.0 {
        for i in 0..p {
            // Sum up this iteration's reflection coefficient.
            let mut rr = 0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            rc[i] = r;
            // Update LPC coefficients and total error.
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= (r * r) * error;
            // Bail out once we get 30 dB gain.
            if error < 0.001f32 * ac[0] {
                break;
            }
        }
    }
    error
}

/// Port of dnn/freq.c:lpcn_compute_band_energy.
pub fn lpcn_compute_band_energy(band_e: &mut [f32], x: &[KissFftCpx]) {
    let mut sum = [0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let v = x[start + j];
            let mut tmp = v.r * v.r;
            tmp += v.i * v.i;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_e[..NB_BANDS].copy_from_slice(&sum);
}

/// Port of dnn/freq.c:compute_burg_cepstrum (static).
pub fn compute_burg_cepstrum(pcm: &[f32], burg_cepstrum: &mut [f32], len: usize, order: usize) {
    let mut burg_in = [0f32; FRAME_SIZE];
    let mut burg_lpc = [0f32; LPC_ORDER];
    let mut x = [0f32; WINDOW_SIZE];
    let mut eburg = [0f32; NB_BANDS];
    let mut lpc = [KissFftCpx::default(); FREQ_SIZE];
    let mut ly = [0f32; NB_BANDS];
    let mut log_max = -2f32;
    let mut follow = -2f32;
    debug_assert!(order <= LPC_ORDER);
    debug_assert!(len <= FRAME_SIZE);
    for i in 0..len - 1 {
        burg_in[i] = pcm[i + 1] - PREEMPHASIS * pcm[i];
    }
    let mut g = silk_burg_analysis(&mut burg_lpc, &burg_in, 1e-3f64 as f32, len - 1, 1, order);
    g /= (len as i32 - 2 * (order as i32 - 1)) as f32;
    x[0] = 1.0;
    for i in 0..order {
        x[i + 1] = (-burg_lpc[i] as f64 * math::pow(0.995, (i + 1) as f64)) as f32;
    }
    forward_transform(&mut lpc, &x);
    compute_band_energy_inverse(&mut eburg, &lpc);
    let norm = (1.0f32 / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32)) as f64;
    for e in &mut eburg {
        *e = (*e as f64 * (0.45 * g as f64 * norm)) as f32;
    }
    for i in 0..NB_BANDS {
        ly[i] = math::log10(1e-2 + eburg[i] as f64) as f32;
        // C: MAX16(logMax-8, MAX16(follow-2.5, Ly[i])) with `follow-2.5` in double.
        let inner = dmax(follow as f64 - 2.5, ly[i] as f64);
        ly[i] = dmax((log_max - 8.0) as f64, inner) as f32;
        log_max = max16(log_max, ly[i]);
        follow = dmax(follow as f64 - 2.5, ly[i] as f64) as f32;
    }
    dct(burg_cepstrum, &ly);
    burg_cepstrum[0] += -4.0;
}

/// C `MAX16(a,b)` evaluated in double (`(a) > (b) ? (a) : (b)`).
#[inline(always)]
const fn dmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Port of dnn/freq.c:burg_cepstral_analysis. `ceps` receives `2*NB_BANDS` values from the
/// `FRAME_SIZE` samples of `x`.
pub fn burg_cepstral_analysis(ceps: &mut [f32], x: &[f32]) {
    compute_burg_cepstrum(x, &mut ceps[..NB_BANDS], FRAME_SIZE / 2, LPC_ORDER);
    compute_burg_cepstrum(
        &x[FRAME_SIZE / 2..],
        &mut ceps[NB_BANDS..2 * NB_BANDS],
        FRAME_SIZE / 2,
        LPC_ORDER,
    );
    for i in 0..NB_BANDS {
        let c0 = ceps[i];
        let c1 = ceps[NB_BANDS + i];
        ceps[i] = (0.5 * (c0 + c1) as f64) as f32;
        ceps[NB_BANDS + i] = c0 - c1;
    }
}

/// Port of dnn/freq.c:interp_band_gain (static). Writes `g[..FREQ_SIZE-1]`.
///
/// C clears only `FREQ_SIZE` *bytes* first (`memset(g, 0, FREQ_SIZE)`); every entry below
/// `FREQ_SIZE-1` is then written by the loop and the caller sets the last one, so the partial
/// clear has no effect.
pub fn interp_band_gain(g: &mut [f32], band_e: &[f32]) {
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[start + j] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

/// Port of dnn/freq.c:dct.
pub fn dct(out: &mut [f32], input: &[f32]) {
    let input = &input[..NB_BANDS];
    let s = math::sqrt(2.0 / NB_BANDS as f64);
    for (i, o) in out[..NB_BANDS].iter_mut().enumerate() {
        let mut sum = 0f32;
        for (j, &v) in input.iter().enumerate() {
            sum += v * DCT_TABLE[j * NB_BANDS + i];
        }
        *o = (sum as f64 * s) as f32;
    }
}

/// Port of dnn/freq.c:idct (static).
pub fn idct(out: &mut [f32], input: &[f32]) {
    let input = &input[..NB_BANDS];
    let s = math::sqrt(2.0 / NB_BANDS as f64);
    for (i, o) in out[..NB_BANDS].iter_mut().enumerate() {
        let mut sum = 0f32;
        for (j, &v) in input.iter().enumerate() {
            sum += v * DCT_TABLE[i * NB_BANDS + j];
        }
        *o = (sum as f64 * s) as f32;
    }
}

/// Port of dnn/freq.c:forward_transform: real 320-point FFT (`out` gets `FREQ_SIZE` bins).
pub fn forward_transform(out: &mut [KissFftCpx], input: &[f32]) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    for (xi, &v) in x.iter_mut().zip(&input[..WINDOW_SIZE]) {
        xi.r = v;
        xi.i = 0.0;
    }
    opus_fft(&KFFT, &x, &mut y);
    out[..FREQ_SIZE].copy_from_slice(&y[..FREQ_SIZE]);
}

/// Port of dnn/freq.c:inverse_transform (static).
pub fn inverse_transform(out: &mut [f32], input: &[KissFftCpx]) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    x[..FREQ_SIZE].copy_from_slice(&input[..FREQ_SIZE]);
    for i in FREQ_SIZE..WINDOW_SIZE {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    opus_fft(&KFFT, &x, &mut y);
    // Output in reverse order for IFFT.
    out[0] = WINDOW_SIZE as f32 * y[0].r;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
    }
}

/// Port of dnn/freq.c:lpc_from_bands (static).
pub fn lpc_from_bands(lpc: &mut [f32], ex: &[f32]) -> f32 {
    let mut ac = [0f32; LPC_ORDER + 1];
    let mut rc = [0f32; LPC_ORDER];
    let mut xr = [0f32; FREQ_SIZE];
    let mut x_auto = [KissFftCpx::default(); FREQ_SIZE];
    let mut x_auto_t = [0f32; WINDOW_SIZE];
    interp_band_gain(&mut xr, ex);
    xr[FREQ_SIZE - 1] = 0.0;
    for (c, &v) in x_auto.iter_mut().zip(&xr) {
        c.r = v;
    }
    inverse_transform(&mut x_auto_t, &x_auto);
    ac.copy_from_slice(&x_auto_t[..LPC_ORDER + 1]);

    // -40 dB noise floor.
    ac[0] = (ac[0] as f64 + (ac[0] as f64 * 1e-4 + (320 / 12) as f64 / 38.0)) as f32;
    // Lag windowing.
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a = (*a as f64 * (1.0 - 6e-5 * i as f64 * i as f64)) as f32;
    }
    lpcn_lpc(lpc, &mut rc, &ac, LPC_ORDER)
}

/// Port of dnn/freq.c:lpc_weighting.
pub fn lpc_weighting(lpc: &mut [f32], gamma: f32) {
    let mut gamma_i = gamma;
    for v in &mut lpc[..LPC_ORDER] {
        *v *= gamma_i;
        gamma_i *= gamma;
    }
}

/// Port of dnn/freq.c:lpc_from_cepstrum. Returns the prediction error.
pub fn lpc_from_cepstrum(lpc: &mut [f32], cepstrum: &[f32]) -> f32 {
    let mut ex = [0f32; NB_BANDS];
    let mut tmp = [0f32; NB_BANDS];
    tmp.copy_from_slice(&cepstrum[..NB_BANDS]);
    tmp[0] += 4.0;
    idct(&mut ex, &tmp);
    for (e, &c) in ex.iter_mut().zip(&COMPENSATION) {
        *e = (math::pow(10.0, *e as f64) * c as f64) as f32;
    }
    lpc_from_bands(lpc, &ex)
}

/// Port of dnn/freq.c:apply_window (in place on `WINDOW_SIZE` samples).
pub fn apply_window(x: &mut [f32]) {
    let x = &mut x[..WINDOW_SIZE];
    for i in 0..OVERLAP_SIZE {
        x[i] *= HALF_WINDOW[i];
        x[WINDOW_SIZE - 1 - i] *= HALF_WINDOW[i];
    }
}

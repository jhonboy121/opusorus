//! Port of celt/celt_lpc.c, celt/celt_lpc.h: LPC analysis and FIR/IIR filters (float build).
//!
//! Only the portable C paths are ported (`celt_fir_c`, the default non-`SMALL_FOOTPRINT`
//! `celt_iir`); SIMD overrides (`OPUS_X86_*`, `OPUS_ARM_*`) are dropped.

use crate::celt::arch::{
    CeltCoef, OpusVal16, OpusVal32, coef2val16, extend32, mac16_16, mult16_16, mult16_16_q15,
    mult32_32_q31, shl32, shr32, sround16,
};
use crate::celt::mathops::frac_div32;
use crate::celt::pitch::{Scratch, celt_pitch_xcorr, xcorr_kernel};

/// `CELT_LPC_ORDER`.
pub const CELT_LPC_ORDER: usize = 24;

/// `SIG_SHIFT` (celt/arch.h); a no-op shift amount in the float build.
const SIG_SHIFT: i32 = 12;

/// Port of celt/celt_lpc.c:_celt_lpc (Levinson-Durbin recursion).
///
/// Writes `lpc[..p]` from the autocorrelation `ac[..=p]`.
pub fn _celt_lpc(lpc: &mut [OpusVal16], ac: &[OpusVal32], p: usize) {
    // FIXED_POINT: not ported (float build) — int32 lpc[] and the Q25->Q12 fit.
    let lpc = &mut lpc[..p];
    let ac = &ac[..=p];
    let mut error: OpusVal32 = ac[0];

    lpc.fill(0.0);
    if ac[0] > 1e-10f32 {
        for i in 0..p {
            // Sum up this iteration's reflection coefficient
            let mut rr: OpusVal32 = 0.0;
            for j in 0..i {
                rr += mult32_32_q31(lpc[j], ac[i - j]);
            }
            rr += shr32(ac[i + 1], 6);
            let r: OpusVal32 = -frac_div32(shl32(rr, 6), error);
            // Update LPC coefficients and total error
            lpc[i] = shr32(r, 6);
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + mult32_32_q31(r, tmp2);
                lpc[i - 1 - j] = tmp2 + mult32_32_q31(r, tmp1);
            }

            error -= mult32_32_q31(mult32_32_q31(r, r), error);
            // Bail out once we get 30 dB gain
            if error <= 0.001f32 * ac[0] {
                break;
            }
        }
    }
}

/// Stack capacity for the reversed coefficients in [`celt_fir`] / [`celt_iir`].
const ORD_MAX: usize = 32;

/// Port of celt/celt_lpc.c:celt_fir_c.
///
/// Layout differs from C because C reads `ord` samples *before* its `x` pointer: here
/// `x[k + ord]` is C's `x[k]`, so `x` must hold `ord` history samples followed by the `n`
/// input samples (`n + ord` total). Writes `y[..n]` (which can therefore never alias `x`,
/// matching the C `celt_assert(x != y)`).
pub fn celt_fir(x: &[OpusVal16], num: &[OpusVal16], y: &mut [OpusVal16], n: usize, ord: usize) {
    let mut rs = Scratch::<ORD_MAX>::new();
    let rnum = rs.get(ord);
    let num = &num[..ord];
    let x = &x[..n + ord];
    let y = &mut y[..n];
    for (i, r) in rnum.iter_mut().enumerate() {
        *r = num[ord - i - 1];
    }
    let mut i = 0usize;
    // C: for (i=0;i<N-3;i+=4)
    while i + 3 < n {
        let mut sum: [OpusVal32; 4] = [
            shl32(extend32(x[ord + i]), SIG_SHIFT),
            shl32(extend32(x[ord + i + 1]), SIG_SHIFT),
            shl32(extend32(x[ord + i + 2]), SIG_SHIFT),
            shl32(extend32(x[ord + i + 3]), SIG_SHIFT),
        ];
        // C: xcorr_kernel(rnum, x+i-ord, sum, ord)
        xcorr_kernel(rnum, &x[i..], &mut sum, ord);
        y[i] = sround16(sum[0], SIG_SHIFT);
        y[i + 1] = sround16(sum[1], SIG_SHIFT);
        y[i + 2] = sround16(sum[2], SIG_SHIFT);
        y[i + 3] = sround16(sum[3], SIG_SHIFT);
        i += 4;
    }
    while i < n {
        let mut sum: OpusVal32 = shl32(extend32(x[ord + i]), SIG_SHIFT);
        for (j, &r) in rnum.iter().enumerate() {
            // C: x[i+j-ord]
            sum = mac16_16(sum, r, x[i + j]);
        }
        y[i] = sround16(sum, SIG_SHIFT);
        i += 1;
    }
}

/// Stack capacity for the filter state buffer in [`celt_iir`] (`n + ord`).
const IIR_Y_MAX: usize = if cfg!(feature = "qext") { 2208 } else { 1104 };

/// Port of celt/celt_lpc.c:celt_iir (default, non-`SMALL_FOOTPRINT` path).
///
/// Filters `x[..n]` into `y[..n]` through `1/A(z)` with coefficients `den[..ord]` and state
/// `mem[..ord]` (updated). `ord` must be a multiple of 4. See [`celt_iir_inplace`] for the
/// in-place form the CELT decoder uses (C passes `_x == _y`).
pub fn celt_iir(
    x: &[OpusVal32],
    den: &[OpusVal16],
    y: &mut [OpusVal32],
    n: usize,
    ord: usize,
    mem: &mut [OpusVal16],
) {
    // Each C `_y[i]` is written only after `_x[i..i+4]` has been read, so filtering a copy of
    // `x` in place is exactly equivalent.
    y[..n].copy_from_slice(&x[..n]);
    celt_iir_inplace(y, den, n, ord, mem);
}

/// In-place [`celt_iir`]: `buf[..n]` holds the input and receives the output (the C call
/// `celt_iir(buf, den, buf, ...)`).
pub fn celt_iir_inplace(
    buf: &mut [OpusVal32],
    den: &[OpusVal16],
    n: usize,
    ord: usize,
    mem: &mut [OpusVal16],
) {
    // SMALL_FOOTPRINT variant: not ported (not enabled in the oracle / default build).
    debug_assert!((ord & 3) == 0);
    let mut rs = Scratch::<ORD_MAX>::new();
    let mut ys = Scratch::<IIR_Y_MAX>::new();
    let rden = rs.get(ord);
    let y = ys.get(n + ord);
    let den = &den[..ord];
    let mem = &mut mem[..ord];
    let buf = &mut buf[..n];
    for (i, r) in rden.iter_mut().enumerate() {
        *r = den[ord - i - 1];
    }
    for i in 0..ord {
        y[i] = -mem[ord - i - 1];
    }
    y[ord..].fill(0.0);
    let mut i = 0usize;
    // C: for (i=0;i<N-3;i+=4)
    while i + 3 < n {
        // Unroll by 4 as if it were an FIR filter
        let mut sum: [OpusVal32; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        xcorr_kernel(rden, &y[i..], &mut sum, ord);
        // Patch up the result to compensate for the fact that this is an IIR
        y[i + ord] = -sround16(sum[0], SIG_SHIFT);
        buf[i] = sum[0];
        sum[1] = mac16_16(sum[1], y[i + ord], den[0]);
        y[i + ord + 1] = -sround16(sum[1], SIG_SHIFT);
        buf[i + 1] = sum[1];
        sum[2] = mac16_16(sum[2], y[i + ord + 1], den[0]);
        sum[2] = mac16_16(sum[2], y[i + ord], den[1]);
        y[i + ord + 2] = -sround16(sum[2], SIG_SHIFT);
        buf[i + 2] = sum[2];

        sum[3] = mac16_16(sum[3], y[i + ord + 2], den[0]);
        sum[3] = mac16_16(sum[3], y[i + ord + 1], den[1]);
        sum[3] = mac16_16(sum[3], y[i + ord], den[2]);
        y[i + ord + 3] = -sround16(sum[3], SIG_SHIFT);
        buf[i + 3] = sum[3];
        i += 4;
    }
    while i < n {
        let mut sum: OpusVal32 = buf[i];
        for (j, &r) in rden.iter().enumerate() {
            sum -= mult16_16(r, y[i + j]);
        }
        // Quirk ported faithfully: unlike the unrolled loop above, the tail stores the
        // output with a positive sign.
        y[i + ord] = sround16(sum, SIG_SHIFT);
        buf[i] = sum;
        i += 1;
    }
    for i in 0..ord {
        mem[i] = buf[n - i - 1];
    }
}

/// Stack capacity for the windowed copy in [`_celt_autocorr`] (`n`).
const AC_XX_MAX: usize = if cfg!(feature = "qext") { 2048 } else { 1024 };

/// Port of celt/celt_lpc.c:_celt_autocorr.
///
/// Writes `ac[..=lag]` from `x[..n]`, optionally windowing the first/last `overlap` samples
/// with `window[..overlap]` (C passes `NULL, 0` for no window: pass `&[]`, `0`). Returns the
/// scaling shift, always 0 in the float build.
pub fn _celt_autocorr(
    x: &[OpusVal16],
    ac: &mut [OpusVal32],
    window: &[CeltCoef],
    overlap: usize,
    lag: usize,
    n: usize,
) -> i32 {
    debug_assert!(n > 0);
    let fast_n = n - lag;
    let mut xxs = Scratch::<AC_XX_MAX>::new();
    let x = &x[..n];
    let ac = &mut ac[..=lag];
    let xptr: &[OpusVal16] = if overlap == 0 {
        x
    } else {
        let xx = xxs.get(n);
        xx.copy_from_slice(x);
        for i in 0..overlap {
            let w: OpusVal16 = coef2val16(window[i]);
            xx[i] = mult16_16_q15(x[i], w);
            xx[n - i - 1] = mult16_16_q15(x[n - i - 1], w);
        }
        xx
    };
    let shift = 0;
    // FIXED_POINT: not ported (float build) — ac0 pre-scaling.
    celt_pitch_xcorr(xptr, xptr, ac, fast_n, lag + 1);
    for k in 0..=lag {
        let mut d: OpusVal32 = 0.0;
        for i in k + fast_n..n {
            d = mac16_16(d, xptr[i], xptr[i - k]);
        }
        ac[k] += d;
    }
    // FIXED_POINT: not ported (float build) — ac[] normalisation.
    shift
}

//! Port of celt/celt_lpc.c, celt/celt_lpc.h: LPC analysis and FIR/IIR filters (float and
//! fixed-point builds).
//!
//! Only the portable C paths are ported (`celt_fir_c`, the default non-`SMALL_FOOTPRINT`
//! `celt_iir`); SIMD overrides (`OPUS_X86_*`, `OPUS_ARM_*`) are dropped. The `#ifdef
//! FIXED_POINT` branches are selected with the `fixed-point` feature (`docs/FIXED_POINT.md`).

use crate::celt::arch::{
    CeltCoef, OpusVal16, OpusVal32, coef2val16, extend32, extract16, mac16_16, mult16_16,
    mult16_16_q15, neg16, shl32, sround16,
};
#[cfg(feature = "fixed-point")]
use crate::celt::arch::{
    OPUS_FAST_INT64, abs32, div32, min32, mult32_32_32, mult32_32_q16, pshr32, qconst32, shr32,
    shr64,
};
#[cfg(all(feature = "fixed-point", not(feature = "fixed-point-debug")))]
use crate::celt::arch::{int32, int64};
#[cfg(not(feature = "fixed-point"))]
use crate::celt::arch::{mult32_32_q31, shr32};
#[cfg(feature = "fixed-point")]
use crate::celt::entcode::ec_ilog;
#[cfg(feature = "fixed-point")]
use crate::celt::mathops::celt_ilog2;
use crate::celt::mathops::frac_div32;
use crate::celt::pitch::{Scratch, celt_pitch_xcorr, xcorr_kernel};

/// `CELT_LPC_ORDER`.
pub const CELT_LPC_ORDER: usize = 24;

/// `SIG_SHIFT` (celt/arch.h); a no-op shift amount in the float build.
const SIG_SHIFT: i32 = 12;

/// Port of celt/celt_lpc.c:_celt_lpc (Levinson-Durbin recursion, float build).
///
/// Writes `lpc[..p]` from the autocorrelation `ac[..=p]`.
#[cfg(not(feature = "fixed-point"))]
pub fn _celt_lpc(lpc: &mut [OpusVal16], ac: &[OpusVal32], p: usize) {
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

/// Port of celt/celt_lpc.c:_celt_lpc (Levinson-Durbin recursion, fixed-point build).
///
/// Writes `lpc[..p]` (Q12) from the autocorrelation `ac[..=p]`; `p <= CELT_LPC_ORDER`. When the
/// coefficients cannot be made to fit in 16 bits, only `lpc[0]` is written (C quirk: the other
/// outputs keep their previous values).
#[cfg(feature = "fixed-point")]
pub fn _celt_lpc(lpc: &mut [OpusVal16], ac: &[OpusVal32], p: usize) {
    _celt_lpc_fast_int64::<OPUS_FAST_INT64>(lpc, ac, p);
}

/// [`_celt_lpc`] with an explicit `OPUS_FAST_INT64` selection (the target's value is
/// [`OPUS_FAST_INT64`]; the other form exists so a 64-bit host can verify the 32-bit one).
#[cfg(feature = "fixed-point")]
pub fn _celt_lpc_fast_int64<const FAST_INT64: bool>(
    lpc_out: &mut [OpusVal16],
    ac: &[OpusVal32],
    p: usize,
) {
    let q31 = |a: i32, b: i32| -> i32 {
        // FIXED_DEBUG: `MULT32_32_Q31` is always the checking 16-bit partial-product form.
        #[cfg(feature = "fixed-point-debug")]
        return crate::celt::arch::mult32_32_q31(a, b);
        #[cfg(not(feature = "fixed-point-debug"))]
        if FAST_INT64 {
            int64::mult32_32_q31(a, b)
        } else {
            int32::mult32_32_q31(a, b)
        }
    };
    let lpc_out = &mut lpc_out[..p];
    let ac = &ac[..=p];
    let mut error: OpusVal32 = ac[0];
    // C: `opus_val32 lpc[CELT_LPC_ORDER]` + OPUS_CLEAR(lpc, p).
    let mut lpc_buf = [0i32; CELT_LPC_ORDER];
    let lpc = &mut lpc_buf[..p];

    if ac[0] != 0 {
        for i in 0..p {
            // Sum up this iteration's reflection coefficient
            let mut rr: OpusVal32;
            if FAST_INT64 {
                let mut acc: i64 = 0;
                for j in 0..i {
                    acc += i64::from(lpc[j]) * i64::from(ac[i - j]);
                }
                rr = shr64(acc, 31) as i32;
            } else {
                rr = 0;
                for j in 0..i {
                    rr += q31(lpc[j], ac[i - j]);
                }
            }
            rr += shr32(ac[i + 1], 6);
            let r: OpusVal32 = -frac_div32(shl32(rr, 6), error);
            // Update LPC coefficients and total error
            lpc[i] = shr32(r, 6);
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + q31(r, tmp2);
                lpc[i - 1 - j] = tmp2 + q31(r, tmp1);
            }

            error -= q31(q31(r, r), error);
            // Bail out once we get 30 dB gain
            if error <= shr32(ac[0], 10) {
                break;
            }
        }
    }

    // Convert the int32 lpcs to int16 and ensure there are no wrap-arounds. This reuses the
    // logic in silk_LPC_fit() and silk_bwexpander_32(). Any bug fixes should also be applied
    // there.
    let mut idx = 0usize;
    let mut iter = 0;
    while iter < 10 {
        let mut maxabs: OpusVal32 = 0;
        for (i, &l) in lpc.iter().enumerate() {
            let absval = abs32(l);
            if absval > maxabs {
                maxabs = absval;
                idx = i;
            }
        }
        maxabs = pshr32(maxabs, 13); // Q25->Q12

        if maxabs > 32767 {
            maxabs = min32(maxabs, 163838);
            let mut chirp_q16: OpusVal32 = qconst32(0.999, 16)
                - div32(
                    shl32(maxabs - 32767, 14),
                    shr32(mult32_32_32(maxabs, idx as i32 + 1), 2),
                );
            let chirp_minus_one_q16: OpusVal32 = chirp_q16 - 65536;

            // Apply bandwidth expansion.
            for l in lpc[..p - 1].iter_mut() {
                *l = mult32_32_q16(chirp_q16, *l);
                chirp_q16 += pshr32(mult32_32_32(chirp_q16, chirp_minus_one_q16), 16);
            }
            lpc[p - 1] = mult32_32_q16(chirp_q16, lpc[p - 1]);
        } else {
            break;
        }
        iter += 1;
    }

    if iter == 10 {
        // If the coeffs still do not fit into the 16 bit range after 10 iterations, fall back
        // to the A(z)=1 filter. (C also clears its local int32 copy, which is not read again.)
        lpc_out[0] = 4096; // Q12
    } else {
        for (o, &l) in lpc_out.iter_mut().zip(lpc.iter()) {
            *o = extract16(pshr32(l, 13)); // Q25->Q12
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
    let mut rs = Scratch::<OpusVal16, ORD_MAX>::new();
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
    let mut rs = Scratch::<OpusVal16, ORD_MAX>::new();
    let mut ys = Scratch::<OpusVal16, IIR_Y_MAX>::new();
    let rden = rs.get(ord);
    let y = ys.get(n + ord);
    let den = &den[..ord];
    let mem = &mut mem[..ord];
    let buf = &mut buf[..n];
    for (i, r) in rden.iter_mut().enumerate() {
        *r = den[ord - i - 1];
    }
    for i in 0..ord {
        // C: `y[i] = -mem[ord-i-1];` (an `int` negation stored in an `opus_val16`).
        y[i] = extract16(neg16(mem[ord - i - 1]));
    }
    y[ord..].fill(OpusVal16::default());
    let mut i = 0usize;
    // C: for (i=0;i<N-3;i+=4)
    while i + 3 < n {
        // Unroll by 4 as if it were an FIR filter
        let mut sum: [OpusVal32; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        xcorr_kernel(rden, &y[i..], &mut sum, ord);
        // Patch up the result to compensate for the fact that this is an IIR (SROUND16
        // saturates to +-32767, so the negation fits in 16 bits).
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
        // C: `mem[i] = _y[N-i-1];` (an `opus_val32` stored in an `opus_val16`: the fixed-point
        // build truncates).
        mem[i] = extract16(buf[n - i - 1]);
    }
}

/// Stack capacity for the windowed copy in [`_celt_autocorr`] (`n`).
const AC_XX_MAX: usize = if cfg!(feature = "qext") { 2048 } else { 1024 };

/// Port of celt/celt_lpc.c:_celt_autocorr.
///
/// Writes `ac[..=lag]` from `x[..n]`, optionally windowing the first/last `overlap` samples
/// with `window[..overlap]` (C passes `NULL, 0` for no window: pass `&[]`, `0`). Returns the
/// scaling shift (always 0 in the float build; the fixed-point build normalises `ac` and
/// returns the total shift of the result).
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
    let mut xxs = Scratch::<OpusVal16, AC_XX_MAX>::new();
    let x = &x[..n];
    let ac = &mut ac[..=lag];
    let xx = xxs.get(n);
    // Whether C's `xptr` points at `xx` (else at `x`).
    let mut use_xx = false;
    if overlap != 0 {
        xx.copy_from_slice(x);
        for i in 0..overlap {
            let w: OpusVal16 = coef2val16(window[i]);
            xx[i] = extract16(mult16_16_q15(x[i], w));
            xx[n - i - 1] = extract16(mult16_16_q15(x[n - i - 1], w));
        }
        use_xx = true;
    }
    // C: `shift=0;`, then (fixed-point build) overwritten by the scaling below.
    #[cfg(not(feature = "fixed-point"))]
    let shift = 0;
    #[cfg(feature = "fixed-point")]
    let mut shift;
    #[cfg(feature = "fixed-point")]
    {
        let xptr: &[OpusVal16] = if use_xx { xx } else { x };
        let ac0_shift = celt_ilog2((n + (n >> 4)) as i32);
        let mut ac0: OpusVal32 = 1 + ((n as i32) << 7);
        if n & 1 != 0 {
            ac0 += shr32(mult16_16(xptr[0], xptr[0]), ac0_shift);
        }
        let mut i = n & 1;
        while i < n {
            ac0 += shr32(mult16_16(xptr[i], xptr[i]), ac0_shift);
            ac0 += shr32(mult16_16(xptr[i + 1], xptr[i + 1]), ac0_shift);
            i += 2;
        }
        // Consider the effect of rounding-to-nearest when scaling the signal.
        ac0 += shr32(ac0, 7);

        shift = celt_ilog2(ac0) - 30 + ac0_shift + 1;
        shift /= 2;
        if shift > 0 {
            // C: `xx[i] = PSHR32(xptr[i], shift)` (in place when `xptr == xx`).
            if use_xx {
                for v in xx.iter_mut() {
                    *v = pshr32(*v, shift) as i16;
                }
            } else {
                for (v, &xi) in xx.iter_mut().zip(x) {
                    *v = pshr32(xi, shift) as i16;
                }
            }
            use_xx = true;
        } else {
            shift = 0;
        }
    }
    let xptr: &[OpusVal16] = if use_xx { xx } else { x };
    celt_pitch_xcorr(xptr, xptr, ac, fast_n, lag + 1);
    for k in 0..=lag {
        let mut d = OpusVal32::default();
        for i in k + fast_n..n {
            d = mac16_16(d, xptr[i], xptr[i - k]);
        }
        ac[k] += d;
    }
    #[cfg(feature = "fixed-point")]
    {
        shift *= 2;
        if shift <= 0 {
            ac[0] += shl32(1, -shift);
        }
        if ac[0] < 268_435_456 {
            let shift2 = 29 - ec_ilog(ac[0] as u32);
            for a in ac.iter_mut() {
                *a = shl32(*a, shift2);
            }
            shift -= shift2;
        } else if ac[0] >= 536_870_912 {
            let mut shift2 = 1;
            if ac[0] >= 1_073_741_824 {
                shift2 += 1;
            }
            for a in ac.iter_mut() {
                *a = shr32(*a, shift2);
            }
            shift += shift2;
        }
    }
    shift
}

//! Port of celt/pitch.c, celt/pitch.h: pitch analysis (float and fixed-point builds).
//!
//! Only the portable C reference kernels are ported (`xcorr_kernel_c`, `dual_inner_prod_c`,
//! `celt_inner_prod_c`, `celt_pitch_xcorr_c`); the SIMD/asm overrides (`OPUS_X86_*`,
//! `OPUS_ARM_*`, MIPS) are not used by the oracle build and are dropped. `comb_filter_const`
//! lives in celt/celt.c and is not part of this file.
//!
//! The `#ifdef FIXED_POINT` branches are selected with the `fixed-point` feature; code shared by
//! both builds is written against the `celt::arch` macro functions (see `docs/FIXED_POINT.md`).
//!
//! Pointer arguments become slices whose element 0 is the C pointer's target; C code that
//! reads *before* a pointer (`x[-i]`) is handled by passing the enclosing slice plus an offset,
//! as documented per function.

use alloc::vec::Vec;

#[cfg(not(feature = "fixed-point"))]
use crate::celt::arch::shr32;
use crate::celt::arch::{
    CeltSig, OpusVal16, OpusVal32, Q15ONE, extend32, extract16, half16, half32, mac16_16, max16,
    max32, mult16_16, mult16_16_q15, mult16_32_q15, qconst16, round16, shl32, vshr32,
};
#[cfg(feature = "fixed-point")]
use crate::celt::arch::{min32, shr16, shr32};
use crate::celt::celt_lpc::{_celt_autocorr, _celt_lpc};
use crate::celt::entcode::celt_udiv;
#[cfg(not(feature = "fixed-point"))]
use crate::celt::mathops::celt_sqrt;
use crate::celt::mathops::frac_div32;
#[cfg(feature = "fixed-point")]
use crate::celt::mathops::{celt_ilog2, celt_maxabs16, celt_maxabs32, celt_rsqrt_norm};

/// `SIG_SHIFT` (celt/arch.h); a no-op shift amount in the float build.
const SIG_SHIFT: i32 = 12;

/// A small C integer constant as an `opus_val32` (float build: converted to `float`).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn v32(x: i32) -> OpusVal32 {
    x as f32
}
/// A small C integer constant as an `opus_val32` (fixed-point build: unchanged).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn v32(x: i32) -> OpusVal32 {
    x
}
/// A small C integer constant as an `opus_val16` (float build: converted to `float`).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn v16(x: i16) -> OpusVal16 {
    x as f32
}
/// A small C integer constant as an `opus_val16` (fixed-point build: unchanged).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn v16(x: i16) -> OpusVal16 {
    x
}
/// `QCONST16(x, bits)` of an `f`-suffixed C literal (float build: the literal itself).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn qc16(x: f32, bits: i32) -> OpusVal16 {
    qconst16(x, bits)
}
/// `QCONST16(x, bits)` of an `f`-suffixed C literal (fixed-point build: Q`bits`).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn qc16(x: f32, bits: i32) -> OpusVal16 {
    qconst16(x as f64, bits)
}

/// Scratch storage replacing a C VLA: lives on the stack for lengths up to `N` (the sizes the
/// codec uses), and falls back to a heap buffer for larger, unusual lengths so that no input
/// size can make the port panic where C would not.
pub(crate) struct Scratch<T, const N: usize> {
    stack: [T; N],
    heap: Vec<T>,
}

impl<T: Copy + Default, const N: usize> Scratch<T, N> {
    /// Creates empty scratch storage (no allocation).
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            stack: [T::default(); N],
            heap: Vec::new(),
        }
    }

    /// Returns a zeroed-or-stale buffer of exactly `len` elements (C VLAs are uninitialised,
    /// so callers must write before reading, exactly like the C code does).
    #[inline(always)]
    pub(crate) fn get(&mut self, len: usize) -> &mut [T] {
        if len <= N {
            &mut self.stack[..len]
        } else {
            self.heap.resize(len, T::default());
            &mut self.heap[..len]
        }
    }
}

/// Port of celt/pitch.h:xcorr_kernel_c.
///
/// Accumulates into `sum[k]` the correlation `sum_j x[j]*y[j+k]` for `k = 0..4`.
/// `x` needs `len` elements and `y` needs `len + 3` elements; `len >= 3`.
#[inline(always)]
pub fn xcorr_kernel(x: &[OpusVal16], y: &[OpusVal16], sum: &mut [OpusVal32; 4], len: usize) {
    debug_assert!(len >= 3);
    let x = &x[..len];
    let y = &y[..len + 3];
    let mut xi = 0usize;
    let mut yi = 0usize;
    let mut y_0 = y[yi];
    yi += 1;
    let mut y_1 = y[yi];
    yi += 1;
    let mut y_2 = y[yi];
    yi += 1;
    // gcc doesn't realize that y_3 can't be used uninitialized
    let mut y_3 = OpusVal16::default();
    let mut j = 0usize;
    // C: for (j=0;j<len-3;j+=4)
    while j + 3 < len {
        let mut tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);
        tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);
        tmp = x[xi];
        xi += 1;
        y_1 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);
        tmp = x[xi];
        xi += 1;
        y_2 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_3);
        sum[1] = mac16_16(sum[1], tmp, y_0);
        sum[2] = mac16_16(sum[2], tmp, y_1);
        sum[3] = mac16_16(sum[3], tmp, y_2);
        j += 4;
    }
    // C: if (j++<len)
    let cond = j < len;
    j += 1;
    if cond {
        let tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);
    }
    // C: if (j++<len)
    let cond = j < len;
    j += 1;
    if cond {
        let tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);
    }
    if j < len {
        let tmp = x[xi];
        y_1 = y[yi];
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);
    }
}

/// Port of celt/pitch.h:dual_inner_prod_c. Returns `(xy1, xy2)`.
#[inline(always)]
#[must_use]
pub fn dual_inner_prod(
    x: &[OpusVal16],
    y01: &[OpusVal16],
    y02: &[OpusVal16],
    n: usize,
) -> (OpusVal32, OpusVal32) {
    let x = &x[..n];
    let y01 = &y01[..n];
    let y02 = &y02[..n];
    let mut xy01 = OpusVal32::default();
    let mut xy02 = OpusVal32::default();
    for i in 0..n {
        xy01 = mac16_16(xy01, x[i], y01[i]);
        xy02 = mac16_16(xy02, x[i], y02[i]);
    }
    (xy01, xy02)
}

/// Port of celt/pitch.h:celt_inner_prod_c.
#[inline(always)]
#[must_use]
pub fn celt_inner_prod(x: &[OpusVal16], y: &[OpusVal16], n: usize) -> OpusVal32 {
    let x = &x[..n];
    let y = &y[..n];
    let mut xy = OpusVal32::default();
    for i in 0..n {
        xy = mac16_16(xy, x[i], y[i]);
    }
    xy
}

/// Port of celt/pitch.c:find_best_pitch.
///
/// `y` needs `len + max_pitch` elements. The fixed-point build takes the C `yshift` and
/// `maxcorr` arguments.
pub fn find_best_pitch(
    xcorr: &[OpusVal32],
    y: &[OpusVal16],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [i32; 2],
    #[cfg(feature = "fixed-point")] yshift: i32,
    #[cfg(feature = "fixed-point")] maxcorr: OpusVal32,
) {
    // Float build: the C shifts are no-ops.
    #[cfg(not(feature = "fixed-point"))]
    let (yshift, xshift) = (0, 0);
    #[cfg(feature = "fixed-point")]
    let xshift = celt_ilog2(maxcorr) - 14;
    let xcorr = &xcorr[..max_pitch];
    let y = &y[..len + max_pitch];
    let mut syy: OpusVal32 = v32(1);
    let mut best_num: [OpusVal16; 2] = [v16(-1), v16(-1)];
    let mut best_den: [OpusVal32; 2] = [OpusVal32::default(); 2];
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for &yj in &y[..len] {
        syy += shr32(mult16_16(yj, yj), yshift);
    }
    for i in 0..max_pitch {
        if xcorr[i] > OpusVal32::default() {
            // C: `opus_val32 xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));`
            let xcorr16: OpusVal32 = extend32(extract16(vshr32(xcorr[i], xshift)));
            // Considering the range of xcorr16, this should avoid both underflows and
            // overflows (inf) when squaring xcorr16.
            #[cfg(not(feature = "fixed-point"))]
            let xcorr16 = xcorr16 * 1e-12f32;
            let num: OpusVal16 = extract16(mult16_16_q15(xcorr16, xcorr16));
            if mult16_32_q15(num, best_den[1]) > mult16_32_q15(best_num[1], syy) {
                if mult16_32_q15(num, best_den[0]) > mult16_32_q15(best_num[0], syy) {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i as i32;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i as i32;
                }
            }
        }
        syy +=
            shr32(mult16_16(y[i + len], y[i + len]), yshift) - shr32(mult16_16(y[i], y[i]), yshift);
        syy = max32(v32(1), syy);
    }
}

/// Port of celt/pitch.c:celt_fir5 (in place on `x[..n]`).
pub fn celt_fir5(x: &mut [OpusVal16], num: &[OpusVal16; 5], n: usize) {
    let x = &mut x[..n];
    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];
    let mut mem0 = OpusVal32::default();
    let mut mem1 = OpusVal32::default();
    let mut mem2 = OpusVal32::default();
    let mut mem3 = OpusVal32::default();
    let mut mem4 = OpusVal32::default();
    for xi in x.iter_mut() {
        let mut sum: OpusVal32 = shl32(extend32(*xi), SIG_SHIFT);
        sum = mac16_16(sum, num0, mem0);
        sum = mac16_16(sum, num1, mem1);
        sum = mac16_16(sum, num2, mem2);
        sum = mac16_16(sum, num3, mem3);
        sum = mac16_16(sum, num4, mem4);
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = extend32(*xi);
        *xi = round16(sum, SIG_SHIFT);
    }
}

/// Port of celt/pitch.c:pitch_downsample.
///
/// `x[c]` is channel `c` of the input (at least `len*factor` samples, plus `offset` past the
/// last decimated sample); only `x[0]` and, when `c == 2`, `x[1]` are read. Writes
/// `x_lp[..len]`.
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn pitch_downsample(
    x: &[&[CeltSig]],
    x_lp: &mut [OpusVal16],
    len: usize,
    c: usize,
    factor: usize,
) {
    let mut ac: [OpusVal32; 5] = [OpusVal32::default(); 5];
    let mut tmp: OpusVal16 = Q15ONE;
    let mut lpc: [OpusVal16; 4] = [OpusVal16::default(); 4];
    let mut lpc2: [OpusVal16; 5] = [OpusVal16::default(); 5];
    let c1: OpusVal16 = qc16(0.8f32, 15);
    let offset = factor / 2;
    let x_lp = &mut x_lp[..len];
    #[cfg(feature = "fixed-point")]
    {
        let x0 = x[0];
        let mut maxabs = celt_maxabs32(&x0[..len * factor]);
        if c == 2 {
            let maxabs_1 = celt_maxabs32(&x[1][..len * factor]);
            maxabs = max32(maxabs, maxabs_1);
        }
        if maxabs < 1 {
            maxabs = 1;
        }
        let mut shift = celt_ilog2(maxabs) - 10;
        if shift < 0 {
            shift = 0;
        }
        if c == 2 {
            shift += 1;
        }
        // C stores the `int` sums in `opus_val16` (implicit narrowing).
        for i in 1..len {
            x_lp[i] = (shr32(x0[factor * i - offset], shift + 2)
                + shr32(x0[factor * i + offset], shift + 2)
                + shr32(x0[factor * i], shift + 1)) as i16;
        }
        x_lp[0] = (shr32(x0[offset], shift + 2) + shr32(x0[0], shift + 1)) as i16;
        if c == 2 {
            let x1 = x[1];
            for i in 1..len {
                x_lp[i] = (i32::from(x_lp[i])
                    + (shr32(x1[factor * i - offset], shift + 2)
                        + shr32(x1[factor * i + offset], shift + 2)
                        + shr32(x1[factor * i], shift + 1))) as i16;
            }
            x_lp[0] = (i32::from(x_lp[0])
                + (shr32(x1[offset], shift + 2) + shr32(x1[0], shift + 1)))
                as i16;
        }
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        let x0 = x[0];
        for i in 1..len {
            x_lp[i] = 0.25f32 * x0[factor * i - offset]
                + 0.25f32 * x0[factor * i + offset]
                + 0.5f32 * x0[factor * i];
        }
        x_lp[0] = 0.25f32 * x0[offset] + 0.5f32 * x0[0];
        if c == 2 {
            let x1 = x[1];
            for i in 1..len {
                x_lp[i] += 0.25f32 * x1[factor * i - offset]
                    + 0.25f32 * x1[factor * i + offset]
                    + 0.5f32 * x1[factor * i];
            }
            x_lp[0] += 0.25f32 * x1[offset] + 0.5f32 * x1[0];
        }
    }
    _celt_autocorr(x_lp, &mut ac, &[], 0, 4, len);

    // Noise floor -40 dB
    #[cfg(feature = "fixed-point")]
    {
        ac[0] += shr32(ac[0], 13);
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        ac[0] *= 1.0001f32;
    }
    // Lag windowing
    for i in 1..=4usize {
        // ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));
        #[cfg(feature = "fixed-point")]
        {
            ac[i] -= mult16_32_q15((2 * i * i) as i32, ac[i]);
        }
        #[cfg(not(feature = "fixed-point"))]
        {
            let fi = i as f32;
            ac[i] -= ac[i] * (0.008f32 * fi) * (0.008f32 * fi);
        }
    }

    _celt_lpc(&mut lpc, &ac, 4);
    for i in 0..4 {
        tmp = extract16(mult16_16_q15(qc16(0.9f32, 15), tmp));
        lpc[i] = extract16(mult16_16_q15(lpc[i], tmp));
    }
    // Add a zero
    lpc2[0] = extract16(extend32(lpc[0]) + extend32(qc16(0.8f32, SIG_SHIFT)));
    lpc2[1] = extract16(extend32(lpc[1]) + mult16_16_q15(c1, lpc[0]));
    lpc2[2] = extract16(extend32(lpc[2]) + mult16_16_q15(c1, lpc[1]));
    lpc2[3] = extract16(extend32(lpc[3]) + mult16_16_q15(c1, lpc[2]));
    lpc2[4] = extract16(mult16_16_q15(c1, lpc[3]));
    celt_fir5(x_lp, &lpc2, len);
}

/// Port of celt/pitch.c:celt_pitch_xcorr_c (unrolled version; the float build returns nothing).
///
/// Writes `xcorr[i] = sum_j x[j]*y[i+j]` for `i < max_pitch`. `x` needs `len` elements and `y`
/// needs `len + max_pitch - 1` (rounded up to the kernel's reads, at most `len + max_pitch`).
#[cfg(not(feature = "fixed-point"))]
#[inline]
pub fn celt_pitch_xcorr(
    x: &[OpusVal16],
    y: &[OpusVal16],
    xcorr: &mut [OpusVal32],
    len: usize,
    max_pitch: usize,
) {
    pitch_xcorr(x, y, xcorr, len, max_pitch);
}

/// Port of celt/pitch.c:celt_pitch_xcorr_c (unrolled version). Returns `maxcorr`, the largest
/// correlation (at least 1).
///
/// Writes `xcorr[i] = sum_j x[j]*y[i+j]` for `i < max_pitch`. `x` needs `len` elements and `y`
/// needs `len + max_pitch - 1` (rounded up to the kernel's reads, at most `len + max_pitch`).
#[cfg(feature = "fixed-point")]
#[inline]
pub fn celt_pitch_xcorr(
    x: &[OpusVal16],
    y: &[OpusVal16],
    xcorr: &mut [OpusVal32],
    len: usize,
    max_pitch: usize,
) -> OpusVal32 {
    pitch_xcorr(x, y, xcorr, len, max_pitch)
}

/// Body of [`celt_pitch_xcorr`]; returns `maxcorr` in the fixed-point build (a dummy 0 in the
/// float build, which does not track it).
#[inline(always)]
fn pitch_xcorr(
    x: &[OpusVal16],
    y: &[OpusVal16],
    xcorr: &mut [OpusVal32],
    len: usize,
    max_pitch: usize,
) -> OpusVal32 {
    debug_assert!(max_pitch > 0);
    #[cfg(feature = "fixed-point")]
    let mut maxcorr: OpusVal32 = 1;
    let xcorr = &mut xcorr[..max_pitch];
    let mut i = 0usize;
    // Perf: C computes four lags at a time (xcorr_kernel), four dependency chains. Each
    // correlation is `sum_j x[j]*y[i+j]`, accumulated from 0 in increasing `j` (C's register
    // rotation in the kernel only reuses loaded `y` values). Sixteen lags at a time, with
    // exactly that per-lag sequence of operations, give the compiler independent vector
    // accumulators; the remaining lags use the C loops. In the fixed-point build every lag's
    // sequence of integer additions is the same as in C (so is any overflow), and `maxcorr`
    // is a maximum over the same values, which does not depend on the order.
    const LAGS: usize = 16;
    let xs = &x[..len];
    while i + LAGS <= max_pitch {
        let mut sum: [OpusVal32; LAGS] = [OpusVal32::default(); LAGS];
        let yb = &y[i..i + len + LAGS - 1];
        for (j, &xj) in xs.iter().enumerate() {
            let yj = &yb[j..j + LAGS];
            for (s, &yk) in sum.iter_mut().zip(yj) {
                *s = mac16_16(*s, xj, yk);
            }
        }
        xcorr[i..i + LAGS].copy_from_slice(&sum);
        #[cfg(feature = "fixed-point")]
        for &s in &sum {
            maxcorr = max32(maxcorr, s);
        }
        i += LAGS;
    }
    // C: for (i=0;i<max_pitch-3;i+=4)
    while i + 3 < max_pitch {
        let mut sum: [OpusVal32; 4] = [OpusVal32::default(); 4];
        xcorr_kernel(x, &y[i..], &mut sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        #[cfg(feature = "fixed-point")]
        {
            sum[0] = max32(sum[0], sum[1]);
            sum[2] = max32(sum[2], sum[3]);
            sum[0] = max32(sum[0], sum[2]);
            maxcorr = max32(maxcorr, sum[0]);
        }
        i += 4;
    }
    // In case max_pitch isn't a multiple of 4, do non-unrolled version.
    while i < max_pitch {
        let sum = celt_inner_prod(x, &y[i..], len);
        xcorr[i] = sum;
        #[cfg(feature = "fixed-point")]
        {
            maxcorr = max32(maxcorr, sum);
        }
        i += 1;
    }
    #[cfg(feature = "fixed-point")]
    {
        maxcorr
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        OpusVal32::default()
    }
}

/// Stack capacity for `x_lp4` in [`pitch_search`] (`len>>2`).
const PS_X4_MAX: usize = if cfg!(feature = "qext") { 512 } else { 256 };
/// Stack capacity for `y_lp4` in [`pitch_search`] (`lag>>2`).
const PS_Y4_MAX: usize = if cfg!(feature = "qext") { 1024 } else { 512 };
/// Stack capacity for `xcorr` in [`pitch_search`] (`max_pitch>>1`).
const PS_XCORR_MAX: usize = if cfg!(feature = "qext") { 1024 } else { 512 };

/// Port of celt/pitch.c:pitch_search. Returns the C `*pitch` output.
///
/// `x_lp` needs `len>>1` elements and `y` needs `(len>>1) + (max_pitch>>1)` elements (the
/// half-rate signals of the C call sites).
#[must_use]
pub fn pitch_search(x_lp: &[OpusVal16], y: &[OpusVal16], len: usize, max_pitch: usize) -> i32 {
    let mut best_pitch: [i32; 2] = [0, 0];
    let mut x4s = Scratch::<OpusVal16, PS_X4_MAX>::new();
    let mut y4s = Scratch::<OpusVal16, PS_Y4_MAX>::new();
    let mut xcs = Scratch::<OpusVal32, PS_XCORR_MAX>::new();

    debug_assert!(len > 0);
    debug_assert!(max_pitch > 0);
    let lag = len + max_pitch;

    let x_lp4 = x4s.get(len >> 2);
    let y_lp4 = y4s.get(lag >> 2);
    let xcorr = xcs.get(max_pitch >> 1);

    // Downsample by 2 again
    for (j, v) in x_lp4.iter_mut().enumerate() {
        *v = x_lp[2 * j];
    }
    for (j, v) in y_lp4.iter_mut().enumerate() {
        *v = y[2 * j];
    }

    #[cfg(feature = "fixed-point")]
    let shift = {
        let xmax = celt_maxabs16(x_lp4);
        let ymax = celt_maxabs16(y_lp4);
        let mut shift = celt_ilog2(max32(1, max32(xmax, ymax))) - 14 + celt_ilog2(len as i32) / 2;
        if shift > 0 {
            for v in x_lp4.iter_mut() {
                *v = shr16(*v, shift) as i16;
            }
            for v in y_lp4.iter_mut() {
                *v = shr16(*v, shift) as i16;
            }
            // Use double the shift for a MAC
            shift *= 2;
        } else {
            shift = 0;
        }
        shift
    };

    // Coarse search with 4x decimation
    #[cfg(feature = "fixed-point")]
    {
        let maxcorr = celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);
        find_best_pitch(
            xcorr,
            y_lp4,
            len >> 2,
            max_pitch >> 2,
            &mut best_pitch,
            0,
            maxcorr,
        );
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);
        find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2, &mut best_pitch);
    }

    // Finer search with 2x decimation
    #[cfg(feature = "fixed-point")]
    let mut maxcorr: OpusVal32 = 1;
    for i in 0..(max_pitch >> 1) {
        xcorr[i] = OpusVal32::default();
        let ii = i as i32;
        if (ii - 2 * best_pitch[0]).abs() > 2 && (ii - 2 * best_pitch[1]).abs() > 2 {
            continue;
        }
        #[cfg(feature = "fixed-point")]
        let sum = {
            let mut sum: OpusVal32 = 0;
            for (&xj, &yj) in x_lp[..len >> 1].iter().zip(&y[i..i + (len >> 1)]) {
                sum += shr32(mult16_16(xj, yj), shift);
            }
            sum
        };
        #[cfg(not(feature = "fixed-point"))]
        let sum = celt_inner_prod(x_lp, &y[i..], len >> 1);
        xcorr[i] = max32(v32(-1), sum);
        #[cfg(feature = "fixed-point")]
        {
            maxcorr = max32(maxcorr, sum);
        }
    }
    #[cfg(feature = "fixed-point")]
    find_best_pitch(
        xcorr,
        y,
        len >> 1,
        max_pitch >> 1,
        &mut best_pitch,
        shift + 1,
        maxcorr,
    );
    #[cfg(not(feature = "fixed-point"))]
    find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1, &mut best_pitch);

    // Refine by pseudo-interpolation
    let offset: i32;
    if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) as i32 - 1 {
        let bp = best_pitch[0] as usize;
        let a = xcorr[bp - 1];
        let b = xcorr[bp];
        let c = xcorr[bp + 1];
        if (c - a) > mult16_32_q15(qc16(0.7f32, 15), b - a) {
            offset = 1;
        } else if (a - c) > mult16_32_q15(qc16(0.7f32, 15), b - c) {
            offset = -1;
        } else {
            offset = 0;
        }
    } else {
        offset = 0;
    }
    2 * best_pitch[0] - offset
}

/// Port of celt/pitch.c:compute_pitch_gain (float version).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
#[must_use]
pub fn compute_pitch_gain(xy: OpusVal32, xx: OpusVal32, yy: OpusVal32) -> OpusVal16 {
    xy / celt_sqrt(1.0 + xx * yy)
}

/// Port of celt/pitch.c:compute_pitch_gain (fixed-point version): `xy/sqrt(xx*yy)` in Q15,
/// clamped to `[-Q15ONE, Q15ONE]`. `xx` and `yy` must be non-negative.
#[cfg(feature = "fixed-point")]
#[must_use]
pub fn compute_pitch_gain(xy: OpusVal32, xx: OpusVal32, yy: OpusVal32) -> OpusVal16 {
    if xy == 0 || xx == 0 || yy == 0 {
        return 0;
    }
    let sx = celt_ilog2(xx) - 14;
    let sy = celt_ilog2(yy) - 14;
    let mut shift = sx + sy;
    let mut x2y2: OpusVal32 = shr32(mult16_16(vshr32(xx, sx), vshr32(yy, sy)), 14);
    if shift & 1 != 0 {
        if x2y2 < 32768 {
            x2y2 <<= 1;
            shift -= 1;
        } else {
            x2y2 >>= 1;
            shift += 1;
        }
    }
    let den: OpusVal16 = celt_rsqrt_norm(x2y2);
    let mut g: OpusVal32 = mult16_32_q15(den, xy);
    g = vshr32(g, (shift >> 1) - 1);
    extract16(max32(-i32::from(Q15ONE), min32(g, i32::from(Q15ONE))))
}

/// Port of celt/pitch.c:second_check.
const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Stack capacity for `yy_lookup` in [`remove_doubling`] (`maxperiod/2 + 1`).
const RD_YY_MAX: usize = if cfg!(feature = "qext") { 1025 } else { 513 };

/// Port of celt/pitch.c:remove_doubling.
///
/// `x` is the C `x` pointer (start of the downsampled pitch buffer); it needs
/// `maxperiod/2 + n/2` elements. `t0` is the C `*T0_` in/out parameter. Returns the pitch gain.
#[must_use]
pub fn remove_doubling(
    x: &[OpusVal16],
    mut maxperiod: i32,
    mut minperiod: i32,
    mut n: i32,
    t0_: &mut i32,
    mut prev_period: i32,
    prev_gain: OpusVal16,
) -> OpusVal16 {
    let mut xcorr: [OpusVal32; 3] = [OpusVal32::default(); 3];
    let mut yys = Scratch::<OpusVal32, RD_YY_MAX>::new();

    let minperiod0 = minperiod;
    maxperiod /= 2;
    minperiod /= 2;
    *t0_ /= 2;
    prev_period /= 2;
    n /= 2;
    // C: x += maxperiod; all x[k] below are x[xo + k].
    let xo = maxperiod as usize;
    let nn = n as usize;
    if *t0_ >= maxperiod {
        *t0_ = maxperiod - 1;
    }

    let mut t = *t0_;
    let t0 = *t0_;
    let yy_lookup = yys.get(maxperiod as usize + 1);
    let xs = &x[xo..];
    let (xx, mut xy) = dual_inner_prod(xs, xs, &x[xo - t0 as usize..], nn);
    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=maxperiod as usize {
        yy = yy + mult16_16(x[xo - i], x[xo - i]) - mult16_16(x[xo + nn - i], x[xo + nn - i]);
        yy_lookup[i] = max32(OpusVal32::default(), yy);
    }
    yy = yy_lookup[t0 as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy, xx, yy);
    let mut g = g0;
    // Look for any pitch at T/k
    for k in 2i32..=15 {
        let t1b: i32;
        let t1 = celt_udiv((2 * t0 + k) as u32, (2 * k) as u32) as i32;
        if t1 < minperiod {
            break;
        }
        // Look for another strong correlation at T1b
        if k == 2 {
            if t1 + t0 > maxperiod {
                t1b = t0;
            } else {
                t1b = t0 + t1;
            }
        } else {
            t1b = celt_udiv(
                (2 * SECOND_CHECK[k as usize] * t0 + k) as u32,
                (2 * k) as u32,
            ) as i32;
        }
        let xy2;
        (xy, xy2) = dual_inner_prod(xs, &x[xo - t1 as usize..], &x[xo - t1b as usize..], nn);
        xy = half32(xy + xy2);
        yy = half32(yy_lookup[t1 as usize] + yy_lookup[t1b as usize]);
        let g1 = compute_pitch_gain(xy, xx, yy);
        let cont: OpusVal16 = if (t1 - prev_period).abs() <= 1 {
            prev_gain
        } else if (t1 - prev_period).abs() <= 2 && 5 * k * k < t0 {
            extract16(half16(prev_gain))
        } else {
            OpusVal16::default()
        };
        // C: `thresh = MAX16(QCONST16(.3f,15), MULT16_16_Q15(QCONST16(.7f,15),g0)-cont);` (an
        // `int` expression stored in an `opus_val16`).
        let mut thresh: OpusVal16 = extract16(max16(
            extend32(qc16(0.3f32, 15)),
            mult16_16_q15(qc16(0.7f32, 15), g0) - extend32(cont),
        ));
        // Bias against very high pitch (very short period) to avoid false-positives
        // due to short-term correlation
        if t1 < 3 * minperiod {
            thresh = extract16(max16(
                extend32(qc16(0.4f32, 15)),
                mult16_16_q15(qc16(0.85f32, 15), g0) - extend32(cont),
            ));
        } else if t1 < 2 * minperiod {
            // Unreachable in C as well (T1 < 2*minperiod implies T1 < 3*minperiod); kept
            // for fidelity.
            thresh = extract16(max16(
                extend32(qc16(0.5f32, 15)),
                mult16_16_q15(qc16(0.9f32, 15), g0) - extend32(cont),
            ));
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }
    best_xy = max32(OpusVal32::default(), best_xy);
    let mut pg: OpusVal16 = if best_yy <= best_xy {
        Q15ONE
    } else {
        extract16(shr32(frac_div32(best_xy, best_yy + v32(1)), 16))
    };

    for (k, xc) in xcorr.iter_mut().enumerate() {
        // C: x-(T+k-1)
        let lagk = t + k as i32 - 1;
        *xc = celt_inner_prod(xs, &x[(xo as i32 - lagk) as usize..], nn);
    }
    let offset: i32 =
        if (xcorr[2] - xcorr[0]) > mult16_32_q15(qc16(0.7f32, 15), xcorr[1] - xcorr[0]) {
            1
        } else if (xcorr[0] - xcorr[2]) > mult16_32_q15(qc16(0.7f32, 15), xcorr[1] - xcorr[2]) {
            -1
        } else {
            0
        };
    if pg > g {
        pg = g;
    }
    *t0_ = 2 * t + offset;

    if *t0_ < minperiod0 {
        *t0_ = minperiod0;
    }
    pg
}

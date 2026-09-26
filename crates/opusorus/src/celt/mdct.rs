//! Port of celt/mdct.c, celt/mdct.h (float and fixed-point builds).
//!
//! [`MdctLookup`] lives in [`crate::celt::static_modes`]. `clt_mdct_clear` has no Rust
//! counterpart: an owned lookup is released by `Drop`.
//!
//! The transforms are shared between the builds and written with the arithmetic macros of
//! [`crate::celt::arch`] and the `S_MUL`/`S_MUL2` helpers of [`super::kiss_fft`] (plain float
//! operations in the float build). The fixed-point build adds the headroom tracking of the
//! forward transform (the FFT then downshifts by `scale_shift - headroom`) and the
//! `pre_shift`/`post_shift`/`fft_shift` normalisation of the backward transform.

use super::arch::{
    CeltCoef, add32, add32_ovflw, neg32, pshr32, pshr32_ovflw, shl32_ovflw, sub32, sub32_ovflw,
};
#[cfg(feature = "fixed-point")]
use super::arch::{abs32, imax, imin, max32, shr32};
use super::kiss_fft::{
    Interleaved, KissFftScalar, KissTwiddleScalar, opus_fft_impl_buf, s_mul, s_mul2,
};
#[cfg(feature = "fixed-point")]
use super::mathops::{celt_ilog2, celt_zlog2};
use super::static_modes::{KissFftCpx, KissFftState, MdctLookup};

#[cfg(feature = "custom-modes")]
use super::static_modes::MAXFACTORS;
#[cfg(feature = "custom-modes")]
use crate::{Error, Result};
#[cfg(feature = "custom-modes")]
use alloc::{borrow::Cow, vec};

/// Largest `N/2` any MDCT lookup can have (sizes the forward-MDCT scratch buffers, which C
/// allocates as VLAs). Static modes: `N = 1920` (`3840` with QEXT); custom modes:
/// `N = 2 * frame_size` with `frame_size <= 1024` (`2048` with QEXT).
#[cfg(not(any(feature = "qext", feature = "custom-modes")))]
pub const MAX_MDCT_N2: usize = 960;
/// Largest `N/2` any MDCT lookup can have (see the non-QEXT definition).
#[cfg(all(not(feature = "qext"), feature = "custom-modes"))]
pub const MAX_MDCT_N2: usize = 1024;
/// Largest `N/2` any MDCT lookup can have (see the non-QEXT definition).
#[cfg(all(feature = "qext", not(feature = "custom-modes")))]
pub const MAX_MDCT_N2: usize = 1920;
/// Largest `N/2` any MDCT lookup can have (see the non-QEXT definition).
#[cfg(all(feature = "qext", feature = "custom-modes"))]
pub const MAX_MDCT_N2: usize = 2048;

/// An empty FFT state used to fill unused `kfft` slots (`maxshift < 3`); C leaves them
/// uninitialized and never reads them.
#[cfg(feature = "custom-modes")]
const fn empty_fft_state() -> KissFftState {
    KissFftState {
        nfft: 0,
        #[cfg(not(feature = "fixed-point"))]
        scale: 0.0,
        #[cfg(feature = "fixed-point")]
        scale: 0,
        #[cfg(feature = "fixed-point")]
        scale_shift: 0,
        shift: 0,
        factors: [0; 2 * MAXFACTORS],
        bitrev: Cow::Borrowed(&[]),
        twiddles: Cow::Borrowed(&[]),
    }
}

/// `trig[i]` of an MDCT of size `n` (the "enough points that sine isn't necessary" loop of
/// celt/mdct.c:clt_mdct_init), float build. `n2` is `n/2`.
#[cfg(all(feature = "custom-modes", not(feature = "fixed-point")))]
fn mdct_trig(i: i32, _n2: i32, n: i32) -> KissTwiddleScalar {
    crate::math::cos(2.0 * super::mathops::PI * (i as f64 + 0.125) / n as f64) as KissTwiddleScalar
}

/// `trig[i]` of an MDCT of size `n`, fixed-point build (Q15 `celt_cos_norm`).
#[cfg(all(
    feature = "custom-modes",
    feature = "fixed-point",
    not(feature = "qext")
))]
fn mdct_trig(i: i32, n2: i32, n: i32) -> KissTwiddleScalar {
    use super::arch::{div32, extend32, shl32};
    // TRIG_UPSCALE == 1
    super::mathops::celt_cos_norm(div32(add32(shl32(extend32(i), 17), n2 + 16384), n))
}

/// `trig[i]` of an MDCT of size `n`, fixed-point + QEXT build (Q31, computed in double).
#[cfg(all(feature = "custom-modes", feature = "fixed-point", feature = "qext"))]
fn mdct_trig(i: i32, _n2: i32, n: i32) -> KissTwiddleScalar {
    // C `M_PI` (the platform <math.h> value, the double nearest to pi).
    const M_PI: f64 = core::f64::consts::PI;
    // (kiss_twiddle_scalar)MAX32(-2147483647,MIN32(2147483647,floor(.5+2147483648*cos(...))))
    let x = crate::math::floor(
        0.5 + 2147483648.0 * crate::math::cos(2.0 * M_PI * (i as f64 + 0.125) / n as f64),
    );
    let x = if 2147483647.0 < x { 2147483647.0 } else { x };
    let x = if -2147483647.0 > x { -2147483647.0 } else { x };
    x as i32
}

/// Port of celt/mdct.c:clt_mdct_init.
///
/// Builds an owned MDCT lookup of size `n` with `maxshift + 1` FFT states.
///
/// # Errors
/// [`Error::AllocFail`] where C returns 0 (an FFT size cannot be factored). [`Error::BadArg`]
/// for `maxshift > 3` (C would overflow `kfft[4]`), `n <= 0`, or `n / 2 > MAX_MDCT_N2`
/// (Rust-only limit of the fixed-size forward scratch buffers).
#[cfg(feature = "custom-modes")]
pub fn clt_mdct_init(n: i32, maxshift: i32) -> Result<MdctLookup> {
    if !(0..=3).contains(&maxshift) || n <= 0 || (n >> 1) as usize > MAX_MDCT_N2 {
        return Err(Error::BadArg);
    }
    let mut big_n = n;
    let mut n2 = n >> 1;
    let mut kfft: [Cow<'static, KissFftState>; 4] = [
        Cow::Owned(empty_fft_state()),
        Cow::Owned(empty_fft_state()),
        Cow::Owned(empty_fft_state()),
        Cow::Owned(empty_fft_state()),
    ];
    for i in 0..=maxshift as usize {
        let st = if i == 0 {
            super::kiss_fft::opus_fft_alloc(n >> 2 >> i)?
        } else {
            super::kiss_fft::opus_fft_alloc_twiddles(n >> 2 >> i, Some(&kfft[0]))?
        };
        kfft[i] = Cow::Owned(st);
    }
    let mut trig = vec![KissTwiddleScalar::default(); (n - (n2 >> maxshift)) as usize];
    let mut off = 0usize;
    for _shift in 0..=maxshift {
        // We have enough points that sine isn't necessary
        for i in 0..n2 {
            trig[off + i as usize] = mdct_trig(i, n2, big_n);
        }
        off += n2 as usize;
        n2 >>= 1;
        big_n >>= 1;
    }
    Ok(MdctLookup {
        n,
        maxshift,
        kfft,
        trig: Cow::Owned(trig),
    })
}

/// Port of celt/mdct.c:clt_mdct_forward_c (the `clt_mdct_forward` macro).
///
/// Forward MDCT of `input` (`N/2 + overlap` samples, `N = l.n >> shift`), scaled by `4/N`,
/// written to `out[0], out[stride], ...` (`N/2` values). Unlike the C prototype the input is
/// not trashed (neither build writes it), so it is taken by shared reference.
pub fn clt_mdct_forward(
    l: &MdctLookup,
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let st = &*l.kfft[shift];

    let mut n = l.n as usize;
    let mut trig_off = 0usize;
    for _ in 0..shift {
        n >>= 1;
        trig_off += n;
    }
    let n4 = n >> 2;
    let trig = &l.trig[trig_off..];

    // Perf: C allocates the FFT buffer `f2` as a VLA of `N4` values. A safe Rust stack array
    // has to be initialized, and zeroing one of the worst-case size costs more than the whole
    // transform for short blocks, so the array is sized by class (48 kHz: 2.5-5 ms, 10-20 ms,
    // larger).
    debug_assert!(n4 <= MAX_MDCT_N2 / 2);
    if n4 <= 120 {
        let mut f2 = [KissFftCpx::default(); 120];
        clt_mdct_forward_impl(st, trig, input, out, window, overlap, stride, &mut f2[..n4]);
    } else if n4 <= 480 {
        let mut f2 = [KissFftCpx::default(); 480];
        clt_mdct_forward_impl(st, trig, input, out, window, overlap, stride, &mut f2[..n4]);
    } else {
        let mut f2 = [KissFftCpx::default(); MAX_MDCT_N2 / 2];
        clt_mdct_forward_impl(st, trig, input, out, window, overlap, stride, &mut f2[..n4]);
    }
}

/// Body of [`clt_mdct_forward`] after the size computation: `trig` starts at this shift's
/// table and `f2` is the C `f2` buffer (`N4` values).
///
/// Perf: C writes the windowed and folded input to a buffer `f` and pre-rotates it in a
/// second loop; here each folded pair is pre-rotated right away (the same operations per
/// element, in the same order), which saves the `f` buffer and a pass over it.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
#[inline(never)]
fn clt_mdct_forward_impl(
    st: &KissFftState,
    trig: &[KissTwiddleScalar],
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[CeltCoef],
    overlap: usize,
    stride: usize,
    f2: &mut [KissFftCpx],
) {
    let scale = st.scale;
    // Allows us to scale with MULT16_32_Q16(), which is faster than MULT16_32_Q15() on ARM.
    #[cfg(feature = "fixed-point")]
    let scale_shift = st.scale_shift - 1;
    let n4 = f2.len();
    let n2 = 2 * n4;
    // trig[i] and trig[N4 + i]
    let (trig0, trig1) = trig[..n2].split_at(n4);
    let bitrev = &st.bitrev[..n4];

    #[cfg(feature = "fixed-point")]
    let mut maxval: i32 = 1;
    // Pre-rotation of the folded pair (C f[2i], f[2i+1]), stored in bit-reversed order.
    let mut pre_rotate = |i: usize, re: KissFftScalar, im: KissFftScalar| {
        let t0 = trig0[i];
        let t1 = trig1[i];
        let yr = sub32(s_mul(re, t0), s_mul(im, t1));
        let yi = add32(s_mul(im, t0), s_mul(re, t1));
        // For QEXT, it's best to scale before the FFT, but otherwise it's best to scale
        // after. For floating-point it doesn't matter.
        #[cfg(feature = "qext")]
        let yc = KissFftCpx { r: yr, i: yi };
        #[cfg(not(feature = "qext"))]
        let yc = KissFftCpx {
            r: s_mul2(yr, scale),
            i: s_mul2(yi, scale),
        };
        #[cfg(feature = "fixed-point")]
        {
            maxval = max32(maxval, max32(abs32(yc.r), abs32(yc.i)));
        }
        f2[bitrev[i] as usize] = yc;
    };

    let ov2 = overlap >> 1;
    let k0 = (overlap + 3) >> 2;
    let input = &input[..n2 + overlap];
    let window = &window[..overlap];
    // Consider the input to be composed of four blocks: [a, b, c, d]
    // Window, shuffle, fold
    {
        // xp1 = in + ov2 + 2*i, xp2 = in + N2 - 1 + ov2 - 2*i in all three loops.
        let mut i = 0usize;
        while i < k0 {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            let wp1 = ov2 + 2 * i;
            let wp2 = ov2 - 1 - 2 * i;
            // Real part arranged as -d-cR, Imag part arranged as -b+aR
            let re = add32(
                s_mul(input[xp1 + n2], window[wp2]),
                s_mul(input[xp2], window[wp1]),
            );
            let im = sub32(
                s_mul(input[xp1], window[wp1]),
                s_mul(input[xp2 - n2], window[wp2]),
            );
            pre_rotate(i, re, im);
            i += 1;
        }
        while i < n4.saturating_sub(k0) {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            // Real part arranged as a-bR, Imag part arranged as -c-dR
            pre_rotate(i, input[xp2], input[xp1]);
            i += 1;
        }
        // wp1 = window + 2*j, wp2 = window + overlap - 1 - 2*j
        let mut j = 0usize;
        while i < n4 {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            let wp1 = 2 * j;
            let wp2 = overlap - 1 - 2 * j;
            // Real part arranged as a-bR, Imag part arranged as -c-dR
            let re = add32(
                neg32(s_mul(input[xp1 - n2], window[wp1])),
                s_mul(input[xp2], window[wp2]),
            );
            let im = add32(
                s_mul(input[xp1], window[wp2]),
                s_mul(input[xp2 + n2], window[wp1]),
            );
            pre_rotate(i, re, im);
            i += 1;
            j += 1;
        }
    }
    #[cfg(feature = "fixed-point")]
    let headroom = imax(0, imin(scale_shift, 28 - celt_ilog2(maxval)));
    #[cfg(feature = "fixed-point")]
    let downshift = scale_shift - headroom;
    #[cfg(not(feature = "fixed-point"))]
    let (headroom, downshift) = (0, 0);

    // N/4 complex FFT, does not downscale anymore
    opus_fft_impl_buf(st, f2, downshift);

    // Post-rotate
    {
        // out[2*stride*i] and out[stride*(N2-1-2i)]
        let out = &mut out[..stride * (n2 - 1) + 1];
        for (i, ((fp, &t0), &t1)) in f2.iter().zip(trig0).zip(trig1).enumerate() {
            #[cfg(feature = "qext")]
            let (t0, t1) = (s_mul2(t0, scale), s_mul2(t1, scale));
            let yr = pshr32(sub32(s_mul(fp.i, t1), s_mul(fp.r, t0)), headroom);
            let yi = pshr32(add32(s_mul(fp.r, t1), s_mul(fp.i, t0)), headroom);
            out[2 * stride * i] = yr;
            out[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }
}

/// One post-rotation step of [`clt_mdct_backward`] for the pair `p0 = y[2i..2i+2]` and
/// `p1 = y[N2-2-2i..N2-2i]` (pair `N4-1-i`), with `t[i], t[N4+i]` (`ta`) and
/// `t[N4-i-1], t[N2-i-1]` (`tb`). Both pairs are read before either is written, as in C.
/// Returns the new `[p0, p1]`.
#[inline(always)]
#[cfg_attr(
    not(feature = "fixed-point"),
    expect(
        clippy::missing_const_for_fn,
        reason = "shared with the fixed-point build, whose arithmetic macros are not const"
    )
)]
fn backward_post_step(
    p0: [KissFftScalar; 2],
    p1: [KissFftScalar; 2],
    ta: (KissTwiddleScalar, KissTwiddleScalar),
    tb: (KissTwiddleScalar, KissTwiddleScalar),
    post_shift: i32,
) -> [[KissFftScalar; 2]; 2] {
    // We swap real and imag because we're using an FFT instead of an IFFT.
    let (re, im) = (p0[1], p0[0]);
    let (t0, t1) = ta;
    // We'd scale up by 2 here, but instead it's done when mixing the windows
    let yr0 = pshr32_ovflw(add32_ovflw(s_mul(re, t0), s_mul(im, t1)), post_shift);
    let yi0 = pshr32_ovflw(sub32_ovflw(s_mul(re, t1), s_mul(im, t0)), post_shift);
    // We swap real and imag because we're using an FFT instead of an IFFT.
    let (re, im) = (p1[1], p1[0]);
    let (t0, t1) = tb;
    // We'd scale up by 2 here, but instead it's done when mixing the windows
    let yr1 = pshr32_ovflw(add32_ovflw(s_mul(re, t0), s_mul(im, t1)), post_shift);
    let yi1 = pshr32_ovflw(sub32_ovflw(s_mul(re, t1), s_mul(im, t0)), post_shift);
    // y[yp0] = yr0, y[yp0+1] = yi1, y[yp1] = yr1, y[yp1+1] = yi0
    [[yr0, yi1], [yr1, yi0]]
}

/// Port of celt/mdct.c:clt_mdct_backward_c (the `clt_mdct_backward` macro).
///
/// Backward MDCT (no scaling) of `input[0], input[stride], ...` (`N/2` values,
/// `N = l.n >> shift`) followed by the TDAC windowing of the first `overlap` output samples.
/// Writes `out[0 .. overlap/2 + N/2]` (and reads `out[0..overlap/2]`, which must hold the
/// previous overlap as in C).
pub fn clt_mdct_backward(
    l: &MdctLookup,
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let mut n = l.n as usize;
    let mut trig_off = 0usize;
    for _ in 0..shift {
        n >>= 1;
        trig_off += n;
    }
    let n2 = n >> 1;
    let n4 = n >> 2;
    let t = &l.trig[trig_off..trig_off + n2];
    let ov2 = overlap >> 1;
    // x1 = in[2*stride*i], x2 = in[stride*(N2-1-2i)]
    let input = &input[..stride * (n2 - 1) + 1];

    #[cfg(feature = "fixed-point")]
    let (pre_shift, post_shift, fft_shift) = {
        let mut sumval: i32 = n2 as i32;
        let mut maxval: i32 = 0;
        for &x in input.iter().step_by(stride) {
            maxval = max32(maxval, abs32(x));
            sumval = add32_ovflw(sumval, abs32(shr32(x, 11)));
        }
        let pre_shift = imax(0, 29 - celt_zlog2(1 + maxval));
        // Worst-case where all the energy goes to a single sample.
        let post_shift = imax(0, 19 - celt_ilog2(abs32(sumval)));
        let post_shift = imin(post_shift, pre_shift);
        (pre_shift, post_shift, pre_shift - post_shift)
    };
    #[cfg(not(feature = "fixed-point"))]
    let (pre_shift, post_shift, fft_shift) = (0, 0, 0);

    // Pre-rotate
    {
        // Perf: iterators and pair views instead of indexing, same arithmetic.
        // t[i] and t[N4 + i]
        let (t0s, t1s) = t.split_at(n4);
        let yp = out[ov2..ov2 + n2].as_chunks_mut::<2>().0;
        let bitrev = &l.kfft[shift].bitrev[..n4];
        for (i, ((&rev, &t0), &t1)) in bitrev.iter().zip(t0s).zip(t1s).enumerate() {
            let x1 = shl32_ovflw(input[2 * stride * i], pre_shift);
            let x2 = shl32_ovflw(input[stride * (n2 - 1 - 2 * i)], pre_shift);
            let yr = add32_ovflw(s_mul(x2, t0), s_mul(x1, t1));
            let yi = sub32_ovflw(s_mul(x1, t0), s_mul(x2, t1));
            // We swap real and imag because we use an FFT instead of an IFFT.
            // Storing the pre-rotation directly in the bitrev order.
            yp[rev as usize] = [yi, yr];
        }
    }

    opus_fft_impl_buf(
        &l.kfft[shift],
        &mut Interleaved(&mut out[ov2..ov2 + n2]),
        fft_shift,
    );

    // Post-rotate and de-shuffle from both ends of the buffer at once to make it in-place.
    {
        let (t0s, t1s) = t.split_at(n4);
        let y = out[ov2..ov2 + n2].as_chunks_mut::<2>().0;
        // Loop to (N4+1)>>1 to handle odd N4. When N4 is odd, the middle pair is computed
        // twice (here: once, with the same result).
        let half = n4 >> 1;
        let (front, rest) = y.split_at_mut(half);
        let (mid, back) = rest.split_at_mut(n4 - 2 * half);
        let ta = t0s.iter().zip(t1s);
        let tb = t0s.iter().rev().zip(t1s.iter().rev());
        for (((p0, p1), (&ta0, &ta1)), (&tb0, &tb1)) in
            front.iter_mut().zip(back.iter_mut().rev()).zip(ta).zip(tb)
        {
            [*p0, *p1] = backward_post_step(*p0, *p1, (ta0, ta1), (tb0, tb1), post_shift);
        }
        if let Some(p) = mid.first_mut() {
            // yp0 == yp1: both halves of the step read the original pair.
            let th = (t0s[half], t1s[half]);
            let [a, b] = backward_post_step(*p, *p, th, th, post_shift);
            // C writes y[yp0] = yr0, y[yp1+1] = yi0, then y[yp1] = yr1, y[yp0+1] = yi1.
            *p = [b[0], a[1]];
        }
    }

    // Mirror on both sides for TDAC
    {
        // x2 = out[i] and x1 = out[overlap-1-i] with window[i] (`w1`) and
        // window[overlap-1-i] (`w2`), for i < overlap/2.
        let h = overlap / 2;
        let (front, back) = out[..overlap].split_at_mut(overlap - h);
        let (wfront, wback) = window[..overlap].split_at(overlap - h);
        for (((yp1, xp1), &w1), &w2) in front[..h]
            .iter_mut()
            .zip(back.iter_mut().rev())
            .zip(&wfront[..h])
            .zip(wback.iter().rev())
        {
            let x1 = *xp1;
            let x2 = *yp1;
            *yp1 = sub32_ovflw(s_mul(x2, w2), s_mul(x1, w1));
            *xp1 = add32_ovflw(s_mul(x2, w1), s_mul(x1, w2));
        }
    }
}

//! Port of celt/mdct.c, celt/mdct.h (float build).
//!
//! [`MdctLookup`] lives in [`crate::celt::static_modes`]. `clt_mdct_clear` has no Rust
//! counterpart: an owned lookup is released by `Drop`.

use super::kiss_fft::{Interleaved, KissFftScalar, opus_fft_impl_buf};
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
        scale: 0.0,
        shift: 0,
        factors: [0; 2 * MAXFACTORS],
        bitrev: Cow::Borrowed(&[]),
        twiddles: Cow::Borrowed(&[]),
    }
}

/// Port of celt/mdct.c:clt_mdct_init (float build).
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
    let mut trig = vec![0.0f32; (n - (n2 >> maxshift)) as usize];
    let mut off = 0usize;
    for _shift in 0..=maxshift {
        // We have enough points that sine isn't necessary
        // FIXED_POINT: not ported (float build).
        for i in 0..n2 {
            trig[off + i as usize] =
                crate::math::cos(2.0 * super::mathops::PI * (i as f64 + 0.125) / big_n as f64)
                    as super::kiss_fft::KissTwiddleScalar;
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

/// Port of celt/mdct.c:clt_mdct_forward_c (the `clt_mdct_forward` macro), float build.
///
/// Forward MDCT of `input` (`N/2 + overlap` samples, `N = l.n >> shift`), scaled by `4/N`,
/// written to `out[0], out[stride], ...` (`N/2` values). Unlike the C prototype the input is
/// not trashed (the float build never writes it), so it is taken by shared reference.
pub fn clt_mdct_forward(
    l: &MdctLookup,
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[f32],
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
    trig: &[f32],
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[f32],
    overlap: usize,
    stride: usize,
    f2: &mut [KissFftCpx],
) {
    let scale = st.scale;
    let n4 = f2.len();
    let n2 = 2 * n4;
    // trig[i] and trig[N4 + i]
    let (trig0, trig1) = trig[..n2].split_at(n4);
    let bitrev = &st.bitrev[..n4];

    // Pre-rotation of the folded pair (C f[2i], f[2i+1]), stored in bit-reversed order.
    let mut pre_rotate = |i: usize, re: KissFftScalar, im: KissFftScalar| {
        let t0 = trig0[i];
        let t1 = trig1[i];
        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;
        // For QEXT, it's best to scale before the FFT, but otherwise it's best to scale
        // after. For floating-point it doesn't matter.
        #[cfg(feature = "qext")]
        let yc = KissFftCpx { r: yr, i: yi };
        #[cfg(not(feature = "qext"))]
        let yc = KissFftCpx {
            r: yr * scale,
            i: yi * scale,
        };
        // FIXED_POINT: not ported (float build) — maxval/headroom tracking.
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
            let re = input[xp1 + n2] * window[wp2] + input[xp2] * window[wp1];
            let im = input[xp1] * window[wp1] - input[xp2 - n2] * window[wp2];
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
            let re = -(input[xp1 - n2] * window[wp1]) + input[xp2] * window[wp2];
            let im = input[xp1] * window[wp2] + input[xp2 + n2] * window[wp1];
            pre_rotate(i, re, im);
            i += 1;
            j += 1;
        }
    }

    // N/4 complex FFT, does not downscale anymore
    opus_fft_impl_buf(st, f2);

    // Post-rotate
    {
        // out[2*stride*i] and out[stride*(N2-1-2i)]
        let out = &mut out[..stride * (n2 - 1) + 1];
        for (i, ((fp, &t0), &t1)) in f2.iter().zip(trig0).zip(trig1).enumerate() {
            #[cfg(feature = "qext")]
            let (t0, t1) = (t0 * scale, t1 * scale);
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            out[2 * stride * i] = yr;
            out[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }
}

/// Port of celt/mdct.c:clt_mdct_backward_c (the `clt_mdct_backward` macro), float build.
///
/// Backward MDCT (no scaling) of `input[0], input[stride], ...` (`N/2` values,
/// `N = l.n >> shift`) followed by the TDAC windowing of the first `overlap` output samples.
/// Writes `out[0 .. overlap/2 + N/2]` (and reads `out[0..overlap/2]`, which must hold the
/// previous overlap as in C).
pub fn clt_mdct_backward(
    l: &MdctLookup,
    input: &[KissFftScalar],
    out: &mut [KissFftScalar],
    window: &[f32],
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

    // FIXED_POINT: not ported (float build) — pre_shift/post_shift/fft_shift computation.

    // Pre-rotate
    {
        // Perf: iterators and pair views instead of indexing, same arithmetic.
        // t[i] and t[N4 + i]
        let (t0s, t1s) = t.split_at(n4);
        let yp = out[ov2..ov2 + n2].as_chunks_mut::<2>().0;
        let bitrev = &l.kfft[shift].bitrev[..n4];
        // x1 = in[2*stride*i], x2 = in[stride*(N2-1-2i)]
        let input = &input[..stride * (n2 - 1) + 1];
        for (i, ((&rev, &t0), &t1)) in bitrev.iter().zip(t0s).zip(t1s).enumerate() {
            let x1 = input[2 * stride * i];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            // We swap real and imag because we use an FFT instead of an IFFT.
            // Storing the pre-rotation directly in the bitrev order.
            yp[rev as usize] = [yi, yr];
        }
    }

    opus_fft_impl_buf(&l.kfft[shift], &mut Interleaved(&mut out[ov2..ov2 + n2]));

    // Post-rotate and de-shuffle from both ends of the buffer at once to make it in-place.
    {
        let (t0s, t1s) = t.split_at(n4);
        let y = out[ov2..ov2 + n2].as_chunks_mut::<2>().0;
        // One step for the pair yp0 = y[2i..2i+2] and yp1 = y[N2-2-2i..N2-2i] (pair N4-1-i),
        // with t[i], t[N4+i] (`ta`) and t[N4-i-1], t[N2-i-1] (`tb`). Both pairs are read
        // before either is written, as in C.
        #[inline(always)]
        fn step(p0: [f32; 2], p1: [f32; 2], ta: (f32, f32), tb: (f32, f32)) -> [[f32; 2]; 2] {
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let (re, im) = (p0[1], p0[0]);
            let (t0, t1) = ta;
            // We'd scale up by 2 here, but instead it's done when mixing the windows
            let yr0 = re * t0 + im * t1;
            let yi0 = re * t1 - im * t0;
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let (re, im) = (p1[1], p1[0]);
            let (t0, t1) = tb;
            // We'd scale up by 2 here, but instead it's done when mixing the windows
            let yr1 = re * t0 + im * t1;
            let yi1 = re * t1 - im * t0;
            // y[yp0] = yr0, y[yp0+1] = yi1, y[yp1] = yr1, y[yp1+1] = yi0
            [[yr0, yi1], [yr1, yi0]]
        }
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
            [*p0, *p1] = step(*p0, *p1, (ta0, ta1), (tb0, tb1));
        }
        if let Some(p) = mid.first_mut() {
            // yp0 == yp1: both halves of the step read the original pair.
            let [a, b] = step(*p, *p, (t0s[half], t1s[half]), (t0s[half], t1s[half]));
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
            *yp1 = x2 * w2 - x1 * w1;
            *xp1 = x2 * w1 + x1 * w2;
        }
    }
}

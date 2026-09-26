//! Port of celt/mdct.c, celt/mdct.h (float build).
//!
//! [`MdctLookup`] lives in [`crate::celt::static_modes`]. `clt_mdct_clear` has no Rust
//! counterpart: an owned lookup is released by `Drop`.

use super::kiss_fft::{Interleaved, KissFftScalar, opus_fft_impl_buf};
use super::static_modes::{KissFftCpx, MdctLookup};

#[cfg(feature = "custom-modes")]
use super::static_modes::{KissFftState, MAXFACTORS};
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
    let scale = st.scale;

    let mut n = l.n as usize;
    let mut trig_off = 0usize;
    for _ in 0..shift {
        n >>= 1;
        trig_off += n;
    }
    let n2 = n >> 1;
    let n4 = n >> 2;
    let trig = &l.trig[trig_off..];

    debug_assert!(n2 <= MAX_MDCT_N2);
    let mut f_buf = [0.0 as KissFftScalar; MAX_MDCT_N2];
    let mut f2_buf = [KissFftCpx::default(); MAX_MDCT_N2 / 2];
    let f = &mut f_buf[..n2];
    let f2 = &mut f2_buf[..n4];

    let ov2 = overlap >> 1;
    let k0 = (overlap + 3) >> 2;
    // Consider the input to be composed of four blocks: [a, b, c, d]
    // Window, shuffle, fold
    {
        // xp1 = in + ov2 + 2*i, xp2 = in + N2 - 1 + ov2 - 2*i in all three loops.
        let mut i = 0usize;
        let mut yp = 0usize;
        while i < k0 {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            let wp1 = ov2 + 2 * i;
            let wp2 = ov2 - 1 - 2 * i;
            // Real part arranged as -d-cR, Imag part arranged as -b+aR
            f[yp] = input[xp1 + n2] * window[wp2] + input[xp2] * window[wp1];
            f[yp + 1] = input[xp1] * window[wp1] - input[xp2 - n2] * window[wp2];
            yp += 2;
            i += 1;
        }
        while i < n4.saturating_sub(k0) {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            // Real part arranged as a-bR, Imag part arranged as -c-dR
            f[yp] = input[xp2];
            f[yp + 1] = input[xp1];
            yp += 2;
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
            f[yp] = -(input[xp1 - n2] * window[wp1]) + input[xp2] * window[wp2];
            f[yp + 1] = input[xp1] * window[wp2] + input[xp2 + n2] * window[wp1];
            yp += 2;
            i += 1;
            j += 1;
        }
    }
    // Pre-rotation
    {
        let bitrev = &st.bitrev[..n4];
        for i in 0..n4 {
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let re = f[2 * i];
            let im = f[2 * i + 1];
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
        }
    }

    // N/4 complex FFT, does not downscale anymore
    opus_fft_impl_buf(st, f2);

    // Post-rotate
    {
        for (i, fp) in f2.iter().enumerate() {
            #[cfg(feature = "qext")]
            let (t0, t1) = (trig[i] * scale, trig[n4 + i] * scale);
            #[cfg(not(feature = "qext"))]
            let (t0, t1) = (trig[i], trig[n4 + i]);
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
        let yp = &mut out[ov2..ov2 + n2];
        let bitrev = &l.kfft[shift].bitrev[..n4];
        for (i, &rev) in bitrev.iter().enumerate() {
            let rev = rev as usize;
            let x1 = input[2 * stride * i];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let yr = x2 * t[i] + x1 * t[n4 + i];
            let yi = x1 * t[i] - x2 * t[n4 + i];
            // We swap real and imag because we use an FFT instead of an IFFT.
            yp[2 * rev + 1] = yr;
            yp[2 * rev] = yi;
            // Storing the pre-rotation directly in the bitrev order.
        }
    }

    opus_fft_impl_buf(&l.kfft[shift], &mut Interleaved(&mut out[ov2..ov2 + n2]));

    // Post-rotate and de-shuffle from both ends of the buffer at once to make it in-place.
    {
        let y = &mut out[ov2..ov2 + n2];
        // yp0 = 2*i, yp1 = N2 - 2 - 2*i.
        // Loop to (N4+1)>>1 to handle odd N4. When N4 is odd, the middle pair will be computed
        // twice.
        for i in 0..(n4 + 1) >> 1 {
            let yp0 = 2 * i;
            let yp1 = n2 - 2 - 2 * i;
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let re = y[yp0 + 1];
            let im = y[yp0];
            let t0 = t[i];
            let t1 = t[n4 + i];
            // We'd scale up by 2 here, but instead it's done when mixing the windows
            let yr = re * t0 + im * t1;
            let yi = re * t1 - im * t0;
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let re = y[yp1 + 1];
            let im = y[yp1];
            y[yp0] = yr;
            y[yp1 + 1] = yi;

            let t0 = t[n4 - i - 1];
            let t1 = t[n2 - i - 1];
            // We'd scale up by 2 here, but instead it's done when mixing the windows
            let yr = re * t0 + im * t1;
            let yi = re * t1 - im * t0;
            y[yp1] = yr;
            y[yp0 + 1] = yi;
        }
    }

    // Mirror on both sides for TDAC
    {
        let out = &mut out[..overlap];
        let window = &window[..overlap];
        for i in 0..overlap / 2 {
            let xp1 = overlap - 1 - i;
            let x1 = out[xp1];
            let x2 = out[i];
            out[i] = x2 * window[overlap - 1 - i] - x1 * window[i];
            out[xp1] = x2 * window[i] + x1 * window[overlap - 1 - i];
        }
    }
}

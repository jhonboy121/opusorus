//! Vertical SIMD (`fearless_simd`) kernels for the MDCT rotations (celt/mdct.c), float and
//! fixed point, over the FFT's lane arithmetic [`Arith`] (see [`super::fft_simd`] and
//! [`crate::simd`]). Each lane runs the scalar operation sequence on an independent pair.

use super::fft_simd::{Arith, with_arith};
use super::static_modes::{KissFftScalar, KissTwiddleScalar};

/// Four complex values, one per lane (`r` and `i` deinterleaved).
#[derive(Clone, Copy)]
struct C4<V> {
    r: V,
    i: V,
}

/// Loads the four consecutive complex values `k..k+4` of the interleaved buffer.
#[inline(always)]
fn ld<A: Arith>(a: A, buf: &[KissFftScalar], k: usize) -> C4<A::V> {
    let x = &buf[2 * k..2 * k + 8];
    let (r, i) = a.deint(a.ld(&x[..4]), a.ld(&x[4..]));
    C4 { r, i }
}

/// Stores four values to the consecutive complex positions `k..k+4`.
#[inline(always)]
fn st<A: Arith>(a: A, buf: &mut [KissFftScalar], k: usize, v: C4<A::V>) {
    let x = &mut buf[2 * k..2 * k + 8];
    let (p, q) = a.inter(v.r, v.i);
    a.st(p, &mut x[..4]);
    a.st(q, &mut x[4..]);
}

/// Loads `x[k..k+4]`.
#[inline(always)]
fn ld4<A: Arith>(a: A, x: &[KissFftScalar], k: usize) -> A::V {
    a.ld(&x[k..k + 4])
}

/// Twiddle lanes `t[k..k+4]`.
#[inline(always)]
fn tw4<A: Arith>(a: A, t: &[KissTwiddleScalar], k: usize) -> A::T {
    a.tw(core::array::from_fn(|j| t[k + j]))
}

/// Twiddle lanes `t[k], t[k-1], t[k-2], t[k-3]`.
#[inline(always)]
fn tw4_rev<A: Arith>(a: A, t: &[KissTwiddleScalar], k: usize) -> A::T {
    a.tw(core::array::from_fn(|j| t[k - j]))
}

/// Whether no value of `t` is `-32768` (the NEON Q15 multiply's condition, see
/// [`with_arith!`]); always true in the float and QEXT builds.
#[inline(always)]
#[cfg_attr(
    not(all(feature = "fixed-point", not(feature = "qext"))),
    expect(clippy::missing_const_for_fn, reason = "the fixed-point body iterates")
)]
fn q15_ok(t: &[KissTwiddleScalar]) -> bool {
    #[cfg(all(feature = "fixed-point", not(feature = "qext")))]
    {
        t.iter().all(|&v| v > i16::MIN)
    }
    #[cfg(not(all(feature = "fixed-point", not(feature = "qext"))))]
    {
        let _ = t;
        true
    }
}

/// Items a kernel handled: `None` means the target has no SIMD (not an error), so nothing.
#[inline(always)]
const fn done_or_none(r: Option<usize>) -> usize {
    match r {
        Some(done) => done,
        None => 0,
    }
}

/// Whether the NEON Q15 multiply may be used with `l`'s twiddles (see [`with_arith!`]): the
/// static tables qualify (checked by the tests), runtime ones (custom modes) are scanned.
pub(crate) fn trig_q15_ok(l: &super::static_modes::MdctLookup) -> bool {
    match &l.trig {
        alloc::borrow::Cow::Borrowed(_) => true,
        alloc::borrow::Cow::Owned(t) => q15_ok(t),
    }
}

/// Pre-rotation of `clt_mdct_forward` for the middle block of the fold (`k0 <= i < n4 - k0`,
/// where the folded pair is `re = in[N2-1+ov2-2i]`, `im = in[ov2+2i]`), four pairs per step,
/// stored to `f2` (interleaved) in bit-reversed order (scaled by `scale` if given, the
/// non-QEXT `S_MUL2`). Handles `i` in `[k0, k0 + 4*j)` for the largest such `j` and returns
/// that end (`k0` without SIMD) and, in the fixed-point build, the `maxval` of the stored values
/// (at least 0; the caller takes the maximum with its own). `trig_q15_ok`: no trig value is
/// `-32768` (fixed point, see [`with_arith!`]).
#[expect(clippy::too_many_arguments, reason = "the state of the C loop")]
#[inline]
pub(crate) fn mdct_forward_pre_mid(
    input: &[KissFftScalar],
    trig0: &[KissTwiddleScalar],
    trig1: &[KissTwiddleScalar],
    bitrev: &[i16],
    scale: Option<KissTwiddleScalar>,
    f2: &mut [KissFftScalar],
    n2: usize,
    ov2: usize,
    (k0, k1): (usize, usize),
    trig_q15_ok: bool,
) -> (usize, KissFftScalar) {
    let r = with_arith!(trig_q15_ok; a => {
        let pairs = f2.as_chunks_mut::<2>().0;
        let mut i = k0;
        #[cfg(feature = "fixed-point")]
        let mut maxv = a.lanes([0; 4]);
        while i + 4 <= k1 {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            // Evens of in[xp1..xp1+8] and in[xp2], in[xp2-2], in[xp2-4], in[xp2-6].
            let (im, _) = a.deint(ld4(a, input, xp1), ld4(a, input, xp1 + 4));
            let (_, re) = a.deint(ld4(a, input, xp2 - 7), ld4(a, input, xp2 - 3));
            let re = a.rev(re);
            let t0 = tw4(a, trig0, i);
            let t1 = tw4(a, trig1, i);
            let mut yr = a.sub(a.mul(re, t0), a.mul(im, t1));
            let mut yi = a.add(a.mul(im, t0), a.mul(re, t1));
            if let Some(scale) = scale {
                yr = a.mul2(yr, scale);
                yi = a.mul2(yi, scale);
            }
            #[cfg(feature = "fixed-point")]
            {
                maxv = a.max_abs(maxv, yr, yi);
            }
            let (yr, yi) = (a.array(yr), a.array(yi));
            for k in 0..4 {
                pairs[bitrev[i + k] as usize] = [yr[k], yi[k]];
            }
            i += 4;
        }
        #[cfg(feature = "fixed-point")]
        let m = a.array(maxv).into_iter().fold(0, i32::max);
        #[cfg(not(feature = "fixed-point"))]
        let m = 0.0;
        (i, m)
    });
    match r {
        Some(r) => r,
        None => (k0, KissFftScalar::default()),
    }
}

/// Post-rotation of `clt_mdct_forward` for the pairs `0..4*(N4/4)`:
/// `out[2*stride*i] = yr`, `out[stride*(N2-1-2i)] = yi`, shifted by `headroom` (fixed point).
/// With QEXT (`scale` given) the twiddles are scaled first. Returns the number of pairs done
/// (0 without SIMD).
#[expect(clippy::too_many_arguments, reason = "the state of the C loop")]
#[inline]
pub(crate) fn mdct_forward_post(
    f2: &[KissFftScalar],
    trig0: &[KissTwiddleScalar],
    trig1: &[KissTwiddleScalar],
    scale: Option<KissTwiddleScalar>,
    headroom: i32,
    out: &mut [KissFftScalar],
    stride: usize,
    trig_q15_ok: bool,
) -> usize {
    let n4 = f2.len() / 2;
    let n2 = 2 * n4;
    let r = with_arith!(trig_q15_ok; a => {
        let nv = n4 & !3;
        for i in (0..nv).step_by(4) {
            let fp = ld(a, f2, i);
            #[cfg(feature = "qext")]
            let (t0, t1) = match scale {
                Some(scale) => (
                    a.tw_from(a.mul2(a.lanes(core::array::from_fn(|j| trig0[i + j])), scale)),
                    a.tw_from(a.mul2(a.lanes(core::array::from_fn(|j| trig1[i + j])), scale)),
                ),
                None => (tw4(a, trig0, i), tw4(a, trig1, i)),
            };
            #[cfg(not(feature = "qext"))]
            let (t0, t1) = {
                let _ = scale;
                (tw4(a, trig0, i), tw4(a, trig1, i))
            };
            let yr = a.pshr(a.sub(a.mul(fp.i, t1), a.mul(fp.r, t0)), headroom);
            let yi = a.pshr(a.add(a.mul(fp.r, t1), a.mul(fp.i, t0)), headroom);
            let (yr, yi) = (a.array(yr), a.array(yi));
            for k in 0..4 {
                out[2 * stride * (i + k)] = yr[k];
                out[stride * (n2 - 1 - 2 * (i + k))] = yi[k];
            }
        }
        nv
    });
    done_or_none(r)
}

/// Pre-rotation of `clt_mdct_backward` for `i` in `0..4*(N4/4)`, with
/// `x1 = SHL32_ovflw(in[2*stride*i], pre_shift)`, `x2 = SHL32_ovflw(in[stride*(N2-1-2i)],
/// pre_shift)` (no shift in the float build), stored as `[yi, yr]` to pair `bitrev[i]` of `y`.
/// Returns the number of pairs done (0 without SIMD).
#[expect(clippy::too_many_arguments, reason = "the state of the C loop")]
#[inline]
pub(crate) fn mdct_backward_pre(
    input: &[KissFftScalar],
    stride: usize,
    pre_shift: i32,
    t0s: &[KissTwiddleScalar],
    t1s: &[KissTwiddleScalar],
    bitrev: &[i16],
    y: &mut [[KissFftScalar; 2]],
    trig_q15_ok: bool,
) -> usize {
    let n4 = bitrev.len();
    let n2 = 2 * n4;
    let r = with_arith!(trig_q15_ok; a => {
        let nv = n4 & !3;
        for i in (0..nv).step_by(4) {
            let (x1, x2) = if stride == 1 {
                // in[2i..2i+8] evens; in[N2-1-2i], in[N2-3-2i], ... (odd positions, descending).
                let (x1, _) = a.deint(ld4(a, input, 2 * i), ld4(a, input, 2 * i + 4));
                let top = n2 - 1 - 2 * i;
                let (_, x2) = a.deint(ld4(a, input, top - 7), ld4(a, input, top - 3));
                (x1, a.rev(x2))
            } else {
                let x1: [KissFftScalar; 4] = core::array::from_fn(|k| input[2 * stride * (i + k)]);
                let x2: [KissFftScalar; 4] =
                    core::array::from_fn(|k| input[stride * (n2 - 1 - 2 * (i + k))]);
                (a.lanes(x1), a.lanes(x2))
            };
            let (x1, x2) = (a.shl(x1, pre_shift), a.shl(x2, pre_shift));
            let t0 = tw4(a, t0s, i);
            let t1 = tw4(a, t1s, i);
            let yr = a.add(a.mul(x2, t0), a.mul(x1, t1));
            let yi = a.sub(a.mul(x1, t0), a.mul(x2, t1));
            let (yr, yi) = (a.array(yr), a.array(yi));
            for k in 0..4 {
                y[bitrev[i + k] as usize] = [yi[k], yr[k]];
            }
        }
        nv
    });
    done_or_none(r)
}

/// Post-rotation of `clt_mdct_backward` for the pair steps `0..4*(half/4)` (`half = N4/2`):
/// step `i` works on pairs `i` and `N4-1-i` of `y` with the twiddles `t[i], t[N4+i]` and
/// `t[N4-1-i], t[N2-1-i]`, both pairs read before either is written; results shifted by
/// `post_shift` (`PSHR32_ovflw`, fixed point). Returns the number of steps done (0 without
/// SIMD).
#[inline]
pub(crate) fn mdct_backward_post(
    y: &mut [KissFftScalar],
    t0s: &[KissTwiddleScalar],
    t1s: &[KissTwiddleScalar],
    post_shift: i32,
    trig_q15_ok: bool,
) -> usize {
    let n4 = t0s.len();
    let half = n4 >> 1;
    let r = with_arith!(trig_q15_ok; a => {
        let nv = half & !3;
        for i in (0..nv).step_by(4) {
            // Pairs i..i+4 (front) and N4-4-i..N4-i (back, lanes reversed so lane k is pair
            // N4-1-i-k).
            let j = n4 - 4 - i;
            let p0 = ld(a, y, i);
            let p1 = ld(a, y, j);
            let (p1r, p1i) = (a.rev(p1.r), a.rev(p1.i));
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let (re, im) = (p0.i, p0.r);
            let (t0, t1) = (tw4(a, t0s, i), tw4(a, t1s, i));
            let yr0 = a.pshr(a.add(a.mul(re, t0), a.mul(im, t1)), post_shift);
            let yi0 = a.pshr(a.sub(a.mul(re, t1), a.mul(im, t0)), post_shift);
            let (re, im) = (p1i, p1r);
            let (t0, t1) = (tw4_rev(a, t0s, n4 - 1 - i), tw4_rev(a, t1s, n4 - 1 - i));
            let yr1 = a.pshr(a.add(a.mul(re, t0), a.mul(im, t1)), post_shift);
            let yi1 = a.pshr(a.sub(a.mul(re, t1), a.mul(im, t0)), post_shift);
            // y[yp0] = yr0, y[yp0+1] = yi1, y[yp1] = yr1, y[yp1+1] = yi0
            st(a, y, i, C4 { r: yr0, i: yi1 });
            st(a, y, j, C4 { r: a.rev(yr1), i: a.rev(yi0) });
        }
        nv
    });
    done_or_none(r)
}

/// TDAC mirror of `clt_mdct_backward` for `i` in `0..4*(h/4)` (`h = overlap/2`):
/// `x1 = out[overlap-1-i]`, `x2 = out[i]` with `w1 = window[i]`, `w2 = window[overlap-1-i]`.
/// Returns the number of `i` done (0 without SIMD).
#[inline]
pub(crate) fn mdct_backward_mirror(
    out: &mut [KissFftScalar],
    window: &[KissTwiddleScalar],
) -> usize {
    let overlap = window.len();
    let h = overlap / 2;
    let r = with_arith!(q15_ok(window); a => {
        let nv = h & !3;
        for i in (0..nv).step_by(4) {
            let hi = overlap - 1 - i;
            let x1 = a.rev(ld4(a, out, hi - 3));
            let x2 = ld4(a, out, i);
            let w1 = tw4(a, window, i);
            let w2 = tw4_rev(a, window, hi);
            let lo = a.sub(a.mul(x2, w2), a.mul(x1, w1));
            let up = a.add(a.mul(x2, w1), a.mul(x1, w2));
            a.st(lo, &mut out[i..i + 4]);
            a.st(a.rev(up), &mut out[hi - 3..=hi]);
        }
        nv
    });
    done_or_none(r)
}

#[cfg(test)]
mod tests {
    //! SIMD == scalar for both MDCTs of the static modes, all shifts and strides, mode and
    //! random windows (fixed point: including `-32768`, which takes the portable multiply), on
    //! every target (the oracle tests only run on the host).

    use alloc::{format, vec::Vec};

    use crate::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
    use crate::celt::static_modes::{CeltMode, KissFftScalar, KissTwiddleScalar, STATIC_MODE_LIST};
    use crate::simd::assert_simd_eq_scalar;

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
        /// Signal values: float in ±32768, fixed point below 2^24 in magnitude.
        fn sig(&mut self) -> KissFftScalar {
            let v = (self.next() >> 7) as i32 - (1 << 24);
            #[cfg(feature = "fixed-point")]
            {
                v
            }
            #[cfg(not(feature = "fixed-point"))]
            {
                v as f32 / 512.0
            }
        }
        /// Window values: float in ±1, fixed point any `celt_coef`.
        fn coef(&mut self) -> KissTwiddleScalar {
            #[cfg(feature = "fixed-point")]
            {
                self.next() as KissTwiddleScalar
            }
            #[cfg(not(feature = "fixed-point"))]
            {
                (self.next() >> 8) as f32 / (1 << 23) as f32 - 1.0
            }
        }
        fn sigs(&mut self, n: usize) -> Vec<KissFftScalar> {
            (0..n).map(|_| self.sig()).collect()
        }
    }

    fn show(x: &[KissFftScalar]) -> Vec<alloc::string::String> {
        x.iter().map(|v| format!("{v:?}")).collect()
    }

    fn modes() -> &'static [&'static CeltMode] {
        &STATIC_MODE_LIST
    }

    #[test]
    fn mdct_matches_scalar() {
        let mut rng = Lcg(2);
        for &mode in modes() {
            let overlap = mode.overlap as usize;
            for shift in 0..=mode.mdct.maxshift as usize {
                let n = mode.mdct.n as usize >> shift;
                let n2 = n / 2;
                for stride in [1usize, 2, 4, 8] {
                    for iter in 0..8 {
                        // Random windows/overlaps too (multiples of 4, and 0), as in the codec.
                        let ov = if iter < 4 {
                            overlap
                        } else {
                            [0, 4, 8 * iter, n2][iter - 4]
                        };
                        let window: Vec<KissTwiddleScalar> = if ov == overlap && iter < 2 {
                            mode.window.to_vec()
                        } else {
                            let w: Vec<KissTwiddleScalar> = (0..ov).map(|_| rng.coef()).collect();
                            // Fixed point: the NEON Q15 multiply's excluded value.
                            #[cfg(feature = "fixed-point")]
                            let w = {
                                let mut w = w;
                                if let Some(first) = w.first_mut() {
                                    *first = KissTwiddleScalar::MIN;
                                }
                                w
                            };
                            w
                        };
                        let input = rng.sigs(n2 + ov);
                        let out0 = rng.sigs(stride * n2);
                        assert_simd_eq_scalar(
                            &format!("forward n={n} stride={stride} ov={ov}"),
                            || {
                                let mut out = out0.clone();
                                clt_mdct_forward(
                                    &mode.mdct, &input, &mut out, &window, ov, shift, stride,
                                );
                                show(&out)
                            },
                        );
                        let input = rng.sigs(stride * n2);
                        let out0 = rng.sigs((n2 + ov / 2).max(ov) + 3);
                        assert_simd_eq_scalar(
                            &format!("backward n={n} stride={stride} ov={ov}"),
                            || {
                                let mut out = out0.clone();
                                clt_mdct_backward(
                                    &mode.mdct, &input, &mut out, &window, ov, shift, stride,
                                );
                                show(&out)
                            },
                        );
                    }
                }
            }
        }
    }

    /// The NEON Q15 multiply needs the static MDCT twiddles above `-32768`.
    #[cfg(all(feature = "fixed-point", not(feature = "qext")))]
    #[test]
    fn static_trig_above_min() {
        for &mode in modes() {
            assert!(mode.mdct.trig.iter().all(|&t| t > i16::MIN));
        }
    }
}

#[cfg(all(test, feature = "custom-modes"))]
mod custom_tests {
    //! SIMD == scalar for custom modes: runtime twiddles (fixed point: the portable Q15
    //! multiply, scanned trig tables), non-power-of-two FFT shapes, large signal values.

    use alloc::{format, vec, vec::Vec};

    use crate::celt::kiss_fft::{Interleaved, opus_fft_impl_buf};
    use crate::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
    use crate::celt::modes::opus_custom_mode_create_custom;
    use crate::celt::static_modes::KissFftScalar;
    use crate::simd::assert_simd_eq_scalar;

    #[test]
    fn custom_modes_match_scalar() {
        let mut seed = 1u32;
        let mut rnd = |mag: u32| -> KissFftScalar {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((seed >> 7) as i32 - (1 << 24)) << mag;
            #[cfg(feature = "fixed-point")]
            {
                v
            }
            #[cfg(not(feature = "fixed-point"))]
            {
                v as f32 / 512.0
            }
        };
        let configs = [
            (32000, 640),
            (44100, 882),
            (44100, 1024),
            (48000, 256),
            (24000, 400),
            (16000, 320),
            (48000, 720),
        ];
        let mut tested = 0;
        for (fs, fsz) in configs {
            // Some layouts are rejected (as by C); the rest must include runtime FFT shapes.
            let Ok(mode) = opus_custom_mode_create_custom(fs, fsz) else {
                continue;
            };
            tested += 1;
            // States beyond `maxshift` are empty placeholders, never used.
            for (k, st) in mode.mdct.kfft[..=mode.mdct.maxshift as usize]
                .iter()
                .enumerate()
            {
                let n = st.nfft as usize;
                let data: Vec<_> = (0..2 * n).map(|_| rnd(0)).collect();
                for ds in [0, 2, 5] {
                    assert_simd_eq_scalar(&format!("fft {fs}/{fsz} k={k} ds={ds}"), || {
                        let mut d = data.clone();
                        opus_fft_impl_buf(st, &mut Interleaved(&mut d), ds);
                        format!("{d:?}")
                    });
                }
            }
            let overlap = mode.overlap as usize;
            for mag in [0, 4] {
                for shift in 0..=mode.mdct.maxshift as usize {
                    let n2 = (mode.mdct.n as usize >> shift) / 2;
                    for stride in [1usize, 2, 4, 8] {
                        let what = format!("{fs}/{fsz} shift={shift} stride={stride} mag={mag}");
                        let input: Vec<_> = (0..n2 + overlap).map(|_| rnd(mag)).collect();
                        assert_simd_eq_scalar(&format!("forward {what}"), || {
                            let mut out = vec![KissFftScalar::default(); stride * n2];
                            clt_mdct_forward(
                                &mode.mdct,
                                &input,
                                &mut out,
                                &mode.window,
                                overlap,
                                shift,
                                stride,
                            );
                            format!("{out:?}")
                        });
                        let input: Vec<_> = (0..stride * n2).map(|_| rnd(mag)).collect();
                        assert_simd_eq_scalar(&format!("backward {what}"), || {
                            let mut out = vec![KissFftScalar::default(); n2 + overlap];
                            clt_mdct_backward(
                                &mode.mdct,
                                &input,
                                &mut out,
                                &mode.window,
                                overlap,
                                shift,
                                stride,
                            );
                            format!("{out:?}")
                        });
                    }
                }
            }
        }
        assert!(tested >= 4, "only {tested} custom modes could be created");
    }
}

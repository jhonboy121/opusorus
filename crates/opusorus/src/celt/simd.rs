//! Vertical SIMD (`fearless_simd`) kernels for the float FFT (celt/kiss_fft.c).
//!
//! Every lane executes exactly the scalar operation sequence of the port (same operations, same
//! order, no FMA contraction) on an independent butterfly, so the results are bit-identical to
//! the scalar code and to the C oracle. IEEE `+`/`*` are commutative and `(-x)*y == -(x*y)`,
//! which some kernels use to share work between lanes; nothing else is reassociated.
//!
//! The kernels run on the target's SIMD baseline (NEON on aarch64, SSE2 on x86/x86-64, SIMD128 on
//! wasm32 built with `+simd128`); elsewhere [`fft_impl`] reports that no SIMD is available and the
//! caller runs the scalar code.

use fearless_simd::{Select, Simd, SimdBase, SimdFrom, f32x4, mask32x4};

use super::kiss_fft::{Interleaved, kf_bfly2, kf_bfly3, kf_bfly4, kf_bfly5};
use super::static_modes::{KissFftState, KissTwiddleCpx};

/// Runs `$body` with the target's baseline SIMD token bound to `$s` (`Some(result)`), or returns
/// `None` if the target has no supported SIMD baseline.
macro_rules! with_simd {
    ($s:ident => $body:expr) => {{
        #[cfg(target_arch = "aarch64")]
        let token = fearless_simd::Level::baseline().as_neon();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let token = fearless_simd::Level::baseline().as_sse2();
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let token = fearless_simd::Level::baseline().as_wasm_simd128();
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        #[cfg(test)]
        let token =
            token.filter(|_| !tests::SCALAR_ONLY.load(core::sync::atomic::Ordering::Relaxed));
        let r = token.map(|$s| {
            $s.vectorize(
                #[inline(always)]
                || $body,
            )
        });
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        // No SIMD: the caller runs the scalar code. The body is still type-checked (with the
        // scalar fallback token, never executed) so it stays warning-free on these targets.
        let r = if false {
            let $s = fearless_simd::Fallback::new();
            Some($body)
        } else {
            None
        };
        r
    }};
}

/// [`with_simd!`] for kernels that do a prefix of the work: evaluates to the `$body` result
/// (how far the kernel got) with SIMD, and to `$none` (nothing done) without.
macro_rules! simd_done {
    ($none:expr; $s:ident => $body:expr) => {
        match with_simd!($s => $body) {
            Some(done) => done,
            None => $none,
        }
    };
}

/// Four complex values, one per lane (`r` and `i` deinterleaved).
#[derive(Clone, Copy)]
struct C4<S: Simd> {
    r: f32x4<S>,
    i: f32x4<S>,
}

/// Loads the four consecutive complex values `k..k+4` of the interleaved buffer.
#[inline(always)]
fn ld<S: Simd>(s: S, buf: &[f32], k: usize) -> C4<S> {
    let x = &buf[2 * k..2 * k + 8];
    let (r, i) = s.deinterleave_f32x4(f32x4::from_slice(s, &x[..4]), f32x4::from_slice(s, &x[4..]));
    C4 { r, i }
}

/// Stores four values to the consecutive complex positions `k..k+4`.
#[inline(always)]
fn st<S: Simd>(s: S, buf: &mut [f32], k: usize, v: C4<S>) {
    let x = &mut buf[2 * k..2 * k + 8];
    let (a, b) = s.interleave_f32x4(v.r, v.i);
    a.store_slice(&mut x[..4]);
    b.store_slice(&mut x[4..]);
}

/// Gathers the twiddles `tw[idx]`, `tw[idx + step]`, `tw[idx + 2*step]`, `tw[idx + 3*step]`.
#[inline(always)]
fn ld_tw<S: Simd>(s: S, tw: &[KissTwiddleCpx], idx: usize, step: usize) -> C4<S> {
    let t = [
        tw[idx],
        tw[idx + step],
        tw[idx + 2 * step],
        tw[idx + 3 * step],
    ];
    C4 {
        r: f32x4::simd_from(s, [t[0].r, t[1].r, t[2].r, t[3].r]),
        i: f32x4::simd_from(s, [t[0].i, t[1].i, t[2].i, t[3].i]),
    }
}

/// `C_MUL`.
#[inline(always)]
fn c_mul<S: Simd>(a: C4<S>, b: C4<S>) -> C4<S> {
    C4 {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

/// `C_ADD`.
#[inline(always)]
fn c_add<S: Simd>(a: C4<S>, b: C4<S>) -> C4<S> {
    C4 {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

/// `C_SUB`.
#[inline(always)]
fn c_sub<S: Simd>(a: C4<S>, b: C4<S>) -> C4<S> {
    C4 {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

/// `kf_bfly2` (`m == 4`), one butterfly per iteration with the four positions `f..f+4` in the
/// lanes. The per-position twiddle expressions of the scalar code are evaluated for all lanes
/// and the right one is selected per lane:
/// position 0: `t = b`; 1: `t = ((b.r+b.i)*tw, (b.i-b.r)*tw)`; 2: `t = (b.i, -b.r)`;
/// 3: `t = ((b.i-b.r)*tw, -((b.i+b.r)*tw))`.
#[inline(always)]
fn bfly2_m4<S: Simd>(s: S, buf: &mut [f32], n: usize) {
    let tw = f32x4::splat(s, 0.7071067812f32);
    let lane = |l: usize| mask32x4::simd_from(s, core::array::from_fn(|k| -i32::from(k == l)));
    let (m0, m1, m2) = (lane(0), lane(1), lane(2));
    for i in 0..n {
        let f = 8 * i;
        let a = ld(s, buf, f);
        let b = ld(s, buf, f + 4);
        // b.r+b.i == b.i+b.r (IEEE addition is commutative).
        let sum = (b.r + b.i) * tw;
        let diff = (b.i - b.r) * tw;
        let t = C4 {
            r: m0.select(b.r, m1.select(sum, m2.select(b.i, diff))),
            i: m0.select(b.i, m1.select(diff, m2.select(-b.r, -sum))),
        };
        st(s, buf, f + 4, c_sub(a, t));
        st(s, buf, f, c_add(a, t));
    }
}

/// `kf_bfly4` with `m == 1` (all twiddles are 1), four butterflies per iteration (lane `k` is
/// butterfly `i + k`). `n` must be a multiple of 4.
#[inline(always)]
fn bfly4_m1<S: Simd>(s: S, buf: &mut [f32], n: usize) {
    // Each butterfly is 4 complex values = 8 floats; one chunk holds 2 butterflies.
    let chunks = buf[..8 * n].as_chunks_mut::<16>().0;
    for pair in chunks.as_chunks_mut::<2>().0 {
        // [f0r f2r g0r g2r], [f0i f2i g0i g2i], [f1r f3r g1r g3r], [f1i f3i g1i g3i]
        let [a0, a1, a2, a3] = s.load_four_interleaved_f32x4(&pair[0]);
        let [b0, b1, b2, b3] = s.load_four_interleaved_f32x4(&pair[1]);
        let (f0r, f2r) = s.deinterleave_f32x4(a0, b0);
        let (f0i, f2i) = s.deinterleave_f32x4(a1, b1);
        let (f1r, f3r) = s.deinterleave_f32x4(a2, b2);
        let (f1i, f3i) = s.deinterleave_f32x4(a3, b3);

        let s0r = f0r - f2r;
        let s0i = f0i - f2i;
        let f0r = f0r + f2r;
        let f0i = f0i + f2i;
        let s1r = f1r + f3r;
        let s1i = f1i + f3i;
        let o2r = f0r - s1r;
        let o2i = f0i - s1i;
        let o0r = f0r + s1r;
        let o0i = f0i + s1i;
        let s1r = f1r - f3r;
        let s1i = f1i - f3i;
        let o1r = s0r + s1i;
        let o1i = s0i - s1r;
        let o3r = s0r - s1i;
        let o3i = s0i + s1r;

        let (a0, b0) = s.interleave_f32x4(o0r, o2r);
        let (a1, b1) = s.interleave_f32x4(o0i, o2i);
        let (a2, b2) = s.interleave_f32x4(o1r, o3r);
        let (a3, b3) = s.interleave_f32x4(o1i, o3i);
        s.store_four_interleaved_f32x4([a0, a1, a2, a3], &mut pair[0]);
        s.store_four_interleaved_f32x4([b0, b1, b2, b3], &mut pair[1]);
    }
}

/// `kf_bfly4` with `m > 1` (`m` a multiple of 4), lanes over four consecutive `f`.
#[inline(always)]
fn bfly4<S: Simd>(
    s: S,
    buf: &mut [f32],
    fstride: usize,
    tw: &[KissTwiddleCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let (m2, m3) = (2 * m, 3 * m);
    for i in 0..n {
        let fbeg = i * mm;
        for u in (0..m).step_by(4) {
            let f = fbeg + u;
            let s0 = c_mul(ld(s, buf, f + m), ld_tw(s, tw, u * fstride, fstride));
            let s1 = c_mul(
                ld(s, buf, f + m2),
                ld_tw(s, tw, 2 * u * fstride, 2 * fstride),
            );
            let s2 = c_mul(
                ld(s, buf, f + m3),
                ld_tw(s, tw, 3 * u * fstride, 3 * fstride),
            );

            let f0 = ld(s, buf, f);
            let s5 = c_sub(f0, s1);
            let f0 = c_add(f0, s1);
            let s3 = c_add(s0, s2);
            let s4 = c_sub(s0, s2);
            st(s, buf, f + m2, c_sub(f0, s3));
            st(s, buf, f, c_add(f0, s3));
            st(
                s,
                buf,
                f + m,
                C4 {
                    r: s5.r + s4.i,
                    i: s5.i - s4.r,
                },
            );
            st(
                s,
                buf,
                f + m3,
                C4 {
                    r: s5.r - s4.i,
                    i: s5.i + s4.r,
                },
            );
        }
    }
}

/// `kf_bfly3` (`m` a multiple of 4), lanes over four consecutive `f`.
#[inline(always)]
fn bfly3<S: Simd>(
    s: S,
    buf: &mut [f32],
    fstride: usize,
    tw: &[KissTwiddleCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let epi3_i = f32x4::splat(s, tw[fstride * m].i);
    let half = f32x4::splat(s, 0.5f32);
    for i in 0..n {
        let fbeg = i * mm;
        for u in (0..m).step_by(4) {
            let f = fbeg + u;
            let s1 = c_mul(ld(s, buf, f + m), ld_tw(s, tw, u * fstride, fstride));
            let s2 = c_mul(
                ld(s, buf, f + m2),
                ld_tw(s, tw, 2 * u * fstride, 2 * fstride),
            );
            let s3 = c_add(s1, s2);
            let s0 = c_sub(s1, s2);

            let f0 = ld(s, buf, f);
            let fm = C4 {
                r: f0.r - s3.r * half,
                i: f0.i - s3.i * half,
            };
            let s0 = C4 {
                r: s0.r * epi3_i,
                i: s0.i * epi3_i,
            };
            st(s, buf, f, c_add(f0, s3));
            st(
                s,
                buf,
                f + m2,
                C4 {
                    r: fm.r + s0.i,
                    i: fm.i - s0.r,
                },
            );
            st(
                s,
                buf,
                f + m,
                C4 {
                    r: fm.r - s0.i,
                    i: fm.i + s0.r,
                },
            );
        }
    }
}

/// `kf_bfly5` (`m` a multiple of 4), lanes over four consecutive `u`.
#[inline(always)]
fn bfly5<S: Simd>(
    s: S,
    buf: &mut [f32],
    fstride: usize,
    tw: &[KissTwiddleCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let (ya, yb) = (tw[fstride * m], tw[fstride * 2 * m]);
    let (yar, yai) = (f32x4::splat(s, ya.r), f32x4::splat(s, ya.i));
    let (ybr, ybi) = (f32x4::splat(s, yb.r), f32x4::splat(s, yb.i));
    for i in 0..n {
        let f0i = i * mm;
        let (f1i, f2i, f3i, f4i) = (f0i + m, f0i + 2 * m, f0i + 3 * m, f0i + 4 * m);
        for u in (0..m).step_by(4) {
            let s0 = ld(s, buf, f0i + u);
            let s1 = c_mul(ld(s, buf, f1i + u), ld_tw(s, tw, u * fstride, fstride));
            let s2 = c_mul(
                ld(s, buf, f2i + u),
                ld_tw(s, tw, 2 * u * fstride, 2 * fstride),
            );
            let s3 = c_mul(
                ld(s, buf, f3i + u),
                ld_tw(s, tw, 3 * u * fstride, 3 * fstride),
            );
            let s4 = c_mul(
                ld(s, buf, f4i + u),
                ld_tw(s, tw, 4 * u * fstride, 4 * fstride),
            );

            let s7 = c_add(s1, s4);
            let s10 = c_sub(s1, s4);
            let s8 = c_add(s2, s3);
            let s9 = c_sub(s2, s3);

            st(
                s,
                buf,
                f0i + u,
                C4 {
                    r: s0.r + (s7.r + s8.r),
                    i: s0.i + (s7.i + s8.i),
                },
            );
            let s5 = C4 {
                r: s0.r + (s7.r * yar + s8.r * ybr),
                i: s0.i + (s7.i * yar + s8.i * ybr),
            };
            let s6 = C4 {
                r: s10.i * yai + s9.i * ybi,
                i: -(s10.r * yai + s9.r * ybi),
            };
            st(s, buf, f1i + u, c_sub(s5, s6));
            st(s, buf, f4i + u, c_add(s5, s6));

            let s11 = C4 {
                r: s0.r + (s7.r * ybr + s8.r * yar),
                i: s0.i + (s7.i * ybr + s8.i * yar),
            };
            let s12 = C4 {
                r: s9.i * yai - s10.i * ybi,
                i: s10.r * ybi - s9.r * yai,
            };
            st(s, buf, f2i + u, c_add(s11, s12));
            st(s, buf, f3i + u, c_sub(s11, s12));
        }
    }
}

/// The stage loop of `opus_fft_impl` (float build) on an interleaved buffer, with the SIMD
/// butterflies where the stage shape allows them and the scalar ones otherwise.
#[inline(always)]
fn fft_stages<S: Simd>(s: S, st: &KissFftState, buf: &mut [f32]) {
    let mut fstride = [0usize; super::static_modes::MAXFACTORS + 1];
    // st->shift can be -1
    let shift = if st.shift > 0 { st.shift } else { 0 };
    let tw = &st.twiddles[..];

    fstride[0] = 1;
    let mut l: usize = 0;
    loop {
        let p = st.factors[2 * l] as usize;
        let m = st.factors[2 * l + 1] as usize;
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }
    let mut m = st.factors[2 * l - 1] as usize;
    for i in (0..l).rev() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as usize
        } else {
            1
        };
        let n = fstride[i];
        let fs = fstride[i] << shift;
        match st.factors[2 * i] {
            2 if m == 4 => bfly2_m4(s, buf, n),
            4 if m == 1 => {
                let nv = n & !3;
                bfly4_m1(s, buf, nv);
                if nv < n {
                    // The degenerate butterfly only touches its own 4 values.
                    kf_bfly4(
                        &mut Interleaved(&mut buf[8 * nv..]),
                        fs,
                        st,
                        1,
                        (n - nv) as i32,
                        m2 as i32,
                    );
                }
            }
            4 if m.is_multiple_of(4) => bfly4(s, buf, fs, tw, m, n, m2),
            3 if m.is_multiple_of(4) => bfly3(s, buf, fs, tw, m, n, m2),
            5 if m.is_multiple_of(4) => bfly5(s, buf, fs, tw, m, n, m2),
            // Shapes only custom modes produce: scalar butterflies.
            2 => kf_bfly2(&mut Interleaved(buf), m as i32, n as i32),
            4 => kf_bfly4(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            3 => kf_bfly3(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            5 => kf_bfly5(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            _ => {}
        }
        m = m2;
    }
}

/// `opus_fft_impl` (float build) on the interleaved buffer `buf` (`2*nfft` values) with the SIMD
/// butterflies. Returns `None` without touching `buf` if the target has no supported SIMD
/// baseline.
#[inline]
pub(crate) fn fft_impl(st: &KissFftState, buf: &mut [f32]) -> Option<()> {
    let buf = &mut buf[..2 * st.nfft as usize];
    with_simd!(s => fft_stages(s, st, buf))
}

// ---- celt/mdct.c rotations (float build) ----

/// Loads `x[k]` for four consecutive `k` into the lanes.
#[inline(always)]
fn ld4<S: Simd>(s: S, x: &[f32], k: usize) -> f32x4<S> {
    f32x4::from_slice(s, &x[k..k + 4])
}

/// Loads `x[k], x[k-1], x[k-2], x[k-3]` (descending) into the lanes.
#[inline(always)]
fn ld4_rev<S: Simd>(s: S, x: &[f32], k: usize) -> f32x4<S> {
    f32x4::from_slice(s, &x[k - 3..=k]).reverse()
}

/// Pre-rotation of `clt_mdct_forward` for the middle block of the fold (`k0 <= i < n4 - k0`,
/// where the folded pair is `re = in[N2-1+ov2-2i]`, `im = in[ov2+2i]`), four pairs per step,
/// stored to `f2` (interleaved) in bit-reversed order. Handles `i` in `[k0, k0 + 4*j)` for the
/// largest such `j` and returns that end (`k0` without SIMD); the caller does the rest.
#[expect(clippy::too_many_arguments, reason = "the state of the C loop")]
#[inline]
pub(crate) fn mdct_forward_pre_mid(
    input: &[f32],
    trig0: &[f32],
    trig1: &[f32],
    bitrev: &[i16],
    scale: Option<f32>,
    f2: &mut [f32],
    n2: usize,
    ov2: usize,
    (k0, k1): (usize, usize),
) -> usize {
    simd_done!(k0; s => {
        let pairs = f2.as_chunks_mut::<2>().0;
        let mut i = k0;
        while i + 4 <= k1 {
            let xp1 = ov2 + 2 * i;
            let xp2 = n2 - 1 + ov2 - 2 * i;
            // Evens of in[xp1..xp1+8] and in[xp2], in[xp2-2], in[xp2-4], in[xp2-6].
            let (im, _) = s.deinterleave_f32x4(ld4(s, input, xp1), ld4(s, input, xp1 + 4));
            let (_, re) = s.deinterleave_f32x4(ld4(s, input, xp2 - 7), ld4(s, input, xp2 - 3));
            let re = re.reverse();
            let t0 = ld4(s, trig0, i);
            let t1 = ld4(s, trig1, i);
            let mut yr = re * t0 - im * t1;
            let mut yi = im * t0 + re * t1;
            if let Some(scale) = scale {
                yr *= scale;
                yi *= scale;
            }
            let (yr, yi) = (yr.as_slice(), yi.as_slice());
            for k in 0..4 {
                pairs[bitrev[i + k] as usize] = [yr[k], yi[k]];
            }
            i += 4;
        }
        i
    })
}

/// Post-rotation of `clt_mdct_forward` (float build: `headroom == 0`) for the pairs
/// `0..4*(N4/4)`: `out[2*stride*i] = yr`, `out[stride*(N2-1-2i)] = yi`. `trig0`/`trig1` are
/// already scaled with QEXT. Returns the number of pairs done (0 without SIMD).
#[inline]
pub(crate) fn mdct_forward_post(
    f2: &[f32],
    trig0: &[f32],
    trig1: &[f32],
    scale: Option<f32>,
    out: &mut [f32],
    stride: usize,
) -> usize {
    let n4 = f2.len() / 2;
    let n2 = 2 * n4;
    simd_done!(0; s => {
        let nv = n4 & !3;
        for i in (0..nv).step_by(4) {
            let fp = ld(s, f2, i);
            let mut t0 = ld4(s, trig0, i);
            let mut t1 = ld4(s, trig1, i);
            if let Some(scale) = scale {
                t0 *= scale;
                t1 *= scale;
            }
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            let (yr, yi) = (yr.as_slice(), yi.as_slice());
            for k in 0..4 {
                out[2 * stride * (i + k)] = yr[k];
                out[stride * (n2 - 1 - 2 * (i + k))] = yi[k];
            }
        }
        nv
    })
}

/// Pre-rotation of `clt_mdct_backward` (float build: no shifts) for `i` in `0..4*(N4/4)`,
/// with `x1 = in[2*stride*i]`, `x2 = in[stride*(N2-1-2i)]`, stored as `[yi, yr]` to pair
/// `bitrev[i]` of `y`. Returns the number of pairs done (0 without SIMD).
#[inline]
pub(crate) fn mdct_backward_pre(
    input: &[f32],
    stride: usize,
    t0s: &[f32],
    t1s: &[f32],
    bitrev: &[i16],
    y: &mut [[f32; 2]],
) -> usize {
    let n4 = bitrev.len();
    let n2 = 2 * n4;
    simd_done!(0; s => {
        let nv = n4 & !3;
        for i in (0..nv).step_by(4) {
            let (x1, x2) = if stride == 1 {
                // in[2i..2i+8] evens; in[N2-1-2i], in[N2-3-2i], ... (odd positions, descending).
                let (x1, _) = s.deinterleave_f32x4(ld4(s, input, 2 * i), ld4(s, input, 2 * i + 4));
                let top = n2 - 1 - 2 * i;
                let (_, x2) = s.deinterleave_f32x4(ld4(s, input, top - 7), ld4(s, input, top - 3));
                (x1, x2.reverse())
            } else {
                let x1: [f32; 4] = core::array::from_fn(|k| input[2 * stride * (i + k)]);
                let x2: [f32; 4] = core::array::from_fn(|k| input[stride * (n2 - 1 - 2 * (i + k))]);
                (f32x4::simd_from(s, x1), f32x4::simd_from(s, x2))
            };
            let t0 = ld4(s, t0s, i);
            let t1 = ld4(s, t1s, i);
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            let (yr, yi) = (yr.as_slice(), yi.as_slice());
            for k in 0..4 {
                y[bitrev[i + k] as usize] = [yi[k], yr[k]];
            }
        }
        nv
    })
}

/// Post-rotation of `clt_mdct_backward` (float build) for the pair steps `0..4*(half/4)`
/// (`half = N4/2`): step `i` works on pairs `i` and `N4-1-i` of `y` with the twiddles
/// `t[i], t[N4+i]` and `t[N4-1-i], t[N2-1-i]`, both pairs read before either is written.
/// Returns the number of steps done (0 without SIMD).
#[inline]
pub(crate) fn mdct_backward_post(y: &mut [f32], t0s: &[f32], t1s: &[f32]) -> usize {
    let n4 = t0s.len();
    let half = n4 >> 1;
    simd_done!(0; s => {
        let nv = half & !3;
        for i in (0..nv).step_by(4) {
            // Pairs i..i+4 (front) and N4-4-i..N4-i (back, lanes reversed so lane k is pair
            // N4-1-i-k).
            let j = n4 - 4 - i;
            let p0 = ld(s, y, i);
            let p1 = ld(s, y, j);
            let (p1r, p1i) = (p1.r.reverse(), p1.i.reverse());
            // We swap real and imag because we're using an FFT instead of an IFFT.
            let (re, im) = (p0.i, p0.r);
            let (t0, t1) = (ld4(s, t0s, i), ld4(s, t1s, i));
            let yr0 = re * t0 + im * t1;
            let yi0 = re * t1 - im * t0;
            let (re, im) = (p1i, p1r);
            let (t0, t1) = (ld4_rev(s, t0s, n4 - 1 - i), ld4_rev(s, t1s, n4 - 1 - i));
            let yr1 = re * t0 + im * t1;
            let yi1 = re * t1 - im * t0;
            // y[yp0] = yr0, y[yp0+1] = yi1, y[yp1] = yr1, y[yp1+1] = yi0
            st(s, y, i, C4 { r: yr0, i: yi1 });
            st(s, y, j, C4 { r: yr1.reverse(), i: yi0.reverse() });
        }
        nv
    })
}

/// TDAC mirror of `clt_mdct_backward` (float build) for `i` in `0..4*(h/4)` (`h = overlap/2`):
/// `x1 = out[overlap-1-i]`, `x2 = out[i]` with `w1 = window[i]`, `w2 = window[overlap-1-i]`.
/// Returns the number of `i` done (0 without SIMD).
#[inline]
pub(crate) fn mdct_backward_mirror(out: &mut [f32], window: &[f32]) -> usize {
    let overlap = window.len();
    let h = overlap / 2;
    simd_done!(0; s => {
        let nv = h & !3;
        for i in (0..nv).step_by(4) {
            let hi = overlap - 1 - i;
            let x1 = ld4_rev(s, out, hi);
            let x2 = ld4(s, out, i);
            let w1 = ld4(s, window, i);
            let w2 = ld4_rev(s, window, hi);
            (x2 * w2 - x1 * w1).store_slice(&mut out[i..i + 4]);
            (x2 * w1 + x1 * w2).reverse().store_slice(&mut out[hi - 3..=hi]);
        }
        nv
    })
}

#[cfg(test)]
mod tests {
    //! SIMD == scalar on every target (the oracle tests only run on the host): the FFT and both
    //! MDCTs of the static modes, all shifts and strides, bit for bit.

    use alloc::{format, vec, vec::Vec};
    use core::sync::atomic::{AtomicBool, Ordering};

    use crate::celt::kiss_fft::{opus_fft, opus_fft_impl};
    use crate::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
    use crate::celt::static_modes::{CeltMode, KissFftCpx, STATIC_MODE_LIST};

    /// Makes the SIMD entry points report "no SIMD" (tests only). Other tests running meanwhile
    /// are unaffected: both paths give the same results.
    pub(super) static SCALAR_ONLY: AtomicBool = AtomicBool::new(false);

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 65536.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    /// Runs `f` once with SIMD and once scalar-only; both results must be bit-identical.
    fn both<T: PartialEq + core::fmt::Debug>(what: &str, mut f: impl FnMut() -> T) {
        let simd = f();
        SCALAR_ONLY.store(true, Ordering::Relaxed);
        let scalar = f();
        SCALAR_ONLY.store(false, Ordering::Relaxed);
        assert_eq!(simd, scalar, "{what}");
    }

    fn modes() -> &'static [&'static CeltMode] {
        &STATIC_MODE_LIST
    }

    #[test]
    fn fft_matches_scalar() {
        let mut rng = Lcg(1);
        for &mode in modes() {
            for st in &mode.mdct.kfft {
                let n = st.nfft as usize;
                for _ in 0..20 {
                    let v = rng.vec(2 * n);
                    let fin: Vec<KissFftCpx> = v
                        .chunks(2)
                        .map(|c| KissFftCpx { r: c[0], i: c[1] })
                        .collect();
                    both(&format!("opus_fft {n}"), || {
                        let mut out = vec![KissFftCpx::default(); n];
                        opus_fft(st, &fin, &mut out);
                        out.iter()
                            .flat_map(|c| [c.r.to_bits(), c.i.to_bits()])
                            .collect::<Vec<_>>()
                    });
                    // opus_fft_impl on `[KissFftCpx]` is always scalar: compare it with the
                    // SIMD kernel directly.
                    let mut scalar = fin.clone();
                    opus_fft_impl(st, &mut scalar);
                    let mut flat = v.clone();
                    if super::fft_impl(st, &mut flat).is_some() {
                        let s: Vec<f32> = scalar.iter().flat_map(|c| [c.r, c.i]).collect();
                        assert_eq!(bits(&flat), bits(&s), "opus_fft_impl {n}");
                    }
                }
            }
        }
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
                        let window = if ov == overlap {
                            mode.window.to_vec()
                        } else {
                            rng.vec(ov)
                        };
                        let input = rng.vec(n2 + ov);
                        let out0 = rng.vec(stride * n2);
                        both(&format!("forward n={n} stride={stride} ov={ov}"), || {
                            let mut out = out0.clone();
                            clt_mdct_forward(
                                &mode.mdct, &input, &mut out, &window, ov, shift, stride,
                            );
                            bits(&out)
                        });
                        let input = rng.vec(stride * n2);
                        let out0 = rng.vec((n2 + ov / 2).max(ov) + 3);
                        both(&format!("backward n={n} stride={stride} ov={ov}"), || {
                            let mut out = out0.clone();
                            clt_mdct_backward(
                                &mode.mdct, &input, &mut out, &window, ov, shift, stride,
                            );
                            bits(&out)
                        });
                    }
                }
            }
        }
    }
}

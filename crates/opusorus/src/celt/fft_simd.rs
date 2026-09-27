//! Vertical SIMD (`fearless_simd`, see [`crate::simd`]) FFT butterflies (celt/kiss_fft.c) for
//! the float and the fixed-point builds.
//!
//! The butterflies are written once, over [`Arith`]: the lane arithmetic of the build's
//! `_kiss_fft_guts.h` macros. Each lane runs the scalar operation sequence on an independent
//! butterfly, so results are bit-identical to the scalar code:
//! * float: IEEE `+`, `-`, `*` (no FMA);
//! * fixed point: wrapping `i32` adds (the `*_ovflw` macros) and an exact `S_MUL`:
//!   `MULT16_32_Q15` (NEON `sqdmulh` with the Q15 twiddle in the high half, which cannot
//!   saturate because the twiddles are above `-32768`; elsewhere an exact split into 16-bit
//!   halves), or with QEXT `MULT32_32_P31` (NEON `sqrdmulh`, with the one saturating case
//!   `-2^31 * -2^31` patched to the wrapped C result; aarch64 only).

#[cfg(feature = "fixed-point")]
use super::kiss_fft::coef_const;
use super::kiss_fft::{Interleaved, kf_bfly2, kf_bfly3, kf_bfly4, kf_bfly5};
use super::static_modes::{KissFftScalar, KissFftState, KissTwiddleCpx, KissTwiddleScalar};

/// Lane arithmetic of the FFT for one build and SIMD level (a value carries the SIMD token).
pub(crate) trait Arith: Copy {
    /// Four `kiss_fft_scalar` lanes.
    type V: Copy;
    /// Four twiddle lanes, in the form [`Arith::mul`] takes them.
    type T: Copy;
    /// Loads 4 scalars.
    fn ld(self, x: &[KissFftScalar]) -> Self::V;
    /// Stores 4 scalars.
    fn st(self, v: Self::V, x: &mut [KissFftScalar]);
    /// `([a0, a2, b0, b2], [a1, a3, b1, b3])`.
    fn deint(self, a: Self::V, b: Self::V) -> (Self::V, Self::V);
    /// `([a0, b0, a1, b1], [a2, b2, a3, b3])`.
    fn inter(self, a: Self::V, b: Self::V) -> (Self::V, Self::V);
    /// 4-way deinterleaving load of 16 scalars.
    fn ld4x4(self, x: &[KissFftScalar; 16]) -> [Self::V; 4];
    /// 4-way interleaving store of 16 scalars.
    fn st4x4(self, v: [Self::V; 4], x: &mut [KissFftScalar; 16]);
    /// `ADD32_ovflw` / float `+`.
    fn add(self, a: Self::V, b: Self::V) -> Self::V;
    /// `SUB32_ovflw` / float `-`.
    fn sub(self, a: Self::V, b: Self::V) -> Self::V;
    /// `NEG32_ovflw` / float unary `-`.
    fn neg(self, a: Self::V) -> Self::V;
    /// `S_MUL(a, t)`.
    fn mul(self, a: Self::V, t: Self::T) -> Self::V;
    /// `HALF_OF(a)`.
    fn half(self, a: Self::V) -> Self::V;
    /// Twiddle lanes from four twiddle scalars.
    fn tw(self, t: [KissTwiddleScalar; 4]) -> Self::T;
    /// Lane `k` from `a` where `m[k]`, else from `b`.
    fn sel(self, m: [bool; 4], a: Self::V, b: Self::V) -> Self::V;
    /// `fft_downshift` of the whole buffer by `shift` (`> 0`): `SHR32(x, 1)` for a shift of 1,
    /// `PSHR32(x, shift)` otherwise.
    fn downshift(self, x: &mut [KissFftScalar], shift: i32);
    /// Lanes reversed.
    fn rev(self, a: Self::V) -> Self::V;
    /// Lanes from an array.
    fn lanes(self, x: [KissFftScalar; 4]) -> Self::V;
    /// Lanes to an array.
    fn array(self, a: Self::V) -> [KissFftScalar; 4];
    /// `S_MUL2(a, scale)`.
    fn mul2(self, a: Self::V, scale: KissTwiddleScalar) -> Self::V;
    /// `PSHR32(a, shift)` / `PSHR32_ovflw` (identity in the float build).
    fn pshr(self, a: Self::V, shift: i32) -> Self::V;
    /// `SHL32_ovflw(a, shift)` (identity in the float build).
    fn shl(self, a: Self::V, shift: i32) -> Self::V;
    /// `MAX32(acc, MAX32(ABS32(a), ABS32(b)))` per lane (fixed point).
    #[cfg(feature = "fixed-point")]
    fn max_abs(self, acc: Self::V, a: Self::V, b: Self::V) -> Self::V;
    /// Twiddle lanes from scalar lanes (QEXT scales the MDCT twiddles).
    #[cfg(feature = "qext")]
    fn tw_from(self, a: Self::V) -> Self::T;
}

/// Evaluates `$body` with `$a` bound to the best [`Arith`] of the build and target, giving
/// `Some(result)`, or `None` if there is none (scalar code). `$q15_ok` (only evaluated for the
/// NEON Q15 flavour) states that no twiddle/coefficient the body multiplies by is `-32768`, the
/// condition of its exact `sqdmulh` multiply; otherwise the portable Q15 multiply is used. The
/// body is expanded once per flavour.
macro_rules! with_arith {
    ($q15_ok:expr; $a:ident => $body:expr) => {{
        #[cfg(not(feature = "fixed-point"))]
        let r = {
            let _ = || $q15_ok;
            with_simd!(s => {
                let $a = $crate::celt::fft_simd::float::Float(s);
                $body
            })
        };
        #[cfg(all(
            feature = "fixed-point",
            not(feature = "qext"),
            not(feature = "fixed-point-debug")
        ))]
        let r = {
            #[cfg(target_arch = "aarch64")]
            let neon = neon_token!().filter(|_| $q15_ok);
            #[cfg(not(target_arch = "aarch64"))]
            let neon: Option<core::convert::Infallible> = {
                let _ = || $q15_ok;
                None
            };
            match neon {
                #[cfg(target_arch = "aarch64")]
                Some(neon) => Some(fearless_simd::Simd::vectorize(
                    neon,
                    #[inline(always)]
                    || {
                        let $a = $crate::celt::fft_simd::fixed::Q15Neon(neon);
                        $body
                    },
                )),
                #[cfg(not(target_arch = "aarch64"))]
                Some(never) => match never {},
                None => with_simd!(s => {
                    let $a = $crate::celt::fft_simd::fixed::Q15(s);
                    $body
                }),
            }
        };
        #[cfg(all(feature = "fixed-point", feature = "qext", not(feature = "fixed-point-debug")))]
        let r = {
            let _ = || $q15_ok;
            #[cfg(target_arch = "aarch64")]
            let r = neon_token!().map(|neon| {
                fearless_simd::Simd::vectorize(
                    neon,
                    #[inline(always)]
                    || {
                        let $a = $crate::celt::fft_simd::fixed::Q31Neon(neon);
                        $body
                    },
                )
            });
            // No exact portable Q31 multiply: scalar.
            #[cfg(not(target_arch = "aarch64"))]
            let r = None;
            r
        };
        r
    }};
}
pub(crate) use with_arith;

/// Four complex values, one per lane.
#[derive(Clone, Copy)]
struct C4<V> {
    r: V,
    i: V,
}

/// Loads the four consecutive complex values `k..k+4`.
#[inline(always)]
fn ld<A: Arith>(a: A, buf: &[KissFftScalar], k: usize) -> C4<A::V> {
    let x = &buf[2 * k..2 * k + 8];
    let (r, i) = a.deint(a.ld(&x[..4]), a.ld(&x[4..]));
    C4 { r, i }
}

/// Stores four values to the complex positions `k..k+4`.
#[inline(always)]
fn st<A: Arith>(a: A, buf: &mut [KissFftScalar], k: usize, v: C4<A::V>) {
    let x = &mut buf[2 * k..2 * k + 8];
    let (p, q) = a.inter(v.r, v.i);
    a.st(p, &mut x[..4]);
    a.st(q, &mut x[4..]);
}

/// Gathers `tw[idx + k*step]`, `k = 0..4`.
#[inline(always)]
fn ld_tw<A: Arith>(a: A, tw: &[KissTwiddleCpx], idx: usize, step: usize) -> C4<A::T> {
    let t: [KissTwiddleCpx; 4] = core::array::from_fn(|k| tw[idx + k * step]);
    C4 {
        r: a.tw(t.map(|c| c.r)),
        i: a.tw(t.map(|c| c.i)),
    }
}

/// `C_MUL`.
#[inline(always)]
fn c_mul<A: Arith>(a: A, x: C4<A::V>, t: C4<A::T>) -> C4<A::V> {
    C4 {
        r: a.sub(a.mul(x.r, t.r), a.mul(x.i, t.i)),
        i: a.add(a.mul(x.r, t.i), a.mul(x.i, t.r)),
    }
}

/// `C_ADD`.
#[inline(always)]
fn c_add<A: Arith>(a: A, x: C4<A::V>, y: C4<A::V>) -> C4<A::V> {
    C4 {
        r: a.add(x.r, y.r),
        i: a.add(x.i, y.i),
    }
}

/// `C_SUB`.
#[inline(always)]
fn c_sub<A: Arith>(a: A, x: C4<A::V>, y: C4<A::V>) -> C4<A::V> {
    C4 {
        r: a.sub(x.r, y.r),
        i: a.sub(x.i, y.i),
    }
}

/// The `kf_bfly2` twiddle `0.7071067812` of the build.
#[inline(always)]
const fn bfly2_tw() -> KissTwiddleScalar {
    #[cfg(feature = "fixed-point")]
    {
        coef_const(0.7071067812f32)
    }
    #[cfg(not(feature = "fixed-point"))]
    {
        0.7071067812f32
    }
}

/// `kf_bfly2` (`m == 4`), one butterfly per iteration with the four positions `f..f+4` in the
/// lanes. The multiplier inputs of positions 1 and 3 are formed per lane, multiplied together,
/// and the right `t` is selected per lane (the scalar expressions: position 0 `t = b`;
/// 1 `t = (S_MUL(b.r+b.i, tw), S_MUL(b.i-b.r, tw))`; 2 `t = (b.i, -b.r)`;
/// 3 `t = (S_MUL(b.i-b.r, tw), S_MUL(-(b.i+b.r), tw))`).
#[inline(always)]
fn bfly2_m4<A: Arith>(a: A, buf: &mut [KissFftScalar], n: usize) {
    let tw = a.tw([bfly2_tw(); 4]);
    const L1: [bool; 4] = [false, true, false, false];
    const L0: [bool; 4] = [true, false, false, false];
    const L2: [bool; 4] = [false, false, true, false];
    for i in 0..n {
        let f = 8 * i;
        let x = ld(a, buf, f);
        let b = ld(a, buf, f + 4);
        let sum = a.add(b.r, b.i);
        let diff = a.sub(b.i, b.r);
        // Real part inputs: lane 1 `b.r+b.i`, lane 3 `b.i-b.r`; imaginary: lane 1 `b.i-b.r`,
        // lane 3 `-(b.i+b.r)` (`b.r+b.i == b.i+b.r` in both builds).
        let mr = a.mul(a.sel(L1, sum, diff), tw);
        let mi = a.mul(a.sel(L1, diff, a.neg(sum)), tw);
        let t = C4 {
            r: a.sel(L0, b.r, a.sel(L2, b.i, mr)),
            i: a.sel(L0, b.i, a.sel(L2, a.neg(b.r), mi)),
        };
        st(a, buf, f + 4, c_sub(a, x, t));
        st(a, buf, f, c_add(a, x, t));
    }
}

/// `kf_bfly4` with `m == 1` (all twiddles are 1), four butterflies per iteration (lane `k` is
/// butterfly `i + k`). `n` must be a multiple of 4.
#[inline(always)]
fn bfly4_m1<A: Arith>(a: A, buf: &mut [KissFftScalar], n: usize) {
    // Each butterfly is 4 complex values = 8 scalars; one chunk holds 2 butterflies.
    let chunks = buf[..8 * n].as_chunks_mut::<16>().0;
    for pair in chunks.as_chunks_mut::<2>().0 {
        // [f0r f2r g0r g2r], [f0i f2i g0i g2i], [f1r f3r g1r g3r], [f1i f3i g1i g3i]
        let [a0, a1, a2, a3] = a.ld4x4(&pair[0]);
        let [b0, b1, b2, b3] = a.ld4x4(&pair[1]);
        let (f0r, f2r) = a.deint(a0, b0);
        let (f0i, f2i) = a.deint(a1, b1);
        let (f1r, f3r) = a.deint(a2, b2);
        let (f1i, f3i) = a.deint(a3, b3);

        let s0r = a.sub(f0r, f2r);
        let s0i = a.sub(f0i, f2i);
        let f0r = a.add(f0r, f2r);
        let f0i = a.add(f0i, f2i);
        let s1r = a.add(f1r, f3r);
        let s1i = a.add(f1i, f3i);
        let o2r = a.sub(f0r, s1r);
        let o2i = a.sub(f0i, s1i);
        let o0r = a.add(f0r, s1r);
        let o0i = a.add(f0i, s1i);
        let s1r = a.sub(f1r, f3r);
        let s1i = a.sub(f1i, f3i);
        let o1r = a.add(s0r, s1i);
        let o1i = a.sub(s0i, s1r);
        let o3r = a.sub(s0r, s1i);
        let o3i = a.add(s0i, s1r);

        let (a0, b0) = a.inter(o0r, o2r);
        let (a1, b1) = a.inter(o0i, o2i);
        let (a2, b2) = a.inter(o1r, o3r);
        let (a3, b3) = a.inter(o1i, o3i);
        a.st4x4([a0, a1, a2, a3], &mut pair[0]);
        a.st4x4([b0, b1, b2, b3], &mut pair[1]);
    }
}

/// `kf_bfly4` with `m > 1` (`m` a multiple of 4), lanes over four consecutive `f`.
#[inline(always)]
fn bfly4<A: Arith>(
    a: A,
    buf: &mut [KissFftScalar],
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
            let s0 = c_mul(a, ld(a, buf, f + m), ld_tw(a, tw, u * fstride, fstride));
            let s1 = c_mul(
                a,
                ld(a, buf, f + m2),
                ld_tw(a, tw, 2 * u * fstride, 2 * fstride),
            );
            let s2 = c_mul(
                a,
                ld(a, buf, f + m3),
                ld_tw(a, tw, 3 * u * fstride, 3 * fstride),
            );

            let f0 = ld(a, buf, f);
            let s5 = c_sub(a, f0, s1);
            let f0 = c_add(a, f0, s1);
            let s3 = c_add(a, s0, s2);
            let s4 = c_sub(a, s0, s2);
            st(a, buf, f + m2, c_sub(a, f0, s3));
            st(a, buf, f, c_add(a, f0, s3));
            let o1 = C4 {
                r: a.add(s5.r, s4.i),
                i: a.sub(s5.i, s4.r),
            };
            st(a, buf, f + m, o1);
            let o3 = C4 {
                r: a.sub(s5.r, s4.i),
                i: a.add(s5.i, s4.r),
            };
            st(a, buf, f + m3, o3);
        }
    }
}

/// `kf_bfly3` (`m` a multiple of 4), lanes over four consecutive `f`.
#[inline(always)]
fn bfly3<A: Arith>(
    a: A,
    buf: &mut [KissFftScalar],
    fstride: usize,
    tw: &[KissTwiddleCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    // epi3.i
    #[cfg(feature = "fixed-point")]
    let epi3_i: KissTwiddleScalar = -coef_const(0.86602540f32);
    #[cfg(not(feature = "fixed-point"))]
    let epi3_i: KissTwiddleScalar = tw[fstride * m].i;
    let epi3_i = a.tw([epi3_i; 4]);
    for i in 0..n {
        let fbeg = i * mm;
        for u in (0..m).step_by(4) {
            let f = fbeg + u;
            let s1 = c_mul(a, ld(a, buf, f + m), ld_tw(a, tw, u * fstride, fstride));
            let s2 = c_mul(
                a,
                ld(a, buf, f + m2),
                ld_tw(a, tw, 2 * u * fstride, 2 * fstride),
            );
            let s3 = c_add(a, s1, s2);
            let s0 = c_sub(a, s1, s2);

            let f0 = ld(a, buf, f);
            let fm = C4 {
                r: a.sub(f0.r, a.half(s3.r)),
                i: a.sub(f0.i, a.half(s3.i)),
            };
            let s0 = C4 {
                r: a.mul(s0.r, epi3_i),
                i: a.mul(s0.i, epi3_i),
            };
            st(a, buf, f, c_add(a, f0, s3));
            let o2 = C4 {
                r: a.add(fm.r, s0.i),
                i: a.sub(fm.i, s0.r),
            };
            st(a, buf, f + m2, o2);
            let o1 = C4 {
                r: a.sub(fm.r, s0.i),
                i: a.add(fm.i, s0.r),
            };
            st(a, buf, f + m, o1);
        }
    }
}

/// `kf_bfly5` (`m` a multiple of 4), lanes over four consecutive `u`.
#[inline(always)]
fn bfly5<A: Arith>(
    a: A,
    buf: &mut [KissFftScalar],
    fstride: usize,
    tw: &[KissTwiddleCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    #[cfg(feature = "fixed-point")]
    let (ya, yb) = (
        KissTwiddleCpx {
            r: coef_const(0.30901699f32),
            i: -coef_const(0.95105652f32),
        },
        KissTwiddleCpx {
            r: -coef_const(0.80901699f32),
            i: -coef_const(0.58778525f32),
        },
    );
    #[cfg(not(feature = "fixed-point"))]
    let (ya, yb) = (tw[fstride * m], tw[fstride * 2 * m]);
    let (yar, yai) = (a.tw([ya.r; 4]), a.tw([ya.i; 4]));
    let (ybr, ybi) = (a.tw([yb.r; 4]), a.tw([yb.i; 4]));
    for i in 0..n {
        let f0i = i * mm;
        let (f1i, f2i, f3i, f4i) = (f0i + m, f0i + 2 * m, f0i + 3 * m, f0i + 4 * m);
        for u in (0..m).step_by(4) {
            let s0 = ld(a, buf, f0i + u);
            let s1 = c_mul(a, ld(a, buf, f1i + u), ld_tw(a, tw, u * fstride, fstride));
            let s2 = c_mul(
                a,
                ld(a, buf, f2i + u),
                ld_tw(a, tw, 2 * u * fstride, 2 * fstride),
            );
            let s3 = c_mul(
                a,
                ld(a, buf, f3i + u),
                ld_tw(a, tw, 3 * u * fstride, 3 * fstride),
            );
            let s4 = c_mul(
                a,
                ld(a, buf, f4i + u),
                ld_tw(a, tw, 4 * u * fstride, 4 * fstride),
            );

            let s7 = c_add(a, s1, s4);
            let s10 = c_sub(a, s1, s4);
            let s8 = c_add(a, s2, s3);
            let s9 = c_sub(a, s2, s3);

            let o0 = C4 {
                r: a.add(s0.r, a.add(s7.r, s8.r)),
                i: a.add(s0.i, a.add(s7.i, s8.i)),
            };
            st(a, buf, f0i + u, o0);
            let s5 = C4 {
                r: a.add(s0.r, a.add(a.mul(s7.r, yar), a.mul(s8.r, ybr))),
                i: a.add(s0.i, a.add(a.mul(s7.i, yar), a.mul(s8.i, ybr))),
            };
            let s6 = C4 {
                r: a.add(a.mul(s10.i, yai), a.mul(s9.i, ybi)),
                i: a.neg(a.add(a.mul(s10.r, yai), a.mul(s9.r, ybi))),
            };
            st(a, buf, f1i + u, c_sub(a, s5, s6));
            st(a, buf, f4i + u, c_add(a, s5, s6));

            let s11 = C4 {
                r: a.add(s0.r, a.add(a.mul(s7.r, ybr), a.mul(s8.r, yar))),
                i: a.add(s0.i, a.add(a.mul(s7.i, ybr), a.mul(s8.i, yar))),
            };
            let s12 = C4 {
                r: a.sub(a.mul(s9.i, yai), a.mul(s10.i, ybi)),
                i: a.sub(a.mul(s10.r, ybi), a.mul(s9.r, yai)),
            };
            st(a, buf, f2i + u, c_add(a, s11, s12));
            st(a, buf, f3i + u, c_sub(a, s11, s12));
        }
    }
}

/// `fft_downshift` (fixed point; nothing in the float build).
#[inline(always)]
fn fft_downshift<A: Arith>(a: A, buf: &mut [KissFftScalar], total: &mut i32, step: i32) {
    let shift = if step < *total { step } else { *total };
    *total -= shift;
    if shift > 0 {
        a.downshift(buf, shift);
    }
}

/// The stage loop of `opus_fft_impl` on an interleaved buffer, with the SIMD butterflies where
/// the stage shape allows them and the scalar ones otherwise.
#[inline(always)]
fn fft_stages<A: Arith>(a: A, st: &KissFftState, buf: &mut [KissFftScalar], mut downshift: i32) {
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
        let radix = st.factors[2 * i];
        let step = match radix {
            2 => 1,
            4 | 3 => 2,
            5 => 3,
            _ => 0,
        };
        if step > 0 {
            fft_downshift(a, buf, &mut downshift, step);
        }
        match radix {
            2 if m == 4 => bfly2_m4(a, buf, n),
            4 if m == 1 => {
                let nv = n & !3;
                bfly4_m1(a, buf, nv);
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
            4 if m.is_multiple_of(4) => bfly4(a, buf, fs, tw, m, n, m2),
            3 if m.is_multiple_of(4) => bfly3(a, buf, fs, tw, m, n, m2),
            5 if m.is_multiple_of(4) => bfly5(a, buf, fs, tw, m, n, m2),
            // Shapes only custom modes produce: scalar butterflies.
            2 => kf_bfly2(&mut Interleaved(buf), m as i32, n as i32),
            4 => kf_bfly4(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            3 => kf_bfly3(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            5 => kf_bfly5(&mut Interleaved(buf), fs, st, m as i32, n as i32, m2 as i32),
            _ => {}
        }
        m = m2;
    }
    let rest = downshift;
    fft_downshift(a, buf, &mut downshift, rest);
}

/// `opus_fft_impl` on the interleaved buffer `buf` (`2*nfft` scalars) with the SIMD
/// butterflies (`downshift`: the fixed-point `ARG_FIXED` argument, 0 in the float build).
/// Returns `None` without touching `buf` if the build/target has no SIMD FFT.
#[inline]
pub(crate) fn fft_impl(st: &KissFftState, buf: &mut [KissFftScalar], downshift: i32) -> Option<()> {
    let buf = &mut buf[..2 * st.nfft as usize];
    // Static twiddle tables are above -32768 (checked by the tests); runtime ones (custom modes)
    // take the portable multiply.
    let q15_ok = matches!(st.twiddles, alloc::borrow::Cow::Borrowed(_));
    with_arith!(q15_ok; a => fft_stages(a, st, buf, downshift))
}

#[cfg(not(feature = "fixed-point"))]
pub(crate) mod float {
    //! Float lane arithmetic.

    use fearless_simd::{Select, Simd, SimdBase, SimdFrom, f32x4, mask32x4};

    use super::Arith;

    #[derive(Clone, Copy)]
    pub(crate) struct Float<S: Simd>(pub(crate) S);

    impl<S: Simd> Arith for Float<S> {
        type V = f32x4<S>;
        type T = f32x4<S>;
        #[inline(always)]
        fn ld(self, x: &[f32]) -> Self::V {
            f32x4::from_slice(self.0, x)
        }
        #[inline(always)]
        fn st(self, v: Self::V, x: &mut [f32]) {
            v.store_slice(x);
        }
        #[inline(always)]
        fn deint(self, a: Self::V, b: Self::V) -> (Self::V, Self::V) {
            self.0.deinterleave_f32x4(a, b)
        }
        #[inline(always)]
        fn inter(self, a: Self::V, b: Self::V) -> (Self::V, Self::V) {
            self.0.interleave_f32x4(a, b)
        }
        #[inline(always)]
        fn ld4x4(self, x: &[f32; 16]) -> [Self::V; 4] {
            self.0.load_four_interleaved_f32x4(x)
        }
        #[inline(always)]
        fn st4x4(self, v: [Self::V; 4], x: &mut [f32; 16]) {
            self.0.store_four_interleaved_f32x4(v, x);
        }
        #[inline(always)]
        fn add(self, a: Self::V, b: Self::V) -> Self::V {
            a + b
        }
        #[inline(always)]
        fn sub(self, a: Self::V, b: Self::V) -> Self::V {
            a - b
        }
        #[inline(always)]
        fn neg(self, a: Self::V) -> Self::V {
            -a
        }
        #[inline(always)]
        fn mul(self, a: Self::V, t: Self::T) -> Self::V {
            a * t
        }
        #[inline(always)]
        fn half(self, a: Self::V) -> Self::V {
            a * 0.5f32
        }
        #[inline(always)]
        fn tw(self, t: [f32; 4]) -> Self::T {
            f32x4::simd_from(self.0, t)
        }
        #[inline(always)]
        fn sel(self, m: [bool; 4], a: Self::V, b: Self::V) -> Self::V {
            mask32x4::simd_from(self.0, m.map(|v| -i32::from(v))).select(a, b)
        }
        #[inline(always)]
        fn downshift(self, _x: &mut [f32], _shift: i32) {}
        #[inline(always)]
        fn rev(self, a: Self::V) -> Self::V {
            a.reverse()
        }
        #[inline(always)]
        fn lanes(self, x: [f32; 4]) -> Self::V {
            f32x4::simd_from(self.0, x)
        }
        #[inline(always)]
        fn array(self, a: Self::V) -> [f32; 4] {
            let mut o = [0f32; 4];
            a.store_slice(&mut o);
            o
        }
        #[inline(always)]
        fn mul2(self, a: Self::V, scale: f32) -> Self::V {
            a * scale
        }
        #[inline(always)]
        fn pshr(self, a: Self::V, _shift: i32) -> Self::V {
            a
        }
        #[inline(always)]
        fn shl(self, a: Self::V, _shift: i32) -> Self::V {
            a
        }
        #[cfg(feature = "qext")]
        #[inline(always)]
        fn tw_from(self, a: Self::V) -> Self::T {
            a
        }
    }
}

#[cfg(all(feature = "fixed-point", not(feature = "fixed-point-debug")))]
pub(crate) mod fixed {
    //! Fixed-point lane arithmetic (`i32` lanes, wrapping adds, exact `S_MUL`).

    use fearless_simd::{Select, Simd, SimdBase, SimdFrom, i32x4, mask32x4};

    #[cfg(target_arch = "aarch64")]
    use fearless_simd::Neon;

    use super::Arith;
    use crate::celt::static_modes::KissTwiddleScalar;

    /// The integer lane operations shared by every fixed-point flavour.
    macro_rules! int_lanes {
        () => {
            type V = i32x4<<Self as Token>::S>;
            #[inline(always)]
            fn ld(self, x: &[i32]) -> Self::V {
                i32x4::from_slice(self.token(), x)
            }
            #[inline(always)]
            fn st(self, v: Self::V, x: &mut [i32]) {
                v.store_slice(x);
            }
            #[inline(always)]
            fn deint(self, a: Self::V, b: Self::V) -> (Self::V, Self::V) {
                self.token().deinterleave_i32x4(a, b)
            }
            #[inline(always)]
            fn inter(self, a: Self::V, b: Self::V) -> (Self::V, Self::V) {
                self.token().interleave_i32x4(a, b)
            }
            #[inline(always)]
            fn ld4x4(self, x: &[i32; 16]) -> [Self::V; 4] {
                self.token().load_four_interleaved_i32x4(x)
            }
            #[inline(always)]
            fn st4x4(self, v: [Self::V; 4], x: &mut [i32; 16]) {
                self.token().store_four_interleaved_i32x4(v, x);
            }
            #[inline(always)]
            fn add(self, a: Self::V, b: Self::V) -> Self::V {
                a + b
            }
            #[inline(always)]
            fn sub(self, a: Self::V, b: Self::V) -> Self::V {
                a - b
            }
            #[inline(always)]
            fn neg(self, a: Self::V) -> Self::V {
                -a
            }
            #[inline(always)]
            fn half(self, a: Self::V) -> Self::V {
                a >> 1
            }
            #[inline(always)]
            fn sel(self, m: [bool; 4], a: Self::V, b: Self::V) -> Self::V {
                mask32x4::simd_from(self.token(), m.map(|v| -i32::from(v))).select(a, b)
            }
            #[inline(always)]
            fn downshift(self, x: &mut [i32], shift: i32) {
                let s = self.token();
                let (chunks, rest) = x.as_chunks_mut::<4>();
                if shift == 1 {
                    // SHR32(x, 1)
                    for c in chunks {
                        (i32x4::from_slice(s, c) >> 1).store_slice(c);
                    }
                    for v in rest {
                        *v >>= 1;
                    }
                } else {
                    // PSHR32(x, shift): (x + ((1<<shift)>>1)) >> shift (the add cannot
                    // overflow for FFT data, which has headroom; it wraps like C would).
                    let r = (1i32 << shift) >> 1;
                    let sh = shift as u32;
                    for c in chunks {
                        ((i32x4::from_slice(s, c) + r) >> sh).store_slice(c);
                    }
                    for v in rest {
                        *v = v.wrapping_add(r) >> sh;
                    }
                }
            }
            #[inline(always)]
            fn rev(self, a: Self::V) -> Self::V {
                a.reverse()
            }
            #[inline(always)]
            fn lanes(self, x: [i32; 4]) -> Self::V {
                i32x4::simd_from(self.token(), x)
            }
            #[inline(always)]
            fn array(self, a: Self::V) -> [i32; 4] {
                let mut o = [0i32; 4];
                a.store_slice(&mut o);
                o
            }
            #[inline(always)]
            fn pshr(self, a: Self::V, shift: i32) -> Self::V {
                // PSHR32 / PSHR32_ovflw: (a + ((1<<shift)>>1)) >> shift (wrapping add).
                (a + ((1i32 << shift) >> 1)) >> shift as u32
            }
            #[inline(always)]
            fn shl(self, a: Self::V, shift: i32) -> Self::V {
                // SHL32_ovflw: (opus_int32)((opus_uint32)a << shift).
                a << shift as u32
            }
            #[inline(always)]
            fn max_abs(self, acc: Self::V, a: Self::V, b: Self::V) -> Self::V {
                // ABS32 wraps at -2^31 like the scalar code in release builds.
                let s = self.token();
                s.max_i32x4(acc, s.max_i32x4(s.abs_i32x4(a), s.abs_i32x4(b)))
            }
        };
    }

    /// Access to the SIMD token of a flavour.
    pub(crate) trait Token: Copy {
        type S: Simd;
        fn token(self) -> Self::S;
    }

    /// Q15 twiddles (`MULT16_32_Q15`), portable: `S_MUL(x, t) = (t*x) >> 15` computed exactly
    /// from the halves `x = xh*2^16 + xl`: `((t*xh) << 1) + ((t*xl) >> 15)` (both products fit
    /// in `i32`; the sum wraps as the C `(opus_val32)` conversion does).
    #[cfg(not(feature = "qext"))]
    #[derive(Clone, Copy)]
    pub(crate) struct Q15<S: Simd>(pub(crate) S);

    #[cfg(not(feature = "qext"))]
    impl<S: Simd> Token for Q15<S> {
        type S = S;
        #[inline(always)]
        fn token(self) -> S {
            self.0
        }
    }

    #[cfg(not(feature = "qext"))]
    impl<S: Simd> Arith for Q15<S> {
        int_lanes!();
        type T = i32x4<S>;
        #[inline(always)]
        fn mul(self, a: Self::V, t: Self::T) -> Self::V {
            let xh = a >> 16;
            let xl = a & 0xffff;
            ((t * xh) << 1) + ((t * xl) >> 15)
        }
        #[inline(always)]
        fn tw(self, t: [KissTwiddleScalar; 4]) -> Self::T {
            i32x4::simd_from(self.0, t.map(i32::from))
        }
        #[inline(always)]
        fn mul2(self, a: Self::V, scale: KissTwiddleScalar) -> Self::V {
            // MULT16_32_Q16(scale, a) = (scale*a) >> 16 = scale*ah + ((scale*al) >> 16).
            let t = i32x4::splat(self.0, i32::from(scale));
            let xh = a >> 16;
            let xl = a & 0xffff;
            (t * xh) + ((t * xl) >> 16)
        }
    }

    /// Q15 twiddles on NEON: `sqdmulh(x, t << 16) = (2*x*t*2^16) >> 32 = (t*x) >> 15`, exact
    /// because `t << 16 > -2^31` (the twiddles are at least `-32767`, checked when the SIMD FFT
    /// is chosen) so the doubling cannot saturate.
    #[cfg(all(target_arch = "aarch64", not(feature = "qext")))]
    #[derive(Clone, Copy)]
    pub(crate) struct Q15Neon(pub(crate) Neon);

    #[cfg(all(target_arch = "aarch64", not(feature = "qext")))]
    impl Token for Q15Neon {
        type S = Neon;
        #[inline(always)]
        fn token(self) -> Self::S {
            self.0
        }
    }

    #[cfg(target_arch = "aarch64")]
    fearless_simd::kernel!(
        /// `sqdmulh`.
        #[inline(always)]
        fn sqdmulh(neon: Neon, a: i32x4<Neon>, b: i32x4<Neon>) -> i32x4<Neon> {
            use core::arch::aarch64::*;
            use fearless_simd::SimdInto;
            vqdmulhq_s32(a.into(), b.into()).simd_into(neon)
        }
    );

    #[cfg(target_arch = "aarch64")]
    fearless_simd::kernel!(
        /// `sqrdmulh`.
        #[inline(always)]
        fn sqrdmulh(neon: Neon, a: i32x4<Neon>, b: i32x4<Neon>) -> i32x4<Neon> {
            use core::arch::aarch64::*;
            use fearless_simd::SimdInto;
            vqrdmulhq_s32(a.into(), b.into()).simd_into(neon)
        }
    );

    #[cfg(all(target_arch = "aarch64", not(feature = "qext")))]
    impl Arith for Q15Neon {
        int_lanes!();
        type T = i32x4<Neon>;
        #[inline(always)]
        fn mul(self, a: Self::V, t: Self::T) -> Self::V {
            sqdmulh(self.0, a, t)
        }
        #[inline(always)]
        fn tw(self, t: [KissTwiddleScalar; 4]) -> Self::T {
            i32x4::simd_from(self.0, t.map(|v| i32::from(v) << 16))
        }
        #[inline(always)]
        fn mul2(self, a: Self::V, scale: KissTwiddleScalar) -> Self::V {
            // MULT16_32_Q16: sqdmulh(a, scale << 15) = (a*scale) >> 16; `scale << 15` is above
            // -2^31 for any 16-bit scale, so it cannot saturate.
            sqdmulh(self.0, a, i32x4::splat(self.0, i32::from(scale) << 15))
        }
    }

    /// Q31 twiddles (QEXT, `MULT32_32_P31` in its 64-bit form `(2^30 + x*t) >> 31`) on NEON:
    /// `sqrdmulh(x, t) = (2*x*t + 2^31) >> 32` is the same value except that `-2^31 * -2^31`
    /// saturates to `2^31-1` where C wraps to `-2^31`; that lane is patched.
    #[cfg(all(target_arch = "aarch64", feature = "qext"))]
    #[derive(Clone, Copy)]
    pub(crate) struct Q31Neon(pub(crate) Neon);

    #[cfg(all(target_arch = "aarch64", feature = "qext"))]
    impl Token for Q31Neon {
        type S = Neon;
        #[inline(always)]
        fn token(self) -> Self::S {
            self.0
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "qext"))]
    impl Arith for Q31Neon {
        int_lanes!();
        type T = i32x4<Neon>;
        #[inline(always)]
        fn mul(self, a: Self::V, t: Self::T) -> Self::V {
            let s = self.0;
            let r = sqrdmulh(s, a, t);
            let min = i32x4::splat(s, i32::MIN);
            let both_min = s.simd_eq_i32x4(a, min) & s.simd_eq_i32x4(t, min);
            both_min.select(min, r)
        }
        #[inline(always)]
        fn tw(self, t: [KissTwiddleScalar; 4]) -> Self::T {
            i32x4::simd_from(self.0, t)
        }
        #[inline(always)]
        fn mul2(self, a: Self::V, scale: KissTwiddleScalar) -> Self::V {
            // S_MUL2 is MULT32_32_P31 with QEXT.
            self.mul(a, i32x4::splat(self.0, scale))
        }
        #[inline(always)]
        fn tw_from(self, a: Self::V) -> Self::T {
            a
        }
    }
}

#[cfg(test)]
mod tests {
    //! SIMD == scalar for the FFT of every static mode state, in every build (the oracle tests
    //! only reach the flavour the host picks; these also run the portable fixed-point multiply
    //! on NEON hosts and run under wasmtime with SIMD128).

    use alloc::{format, vec::Vec};

    use crate::celt::kiss_fft::opus_fft_impl;
    use crate::celt::static_modes::{KissFftCpx, KissFftScalar, KissFftState, STATIC_MODE_LIST};

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
        /// Test data: float in ±32768, fixed-point below 2^24 (FFT data has headroom).
        fn scalar(&mut self) -> KissFftScalar {
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
    }

    fn states() -> Vec<&'static KissFftState> {
        STATIC_MODE_LIST
            .iter()
            .flat_map(|m| m.mdct.kfft.iter().map(|k| &**k))
            .collect()
    }

    fn scalar_fft(st: &KissFftState, data: &[KissFftScalar], downshift: i32) -> Vec<KissFftScalar> {
        let mut cpx: Vec<KissFftCpx> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[r, i]| KissFftCpx { r, i })
            .collect();
        #[cfg(feature = "fixed-point")]
        opus_fft_impl(st, &mut cpx, downshift);
        #[cfg(not(feature = "fixed-point"))]
        {
            let _ = downshift;
            opus_fft_impl(st, &mut cpx);
        }
        cpx.iter().flat_map(|c| [c.r, c.i]).collect()
    }

    /// Runs `simd` (a SIMD FFT flavour) against the scalar FFT on random data.
    fn check(what: &str, mut simd: impl FnMut(&KissFftState, &mut [KissFftScalar], i32) -> bool) {
        let mut rng = Lcg(0x5EED);
        for st in states() {
            let n = st.nfft as usize;
            for downshift in [0, 1, 3, 7] {
                if cfg!(not(feature = "fixed-point")) && downshift != 0 {
                    continue;
                }
                let data: Vec<KissFftScalar> = (0..2 * n).map(|_| rng.scalar()).collect();
                let reference = scalar_fft(st, &data, downshift);
                let mut out = data.clone();
                if !simd(st, &mut out, downshift) {
                    return;
                }
                let bits =
                    |v: &[KissFftScalar]| v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>();
                assert_eq!(
                    bits(&out),
                    bits(&reference),
                    "{what} nfft={n} downshift={downshift}"
                );
            }
        }
    }

    #[test]
    fn fft_matches_scalar() {
        check("fft_impl", |st, buf, d| {
            super::fft_impl(st, buf, d).is_some()
        });
    }

    /// The portable Q15 multiply on any SIMD level of the host (NEON hosts use `sqdmulh`
    /// in `fft_impl`).
    #[cfg(all(
        feature = "fixed-point",
        not(feature = "qext"),
        not(feature = "fixed-point-debug")
    ))]
    #[test]
    fn portable_q15_matches_scalar() {
        check("portable Q15", |st, buf, d| {
            with_simd!(s => super::fft_stages(super::fixed::Q15(s), st, buf, d)).is_some()
        });
    }

    /// `sqrdmulh` + patch == `MULT32_32_P31` (64-bit form) on edge values.
    #[cfg(all(
        target_arch = "aarch64",
        feature = "fixed-point",
        feature = "qext",
        not(feature = "fixed-point-debug")
    ))]
    #[test]
    fn q31_neon_mul_matches_scalar() {
        use super::Arith;
        use crate::celt::arch::mult32_32_p31_ovflw;
        use fearless_simd::{SimdBase, SimdFrom, i32x4};
        let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
            return;
        };
        let a = super::fixed::Q31Neon(neon);
        let edge = [
            i32::MIN,
            i32::MIN + 1,
            -1,
            0,
            1,
            1 << 30,
            i32::MAX - 1,
            i32::MAX,
        ];
        let mut rng = Lcg(3);
        let vals: Vec<i32> = edge
            .iter()
            .copied()
            .chain((0..4000).map(|_| rng.next() as i32))
            .collect();
        for x in vals.chunks(4).filter(|c| c.len() == 4) {
            for t in vals.chunks(4).take(64).filter(|c| c.len() == 4) {
                let xs: [i32; 4] = [x[0], x[1], x[2], x[3]];
                let ts: [i32; 4] = [t[0], t[1], t[2], t[3]];
                let r = a.mul(i32x4::simd_from(neon, xs), a.tw(ts));
                let expect: [i32; 4] = core::array::from_fn(|k| mult32_32_p31_ovflw(ts[k], xs[k]));
                assert_eq!(r.as_slice(), &expect, "{xs:?} * {ts:?}");
            }
        }
        // Every Q31 combination of the extremes.
        for &x in &edge {
            for &t in &edge {
                let r = a.mul(i32x4::splat(neon, x), a.tw([t; 4]));
                assert_eq!(r.as_slice()[0], mult32_32_p31_ovflw(t, x), "{x} * {t}");
            }
        }
    }

    /// The NEON Q15 multiply needs every static twiddle above `-32768`.
    #[cfg(all(feature = "fixed-point", not(feature = "qext")))]
    #[test]
    fn static_q15_twiddles_above_min() {
        for st in states() {
            let tw: &[crate::celt::static_modes::KissTwiddleCpx] = &st.twiddles;
            assert!(tw.iter().all(|c| c.r > i16::MIN && c.i > i16::MIN));
        }
    }
}

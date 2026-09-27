//! Port of celt/kiss_fft.c, celt/kiss_fft.h, celt/_kiss_fft_guts.h (float and fixed-point
//! builds).
//!
//! The state types ([`KissFftState`], [`KissFftCpx`], [`KissTwiddleCpx`]) and the static
//! 48 kHz (and QEXT 96 kHz) FFT tables live in [`crate::celt::static_modes`].
//!
//! Notes on the translation:
//! * The complex macros of `_kiss_fft_guts.h` (`C_MUL`, `C_ADD`, `C_SUB`, `C_ADDTO`,
//!   `C_MULBYSCALAR`, `HALF_OF`, `S_MUL`, `S_MUL2`) are small per-build helpers keeping the exact
//!   operation order; the butterflies are shared and written with the `*_ovflw` arithmetic
//!   macros of [`crate::celt::arch`] (plain float ops in the float build, wrapping integer ops
//!   in the fixed-point build).
//! * Fixed-point build: `kiss_fft_scalar` is `opus_int32`, twiddles are `celt_coef` (Q15, or
//!   Q31 with QEXT, `COEF_SHIFT`), `S_MUL` is `MULT16_32_Q15` (`MULT32_32_P31_ovflw` with QEXT),
//!   and `opus_fft_impl` takes the extra `downshift` argument (`ARG_FIXED`) consumed stage by
//!   stage by `fft_downshift`.
//! * The FFT core works on anything implementing the private `CpxBuf` trait so that the MDCT can
//!   run the FFT in place on an interleaved scalar buffer (C casts it to `kiss_fft_cpx*`).
//! * `opus_fft_free` / `opus_fft_alloc_arch_c` / `opus_fft_free_arch_c` have no Rust counterpart:
//!   owned states are released by `Drop` and there is no arch-specific FFT state.

use super::arch::{add32_ovflw, neg32_ovflw, sub32_ovflw};
#[cfg(feature = "fixed-point")]
use super::arch::{imin, pshr32, shr32};
use super::static_modes::{KissFftCpx, KissFftState, KissTwiddleCpx, MAXFACTORS};
pub use super::static_modes::{KissFftScalar, KissTwiddleScalar};

#[cfg(feature = "custom-modes")]
use crate::{Error, Result};
#[cfg(feature = "custom-modes")]
use alloc::{borrow::Cow, vec, vec::Vec};

/// `COEF_SHIFT` (fixed-point build): Q format of `celt_coef` plus one.
#[cfg(all(feature = "fixed-point", feature = "qext"))]
pub const COEF_SHIFT: i32 = 32;
/// `COEF_SHIFT` (fixed-point build): Q format of `celt_coef` plus one.
#[cfg(all(feature = "fixed-point", not(feature = "qext")))]
pub const COEF_SHIFT: i32 = 16;

/// `QCONST32(x, COEF_SHIFT-1)` stored in a `kiss_twiddle_scalar` (`x` is an `f`-suffixed C
/// literal).
#[cfg(feature = "fixed-point")]
const fn coef_const(x: f32) -> KissTwiddleScalar {
    super::arch::qconst32(x as f64, COEF_SHIFT - 1) as KissTwiddleScalar
}

/// A buffer of complex values the FFT can operate on in place.
///
/// Implemented for `[KissFftCpx]` and for interleaved `re, im` scalar data (the C code casts
/// `kiss_fft_scalar *` buffers to `kiss_fft_cpx *` in the MDCT).
pub(crate) trait CpxBuf {
    /// Loads element `i`.
    fn ld(&self, i: usize) -> KissFftCpx;
    /// Stores element `i`.
    fn st(&mut self, i: usize, v: KissFftCpx);
    /// The buffer as interleaved `re, im` scalars, if it is laid out that way (the SIMD FFT
    /// works on those).
    #[cfg(not(feature = "fixed-point"))]
    #[inline(always)]
    fn interleaved(&mut self) -> Option<&mut [KissFftScalar]> {
        None
    }
}

impl CpxBuf for [KissFftCpx] {
    #[inline(always)]
    fn ld(&self, i: usize) -> KissFftCpx {
        self[i]
    }
    #[inline(always)]
    fn st(&mut self, i: usize, v: KissFftCpx) {
        self[i] = v;
    }
}

/// Interleaved complex view of a scalar slice (`[r0, i0, r1, i1, ...]`).
pub(crate) struct Interleaved<'a>(pub(crate) &'a mut [KissFftScalar]);

// Perf: element `i` is accessed as pair `i` of the `[_; 2]` view (one bounds check per
// access instead of two).
impl CpxBuf for Interleaved<'_> {
    #[inline(always)]
    fn ld(&self, i: usize) -> KissFftCpx {
        let [r, im] = self.0.as_chunks::<2>().0[i];
        KissFftCpx { r, i: im }
    }
    #[inline(always)]
    fn st(&mut self, i: usize, v: KissFftCpx) {
        let p = &mut self.0.as_chunks_mut::<2>().0[i];
        p[0] = v.r;
        p[1] = v.i;
    }
    #[cfg(not(feature = "fixed-point"))]
    #[inline(always)]
    fn interleaved(&mut self) -> Option<&mut [KissFftScalar]> {
        Some(self.0)
    }
}

// ---- _kiss_fft_guts.h, float build ----

/// `S_MUL(a, b)` (float build: `a*b`).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
pub(crate) const fn s_mul(a: f32, b: f32) -> f32 {
    a * b
}
/// `S_MUL2(a, b)` (float build: `a*b`).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
pub(crate) const fn s_mul2(a: f32, b: f32) -> f32 {
    a * b
}
/// `HALF_OF(x)` (float build: `x*.5f`).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn half_of(x: f32) -> f32 {
    x * 0.5f32
}

/// `C_MUL(m, a, b)`: complex multiply.
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn c_mul(a: KissFftCpx, b: KissTwiddleCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

/// `C_ADD(res, a, b)`.
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

/// `C_SUB(res, a, b)`.
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

// ---- _kiss_fft_guts.h, fixed-point build ----

/// `S_MUL(a, b)` (fixed point: `MULT16_32_Q15(b, a)`).
#[cfg(all(feature = "fixed-point", not(feature = "qext")))]
#[inline(always)]
pub(crate) fn s_mul(a: KissFftScalar, b: KissTwiddleScalar) -> KissFftScalar {
    super::arch::mult16_32_q15(b, a)
}
/// `S_MUL2(a, b)` (fixed point: `MULT16_32_Q16(b, a)`).
#[cfg(all(feature = "fixed-point", not(feature = "qext")))]
#[inline(always)]
pub(crate) fn s_mul2(a: KissFftScalar, b: KissTwiddleScalar) -> KissFftScalar {
    super::arch::mult16_32_q16(b, a)
}
/// `S_MUL(a, b)` (fixed point + QEXT: `MULT32_32_P31_ovflw(b, a)`).
#[cfg(all(feature = "fixed-point", feature = "qext"))]
#[inline(always)]
pub(crate) fn s_mul(a: KissFftScalar, b: KissTwiddleScalar) -> KissFftScalar {
    super::arch::mult32_32_p31_ovflw(b, a)
}
/// `S_MUL2(a, b)` (fixed point + QEXT: `MULT32_32_P31_ovflw(b, a)`).
#[cfg(all(feature = "fixed-point", feature = "qext"))]
#[inline(always)]
pub(crate) fn s_mul2(a: KissFftScalar, b: KissTwiddleScalar) -> KissFftScalar {
    super::arch::mult32_32_p31_ovflw(b, a)
}
/// `HALF_OF(x)` (fixed point: `x>>1`).
#[cfg(feature = "fixed-point")]
#[inline(always)]
const fn half_of(x: KissFftScalar) -> KissFftScalar {
    x >> 1
}

/// `C_MUL(m, a, b)`: complex multiply.
#[cfg(feature = "fixed-point")]
#[inline(always)]
fn c_mul(a: KissFftCpx, b: KissTwiddleCpx) -> KissFftCpx {
    KissFftCpx {
        r: sub32_ovflw(s_mul(a.r, b.r), s_mul(a.i, b.i)),
        i: add32_ovflw(s_mul(a.r, b.i), s_mul(a.i, b.r)),
    }
}

/// `C_ADD(res, a, b)`.
#[cfg(feature = "fixed-point")]
#[inline(always)]
fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: add32_ovflw(a.r, b.r),
        i: add32_ovflw(a.i, b.i),
    }
}

/// `C_SUB(res, a, b)`.
#[cfg(feature = "fixed-point")]
#[inline(always)]
fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: sub32_ovflw(a.r, b.r),
        i: sub32_ovflw(a.i, b.i),
    }
}

/// Port of celt/kiss_fft.c:kf_bfly2 (radix-2 butterfly).
#[inline]
pub(crate) fn kf_bfly2<B: CpxBuf + ?Sized>(fout: &mut B, m: i32, n: i32) {
    #[cfg(feature = "custom-modes")]
    if m == 1 {
        celt_assert!(m == 1);
        for i in 0..n as usize {
            let f = 2 * i;
            let t = fout.ld(f + 1);
            let f0 = fout.ld(f);
            fout.st(f + 1, c_sub(f0, t));
            fout.st(f, c_add(f0, t));
        }
        return;
    }
    // QCONST32(0.7071067812f, COEF_SHIFT-1)
    #[cfg(feature = "fixed-point")]
    let tw: KissTwiddleScalar = coef_const(0.7071067812f32);
    #[cfg(not(feature = "fixed-point"))]
    let tw: KissTwiddleScalar = 0.7071067812f32;
    // We know that m==4 here because the radix-2 is just after a radix-4.
    celt_assert!(m == 4);
    for i in 0..n as usize {
        let f = 8 * i;
        let f2 = f + 4;

        let t = fout.ld(f2);
        let a = fout.ld(f);
        fout.st(f2, c_sub(a, t));
        fout.st(f, c_add(a, t));

        let b = fout.ld(f2 + 1);
        let t = KissFftCpx {
            r: s_mul(add32_ovflw(b.r, b.i), tw),
            i: s_mul(sub32_ovflw(b.i, b.r), tw),
        };
        let a = fout.ld(f + 1);
        fout.st(f2 + 1, c_sub(a, t));
        fout.st(f + 1, c_add(a, t));

        let b = fout.ld(f2 + 2);
        let t = KissFftCpx {
            r: b.i,
            i: neg32_ovflw(b.r),
        };
        let a = fout.ld(f + 2);
        fout.st(f2 + 2, c_sub(a, t));
        fout.st(f + 2, c_add(a, t));

        let b = fout.ld(f2 + 3);
        let t = KissFftCpx {
            r: s_mul(sub32_ovflw(b.i, b.r), tw),
            i: s_mul(neg32_ovflw(add32_ovflw(b.i, b.r)), tw),
        };
        let a = fout.ld(f + 3);
        fout.st(f2 + 3, c_sub(a, t));
        fout.st(f + 3, c_add(a, t));
    }
}

/// Port of celt/kiss_fft.c:kf_bfly4 (radix-4 butterfly).
#[inline]
pub(crate) fn kf_bfly4<B: CpxBuf + ?Sized>(
    fout: &mut B,
    fstride: usize,
    st: &KissFftState,
    m: i32,
    n: i32,
    mm: i32,
) {
    if m == 1 {
        // Degenerate case where all the twiddles are 1.
        for i in 0..n as usize {
            let f = 4 * i;
            let f0 = fout.ld(f);
            let f1 = fout.ld(f + 1);
            let f2 = fout.ld(f + 2);
            let f3 = fout.ld(f + 3);

            let scratch0 = c_sub(f0, f2);
            let f0 = c_add(f0, f2);
            let scratch1 = c_add(f1, f3);
            fout.st(f + 2, c_sub(f0, scratch1));
            fout.st(f, c_add(f0, scratch1));
            let scratch1 = c_sub(f1, f3);

            fout.st(
                f + 1,
                KissFftCpx {
                    r: add32_ovflw(scratch0.r, scratch1.i),
                    i: sub32_ovflw(scratch0.i, scratch1.r),
                },
            );
            fout.st(
                f + 3,
                KissFftCpx {
                    r: sub32_ovflw(scratch0.r, scratch1.i),
                    i: add32_ovflw(scratch0.i, scratch1.r),
                },
            );
        }
    } else {
        let tw = &st.twiddles[..];
        let m = m as usize;
        let m2 = 2 * m;
        let m3 = 3 * m;
        for i in 0..n as usize {
            let fbeg = i * mm as usize;
            let (mut tw1, mut tw2, mut tw3) = (0usize, 0usize, 0usize);
            // m is guaranteed to be a multiple of 4.
            for f in fbeg..fbeg + m {
                let s0 = c_mul(fout.ld(f + m), tw[tw1]);
                let s1 = c_mul(fout.ld(f + m2), tw[tw2]);
                let s2 = c_mul(fout.ld(f + m3), tw[tw3]);

                let f0 = fout.ld(f);
                let s5 = c_sub(f0, s1);
                let f0 = c_add(f0, s1);
                let s3 = c_add(s0, s2);
                let s4 = c_sub(s0, s2);
                fout.st(f + m2, c_sub(f0, s3));
                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;
                fout.st(f, c_add(f0, s3));

                fout.st(
                    f + m,
                    KissFftCpx {
                        r: add32_ovflw(s5.r, s4.i),
                        i: sub32_ovflw(s5.i, s4.r),
                    },
                );
                fout.st(
                    f + m3,
                    KissFftCpx {
                        r: sub32_ovflw(s5.r, s4.i),
                        i: add32_ovflw(s5.i, s4.r),
                    },
                );
            }
        }
    }
}

/// Port of celt/kiss_fft.c:kf_bfly3 (radix-3 butterfly).
#[inline]
pub(crate) fn kf_bfly3<B: CpxBuf + ?Sized>(
    fout: &mut B,
    fstride: usize,
    st: &KissFftState,
    m: i32,
    n: i32,
    mm: i32,
) {
    let tw = &st.twiddles[..];
    let m = m as usize;
    let m2 = 2 * m;
    // epi3.i (epi3.r is unused)
    #[cfg(feature = "fixed-point")]
    let epi3_i: KissTwiddleScalar = -coef_const(0.86602540f32);
    #[cfg(not(feature = "fixed-point"))]
    let epi3_i: KissTwiddleScalar = tw[fstride * m].i;
    for i in 0..n as usize {
        let mut f = i * mm as usize;
        let (mut tw1, mut tw2) = (0usize, 0usize);
        // For non-custom modes, m is guaranteed to be a multiple of 4.
        let mut k = m;
        loop {
            let s1 = c_mul(fout.ld(f + m), tw[tw1]);
            let s2 = c_mul(fout.ld(f + m2), tw[tw2]);

            let s3 = c_add(s1, s2);
            let mut s0 = c_sub(s1, s2);
            tw1 += fstride;
            tw2 += fstride * 2;

            let f0 = fout.ld(f);
            let fm = KissFftCpx {
                r: sub32_ovflw(f0.r, half_of(s3.r)),
                i: sub32_ovflw(f0.i, half_of(s3.i)),
            };

            // C_MULBYSCALAR(scratch[0], epi3.i)
            s0.r = s_mul(s0.r, epi3_i);
            s0.i = s_mul(s0.i, epi3_i);

            fout.st(f, c_add(f0, s3));

            fout.st(
                f + m2,
                KissFftCpx {
                    r: add32_ovflw(fm.r, s0.i),
                    i: sub32_ovflw(fm.i, s0.r),
                },
            );

            fout.st(
                f + m,
                KissFftCpx {
                    r: sub32_ovflw(fm.r, s0.i),
                    i: add32_ovflw(fm.i, s0.r),
                },
            );

            f += 1;
            k -= 1;
            if k == 0 {
                break;
            }
        }
    }
}

/// Port of celt/kiss_fft.c:kf_bfly5 (radix-5 butterfly).
#[inline]
pub(crate) fn kf_bfly5<B: CpxBuf + ?Sized>(
    fout: &mut B,
    fstride: usize,
    st: &KissFftState,
    m: i32,
    n: i32,
    mm: i32,
) {
    let tw = &st.twiddles[..];
    let m = m as usize;
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

    for i in 0..n as usize {
        let f0i = i * mm as usize;
        let f1i = f0i + m;
        let f2i = f0i + 2 * m;
        let f3i = f0i + 3 * m;
        let f4i = f0i + 4 * m;

        // For non-custom modes, m is guaranteed to be a multiple of 4.
        for u in 0..m {
            let s0 = fout.ld(f0i + u);

            let s1 = c_mul(fout.ld(f1i + u), tw[u * fstride]);
            let s2 = c_mul(fout.ld(f2i + u), tw[2 * u * fstride]);
            let s3 = c_mul(fout.ld(f3i + u), tw[3 * u * fstride]);
            let s4 = c_mul(fout.ld(f4i + u), tw[4 * u * fstride]);

            let s7 = c_add(s1, s4);
            let s10 = c_sub(s1, s4);
            let s8 = c_add(s2, s3);
            let s9 = c_sub(s2, s3);

            fout.st(
                f0i + u,
                KissFftCpx {
                    r: add32_ovflw(s0.r, add32_ovflw(s7.r, s8.r)),
                    i: add32_ovflw(s0.i, add32_ovflw(s7.i, s8.i)),
                },
            );

            let s5 = KissFftCpx {
                r: add32_ovflw(s0.r, add32_ovflw(s_mul(s7.r, ya.r), s_mul(s8.r, yb.r))),
                i: add32_ovflw(s0.i, add32_ovflw(s_mul(s7.i, ya.r), s_mul(s8.i, yb.r))),
            };

            let s6 = KissFftCpx {
                r: add32_ovflw(s_mul(s10.i, ya.i), s_mul(s9.i, yb.i)),
                i: neg32_ovflw(add32_ovflw(s_mul(s10.r, ya.i), s_mul(s9.r, yb.i))),
            };

            fout.st(f1i + u, c_sub(s5, s6));
            fout.st(f4i + u, c_add(s5, s6));

            let s11 = KissFftCpx {
                r: add32_ovflw(s0.r, add32_ovflw(s_mul(s7.r, yb.r), s_mul(s8.r, ya.r))),
                i: add32_ovflw(s0.i, add32_ovflw(s_mul(s7.i, yb.r), s_mul(s8.i, ya.r))),
            };
            let s12 = KissFftCpx {
                r: sub32_ovflw(s_mul(s9.i, ya.i), s_mul(s10.i, yb.i)),
                i: sub32_ovflw(s_mul(s10.r, yb.i), s_mul(s9.r, ya.i)),
            };

            fout.st(f2i + u, c_add(s11, s12));
            fout.st(f3i + u, c_sub(s11, s12));
        }
    }
}

/// Port of celt/kiss_fft.c:compute_bitrev_table.
///
/// `f` is the whole bitrev table and `fi` the C `f` pointer offset into it. The unused `st`
/// parameter of the C function is dropped.
#[cfg(feature = "custom-modes")]
fn compute_bitrev_table(
    mut fout: i32,
    f: &mut [i16],
    mut fi: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[i16],
) {
    let p = factors[0] as i32; // the radix
    let m = factors[1] as i32; // stage's fft length/p

    if m == 1 {
        for j in 0..p {
            f[fi] = (fout + j) as i16;
            fi += fstride * in_stride;
        }
    } else {
        for _ in 0..p {
            compute_bitrev_table(fout, f, fi, fstride * p as usize, in_stride, &factors[2..]);
            fi += fstride * in_stride;
            fout += m;
        }
    }
}

/// Port of celt/kiss_fft.c:kf_factor.
///
/// `facbuf` is populated by `p1, m1, p2, m2, ...` where `p[i] * m[i] = m[i-1]`, `m0 = n`.
/// Returns `false` if `n` has a prime factor larger than 5.
///
/// Deviation: C would write past `facbuf` for sizes needing more than `MAXFACTORS` stages
/// (undefined behaviour, `nfft >= 2^17`); the port reports failure instead.
#[cfg(feature = "custom-modes")]
fn kf_factor(mut n: i32, facbuf: &mut [i16; 2 * MAXFACTORS]) -> bool {
    let mut p: i32 = 4;
    let mut stages: usize = 0;
    let nbak = n;

    // factor out powers of 4, powers of 2, then any remaining primes
    loop {
        while n % p != 0 {
            match p {
                4 => p = 2,
                2 => p = 3,
                _ => p += 2,
            }
            if p > 32000 || p * p > n {
                p = n; // no more factors, skip to end
            }
        }
        n /= p;
        if p > 5 {
            return false;
        }
        if stages >= MAXFACTORS {
            return false;
        }
        facbuf[2 * stages] = p as i16;
        if p == 2 && stages > 1 {
            facbuf[2 * stages] = 4;
            facbuf[2] = 2;
        }
        stages += 1;
        if n <= 1 {
            break;
        }
    }
    n = nbak;
    // Reverse the order to get the radix 4 at the end, so we can use the fast degenerate case.
    // It turns out that reversing the order also improves the noise behaviour.
    for i in 0..stages / 2 {
        facbuf.swap(2 * i, 2 * (stages - i - 1));
    }
    for i in 0..stages {
        n /= facbuf[2 * i] as i32;
        facbuf[2 * i + 1] = n as i16;
    }
    true
}

/// Port of celt/kiss_fft.c:compute_twiddles (float build).
#[cfg(all(feature = "custom-modes", not(feature = "fixed-point")))]
fn compute_twiddles(twiddles: &mut [KissTwiddleCpx], nfft: i32) {
    for (i, tw) in twiddles[..nfft as usize].iter_mut().enumerate() {
        const PI: f64 = 3.14159265358979323846264338327;
        let phase: f64 = (-2.0 * PI / nfft as f64) * i as f64;
        // kf_cexp(twiddles+i, phase)
        tw.r = crate::math::cos(phase) as f32;
        tw.i = crate::math::sin(phase) as f32;
    }
}

/// Port of celt/kiss_fft.c:compute_twiddles (fixed-point build, QEXT: Q31 twiddles).
#[cfg(all(feature = "custom-modes", feature = "fixed-point", feature = "qext"))]
fn compute_twiddles(twiddles: &mut [KissTwiddleCpx], nfft: i32) {
    // C `M_PI` (the platform <math.h> value, the double nearest to pi).
    const M_PI: f64 = core::f64::consts::PI;
    for (i, tw) in twiddles[..nfft as usize].iter_mut().enumerate() {
        let phase: i32 = -(i as i32);
        // (int)MIN32(2147483647, floor(.5+2147483648*cos((2*M_PI/nfft)*phase)))
        let arg = (2.0 * M_PI / nfft as f64) * phase as f64;
        let r = crate::math::floor(0.5 + 2147483648.0 * crate::math::cos(arg));
        let im = crate::math::floor(0.5 + 2147483648.0 * crate::math::sin(arg));
        tw.r = (if 2147483647.0 < r { 2147483647.0 } else { r }) as i32;
        tw.i = (if 2147483647.0 < im { 2147483647.0 } else { im }) as i32;
    }
}

/// Port of celt/kiss_fft.c:compute_twiddles (fixed-point build: `kf_cexp2` on Q15 phases).
#[cfg(all(
    feature = "custom-modes",
    feature = "fixed-point",
    not(feature = "qext")
))]
fn compute_twiddles(twiddles: &mut [KissTwiddleCpx], nfft: i32) {
    use super::arch::{div32, shl32};
    use super::mathops::celt_cos_norm;
    for (i, tw) in twiddles[..nfft as usize].iter_mut().enumerate() {
        let phase: i32 = -(i as i32);
        // kf_cexp2(twiddles+i, DIV32(SHL32(phase,17),nfft)), TRIG_UPSCALE == 1
        let p = div32(shl32(phase, 17), nfft);
        tw.r = celt_cos_norm(p);
        tw.i = celt_cos_norm(p - 32768);
    }
}

/// `scale` and `scale_shift` of an FFT state of size `nfft` (the `FIXED_POINT` part of
/// celt/kiss_fft.c:opus_fft_alloc_twiddles).
#[cfg(all(feature = "custom-modes", feature = "fixed-point"))]
const fn fft_scale(nfft: i32) -> (super::arch::CeltCoef, i32) {
    let scale_shift = super::mathops::celt_ilog2(nfft);
    #[cfg(feature = "qext")]
    let scale = if nfft == 1 << scale_shift {
        super::arch::qconst32(1.0f32 as f64, 30)
    } else {
        (((1_073_741_824i64 << scale_shift) + (nfft / 2) as i64) / nfft as i64) as i32
    };
    #[cfg(not(feature = "qext"))]
    let scale = if nfft == 1 << scale_shift {
        super::arch::Q15ONE
    } else {
        (((1_073_741_824 + nfft / 2) / nfft) >> (15 - scale_shift)) as i16
    };
    (scale, scale_shift)
}

/// Port of celt/kiss_fft.c:opus_fft_alloc_twiddles.
///
/// Allocates an FFT state of size `nfft`. With `base`, the twiddles of `base` are reused
/// (`base.nfft` must be `nfft << shift` for some `shift < 32`); the Rust state owns a copy of
/// them since it cannot borrow from `base`. Returns [`Error::AllocFail`] where C returns `NULL`
/// (unsupported size or no matching shift) and [`Error::BadArg`] for `nfft <= 0`.
///
/// The `mem`/`lenmem` caller-provided storage and `arch` parameters of C are dropped.
#[cfg(feature = "custom-modes")]
pub fn opus_fft_alloc_twiddles(nfft: i32, base: Option<&KissFftState>) -> Result<KissFftState> {
    if nfft <= 0 {
        return Err(Error::BadArg);
    }
    #[cfg(feature = "fixed-point")]
    let (scale, scale_shift) = fft_scale(nfft);
    #[cfg(not(feature = "fixed-point"))]
    let scale = 1.0f32 / nfft as f32;
    let (twiddles, shift) = match base {
        Some(base) => {
            let mut shift = 0i32;
            while shift < 32 && nfft << shift != base.nfft {
                shift += 1;
            }
            if shift >= 32 {
                return Err(Error::AllocFail);
            }
            (base.twiddles.clone(), shift)
        }
        None => {
            let mut tw = vec![KissTwiddleCpx::default(); nfft as usize];
            compute_twiddles(&mut tw, nfft);
            (Cow::Owned(tw), -1)
        }
    };
    let mut factors = [0i16; 2 * MAXFACTORS];
    if !kf_factor(nfft, &mut factors) {
        return Err(Error::AllocFail);
    }

    // bitrev
    let mut bitrev: Vec<i16> = vec![0; nfft as usize];
    compute_bitrev_table(0, &mut bitrev, 0, 1, 1, &factors);

    // opus_fft_alloc_arch: no architecture specific state.
    Ok(KissFftState {
        nfft,
        scale,
        #[cfg(feature = "fixed-point")]
        scale_shift,
        shift,
        factors,
        bitrev: Cow::Owned(bitrev),
        twiddles,
    })
}

/// Port of celt/kiss_fft.c:opus_fft_alloc.
///
/// # Errors
/// See [`opus_fft_alloc_twiddles`].
#[cfg(feature = "custom-modes")]
pub fn opus_fft_alloc(nfft: i32) -> Result<KissFftState> {
    opus_fft_alloc_twiddles(nfft, None)
}

/// Port of celt/kiss_fft.c:fft_downshift (fixed-point build): shifts the `n` values of `x`
/// right by `min(step, *total)` (rounding unless the shift is 1) and deducts it from `*total`.
#[cfg(feature = "fixed-point")]
fn fft_downshift<B: CpxBuf + ?Sized>(x: &mut B, n: usize, total: &mut i32, step: i32) {
    let shift = imin(step, *total);
    *total -= shift;
    if shift == 1 {
        for i in 0..n {
            let v = x.ld(i);
            x.st(
                i,
                KissFftCpx {
                    r: shr32(v.r, 1),
                    i: shr32(v.i, 1),
                },
            );
        }
    } else if shift > 0 {
        for i in 0..n {
            let v = x.ld(i);
            x.st(
                i,
                KissFftCpx {
                    r: pshr32(v.r, shift),
                    i: pshr32(v.i, shift),
                },
            );
        }
    }
}

/// `fft_downshift` (float build: expands to nothing).
#[cfg(not(feature = "fixed-point"))]
#[inline(always)]
const fn fft_downshift<B: CpxBuf + ?Sized>(_x: &mut B, _n: usize, _total: &mut i32, _step: i32) {}

/// Generic core of [`opus_fft_impl`], usable on any [`CpxBuf`]. `downshift` is the C
/// `ARG_FIXED(downshift)` argument (ignored in the float build).
pub(crate) fn opus_fft_impl_buf<B: CpxBuf + ?Sized>(
    st: &KissFftState,
    fout: &mut B,
    mut downshift: i32,
) {
    // Float build: vertical SIMD butterflies (bit-identical) when the target has SIMD.
    #[cfg(not(feature = "fixed-point"))]
    if let Some(buf) = fout.interleaved()
        && super::simd::fft_impl(st, buf).is_some()
    {
        return;
    }
    // One extra entry compared to C (`fstride[MAXFACTORS]`) so a state with MAXFACTORS stages
    // cannot index out of bounds (C would write past the array).
    let mut fstride = [0i32; MAXFACTORS + 1];
    let nfft = st.nfft as usize;

    // st->shift can be -1
    let shift = if st.shift > 0 { st.shift } else { 0 };

    fstride[0] = 1;
    let mut l: usize = 0;
    loop {
        let p = st.factors[2 * l] as i32;
        let m = st.factors[2 * l + 1] as i32;
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }
    let mut m = st.factors[2 * l - 1] as i32;
    for i in (0..l).rev() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as i32
        } else {
            1
        };
        let fs = (fstride[i] << shift) as usize;
        match st.factors[2 * i] {
            2 => {
                fft_downshift(fout, nfft, &mut downshift, 1);
                kf_bfly2(fout, m, fstride[i]);
            }
            4 => {
                fft_downshift(fout, nfft, &mut downshift, 2);
                kf_bfly4(fout, fs, st, m, fstride[i], m2);
            }
            3 => {
                fft_downshift(fout, nfft, &mut downshift, 2);
                kf_bfly3(fout, fs, st, m, fstride[i], m2);
            }
            5 => {
                fft_downshift(fout, nfft, &mut downshift, 3);
                kf_bfly5(fout, fs, st, m, fstride[i], m2);
            }
            _ => {}
        }
        m = m2;
    }
    let rest = downshift;
    fft_downshift(fout, nfft, &mut downshift, rest);
}

/// Port of celt/kiss_fft.c:opus_fft_impl: in-place FFT of bit-reversed input `fout`
/// (`st.nfft` elements). The fixed-point build takes the extra `downshift` argument (total
/// right shift applied across the stages, `ARG_FIXED`).
pub fn opus_fft_impl(
    st: &KissFftState,
    fout: &mut [KissFftCpx],
    #[cfg(feature = "fixed-point")] downshift: i32,
) {
    #[cfg(not(feature = "fixed-point"))]
    let downshift = 0;
    opus_fft_impl_buf(st, &mut fout[..st.nfft as usize], downshift);
}

/// Port of celt/kiss_fft.c:opus_fft_c (the `opus_fft` macro): forward FFT with `1/nfft`
/// scaling. `fin` and `fout` must hold at least `st.nfft` elements (in-place is not supported,
/// which the borrow checker enforces).
pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let n = st.nfft as usize;
    let scale = st.scale;
    // Allows us to scale with MULT16_32_Q16(), which is faster than MULT16_32_Q15() on ARM.
    #[cfg(feature = "fixed-point")]
    let scale_shift = st.scale_shift - 1;
    #[cfg(not(feature = "fixed-point"))]
    let scale_shift = 0;
    let bitrev = &st.bitrev[..n];
    let fin = &fin[..n];
    let fout = &mut fout[..n];
    // Float build: the SIMD FFT needs interleaved scalars, which `[KissFftCpx]` cannot be viewed
    // as in safe code; run it on a copy (up to the 480-point analysis FFT).
    #[cfg(not(feature = "fixed-point"))]
    if n <= SIMD_FFT_MAX {
        let mut buf = [0f32; 2 * SIMD_FFT_MAX];
        let pairs = buf[..2 * n].as_chunks_mut::<2>().0;
        // Bit-reverse the input
        for (x, &rev) in fin.iter().zip(bitrev) {
            pairs[rev as usize] = [s_mul2(x.r, scale), s_mul2(x.i, scale)];
        }
        if super::simd::fft_impl(st, &mut buf[..2 * n]).is_some() {
            for (o, &[r, i]) in fout.iter_mut().zip(buf[..2 * n].as_chunks::<2>().0) {
                *o = KissFftCpx { r, i };
            }
            return;
        }
    }
    // Bit-reverse the input
    for (x, &rev) in fin.iter().zip(bitrev) {
        let o = &mut fout[rev as usize];
        o.r = s_mul2(x.r, scale);
        o.i = s_mul2(x.i, scale);
    }
    opus_fft_impl_buf(st, fout, scale_shift);
}

/// Largest `nfft` for which [`opus_fft`] (float build) runs the SIMD FFT on a stack copy.
#[cfg(not(feature = "fixed-point"))]
const SIMD_FFT_MAX: usize = 480;

/// Port of celt/kiss_fft.c:opus_ifft_c (the `opus_ifft` macro): unscaled inverse FFT.
pub fn opus_ifft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let n = st.nfft as usize;
    let bitrev = &st.bitrev[..n];
    let fin = &fin[..n];
    let fout = &mut fout[..n];
    // Bit-reverse the input
    for (x, &rev) in fin.iter().zip(bitrev) {
        fout[rev as usize] = *x;
    }
    for x in fout.iter_mut() {
        x.i = -x.i;
    }
    opus_fft_impl_buf(st, fout, 0);
    for x in fout.iter_mut() {
        x.i = -x.i;
    }
}

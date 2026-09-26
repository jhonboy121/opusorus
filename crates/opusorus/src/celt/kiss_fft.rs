//! Port of celt/kiss_fft.c, celt/kiss_fft.h, celt/_kiss_fft_guts.h (float build).
//!
//! The state types ([`KissFftState`], [`KissFftCpx`], [`KissTwiddleCpx`]) and the static
//! 48 kHz (and QEXT 96 kHz) FFT tables live in [`crate::celt::static_modes`].
//!
//! Notes on the translation:
//! * The complex macros of `_kiss_fft_guts.h` (`C_MUL`, `C_ADD`, `C_SUB`, `C_ADDTO`,
//!   `C_MULBYSCALAR`, `HALF_OF`, ...) are expanded inline with their float definitions, keeping
//!   the exact operation order. The `*_ovflw` arithmetic macros are plain float ops in the float
//!   build.
//! * The FFT core works on anything implementing the private `CpxBuf` trait so that the MDCT can
//!   run the FFT in place on an interleaved `f32` buffer (C casts `float*` to `kiss_fft_cpx*`).
//! * `opus_fft_free` / `opus_fft_alloc_arch_c` / `opus_fft_free_arch_c` have no Rust counterpart:
//!   owned states are released by `Drop` and there is no arch-specific FFT state.

use super::static_modes::{KissFftCpx, KissFftState, KissTwiddleCpx, MAXFACTORS};

#[cfg(feature = "custom-modes")]
use crate::{Error, Result};
#[cfg(feature = "custom-modes")]
use alloc::{borrow::Cow, vec, vec::Vec};

/// `kiss_fft_scalar` (float build).
pub type KissFftScalar = f32;
/// `kiss_twiddle_scalar` (float build).
pub type KissTwiddleScalar = f32;

// FIXED_POINT: not ported (float build) — `fft_downshift`, `S_MUL`/`C_MUL` fixed variants,
// `scale_shift`, `KISS_FFT_COS/SIN` fixed variants, `kf_cexp2`.

/// A buffer of complex values the FFT can operate on in place.
///
/// Implemented for `[KissFftCpx]` and for interleaved `re, im` `f32` data (the C code casts
/// `float *` buffers to `kiss_fft_cpx *` in the MDCT).
pub(crate) trait CpxBuf {
    /// Loads element `i`.
    fn ld(&self, i: usize) -> KissFftCpx;
    /// Stores element `i`.
    fn st(&mut self, i: usize, v: KissFftCpx);
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

/// Interleaved complex view of an `f32` slice (`[r0, i0, r1, i1, ...]`).
pub(crate) struct Interleaved<'a>(pub(crate) &'a mut [f32]);

impl CpxBuf for Interleaved<'_> {
    #[inline(always)]
    fn ld(&self, i: usize) -> KissFftCpx {
        KissFftCpx {
            r: self.0[2 * i],
            i: self.0[2 * i + 1],
        }
    }
    #[inline(always)]
    fn st(&mut self, i: usize, v: KissFftCpx) {
        self.0[2 * i] = v.r;
        self.0[2 * i + 1] = v.i;
    }
}

/// `C_MUL(m, a, b)`: complex multiply.
#[inline(always)]
const fn c_mul(a: KissFftCpx, b: KissTwiddleCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

/// `C_ADD(res, a, b)`.
#[inline(always)]
const fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

/// `C_SUB(res, a, b)`.
#[inline(always)]
const fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

/// Port of celt/kiss_fft.c:kf_bfly2 (radix-2 butterfly).
///
#[inline]
fn kf_bfly2<B: CpxBuf + ?Sized>(fout: &mut B, m: i32, n: i32) {
    #[cfg(feature = "custom-modes")]
    if m == 1 {
        debug_assert!(m == 1);
        for i in 0..n as usize {
            let f = 2 * i;
            let t = fout.ld(f + 1);
            let f0 = fout.ld(f);
            fout.st(f + 1, c_sub(f0, t));
            fout.st(f, c_add(f0, t));
        }
        return;
    }
    let tw: f32 = 0.7071067812f32;
    // We know that m==4 here because the radix-2 is just after a radix-4.
    debug_assert!(m == 4);
    for i in 0..n as usize {
        let f = 8 * i;
        let f2 = f + 4;

        let t = fout.ld(f2);
        let a = fout.ld(f);
        fout.st(f2, c_sub(a, t));
        fout.st(f, c_add(a, t));

        let b = fout.ld(f2 + 1);
        let t = KissFftCpx {
            r: (b.r + b.i) * tw,
            i: (b.i - b.r) * tw,
        };
        let a = fout.ld(f + 1);
        fout.st(f2 + 1, c_sub(a, t));
        fout.st(f + 1, c_add(a, t));

        let b = fout.ld(f2 + 2);
        let t = KissFftCpx { r: b.i, i: -b.r };
        let a = fout.ld(f + 2);
        fout.st(f2 + 2, c_sub(a, t));
        fout.st(f + 2, c_add(a, t));

        let b = fout.ld(f2 + 3);
        let t = KissFftCpx {
            r: (b.i - b.r) * tw,
            i: (-(b.i + b.r)) * tw,
        };
        let a = fout.ld(f + 3);
        fout.st(f2 + 3, c_sub(a, t));
        fout.st(f + 3, c_add(a, t));
    }
}

/// Port of celt/kiss_fft.c:kf_bfly4 (radix-4 butterfly).
#[inline]
fn kf_bfly4<B: CpxBuf + ?Sized>(
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
                    r: scratch0.r + scratch1.i,
                    i: scratch0.i - scratch1.r,
                },
            );
            fout.st(
                f + 3,
                KissFftCpx {
                    r: scratch0.r - scratch1.i,
                    i: scratch0.i + scratch1.r,
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
                        r: s5.r + s4.i,
                        i: s5.i - s4.r,
                    },
                );
                fout.st(
                    f + m3,
                    KissFftCpx {
                        r: s5.r - s4.i,
                        i: s5.i + s4.r,
                    },
                );
            }
        }
    }
}

/// Port of celt/kiss_fft.c:kf_bfly3 (radix-3 butterfly).
#[inline]
fn kf_bfly3<B: CpxBuf + ?Sized>(
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
    // FIXED_POINT: not ported (float build) — fixed epi3 constant.
    let epi3 = tw[fstride * m];
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
                r: f0.r - s3.r * 0.5f32,
                i: f0.i - s3.i * 0.5f32,
            };

            // C_MULBYSCALAR(scratch[0], epi3.i)
            s0.r *= epi3.i;
            s0.i *= epi3.i;

            fout.st(f, c_add(f0, s3));

            fout.st(
                f + m2,
                KissFftCpx {
                    r: fm.r + s0.i,
                    i: fm.i - s0.r,
                },
            );

            fout.st(
                f + m,
                KissFftCpx {
                    r: fm.r - s0.i,
                    i: fm.i + s0.r,
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
fn kf_bfly5<B: CpxBuf + ?Sized>(
    fout: &mut B,
    fstride: usize,
    st: &KissFftState,
    m: i32,
    n: i32,
    mm: i32,
) {
    let tw = &st.twiddles[..];
    let m = m as usize;
    // FIXED_POINT: not ported (float build) — fixed ya/yb constants.
    let ya = tw[fstride * m];
    let yb = tw[fstride * 2 * m];

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
                    r: s0.r + (s7.r + s8.r),
                    i: s0.i + (s7.i + s8.i),
                },
            );

            let s5 = KissFftCpx {
                r: s0.r + (s7.r * ya.r + s8.r * yb.r),
                i: s0.i + (s7.i * ya.r + s8.i * yb.r),
            };

            let s6 = KissFftCpx {
                r: s10.i * ya.i + s9.i * yb.i,
                i: -(s10.r * ya.i + s9.r * yb.i),
            };

            fout.st(f1i + u, c_sub(s5, s6));
            fout.st(f4i + u, c_add(s5, s6));

            let s11 = KissFftCpx {
                r: s0.r + (s7.r * yb.r + s8.r * ya.r),
                i: s0.i + (s7.i * yb.r + s8.i * ya.r),
            };
            let s12 = KissFftCpx {
                r: s9.i * ya.i - s10.i * yb.i,
                i: s10.r * yb.i - s9.r * ya.i,
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
#[cfg(feature = "custom-modes")]
fn compute_twiddles(twiddles: &mut [KissTwiddleCpx], nfft: i32) {
    // FIXED_POINT: not ported (float build).
    for (i, tw) in twiddles[..nfft as usize].iter_mut().enumerate() {
        const PI: f64 = 3.14159265358979323846264338327;
        let phase: f64 = (-2.0 * PI / nfft as f64) * i as f64;
        // kf_cexp(twiddles+i, phase)
        tw.r = crate::math::cos(phase) as f32;
        tw.i = crate::math::sin(phase) as f32;
    }
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
    // FIXED_POINT: not ported (float build) — scale_shift / fixed scale.
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

/// Generic core of [`opus_fft_impl`], usable on any [`CpxBuf`].
pub(crate) fn opus_fft_impl_buf<B: CpxBuf + ?Sized>(st: &KissFftState, fout: &mut B) {
    // One extra entry compared to C (`fstride[MAXFACTORS]`) so a state with MAXFACTORS stages
    // cannot index out of bounds (C would write past the array).
    let mut fstride = [0i32; MAXFACTORS + 1];

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
            2 => kf_bfly2(fout, m, fstride[i]),
            4 => kf_bfly4(fout, fs, st, m, fstride[i], m2),
            3 => kf_bfly3(fout, fs, st, m, fstride[i], m2),
            5 => kf_bfly5(fout, fs, st, m, fstride[i], m2),
            _ => {}
        }
        m = m2;
    }
}

/// Port of celt/kiss_fft.c:opus_fft_impl: in-place FFT of bit-reversed input `fout`
/// (`st.nfft` elements).
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx]) {
    opus_fft_impl_buf(st, &mut fout[..st.nfft as usize]);
}

/// Port of celt/kiss_fft.c:opus_fft_c (the `opus_fft` macro): forward FFT with `1/nfft`
/// scaling. `fin` and `fout` must hold at least `st.nfft` elements (in-place is not supported,
/// which the borrow checker enforces).
pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let n = st.nfft as usize;
    let scale = st.scale;
    let bitrev = &st.bitrev[..n];
    let fin = &fin[..n];
    let fout = &mut fout[..n];
    // Bit-reverse the input
    for (x, &rev) in fin.iter().zip(bitrev) {
        let o = &mut fout[rev as usize];
        o.r = x.r * scale;
        o.i = x.i * scale;
    }
    opus_fft_impl_buf(st, fout);
}

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
    opus_fft_impl_buf(st, fout);
    for x in fout.iter_mut() {
        x.i = -x.i;
    }
}

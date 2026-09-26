//! Port of celt/mini_kfft.c (QEXT): a minimalist, concatenated kiss-fft used to compute real
//! scalar FFTs (used upstream by `qext_compare` and DNN tooling).
//!
//! Notes on the translation:
//! * `mini_kiss_fft_cpx` is its own float complex type ([`MiniKissFftCpx`]): `mini_kfft.c` is
//!   float-only, also in fixed-point builds (where `kiss_fft_cpx` holds integers).
//! * The `mem`/`lenmem` caller-provided storage mode of the C allocators is dropped; states own
//!   `Vec`s and are released by `Drop` (C uses `free`).
//! * C `assert`s that would abort (odd real-FFT size, radix not in 2..=5) are turned into
//!   [`Error::BadArg`] at allocation time, so the transforms themselves cannot fail.

use crate::{Error, Result};
use alloc::{vec, vec::Vec};

/// `mini_kiss_fft_scalar`.
pub type MiniKissFftScalar = f32;
/// `mini_kiss_fft_cpx`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MiniKissFftCpx {
    /// Real part.
    pub r: MiniKissFftScalar,
    /// Imaginary part.
    pub i: MiniKissFftScalar,
}

/// `MINI_MAXFACTORS`.
pub const MINI_MAXFACTORS: usize = 32;

/// `mini_kiss_fft_state`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniKissFftState {
    pub nfft: i32,
    pub inverse: bool,
    pub factors: [i32; 2 * MINI_MAXFACTORS],
    pub twiddles: Vec<MiniKissFftCpx>,
}

/// `mini_kiss_fftr_state`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniKissFftrState {
    pub substate: MiniKissFftState,
    pub tmpbuf: Vec<MiniKissFftCpx>,
    pub super_twiddles: Vec<MiniKissFftCpx>,
}

/// Read-only source of complex input values (`const mini_kiss_fft_cpx *`), either a complex
/// slice or a real slice reinterpreted as `re, im` pairs (C casts `timedata` in
/// `mini_kiss_fftr`).
trait CpxSrc {
    fn get(&self, i: usize) -> MiniKissFftCpx;
}

impl CpxSrc for [MiniKissFftCpx] {
    #[inline(always)]
    fn get(&self, i: usize) -> MiniKissFftCpx {
        self[i]
    }
}

/// Real samples viewed as interleaved complex pairs.
struct RealPairs<'a>(&'a [MiniKissFftScalar]);

impl CpxSrc for RealPairs<'_> {
    #[inline(always)]
    fn get(&self, i: usize) -> MiniKissFftCpx {
        MiniKissFftCpx {
            r: self.0[2 * i],
            i: self.0[2 * i + 1],
        }
    }
}

/// `C_MUL`.
#[inline(always)]
const fn c_mul(a: MiniKissFftCpx, b: MiniKissFftCpx) -> MiniKissFftCpx {
    MiniKissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

/// `C_ADD`.
#[inline(always)]
const fn c_add(a: MiniKissFftCpx, b: MiniKissFftCpx) -> MiniKissFftCpx {
    MiniKissFftCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

/// `C_SUB`.
#[inline(always)]
const fn c_sub(a: MiniKissFftCpx, b: MiniKissFftCpx) -> MiniKissFftCpx {
    MiniKissFftCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

/// `MINI_HALF_OF(x)`: `x * (mini_kiss_fft_scalar).5` (float multiply).
#[inline(always)]
const fn mini_half_of(x: MiniKissFftScalar) -> MiniKissFftScalar {
    x * 0.5f32
}

/// `mini_kf_cexp(x, phase)`.
#[inline(always)]
fn mini_kf_cexp(phase: f64) -> MiniKissFftCpx {
    MiniKissFftCpx {
        r: crate::math::cos(phase) as MiniKissFftScalar,
        i: crate::math::sin(phase) as MiniKissFftScalar,
    }
}

/// Port of celt/mini_kfft.c:kf_bfly2.
fn kf_bfly2(fout: &mut [MiniKissFftCpx], fstride: usize, st: &MiniKissFftState, m: usize) {
    let tw = &st.twiddles[..];
    let mut tw1 = 0usize;
    // C_FIXDIV: no-op in the float build.
    for k in 0..m {
        let t = c_mul(fout[m + k], tw[tw1]);
        tw1 += fstride;
        fout[m + k] = c_sub(fout[k], t);
        fout[k] = c_add(fout[k], t);
    }
}

/// Port of celt/mini_kfft.c:kf_bfly4.
fn kf_bfly4(fout: &mut [MiniKissFftCpx], fstride: usize, st: &MiniKissFftState, m: usize) {
    let tw = &st.twiddles[..];
    let (mut tw1, mut tw2, mut tw3) = (0usize, 0usize, 0usize);
    let m2 = 2 * m;
    let m3 = 3 * m;

    for f in 0..m {
        let s0 = c_mul(fout[f + m], tw[tw1]);
        let s1 = c_mul(fout[f + m2], tw[tw2]);
        let s2 = c_mul(fout[f + m3], tw[tw3]);

        let s5 = c_sub(fout[f], s1);
        fout[f] = c_add(fout[f], s1);
        let s3 = c_add(s0, s2);
        let s4 = c_sub(s0, s2);
        fout[f + m2] = c_sub(fout[f], s3);
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        fout[f] = c_add(fout[f], s3);

        if st.inverse {
            fout[f + m] = MiniKissFftCpx {
                r: s5.r - s4.i,
                i: s5.i + s4.r,
            };
            fout[f + m3] = MiniKissFftCpx {
                r: s5.r + s4.i,
                i: s5.i - s4.r,
            };
        } else {
            fout[f + m] = MiniKissFftCpx {
                r: s5.r + s4.i,
                i: s5.i - s4.r,
            };
            fout[f + m3] = MiniKissFftCpx {
                r: s5.r - s4.i,
                i: s5.i + s4.r,
            };
        }
    }
}

/// Port of celt/mini_kfft.c:kf_bfly3.
fn kf_bfly3(fout: &mut [MiniKissFftCpx], fstride: usize, st: &MiniKissFftState, m: usize) {
    let tw = &st.twiddles[..];
    let m2 = 2 * m;
    let epi3 = tw[fstride * m];
    let (mut tw1, mut tw2) = (0usize, 0usize);

    for f in 0..m {
        let s1 = c_mul(fout[f + m], tw[tw1]);
        let s2 = c_mul(fout[f + m2], tw[tw2]);

        let s3 = c_add(s1, s2);
        let mut s0 = c_sub(s1, s2);
        tw1 += fstride;
        tw2 += fstride * 2;

        let mut fm = MiniKissFftCpx {
            r: fout[f].r - mini_half_of(s3.r),
            i: fout[f].i - mini_half_of(s3.i),
        };

        // C_MULBYSCALAR(scratch[0], epi3.i)
        s0.r *= epi3.i;
        s0.i *= epi3.i;

        fout[f] = c_add(fout[f], s3);

        fout[f + m2] = MiniKissFftCpx {
            r: fm.r + s0.i,
            i: fm.i - s0.r,
        };

        fm.r -= s0.i;
        fm.i += s0.r;
        fout[f + m] = fm;
    }
}

/// Port of celt/mini_kfft.c:kf_bfly5.
fn kf_bfly5(fout: &mut [MiniKissFftCpx], fstride: usize, st: &MiniKissFftState, m: usize) {
    let tw = &st.twiddles[..];
    let ya = tw[fstride * m];
    let yb = tw[fstride * 2 * m];

    let (f1, f2, f3, f4) = (m, 2 * m, 3 * m, 4 * m);

    for u in 0..m {
        let s0 = fout[u];

        let s1 = c_mul(fout[f1 + u], tw[u * fstride]);
        let s2 = c_mul(fout[f2 + u], tw[2 * u * fstride]);
        let s3 = c_mul(fout[f3 + u], tw[3 * u * fstride]);
        let s4 = c_mul(fout[f4 + u], tw[4 * u * fstride]);

        let s7 = c_add(s1, s4);
        let s10 = c_sub(s1, s4);
        let s8 = c_add(s2, s3);
        let s9 = c_sub(s2, s3);

        fout[u].r += s7.r + s8.r;
        fout[u].i += s7.i + s8.i;

        let s5 = MiniKissFftCpx {
            r: s0.r + s7.r * ya.r + s8.r * yb.r,
            i: s0.i + s7.i * ya.r + s8.i * yb.r,
        };

        let s6 = MiniKissFftCpx {
            r: s10.i * ya.i + s9.i * yb.i,
            i: -(s10.r * ya.i) - s9.r * yb.i,
        };

        fout[f1 + u] = c_sub(s5, s6);
        fout[f4 + u] = c_add(s5, s6);

        let s11 = MiniKissFftCpx {
            r: s0.r + s7.r * yb.r + s8.r * ya.r,
            i: s0.i + s7.i * yb.r + s8.i * ya.r,
        };
        let s12 = MiniKissFftCpx {
            r: -(s10.i * yb.i) + s9.i * ya.i,
            i: s10.r * yb.i - s9.r * ya.i,
        };

        fout[f2 + u] = c_add(s11, s12);
        fout[f3 + u] = c_sub(s11, s12);
    }
}

/// Port of celt/mini_kfft.c:kf_work (recursive decimation-in-time driver).
///
/// `f` is the input with `fi` the C `f` pointer offset.
fn kf_work<S: CpxSrc + ?Sized>(
    fout: &mut [MiniKissFftCpx],
    f: &S,
    mut fi: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[i32],
    st: &MiniKissFftState,
) {
    let p = factors[0] as usize; // the radix
    let m = factors[1] as usize; // stage's fft length/p

    if m == 1 {
        for o in &mut fout[..p] {
            *o = f.get(fi);
            fi += fstride * in_stride;
        }
    } else {
        for k in 0..p {
            // recursive call: DFT of size m*p performed by doing p instances of smaller DFTs of
            // size m, each one takes a decimated version of the input
            kf_work(
                &mut fout[k * m..],
                f,
                fi,
                fstride * p,
                in_stride,
                &factors[2..],
                st,
            );
            fi += fstride * in_stride;
        }
    }

    // recombine the p smaller DFTs
    match p {
        2 => kf_bfly2(fout, fstride, st, m),
        3 => kf_bfly3(fout, fstride, st, m),
        4 => kf_bfly4(fout, fstride, st, m),
        5 => kf_bfly5(fout, fstride, st, m),
        // C: assert(0). Rejected by `mini_kiss_fft_alloc`.
        _ => unreachable!("mini_kfft: unsupported radix {p}"),
    }
}

/// Port of celt/mini_kfft.c:kf_factor.
///
/// `facbuf` is populated by `p1, m1, p2, m2, ...` where `p[i] * m[i] = m[i-1]`, `m0 = n`.
/// Returns `false` if a radix other than 2..=5 would be produced (C would `assert(0)` when
/// running the FFT) or the factor buffer would overflow.
fn kf_factor(mut n: i32, facbuf: &mut [i32; 2 * MINI_MAXFACTORS]) -> bool {
    let mut p: i32 = 4;
    let floor_sqrt: f64 = crate::math::floor(crate::math::sqrt(n as f64));
    let mut k = 0usize;

    // factor out powers of 4, powers of 2, then any remaining primes
    loop {
        while n % p != 0 {
            match p {
                4 => p = 2,
                2 => p = 3,
                _ => p += 2,
            }
            if p as f64 > floor_sqrt {
                p = n; // no more factors, skip to end
            }
        }
        n /= p;
        if k >= MINI_MAXFACTORS || !(2..=5).contains(&p) {
            return false;
        }
        facbuf[2 * k] = p;
        facbuf[2 * k + 1] = n;
        k += 1;
        if n <= 1 {
            break;
        }
    }
    true
}

/// Port of celt/mini_kfft.c:mini_kiss_fft_alloc.
///
/// # Errors
/// [`Error::BadArg`] if `nfft < 2` or `nfft` has a prime factor larger than 5 (C accepts these
/// but aborts on an assertion when transforming).
pub fn mini_kiss_fft_alloc(nfft: i32, inverse_fft: bool) -> Result<MiniKissFftState> {
    if nfft < 2 {
        return Err(Error::BadArg);
    }
    let mut twiddles = vec![MiniKissFftCpx::default(); nfft as usize];
    for (i, tw) in twiddles.iter_mut().enumerate() {
        const PI: f64 = 3.141592653589793238462643383279502884197169399375105820974944;
        let mut phase: f64 = -2.0 * PI * i as f64 / nfft as f64;
        if inverse_fft {
            phase *= -1.0;
        }
        *tw = mini_kf_cexp(phase);
    }
    let mut factors = [0i32; 2 * MINI_MAXFACTORS];
    if !kf_factor(nfft, &mut factors) {
        return Err(Error::BadArg);
    }
    Ok(MiniKissFftState {
        nfft,
        inverse: inverse_fft,
        factors,
        twiddles,
    })
}

/// Generic core of [`mini_kiss_fft_stride`].
fn mini_kiss_fft_stride_src<S: CpxSrc + ?Sized>(
    st: &MiniKissFftState,
    fin: &S,
    fout: &mut [MiniKissFftCpx],
    in_stride: usize,
) {
    kf_work(
        &mut fout[..st.nfft as usize],
        fin,
        0,
        1,
        in_stride,
        &st.factors,
        st,
    );
}

/// Port of celt/mini_kfft.c:mini_kiss_fft_stride: FFT of `fin[0], fin[in_stride], ...`
/// (`st.nfft` values) into `fout`. `fin != fout` is enforced by the borrow checker.
pub fn mini_kiss_fft_stride(
    st: &MiniKissFftState,
    fin: &[MiniKissFftCpx],
    fout: &mut [MiniKissFftCpx],
    in_stride: usize,
) {
    mini_kiss_fft_stride_src(st, fin, fout, in_stride);
}

/// Port of celt/mini_kfft.c:mini_kiss_fft.
pub fn mini_kiss_fft(cfg: &MiniKissFftState, fin: &[MiniKissFftCpx], fout: &mut [MiniKissFftCpx]) {
    mini_kiss_fft_stride(cfg, fin, fout, 1);
}

/// Port of celt/mini_kfft.c:mini_kiss_fftr_alloc: real FFT of (even) size `nfft`.
///
/// # Errors
/// [`Error::BadArg`] if `nfft` is odd (C `assert`) or `nfft / 2` is not supported by
/// [`mini_kiss_fft_alloc`].
pub fn mini_kiss_fftr_alloc(nfft: i32, inverse_fft: bool) -> Result<MiniKissFftrState> {
    if nfft & 1 != 0 {
        return Err(Error::BadArg);
    }
    let nfft = nfft >> 1;

    let substate = mini_kiss_fft_alloc(nfft, inverse_fft)?;
    let tmpbuf = vec![MiniKissFftCpx::default(); nfft as usize];
    let mut super_twiddles = vec![MiniKissFftCpx::default(); (nfft / 2) as usize];

    for (i, tw) in super_twiddles.iter_mut().enumerate() {
        let mut phase: f64 =
            -3.14159265358979323846264338327 * ((i + 1) as f64 / nfft as f64 + 0.5);
        if inverse_fft {
            phase *= -1.0;
        }
        *tw = mini_kf_cexp(phase);
    }
    Ok(MiniKissFftrState {
        substate,
        tmpbuf,
        super_twiddles,
    })
}

/// Port of celt/mini_kfft.c:mini_kiss_fftr: forward real FFT of `timedata` (`nfft` samples)
/// into `freqdata` (`nfft/2 + 1` bins).
pub fn mini_kiss_fftr(
    st: &mut MiniKissFftrState,
    timedata: &[MiniKissFftScalar],
    freqdata: &mut [MiniKissFftCpx],
) {
    // input buffer timedata is stored row-wise
    debug_assert!(!st.substate.inverse);

    let ncfft = st.substate.nfft as usize;

    // perform the parallel fft of two real signals packed in real,imag
    mini_kiss_fft_stride_src(&st.substate, &RealPairs(timedata), &mut st.tmpbuf, 1);
    // The real part of the DC element of the frequency spectrum in st->tmpbuf contains the sum
    // of the even-numbered elements of the input time sequence. The imag part is the sum of the
    // odd-numbered elements.
    //
    // The sum of tdc.r and tdc.i is the sum of the input time sequence, yielding DC of input
    // time sequence. The difference of tdc.r - tdc.i is the sum of the input (dot product)
    // [1,-1,1,-1..., yielding Nyquist bin of input time sequence.

    let tmpbuf = &st.tmpbuf[..ncfft];
    let tdc = tmpbuf[0];
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
    freqdata[ncfft].i = 0.0;
    freqdata[0].i = 0.0;

    for k in 1..=ncfft / 2 {
        let fpk = tmpbuf[k];
        let fpnk = MiniKissFftCpx {
            r: tmpbuf[ncfft - k].r,
            i: -tmpbuf[ncfft - k].i,
        };

        let f1k = c_add(fpk, fpnk);
        let f2k = c_sub(fpk, fpnk);
        let tw = c_mul(f2k, st.super_twiddles[k - 1]);

        freqdata[k].r = mini_half_of(f1k.r + tw.r);
        freqdata[k].i = mini_half_of(f1k.i + tw.i);
        freqdata[ncfft - k].r = mini_half_of(f1k.r - tw.r);
        freqdata[ncfft - k].i = mini_half_of(tw.i - f1k.i);
    }
}

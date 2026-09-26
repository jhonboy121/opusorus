//! Differential tests for unit `celt_fft`: kiss FFT (celt/kiss_fft.c), MDCT (celt/mdct.c) and,
//! with QEXT, the mini kiss FFT (celt/mini_kfft.c) vs the C oracle. All comparisons are
//! bit-exact.
//!
//! Shared by the float and the fixed-point builds (`fixed-point`, `fixed-res24`, with or
//! without `qext` / `custom-modes`): the data type is `kiss_fft_scalar` (`f32` / `i32`) and the
//! window/twiddle type `celt_coef` (`f32` / Q15 `i16` / Q31 `i32` with QEXT). The mini kiss
//! FFT is float-only in both builds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::celt::kiss_fft;
use opusorus::celt::mdct;
use opusorus::celt::static_modes::{
    self, CeltMode, KissFftCpx, KissFftScalar as S, KissFftState, KissTwiddleScalar as Coef,
};
use opusorus_conformance::{Rng, assert_slice_eq};
use opusorus_oracle::celt_fft::{self as c, FftKind};

/// Static modes available in this build: `(fs, mode)`.
fn static_modes() -> Vec<(i32, &'static CeltMode)> {
    #[allow(unused_mut, reason = "extended only with qext")]
    let mut v = vec![(48000, &static_modes::MODE48000_960_120)];
    #[cfg(feature = "qext")]
    v.push((96000, &static_modes::MODE96000_1920_240));
    v
}

fn to_cpx(x: &[S]) -> Vec<KissFftCpx> {
    x.as_chunks::<2>()
        .0
        .iter()
        .map(|p| KissFftCpx { r: p[0], i: p[1] })
        .collect()
}

fn from_cpx(x: &[KissFftCpx]) -> Vec<S> {
    x.iter().flat_map(|c| [c.r, c.i]).collect()
}

/// Bit-exact comparison of data (`kiss_fft_scalar`) buffers.
#[cfg(not(feature = "fixed-point"))]
#[track_caller]
fn assert_s_eq(what: &str, rust: &[S], c: &[S]) {
    opusorus_conformance::assert_bits_eq_f32(what, rust, c);
}
/// Bit-exact comparison of data (`kiss_fft_scalar`) buffers.
#[cfg(feature = "fixed-point")]
#[track_caller]
fn assert_s_eq(what: &str, rust: &[S], c: &[S]) {
    assert_slice_eq(what, rust, c);
}

/// Bit-exact comparison of `celt_coef` tables (twiddles, trig, scale).
#[cfg(not(feature = "fixed-point"))]
#[track_caller]
fn assert_coef_eq(what: &str, rust: &[Coef], c: &[Coef]) {
    opusorus_conformance::assert_bits_eq_f32(what, rust, c);
}
/// Bit-exact comparison of `celt_coef` tables (twiddles, trig, scale).
#[cfg(feature = "fixed-point")]
#[track_caller]
fn assert_coef_eq(what: &str, rust: &[Coef], c: &[Coef]) {
    assert_slice_eq(what, rust, c);
}

/// Random real buffer of length `n`; `case` selects the amplitude class / edge pattern.
#[cfg(not(feature = "fixed-point"))]
fn signal(rng: &mut Rng, n: usize, case: usize) -> Vec<S> {
    match case % 10 {
        0 => vec![0.0; n],
        1 => {
            let mut v = vec![0.0; n];
            v[rng.range_i32(0, n as i32 - 1) as usize] = 1.0;
            v
        }
        2 => vec![rng.f32_sym() * 1000.0; n],
        3 => (0..n).map(|_| rng.f32_sym() * 1e-30).collect(),
        4 => (0..n).map(|_| rng.f32_sym() * 1e30).collect(),
        5 => (0..n)
            .map(|i| if i % 2 == 0 { 32767.0 } else { -32768.0 })
            .collect(),
        6 => (0..n).map(|_| rng.f32_sym() * 32768.0).collect(),
        7 => {
            let f = rng.f32_sym() * 0.5;
            (0..n)
                .map(|i| (i as f32 * f * 3.0).sin() + 0.001 * rng.f32_sym())
                .collect()
        }
        _ => (0..n).map(|_| rng.f32_sym()).collect(),
    }
}

/// Random `i32` in `[-amp, amp]`.
#[cfg(feature = "fixed-point")]
const fn rand_amp(rng: &mut Rng, amp: i32) -> i32 {
    rng.range_i32(-amp, amp)
}

/// Random real buffer of length `n` in the fixed-point domain of the transforms (`celt_sig`
/// scale, at most `2^28` in magnitude so that the non-wrapping C additions of the forward MDCT
/// cannot overflow); `case` selects the amplitude class / edge pattern.
#[cfg(feature = "fixed-point")]
fn signal(rng: &mut Rng, n: usize, case: usize) -> Vec<S> {
    match case % 10 {
        0 => vec![0; n],
        1 => {
            let mut v = vec![0; n];
            v[rng.range_i32(0, n as i32 - 1) as usize] = 1 << 27;
            v
        }
        2 => vec![rand_amp(rng, 1 << 27); n],
        3 => (0..n).map(|_| rand_amp(rng, 8)).collect(),
        4 => (0..n).map(|_| rand_amp(rng, 1 << 28)).collect(),
        5 => (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    32767 << 12
                } else {
                    -32768 << 12
                }
            })
            .collect(),
        6 => (0..n).map(|_| rand_amp(rng, 32768)).collect(),
        7 => {
            let f = rng.f32_sym() * 0.5;
            (0..n)
                .map(|i| ((i as f32 * f * 3.0).sin() * (1 << 26) as f32) as i32 + rand_amp(rng, 64))
                .collect()
        }
        8 => (0..n).map(|_| rand_amp(rng, 1 << 20)).collect(),
        _ => (0..n).map(|_| rand_amp(rng, 1 << 27)).collect(),
    }
}

/// Like [`signal`], but every fourth case uses (almost) the full `i32` range (the backward MDCT
/// and the inverse FFT accept any value except `i32::MIN`, which C would negate).
#[cfg(feature = "fixed-point")]
fn signal_wide(rng: &mut Rng, n: usize, case: usize) -> Vec<S> {
    if case % 4 == 3 {
        (0..n).map(|_| rand_amp(rng, i32::MAX - 1)).collect()
    } else {
        signal(rng, n, case)
    }
}
/// Float build: [`signal`].
#[cfg(not(feature = "fixed-point"))]
fn signal_wide(rng: &mut Rng, n: usize, case: usize) -> Vec<S> {
    signal(rng, n, case)
}

/// The used prefix of a factor list (`p, m` pairs up to and including `m == 1`); C leaves the
/// rest of runtime-allocated factor buffers uninitialized.
#[cfg(any(feature = "custom-modes", feature = "qext"))]
fn used_factors<T: Copy + Into<i32>>(f: &[T]) -> Vec<i32> {
    let mut out = Vec::new();
    for pair in f.as_chunks::<2>().0 {
        out.push(pair[0].into());
        out.push(pair[1].into());
        if pair[1].into() == 1 {
            break;
        }
    }
    out
}

/// Asserts `a` and `b` differ by at most one ulp elementwise (only used against the static
/// tables, which C itself does not reproduce bit-exactly at run time).
#[cfg(all(feature = "custom-modes", not(feature = "fixed-point")))]
fn assert_static_close(what: &str, a: &[Coef], b: &[Coef]) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x.to_bits() as i32)
            .wrapping_sub(y.to_bits() as i32)
            .unsigned_abs();
        assert!(d <= 1, "{what}: {i}: {x} vs {y}");
    }
}
/// Fixed-point build: asserts `a` and `b` differ by at most 2 elementwise. The Q15 static
/// tables (`static_modes_fixed.h`) do not reproduce C's run-time `celt_cos_norm` twiddles and
/// trig values exactly either (up to 2 LSB off); the Q31 (QEXT) ones match them exactly.
#[cfg(all(feature = "custom-modes", feature = "fixed-point"))]
fn assert_static_close(what: &str, a: &[Coef], b: &[Coef]) {
    let tol = if cfg!(feature = "qext") { 0 } else { 2 };
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (i64::from(x) - i64::from(y)).abs();
        assert!(d <= tol, "{what}: {i}: {x} vs {y}");
    }
}

fn run_rust_fft(st: &KissFftState, kind: FftKind, fin: &[S]) -> Vec<S> {
    let fin = to_cpx(fin);
    let mut fout = vec![KissFftCpx::default(); fin.len()];
    match kind {
        FftKind::Fft => kiss_fft::opus_fft(st, &fin, &mut fout),
        FftKind::Ifft => kiss_fft::opus_ifft(st, &fin, &mut fout),
        FftKind::Impl(_downshift) => {
            fout.copy_from_slice(&fin);
            #[cfg(feature = "fixed-point")]
            kiss_fft::opus_fft_impl(st, &mut fout, _downshift);
            #[cfg(not(feature = "fixed-point"))]
            kiss_fft::opus_fft_impl(st, &mut fout);
        }
    }
    from_cpx(&fout)
}

/// The transforms run on an FFT state for test iteration `iter`: forward, inverse and
/// `opus_fft_impl` (fixed point: with a downshift varying with `iter`).
const fn fft_kinds(iter: usize) -> [FftKind; 3] {
    let downshift = if cfg!(feature = "fixed-point") {
        [0, 1, 2, 3, 5, 8, 13][iter % 7]
    } else {
        0
    };
    [FftKind::Fft, FftKind::Ifft, FftKind::Impl(downshift)]
}

/// Input for `kind` on an FFT of size `n`.
#[cfg(not(feature = "fixed-point"))]
fn fft_input(rng: &mut Rng, n: usize, iter: usize, _kind: FftKind) -> Vec<S> {
    signal(rng, 2 * n, iter)
}
/// Input for `kind` on an FFT of size `n` (fixed point). The forward FFT scales its input
/// down first; `opus_fft_impl` without downshift only uses wrapping arithmetic, so it gets any
/// value (exercising the `*_ovflw` wrap-around). The inverse FFT (whose final `-x` is undefined
/// for `i32::MIN` in C) and `opus_fft_impl` with a downshift (whose rounding `PSHR32` must not
/// overflow) get inputs that cannot wrap in the `n`-point transform.
#[cfg(feature = "fixed-point")]
fn fft_input(rng: &mut Rng, n: usize, iter: usize, kind: FftKind) -> Vec<S> {
    match kind {
        FftKind::Fft => signal(rng, 2 * n, iter),
        FftKind::Impl(0) => signal_wide(rng, 2 * n, iter),
        FftKind::Ifft | FftKind::Impl(_) => {
            let headroom = (usize::BITS - n.leading_zeros()) as i32 - 1;
            signal(rng, 2 * n, iter)
                .into_iter()
                .map(|x| x >> headroom)
                .collect()
        }
    }
}

#[test]
fn static_fft_states_match_c() {
    for (fs, mode) in static_modes() {
        for idx in 0..4 {
            let c = c::fft_static_info(fs, idx);
            let r = &*mode.mdct.kfft[idx];
            assert_eq!(r.nfft, c.nfft, "fs={fs} idx={idx}");
            assert_coef_eq(&format!("fs={fs} idx={idx} scale"), &[r.scale], &[c.scale]);
            #[cfg(feature = "fixed-point")]
            assert_eq!(r.scale_shift, c.scale_shift, "fs={fs} idx={idx}");
            assert_eq!(r.shift, c.shift, "fs={fs} idx={idx}");
            assert_eq!(r.factors, c.factors, "fs={fs} idx={idx}");
            assert_slice_eq("bitrev", &r.bitrev[..r.nfft as usize], &c.bitrev);
        }
    }
}

#[test]
fn fft_static_states() {
    let mut rng = Rng::new(0xF0F7_0001);
    for (fs, mode) in static_modes() {
        for idx in 0..4 {
            let st = &*mode.mdct.kfft[idx];
            let n = st.nfft as usize;
            for iter in 0..250 {
                for kind in fft_kinds(iter) {
                    let fin = fft_input(&mut rng, n, iter, kind);
                    let what = format!("fs={fs} idx={idx} iter={iter} {kind:?}");
                    let cout = c::fft_static(fs, idx, kind, &fin);
                    let rout = run_rust_fft(st, kind, &fin);
                    assert_s_eq(&what, &rout, &cout);
                }
            }
        }
    }
}

/// Random window of `overlap` values in `[0, 1]`.
#[cfg(not(feature = "fixed-point"))]
fn random_window(rng: &mut Rng, overlap: usize) -> Vec<Coef> {
    (0..overlap).map(|_| 0.5 + 0.5 * rng.f32_sym()).collect()
}
/// Random window of `overlap` values in `[0, COEF_ONE]`.
#[cfg(feature = "fixed-point")]
fn random_window(rng: &mut Rng, overlap: usize) -> Vec<Coef> {
    // COEF_ONE: Q31ONE with QEXT, Q15ONE otherwise.
    let one = if cfg!(feature = "qext") {
        i32::MAX
    } else {
        32767
    };
    (0..overlap)
        .map(|_| rng.range_i32(0, one) as Coef)
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "test helper mirrors the MDCT signature"
)]
fn run_rust_mdct(
    l: &static_modes::MdctLookup,
    dir: i32,
    input: &[S],
    out: &mut [S],
    window: &[Coef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    if dir == 0 {
        mdct::clt_mdct_forward(l, input, out, window, overlap, shift, stride);
    } else {
        mdct::clt_mdct_backward(l, input, out, window, overlap, shift, stride);
    }
}

/// Builds input/output buffers for an MDCT of size `n` (already shifted).
fn mdct_buffers(
    rng: &mut Rng,
    dir: i32,
    n: usize,
    overlap: usize,
    stride: usize,
    iter: usize,
) -> (Vec<S>, Vec<S>) {
    let n2 = n / 2;
    if dir == 0 {
        let input = signal(rng, n2 + overlap, iter);
        let out = signal(rng, stride * n2, iter + 3);
        (input, out)
    } else {
        let input = signal_wide(rng, stride * n2, iter);
        let out = signal_wide(rng, (n2 + overlap / 2).max(overlap) + 3, iter + 7);
        (input, out)
    }
}

#[test]
fn mdct_static_mode_window() {
    let mut rng = Rng::new(0xF0F7_0002);
    for (fs, mode) in static_modes() {
        let overlap = mode.overlap as usize;
        for dir in [0, 1] {
            for shift in 0..=mode.mdct.maxshift as usize {
                let n = mode.mdct.n as usize >> shift;
                for stride in [1usize, 2, 4, 8] {
                    for iter in 0..100 {
                        let (input, out0) = mdct_buffers(&mut rng, dir, n, overlap, stride, iter);
                        let mut cout = out0.clone();
                        c::mdct_static(fs, dir, &input, &mut cout, None, overlap, shift, stride);
                        let mut rout = out0;
                        run_rust_mdct(
                            &mode.mdct,
                            dir,
                            &input,
                            &mut rout,
                            &mode.window,
                            overlap,
                            shift,
                            stride,
                        );
                        let what =
                            format!("fs={fs} dir={dir} shift={shift} stride={stride} iter={iter}");
                        assert_s_eq(&what, &rout, &cout);
                    }
                }
            }
        }
    }
}

#[test]
fn mdct_static_random_windows() {
    let mut rng = Rng::new(0xF0F7_0003);
    for (fs, mode) in static_modes() {
        for dir in [0, 1] {
            for shift in 0..=mode.mdct.maxshift as usize {
                let n = mode.mdct.n as usize >> shift;
                let n2 = n / 2;
                for iter in 0..250 {
                    let overlap = match iter % 4 {
                        0 => 0,
                        1 => n2,
                        _ => 4 * rng.range_i32(1, (n2 / 4) as i32) as usize,
                    };
                    let stride = 1usize << rng.range_i32(0, 3);
                    let window = random_window(&mut rng, overlap);
                    let (input, out0) = mdct_buffers(&mut rng, dir, n, overlap, stride, iter);
                    let mut cout = out0.clone();
                    c::mdct_static(
                        fs,
                        dir,
                        &input,
                        &mut cout,
                        Some(&window),
                        overlap,
                        shift,
                        stride,
                    );
                    let mut rout = out0;
                    run_rust_mdct(
                        &mode.mdct, dir, &input, &mut rout, &window, overlap, shift, stride,
                    );
                    let what = format!(
                        "fs={fs} dir={dir} shift={shift} overlap={overlap} stride={stride} \
                         iter={iter}"
                    );
                    assert_s_eq(&what, &rout, &cout);
                }
            }
        }
    }
}

/// Fixed point: the MDCT round trip (forward then backward with the mode window, as the CELT
/// encoder and decoder use it) on realistic 16-bit PCM scaled to `celt_sig`, checking every
/// stage against C.
#[cfg(feature = "fixed-point")]
#[test]
fn mdct_static_round_trip_pcm() {
    use opusorus::celt::arch::SIG_SHIFT;
    let mut rng = Rng::new(0xF0F7_0008);
    for (fs, mode) in static_modes() {
        let overlap = mode.overlap as usize;
        for shift in 0..=mode.mdct.maxshift as usize {
            let n = mode.mdct.n as usize >> shift;
            let n2 = n / 2;
            for iter in 0..40 {
                let pcm = opusorus_conformance::signals::to_i16(
                    &opusorus_conformance::signals::music_like(
                        n2 + overlap,
                        1,
                        fs as u32,
                        0x5EED + iter as u64,
                    ),
                );
                let input: Vec<S> = pcm
                    .iter()
                    .map(|&x| i32::from(x) << SIG_SHIFT)
                    .map(|x| x + rng.range_i32(-8, 8))
                    .collect();
                let mut cfreq = vec![0; n2];
                c::mdct_static(fs, 0, &input, &mut cfreq, None, overlap, shift, 1);
                let mut rfreq = vec![0; n2];
                mdct::clt_mdct_forward(
                    &mode.mdct,
                    &input,
                    &mut rfreq,
                    &mode.window,
                    overlap,
                    shift,
                    1,
                );
                let what = format!("fs={fs} shift={shift} iter={iter}");
                assert_s_eq(&format!("{what} forward"), &rfreq, &cfreq);
                let out0 = signal(&mut rng, n2 + overlap, iter);
                let mut cout = out0.clone();
                c::mdct_static(fs, 1, &cfreq, &mut cout, None, overlap, shift, 1);
                let mut rout = out0;
                mdct::clt_mdct_backward(
                    &mode.mdct,
                    &rfreq,
                    &mut rout,
                    &mode.window,
                    overlap,
                    shift,
                    1,
                );
                assert_s_eq(&format!("{what} backward"), &rout, &cout);
            }
        }
    }
}

#[cfg(feature = "custom-modes")]
mod custom {
    use super::*;
    use opusorus::celt::static_modes::KissTwiddleCpx;

    fn tw_flat(tw: &[KissTwiddleCpx]) -> Vec<Coef> {
        tw.iter().flat_map(|c| [c.r, c.i]).collect()
    }

    /// Compares Rust `opus_fft_alloc_twiddles(nfft, base)` against C, including transforms.
    fn check_alloc(rng: &mut Rng, nfft: i32, base_nfft: i32) {
        let base = if base_nfft > 0 {
            Some(kiss_fft::opus_fft_alloc(base_nfft).expect("base alloc"))
        } else {
            None
        };
        let r = kiss_fft::opus_fft_alloc_twiddles(nfft, base.as_ref())
            .unwrap_or_else(|e| panic!("nfft={nfft} base={base_nfft}: {e:?}"));
        let (cinfo, _) = c::fft_alloc(nfft, base_nfft, None).unwrap();
        let what = format!("nfft={nfft} base={base_nfft}");
        assert_eq!(r.nfft, cinfo.nfft, "{what}");
        assert_coef_eq(&format!("{what} scale"), &[r.scale], &[cinfo.scale]);
        #[cfg(feature = "fixed-point")]
        assert_eq!(r.scale_shift, cinfo.scale_shift, "{what}");
        assert_eq!(r.shift, cinfo.shift, "{what}");
        assert_eq!(
            used_factors(&r.factors),
            used_factors(&cinfo.factors),
            "{what}"
        );
        assert_slice_eq(&what, &r.bitrev, &cinfo.bitrev);
        assert_coef_eq(&what, &tw_flat(&r.twiddles), &cinfo.twiddles);

        let n = nfft as usize;
        for iter in 0..20 {
            for kind in fft_kinds(iter) {
                let fin = fft_input(rng, n, iter, kind);
                let (_, cout) = c::fft_alloc(nfft, base_nfft, Some((kind, &fin))).expect("c ok");
                let rout = run_rust_fft(&r, kind, &fin);
                assert_s_eq(&format!("{what} iter={iter} {kind:?}"), &rout, &cout);
            }
        }
    }

    #[test]
    fn fft_alloc_matches_c() {
        let mut rng = Rng::new(0xF0F7_0004);
        let sizes = [
            1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50,
            60, 64, 72, 75, 80, 90, 96, 100, 120, 125, 128, 144, 150, 160, 180, 192, 200, 225, 240,
            250, 256, 300, 320, 360, 384, 400, 480, 500, 512, 600, 640, 720, 768, 800, 960, 1000,
            1024, 1920, 2048,
        ];
        for &n in &sizes {
            check_alloc(&mut rng, n, 0);
        }
        let pairs = [
            (480, 480),
            (240, 480),
            (120, 480),
            (60, 480),
            (200, 400),
            (100, 400),
            (50, 400),
            (25, 400),
            (100, 200),
            (30, 60),
            (15, 60),
            (15, 30),
            (256, 1024),
            (480, 1920),
            (6, 12),
            (3, 6),
        ];
        for &(n, b) in &pairs {
            check_alloc(&mut rng, n, b);
        }
    }

    /// Sizes C rejects (returns `NULL`). C's failure path calls `opus_fft_free` on a state whose
    /// `bitrev` (and possibly `twiddles`) pointer is uninitialized, which is undefined behaviour
    /// in the oracle, so these are only checked on the Rust side.
    #[test]
    fn fft_alloc_failures() {
        for n in [7, 11, 13, 14, 21, 28, 49, 121, 202, 1001] {
            assert_eq!(
                kiss_fft::opus_fft_alloc(n),
                Err(opusorus::Error::AllocFail),
                "n={n}"
            );
        }
        let base480 = kiss_fft::opus_fft_alloc(480).unwrap();
        for n in [100, 960, 7, 481, 1] {
            assert_eq!(
                kiss_fft::opus_fft_alloc_twiddles(n, Some(&base480)),
                Err(opusorus::Error::AllocFail),
                "n={n}"
            );
        }
        assert_eq!(kiss_fft::opus_fft_alloc(0), Err(opusorus::Error::BadArg));
        assert_eq!(kiss_fft::opus_fft_alloc(-4), Err(opusorus::Error::BadArg));
        // N4 = 28 = 4*7 (unsupported radix); N4 >> 3 = 6 is not 50 >> shift.
        assert_eq!(
            mdct::clt_mdct_init(112, 1).map(|_| ()),
            Err(opusorus::Error::AllocFail)
        );
        assert_eq!(
            mdct::clt_mdct_init(200, 3).map(|_| ()),
            Err(opusorus::Error::AllocFail)
        );
        assert_eq!(
            mdct::clt_mdct_init(240, 3).map(|_| ()),
            Err(opusorus::Error::AllocFail)
        );
        assert_eq!(
            mdct::clt_mdct_init(1920, 4).map(|_| ()),
            Err(opusorus::Error::BadArg)
        );
    }

    #[test]
    fn fft_alloc_matches_static_tables() {
        let base = kiss_fft::opus_fft_alloc(480).unwrap();
        let st0 = &static_modes::FFT_STATE48000_960_0;
        assert_eq!(base.factors, st0.factors);
        assert_coef_eq("scale480", &[base.scale], &[st0.scale]);
        #[cfg(feature = "fixed-point")]
        assert_eq!(base.scale_shift, st0.scale_shift);
        assert_eq!(base.shift, st0.shift);
        assert_slice_eq("bitrev480", &base.bitrev, &st0.bitrev);
        // Float: the static tables were generated offline; C's own run-time twiddles differ from
        // them by up to one ulp (checked here on the C side), while the port matches C's
        // run-time values bit-exactly. Fixed point: they are identical.
        let (c480, _) = c::fft_alloc(480, 0, None).unwrap();
        assert_coef_eq("twiddles480 vs C", &tw_flat(&base.twiddles), &c480.twiddles);
        assert_static_close(
            "C twiddles480 vs static",
            &c480.twiddles,
            &tw_flat(&st0.twiddles),
        );
        let others = [
            &static_modes::FFT_STATE48000_960_1,
            &static_modes::FFT_STATE48000_960_2,
            &static_modes::FFT_STATE48000_960_3,
        ];
        for st in others {
            let r = kiss_fft::opus_fft_alloc_twiddles(st.nfft, Some(&base)).unwrap();
            assert_eq!(r.factors, st.factors);
            assert_coef_eq("scale", &[r.scale], &[st.scale]);
            #[cfg(feature = "fixed-point")]
            assert_eq!(r.scale_shift, st.scale_shift);
            assert_eq!(r.shift, st.shift);
            assert_slice_eq("bitrev", &r.bitrev, &st.bitrev);
        }
        #[cfg(feature = "qext")]
        {
            let base = kiss_fft::opus_fft_alloc(960).unwrap();
            let st0 = &static_modes::FFT_STATE96000_1920_0;
            assert_eq!(base.factors, st0.factors);
            assert_slice_eq("bitrev960", &base.bitrev, &st0.bitrev);
            let (c960, _) = c::fft_alloc(960, 0, None).unwrap();
            assert_coef_eq("twiddles960 vs C", &tw_flat(&base.twiddles), &c960.twiddles);
            assert_static_close(
                "C twiddles960 vs static",
                &c960.twiddles,
                &tw_flat(&st0.twiddles),
            );
        }
    }

    #[test]
    fn mdct_init_matches_c_and_static() {
        let m = mdct::clt_mdct_init(1920, 3).unwrap();
        let s = &static_modes::MODE48000_960_120.mdct;
        assert_eq!(m.n, s.n);
        assert_eq!(m.maxshift, s.maxshift);
        // As for the FFT twiddles, C's run-time trig table is within one ulp of the static one
        // (float) or identical to it (fixed point).
        let c1920 = c::mdct_init(1920, 3, None).unwrap();
        assert_coef_eq("trig1920 vs C", &m.trig, &c1920.trig);
        assert_static_close("C trig1920 vs static", &c1920.trig, &s.trig);
        #[cfg(feature = "qext")]
        {
            let m = mdct::clt_mdct_init(3840, 3).unwrap();
            let s = &static_modes::MODE96000_1920_240.mdct;
            let c3840 = c::mdct_init(3840, 3, None).unwrap();
            assert_coef_eq("trig3840 vs C", &m.trig, &c3840.trig);
            assert_static_close("C trig3840 vs static", &c3840.trig, &s.trig);
        }

        let mut rng = Rng::new(0xF0F7_0005);
        let configs = [
            (1920, 3),
            (1920, 0),
            (960, 2),
            (800, 3),
            (480, 3),
            (400, 2),
            (240, 2),
            (240, 1),
            (160, 3),
            (120, 1),
            (120, 0),
            (96, 3),
            (80, 2),
            (2048, 3),
            (1000, 1),
            (512, 3),
            (200, 1),
        ];
        for &(n, maxshift) in &configs {
            let what = format!("n={n} maxshift={maxshift}");
            let r = mdct::clt_mdct_init(n, maxshift).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let ci = c::mdct_init(n, maxshift, None).unwrap();
            assert_coef_eq(&what, &r.trig, &ci.trig);
            for i in 0..=maxshift as usize {
                assert_eq!(r.kfft[i].nfft, ci.nffts[i], "{what}");
                assert_eq!(r.kfft[i].shift, ci.shifts[i], "{what}");
            }
            for dir in [0, 1] {
                for shift in 0..=maxshift as usize {
                    let ns = n as usize >> shift;
                    if !ns.is_multiple_of(4) {
                        continue;
                    }
                    let n2 = ns / 2;
                    for iter in 0..40 {
                        let overlap = match iter % 3 {
                            0 => 0,
                            1 => n2 / 4 * 4,
                            _ => 4 * rng.range_i32(0, (n2 / 4) as i32) as usize,
                        };
                        let stride = 1usize << rng.range_i32(0, 3);
                        let window = random_window(&mut rng, overlap);
                        let (input, out0) = mdct_buffers(&mut rng, dir, ns, overlap, stride, iter);
                        let mut cout = out0.clone();
                        c::mdct_init(
                            n,
                            maxshift,
                            Some(c::MdctRun {
                                dir,
                                input: &input,
                                out: &mut cout,
                                window: &window,
                                overlap,
                                shift,
                                stride,
                            }),
                        )
                        .unwrap();
                        let mut rout = out0;
                        run_rust_mdct(&r, dir, &input, &mut rout, &window, overlap, shift, stride);
                        let what = format!(
                            "{what} dir={dir} shift={shift} overlap={overlap} stride={stride} \
                             iter={iter}"
                        );
                        assert_s_eq(&what, &rout, &cout);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "qext")]
mod qext {
    use super::used_factors;
    use opusorus::celt::mini_kfft::{self, MiniKissFftCpx};
    use opusorus_conformance::{Rng, assert_bits_eq_f32};
    use opusorus_oracle::celt_fft as c;

    /// Random float buffer of length `n` (the mini kiss FFT is float-only in both builds);
    /// `case` selects the amplitude class / edge pattern.
    fn fsignal(rng: &mut Rng, n: usize, case: usize) -> Vec<f32> {
        match case % 10 {
            0 => vec![0.0; n],
            1 => {
                let mut v = vec![0.0; n];
                v[rng.range_i32(0, n as i32 - 1) as usize] = 1.0;
                v
            }
            2 => vec![rng.f32_sym() * 1000.0; n],
            3 => (0..n).map(|_| rng.f32_sym() * 1e-30).collect(),
            4 => (0..n).map(|_| rng.f32_sym() * 1e30).collect(),
            5 => (0..n)
                .map(|i| if i % 2 == 0 { 32767.0 } else { -32768.0 })
                .collect(),
            6 => (0..n).map(|_| rng.f32_sym() * 32768.0).collect(),
            7 => {
                let f = rng.f32_sym() * 0.5;
                (0..n)
                    .map(|i| (i as f32 * f * 3.0).sin() + 0.001 * rng.f32_sym())
                    .collect()
            }
            _ => (0..n).map(|_| rng.f32_sym()).collect(),
        }
    }

    fn to_mcpx(x: &[f32]) -> Vec<MiniKissFftCpx> {
        x.as_chunks::<2>()
            .0
            .iter()
            .map(|p| MiniKissFftCpx { r: p[0], i: p[1] })
            .collect()
    }

    fn from_mcpx(x: &[MiniKissFftCpx]) -> Vec<f32> {
        x.iter().flat_map(|c| [c.r, c.i]).collect()
    }

    #[test]
    fn mini_kfft_matches_c() {
        let mut rng = Rng::new(0xF0F7_0006);
        let sizes = [
            2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25, 27, 30, 32, 45, 48, 60, 64, 81, 100, 120,
            125, 128, 240, 256, 480, 960, 1000, 1024, 1920, 2048,
        ];
        for &n in &sizes {
            for inverse in [false, true] {
                let st = mini_kfft::mini_kiss_fft_alloc(n, inverse).unwrap();
                for iter in 0..12 {
                    let stride = 1 + iter % 3;
                    let len = 2 * ((n as usize - 1) * stride + 1);
                    let fin = fsignal(&mut rng, len, iter);
                    let co = c::mini_kfft(n, inverse, &fin, stride);
                    let what = format!("n={n} inverse={inverse} stride={stride} iter={iter}");
                    if iter == 0 {
                        assert_eq!(
                            used_factors(&st.factors),
                            used_factors(&co.factors),
                            "{what}"
                        );
                        let tw: Vec<f32> = st.twiddles.iter().flat_map(|c| [c.r, c.i]).collect();
                        assert_bits_eq_f32(&what, &tw, &co.twiddles);
                    }
                    let fin_c = to_mcpx(&fin);
                    let mut fout = vec![MiniKissFftCpx::default(); n as usize];
                    mini_kfft::mini_kiss_fft_stride(&st, &fin_c, &mut fout, stride);
                    assert_bits_eq_f32(&what, &from_mcpx(&fout), &co.fout);
                    if stride == 1 {
                        let mut fout2 = vec![MiniKissFftCpx::default(); n as usize];
                        mini_kfft::mini_kiss_fft(&st, &fin_c, &mut fout2);
                        assert_eq!(fout, fout2);
                    }
                }
            }
        }
        // Sizes with a radix > 5 (C would assert when transforming) are rejected.
        for n in [0, 1, 7, 14, 22, 49, 1001] {
            assert!(mini_kfft::mini_kiss_fft_alloc(n, false).is_err(), "n={n}");
        }
    }

    #[test]
    fn mini_kfftr_matches_c() {
        let mut rng = Rng::new(0xF0F7_0007);
        let sizes = [
            4, 6, 8, 10, 12, 16, 20, 24, 30, 32, 40, 50, 64, 96, 120, 128, 240, 256, 480, 512, 960,
            1024, 1920, 2048, 4096,
        ];
        for &n in &sizes {
            let mut st = mini_kfft::mini_kiss_fftr_alloc(n, false).unwrap();
            for iter in 0..20 {
                let x = fsignal(&mut rng, n as usize, iter);
                let (cf, ctw) = c::mini_kfftr(n, &x);
                let what = format!("n={n} iter={iter}");
                if iter == 0 {
                    let tw: Vec<f32> = st.super_twiddles.iter().flat_map(|c| [c.r, c.i]).collect();
                    assert_bits_eq_f32(&what, &tw, &ctw);
                }
                let mut freq = vec![MiniKissFftCpx::default(); n as usize / 2 + 1];
                mini_kfft::mini_kiss_fftr(&mut st, &x, &mut freq);
                assert_bits_eq_f32(&what, &from_mcpx(&freq), &cf);
            }
        }
        assert!(mini_kfft::mini_kiss_fftr_alloc(10, true).is_ok());
        assert!(mini_kfft::mini_kiss_fftr_alloc(11, false).is_err());
        assert!(mini_kfft::mini_kiss_fftr_alloc(28, false).is_err());
    }
}

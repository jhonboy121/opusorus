//! Differential tests for unit `celt_fft`: kiss FFT (celt/kiss_fft.c), MDCT (celt/mdct.c) and,
//! with QEXT, the mini kiss FFT (celt/mini_kfft.c) vs the C oracle. All comparisons are
//! bit-exact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::celt::kiss_fft;
use opusorus::celt::mdct;
use opusorus::celt::static_modes::{self, CeltMode, KissFftCpx, KissFftState};
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq};
use opusorus_oracle::celt_fft::{self as c, FftKind};

/// Static modes available in this build: `(fs, mode)`.
fn static_modes() -> Vec<(i32, &'static CeltMode)> {
    #[allow(unused_mut, reason = "extended only with qext")]
    let mut v = vec![(48000, &static_modes::MODE48000_960_120)];
    #[cfg(feature = "qext")]
    v.push((96000, &static_modes::MODE96000_1920_240));
    v
}

fn to_cpx(x: &[f32]) -> Vec<KissFftCpx> {
    x.as_chunks::<2>()
        .0
        .iter()
        .map(|p| KissFftCpx { r: p[0], i: p[1] })
        .collect()
}

fn from_cpx(x: &[KissFftCpx]) -> Vec<f32> {
    x.iter().flat_map(|c| [c.r, c.i]).collect()
}

/// Random real buffer of length `n`; `case` selects the amplitude class / edge pattern.
fn signal(rng: &mut Rng, n: usize, case: usize) -> Vec<f32> {
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
#[cfg(feature = "custom-modes")]
fn assert_within_1ulp(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x.to_bits() as i32)
            .wrapping_sub(y.to_bits() as i32)
            .unsigned_abs();
        assert!(d <= 1, "{what}: {i}: {x} vs {y}");
    }
}

fn run_rust_fft(st: &KissFftState, kind: FftKind, fin: &[f32]) -> Vec<f32> {
    let fin = to_cpx(fin);
    let mut fout = vec![KissFftCpx::default(); fin.len()];
    match kind {
        FftKind::Fft => kiss_fft::opus_fft(st, &fin, &mut fout),
        FftKind::Ifft => kiss_fft::opus_ifft(st, &fin, &mut fout),
        FftKind::Impl => {
            fout.copy_from_slice(&fin);
            kiss_fft::opus_fft_impl(st, &mut fout);
        }
    }
    from_cpx(&fout)
}

#[test]
fn static_fft_states_match_c() {
    for (fs, mode) in static_modes() {
        for idx in 0..4 {
            let c = c::fft_static_info(fs, idx);
            let r = &*mode.mdct.kfft[idx];
            assert_eq!(r.nfft, c.nfft, "fs={fs} idx={idx}");
            assert_eq!(r.scale.to_bits(), c.scale.to_bits(), "fs={fs} idx={idx}");
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
                let fin = signal(&mut rng, 2 * n, iter);
                for kind in [FftKind::Fft, FftKind::Ifft, FftKind::Impl] {
                    let what = format!("fs={fs} idx={idx} iter={iter} {kind:?}");
                    let cout = c::fft_static(fs, idx, kind, &fin);
                    let rout = run_rust_fft(st, kind, &fin);
                    assert_bits_eq_f32(&what, &rout, &cout);
                }
            }
        }
    }
}

/// Random window of `overlap` values in `[0, 1]` (or the mode window).
fn random_window(rng: &mut Rng, overlap: usize) -> Vec<f32> {
    (0..overlap).map(|_| 0.5 + 0.5 * rng.f32_sym()).collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "test helper mirrors the MDCT signature"
)]
fn run_rust_mdct(
    l: &static_modes::MdctLookup,
    dir: i32,
    input: &[f32],
    out: &mut [f32],
    window: &[f32],
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
) -> (Vec<f32>, Vec<f32>) {
    let n2 = n / 2;
    if dir == 0 {
        let input = signal(rng, n2 + overlap, iter);
        let out = signal(rng, stride * n2, iter + 3);
        (input, out)
    } else {
        let input = signal(rng, stride * n2, iter);
        let out = signal(rng, (n2 + overlap / 2).max(overlap) + 3, iter + 7);
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
                        assert_bits_eq_f32(&what, &rout, &cout);
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
                    assert_bits_eq_f32(&what, &rout, &cout);
                }
            }
        }
    }
}

#[cfg(feature = "custom-modes")]
mod custom {
    use super::*;
    use opusorus::celt::static_modes::KissTwiddleCpx;

    fn tw_bits(tw: &[KissTwiddleCpx]) -> Vec<f32> {
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
        assert_eq!(r.scale.to_bits(), cinfo.scale.to_bits(), "{what}");
        assert_eq!(r.shift, cinfo.shift, "{what}");
        assert_eq!(
            used_factors(&r.factors),
            used_factors(&cinfo.factors),
            "{what}"
        );
        assert_slice_eq(&what, &r.bitrev, &cinfo.bitrev);
        assert_bits_eq_f32(&what, &tw_bits(&r.twiddles), &cinfo.twiddles);

        let n = nfft as usize;
        for iter in 0..20 {
            let fin = signal(rng, 2 * n, iter);
            for kind in [FftKind::Fft, FftKind::Ifft, FftKind::Impl] {
                let (_, cout) = c::fft_alloc(nfft, base_nfft, Some((kind, &fin))).expect("c ok");
                let rout = run_rust_fft(&r, kind, &fin);
                assert_bits_eq_f32(&format!("{what} iter={iter} {kind:?}"), &rout, &cout);
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
        assert_eq!(base.scale.to_bits(), st0.scale.to_bits());
        assert_eq!(base.shift, st0.shift);
        assert_slice_eq("bitrev480", &base.bitrev, &st0.bitrev);
        // The static tables were generated offline; C's own run-time twiddles differ from them
        // by up to one ulp (checked here on the C side), while the port matches C's run-time
        // values bit-exactly.
        let (c480, _) = c::fft_alloc(480, 0, None).unwrap();
        assert_bits_eq_f32("twiddles480 vs C", &tw_bits(&base.twiddles), &c480.twiddles);
        assert_within_1ulp(
            "C twiddles480 vs static",
            &c480.twiddles,
            &tw_bits(&st0.twiddles),
        );
        let others = [
            &static_modes::FFT_STATE48000_960_1,
            &static_modes::FFT_STATE48000_960_2,
            &static_modes::FFT_STATE48000_960_3,
        ];
        for st in others {
            let r = kiss_fft::opus_fft_alloc_twiddles(st.nfft, Some(&base)).unwrap();
            assert_eq!(r.factors, st.factors);
            assert_eq!(r.scale.to_bits(), st.scale.to_bits());
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
            assert_bits_eq_f32("twiddles960 vs C", &tw_bits(&base.twiddles), &c960.twiddles);
            assert_within_1ulp(
                "C twiddles960 vs static",
                &c960.twiddles,
                &tw_bits(&st0.twiddles),
            );
        }
    }

    #[test]
    fn mdct_init_matches_c_and_static() {
        let m = mdct::clt_mdct_init(1920, 3).unwrap();
        let s = &static_modes::MODE48000_960_120.mdct;
        assert_eq!(m.n, s.n);
        assert_eq!(m.maxshift, s.maxshift);
        // As for the FFT twiddles, C's run-time trig table is within one ulp of the static one.
        let c1920 = c::mdct_init(1920, 3, None).unwrap();
        assert_bits_eq_f32("trig1920 vs C", &m.trig, &c1920.trig);
        assert_within_1ulp("C trig1920 vs static", &c1920.trig, &s.trig);
        #[cfg(feature = "qext")]
        {
            let m = mdct::clt_mdct_init(3840, 3).unwrap();
            let s = &static_modes::MODE96000_1920_240.mdct;
            let c3840 = c::mdct_init(3840, 3, None).unwrap();
            assert_bits_eq_f32("trig3840 vs C", &m.trig, &c3840.trig);
            assert_within_1ulp("C trig3840 vs static", &c3840.trig, &s.trig);
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
            assert_bits_eq_f32(&what, &r.trig, &ci.trig);
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
                        assert_bits_eq_f32(&what, &rout, &cout);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "qext")]
mod qext {
    use super::*;
    use opusorus::celt::mini_kfft;

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
                    let fin = signal(&mut rng, len, iter);
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
                    let fin_c = to_cpx(&fin);
                    let mut fout = vec![KissFftCpx::default(); n as usize];
                    mini_kfft::mini_kiss_fft_stride(&st, &fin_c, &mut fout, stride);
                    assert_bits_eq_f32(&what, &from_cpx(&fout), &co.fout);
                    if stride == 1 {
                        let mut fout2 = vec![KissFftCpx::default(); n as usize];
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
                let x = signal(&mut rng, n as usize, iter);
                let (cf, ctw) = c::mini_kfftr(n, &x);
                let what = format!("n={n} iter={iter}");
                if iter == 0 {
                    let tw: Vec<f32> = st.super_twiddles.iter().flat_map(|c| [c.r, c.i]).collect();
                    assert_bits_eq_f32(&what, &tw, &ctw);
                }
                let mut freq = vec![KissFftCpx::default(); n as usize / 2 + 1];
                mini_kfft::mini_kiss_fftr(&mut st, &x, &mut freq);
                assert_bits_eq_f32(&what, &from_cpx(&freq), &cf);
            }
        }
        assert!(mini_kfft::mini_kiss_fftr_alloc(10, true).is_ok());
        assert!(mini_kfft::mini_kiss_fftr_alloc(11, false).is_err());
        assert!(mini_kfft::mini_kiss_fftr_alloc(28, false).is_err());
    }
}

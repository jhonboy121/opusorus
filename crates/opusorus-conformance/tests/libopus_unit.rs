//! Ports of the libopus unit tests that need codec internals (the `internals` feature):
//!
//! * `celt/tests/test_unit_{types,cwrs32,dft,entropy,laplace,mathops,mdct,rotation}.c` and
//!   `test_unit_mini_kfft.c` (`qext`);
//! * `test_simple_matrix` of `tests/test_opus_projection.c` (the mapping-matrix routines);
//! * `tests/test_opus_custom.c` (`custom-modes`; the Opus Custom API is internal in opusorus).
//!
//! These are self-checks of opusorus (no C oracle involved). Tests that call the C library
//! `rand()` use glibc's generator (`GlibcRand`), so they draw the same values as the C
//! programs on glibc. Both builds are ported: in fixed-point builds (`fixed-point`,
//! `fixed-res24`) the tests use the integer `kiss_fft_scalar` / `celt_norm` / `opus_res` types
//! and test_unit_mathops.c runs its `FIXED_POINT` tests (`testilog2`, fixed `testlog2`/`testexp2`
//! /`testdiv`/`testsqrt`, `testrsqrt`, `testsqrt32`, `test_cos_norm32`, `test_rcp_norm32`, and
//! with `qext` `testlog2_db`, `testexp2_db`, `testatan_norm`, `testatan2p_norm`) instead of the
//! float ones, as the C programs do; `test_simple_matrix` skips the float matrix variants
//! (`!defined(FIXED_POINT)` in C).
//!
//! Adaptations:
//! * `cwrsi`/`icwrs` are private to `celt/cwrs.rs`; the cwrs32 test reaches them through
//!   `decode_pulses`/`encode_pulses` with the combination index range-coded as
//!   `ec_enc_uint(i, CELT_PVQ_V(n,k))`, which is exactly how those wrappers call them.
//! * test_unit_entropy prints (but does not fail on) encoder/decoder `tell` mismatches; the
//!   port asserts them.
//! * test_unit_mathops runs the SIMD and the C implementation of `celt_float2int16` and
//!   `opus_limit2_checkwithin1`; opusorus has only the C implementation, checked once per
//!   `use_ref_impl` value.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "numeric literals are copied verbatim from the C tests"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index loops mirror the C tests across several arrays"
)]

use std::f64::consts::PI as M_PI;

use opusorus::celt::arch::{CeltCoef, CeltNorm, OpusRes};
use opusorus::celt::arch::{int16tores, res2int16};
use opusorus::celt::bands::{SPREAD_NORMAL, bitexact_cos, bitexact_log2tan};
use opusorus::celt::cwrs::{celt_pvq_v, decode_pulses, encode_pulses};
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::kiss_fft::{opus_fft, opus_ifft};
use opusorus::celt::laplace::{ec_laplace_decode, ec_laplace_encode};
use opusorus::celt::mathops::{celt_float2int16, float2int16, opus_limit2_checkwithin1};
use opusorus::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
#[cfg(not(feature = "custom-modes"))]
use opusorus::celt::modes::opus_custom_mode_create;
use opusorus::celt::rate::get_pulses;
use opusorus::celt::static_modes::{KissFftCpx, KissFftScalar};
use opusorus::celt::vq::exp_rotation;
use opusorus::mapping_matrix::{
    MappingMatrix, mapping_matrix_get_size, mapping_matrix_multiply_channel_in_short,
    mapping_matrix_multiply_channel_out_short,
};
#[cfg(not(feature = "fixed-point"))]
use opusorus::mapping_matrix::{
    mapping_matrix_multiply_channel_in_float, mapping_matrix_multiply_channel_out_float,
};

/// An `int` stored in a `kiss_fft_scalar` (float, or `opus_int32` in fixed-point builds).
#[cfg(not(feature = "fixed-point"))]
const fn scalar(v: i32) -> KissFftScalar {
    v as KissFftScalar
}
/// An `int` stored in a `kiss_fft_scalar` (float, or `opus_int32` in fixed-point builds).
#[cfg(feature = "fixed-point")]
const fn scalar(v: i32) -> KissFftScalar {
    v
}

/// An `int` stored in a `celt_norm` (float, or `opus_int32` in fixed-point builds).
#[cfg(not(feature = "fixed-point"))]
const fn norm(v: i32) -> CeltNorm {
    v as CeltNorm
}
/// An `int` stored in a `celt_norm` (float, or `opus_int32` in fixed-point builds).
#[cfg(feature = "fixed-point")]
const fn norm(v: i32) -> CeltNorm {
    v
}

/// glibc's `rand()` (`TYPE_3` additive feedback generator); `new(1)` is an unseeded `rand()`.
#[derive(Debug, Clone)]
struct GlibcRand {
    r: [u32; 34],
    pos: usize,
}

impl GlibcRand {
    const RAND_MAX: u32 = 2_147_483_647;

    fn new(seed: u32) -> Self {
        let mut r = [0u32; 34];
        r[0] = if seed == 0 { 1 } else { seed };
        for i in 1..31 {
            let prev = r[i - 1] as i32;
            let hi = prev / 127_773;
            let lo = prev % 127_773;
            let mut word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            r[i] = word as u32;
        }
        for i in 31..34 {
            r[i] = r[i - 31];
        }
        let mut g = Self { r, pos: 0 };
        for _ in 34..344 {
            g.step();
        }
        g
    }

    const fn step(&mut self) -> u32 {
        let v = self.r[(self.pos + 3) % 34].wrapping_add(self.r[(self.pos + 31) % 34]);
        self.r[self.pos] = v;
        self.pos = (self.pos + 1) % 34;
        v
    }

    /// `rand()`.
    const fn next(&mut self) -> i32 {
        (self.step() >> 1) as i32
    }

    /// `rand()` as the unsigned value C gets in mixed `unsigned` arithmetic.
    const fn next_u(&mut self) -> u32 {
        self.next() as u32
    }
}

/// The `SEED` environment variable (as the C programs accept), or a fixed default.
fn seed() -> u32 {
    match std::env::var("SEED") {
        Ok(s) => match s.trim().parse() {
            Ok(v) => v,
            Err(e) => panic!("invalid SEED {s:?}: {e}"),
        },
        Err(_) => 0x5EED_0105,
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_types.c
// ---------------------------------------------------------------------------------------------

#[test]
fn unit_types() {
    let mut i: i16 = 1;
    i <<= 14;
    assert_eq!(i >> 14, 1, "opus_int16 isn't 16 bits");
    assert_eq!(size_of::<i16>() * 2, size_of::<i32>(), "16*2 != 32");
}

// ---------------------------------------------------------------------------------------------
// test_unit_cwrs32.c
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "custom-modes")]
const PN: &[i32] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44,
    48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208,
];
#[cfg(feature = "custom-modes")]
const PKMAX: &[i32] = &[
    128, 128, 128, 128, 88, 52, 36, 26, 22, 18, 16, 15, 13, 12, 12, 11, 10, 9, 9, 8, 8, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
];
#[cfg(not(feature = "custom-modes"))]
const PN: &[i32] = &[
    2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 36, 44, 48, 64, 72, 88, 96, 144, 176,
];
#[cfg(not(feature = "custom-modes"))]
const PKMAX: &[i32] = &[
    128, 128, 128, 88, 36, 26, 18, 16, 12, 11, 9, 9, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4,
];

/// `cwrsi(n,k,i,y)`: the pulse vector of combination index `i`.
fn cwrsi(n: i32, k: i32, i: u32, nc: u32, y: &mut [i32]) {
    let mut buf = [0u8; 16];
    let mut enc = EcEnc::new(&mut buf);
    enc.enc_uint(i, nc);
    enc.done();
    assert_eq!(enc.get_error(), 0);
    let mut dec = EcDec::new(&buf);
    decode_pulses(y, n, k, &mut dec);
}

/// `icwrs(n,y)`: the combination index of pulse vector `y`.
fn icwrs(n: i32, k: i32, y: &[i32], nc: u32) -> u32 {
    let mut buf = [0u8; 16];
    let mut enc = EcEnc::new(&mut buf);
    encode_pulses(y, n, k, &mut enc);
    enc.done();
    assert_eq!(enc.get_error(), 0);
    let mut dec = EcDec::new(&buf);
    dec.dec_uint(nc)
}

#[test]
fn unit_cwrs32() {
    const NMAX: usize = 240;
    for (&n, &kmax) in PN.iter().zip(PKMAX) {
        for pseudo in 1..41 {
            let k = get_pulses(pseudo);
            if k > kmax {
                break;
            }
            // Testing CWRS with N=n, K=k
            let nc = celt_pvq_v(n, k);
            let inc = (nc / 20000).max(1);
            let mut i = 0u32;
            while i < nc {
                let mut y = [0i32; NMAX];
                cwrsi(n, k, i, nc, &mut y[..n as usize]);
                let sy: i32 = y[..n as usize].iter().map(|v| v.abs()).sum();
                assert_eq!(sy, k, "N={n} Pulse count mismatch in cwrsi ({sy}!={k})");
                let ii = icwrs(n, k, &y[..n as usize], nc);
                let v = celt_pvq_v(n, k);
                assert_eq!(ii, i, "Combination-index mismatch ({ii}!={i})");
                assert_eq!(v, nc, "Combination count mismatch ({v}!={nc})");
                i = match i.checked_add(inc) {
                    Some(v) => v,
                    None => break,
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_dft.c
// ---------------------------------------------------------------------------------------------

/// Complex values as `(re, im)` doubles (C: `in[k].r * re` promotes to double).
fn pairs(v: &[KissFftCpx]) -> Vec<(f64, f64)> {
    v.iter().map(|c| (f64::from(c.r), f64::from(c.i))).collect()
}

/// `check` of test_unit_dft.c / test_unit_mini_kfft.c: SNR of `out` vs a double-precision DFT
/// of `input` (scaled by `1/nfft` for the forward transform when `scale_forward`).
fn check_dft(input: &[(f64, f64)], out: &[(f64, f64)], isinverse: bool, scale_forward: bool) {
    let nfft = input.len();
    let mut errpow = 0f64;
    let mut sigpow = 0f64;
    for bin in 0..nfft {
        let mut ansr = 0f64;
        let mut ansi = 0f64;
        for (k, x) in input.iter().enumerate() {
            let phase = -2.0 * M_PI * bin as f64 * k as f64 / nfft as f64;
            let mut re = phase.cos();
            let mut im = phase.sin();
            if isinverse {
                im = -im;
            }
            if !isinverse && scale_forward {
                re /= nfft as f64;
                im /= nfft as f64;
            }
            ansr += x.0 * re - x.1 * im;
            ansi += x.0 * im + x.1 * re;
        }
        let difr = ansr - out[bin].0;
        let difi = ansi - out[bin].1;
        errpow += difr * difr + difi * difi;
        sigpow += ansr * ansr + ansi * ansi;
    }
    let snr = 10.0 * (sigpow / errpow).log10();
    assert!(
        snr >= 60.0,
        "nfft={nfft} inverse={isinverse}: poor snr: {snr}"
    );
}

/// The random input of test1d: `(rand() % 32767) - 16384`, times 32768 (and `/nfft` for the
/// inverse), in `kiss_fft_scalar` arithmetic (integer division in fixed-point builds).
fn random_cpx(nfft: usize, isinverse: bool, rand: &mut GlibcRand) -> Vec<KissFftCpx> {
    let mut input = vec![KissFftCpx::default(); nfft];
    for x in &mut input {
        x.r = scalar((rand.next() % 32767) - 16384);
        x.i = scalar((rand.next() % 32767) - 16384);
    }
    for x in &mut input {
        x.r *= scalar(32768);
        x.i *= scalar(32768);
    }
    if isinverse {
        for x in &mut input {
            x.r /= scalar(nfft as i32);
            x.i /= scalar(nfft as i32);
        }
    }
    input
}

/// `test1d` of test_unit_dft.c.
fn dft_test1d(nfft: usize, isinverse: bool, rand: &mut GlibcRand) {
    #[cfg(feature = "custom-modes")]
    let owned = opusorus::celt::kiss_fft::opus_fft_alloc(nfft as i32).unwrap();
    #[cfg(feature = "custom-modes")]
    let cfg = &owned;
    #[cfg(not(feature = "custom-modes"))]
    let cfg = {
        let mode = opus_custom_mode_create(48000, 960).unwrap();
        let id = match nfft {
            480 => 0,
            240 => 1,
            120 => 2,
            60 => 3,
            _ => return,
        };
        &*mode.mdct.kfft[id]
    };

    let input = random_cpx(nfft, isinverse, rand);
    let mut out = vec![KissFftCpx::default(); nfft];
    if isinverse {
        opus_ifft(cfg, &input, &mut out);
    } else {
        opus_fft(cfg, &input, &mut out);
    }
    check_dft(&pairs(&input), &pairs(&out), isinverse, true);
}

#[test]
fn unit_dft() {
    let mut rand = GlibcRand::new(1);
    for nfft in [32, 128, 256, 36, 50, 60, 120, 240, 480] {
        dft_test1d(nfft, false, &mut rand);
        dft_test1d(nfft, true, &mut rand);
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_mini_kfft.c (QEXT)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "qext")]
#[test]
fn unit_mini_kfft() {
    use opusorus::celt::mini_kfft::{MiniKissFftCpx, mini_kiss_fft, mini_kiss_fft_alloc};
    let mut rand = GlibcRand::new(1);
    for nfft in [32, 128, 256, 36, 50, 60, 120, 240, 480] {
        for isinverse in [false, true] {
            let fft = mini_kiss_fft_alloc(nfft as i32, false).unwrap();
            let ifft = mini_kiss_fft_alloc(nfft as i32, true).unwrap();
            // The mini FFT is float code in every build.
            let mut min = vec![MiniKissFftCpx::default(); nfft];
            for x in &mut min {
                x.r = ((rand.next() % 32767) - 16384) as f32;
                x.i = ((rand.next() % 32767) - 16384) as f32;
            }
            for x in &mut min {
                x.r *= 32768.0;
                x.i *= 32768.0;
            }
            if isinverse {
                for x in &mut min {
                    x.r /= nfft as f32;
                    x.i /= nfft as f32;
                }
            }
            let mut mout = vec![MiniKissFftCpx::default(); nfft];
            mini_kiss_fft(if isinverse { &ifft } else { &fft }, &min, &mut mout);
            let p = |v: &[MiniKissFftCpx]| -> Vec<(f64, f64)> {
                v.iter().map(|c| (f64::from(c.r), f64::from(c.i))).collect()
            };
            // The mini FFT does not scale the forward transform (`if (0&&isinverse)` in C).
            check_dft(&p(&min), &p(&mout), isinverse, false);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_mdct.c
// ---------------------------------------------------------------------------------------------

/// `check` of test_unit_mdct.c (forward).
fn check_mdct(input: &[KissFftScalar], out: &[KissFftScalar]) {
    let nfft = input.len();
    let mut errpow = 0f64;
    let mut sigpow = 0f64;
    for bin in 0..nfft / 2 {
        let mut ansr = 0f64;
        for (k, &x) in input.iter().enumerate() {
            let phase = 2.0 * M_PI * (k as f64 + 0.5 + 0.25 * nfft as f64) * (bin as f64 + 0.5)
                / nfft as f64;
            let mut re = phase.cos();
            re /= (nfft / 4) as f64;
            ansr += f64::from(x) * re;
        }
        let difr = ansr - f64::from(out[bin]);
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }
    let snr = 10.0 * (sigpow / errpow).log10();
    assert!(snr >= 60.0, "nfft={nfft} inverse=0: poor snr: {snr}");
}

/// `check_inv` of test_unit_mdct.c.
fn check_mdct_inv(input: &[KissFftScalar], out: &[KissFftScalar]) {
    let nfft = input.len();
    let mut errpow = 0f64;
    let mut sigpow = 0f64;
    for bin in 0..nfft {
        let mut ansr = 0f64;
        for (k, &x) in input[..nfft / 2].iter().enumerate() {
            let phase = 2.0 * M_PI * (bin as f64 + 0.5 + 0.25 * nfft as f64) * (k as f64 + 0.5)
                / nfft as f64;
            ansr += f64::from(x) * phase.cos();
        }
        let difr = ansr - f64::from(out[bin]);
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }
    let snr = 10.0 * (sigpow / errpow).log10();
    assert!(snr >= 60.0, "nfft={nfft} inverse=1: poor snr: {snr}");
}

/// `test1d` of test_unit_mdct.c.
fn mdct_test1d(nfft: usize, isinverse: bool, rand: &mut GlibcRand) {
    #[cfg(feature = "custom-modes")]
    let owned = opusorus::celt::mdct::clt_mdct_init(nfft as i32, 0).unwrap();
    #[cfg(feature = "custom-modes")]
    let (cfg, shift) = (&owned, 0);
    #[cfg(not(feature = "custom-modes"))]
    let (cfg, shift) = {
        let mode = opus_custom_mode_create(48000, 960).unwrap();
        let shift = match nfft {
            1920 => 0,
            960 => 1,
            480 => 2,
            240 => 3,
            _ => return,
        };
        (&mode.mdct, shift)
    };

    let mut input: Vec<KissFftScalar> = (0..nfft)
        .map(|_| scalar((rand.next() % 32768) - 16384))
        .collect();
    // Q15ONE (Q31ONE with QEXT): 1.0 in the float build.
    #[cfg(feature = "qext")]
    let one: CeltCoef = opusorus::celt::arch::Q31ONE;
    #[cfg(not(feature = "qext"))]
    let one: CeltCoef = opusorus::celt::arch::Q15ONE;
    let window = vec![one; nfft / 2];
    for x in &mut input {
        *x *= scalar(32768);
    }
    if isinverse {
        for x in &mut input {
            *x /= scalar(nfft as i32);
        }
    }
    let in_copy = input.clone();
    let mut out = vec![scalar(0); nfft];

    if isinverse {
        clt_mdct_backward(cfg, &input, &mut out, &window, nfft / 2, shift, 1);
        // apply TDAC because clt_mdct_backward() no longer does that
        for k in 0..nfft / 4 {
            out[nfft - k - 1] = out[nfft / 2 + k];
        }
        check_mdct_inv(&input, &out);
    } else {
        clt_mdct_forward(cfg, &input, &mut out, &window, nfft / 2, shift, 1);
        check_mdct(&in_copy, &out);
    }
}

#[test]
fn unit_mdct() {
    let mut rand = GlibcRand::new(1);
    for nfft in [
        32, 256, 512, 1024, 2048, 36, 40, 60, 120, 240, 480, 960, 1920,
    ] {
        mdct_test1d(nfft, false, &mut rand);
        mdct_test1d(nfft, true, &mut rand);
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_entropy.c
// ---------------------------------------------------------------------------------------------

const M_LOG2E: f64 = 1.442_695_040_888_963_4;
const DATA_SIZE: usize = 10_000_000;
const DATA_SIZE2: usize = 10000;

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "straight port of one C test program; splitting it would obscure the correspondence"
)]
fn unit_entropy() {
    let mut entropy = 0f64;
    let mut buf = vec![0u8; DATA_SIZE];

    // Testing encoding of raw bit values.
    let nbits;
    {
        let mut enc = EcEnc::new(&mut buf);
        for ft in 2u32..1024 {
            for i in 0..ft {
                entropy += f64::from(ft).ln() * M_LOG2E;
                enc.enc_uint(i, ft);
            }
        }
        // Testing encoding of raw bit values.
        for ftb in 1u32..16 {
            for i in 0..(1u32 << ftb) {
                entropy += f64::from(ftb);
                let nbits = enc.tell();
                enc.enc_bits(i, ftb);
                let nbits2 = enc.tell();
                assert_eq!(
                    nbits2 - nbits,
                    ftb as i32,
                    "Used {} bits to encode {ftb} bits directly.",
                    nbits2 - nbits
                );
            }
        }
        nbits = enc.tell_frac();
        enc.done();
        assert!(f64::from(nbits) >= entropy * 8.0 * 0.99);
    }
    {
        let mut dec = EcDec::new(&buf);
        for ft in 2u32..1024 {
            for i in 0..ft {
                let sym = dec.dec_uint(ft);
                assert_eq!(sym, i, "Decoded {sym} instead of {i} with ft of {ft}.");
            }
        }
        for ftb in 1u32..16 {
            for i in 0..(1u32 << ftb) {
                let sym = dec.dec_bits(ftb);
                assert_eq!(sym, i, "Decoded {sym} instead of {i} with ftb of {ftb}.");
            }
        }
        let nbits2 = dec.tell_frac();
        assert_eq!(nbits, nbits2, "Reported number of bits used was wrong.");
    }

    // Testing an encoder bust prefers range coder data over raw bits. This isn't a general
    // guarantee, will only work for data that is buffered in the encoder state and not yet
    // stored in the user buffer, and should never get used in practice. It's mostly here for
    // code coverage completeness.
    {
        // Start with a 16-bit buffer.
        let mut enc = EcEnc::new(&mut buf[..2]);
        // Write 7 raw bits.
        enc.enc_bits(0x55, 7);
        // Write 12.3 bits of range coder data.
        enc.enc_uint(1, 2);
        enc.enc_uint(1, 3);
        enc.enc_uint(1, 4);
        enc.enc_uint(1, 5);
        enc.enc_uint(2, 6);
        enc.enc_uint(6, 7);
        enc.done();
        assert_ne!(enc.get_error(), 0);
    }
    {
        let mut dec = EcDec::new(&buf[..2]);
        // The raw bits should have been overwritten by the range coder data.
        assert_eq!(dec.dec_bits(7), 0x05);
        // And all the range coder data should have been encoded correctly.
        assert_eq!(dec.dec_uint(2), 1);
        assert_eq!(dec.dec_uint(3), 1);
        assert_eq!(dec.dec_uint(4), 1);
        assert_eq!(dec.dec_uint(5), 1);
        assert_eq!(dec.dec_uint(6), 2);
        assert_eq!(dec.dec_uint(7), 6);
    }

    let seed = seed();
    let mut rand = GlibcRand::new(seed);
    // "Testing random streams... Random seed: %u (%.4X)" prints `rand() % 65536`.
    rand.next();
    for _ in 0..409_600 {
        let a = rand.next_u();
        let b = rand.next_u() % 11;
        let ft = a / ((GlibcRand::RAND_MAX >> b) + 1) + 10;
        let a = rand.next_u();
        let b = rand.next_u() % 9;
        let sz = (a / ((GlibcRand::RAND_MAX >> b) + 1)) as usize;
        let mut data = vec![0u32; sz];
        let mut tell = vec![0u32; sz + 1];
        let tell_bits;
        {
            let mut enc = EcEnc::new(&mut buf[..DATA_SIZE2]);
            let zeros = rand.next() % 13 == 0;
            tell[0] = enc.tell_frac();
            for j in 0..sz {
                data[j] = if zeros { 0 } else { rand.next_u() % ft };
                enc.enc_uint(data[j], ft);
                tell[j + 1] = enc.tell_frac();
            }
            if rand.next() % 2 == 0 {
                while enc.tell() % 8 != 0 {
                    enc.enc_uint(rand.next_u() % 2, 2);
                }
            }
            tell_bits = enc.tell();
            enc.done();
            assert_eq!(
                tell_bits,
                enc.tell(),
                "ec_tell() changed after ec_enc_done() (Random seed: {seed})"
            );
            assert!(
                (tell_bits as u32).div_ceil(8) >= enc.range_bytes(),
                "ec_tell() lied (Random seed: {seed})"
            );
        }
        let mut dec = EcDec::new(&buf[..DATA_SIZE2]);
        assert_eq!(dec.tell_frac(), tell[0], "Tell mismatch at symbol 0");
        for j in 0..sz {
            let sym = dec.dec_uint(ft);
            assert_eq!(
                sym, data[j],
                "Decoded {sym} instead of {} with ft of {ft} at position {j} of {sz} (Random seed: {seed}).",
                data[j]
            );
            assert_eq!(
                dec.tell_frac(),
                tell[j + 1],
                "Tell mismatch between encoder and decoder at symbol {} (Random seed: {seed}).",
                j + 1
            );
        }
    }

    // Test compatibility between multiple different encode/decode routines.
    for _ in 0..409_600 {
        let a = rand.next_u();
        let b = rand.next_u() % 9;
        let sz = (a / ((GlibcRand::RAND_MAX >> b) + 1)) as usize;
        let mut logp1 = vec![0u32; sz];
        let mut data = vec![0u32; sz];
        let mut tell = vec![0u32; sz + 1];
        let mut enc_method = vec![0u32; sz];
        {
            let mut enc = EcEnc::new(&mut buf[..DATA_SIZE2]);
            tell[0] = enc.tell_frac();
            for j in 0..sz {
                data[j] = rand.next_u() / ((GlibcRand::RAND_MAX >> 1) + 1);
                logp1[j] = (rand.next_u() % 15) + 1;
                enc_method[j] = rand.next_u() / ((GlibcRand::RAND_MAX >> 2) + 1);
                let (l, d) = (logp1[j], data[j] != 0);
                match enc_method[j] {
                    0 => enc.encode(
                        if d { (1 << l) - 1 } else { 0 },
                        (1 << l) - u32::from(!d),
                        1 << l,
                    ),
                    1 => enc.encode_bin(
                        if d { (1 << l) - 1 } else { 0 },
                        (1 << l) - u32::from(!d),
                        l,
                    ),
                    2 => enc.enc_bit_logp(d, l),
                    _ => enc.enc_icdf(usize::from(d), &[1, 0], l),
                }
                tell[j + 1] = enc.tell_frac();
            }
            enc.done();
            assert!(
                (enc.tell() as u32).div_ceil(8) >= enc.range_bytes(),
                "tell() lied (Random seed: {seed})"
            );
        }
        let mut dec = EcDec::new(&buf[..DATA_SIZE2]);
        assert_eq!(dec.tell_frac(), tell[0], "Tell mismatch at symbol 0");
        for j in 0..sz {
            let dec_method = rand.next_u() / ((GlibcRand::RAND_MAX >> 2) + 1);
            let l = logp1[j];
            let sym = match dec_method {
                0 => {
                    let fs = dec.decode(1 << l);
                    let sym = fs >= (1 << l) - 1;
                    dec.update(
                        if sym { (1 << l) - 1 } else { 0 },
                        (1 << l) - u32::from(!sym),
                        1 << l,
                    );
                    sym
                }
                1 => {
                    let fs = dec.decode_bin(l);
                    let sym = fs >= (1 << l) - 1;
                    dec.update(
                        if sym { (1 << l) - 1 } else { 0 },
                        (1 << l) - u32::from(!sym),
                        1 << l,
                    );
                    sym
                }
                2 => dec.dec_bit_logp(l),
                _ => dec.dec_icdf(&[1, 0], l) != 0,
            };
            assert_eq!(
                u32::from(sym),
                data[j],
                "Decoded {sym} instead of {} with logp1 of {l} at position {j} of {sz} \
                 (Random seed: {seed}). Encoding method: {}, decoding method: {dec_method}",
                data[j],
                enc_method[j]
            );
            assert_eq!(
                dec.tell_frac(),
                tell[j + 1],
                "Tell mismatch between encoder and decoder at symbol {} (Random seed: {seed}).",
                j + 1
            );
        }
    }

    {
        let mut enc = EcEnc::new(&mut buf[..DATA_SIZE2]);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(false, 2);
        enc.patch_initial_bits(3, 2);
        assert_eq!(enc.get_error(), 0, "patch_initial_bits failed");
        enc.patch_initial_bits(0, 5);
        assert_ne!(
            enc.get_error(),
            0,
            "patch_initial_bits didn't fail when it should have"
        );
        enc.done();
        assert_eq!(enc.range_bytes(), 1);
    }
    assert_eq!(
        buf[0], 192,
        "Got {} when expecting 192 for patch_initial_bits",
        buf[0]
    );
    {
        let mut enc = EcEnc::new(&mut buf[..DATA_SIZE2]);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(false, 1);
        enc.enc_bit_logp(true, 6);
        enc.enc_bit_logp(false, 2);
        enc.patch_initial_bits(0, 2);
        assert_eq!(enc.get_error(), 0, "patch_initial_bits failed");
        enc.done();
        assert_eq!(enc.range_bytes(), 2);
    }
    assert_eq!(
        buf[0], 63,
        "Got {} when expecting 63 for patch_initial_bits",
        buf[0]
    );
    {
        let mut enc = EcEnc::new(&mut buf[..2]);
        enc.enc_bit_logp(false, 2);
        for _ in 0..48 {
            enc.enc_bits(0, 1);
        }
        enc.done();
        assert_ne!(
            enc.get_error(),
            0,
            "Raw bits overfill didn't fail when it should have"
        );
    }
    {
        let mut enc = EcEnc::new(&mut buf[..2]);
        for _ in 0..17 {
            enc.enc_bits(0, 1);
        }
        enc.done();
        assert_ne!(enc.get_error(), 0, "17 raw bits encoded in two bytes");
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_laplace.c
// ---------------------------------------------------------------------------------------------

/// `LAPLACE_MINP` / `LAPLACE_NMIN` of celt/laplace.c.
const LAPLACE_MINP: u32 = 1;
const LAPLACE_NMIN: u32 = 16;

/// Port of `ec_laplace_get_start_freq`.
const fn ec_laplace_get_start_freq(decay: i32) -> u32 {
    let ft: u32 = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN + 1);
    let fs = (ft * (16384 - decay) as u32) / (16384 + decay) as u32;
    fs + LAPLACE_MINP
}

#[test]
fn unit_laplace() {
    const DATA_SIZE: usize = 40000;
    let mut buf = vec![0u8; DATA_SIZE];
    let mut val = [0i32; 10000];
    let mut decay = [0i32; 10000];
    let mut rand = GlibcRand::new(1);

    val[0] = 3;
    decay[0] = 6000;
    val[1] = 0;
    decay[1] = 5800;
    val[2] = -1;
    decay[2] = 5600;
    for i in 3..10000 {
        val[i] = rand.next() % 15 - 7;
        decay[i] = rand.next() % 11000 + 5000;
    }
    let range_bytes = {
        let mut enc = EcEnc::new(&mut buf);
        for i in 0..10000 {
            ec_laplace_encode(
                &mut enc,
                &mut val[i],
                ec_laplace_get_start_freq(decay[i]),
                decay[i],
            );
        }
        enc.done();
        enc.range_bytes() as usize
    };

    let mut dec = EcDec::new(&buf[..range_bytes]);
    for i in 0..10000 {
        let d = ec_laplace_decode(&mut dec, ec_laplace_get_start_freq(decay[i]), decay[i]);
        assert_eq!(d, val[i], "Got {d} instead of {}", val[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_mathops.c
// ---------------------------------------------------------------------------------------------

/// `PI` of celt/mathops.h.
#[cfg(not(feature = "fixed-point"))]
const PI: f64 = 3.141_592_653_589_793_1;

#[test]
fn unit_mathops_bitexact() {
    // testbitexactcos
    let mut chk = 0i32;
    let mut max_d = 0i32;
    let mut last = 32767i32;
    let mut min_d = 32767i32;
    for i in 64..=16320i32 {
        let q = i32::from(bitexact_cos(i as i16));
        chk ^= q * i;
        let d = last - q;
        max_d = max_d.max(d);
        min_d = min_d.min(d);
        last = q;
    }
    assert_eq!(
        (chk, max_d, min_d),
        (89_408_644, 5, 0),
        "bitexact_cos failed"
    );
    assert_eq!(bitexact_cos(64), 32767);
    assert_eq!(bitexact_cos(16320), 200);
    assert_eq!(bitexact_cos(8192), 23171);

    // testbitexactlog2tan
    let mut fail = false;
    let mut chk = 0i32;
    let mut max_d = 0i32;
    let mut last = 15059i32;
    let mut min_d = 15059i32;
    for i in 64..8193i32 {
        let mid = i32::from(bitexact_cos(i as i16));
        let side = i32::from(bitexact_cos((16384 - i) as i16));
        let q = bitexact_log2tan(mid, side);
        chk ^= q * i;
        let d = last - q;
        if q != -bitexact_log2tan(side, mid) {
            fail = true;
        }
        max_d = max_d.max(d);
        min_d = min_d.min(d);
        last = q;
    }
    assert_eq!(
        (chk, max_d, min_d, fail),
        (15_821_257, 61, -2, false),
        "bitexact_log2tan failed"
    );
    assert_eq!(bitexact_log2tan(32767, 200), 15059);
    assert_eq!(bitexact_log2tan(30274, 12540), 2611);
    assert_eq!(bitexact_log2tan(23171, 23171), 0);
}

#[cfg(not(feature = "fixed-point"))]
#[test]
fn unit_mathops_div_sqrt() {
    use opusorus::celt::mathops::{celt_rcp, celt_sqrt};
    // testdiv
    for i in 1..=327_670i32 {
        let val = celt_rcp(i as f32);
        // C: `prod = val*i;` (float product, stored to double)
        let prod = f64::from(val * i as f32);
        assert!(
            (prod - 1.0).abs() <= 0.00025,
            "div failed: 1/{i}={val} (product = {prod})"
        );
    }
    // testsqrt
    let mut i = 1i32;
    while i <= 1_000_000_000 {
        let val = celt_sqrt(i as f32);
        let r = f64::from(i).sqrt();
        let ratio = f64::from(val) / r;
        assert!(
            (ratio - 1.0).abs() <= 0.0005 || (f64::from(val) - r).abs() <= 2.0,
            "sqrt failed: sqrt({i})={val} (ratio = {ratio})"
        );
        i += i >> 10;
        i += 1;
    }
}

#[cfg(not(feature = "fixed-point"))]
#[test]
fn unit_mathops_log2_exp2() {
    use opusorus::celt::mathops::{celt_exp2, celt_log2};
    // testlog2
    let error_threshold = 2.2e-06f32;
    let mut x = 0.001f32;
    while f64::from(x) < 1_677_700.0 {
        let error = ((1.442_695_040_888_963_387 * f64::from(x).ln()) - f64::from(celt_log2(x)))
            .abs() as f32;
        assert!(
            error <= error_threshold,
            "celt_log2 failed: (x = {x}, error = {error})"
        );
        x = (f64::from(x) + f64::from(x) / 8.0) as f32;
    }

    // testexp2
    let error_threshold = 2.3e-07f32;
    let mut x = -11.0f32;
    while x < 24.0 {
        let error = (f64::from(x) - (1.442_695_040_888_963_387 * f64::from(celt_exp2(x)).ln()))
            .abs() as f32;
        assert!(
            error <= error_threshold,
            "celt_exp2 failed: (x = {x}, error = {error})"
        );
        x += 0.0007f32;
    }

    // testexp2log2
    let error_threshold = 2.0e-06f32;
    let mut x = -11.0f32;
    while x < 24.0 {
        let error = (x - celt_log2(celt_exp2(x))).abs();
        assert!(
            error <= error_threshold,
            "celt_log2/celt_exp2 failed: (x = {x}, error = {error})"
        );
        x += 0.0007f32;
    }
}

#[cfg(not(feature = "fixed-point"))]
#[test]
fn unit_mathops_cos_atan2() {
    use opusorus::celt::mathops::{celt_atan2p_norm, celt_cos_norm2};
    // test_cos
    let error_threshold = 6.0e-07f32;
    let mut x = -4.0f32;
    while x < 4.0 {
        let error = (((0.5f32 as f64 * PI) * f64::from(x)).cos() as f32 - celt_cos_norm2(x)).abs();
        assert!(
            error <= error_threshold,
            "celt_cos_norm2 failed: (x = {x}, error = {error})"
        );
        x += 0.0007f32;
    }

    // test_atan2
    let error_threshold = 1.5e-07f32;
    let mut x = 0.0f32;
    while x < 1.0 {
        let mut y = 0.0f32;
        while y < 1.0 {
            if x == 0.0 && y == 0.0 {
                // atan2(0,0) is undefined behavior.
                y += 0.007f32;
                continue;
            }
            let error = (0.636_619_772_367_581f32 * f64::from(y).atan2(f64::from(x)) as f32
                - celt_atan2p_norm(y, x))
            .abs();
            assert!(
                error <= error_threshold,
                "celt_atan2p_norm failed: (x = {x}, y = {y}, error = {error})"
            );
            y += 0.007f32;
        }
        x += 0.007f32;
    }
}

/// The `FIXED_POINT` tests of test_unit_mathops.c. `FIX_INT_TO_DOUBLE(x, q)` is
/// `ldexp((double)x, -q)` (exact) and `DOUBLE_TO_FIX_INT(x, q)` is `ldexp((double)x, q)`
/// converted to `opus_int32` (truncation). Error values are stored in `float` variables as in C.
#[cfg(feature = "fixed-point")]
mod mathops_fixed {
    use opusorus::celt::mathops::{
        celt_cos_norm32, celt_exp2, celt_ilog2, celt_log2, celt_rcp, celt_rcp_norm32,
        celt_rsqrt_norm32, celt_sqrt, celt_sqrt32,
    };

    /// `FIX_INT_TO_DOUBLE(x, q)`.
    fn fix_to_double(x: i32, q: i32) -> f64 {
        f64::from(x) * 2f64.powi(-q)
    }

    /// `DOUBLE_TO_FIX_INT(x, q)` stored in an `opus_int32`. C converts an out-of-range double
    /// with undefined behaviour (`test_rcp_norm32` reaches `1.0` in Q31); `as` saturates, as
    /// AArch64 does.
    fn double_to_fix(x: f32, q: i32) -> i32 {
        (f64::from(x) * 2f64.powi(q)) as i32
    }

    #[test]
    fn unit_mathops_div_sqrt() {
        // testdiv
        for i in 1..=327_670i32 {
            let val = celt_rcp(i);
            let prod = (1.0 / 32768.0 / 65526.0) * f64::from(val) * f64::from(i);
            assert!(
                (prod - 1.0).abs() <= 0.00025,
                "div failed: 1/{i}={val} (product = {prod})"
            );
        }
        // testsqrt: `opus_val16 val = celt_sqrt(i)`.
        let mut i = 1i32;
        while i <= 1_000_000_000 {
            let val = celt_sqrt(i) as i16;
            let r = f64::from(i).sqrt();
            let ratio = f64::from(val) / r;
            assert!(
                (ratio - 1.0).abs() <= 0.0005 || (f64::from(val) - r).abs() <= 2.0,
                "sqrt failed: sqrt({i})={val} (ratio = {ratio})"
            );
            i += i >> 10;
            i += 1;
        }
    }

    #[test]
    fn unit_mathops_log2_exp2() {
        // testlog2
        let mut x = 8i32;
        while x < 1_073_741_824 {
            let error = ((1.442_695_040_888_963_387 * (f64::from(x) / 16384.0).ln())
                - f64::from(celt_log2(x)) / 1024.0)
                .abs() as f32;
            assert!(
                f64::from(error) <= 0.003,
                "celt_log2 failed: x = {x}, error = {error}"
            );
            x += x >> 3;
        }
        // testexp2
        for x in -32768..15360i32 {
            let x = x as i16;
            let e = f64::from(celt_exp2(x));
            let error1 = (f64::from(x) / 1024.0 - (1.442_695_040_888_963_387 * (e / 65536.0).ln()))
                .abs() as f32;
            let error2 = ((0.693_147_180_559_945_309_4 * f64::from(x) / 1024.0).exp() - e / 65536.0)
                .abs() as f32;
            assert!(
                f64::from(error1) <= 0.0002 || f64::from(error2) <= 0.00004,
                "celt_exp2 failed: x = {x}, error1 = {error1}, error2 = {error2}"
            );
        }
        // testexp2log2
        let mut x = 8i32;
        while x < 65536 {
            let error =
                ((f64::from(x) - 0.25 * f64::from(celt_exp2(celt_log2(x)))).abs() / 16384.0) as f32;
            assert!(
                f64::from(error) <= 0.004,
                "celt_log2/celt_exp2 failed: (x = {x}, error = {error})"
            );
            x += x >> 3;
        }
    }

    #[test]
    fn unit_mathops_cos_norm32() {
        let error_threshold = 1e-07f32;
        let mut fx = -1.0f32;
        while fx <= 1.0 {
            let x = double_to_fix(fx, 30);
            let error = ((1.570_796_326_794_896_6 * fix_to_double(x, 30)).cos()
                - fix_to_double(celt_cos_norm32(x), 31))
            .abs() as f32;
            assert!(
                error <= error_threshold,
                "celt_cos_norm32 failed: error: [{error} > {error_threshold}] (x = {fx})"
            );
            fx += 0.007f32;
        }
    }

    #[test]
    fn unit_mathops_ilog2() {
        let mut x = 1i32;
        while x <= 268_435_455 {
            let lg = celt_ilog2(x);
            assert!(
                (0..31).contains(&lg),
                "celt_ilog2 failed: 0<=celt_ilog2(x)<31 (x = {x}, celt_ilog2(x) = {lg})"
            );
            let y = 1i32 << lg;
            assert!(
                x >= y && (x >> 1) < y,
                "celt_ilog2 failed: 2**celt_ilog2(x)<=x<2**(celt_ilog2(x)+1) (x = {x}, \
                 2**celt_ilog2(x) = {y})"
            );
            x += 127;
        }
    }

    #[test]
    fn unit_mathops_rsqrt_sqrt32() {
        // testrsqrt
        let error_threshold = 6.0e-08f32;
        let mut fx = 0.25f32;
        while fx < 1.0 {
            let x = double_to_fix(fx, 31);
            let quantized_fx = fix_to_double(x, 31) as f32;
            let error = (fix_to_double(celt_rsqrt_norm32(x), 29)
                - 1.0 / f64::from(quantized_fx).sqrt())
            .abs() as f32;
            assert!(
                error <= error_threshold,
                "celt_rsqrt_norm32 failed: (x = {quantized_fx}, error = {error})"
            );
            fx += 0.007f32;
        }
        // testsqrt32
        let two_lsbs = fix_to_double(2, 16) as f32;
        let mut i = 0i32;
        while i <= 1_073_741_824 + 64 {
            let r = f64::from(i).sqrt();
            let absolute_error = (r - fix_to_double(celt_sqrt32(i), 16)).abs() as f32;
            let relative_error_threshold = (8e-8 * r) as f32;
            assert!(
                absolute_error <= two_lsbs || absolute_error <= relative_error_threshold,
                "celt_sqrt32 failed: absolute_error: [{absolute_error} > {two_lsbs}] \
                 relative_error: [{absolute_error} > {relative_error_threshold}] (x = {i})"
            );
            i += i >> 25;
            i += 1;
        }
    }

    #[test]
    fn unit_mathops_rcp_norm32() {
        let two_lsbs = fix_to_double(2, 29) as f32;
        let relative_error_threshold = 6.51e-08f32;
        let mut fx = 0.5f32;
        while f64::from(fx) <= 1.0 {
            let x = double_to_fix(fx, 31);
            let quantized_fx = fix_to_double(x, 31) as f32;
            // `1 / quantized_fx` is a float division.
            let ground_truth = f64::from(1.0 / quantized_fx);
            let absolute_error =
                (ground_truth - fix_to_double(celt_rcp_norm32(x), 30)).abs() as f32;
            let relative_error = (f64::from(absolute_error) / ground_truth) as f32;
            assert!(
                absolute_error <= two_lsbs
                    || f64::from(absolute_error)
                        <= f64::from(relative_error_threshold) * ground_truth,
                "celt_rcp_norm32 failed: absolute_error: [{absolute_error} > {two_lsbs}] \
                 relative_error: [{relative_error} > {relative_error_threshold}] (x = \
                 {quantized_fx})"
            );
            fx = (f64::from(fx) + 0.000_000_7) as f32;
        }
    }

    #[cfg(feature = "qext")]
    #[test]
    fn unit_mathops_log2_exp2_db() {
        use opusorus::celt::arch::DB_SHIFT;
        use opusorus::celt::mathops::{celt_exp2_db, celt_log2_db};
        // testlog2_db
        let error_threshold = 2.0e-07f32;
        let mut x = 8i32;
        while x < 1_073_741_824 {
            let error = ((1.442_695_040_888_963_387 * fix_to_double(x, 14).ln())
                - fix_to_double(celt_log2_db(x), DB_SHIFT))
            .abs() as f32;
            assert!(
                error <= error_threshold,
                "celt_log2_db failed: error: [{error} > {error_threshold}] (x = {x})"
            );
            x += x >> 3;
        }
        // testexp2_db
        let absolute_error_threshold = fix_to_double(2, 16) as f32;
        let mut fx = -32.0f32;
        while f64::from(fx) < 15.0 {
            let x_32 = double_to_fix(fx, DB_SHIFT);
            let quantized_fx = fix_to_double(x_32, DB_SHIFT) as f32;
            let ground_truth = (0.693_147_180_559_945_309_4 * f64::from(quantized_fx)).exp();
            let absolute_error =
                (ground_truth - fix_to_double(celt_exp2_db(x_32), 16)).abs() as f32;
            let relative_error_threshold = (1.24e-7 * ground_truth) as f32;
            assert!(
                absolute_error <= absolute_error_threshold
                    || absolute_error <= relative_error_threshold,
                "celt_exp2_db failed: absolute_error: [{absolute_error} > \
                 {absolute_error_threshold}] relative_error: [{absolute_error} > \
                 {relative_error_threshold}] (x = {quantized_fx})"
            );
            fx = (f64::from(fx) + 0.0007) as f32;
        }
    }

    #[cfg(feature = "qext")]
    #[test]
    fn unit_mathops_atan() {
        use opusorus::celt::mathops::{celt_atan_norm, celt_atan2p_norm};
        const ATAN2_2_OVER_PI: f32 = 0.636_619_772_367_581;
        // testatan_norm
        let error_threshold = 5.97e-08f32;
        let mut fx = -1.0f32;
        while fx <= 1.0 {
            let x = double_to_fix(fx, 30);
            let error = (fix_to_double(x, 30).atan() * f64::from(ATAN2_2_OVER_PI)
                - fix_to_double(celt_atan_norm(x), 30))
            .abs() as f32;
            assert!(
                error <= error_threshold,
                "celt_atan_norm failed: error: [{error} > {error_threshold}] (x = {fx})"
            );
            fx += 0.007f32;
        }
        // testatan2p_norm
        let error_threshold = 1.2e-07f32;
        let mut fx = 0.0f32;
        while fx <= 1.0 {
            let x = double_to_fix(fx, 30);
            let mut fy = 0.0f32;
            while fy <= 1.0 {
                let y = double_to_fix(fy, 30);
                // C: `if (x == 0 && x == 0) continue;` (skips the whole x == 0 column).
                if x != 0 {
                    let error = (fix_to_double(y, 30).atan2(fix_to_double(x, 30))
                        * f64::from(ATAN2_2_OVER_PI)
                        - fix_to_double(celt_atan2p_norm(y, x), 30))
                    .abs() as f32;
                    assert!(
                        error <= error_threshold,
                        "celt_atan2p_norm failed: error: [{error} > {error_threshold}] (x = \
                         {fx}, y = {fy})"
                    );
                }
                fy += 0.007f32;
            }
            fx += 0.007f32;
        }
    }
}

/// `testcelt_float2int16`.
fn testcelt_float2int16(buffer_size: usize) {
    const MAX_BUFFER_SIZE: usize = 2080;
    let mut floats = [0f32; MAX_BUFFER_SIZE];
    let mut results = [0i16; MAX_BUFFER_SIZE];
    let scale = 1.0f32 / 32768.0;
    let mut cnt = 0;
    while cnt + 15 < buffer_size && cnt < buffer_size / 2 {
        for v in [
            77777.0f32 * scale,
            33000.0 * scale,
            32768.0 * scale,
            32767.4 * scale,
            32766.6 * scale,
            (0.501f64 * f64::from(scale)) as f32,
            0.499 * scale,
            0.0,
            -0.499 * scale,
            -0.501 * scale,
            -32767.6 * scale,
            -32768.4 * scale,
            -32769.0 * scale,
            -33000.0 * scale,
            -77777.0 * scale,
        ] {
            floats[cnt] = v;
            cnt += 1;
        }
        assert!(cnt < buffer_size);
    }
    while cnt < buffer_size {
        let mut in_range = (cnt as f64 * 7.0 + 0.5) as f32;
        in_range = (f64::from(in_range) + if cnt & 1 != 0 { 0.1 } else { -0.1 }) as f32;
        in_range *= if cnt & 2 != 0 { 1.0 } else { -1.0 };
        floats[cnt] = in_range * scale;
        cnt += 1;
    }
    results.fill(42);
    celt_float2int16(&floats[..cnt], &mut results[..cnt]);
    for i in 0..cnt {
        let expected = float2int16(floats[i]);
        assert_eq!(
            results[i], expected,
            "testcelt_float2int16 failed: converted {} (index: {i}, cnt: {buffer_size})",
            floats[i]
        );
    }
    assert!(
        results[cnt..].iter().all(|&v| v == 42),
        "testcelt_float2int16 failed: buffer overflow (cnt: {buffer_size})"
    );
}

/// `testopus_limit2_checkwithin1`.
fn testopus_limit2_checkwithin1() {
    // strange float count to trigger residue loop of SIMD implementation
    const BUFFER_SIZE: usize = 37;
    let mut pattern = [0f32; BUFFER_SIZE];
    for (i, p) in pattern.iter_mut().enumerate() {
        *p = if i % 2 != 0 { -1.0 } else { 1.0 };
    }
    // All values within -1..1: Nothing changed. Return value is implementation-dependent (not
    // expected to recognise nothing exceeds -1..1)
    let mut buffer = pattern;
    let _within1 = opus_limit2_checkwithin1(&mut buffer);
    assert_eq!(
        buffer, pattern,
        "opus_limit2_checkwithin1() modified values not exceeding -1..1"
    );
    // One value exceeds -1..1, within -2..2: Values unchanged. Return value says not all
    // values are within -1..1
    for i in 0..BUFFER_SIZE {
        let replace_value = pattern[i] * 1.001f32;
        buffer = pattern;
        buffer[i] = replace_value;
        let within1 = opus_limit2_checkwithin1(&mut buffer);
        assert!(
            !within1 && buffer[i] == replace_value,
            "opus_limit2_checkwithin1() handled value exceeding -1..1 erroneously (i={i})"
        );
        buffer[i] = pattern[i];
        assert_eq!(
            buffer, pattern,
            "opus_limit2_checkwithin1() modified value within -2..2 (i={i})"
        );
    }
    // One value exceeds -2..2: One value is hardclipped, others are unchanged. Return value
    // says not all values are within -1..1
    for i in 0..BUFFER_SIZE {
        let replace_value = (f64::from(pattern[i]) * 2.1) as f32;
        buffer = pattern;
        buffer[i] = replace_value;
        let within1 = opus_limit2_checkwithin1(&mut buffer);
        let clipped = if replace_value > 0.0 { 2.0 } else { -2.0 };
        assert!(
            !within1 && buffer[i] == clipped,
            "opus_limit2_checkwithin1() handled value exceeding -2..2 erroneously (i={i})"
        );
        buffer[i] = pattern[i];
        assert_eq!(
            buffer, pattern,
            "opus_limit2_checkwithin1() modified value within -2..2 (i={i})"
        );
    }
}

#[test]
fn unit_mathops_float2int16_limit2() {
    // use_ref_impl = 0 and 1 (opusorus has one implementation).
    for _use_ref_impl in [0, 1] {
        testcelt_float2int16(1);
        testcelt_float2int16(32);
        testcelt_float2int16(127);
        testcelt_float2int16(1031);
        testopus_limit2_checkwithin1();
    }
}

// ---------------------------------------------------------------------------------------------
// test_unit_rotation.c
// ---------------------------------------------------------------------------------------------

fn test_rotation(n: usize, k: i32, rand: &mut GlibcRand) {
    const MAX_SIZE: usize = 100;
    let mut x0 = [norm(0); MAX_SIZE];
    let mut x1 = [norm(0); MAX_SIZE];
    for i in 0..n {
        let v = norm(rand.next() % 16_777_215 - 8_388_608);
        x0[i] = v;
        x1[i] = v;
    }
    let snr_of = |x0: &[CeltNorm], x1: &[CeltNorm]| {
        let mut err = 0f64;
        let mut ener = 0f64;
        for i in 0..n {
            let d = f64::from(x0[i]) - f64::from(x1[i]);
            err += d * d;
            ener += f64::from(x0[i]) * f64::from(x0[i]);
        }
        20.0 * (ener / err).log10()
    };
    exp_rotation(&mut x1[..n], n as i32, 1, 1, k, SPREAD_NORMAL);
    let snr0 = snr_of(&x0, &x1);
    exp_rotation(&mut x1[..n], n as i32, -1, 1, k, SPREAD_NORMAL);
    let snr = snr_of(&x0, &x1);
    assert!(
        snr >= 60.0 && snr0 <= 20.0,
        "SNR for size {n} ({k} pulses) is {snr} (was {snr0} without inverse)"
    );
}

#[test]
fn unit_rotation() {
    let mut rand = GlibcRand::new(1);
    test_rotation(15, 3, &mut rand);
    test_rotation(23, 5, &mut rand);
    test_rotation(50, 3, &mut rand);
    test_rotation(80, 1, &mut rand);
}

// ---------------------------------------------------------------------------------------------
// test_opus_projection.c: test_simple_matrix
// ---------------------------------------------------------------------------------------------

const SIMPLE_MATRIX_FRAME_SIZE: usize = 10;
const SIMPLE_MATRIX_INPUT_SIZE: usize = 30;
const SIMPLE_MATRIX_OUTPUT_SIZE: usize = 40;
const ERROR_TOLERANCE: i32 = 1;

fn assert_is_equal(a: &[OpusRes], b: &[i16]) {
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (i32::from(res2int16(x)) - i32::from(y)).abs() <= ERROR_TOLERANCE,
            "index {i}: {} vs {y}",
            res2int16(x)
        );
    }
}

fn assert_is_equal_short(a: &[i16], b: &[i16]) {
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (i32::from(x) - i32::from(y)).abs() <= ERROR_TOLERANCE,
            "index {i}: {x} vs {y}"
        );
    }
}

#[test]
fn projection_simple_matrix() {
    let (rows, cols, gain) = (4, 3, 0);
    let simple_matrix_data: [i16; 12] = [0, 32767, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767];
    let input_int16: [i16; SIMPLE_MATRIX_INPUT_SIZE] = [
        32767, 0, -32768, 29491, -3277, -29491, 26214, -6554, -26214, 22938, -9830, -22938, 19661,
        -13107, -19661, 16384, -16384, -16384, 13107, -19661, -13107, 9830, -22938, -9830, 6554,
        -26214, -6554, 3277, -29491, -3277,
    ];
    let expected_output_int16: [i16; SIMPLE_MATRIX_OUTPUT_SIZE] = [
        0, 32767, 0, -32768, -3277, 29491, 0, -29491, -6554, 26214, 0, -26214, -9830, 22938, 0,
        -22938, -13107, 19661, 0, -19661, -16384, 16384, 0, -16384, -19661, 13107, 0, -13107,
        -22938, 9830, 0, -9830, -26214, 6554, 0, -6554, -29491, 3277, 0, -3277,
    ];

    // Initialize matrix
    assert_ne!(mapping_matrix_get_size(rows, cols), 0);
    let simple_matrix = MappingMatrix::new(rows, cols, gain, &simple_matrix_data);
    let (rows, cols) = (rows as usize, cols as usize);

    // Copy inputs.
    let input_pcm: Vec<OpusRes> = input_int16.iter().map(|&v| int16tores(v)).collect();

    // _in_short
    let mut output_pcm = [OpusRes::default(); SIMPLE_MATRIX_OUTPUT_SIZE];
    for i in 0..rows {
        mapping_matrix_multiply_channel_in_short(
            &simple_matrix,
            &input_int16,
            cols,
            &mut output_pcm[i..],
            i,
            rows,
            SIMPLE_MATRIX_FRAME_SIZE,
        );
    }
    assert_is_equal(&output_pcm, &expected_output_int16);

    // _out_short
    let mut output_int16 = [0i16; SIMPLE_MATRIX_OUTPUT_SIZE];
    for i in 0..cols {
        mapping_matrix_multiply_channel_out_short(
            &simple_matrix,
            &input_pcm[i..],
            i,
            cols,
            &mut output_int16,
            rows,
            SIMPLE_MATRIX_FRAME_SIZE,
        );
    }
    assert_is_equal_short(&output_int16, &expected_output_int16);

    // `!defined(DISABLE_FLOAT_API) && !defined(FIXED_POINT)` in C.
    #[cfg(not(feature = "fixed-point"))]
    projection_simple_matrix_float(&simple_matrix, &input_pcm, &expected_output_int16);
}

/// The `_in_float` / `_out_float` part of `test_simple_matrix` (float build only).
#[cfg(not(feature = "fixed-point"))]
fn projection_simple_matrix_float(
    simple_matrix: &MappingMatrix,
    input_pcm: &[f32],
    expected_output_int16: &[i16],
) {
    let (rows, cols) = (4, 3);
    // _in_float
    let mut output_pcm = [0f32; SIMPLE_MATRIX_OUTPUT_SIZE];
    for i in 0..rows {
        mapping_matrix_multiply_channel_in_float(
            simple_matrix,
            input_pcm,
            cols,
            &mut output_pcm[i..],
            i,
            rows,
            SIMPLE_MATRIX_FRAME_SIZE,
        );
    }
    assert_is_equal(&output_pcm, expected_output_int16);

    // _out_float
    let mut output_pcm = [0f32; SIMPLE_MATRIX_OUTPUT_SIZE];
    for i in 0..cols {
        mapping_matrix_multiply_channel_out_float(
            simple_matrix,
            &input_pcm[i..],
            i,
            cols,
            &mut output_pcm,
            rows,
            SIMPLE_MATRIX_FRAME_SIZE,
        );
    }
    assert_is_equal(&output_pcm, expected_output_int16);
}

// ---------------------------------------------------------------------------------------------
// test_opus_custom.c
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "custom-modes")]
mod custom {
    //! Port of `tests/test_opus_custom.c`: random combinations of the Opus and Opus Custom
    //! encoders/decoders (float/16-bit/24-bit I/O, random settings) on a sine sweep, with
    //! corrupted copies of every packet fed to a second decoder.
    //!
    //! libopus fuzzes 5 encoders x 40 settings over a 60 s sweep; the default test runs 5 x 10
    //! settings over a 3 s sweep, the full length is the `#[ignore]`d `opus_custom_full`.
    //! `RESYNTH` (the RMS encoder/decoder comparison) is a compile-time C option that is not
    //! enabled in the C test build and is not ported.

    use super::seed;
    use opusorus::celt::celt_decoder::CustomDecoder;
    use opusorus::celt::celt_encoder::CeltEncoder;
    use opusorus::celt::modes::opus_custom_mode_create_custom;
    use opusorus::celt::static_modes::CeltMode;
    use opusorus::constants::raw::OPUS_BITRATE_MAX;
    use opusorus::{Application, Bitrate, Decoder, Encoder, Error};

    const MAX_PACKET: usize = 1500;
    const PI: f64 = 3.141_592_653_589_793_238_462_643;
    const SINE_SWEEP_AMPLITUDE: f64 = 0.5;

    /// `fast_rand` of `tests/test_opus_common.h`.
    struct FastRand {
        rz: u32,
        rw: u32,
    }

    impl FastRand {
        const fn next(&mut self) -> u32 {
            self.rz = 36969u32
                .wrapping_mul(self.rz & 65535)
                .wrapping_add(self.rz >> 16);
            self.rw = 18000u32
                .wrapping_mul(self.rw & 65535)
                .wrapping_add(self.rw >> 16);
            (self.rz << 16).wrapping_add(self.rw)
        }

        const fn sample<T: Copy>(&mut self, a: &[T]) -> T {
            a[(self.next() % a.len() as u32) as usize]
        }

        /// `log(1e-10 + fast_rand()/4294967296.)`.
        fn log_uniform(&mut self) -> f64 {
            (1e-10 + f64::from(self.next()) / 4_294_967_296.0).ln()
        }
    }

    /// The input of one test: interleaved samples in one of the three formats.
    enum Pcm {
        I16(Vec<i16>),
        I24(Vec<i32>),
        F32(Vec<f32>),
    }

    /// Port of `generate_sine_sweep`.
    fn generate_sine_sweep(
        amplitude: f64,
        bit_depth: i32,
        sample_rate: i32,
        channels: usize,
        use_float: bool,
        duration_seconds: f64,
    ) -> (Pcm, usize) {
        let start_freq = 100.0f64;
        let end_freq = f64::from(sample_rate) / 2.0;
        // Calculate the maximum sample value based on bit depth.
        let max_sample_value = ((1i64 << (bit_depth - 1)) - 1) as f64;
        let num_samples = (0.5f64 + duration_seconds * f64::from(sample_rate)).floor() as usize;
        let mut f = vec![0f32; if use_float { num_samples * channels } else { 0 }];
        let mut s16 = vec![
            0i16;
            if !use_float && bit_depth == 16 {
                num_samples * channels
            } else {
                0
            }
        ];
        let mut s24 = vec![
            0i32;
            if !use_float && bit_depth == 24 {
                num_samples * channels
            } else {
                0
            }
        ];
        for i in 0..num_samples {
            // Calculate the time in seconds for the current sample
            let t = i as f64 / f64::from(sample_rate);
            // Calculate the frequency at this time point
            let b = ((end_freq + start_freq) / start_freq).ln() / duration_seconds;
            let a = start_freq / b;
            let sample = amplitude * (2.0 * PI * a * (b * t).exp() - (b * t) - 1.0).sin();
            let idx = i * channels;
            if use_float {
                f[idx] = sample as f32;
                if channels == 2 {
                    f[idx + 1] = f[idx];
                }
            } else if bit_depth == 16 {
                s16[idx] = (0.5 + sample * max_sample_value).floor() as i16;
                if channels == 2 {
                    s16[idx + 1] = s16[idx];
                }
            } else {
                s24[idx] = (0.5 + sample * max_sample_value).floor() as i32;
                if channels == 2 {
                    s24[idx + 1] = s24[idx];
                }
            }
        }
        let pcm = if use_float {
            Pcm::F32(f)
        } else if bit_depth == 16 {
            Pcm::I16(s16)
        } else {
            Pcm::I24(s24)
        };
        (pcm, num_samples)
    }

    enum Enc {
        Opus(Box<Encoder>),
        Custom(Box<CeltEncoder>),
    }

    enum Dec<'m> {
        Opus(Box<Decoder>),
        Custom(Box<CustomDecoder<'m>>),
    }

    impl Clone for Dec<'_> {
        fn clone(&self) -> Self {
            match self {
                Self::Opus(d) => Self::Opus(d.clone()),
                Self::Custom(d) => Self::Custom(d.clone()),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Params {
        sample_rate: i32,
        num_channels: usize,
        frame_size: usize,
        float_encode: bool,
        float_decode: bool,
        encoder_bit_depth: i32,
        decoder_bit_depth: i32,
    }

    /// Accepted results of decoding a corrupted packet.
    const fn corrupt_ok(r: Result<usize, Error>) -> bool {
        match r {
            Ok(n) => n > 0,
            Err(e) => matches!(
                e,
                Error::BadArg | Error::InvalidPacket | Error::BufferTooSmall
            ),
        }
    }

    /// Port of `test_encode`.
    #[allow(
        clippy::too_many_lines,
        reason = "straight port of one C test function; splitting it would obscure the \
                  correspondence"
    )]
    fn test_encode(
        params: Params,
        enc: &mut Enc,
        dec: &mut Dec<'_>,
        duration: f64,
        rng: &mut FastRand,
    ) {
        let mut packet = [0u8; MAX_PACKET + 257];
        let ch = params.num_channels;
        let frame_size = params.frame_size;

        // Generate input data
        let (inbuf, input_samples) = generate_sine_sweep(
            SINE_SWEEP_AMPLITUDE,
            params.encoder_bit_depth,
            params.sample_rate,
            ch,
            params.float_encode,
            duration,
        );

        // Allocate memory for output data
        let n = input_samples * ch;
        let mut out_f = vec![0f32; if params.float_decode { n } else { 0 }];
        let mut out_24 = vec![
            0i32;
            if !params.float_decode && params.decoder_bit_depth == 24 {
                n
            } else {
                0
            }
        ];
        let mut out_16 = vec![
            0i16;
            if !params.float_decode && params.decoder_bit_depth != 24 {
                n
            } else {
                0
            }
        ];

        let mut dec_copy = dec.clone();

        let mut samp_count = 0usize;
        // Encode data, then decode for sanity check
        loop {
            let off = samp_count * ch;
            let len: usize = match (&mut *enc, &inbuf) {
                (Enc::Custom(e), Pcm::F32(x)) => {
                    let r = e.opus_custom_encode_float(
                        &x[off..],
                        frame_size as i32,
                        &mut packet,
                        MAX_PACKET as i32,
                    );
                    assert!(r > 0, "opus_custom_encode_float() failed: {r}");
                    r as usize
                }
                (Enc::Custom(e), Pcm::I24(x)) => {
                    let r = e.opus_custom_encode24(
                        &x[off..],
                        frame_size as i32,
                        &mut packet,
                        MAX_PACKET as i32,
                    );
                    assert!(r > 0, "opus_custom_encode24() failed: {r}");
                    r as usize
                }
                (Enc::Custom(e), Pcm::I16(x)) => {
                    let r = e.opus_custom_encode(
                        &x[off..],
                        frame_size as i32,
                        &mut packet,
                        MAX_PACKET as i32,
                    );
                    assert!(r > 0, "opus_custom_encode() failed: {r}");
                    r as usize
                }
                (Enc::Opus(e), Pcm::F32(x)) => e
                    .encode_float(&x[off..], frame_size, &mut packet[..MAX_PACKET])
                    .unwrap(),
                (Enc::Opus(e), Pcm::I24(x)) => e
                    .encode24(&x[off..], frame_size, &mut packet[..MAX_PACKET])
                    .unwrap(),
                (Enc::Opus(e), Pcm::I16(x)) => e
                    .encode(&x[off..], frame_size, &mut packet[..MAX_PACKET])
                    .unwrap(),
            };
            assert!(len > 0);

            // Generate bit/byte errors and check that nothing bad happens.
            {
                // Draw the inverse bit error rate from an exponential distribution.
                let ber_1 = (1.0 - 100.0 * rng.log_uniform()) as i32;
                let mut scratch = [0i16; 1920 * 6];
                let mut packet_corrupt = packet;
                // Randomly flip the 5 first bytes.
                for error_pos in 0..5 {
                    // 25% chance of flipping each byte.
                    if error_pos < len && rng.next().is_multiple_of(5) {
                        packet_corrupt[error_pos] = (rng.next() & 0xFF) as u8;
                    }
                }
                // Arbitrarily truncate the packet.
                let len2 = (1.0 - len as f64 * rng.log_uniform()) as i32;
                let len2 = len2.min(len as i32) as usize;
                let mut error_pos = 0i32;
                // Generate bit errors using "run-length" flipping.
                loop {
                    // C: `error_pos += (int)-ber_1*log(...)` (double sum converted to int).
                    error_pos =
                        (f64::from(error_pos) + f64::from(-ber_1) * rng.log_uniform()) as i32;
                    if error_pos >= (len2 * 8) as i32 {
                        break;
                    }
                    packet_corrupt[(error_pos / 8) as usize] ^= 1 << (error_pos & 7);
                }
                let r = match &mut dec_copy {
                    Dec::Custom(d) => d
                        .opus_custom_decode(
                            Some(&packet_corrupt[..len2]),
                            len2 as i32,
                            &mut scratch,
                            frame_size as i32,
                        )
                        .map(|n| n as usize),
                    Dec::Opus(d) => d.decode(
                        Some(&packet_corrupt[..len2]),
                        &mut scratch,
                        frame_size,
                        false,
                    ),
                };
                assert!(
                    corrupt_ok(r),
                    "decode with corrupt stream failed with: {r:?}"
                );
            }

            let data = Some(&packet[..len]);
            let samples_decoded = if params.float_decode {
                let out = &mut out_f[off..];
                match dec {
                    Dec::Custom(d) => d
                        .opus_custom_decode_float(data, len as i32, out, frame_size as i32)
                        .map(|n| n as usize),
                    Dec::Opus(d) => d.decode_float(data, out, frame_size, false),
                }
            } else if params.decoder_bit_depth == 24 {
                let out = &mut out_24[off..];
                match dec {
                    Dec::Custom(d) => d
                        .opus_custom_decode24(data, len as i32, out, frame_size as i32)
                        .map(|n| n as usize),
                    Dec::Opus(d) => d.decode24(data, out, frame_size, false),
                }
            } else {
                let out = &mut out_16[off..];
                match dec {
                    Dec::Custom(d) => d
                        .opus_custom_decode(data, len as i32, out, frame_size as i32)
                        .map(|n| n as usize),
                    Dec::Opus(d) => d.decode(data, out, frame_size, false),
                }
            };
            assert_eq!(samples_decoded, Ok(frame_size));

            samp_count += frame_size;
            if samp_count + frame_size > input_samples {
                break;
            }
        }
    }

    /// Port of `test_opus_custom`.
    fn test_opus_custom(num_encoders: usize, num_setting_changes: usize, duration: f64) {
        let mut rng = FastRand {
            rz: seed(),
            rw: seed(),
        };
        // `main` prints `fast_rand() % 65535` with the seed.
        rng.next();

        // Parameters to fuzz. Some values are duplicated to increase their probability of being
        // tested. (The C list also has 8/12/16/24 kHz under `#ifdef CUSTOM_MODEES`, a typo
        // that is never defined, so they are not used.)
        #[cfg(feature = "qext")]
        let sampling_rates: &[i32] = &[48000, 96000];
        #[cfg(not(feature = "qext"))]
        let sampling_rates: &[i32] = &[48000];
        let channels = [1usize, 2];
        let bitrates = [
            6000,
            12000,
            16000,
            24000,
            32000,
            48000,
            64000,
            96000,
            510_000,
            OPUS_BITRATE_MAX,
        ];
        let use_vbr = [0, 1, 1];
        let vbr_constraints = [0, 1, 1];
        let complexities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let packet_loss_perc = [0, 1, 2, 5];
        let lsb_depths = [8, 24];
        let frame_sizes_ms_x2 = [5, 10, 20, 40]; // x2 to avoid 2.5 ms
        let use_float_encode = [false, true];
        let use_float_decode = [false, true];
        let use_custom_encode = [false, true];
        let use_custom_decode = [false, true];
        let encoder_bit_depths = [16, 24];
        let decoder_bit_depths = [16, 24];

        for _ in 0..num_encoders {
            let sample_rate = rng.sample(sampling_rates);
            let mut custom_encode = true;
            let mut custom_decode = true;
            // Can only mix and match Opus and OpusCustom with 48kHz (and optionally 96 kHz).
            if sample_rate == 48000 || sample_rate == 96000 {
                custom_encode = rng.sample(&use_custom_encode);
                custom_decode = rng.sample(&use_custom_decode);
                // No point in testing this as OpusCustom isn't involved
                if !(custom_encode || custom_decode) {
                    continue;
                }
            }
            let num_channels = rng.sample(&channels);
            let frame_size_ms_x2 = rng.sample(&frame_sizes_ms_x2);
            let frame_size = (frame_size_ms_x2 * sample_rate / 2000) as usize;

            // OpusCustom isn't supporting this case at the moment (frame < 40)
            if (sample_rate == 8000 || sample_rate == 12000) && frame_size_ms_x2 == 5 {
                continue;
            }

            let mode = match opus_custom_mode_create_custom(sample_rate, frame_size as i32) {
                Ok(m) => m,
                Err(e) => {
                    panic!("test_opus_custom error: {sample_rate} Hz, {frame_size} samples: {e}")
                }
            };
            let mode_ref: &CeltMode = &mode;

            let mut dec = if custom_decode {
                Dec::Custom(Box::new(
                    CustomDecoder::opus_custom_decoder_create(mode_ref, num_channels as i32)
                        .unwrap(),
                ))
            } else {
                Dec::Opus(Box::new(
                    Decoder::new(sample_rate, num_channels as i32).unwrap(),
                ))
            };
            let mut enc = if custom_encode {
                Enc::Custom(Box::new(
                    CeltEncoder::opus_custom_encoder_create(mode.clone(), num_channels as i32)
                        .unwrap(),
                ))
            } else {
                Enc::Opus(Box::new(
                    Encoder::new(
                        sample_rate,
                        num_channels as i32,
                        Application::RestrictedLowDelay,
                    )
                    .unwrap(),
                ))
            };

            for _ in 0..num_setting_changes {
                let bitrate = rng.sample(&bitrates);
                let vbr = rng.sample(&use_vbr);
                let vbr_constraint = rng.sample(&vbr_constraints);
                let complexity = rng.sample(&complexities);
                let pkt_loss = rng.sample(&packet_loss_perc);
                let lsb_depth = rng.sample(&lsb_depths);
                let float_encode = rng.sample(&use_float_encode);
                let float_decode = rng.sample(&use_float_decode);
                let encoder_bit_depth = rng.sample(&encoder_bit_depths);
                let decoder_bit_depth = rng.sample(&decoder_bit_depths);
                #[cfg(feature = "qext")]
                let qext = rng.next() & 1;

                match &mut enc {
                    Enc::Custom(e) => {
                        e.set_bitrate(bitrate).unwrap();
                        e.set_vbr(vbr);
                        e.set_vbr_constraint(vbr_constraint);
                        e.set_complexity(complexity).unwrap();
                        e.set_packet_loss_perc(pkt_loss).unwrap();
                        e.set_lsb_depth(lsb_depth).unwrap();
                        #[cfg(feature = "qext")]
                        e.set_qext(qext as i32).unwrap();
                    }
                    Enc::Opus(e) => {
                        e.set_bitrate(Bitrate::from_raw(bitrate)).unwrap();
                        e.set_vbr(vbr != 0);
                        e.set_vbr_constraint(vbr_constraint != 0);
                        e.set_complexity(complexity).unwrap();
                        e.set_packet_loss_perc(pkt_loss).unwrap();
                        e.set_lsb_depth(lsb_depth).unwrap();
                    }
                }
                let params = Params {
                    sample_rate,
                    num_channels,
                    frame_size,
                    float_encode,
                    float_decode,
                    encoder_bit_depth,
                    decoder_bit_depth,
                };
                test_encode(params, &mut enc, &mut dec, duration, &mut rng);
            }
        }
    }

    #[test]
    fn opus_custom() {
        test_opus_custom(5, 10, 3.0);
    }

    #[test]
    #[ignore = "full libopus length (5 x 40 settings over a 60 s sweep); run with --ignored"]
    fn opus_custom_full() {
        test_opus_custom(5, 40, 60.0);
    }
}

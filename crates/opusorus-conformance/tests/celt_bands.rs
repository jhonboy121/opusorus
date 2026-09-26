//! Differential tests for unit `celt_bands`: celt/vq.c, celt/quant_bands.c, celt/bands.c and
//! celt/celt.c vs the C oracle (bit-exact).
//!
//! Shared by the float and the fixed-point builds (`fixed-point`, `fixed-res24`, with or
//! without `qext`): inputs are generated as float values and converted to the build's types
//! (`celt_norm` Q24, `celt_sig`/`celt_ener` with `SIG_SHIFT`, `celt_glog` Q24, Q15/Q31 gains in
//! the fixed-point build), and every comparison is bit-exact.

#![allow(
    clippy::too_many_arguments,
    reason = "test runners mirror the flat oracle shim signatures"
)]

use opusorus::celt::arch::CeltCoef;
use opusorus::celt::bands::{self, BandsScratch};
use opusorus::celt::celt as celtc;
use opusorus::celt::entcode::EcCoder;
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::quant_bands as qb;
use opusorus::celt::rate;
use opusorus::celt::static_modes::{CeltMode, MODE48000_960_120};
use opusorus::celt::vq;
use opusorus_conformance::{Rng, assert_slice_eq};
use opusorus_oracle::celt_bands::{self as c, Ener, Glog, Norm, Sig, Val16, Val32};

// ---------------------------------------------------------------------------------------------
// Build-generic values
// ---------------------------------------------------------------------------------------------

/// Bit pattern of a value, for bit-exact comparisons of float and integer data alike.
trait Bits: Copy + core::fmt::Debug {
    fn bits(self) -> u64;
}
impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl Bits for i32 {
    fn bits(self) -> u64 {
        u64::from(self as u32)
    }
}
impl Bits for i16 {
    fn bits(self) -> u64 {
        u64::from(self as u16)
    }
}
impl Bits for i8 {
    fn bits(self) -> u64 {
        u64::from(self as u8)
    }
}

/// Asserts that two slices are bit-identical.
#[track_caller]
fn assert_v<T: Bits>(what: &str, rust: &[T], c: &[T]) {
    assert_eq!(rust.len(), c.len(), "{what}: length mismatch");
    if let Some(i) = rust.iter().zip(c).position(|(a, b)| a.bits() != b.bits()) {
        panic!(
            "{what}: first mismatch at {i}: rust={:?} c={:?}",
            rust[i], c[i]
        );
    }
}

/// Asserts that two values are bit-identical.
#[track_caller]
fn assert_v1<T: Bits>(what: &str, rust: T, c: T) {
    assert_v(what, &[rust], &[c]);
}

/// Scales `x` by `2^q` with rounding and saturation to `[-lim, lim]` (fixed-point inputs).
#[cfg(feature = "fixed-point")]
fn fx(x: f32, q: i32, lim: i64) -> i64 {
    ((f64::from(x) * (1i64 << q) as f64).round() as i64).clamp(-lim, lim)
}

/// A `celt_norm` from a float value (Q24 in the fixed-point build).
#[cfg(not(feature = "fixed-point"))]
const fn nrm(x: f32) -> Norm {
    x
}
#[cfg(feature = "fixed-point")]
fn nrm(x: f32) -> Norm {
    fx(x, 24, 1 << 30) as i32
}
/// A `celt_sig` / `celt_ener` from a float value in the CELT signal scale (`SIG_SHIFT` above
/// 16-bit PCM in the fixed-point build).
#[cfg(not(feature = "fixed-point"))]
const fn sig(x: f32) -> Sig {
    x
}
#[cfg(feature = "fixed-point")]
fn sig(x: f32) -> Sig {
    fx(x, 12, 1 << 30) as i32
}
/// A `celt_glog` from a value in log2 units (Q24 in the fixed-point build).
#[cfg(not(feature = "fixed-point"))]
const fn glog(x: f32) -> Glog {
    x
}
#[cfg(feature = "fixed-point")]
fn glog(x: f32) -> Glog {
    fx(x, 24, 1 << 30) as i32
}
/// An `opus_val16` Q15 value (e.g. a gain or a cosine).
#[cfg(not(feature = "fixed-point"))]
const fn q15(x: f32) -> Val16 {
    x
}
#[cfg(feature = "fixed-point")]
fn q15(x: f32) -> Val16 {
    fx(x, 15, 32767) as i16
}
/// An `opus_val16` Q14 value (the hysteresis thresholds).
#[cfg(not(feature = "fixed-point"))]
const fn q14(x: f32) -> Val16 {
    x
}
#[cfg(feature = "fixed-point")]
fn q14(x: f32) -> Val16 {
    fx(x, 14, 32767) as i16
}
/// An `opus_val32` Q31 gain (at most 1 in the fixed-point build, as the codec's gains).
#[cfg(not(feature = "fixed-point"))]
const fn q31(x: f32) -> Val32 {
    x
}
#[cfg(feature = "fixed-point")]
fn q31(x: f32) -> Val32 {
    fx(x, 31, i64::from(i32::MAX)) as i32
}
/// An `opus_val32` in plain units (`delayedIntra`).
#[cfg(not(feature = "fixed-point"))]
const fn val32(x: f32) -> Val32 {
    x
}
#[cfg(feature = "fixed-point")]
const fn val32(x: f32) -> Val32 {
    x as i32
}
/// An `opus_val16` returned by `op_pvq_search` as the `opus_val32` `Ryy`.
#[cfg(not(feature = "fixed-point"))]
const fn yy32(x: Val16) -> Val32 {
    x
}
#[cfg(feature = "fixed-point")]
const fn yy32(x: Val16) -> Val32 {
    x as i32
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// The Rust mode matching the oracle's `cb_mode(fs, qext)`.
fn mode(fs: i32, qext: bool) -> CeltMode {
    let base: &CeltMode = match fs {
        48000 => &MODE48000_960_120,
        #[cfg(feature = "qext")]
        96000 => &opusorus::celt::static_modes::MODE96000_1920_240,
        _ => panic!("unsupported fs {fs}"),
    };
    #[cfg(feature = "qext")]
    if qext {
        return opusorus::celt::modes::compute_qext_mode(base);
    }
    assert!(!qext);
    base.clone()
}

/// Modes available in this build: `(fs, qext_mode)`.
fn all_modes() -> Vec<(i32, bool)> {
    let mut v = vec![(48000, false)];
    if cfg!(feature = "qext") {
        v.push((96000, false));
        v.push((48000, true));
        v.push((96000, true));
    }
    v
}

fn rand_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| amp * rng.f32_sym()).collect()
}

/// A random unit-norm-ish vector with a random structure (sparse, peaky, flat, tiny, zero).
fn shape_vec(rng: &mut Rng, n: usize) -> Vec<Norm> {
    shape_vec_f(rng, n).into_iter().map(nrm).collect()
}

fn shape_vec_f(rng: &mut Rng, n: usize) -> Vec<f32> {
    let kind = rng.range_i32(0, 9);
    let mut v: Vec<f32> = match kind {
        0 => vec![0.0; n],
        1 => (0..n).map(|_| 1e-20 * rng.f32_sym()).collect(),
        2 => {
            let mut v = vec![0.0; n];
            v[rng.range_i32(0, n as i32 - 1) as usize] =
                if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            v
        }
        3 => (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    rng.f32_sym()
                } else {
                    0.01 * rng.f32_sym()
                }
            })
            .collect(),
        _ => rand_vec(rng, n, 1.0),
    };
    if kind >= 3 {
        let e: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if e > 0.0 {
            for x in &mut v {
                *x /= e;
            }
        }
    }
    v
}

/// A realistic MDCT-like spectrum of `c` channels of `n` bins with a random tilt, some silent
/// bands and optionally identical/near-identical channels.
fn spectrum(rng: &mut Rng, m: &CeltMode, lm: i32, c: usize) -> Vec<Sig> {
    spectrum_f(rng, m, lm, c).into_iter().map(sig).collect()
}

fn spectrum_f(rng: &mut Rng, m: &CeltMode, lm: i32, c: usize) -> Vec<f32> {
    let n = (m.short_mdct_size << lm) as usize;
    let mut x = vec![0.0f32; c * n];
    let tilt = 0.5 + 3.0 * (rng.next_u32() as f32 / u32::MAX as f32);
    let stereo_kind = rng.range_i32(0, 5);
    for ch in 0..c {
        for i in 0..m.eff_ebands as usize {
            let lo = (i32::from(m.e_bands[i]) << lm) as usize;
            let hi = (i32::from(m.e_bands[i + 1]) << lm) as usize;
            let silent = rng.range_i32(0, 12) == 0;
            let amp = if silent {
                0.0
            } else {
                1000.0 * (-(tilt * i as f32 / 8.0)).exp() * (0.2 + rng.f32_sym().abs())
            };
            let peaky = rng.range_i32(0, 3) == 0;
            for j in lo..hi {
                let v = if peaky && !(j - lo).is_multiple_of(4) {
                    0.02 * amp * rng.f32_sym()
                } else {
                    amp * rng.f32_sym()
                };
                x[ch * n + j] = v;
            }
        }
    }
    if c == 2 {
        match stereo_kind {
            0 => {
                let (a, b) = x.split_at_mut(n);
                b.copy_from_slice(a);
            }
            1 => {
                let (a, b) = x.split_at_mut(n);
                for (bb, aa) in b.iter_mut().zip(a.iter()) {
                    *bb = 0.9 * *aa + 0.01 * *bb;
                }
            }
            2 => {
                let (a, b) = x.split_at_mut(n);
                for (bb, aa) in b.iter_mut().zip(a.iter()) {
                    *bb = -*aa;
                }
            }
            _ => {}
        }
    }
    x
}

/// `(X, bandE)` normalised from a spectrum (via the already-verified Rust band functions).
fn normalised(m: &CeltMode, lm: i32, c: usize, freq: &[Sig]) -> (Vec<Norm>, Vec<Ener>) {
    let nb = m.nb_ebands as usize;
    let eff = m.eff_ebands;
    let mut band_e = vec![Ener::default(); 2 * 21.max(nb)];
    bands::compute_band_energies(m, freq, &mut band_e, eff, c as i32, lm);
    let mut x = vec![Norm::default(); freq.len()];
    bands::normalise_bands(m, freq, &mut x, &band_e, eff, c as i32, 1 << lm);
    (x, band_e)
}

/// A random `tf_res` entry from the TF select table.
fn tf_res(rng: &mut Rng, lm: i32, transient: bool) -> Vec<i32> {
    let sel = rng.range_i32(0, 1);
    (0..21)
        .map(|_| {
            let flag = rng.range_i32(0, 1);
            i32::from(
                celtc::TF_SELECT_TABLE[lm as usize]
                    [(4 * i32::from(transient) + 2 * sel + flag) as usize],
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// celt.c
// ---------------------------------------------------------------------------------------------

#[test]
fn celt_tables_match() {
    let (tf, trim, spread, tapset) = c::celt_tables();
    assert_eq!(celtc::TF_SELECT_TABLE, tf);
    assert_eq!(celtc::TRIM_ICDF, trim);
    assert_eq!(celtc::SPREAD_ICDF, spread);
    assert_eq!(celtc::TAPSET_ICDF, tapset);
    assert_v("eMeans", &qb::E_MEANS, &c::e_means());
    for rate in [8000, 12000, 16000, 24000, 48000, 96000] {
        if rate == 96000 && !cfg!(feature = "qext") {
            continue;
        }
        assert_eq!(
            celtc::resampling_factor(rate),
            c::resampling_factor(rate),
            "rate {rate}"
        );
    }
    assert_eq!(celtc::opus_strerror(0), "success");
    for e in -8..=1 {
        let o = c::strerror(e);
        assert_eq!(celtc::opus_strerror(e), o, "strerror({e})");
    }
    assert!(celtc::opus_get_version_string().starts_with("libopus "));
    assert_eq!(
        celtc::opus_get_version_string().contains("-fixed"),
        opusorus_oracle::api::version_string().contains("-fixed")
    );
    assert_eq!(
        celtc::opus_get_version_string().contains("-fixed"),
        cfg!(feature = "fixed-point")
    );
}

#[test]
fn init_caps_matches() {
    for (fs, q) in all_modes() {
        let m = mode(fs, q);
        for lm in 0..=3 {
            for cc in 1..=2 {
                let mut r = vec![0i32; 21];
                let mut o = vec![0i32; 21];
                celtc::init_caps(&m, &mut r, lm, cc);
                c::init_caps(fs, q, &mut o, lm, cc);
                let nb = m.nb_ebands as usize;
                assert_slice_eq(
                    &format!("init_caps fs={fs} q={q} lm={lm} c={cc}"),
                    &r[..nb],
                    &o[..nb],
                );
            }
        }
    }
}

#[test]
fn comb_filter_matches() {
    let mut rng = Rng::new(0xC0B);
    let hist = 2 * 1024 + 8;
    let mut cases = vec![(48000, 120)];
    if cfg!(feature = "qext") {
        cases.push((96000, 240));
    }
    for (fs, overlap) in cases {
        for it in 0..3000 {
            let n: i32 = *[overlap, 2 * overlap, 480, 960, 1920, 64, 240, 16]
                .get(rng.range_i32(0, 7) as usize)
                .unwrap_or(&120);
            let n = if overlap == 240 {
                n.max(240) & !1
            } else {
                n.max(overlap)
            };
            let tsel = |rng: &mut Rng| match rng.range_i32(0, 4) {
                0 => 0,
                1 => rng.range_i32(15, 40),
                // COMBFILTER_MAXPERIOD-2 is the largest period the codec uses.
                _ => rng.range_i32(15, 1022),
            };
            let gsel = |rng: &mut Rng| match rng.range_i32(0, 3) {
                0 => q15(0.0),
                _ => q15(0.8 * rng.f32_sym()),
            };
            let mut p = c::CombParams {
                t0: tsel(&mut rng),
                t1: tsel(&mut rng),
                n,
                g0: gsel(&mut rng),
                g1: gsel(&mut rng),
                tapset0: rng.range_i32(0, 2),
                tapset1: rng.range_i32(0, 2),
                fs,
                use_window: true,
                overlap,
            };
            if it % 7 == 0 {
                p.t1 = p.t0;
                p.g1 = p.g0;
                p.tapset1 = p.tapset0;
            }
            if it % 11 == 0 {
                // Decoder's first call: no window, overlap 0.
                p.use_window = false;
                p.overlap = 0;
            }
            let m = mode(fs, false);
            let window: &[CeltCoef] = if p.use_window { &m.window } else { &[] };
            let x: Vec<Val32> = rand_vec(&mut rng, hist + n as usize, 1000.0)
                .into_iter()
                .map(sig)
                .collect();
            // Separate buffers.
            let y0: Vec<Val32> = rand_vec(&mut rng, n as usize, 1.0)
                .into_iter()
                .map(sig)
                .collect();
            let (mut yr, mut yc) = (y0.clone(), y0.clone());
            let mut xc = x.clone();
            celtc::comb_filter(
                &mut yr, &x, hist, p.t0, p.t1, p.n, p.g0, p.g1, p.tapset0, p.tapset1, window,
                p.overlap,
            );
            c::comb_filter(&mut yc, &mut xc, hist, p);
            assert_v(&format!("comb_filter {it} {p:?}"), &yr, &yc);
            // In place.
            let (mut br, mut bc) = (x.clone(), x.clone());
            celtc::comb_filter_inplace(
                &mut br, hist, p.t0, p.t1, p.n, p.g0, p.g1, p.tapset0, p.tapset1, window, p.overlap,
            );
            c::comb_filter_inplace(&mut bc, hist, p);
            assert_v(&format!("comb_filter_inplace {it} {p:?}"), &br, &bc);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// vq.c
// ---------------------------------------------------------------------------------------------

/// A random `(N, K)` pair the codec can actually code: `N` is a band size (possibly split, at
/// `LM` in `-1..=3`) and `K = get_pulses(q)` for a `q` its pulse cache allows.
fn valid_nk(rng: &mut Rng) -> (i32, i32) {
    let m = &MODE48000_960_120;
    let nb = m.nb_ebands;
    loop {
        let lm = rng.range_i32(-1, 3);
        let i = rng.range_i32(0, nb - 1);
        let w = i32::from(m.e_bands[i as usize + 1] - m.e_bands[i as usize]);
        let n = if lm < 0 { w >> 1 } else { w << lm };
        if n < 2 {
            continue;
        }
        let idx = m.cache.index[((lm + 1) * nb + i) as usize];
        if idx < 0 {
            continue;
        }
        let maxq = i32::from(m.cache.bits[idx as usize]);
        if maxq < 1 {
            continue;
        }
        let q = rng.range_i32(1, maxq);
        return (n, rate::get_pulses(q));
    }
}

const BAND_NS: [i32; 16] = [2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 22, 24, 32, 44, 88, 176];

#[test]
fn vq_primitives_match() {
    let mut rng = Rng::new(0x51);
    for it in 0..20000 {
        let n = BAND_NS[rng.range_i32(0, 15) as usize];
        let nu = n as usize;
        let x = shape_vec(&mut rng, nu);
        // exp_rotation
        let stride = *[1, 2, 4, 8].get(rng.range_i32(0, 3) as usize).unwrap_or(&1);
        let k = rng.range_i32(1, 2 * n + 2);
        let spread = rng.range_i32(0, 3);
        let dir = if rng.next_u32() & 1 == 0 { 1 } else { -1 };
        if n % stride == 0 {
            let (mut a, mut b) = (x.clone(), x.clone());
            vq::exp_rotation(&mut a, n, dir, stride, k, spread);
            c::exp_rotation(&mut b, n, dir, stride, k, spread);
            assert_v(&format!("exp_rotation {it}"), &a, &b);
        }
        // exp_rotation1
        {
            let (cc, ss) = (q15(rng.f32_sym()), q15(rng.f32_sym()));
            let st = rng.range_i32(1, n.max(2) - 1);
            let (mut a, mut b) = (x.clone(), x.clone());
            vq::exp_rotation1(&mut a, n, st, cc, ss);
            c::exp_rotation1(&mut b, n, st, cc, ss);
            assert_v(&format!("exp_rotation1 {it}"), &a, &b);
        }
        // op_pvq_search
        {
            let k = if n <= 32 {
                rng.range_i32(1, 3 * n + 10).min(128)
            } else {
                rng.range_i32(1, 40)
            };
            let (mut a, mut b) = (x.clone(), x.clone());
            let (mut ia, mut ib) = (vec![0i32; nu], vec![0i32; nu]);
            let ya = vq::op_pvq_search(&mut a, &mut ia, k, n);
            let yb = c::op_pvq_search(&mut b, &mut ib, k, n);
            assert_v1(&format!("pvq yy {it}"), ya, yb);
            assert_slice_eq(&format!("pvq iy {it}"), &ia, &ib);
            assert_v(&format!("pvq x {it}"), &a, &b);
            // normalise_residual / extract_collapse_mask on the result
            let gain = q31(rng.f32_sym().abs() + 0.1);
            let (mut na, mut nb) = (vec![Norm::default(); nu], vec![Norm::default(); nu]);
            vq::normalise_residual(&ia, &mut na, n, yy32(ya), gain, 0);
            c::normalise_residual(&mut ib, &mut nb, n, yy32(yb), gain, 0);
            assert_v(&format!("normalise_residual {it}"), &na, &nb);
            for bl in [1, 2, 4, 8] {
                if n % bl == 0 {
                    assert_eq!(
                        vq::extract_collapse_mask(&ia, n, bl),
                        c::extract_collapse_mask(&mut ib, n, bl),
                        "collapse mask {it}"
                    );
                }
            }
        }
        // renormalise_vector (the codec's gains are positive Q31 values in fixed point)
        {
            let gain = if cfg!(feature = "fixed-point") {
                q31(rng.f32_sym().abs())
            } else {
                q31(2.0 * rng.f32_sym())
            };
            let (mut a, mut b) = (x.clone(), x.clone());
            vq::renormalise_vector(&mut a, n, gain);
            c::renormalise_vector(&mut b, n, gain);
            assert_v(&format!("renormalise {it}"), &a, &b);
        }
        // celt_inner_prod_norm(_shift). The codec only calls celt_inner_prod_norm on vectors
        // scaled down to Q14 (its `int` sum would overflow on Q24 vectors).
        {
            let y = shape_vec(&mut rng, nu);
            let q14v = |v: &[Norm]| -> Vec<Norm> {
                v.iter()
                    .map(|&e| {
                        if cfg!(feature = "fixed-point") {
                            e / 1024 as Norm
                        } else {
                            e
                        }
                    })
                    .collect()
            };
            let (xs, ys) = (q14v(&x), q14v(&y));
            assert_v1(
                &format!("inner_prod_norm {it}"),
                vq::celt_inner_prod_norm(&xs, &ys, nu),
                c::celt_inner_prod_norm(&xs, &ys, nu),
            );
            assert_v1(
                &format!("inner_prod_norm_shift {it}"),
                vq::celt_inner_prod_norm_shift(&x, &y, nu),
                c::celt_inner_prod_norm_shift(&x, &y, nu),
            );
        }
        // stereo_itheta
        {
            let y = shape_vec(&mut rng, nu);
            for stereo in [false, true] {
                assert_eq!(
                    vq::stereo_itheta(&x, &y, stereo, n),
                    c::stereo_itheta(&x, &y, stereo, n),
                    "itheta {it} {stereo}"
                );
            }
        }
    }
}

#[cfg(feature = "qext")]
#[test]
fn vq_qext_search_match() {
    let mut rng = Rng::new(0x5E);
    for it in 0..20000 {
        let n = BAND_NS[rng.range_i32(0, 15) as usize];
        let nu = n as usize;
        let x = shape_vec(&mut rng, nu);
        let k = rng.range_i32(1, 40);
        let extra = rng.range_i32(2, 12);
        let up = (1 << extra) - 1;
        // The codec passes `IMAX(0, extra_bits-7)` (only used by the fixed-point build).
        let shift = (extra - 7).max(0);
        let (mut ia, mut ib) = (vec![0i32; nu + 3], vec![0i32; nu + 3]);
        let (mut ua, mut ub) = (vec![0i32; nu], vec![0i32; nu]);
        if n == 2 {
            let mut refa = 0;
            let ya = vq::op_pvq_search_n2(&x, &mut ia, &mut ua, k, up, &mut refa, shift);
            let (yb, refb) = c::op_pvq_search_n2(&x, &mut ib, &mut ub, k, up, shift);
            assert_v1(&format!("n2 yy {it}"), ya, yb);
            assert_eq!(refa, refb, "n2 refine {it}");
            assert_slice_eq("n2 iy", &ia[..2], &ib[..2]);
            assert_slice_eq("n2 up_iy", &ua, &ub);
        } else {
            let (mut ra, mut rb) = (vec![0i32; nu], vec![0i32; nu]);
            let ya = vq::op_pvq_search_extra(&x, &mut ia, &mut ua, k, up, &mut ra, n, shift);
            let yb = c::op_pvq_search_extra(&x, &mut ib, &mut ub, k, up, &mut rb, n, shift);
            assert_v1(&format!("extra yy {it}"), ya, yb);
            assert_slice_eq(&format!("extra iy {it}"), &ia[..nu], &ib[..nu]);
            assert_slice_eq(&format!("extra up_iy {it}"), &ua, &ub);
            assert_slice_eq(&format!("extra refine {it}"), &ra, &rb);
        }
    }
}

/// `alg_quant` on fresh coders, returning the same layout as the oracle.
fn rust_alg_quant(
    x: &mut [Norm],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    buf: &mut [u8],
    gain: Val32,
    resynth: bool,
    ext_buf: &mut [u8],
    extra_bits: i32,
) -> c::VqOut {
    let mut enc = EcEnc::new(buf);
    let mut ext = EcEnc::new(ext_buf);
    #[cfg(feature = "qext")]
    let cm = vq::alg_quant(
        x, n, k, spread, b, &mut enc, gain, resynth, &mut ext, extra_bits,
    );
    #[cfg(not(feature = "qext"))]
    let cm = {
        let _ = extra_bits;
        vq::alg_quant(x, n, k, spread, b, &mut enc, gain, resynth)
    };
    let t = enc.tell_frac();
    let te = ext.tell_frac();
    enc.done();
    ext.done();
    [
        cm,
        t,
        enc.rng,
        enc.error as u32,
        te,
        ext.rng,
        ext.error as u32,
    ]
}

fn rust_alg_unquant(
    x: &mut [Norm],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    buf: &[u8],
    gain: Val32,
    ext_buf: &[u8],
    extra_bits: i32,
) -> c::VqOut {
    let mut dec = EcDec::new(buf);
    #[cfg_attr(
        not(feature = "qext"),
        allow(unused_mut, reason = "only used mutably with qext")
    )]
    let mut ext = EcDec::new(ext_buf);
    #[cfg(feature = "qext")]
    let cm = vq::alg_unquant(x, n, k, spread, b, &mut dec, gain, &mut ext, extra_bits);
    #[cfg(not(feature = "qext"))]
    let cm = {
        let _ = extra_bits;
        vq::alg_unquant(x, n, k, spread, b, &mut dec, gain)
    };
    [
        cm,
        dec.tell_frac(),
        dec.rng,
        dec.error as u32,
        ext.tell_frac(),
        ext.rng,
        ext.error as u32,
    ]
}

#[test]
fn alg_quant_unquant_match() {
    let mut rng = Rng::new(0xA1);
    for it in 0..100000 {
        let (n, k) = valid_nk(&mut rng);
        let nu = n as usize;
        let bl = *[1, 2, 4, 8].get(rng.range_i32(0, 3) as usize).unwrap_or(&1);
        let bl = if n % bl == 0 { bl } else { 1 };
        let spread = rng.range_i32(0, 3);
        let gain = q31(0.25 + rng.f32_sym().abs());
        let resynth = rng.next_u32() & 1 == 0;
        let extra_bits = if cfg!(feature = "qext") && rng.range_i32(0, 2) == 0 {
            rng.range_i32(0, 12)
        } else {
            0
        };
        let size = rng.range_i32(8, 64) as usize;
        let ext_size = if extra_bits > 0 {
            rng.range_i32(0, 64) as usize
        } else {
            0
        };
        let x = shape_vec(&mut rng, nu);
        let (mut xa, mut xb) = (x.clone(), x.clone());
        let (mut ba, mut bb) = (vec![0u8; size], vec![0u8; size]);
        let (mut ea, mut eb) = (vec![0u8; ext_size], vec![0u8; ext_size]);
        let ra = rust_alg_quant(
            &mut xa, n, k, spread, bl, &mut ba, gain, resynth, &mut ea, extra_bits,
        );
        let rb = c::alg_quant(
            &mut xb, n, k, spread, bl, &mut bb, gain, resynth, &mut eb, extra_bits,
        );
        let what = format!("alg_quant {it} n={n} k={k} B={bl} spread={spread} extra={extra_bits}");
        assert_eq!(ra, rb, "{what}");
        assert_v(&what, &xa, &xb);
        assert_slice_eq(&what, &ba, &bb);
        assert_slice_eq(&what, &ea, &eb);
        // Decode (the produced bytes, sometimes garbage).
        if it % 5 == 0 {
            rng.fill_bytes(&mut ba);
            rng.fill_bytes(&mut ea);
            bb.copy_from_slice(&ba);
            eb.copy_from_slice(&ea);
        }
        let (mut da, mut db) = (vec![Norm::default(); nu], vec![Norm::default(); nu]);
        let ua = rust_alg_unquant(&mut da, n, k, spread, bl, &ba, gain, &ea, extra_bits);
        let ub = c::alg_unquant(
            &mut db, n, k, spread, bl, &mut bb, gain, &mut eb, extra_bits,
        );
        assert_eq!(ua, ub, "unquant {what}");
        assert_v(&format!("unquant {what}"), &da, &db);
    }
}

#[cfg(feature = "qext")]
#[test]
fn cubic_quant_unquant_match() {
    let mut rng = Rng::new(0xCB);
    for it in 0..20000 {
        let n = BAND_NS[rng.range_i32(0, 15) as usize];
        let nu = n as usize;
        let res = rng.range_i32(0, 14);
        let bl = *[1, 2, 4, 8].get(rng.range_i32(0, 3) as usize).unwrap_or(&1);
        let gain = q31(0.25 + rng.f32_sym().abs());
        let resynth = rng.next_u32() & 1 == 0;
        let size = rng.range_i32(2, 300) as usize;
        let x = shape_vec(&mut rng, nu);
        let (mut xa, mut xb) = (x.clone(), x.clone());
        let (mut ba, mut bb) = (vec![0u8; size], vec![0u8; size]);
        let ra = {
            let mut enc = EcEnc::new(&mut ba);
            let cm = vq::cubic_quant(&mut xa, n, res, bl, &mut enc, gain, resynth);
            let t = enc.tell_frac();
            enc.done();
            [cm, t, enc.rng, enc.error as u32]
        };
        let rb = c::cubic_quant(&mut xb, n, res, bl, &mut bb, gain, resynth);
        let what = format!("cubic_quant {it} n={n} res={res} B={bl}");
        assert_eq!(ra, rb, "{what}");
        assert_v(&what, &xa, &xb);
        assert_slice_eq(&what, &ba, &bb);
        if it % 4 == 0 {
            rng.fill_bytes(&mut ba);
            bb.copy_from_slice(&ba);
        }
        let (mut da, mut db) = (vec![Norm::default(); nu], vec![Norm::default(); nu]);
        let ua = {
            let mut dec = EcDec::new(&ba);
            let cm = vq::cubic_unquant(&mut da, n, res, bl, &mut dec, gain);
            [cm, dec.tell_frac(), dec.rng, dec.error as u32]
        };
        let ub = c::cubic_unquant(&mut db, n, res, bl, &mut bb, gain);
        assert_eq!(ua, ub, "un{what}");
        assert_v(&format!("un{what}"), &da, &db);
    }
}

// ---------------------------------------------------------------------------------------------
// bands.c primitives
// ---------------------------------------------------------------------------------------------

#[test]
fn bands_integer_helpers_match() {
    // The codec only evaluates bitexact_cos() on multiples of 16384/qn with qn <= 256, strictly
    // inside (0, 16384) (the C `celt_sig_assert` rejects smaller arguments).
    for x in 64..=16320i16 {
        assert_eq!(
            bands::bitexact_cos(x),
            c::bitexact_cos(x),
            "bitexact_cos({x})"
        );
    }
    let mut rng = Rng::new(0xB7);
    for _ in 0..100000 {
        let (s, co) = (rng.range_i32(1, 32767), rng.range_i32(1, 32767));
        assert_eq!(bands::bitexact_log2tan(s, co), c::bitexact_log2tan(s, co));
        let seed = rng.next_u32();
        assert_eq!(bands::celt_lcg_rand(seed), c::celt_lcg_rand(seed));
    }
    for n in 1..=176 {
        for b in (-50..3000).step_by(7) {
            for offset in [-16, -4, 0, 5, 20, 40] {
                for pulse_cap in [0, 16, 40, 60] {
                    for stereo in [false, true] {
                        assert_eq!(
                            bands::compute_qn(n, b, offset, pulse_cap, stereo),
                            c::compute_qn(n, b, offset, pulse_cap, stereo),
                            "compute_qn({n},{b},{offset},{pulse_cap},{stereo})"
                        );
                    }
                }
            }
        }
    }
    // hysteresis_decision (the encoder's spread/tapset thresholds)
    let thr = [q14(-0.25), q14(0.5), q14(1.0)];
    let hys = [q14(0.05), q14(0.1), q14(0.05)];
    for _ in 0..20000 {
        let v = q14(1.5 * rng.f32_sym());
        let n = rng.range_i32(1, 3);
        let prev = rng.range_i32(0, n);
        assert_eq!(
            bands::hysteresis_decision(v, &thr, &hys, n, prev),
            c::hysteresis_decision(v, &thr, &hys, n, prev)
        );
    }
}

#[test]
fn band_vector_helpers_match() {
    let mut rng = Rng::new(0xBA);
    for it in 0..20000 {
        let n = BAND_NS[rng.range_i32(0, 15) as usize];
        let nu = n as usize;
        let x = shape_vec(&mut rng, nu);
        let y = shape_vec(&mut rng, nu);
        // haar1
        for stride in [1, 2, 4, 8] {
            if n % (2 * stride) == 0 {
                let (mut a, mut b) = (x.clone(), x.clone());
                bands::haar1(&mut a, n / stride, stride);
                c::haar1(&mut b, n / stride, stride);
                assert_v(&format!("haar1 {it}"), &a, &b);
            }
        }
        // (de)interleave_hadamard
        for stride in [2, 4, 8, 16] {
            if n % stride == 0 {
                for h in [false, true] {
                    let (mut a, mut b) = (x.clone(), x.clone());
                    bands::deinterleave_hadamard(&mut a, n / stride, stride, h);
                    c::deinterleave_hadamard(&mut b, n / stride, stride, h);
                    assert_v("deinterleave", &a, &b);
                    bands::interleave_hadamard(&mut a, n / stride, stride, h);
                    c::interleave_hadamard(&mut b, n / stride, stride, h);
                    assert_v("interleave", &a, &b);
                }
            }
        }
        // stereo split / merge / intensity
        {
            let (mut xa, mut ya, mut xb, mut yb) = (x.clone(), y.clone(), x.clone(), y.clone());
            bands::stereo_split(&mut xa, &mut ya, n);
            c::stereo_split(&mut xb, &mut yb, n);
            assert_v("split x", &xa, &xb);
            assert_v("split y", &ya, &yb);
            let mid = if it % 13 == 0 {
                q31(0.0)
            } else {
                q31(rng.f32_sym().abs())
            };
            bands::stereo_merge(&mut xa, &mut ya, mid, n);
            c::stereo_merge(&mut xb, &mut yb, mid, n);
            assert_v("merge x", &xa, &xb);
            assert_v("merge y", &ya, &yb);
            let band_e: Vec<Ener> = rand_vec(&mut rng, 42, 100.0)
                .into_iter()
                .map(|v| sig(v.abs()))
                .collect();
            let id = rng.range_i32(0, 20);
            let (mut xa, mut xb) = (x.clone(), x.clone());
            bands::intensity_stereo(&MODE48000_960_120, &mut xa, &y, &band_e, id, n);
            c::intensity_stereo(48000, &mut xb, &y, &band_e, id, n);
            assert_v("intensity", &xa, &xb);
        }
        let (ex, ey) = (
            sig(rng.f32_sym().abs() * 1e3),
            sig(rng.f32_sym().abs() * 1e3),
        );
        assert_v(
            "weights",
            &bands::compute_channel_weights(ex, ey),
            &c::compute_channel_weights(ex, ey),
        );
    }
}

#[test]
fn band_energies_and_normalisation_match() {
    let mut rng = Rng::new(0xE0);
    for (fs, q) in all_modes() {
        let m = mode(fs, q);
        let nb = m.nb_ebands;
        for it in 0..1000 {
            let lm = rng.range_i32(0, 3);
            let cc = rng.range_i32(1, 2);
            let n = (m.short_mdct_size << lm) as usize;
            let freq: Vec<Sig> = if q {
                rand_vec(&mut rng, cc as usize * n, 100.0)
                    .into_iter()
                    .map(sig)
                    .collect()
            } else {
                spectrum(&mut rng, &m, lm, cc as usize)
            };
            let end = rng.range_i32(1, m.eff_ebands);
            let _ = nb;
            let (mut ea, mut eb) = (vec![Ener::default(); 42], vec![Ener::default(); 42]);
            bands::compute_band_energies(&m, &freq, &mut ea, end, cc, lm);
            c::compute_band_energies(fs, q, &freq, &mut eb, end, cc, lm);
            assert_v(&format!("bandE {fs} {q} {it}"), &ea, &eb);
            let (mut xa, mut xb) = (
                vec![Norm::default(); freq.len()],
                vec![Norm::default(); freq.len()],
            );
            bands::normalise_bands(&m, &freq, &mut xa, &ea, end, cc, 1 << lm);
            c::normalise_bands(fs, q, &freq, &mut xb, &eb, end, cc, 1 << lm);
            assert_v("normalise", &xa, &xb);
            // amp2Log2
            let eff_end = rng.range_i32(1, end);
            let (mut la, mut lb) = (vec![Glog::default(); 42], vec![Glog::default(); 42]);
            qb::amp2_log2(&m, eff_end, end, &ea, &mut la, cc);
            c::amp2log2(fs, q, eff_end, end, &mut eb, &mut lb, cc);
            assert_v("amp2Log2", &la, &lb);
            // denormalise_bands (one channel)
            let start = if rng.range_i32(0, 3) == 0 {
                rng.range_i32(0, end - 1)
            } else {
                0
            };
            let downsample = *[1, 1, 2, 3, 6]
                .get(rng.range_i32(0, 4) as usize)
                .unwrap_or(&1);
            let silence = rng.range_i32(0, 10) == 0;
            let mut ble = la.clone();
            if it % 9 == 0 {
                ble[0] = glog(40.0);
            }
            let (mut fa, mut fb) = (vec![sig(1.0); n], vec![sig(1.0); n]);
            bands::denormalise_bands(
                &m,
                &xa,
                &mut fa,
                &ble,
                start,
                end,
                1 << lm,
                downsample,
                silence,
            );
            c::denormalise_bands(
                fs,
                q,
                &xa,
                &mut fb,
                &ble,
                start,
                end,
                1 << lm,
                downsample,
                silence,
            );
            assert_v(&format!("denormalise {it}"), &fa, &fb);
        }
    }
}

#[test]
fn anti_collapse_and_spreading_match() {
    let mut rng = Rng::new(0xAC);
    let m = &MODE48000_960_120;
    for it in 0..4000 {
        let lm = rng.range_i32(0, 3);
        let cc = rng.range_i32(1, 2);
        let n = (m.short_mdct_size << lm) as usize;
        let freq = spectrum(&mut rng, m, lm, cc as usize);
        let (x, _) = normalised(m, lm, cc as usize, &freq);
        // anti_collapse
        let start = rng.range_i32(0, 5);
        let end = rng.range_i32(start + 1, 21);
        let mut cm = vec![0u8; 42];
        rng.fill_bytes(&mut cm);
        let log_e: Vec<Glog> = rand_vec(&mut rng, 42, 10.0).into_iter().map(glog).collect();
        let p1: Vec<Glog> = rand_vec(&mut rng, 42, 10.0).into_iter().map(glog).collect();
        let p2: Vec<Glog> = rand_vec(&mut rng, 42, 10.0).into_iter().map(glog).collect();
        let pulses: Vec<i32> = (0..21).map(|_| rng.range_i32(0, 400)).collect();
        let seed = rng.next_u32();
        let encode = rng.next_u32() & 1 == 0;
        let (mut xa, mut xb) = (x.clone(), x.clone());
        bands::anti_collapse(
            m, &mut xa, &cm, lm, cc, n as i32, start, end, &log_e, &p1, &p2, &pulses, seed, encode,
        );
        c::anti_collapse(
            48000, false, &mut xb, &mut cm, lm, cc, n as i32, start, end, &log_e, &p1, &p2,
            &pulses, seed, encode,
        );
        assert_v(&format!("anti_collapse {it}"), &xa, &xb);
        // spreading_decision (state evolves over several frames)
        let mut sa = [
            rng.range_i32(0, 400),
            rng.range_i32(0, 30),
            rng.range_i32(0, 2),
        ];
        let mut sb = sa;
        let weights: Vec<i32> = (0..21).map(|_| rng.range_i32(1, 32)).collect();
        let mut last = rng.range_i32(0, 3);
        for f in 0..4 {
            let freq = spectrum(&mut rng, m, lm, cc as usize);
            let (x, _) = normalised(m, lm, cc as usize, &freq);
            let end = if f == 0 { 21 } else { rng.range_i32(1, 21) };
            let upd = rng.next_u32() & 1 == 0;
            let (mut a0, mut a1, mut a2) = (sa[0], sa[1], sa[2]);
            let da = bands::spreading_decision(
                m,
                &x,
                &mut a0,
                last,
                &mut a1,
                &mut a2,
                upd,
                end,
                cc,
                1 << lm,
                &weights,
            );
            sa = [a0, a1, a2];
            let db =
                c::spreading_decision(48000, &x, &mut sb, last, upd, end, cc, 1 << lm, &weights);
            assert_eq!(da, db, "spreading {it}/{f}");
            assert_eq!(sa, sb, "spreading state {it}/{f}");
            last = da;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// quant_all_bands
// ---------------------------------------------------------------------------------------------

/// Array inputs of quant_all_bands.
#[derive(Clone)]
struct QabIn {
    x: Vec<Norm>,
    band_e: Vec<Ener>,
    pulses: Vec<i32>,
    tf_res: Vec<i32>,
    offsets: Vec<i32>,
    cap: Vec<i32>,
    extra_pulses: Vec<i32>,
    seed: u32,
}

/// Rust counterpart of `oracle_quant_all_bands` (same outputs).
fn rust_qab(
    p: &c::QabParams,
    inp: &mut QabIn,
    cm: &mut [u8],
    buf: &mut [u8],
    ext_buf: &mut [u8],
    scratch: &mut BandsScratch,
) -> [i32; 11] {
    let m = mode(p.fs, p.qext_mode);
    let n = (m.short_mdct_size << p.lm) as usize;
    let size = buf.len() as i32;
    let mut intensity = p.intensity;
    let mut dual = i32::from(p.dual_stereo);
    let mut balance = p.balance;
    let mut coded = p.coded_bands;
    let mut fine_quant = [0i32; 32];
    let mut fine_priority = [0i32; 32];
    let (x0, x1) = inp.x.split_at_mut(n);
    let y = if p.c == 2 { Some(&mut x1[..n]) } else { None };
    let mut out = [0i32; 11];

    macro_rules! run {
        ($ec:expr, $ext:expr) => {{
            let ec: &mut EcCoder<'_, '_> = $ec;
            if p.do_alloc {
                let bits = ((size * 8) << 3) - ec.tell_frac() as i32 - 1;
                coded = rate::clt_compute_allocation(
                    &m,
                    p.start,
                    p.end,
                    &inp.offsets,
                    &inp.cap,
                    p.alloc_trim,
                    &mut intensity,
                    &mut dual,
                    bits,
                    &mut balance,
                    &mut inp.pulses,
                    &mut fine_quant,
                    &mut fine_priority,
                    p.c,
                    p.lm,
                    ec,
                    p.prev,
                    p.signal_bandwidth,
                );
            }
            out[10] = ec.tell_frac() as i32;
            #[cfg(feature = "qext")]
            bands::quant_all_bands(
                &m,
                p.start,
                p.end,
                x0,
                y,
                cm,
                &inp.band_e,
                &inp.pulses,
                p.short_blocks,
                p.spread,
                dual != 0,
                intensity,
                &inp.tf_res,
                p.total_bits,
                balance,
                ec,
                p.lm,
                coded,
                &mut inp.seed,
                p.complexity,
                p.disable_inv,
                $ext,
                &inp.extra_pulses,
                p.ext_total_bits,
                if p.has_cap { Some(&inp.cap[..]) } else { None },
                scratch,
            );
            #[cfg(not(feature = "qext"))]
            {
                let _ = $ext;
                bands::quant_all_bands(
                    &m,
                    p.start,
                    p.end,
                    x0,
                    y,
                    cm,
                    &inp.band_e,
                    &inp.pulses,
                    p.short_blocks,
                    p.spread,
                    dual != 0,
                    intensity,
                    &inp.tf_res,
                    p.total_bits,
                    balance,
                    ec,
                    p.lm,
                    coded,
                    &mut inp.seed,
                    p.complexity,
                    p.disable_inv,
                    scratch,
                );
            }
            out[4] = ec.tell_frac() as i32;
        }};
    }

    if p.encode {
        let mut enc = EcEnc::new(buf);
        let mut ext = EcEnc::new(ext_buf);
        if p.prefix_ft > 1 {
            enc.enc_uint(p.prefix_val as u32, p.prefix_ft as u32);
        }
        run!(&mut EcCoder::Enc(&mut enc), &mut EcCoder::Enc(&mut ext));
        out[7] = ext.tell_frac() as i32;
        enc.done();
        ext.done();
        out[5] = enc.rng as i32;
        out[6] = enc.error;
        out[8] = ext.rng as i32;
        out[9] = ext.error;
    } else {
        let mut dec = EcDec::new(buf);
        let mut ext = EcDec::new(ext_buf);
        if p.prefix_ft > 1 {
            let _ = dec.dec_uint(p.prefix_ft as u32);
        }
        run!(&mut EcCoder::Dec(&mut dec), &mut EcCoder::Dec(&mut ext));
        out[7] = ext.tell_frac() as i32;
        out[5] = dec.rng as i32;
        out[6] = dec.error;
        out[8] = ext.rng as i32;
        out[9] = ext.error;
    }
    out[0] = coded;
    out[1] = balance;
    out[2] = intensity;
    out[3] = dual;
    out
}

/// Runs Rust and C on the same inputs and compares everything.
fn check_qab(
    what: &str,
    p: &c::QabParams,
    inp: &QabIn,
    buf: &mut Vec<u8>,
    ext_buf: &mut Vec<u8>,
    scratch: &mut BandsScratch,
) -> QabIn {
    let mut ri = inp.clone();
    let mut ci = inp.clone();
    let (mut rbuf, mut cbuf) = (buf.clone(), buf.clone());
    let (mut rext, mut cext) = (ext_buf.clone(), ext_buf.clone());
    let (mut rcm, mut ccm) = (vec![0xAAu8; 42], vec![0xAAu8; 42]);
    let ro = rust_qab(p, &mut ri, &mut rcm, &mut rbuf, &mut rext, scratch);
    let co = c::quant_all_bands(
        p,
        &mut ci.x,
        &mut ccm,
        &ci.band_e,
        &mut ci.pulses,
        &ci.tf_res,
        &ci.offsets,
        &ci.cap,
        &ci.extra_pulses,
        &mut cbuf,
        &mut cext,
        &mut ci.seed,
    );
    let w = format!("{what} {p:?}");
    assert_eq!(ro, co, "outputs: {w}");
    assert_slice_eq(&format!("bytes: {w}"), &rbuf, &cbuf);
    assert_slice_eq(&format!("ext bytes: {w}"), &rext, &cext);
    assert_slice_eq(&format!("collapse masks: {w}"), &rcm, &ccm);
    assert_eq!(ri.seed, ci.seed, "seed: {w}");
    assert_slice_eq(&format!("pulses: {w}"), &ri.pulses, &ci.pulses);
    assert_v(&format!("X: {w}"), &ri.x, &ci.x);
    *buf = rbuf;
    *ext_buf = rext;
    ri
}

fn qab_case(rng: &mut Rng, fs: i32, qext_mode: bool, scratch: &mut BandsScratch, it: usize) {
    let m = mode(fs, qext_mode);
    let nb = m.nb_ebands;
    let lm = rng.range_i32(0, 3);
    let cc = rng.range_i32(1, 2);
    let n = (m.short_mdct_size << lm) as usize;
    let freq = spectrum(rng, &m, lm, cc as usize);
    let (x, band_e) = normalised(&m, lm, cc as usize, &freq);
    let short_blocks = lm > 0 && rng.range_i32(0, 2) == 0;
    let (start, end) = if qext_mode {
        // The codec codes all extra bands at 96 kHz (NB_QEXT_BANDS) and 2 at 48 kHz, which is
        // the mode's effEBands in both cases.
        let _ = nb;
        (0, m.eff_ebands)
    } else {
        let start = if rng.range_i32(0, 4) == 0 { 17 } else { 0 };
        let end = *[21, 21, 21, 13, 17, 19, 20]
            .get(rng.range_i32(0, 6) as usize)
            .unwrap_or(&21);
        (start, end.max(start + 2))
    };
    let size = match rng.range_i32(0, 4) {
        0 => rng.range_i32(2, 20),
        1 => rng.range_i32(20, 80),
        2 => rng.range_i32(80, 300),
        3 => rng.range_i32(300, 1275),
        _ => 1275,
    } as usize;
    let mut cap = vec![0i32; 21];
    if !qext_mode {
        celtc::init_caps(&m, &mut cap, lm, cc);
    } else {
        for v in &mut cap {
            *v = rng.range_i32(0, 3000);
        }
    }
    let ext_on = cfg!(feature = "qext") && !qext_mode && rng.range_i32(0, 2) == 0;
    let ext_size = if ext_on {
        rng.range_i32(0, 400) as usize
    } else {
        0
    };
    let mut extra_pulses = vec![0i32; 21];
    if ext_on {
        for v in extra_pulses.iter_mut().take(end as usize) {
            *v = if rng.range_i32(0, 2) == 0 {
                0
            } else {
                rng.range_i32(0, 1200)
            };
        }
    }
    let do_alloc = !qext_mode && rng.range_i32(0, 5) != 0;
    let mut pulses = vec![0i32; 21];
    if !do_alloc {
        for v in pulses.iter_mut().take(end as usize) {
            *v = rng.range_i32(0, 1500);
        }
    }
    let offsets: Vec<i32> = (0..21)
        .map(|_| {
            if rng.range_i32(0, 4) == 0 {
                rng.range_i32(0, 60)
            } else {
                0
            }
        })
        .collect();
    let p = c::QabParams {
        encode: true,
        fs,
        qext_mode,
        start,
        end,
        c: cc,
        lm,
        short_blocks,
        spread: rng.range_i32(0, 3),
        dual_stereo: cc == 2 && rng.range_i32(0, 3) == 0,
        intensity: rng.range_i32(start, end),
        total_bits: (size as i32 * 64) - if rng.next_u32() & 1 == 0 { 0 } else { 8 },
        balance: if do_alloc {
            0
        } else {
            rng.range_i32(-500, 500)
        },
        coded_bands: if do_alloc {
            0
        } else {
            rng.range_i32(start + 1, end)
        },
        complexity: if rng.range_i32(0, 2) == 0 {
            10
        } else {
            rng.range_i32(0, 10)
        },
        disable_inv: rng.range_i32(0, 4) == 0,
        do_alloc,
        alloc_trim: rng.range_i32(0, 10),
        prev: if rng.next_u32() & 1 == 0 {
            0
        } else {
            rng.range_i32(start + 1, end)
        },
        signal_bandwidth: rng.range_i32(start, end - 1),
        ext_total_bits: ext_size as i32 * 64 - if rng.next_u32() & 1 == 0 { 0 } else { 8 },
        has_cap: rng.next_u32() & 1 == 0,
        prefix_ft: if rng.range_i32(0, 3) == 0 {
            rng.range_i32(2, 5000)
        } else {
            0
        },
        prefix_val: 0,
    };
    let mut p = p;
    if p.prefix_ft > 1 {
        p.prefix_val = rng.range_i32(0, p.prefix_ft - 1);
    }
    let inp = QabIn {
        x,
        band_e,
        pulses,
        tf_res: tf_res(rng, lm, short_blocks),
        offsets,
        cap,
        extra_pulses,
        seed: rng.next_u32(),
    };
    let mut buf = vec![0u8; size];
    let mut ext_buf = vec![0u8; ext_size];
    let _ = check_qab(
        &format!("enc {it}"),
        &p,
        &inp,
        &mut buf,
        &mut ext_buf,
        scratch,
    );

    // Decode what was produced (sometimes corrupted), starting from garbage X.
    let mut pd = p.clone();
    pd.encode = false;
    let mut di = inp;
    di.x = rand_vec(rng, cc as usize * n, 3.0)
        .into_iter()
        .map(nrm)
        .collect();
    if it % 7 == 3 {
        rng.fill_bytes(&mut buf);
        rng.fill_bytes(&mut ext_buf);
    }
    let _ = check_qab(
        &format!("dec {it}"),
        &pd,
        &di,
        &mut buf,
        &mut ext_buf,
        scratch,
    );
}

#[test]
fn quant_all_bands_48k_matches() {
    let mut rng = Rng::new(0x0AB);
    let mut scratch = BandsScratch::new();
    for it in 0..30000 {
        qab_case(&mut rng, 48000, false, &mut scratch, it);
    }
}

#[cfg(feature = "qext")]
#[test]
fn quant_all_bands_qext_matches() {
    let mut rng = Rng::new(0x0AE);
    let mut scratch = BandsScratch::new();
    for it in 0..15000 {
        let (fs, q) = match it % 3 {
            0 => (96000, false),
            1 => (96000, true),
            _ => (48000, true),
        };
        qab_case(&mut rng, fs, q, &mut scratch, it);
    }
}

// ---------------------------------------------------------------------------------------------
// quant_bands.c
// ---------------------------------------------------------------------------------------------

fn rust_quant_energy(
    p: &c::EnergyParams,
    e_bands: &[Glog],
    old: &mut [Glog],
    error: &mut [Glog],
    buf: &mut [u8],
    delayed: &mut Val32,
    fine_quant: &[i32],
    prev_quant: Option<&[i32]>,
    fine_priority: &[i32],
) -> [i32; 5] {
    let m = mode(p.fs, p.qext);
    let size = buf.len() as i32;
    let mut enc = EcEnc::new(buf);
    if p.prefix_ft > 1 {
        enc.enc_uint(p.prefix_val as u32, p.prefix_ft as u32);
    }
    qb::quant_coarse_energy(
        &m,
        p.start,
        p.end,
        p.eff_end,
        e_bands,
        old,
        p.budget,
        error,
        &mut enc,
        p.c,
        p.lm,
        p.nb_available_bytes,
        p.force_intra,
        delayed,
        p.two_pass,
        p.loss_rate,
        p.lfe,
    );
    let t0 = enc.tell_frac() as i32;
    qb::quant_fine_energy(
        &m, p.start, p.end, old, error, prev_quant, fine_quant, &mut enc, p.c,
    );
    let t1 = enc.tell_frac() as i32;
    let bits_left = size * 8 - enc.tell();
    qb::quant_energy_finalise(
        &m,
        p.start,
        p.end,
        if p.null_old { None } else { Some(old) },
        error,
        fine_quant,
        fine_priority,
        bits_left,
        &mut enc,
        p.c,
    );
    let t2 = enc.tell_frac() as i32;
    enc.done();
    [t0, t1, t2, enc.rng as i32, enc.error]
}

fn rust_unquant_energy(
    fs: i32,
    qext: bool,
    start: i32,
    end: i32,
    cc: i32,
    lm: i32,
    intra: i32,
    prefix_ft: i32,
    null_old: bool,
    old: &mut [Glog],
    buf: &[u8],
    fine_quant: &[i32],
    prev_quant: Option<&[i32]>,
    fine_priority: &[i32],
) -> [i32; 6] {
    let m = mode(fs, qext);
    let size = buf.len() as i32;
    let mut dec = EcDec::new(buf);
    if prefix_ft > 1 {
        let _ = dec.dec_uint(prefix_ft as u32);
    }
    let intra = if intra < 0 {
        if dec.tell() + 3 <= size * 8 {
            i32::from(dec.dec_bit_logp(3))
        } else {
            0
        }
    } else {
        intra
    };
    qb::unquant_coarse_energy(&m, start, end, old, intra != 0, &mut dec, cc, lm);
    let t0 = dec.tell_frac() as i32;
    qb::unquant_fine_energy(&m, start, end, old, prev_quant, fine_quant, &mut dec, cc);
    let t1 = dec.tell_frac() as i32;
    let bits_left = size * 8 - dec.tell();
    qb::unquant_energy_finalise(
        &m,
        start,
        end,
        if null_old { None } else { Some(old) },
        fine_quant,
        fine_priority,
        bits_left,
        &mut dec,
        cc,
    );
    [
        intra,
        t0,
        t1,
        dec.tell_frac() as i32,
        dec.rng as i32,
        dec.error,
    ]
}

#[test]
fn energy_quantisation_matches() {
    let mut rng = Rng::new(0xE11);
    for (fs, q) in all_modes() {
        let m = mode(fs, q);
        let nb = m.nb_ebands;
        for it in 0..10000 {
            let cc = rng.range_i32(1, 2);
            let lm = rng.range_i32(0, 3);
            let start = if !q && rng.range_i32(0, 4) == 0 {
                17
            } else {
                0
            };
            let end = if q { nb } else { rng.range_i32(start + 1, nb) };
            let eff_end = rng.range_i32(start + 1, end);
            let size = match rng.range_i32(0, 3) {
                0 => rng.range_i32(1, 8),
                1 => rng.range_i32(8, 60),
                _ => rng.range_i32(60, 1275),
            } as usize;
            let budget = if rng.range_i32(0, 4) == 0 {
                rng.range_i32(0, size as i32 * 8) as u32
            } else {
                size as u32 * 8
            };
            let p = c::EnergyParams {
                fs,
                qext: q,
                start,
                end,
                eff_end,
                c: cc,
                lm,
                budget,
                nb_available_bytes: rng.range_i32(1, size as i32 + 5),
                force_intra: rng.range_i32(0, 5) == 0,
                two_pass: rng.next_u32() & 1 == 0,
                loss_rate: rng.range_i32(0, 30),
                lfe: rng.range_i32(0, 8) == 0,
                prefix_ft: if rng.range_i32(0, 3) == 0 {
                    rng.range_i32(2, 3000)
                } else {
                    0
                },
                prefix_val: 0,
                null_old: rng.range_i32(0, 4) == 0,
            };
            let mut p = p;
            if p.prefix_ft > 1 {
                p.prefix_val = rng.range_i32(0, p.prefix_ft - 1);
            }
            // Mostly typical energies; sometimes very low ones (the fixed-point predictor clamps
            // at -28).
            let e_bands: Vec<f32> = if rng.range_i32(0, 4) == 0 {
                (0..42).map(|_| 8.0 * rng.f32_sym() - 24.0).collect()
            } else {
                (0..42).map(|_| 12.0 * rng.f32_sym() + 2.0).collect()
            };
            let mut old: Vec<Glog> = match rng.range_i32(0, 3) {
                0 => vec![glog(-28.0); 42],
                1 => e_bands
                    .iter()
                    .map(|&e| glog(e + 2.0 * rng.f32_sym()))
                    .collect(),
                _ => (0..42).map(|_| glog(15.0 * rng.f32_sym())).collect(),
            };
            let fine_quant: Vec<i32> = (0..21).map(|_| rng.range_i32(0, 8)).collect();
            let fine_priority: Vec<i32> = (0..21).map(|_| rng.range_i32(0, 1)).collect();
            let prev_quant: Option<Vec<i32>> = if rng.range_i32(0, 3) == 0 {
                Some((0..21).map(|_| rng.range_i32(0, 8)).collect())
            } else {
                None
            };
            let mut delayed = if rng.next_u32() & 1 == 0 {
                val32(0.0)
            } else {
                val32(50.0 * rng.f32_sym().abs())
            };
            // Several frames: the predictor state (oldEBands, delayedIntra) evolves.
            for f in 0..3 {
                let e_bands: Vec<Glog> = e_bands
                    .iter()
                    .map(|&e| glog(e + f as f32 * rng.f32_sym()))
                    .collect();
                let (mut oa, mut ob) = (old.clone(), old.clone());
                let (mut era, mut erb) = (vec![Glog::default(); 42], vec![Glog::default(); 42]);
                let (mut ba, mut bb) = (vec![0u8; size], vec![0u8; size]);
                let (mut da, mut db) = (delayed, delayed);
                let (fqa, mut fqb) = (fine_quant.clone(), fine_quant.clone());
                let mut fpb = fine_priority.clone();
                let mut pqb = prev_quant.clone();
                let ra = rust_quant_energy(
                    &p,
                    &e_bands,
                    &mut oa,
                    &mut era,
                    &mut ba,
                    &mut da,
                    &fqa,
                    prev_quant.as_deref(),
                    &fine_priority,
                );
                let rb = c::quant_energy(
                    &p,
                    &e_bands,
                    &mut ob,
                    &mut erb,
                    &mut bb,
                    &mut db,
                    &mut fqb,
                    pqb.as_deref_mut(),
                    &mut fpb,
                );
                let w = format!("quant_energy {fs} {q} {it}/{f} {p:?}");
                assert_eq!(ra, rb, "{w}");
                assert_slice_eq(&format!("bytes {w}"), &ba, &bb);
                assert_v(&format!("old {w}"), &oa, &ob);
                // C copies the whole (partly uninitialised) intra error array: compare the
                // bands that are written.
                for ch in 0..cc as usize {
                    let (lo, hi) = (
                        ch * nb as usize + start as usize,
                        ch * nb as usize + end as usize,
                    );
                    assert_v(&format!("error {w}"), &era[lo..hi], &erb[lo..hi]);
                }
                assert_v1(&format!("delayedIntra {w}"), da, db);
                // loss_distortion (static)
                let mut ob2 = ob.clone();
                assert_v1(
                    "loss_distortion",
                    qb::loss_distortion(&e_bands, &oa, start, eff_end, nb, cc),
                    c::loss_distortion(&e_bands, &mut ob2, start, eff_end, nb, cc),
                );
                // Decode.
                if it % 6 == 5 {
                    rng.fill_bytes(&mut ba);
                    bb.copy_from_slice(&ba);
                }
                let intra = if rng.range_i32(0, 4) == 0 {
                    rng.range_i32(0, 1)
                } else {
                    -1
                };
                let (mut ua, mut ub) = (old.clone(), old.clone());
                let dra = rust_unquant_energy(
                    fs,
                    q,
                    start,
                    end,
                    cc,
                    lm,
                    intra,
                    p.prefix_ft,
                    p.null_old,
                    &mut ua,
                    &ba,
                    &fqa,
                    prev_quant.as_deref(),
                    &fine_priority,
                );
                let drb = c::unquant_energy(
                    fs,
                    q,
                    start,
                    end,
                    cc,
                    lm,
                    intra,
                    p.prefix_ft,
                    p.null_old,
                    &mut ub,
                    &mut bb,
                    &mut fqb,
                    pqb.as_deref_mut(),
                    &mut fpb,
                );
                assert_eq!(dra, drb, "unquant {w}");
                assert_v(&format!("unquant old {w}"), &ua, &ub);
                old = oa;
                delayed = da;
            }
        }
    }
}

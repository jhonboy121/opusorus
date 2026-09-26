//! Differential tests for unit `silk_encoder_flp`: the floating-point SILK encoder
//! (`silk/float/*.c`, `silk/enc_API.c`, `silk/init_encoder.c`, `silk/control_codec.c`) vs the C
//! oracle.
//!
//! * Leaf DSP functions (`SigProc_FLP.h` helpers, energy / inner product / autocorrelation,
//!   Burg, Schur, k2a, windows, LPC/LTP filters, sorting, warped autocorrelation, NLSF
//!   wrappers, LTP analysis + quantization, LPC analysis, the pitch analyser and its static
//!   stage-3 helpers, the static noise-shaping coefficient limiters): randomized and realistic
//!   inputs, bit-exact (`to_bits`) outputs.
//! * Full encoder: long multi-packet streams through `silk_Encode` on both sides with identical
//!   control parameters (API rates 8..48 kHz (+96 kHz with qext), NB/MB/WB internal rates with
//!   switching, 10/20/40/60 ms, complexity 0..10, 5..80 kb/s, CBR/VBR with tight caps, FEC,
//!   DTX on silence, mono/stereo incl. mono<->stereo switches and toMono, reducedDependency,
//!   prefill): return codes, nBytesOut, the control struct, the range encoder state, the
//!   finished payload bytes, final range, and the complete encoder state (every float as its
//!   bit pattern) after every call. The payloads are also decoded with the Rust and C SILK
//!   decoders as a sanity check.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "test code: failures should panic; index loops mirror the C layouts"
)]

use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::{EcEnc, EcEncSnapshot};
use opusorus::silk::decoder::SilkDecoder;
use opusorus::silk::encoder::{SilkEncoder, silk_init_encoder};
use opusorus::silk::float::{self as flp, SilkEncoderStateFlp};
use opusorus::silk::resampler::{ResamplerFunction, SilkResamplerState};
use opusorus::silk::structs::{SideInfoIndices, SilkDecControlStruct, SilkEncControlStruct};
use opusorus_conformance::{Rng, assert_slice_eq, signals};
use opusorus_oracle::silk_decoder as cdec;
use opusorus_oracle::silk_encoder_flp as c;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

#[track_caller]
fn eq_f32(what: &str, r: &[f32], c: &[f32]) {
    assert_eq!(r.len(), c.len(), "{what}: length");
    for i in 0..r.len() {
        assert!(
            r[i].to_bits() == c[i].to_bits(),
            "{what}: mismatch at {i}: rust={} ({:#010x}) c={} ({:#010x})",
            r[i],
            r[i].to_bits(),
            c[i],
            c[i].to_bits()
        );
    }
}

#[track_caller]
fn eq1(what: &str, r: f32, c: f32) {
    assert!(
        r.to_bits() == c.to_bits(),
        "{what}: rust={r} ({:#010x}) c={c} ({:#010x})",
        r.to_bits(),
        c.to_bits()
    );
}

#[track_caller]
fn eq1d(what: &str, r: f64, c: f64) {
    assert!(
        r.to_bits() == c.to_bits(),
        "{what}: rust={r} ({:#018x}) c={c} ({:#018x})",
        r.to_bits(),
        c.to_bits()
    );
}

/// Random float with random magnitude (exponent spread), some exact zeros.
fn rand_f(rng: &mut Rng, max_log2: i32) -> f32 {
    match rng.range_i32(0, 15) {
        0 => 0.0,
        _ => {
            let e = rng.range_i32(-20, max_log2);
            rng.f32_sym() * (2.0f32).powi(e)
        }
    }
}

/// AR(2) resonance filtered noise at int16-ish scale, plus a pitch pulse train sometimes.
fn ar_signal(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    let r = 0.5 + 0.49 * (rng.f32_sym() * 0.5 + 0.5);
    let w = 0.1 + 2.9 * (rng.f32_sym() * 0.5 + 0.5);
    let a1 = 2.0 * r * w.cos();
    let a2 = -r * r;
    let period = rng.range_i32(20, 300) as usize;
    let pulses = rng.range_i32(0, 1) == 1;
    let (mut y1, mut y2) = (0.0f32, 0.0f32);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut e = rng.f32_sym();
        if pulses && i % period == 0 {
            e += 8.0;
        }
        let y = e + a1 * y1 + a2 * y2;
        y2 = y1;
        y1 = y;
        out.push(y);
    }
    let m = out.iter().fold(1e-9f32, |m, &v| m.max(v.abs()));
    for v in &mut out {
        *v = (*v / m * amp).round();
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Leaf functions
// ---------------------------------------------------------------------------------------------

#[test]
fn sigproc_flp_helpers() {
    let mut rng = Rng::new(1);
    for _ in 0..20000 {
        let x = rand_f(&mut rng, 8);
        eq1("sigmoid", flp::silk_sigmoid(x), c::sigmoid(x));
        let y = (rand_f(&mut rng, 30).abs() as f64) + 1e-3;
        eq1("log2", flp::silk_log2(y), c::log2(y));
        let z = rand_f(&mut rng, 20);
        assert_eq!(flp::silk_float2int(z), c::float2int(z), "float2int {z}");
    }
    for _ in 0..200 {
        let n = rng.range_i32(1, 700) as usize;
        let x: Vec<f32> = (0..n).map(|_| rand_f(&mut rng, 17)).collect();
        let mut r = vec![0i16; n];
        let mut cc = vec![0i16; n];
        flp::silk_float2short_array(&mut r, &x, n);
        c::float2short_array(&mut cc, &x, n);
        assert_slice_eq("float2short", &r, &cc);
        let mut rf = vec![0f32; n];
        let mut cf = vec![0f32; n];
        flp::silk_short2float_array(&mut rf, &r, n);
        c::short2float_array(&mut cf, &r, n);
        eq_f32("short2float", &rf, &cf);
    }
    // half-way rounding cases
    for v in [
        0.5f32, 1.5, 2.5, -0.5, -1.5, 32767.5, -32768.5, 40000.0, -40000.0,
    ] {
        assert_eq!(flp::silk_float2int(v), c::float2int(v));
    }
}

#[test]
fn energy_inner_product_autocorr() {
    let mut rng = Rng::new(2);
    for it in 0..3000 {
        let n = rng.range_i32(0, 600) as usize;
        let amp = [1.0f32, 100.0, 30000.0, 1e-3][it % 4];
        let a: Vec<f32> = (0..n + 30).map(|_| rng.f32_sym() * amp).collect();
        let b: Vec<f32> = (0..n + 30).map(|_| rand_f(&mut rng, 15)).collect();
        eq1d("energy", flp::silk_energy_flp(&a, n), c::energy(&a, n));
        eq1d(
            "inner",
            flp::silk_inner_product_flp(&a, &b, n),
            c::inner_product(&a, &b, n),
        );
        if n > 0 {
            let cnt = rng.range_i32(1, 30) as usize;
            let mut r = vec![0f32; 30];
            let mut cc = vec![0f32; 30];
            flp::silk_autocorrelation_flp(&mut r, &a, n, cnt);
            c::autocorrelation(&mut cc, &a, n, cnt);
            eq_f32("autocorr", &r, &cc);
        }
    }
}

#[test]
fn sine_window_bwexpander_k2a_schur_inv_pred_gain() {
    let mut rng = Rng::new(3);
    for _ in 0..3000 {
        // sine window
        let len = 4 * rng.range_i32(1, 40) as usize;
        let x: Vec<f32> = (0..len).map(|_| rng.f32_sym() * 20000.0).collect();
        for wt in [1, 2] {
            let mut r = vec![0f32; len];
            let mut cc = vec![0f32; len];
            flp::silk_apply_sine_window_flp(&mut r, &x, wt, len);
            c::apply_sine_window(&mut cc, &x, wt, len);
            eq_f32("sine window", &r, &cc);
        }
        // Schur / k2a / bwexpander / inverse prediction gain on a realistic autocorrelation
        let order = rng.range_i32(1, 24) as usize;
        let sig = ar_signal(&mut rng, 320, 10000.0);
        let mut ac = [0f32; 25];
        flp::silk_autocorrelation_flp(&mut ac, &sig, 320, order + 1);
        if rng.range_i32(0, 3) == 0 {
            ac[0] += ac[0] * 1e-3 + 1.0;
        }
        let mut rr = [0f32; 24];
        let mut cr = [0f32; 24];
        let er = flp::silk_schur_flp(&mut rr, &ac, order);
        let ec = c::schur(&mut cr, &ac, order);
        eq1("schur nrg", er, ec);
        eq_f32("schur rc", &rr, &cr);
        let mut ar = [0f32; 24];
        let mut ac2 = [0f32; 24];
        flp::silk_k2a_flp(&mut ar, &rr, order);
        c::k2a(&mut ac2, &rr, order);
        eq_f32("k2a", &ar, &ac2);
        eq1(
            "inv pred gain",
            flp::silk_lpc_inverse_pred_gain_flp(&ar, order),
            c::lpc_inverse_pred_gain(&ar, order),
        );
        let chirp = 0.5 + 0.5 * (rng.f32_sym() * 0.5 + 0.5);
        flp::silk_bwexpander_flp(&mut ar, order, chirp);
        c::bwexpander(&mut ac2, order, chirp);
        eq_f32("bwexpander", &ar, &ac2);
        // random (possibly unstable) coefficients for the inverse prediction gain
        let rnd: Vec<f32> = (0..order).map(|_| rng.f32_sym() * 1.2).collect();
        eq1(
            "inv pred gain rnd",
            flp::silk_lpc_inverse_pred_gain_flp(&rnd, order),
            c::lpc_inverse_pred_gain(&rnd, order),
        );
    }
    // Degenerate autocorrelations (zero energy) for Schur
    let ac = [0f32; 25];
    let mut rr = [0f32; 24];
    let mut cr = [0f32; 24];
    eq1(
        "schur zero",
        flp::silk_schur_flp(&mut rr, &ac, 16),
        c::schur(&mut cr, &ac, 16),
    );
    eq_f32("schur zero rc", &rr, &cr);
}

#[test]
fn burg_modified() {
    let mut rng = Rng::new(4);
    for it in 0..2500 {
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let d = [6usize, 8, 10, 12, 16, 24][rng.range_i32(0, 5) as usize];
        let max_sl = 384 / nb_subfr;
        let sl = rng.range_i32(d as i32 + 1, max_sl as i32) as usize;
        let amp = [30000.0f32, 3000.0, 30.0, 1.0][it % 4];
        let x = ar_signal(&mut rng, sl * nb_subfr, amp);
        let min_inv_gain = [1e-4f32, 1e-2, 0.1, 0.5, 0.99][rng.range_i32(0, 4) as usize];
        let mut ar = [0f32; 24];
        let mut ac = [0f32; 24];
        let nr = flp::silk_burg_modified_flp(&mut ar, &x, min_inv_gain, sl, nb_subfr, d);
        let nc = c::burg_modified(&mut ac, &x, min_inv_gain, sl, nb_subfr, d);
        eq1("burg nrg", nr, nc);
        eq_f32("burg A", &ar, &ac);
    }
    // silence
    let x = [0f32; 384];
    let mut ar = [0f32; 24];
    let mut ac = [0f32; 24];
    eq1(
        "burg silence",
        flp::silk_burg_modified_flp(&mut ar, &x, 1e-4, 96, 4, 16),
        c::burg_modified(&mut ac, &x, 1e-4, 96, 4, 16),
    );
    eq_f32("burg silence A", &ar, &ac);
}

#[test]
fn corr_matrix_vector_lpc_ltp_filters() {
    let mut rng = Rng::new(5);
    for _ in 0..2000 {
        // corrMatrix / corrVector
        let order = rng.range_i32(2, 16) as usize;
        let l = rng.range_i32(1, 120) as usize;
        let x = ar_signal(&mut rng, l + order - 1, 20000.0);
        let t: Vec<f32> = (0..l).map(|_| rng.f32_sym() * 1000.0).collect();
        let mut rm = vec![0f32; order * order];
        let mut cm = vec![0f32; order * order];
        flp::silk_corr_matrix_flp(&x, l, order, &mut rm);
        c::corr_matrix(&x, l, order, &mut cm);
        eq_f32("corrMatrix", &rm, &cm);
        let mut rv = vec![0f32; order];
        let mut cv = vec![0f32; order];
        flp::silk_corr_vector_flp(&x, &t, l, order, &mut rv);
        c::corr_vector(&x, &t, l, order, &mut cv);
        eq_f32("corrVector", &rv, &cv);

        // LPC analysis filter
        let ord = [6usize, 8, 10, 12, 16][rng.range_i32(0, 4) as usize];
        let len = rng.range_i32(ord as i32, 400) as usize;
        let s = ar_signal(&mut rng, len, 30000.0);
        let coef: Vec<f32> = (0..ord).map(|_| rng.f32_sym() * 1.5).collect();
        let mut rr = vec![7f32; len];
        let mut cr = vec![7f32; len];
        flp::silk_lpc_analysis_filter_flp(&mut rr, &coef, &s, len, ord);
        c::lpc_analysis_filter(&mut cr, &coef, &s, len, ord);
        eq_f32("LPC analysis filter", &rr, &cr);

        // LTP analysis filter
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let sl = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let pre = [10usize, 16][rng.range_i32(0, 1) as usize];
        let pitch: Vec<i32> = (0..4).map(|_| rng.range_i32(16, 288)).collect();
        let x_off = 290 + pre;
        let xb = ar_signal(&mut rng, x_off + nb_subfr * sl + pre, 20000.0);
        let b: Vec<f32> = (0..20).map(|_| rng.f32_sym() * 0.5).collect();
        let inv_g: Vec<f32> = (0..4)
            .map(|_| 1.0 / (1.0 + 500.0 * rng.f32_sym().abs()))
            .collect();
        let n = nb_subfr * (sl + pre);
        let mut rl = vec![0f32; n];
        let mut cl = vec![0f32; n];
        flp::silk_ltp_analysis_filter_flp(
            &mut rl, &xb, x_off, &b, &pitch, &inv_g, sl, nb_subfr, pre,
        );
        c::ltp_analysis_filter(&mut cl, &xb, x_off, &b, &pitch, &inv_g, sl, nb_subfr, pre);
        eq_f32("LTP analysis filter", &rl, &cl);
    }
}

#[test]
fn residual_energy_scale_sort_warped() {
    let mut rng = Rng::new(6);
    for _ in 0..2000 {
        // residual_energy_FLP
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let order = if rng.range_i32(0, 1) == 0 { 10 } else { 16 };
        let sl = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let x = ar_signal(&mut rng, nb_subfr * (sl + order), 3000.0);
        let mut a = [[0f32; 16]; 2];
        for row in &mut a {
            for v in row.iter_mut().take(order) {
                *v = rng.f32_sym() * 0.8;
            }
        }
        let gains: Vec<f32> = (0..4).map(|_| 1.0 + 1000.0 * rng.f32_sym().abs()).collect();
        let mut rn = [0f32; 4];
        flp::silk_residual_energy_flp(&mut rn, &x, &a, &gains, sl, nb_subfr, order);
        let cn = c::residual_energy(&x, &a, &gains, sl, nb_subfr, order);
        eq_f32("residual energy", &rn, &cn);

        // residual_energy_covar (+ regularize)
        let d = rng.range_i32(1, 16) as usize;
        let mut m = vec![0f32; d * d];
        for i in 0..d {
            for j in i..d {
                let v = rng.f32_sym() * if i == j { 10.0 } else { 3.0 };
                m[i * d + j] = v;
                m[j * d + i] = v;
            }
            m[i * d + i] = m[i * d + i].abs();
        }
        let cvec: Vec<f32> = (0..d).map(|_| rng.f32_sym()).collect();
        let wv: Vec<f32> = (0..d).map(|_| rng.f32_sym() * 5.0).collect();
        let wxx = rng.f32_sym() * 10.0;
        let mut mr = m.clone();
        let mut mc = m.clone();
        // It ends in `silk_assert( nrg == 0 )` when the regularization does not converge: with
        // the `assertions` feature the port panics there and the C oracle would abort.
        let converges = !cfg!(feature = "assertions")
            || std::panic::catch_unwind(|| {
                let mut m2 = m.clone();
                flp::silk_residual_energy_covar_flp(&cvec, &mut m2, &wv, wxx, d)
            })
            .is_ok();
        if converges {
            eq1(
                "residual energy covar",
                flp::silk_residual_energy_covar_flp(&cvec, &mut mr, &wv, wxx, d),
                c::residual_energy_covar(&cvec, &mut mc, &wv, wxx, d),
            );
            eq_f32("covar regularized", &mr, &mc);
        }
        let noise = rng.f32_sym().abs() * 3.0;
        let mut xr = [wxx];
        let mut xc = [wxx];
        flp::silk_regularize_correlations_flp(&mut mr, &mut xr, noise, d);
        c::regularize_correlations(&mut mc, &mut xc, noise, d);
        eq_f32("regularize XX", &mr, &mc);
        eq_f32("regularize xx", &xr, &xc);

        // scale vectors
        let n = rng.range_i32(0, 100) as usize;
        let v: Vec<f32> = (0..n).map(|_| rand_f(&mut rng, 15)).collect();
        let g = rand_f(&mut rng, 4);
        let mut r1 = vec![0f32; n];
        let mut c1 = vec![0f32; n];
        flp::silk_scale_copy_vector_flp(&mut r1, &v, g, n);
        c::scale_copy_vector(&mut c1, &v, g, n);
        eq_f32("scale copy", &r1, &c1);
        let mut r2 = v.clone();
        let mut c2 = v.clone();
        flp::silk_scale_vector_flp(&mut r2, g, n);
        c::scale_vector(&mut c2, g, n);
        eq_f32("scale", &r2, &c2);

        // insertion sort (with ties)
        let l = rng.range_i32(1, 80) as usize;
        let k = rng.range_i32(1, l as i32) as usize;
        let a0: Vec<f32> = (0..l)
            .map(|_| (rng.range_i32(-20, 20) as f32) * 0.25)
            .collect();
        let mut ra = a0.clone();
        let mut ca = a0.clone();
        let mut ri = vec![0i32; k];
        let mut ci = vec![0i32; k];
        flp::silk_insertion_sort_decreasing_flp(&mut ra, &mut ri, l, k);
        c::insertion_sort_decreasing(&mut ca, &mut ci, l, k);
        eq_f32("sort a", &ra, &ca);
        assert_slice_eq("sort idx", &ri, &ci);

        // warped autocorrelation
        let order = 2 * rng.range_i32(1, 12) as usize;
        let len = rng.range_i32(1, 240) as usize;
        let x = ar_signal(&mut rng, len, 20000.0);
        let warping = rng.f32_sym().abs() * 0.3;
        let mut rw = [0f32; 25];
        let mut cw = [0f32; 25];
        flp::silk_warped_autocorrelation_flp(&mut rw, &x, warping, len, order);
        c::warped_autocorrelation(&mut cw, &x, warping, len, order);
        eq_f32("warped autocorr", &rw, &cw);
    }
}

#[test]
fn noise_shape_coef_limiters() {
    let mut rng = Rng::new(7);
    for _ in 0..5000 {
        let order = 2 * rng.range_i32(1, 12) as usize;
        let scale = [0.5f32, 2.0, 6.0, 20.0][rng.range_i32(0, 3) as usize];
        let coefs: Vec<f32> = (0..order).map(|_| rng.f32_sym() * scale).collect();
        let lambda = rng.f32_sym().abs() * 0.3;
        eq1(
            "warped_gain",
            flp::warped_gain(&coefs, lambda, order),
            c::warped_gain(&coefs, lambda, order),
        );
        // Both limiters end in `silk_assert( 0 )` when they do not converge: with the
        // `assertions` feature the port panics there and the C oracle would abort, so the
        // oracle is only called when the port returned.
        let converges = |f: &dyn Fn(&mut [f32])| {
            let mut r = coefs.clone();
            !cfg!(feature = "assertions")
                || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut r))).is_ok()
        };
        if converges(&|r| flp::warped_true2monic_coefs(r, lambda, 3.999, order)) {
            let mut r = coefs.clone();
            let mut cc = coefs.clone();
            flp::warped_true2monic_coefs(&mut r, lambda, 3.999, order);
            c::warped_true2monic_coefs(&mut cc, lambda, 3.999, order);
            eq_f32("warped_true2monic", &r, &cc);
        }
        if converges(&|r| flp::limit_coefs(r, 3.999, order)) {
            let mut r = coefs.clone();
            let mut cc = coefs;
            flp::limit_coefs(&mut r, 3.999, order);
            c::limit_coefs(&mut cc, 3.999, order);
            eq_f32("limit_coefs", &r, &cc);
        }
    }
}

#[test]
fn nlsf_wrappers_find_lpc_ltp() {
    let mut rng = Rng::new(8);
    for it in 0..1500 {
        let order = if rng.range_i32(0, 1) == 0 { 10 } else { 16 };
        // A2NLSF / NLSF2A on Burg-derived stable filters
        let sig = ar_signal(&mut rng, 320, 10000.0);
        let mut a = [0f32; 24];
        flp::silk_burg_modified_flp(&mut a, &sig, 1e-3, 80, 4, order);
        let mut rn = [0i16; 16];
        let mut cn = [0i16; 16];
        flp::silk_a2nlsf_flp(&mut rn, &a, order);
        c::a2nlsf(&mut cn, &a, order);
        assert_slice_eq("A2NLSF_FLP", &rn, &cn);
        let mut ra = [0f32; 16];
        let mut ca = [0f32; 16];
        flp::silk_nlsf2a_flp(&mut ra, &rn, order);
        c::nlsf2a(&mut ca, &rn, order);
        eq_f32("NLSF2A_FLP", &ra, &ca);

        // find_LPC (with and without the interpolation search)
        let nb_subfr = if it % 3 == 0 { 2 } else { 4 };
        let sl = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let x = ar_signal(&mut rng, nb_subfr * (sl + order), 5000.0);
        let mut prev = [0i16; 16];
        let psig = ar_signal(&mut rng, 320, 10000.0);
        let mut pa = [0f32; 24];
        flp::silk_burg_modified_flp(&mut pa, &psig, 1e-3, 80, 4, order);
        flp::silk_a2nlsf_flp(&mut prev, &pa, order);
        let use_interp = rng.range_i32(0, 3) != 0;
        let first = rng.range_i32(0, 4) == 0;
        let mig = [1e-2f32, 1e-3, 1e-4][rng.range_i32(0, 2) as usize];
        let mut st = SilkEncoderStateFlp::new();
        let s = &mut st.s_cmn;
        s.predict_lpc_order = order as i32;
        s.subfr_length = sl as i32;
        s.nb_subfr = nb_subfr as i32;
        s.use_interpolated_nlsfs = i32::from(use_interp);
        s.first_frame_after_reset = i32::from(first);
        s.prev_nlsfq_q15 = prev;
        let mut rnl = [0i16; 16];
        flp::silk_find_lpc_flp(s, &mut rnl, &x, mig);
        let (cnl, cint) = c::find_lpc(order, sl, nb_subfr, use_interp, first, &prev, &x, mig);
        assert_eq!(
            s.indices.nlsf_interp_coef_q2 as i32, cint,
            "find_LPC interp"
        );
        assert_slice_eq("find_LPC NLSF", &rnl[..order], &cnl[..order]);

        // find_LTP + quant_LTP_gains_FLP
        let nb = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let sl = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let r_off = 300usize;
        let rb = ar_signal(&mut rng, r_off + nb * sl + 5, 2000.0);
        let mut lag = [0i32; 4];
        let base = rng.range_i32(20, 280);
        for l in &mut lag {
            *l = (base + rng.range_i32(-4, 4)).clamp(16, 290);
        }
        let mut rxx = [0f32; 100];
        let mut rxv = [0f32; 20];
        flp::silk_find_ltp_flp(&mut rxx, &mut rxv, &rb, r_off, &lag, sl, nb);
        let (cxx, cxv) = c::find_ltp(&rb, r_off, &lag, sl, nb);
        eq_f32("find_LTP XX", &rxx[..nb * 25], &cxx[..nb * 25]);
        eq_f32("find_LTP xX", &rxv[..nb * 5], &cxv[..nb * 5]);
        let slg = rng.range_i32(0, 4000);
        let mut b = [0f32; 20];
        let mut cbk = [0i8; 4];
        let mut per = 0i8;
        let mut slg_r = slg;
        let mut pg = 0f32;
        flp::silk_quant_ltp_gains_flp(
            &mut b, &mut cbk, &mut per, &mut slg_r, &mut pg, &rxx, &rxv, sl as i32, nb,
        );
        let (cb, ccbk, cper, cslg, cpg) = c::quant_ltp_gains(&rxx, &rxv, sl as i32, nb, slg);
        eq_f32("quant LTP B", &b[..nb * 5], &cb[..nb * 5]);
        assert_slice_eq("quant LTP cbk", &cbk[..nb], &ccbk[..nb]);
        assert_eq!(per, cper, "quant LTP per");
        assert_eq!(slg_r, cslg, "quant LTP sum_log_gain");
        eq1("quant LTP pred gain", pg, cpg);
    }
}

/// A frame for the pitch analyser: a voiced-ish signal (pulse train through a resonance) at
/// `fs_khz`, or noise / silence.
fn pitch_frame(rng: &mut Rng, fs_khz: i32, nb_subfr: i32, kind: i32) -> Vec<f32> {
    let n = ((20 + 5 * nb_subfr) * fs_khz) as usize;
    match kind {
        0 => vec![0.0; n],
        1 => (0..n).map(|_| (rng.f32_sym() * 3000.0).round()).collect(),
        _ => {
            let period = rng.range_i32(2 * fs_khz, 18 * fs_khz) as f32 + rng.f32_sym();
            let r = 0.9f32;
            let w = 0.3 + rng.f32_sym().abs() * 1.5;
            let (mut y1, mut y2) = (0f32, 0f32);
            let mut ph = 0f32;
            let drift = rng.f32_sym() * 0.002;
            let mut per = period;
            (0..n)
                .map(|_| {
                    ph += 1.0;
                    let mut e = 0.05 * rng.f32_sym();
                    if ph >= per {
                        ph -= per;
                        e += 1.0;
                        per *= 1.0 + drift;
                    }
                    let y = e + 2.0 * r * w.cos() * y1 - r * r * y2;
                    y2 = y1;
                    y1 = y;
                    (y * 2000.0).round().clamp(-32000.0, 32000.0)
                })
                .collect()
        }
    }
}

#[test]
fn pitch_analysis_core() {
    let mut rng = Rng::new(9);
    let mut voiced = 0;
    for it in 0..2500 {
        let fs_khz = [8, 12, 16][it % 3];
        let nb_subfr = if (it / 3) % 2 == 0 { 4 } else { 2 };
        let complexity = rng.range_i32(0, 2);
        let kind = if it % 17 == 0 {
            0
        } else if it % 5 == 0 {
            1
        } else {
            2
        };
        let frame = pitch_frame(&mut rng, fs_khz, nb_subfr, kind);
        let prev_lag = if rng.range_i32(0, 2) == 0 {
            0
        } else {
            rng.range_i32(2 * fs_khz, 18 * fs_khz)
        };
        let ltp_corr = rng.f32_sym().abs();
        let th1 = 0.7 + 0.1 * rng.f32_sym().abs();
        let th2 = 0.1 + 0.4 * rng.f32_sym().abs();
        let init = [rng.range_i32(0, 99), 7, -3, 1234];
        let mut pitch = init;
        let mut lag_index = 0i16;
        let mut contour = 0i8;
        let mut corr = ltp_corr;
        let r = flp::silk_pitch_analysis_core_flp(
            &frame,
            &mut pitch,
            &mut lag_index,
            &mut contour,
            &mut corr,
            prev_lag,
            th1,
            th2,
            fs_khz,
            complexity,
            nb_subfr,
        );
        let (cr, cp, cl, cc, ccorr) = c::pitch_analysis_core(
            &frame, ltp_corr, prev_lag, th1, th2, fs_khz, complexity, nb_subfr, init,
        );
        let what = format!("pitch it {it} fs {fs_khz} nb {nb_subfr} cx {complexity} kind {kind}");
        assert_eq!(r, cr, "{what}: voicing");
        assert_slice_eq(&what, &pitch, &cp);
        assert_eq!(lag_index, cl, "{what}: lagIndex");
        assert_eq!(contour, cc, "{what}: contourIndex");
        eq1(&what, corr, ccorr);
        if r == 0 {
            voiced += 1;
        }

        // stage-3 helpers directly
        if fs_khz > 8 {
            let sf = (5 * fs_khz) as usize;
            let start_lag = rng.range_i32(2 * fs_khz, 18 * fs_khz - 1 - 8);
            let mut rc: flp::Stage3Array = [[[0.0; 5]; 34]; 4];
            let mut re: flp::Stage3Array = [[[0.0; 5]; 34]; 4];
            flp::silk_p_ana_calc_corr_st3(&mut rc, &frame, start_lag, sf, nb_subfr, complexity);
            flp::silk_p_ana_calc_energy_st3(&mut re, &frame, start_lag, sf, nb_subfr, complexity);
            let cc3 = c::calc_corr_st3(&frame, start_lag, sf, nb_subfr, complexity);
            let ce3 = c::calc_energy_st3(&frame, start_lag, sf, nb_subfr, complexity);
            eq_f32("corr_st3", rc.as_flattened().as_flattened(), &cc3);
            eq_f32("energy_st3", re.as_flattened().as_flattened(), &ce3);
        }
    }
    assert!(voiced > 500, "too few voiced frames: {voiced}");
}

// ---------------------------------------------------------------------------------------------
// Full encoder: state dump (same order as oracle_sef_dump in csrc/silk_encoder_flp.c)
// ---------------------------------------------------------------------------------------------

fn dump_indices(x: &SideInfoIndices, v: &mut Vec<i32>) {
    v.extend(x.gains_indices.iter().map(|&a| a as i32));
    v.extend(x.ltp_index.iter().map(|&a| a as i32));
    v.extend(x.nlsf_indices.iter().map(|&a| a as i32));
    v.extend([
        x.lag_index as i32,
        x.contour_index as i32,
        x.signal_type as i32,
        x.quant_offset_type as i32,
        x.nlsf_interp_coef_q2 as i32,
        x.per_index as i32,
        x.ltp_scale_index as i32,
        x.seed as i32,
    ]);
}

fn dump_resampler(r: &SilkResamplerState, v: &mut Vec<i32>) {
    v.extend(r.s_iir.iter().copied());
    match r.resampler_function {
        ResamplerFunction::DownFir => v.extend(r.s_fir_i32.iter().copied()),
        ResamplerFunction::IirFir => v.extend(r.s_fir_i16.iter().map(|&x| x as i32)),
        _ => v.extend([0; 36]),
    }
    v.extend(r.delay_buf.iter().map(|&x| x as i32));
    v.push(r.resampler_function.to_c());
    v.extend([
        r.batch_size,
        r.inv_ratio_q16,
        r.fir_order,
        r.fir_fracs,
        r.fs_in_khz,
        r.fs_out_khz,
        r.input_delay,
    ]);
}

fn dump_channel(e: &SilkEncoderStateFlp, v: &mut Vec<i32>) {
    let s = &e.s_cmn;
    v.extend(s.in_hp_state);
    v.extend([s.variable_hp_smth1_q15, s.variable_hp_smth2_q15]);
    v.extend(s.s_lp.in_lp_state);
    v.extend([s.s_lp.transition_frame_no, s.s_lp.mode, s.s_lp.saved_fs_khz]);
    let vad = &s.s_vad;
    v.extend(vad.ana_state);
    v.extend(vad.ana_state1);
    v.extend(vad.ana_state2);
    v.extend(vad.xnrg_subfr);
    v.extend(vad.nrg_ratio_smth_q8);
    v.push(vad.hp_state as i32);
    v.extend(vad.nl);
    v.extend(vad.inv_nl);
    v.extend(vad.noise_level_bias);
    v.push(vad.counter);
    let n = &s.s_nsq;
    v.extend(n.xq.iter().map(|&x| x as i32));
    v.extend(n.s_ltp_shp_q14.iter().copied());
    v.extend(n.s_lpc_q14.iter().copied());
    v.extend(n.s_ar2_q14.iter().copied());
    v.extend([
        n.s_lf_ar_shp_q14,
        n.s_diff_shp_q14,
        n.lag_prev,
        n.s_ltp_buf_idx,
        n.s_ltp_shp_buf_idx,
        n.rand_seed,
        n.prev_gain_q16,
        n.rewhite_flag,
    ]);
    v.extend(s.prev_nlsfq_q15.iter().map(|&x| x as i32));
    v.extend([
        s.speech_activity_q8,
        s.allow_bandwidth_switch,
        s.lbrr_prev_last_gain_index as i32,
        s.prev_signal_type as i32,
        s.prev_lag,
        s.pitch_lpc_win_length,
        s.max_pitch_lag,
        s.api_fs_hz,
        s.prev_api_fs_hz,
        s.max_internal_fs_hz,
        s.min_internal_fs_hz,
        s.desired_internal_fs_hz,
        s.fs_khz,
        s.nb_subfr,
        s.frame_length,
        s.subfr_length,
        s.ltp_mem_length,
        s.la_pitch,
        s.la_shape,
        s.shape_win_length,
        s.target_rate_bps,
        s.packet_size_ms,
        s.packet_loss_perc,
        s.frame_counter,
        s.complexity,
        s.n_states_delayed_decision,
        s.use_interpolated_nlsfs,
        s.shaping_lpc_order,
        s.predict_lpc_order,
        s.pitch_estimation_complexity,
        s.pitch_estimation_lpc_order,
        s.pitch_estimation_threshold_q16,
        s.sum_log_gain_q7,
        s.nlsf_msvq_survivors,
        s.first_frame_after_reset,
        s.controlled_since_last_payload,
        s.warping_q16,
        s.use_cbr,
        s.prefill_flag,
        s.pitch_lag_low_bits_icdf.len() as i32,
        s.pitch_contour_icdf.len() as i32,
        if s.fs_khz == 0 {
            0
        } else {
            s.ps_nlsf_cb.order as i32
        },
    ]);
    v.extend(s.input_quality_bands_q15);
    v.extend([s.input_tilt_q15, s.snr_db_q7]);
    v.extend(s.vad_flags.iter().map(|&x| x as i32));
    v.push(s.lbrr_flag as i32);
    v.extend(s.lbrr_flags);
    dump_indices(&s.indices, v);
    v.extend(s.pulses.iter().map(|&x| x as i32));
    v.extend(s.input_buf.iter().map(|&x| x as i32));
    v.extend([
        s.input_buf_ix,
        s.n_frames_per_packet,
        s.n_frames_encoded,
        s.n_channels_api,
        s.n_channels_internal,
        s.channel_nb,
        s.frames_since_onset,
        s.ec_prev_signal_type,
        s.ec_prev_lag_index as i32,
    ]);
    dump_resampler(&s.resampler_state, v);
    v.extend([
        s.use_dtx,
        s.in_dtx,
        s.no_speech_counter,
        s.use_in_band_fec,
        s.lbrr_enabled,
        s.lbrr_gain_increases,
    ]);
    for x in &s.indices_lbrr {
        dump_indices(x, v);
    }
    for p in &s.pulses_lbrr {
        v.extend(p.iter().map(|&x| x as i32));
    }
    v.push(e.s_shape.last_gain_index as i32);
    v.push(e.s_shape.harm_shape_gain_smth.to_bits() as i32);
    v.push(e.s_shape.tilt_smth.to_bits() as i32);
    v.extend(e.x_buf.iter().map(|&x| x.to_bits() as i32));
    v.push(e.ltp_corr.to_bits() as i32);
}

fn dump_state(e: &SilkEncoder) -> Vec<i32> {
    let mut v = Vec::with_capacity(12000);
    let st = &e.s_stereo;
    v.extend(st.pred_prev_q13.iter().map(|&x| x as i32));
    v.extend(st.s_mid.iter().map(|&x| x as i32));
    v.extend(st.s_side.iter().map(|&x| x as i32));
    v.extend(st.mid_side_amp_q0);
    v.extend([
        st.smth_width_q14 as i32,
        st.width_prev_q14 as i32,
        st.silent_side_len as i32,
    ]);
    for p in &st.pred_ix {
        v.extend(p[0].iter().map(|&x| x as i32));
        v.extend(p[1].iter().map(|&x| x as i32));
    }
    v.extend(st.mid_only_flags.iter().map(|&x| x as i32));
    v.extend([
        e.n_bits_used_lbrr,
        e.n_bits_exceeded,
        e.n_channels_api,
        e.n_channels_internal,
        e.n_prev_channels_internal,
        e.time_since_switch_allowed_ms,
        e.allow_bandwidth_switch,
        e.prev_decode_only_middle,
    ]);
    for ch in &e.state_fxx {
        dump_channel(ch, &mut v);
    }
    v
}

const fn ctl_to_c(r: &SilkEncControlStruct) -> c::EncCtrl {
    [
        r.n_channels_api,
        r.n_channels_internal,
        r.api_sample_rate,
        r.max_internal_sample_rate,
        r.min_internal_sample_rate,
        r.desired_internal_sample_rate,
        r.payload_size_ms,
        r.bit_rate,
        r.packet_loss_percentage,
        r.complexity,
        r.use_in_band_fec,
        r.use_dred,
        r.lbrr_coded,
        r.use_dtx,
        r.use_cbr,
        r.max_bits,
        r.to_mono,
        r.opus_can_switch,
        r.reduced_dependency,
        r.internal_sample_rate,
        r.allow_bandwidth_switch,
        r.in_wb_mode_without_variable_lp,
        r.stereo_width_q14,
        r.switch_ready,
        r.signal_type,
        r.offset,
    ]
}

fn ec_dump(e: &EcEnc<'_>) -> c::EcStateDump {
    [
        e.storage,
        e.end_offs,
        e.end_window,
        e.nend_bits as u32,
        e.nbits_total as u32,
        e.offs,
        e.rng,
        e.val,
        e.ext,
        e.rem as u32,
        e.error as u32,
        e.tell_frac(),
    ]
}

#[track_caller]
fn compare_dumps(what: &str, r: &[i32], c: &[i32]) {
    assert_eq!(r.len(), c.len(), "{what}: dump length");
    if let Some(i) = r.iter().zip(c).position(|(a, b)| a != b) {
        panic!(
            "{what}: state mismatch at dump index {i} (of {}): rust={} c={} (channel 0 starts at 43, each channel has {} values)",
            r.len(),
            r[i],
            c[i],
            (r.len() - 43) / 2
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Full encoder harness
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sig {
    Speech,
    Music,
    Noise,
    /// Speech with silent / near-silent stretches (DTX).
    Bursty,
    /// Loud clipped speech (input saturation).
    Loud,
    /// Stereo with identical / correlated / nearly identical channels.
    StereoMix,
    /// Garbage input: full-scale noise and square waves, pure tones, DC, NaN / infinities.
    Garbage,
    /// Pure tone (Hz), amplitude slowly swept (very sharp LPC resonances).
    Tone(u32),
}

#[derive(Clone, Debug)]
struct Cfg {
    api_fs: i32,
    api_ch: i32,
    int_ch: i32,
    /// (max, min, desired) internal rate.
    rates: (i32, i32, i32),
    packet_ms: i32,
    bitrate: i32,
    complexity: i32,
    cbr: bool,
    /// Max payload bytes for VBR (0 = 1275).
    cap: i32,
    fec: bool,
    loss: i32,
    dtx: bool,
    sig: Sig,
    n_packets: usize,
    seed: u64,
    /// Randomly vary parameters per packet.
    vary: bool,
    /// Randomly switch the internal channel count (stereo API only).
    ch_switch: bool,
    reduced_dependency: bool,
    /// Prefill (1 or 2) every n packets (0 = never).
    prefill_every: usize,
    /// Decode the payloads with the Rust and C SILK decoders.
    decode: bool,
    /// Feed each packet in 10 ms `silk_Encode` calls (the range coder spans the calls).
    chunked: bool,
    /// Randomly switch the API sampling rate between packets.
    api_switch: bool,
}

impl Cfg {
    const fn new(api_fs: i32, api_ch: i32, packet_ms: i32, bitrate: i32, complexity: i32) -> Self {
        Self {
            api_fs,
            api_ch,
            int_ch: api_ch,
            rates: (16000, 8000, 16000),
            packet_ms,
            bitrate,
            complexity,
            cbr: false,
            cap: 0,
            fec: false,
            loss: 0,
            dtx: false,
            sig: Sig::Speech,
            n_packets: 50,
            seed: 1,
            vary: false,
            ch_switch: false,
            reduced_dependency: false,
            prefill_every: 0,
            decode: false,
            chunked: false,
            api_switch: false,
        }
    }
}

fn gen_signal(cfg: &Cfg, fs: i32, n: usize) -> Vec<f32> {
    let ch = cfg.api_ch as usize;
    let fs = fs as u32;
    let mut x = match cfg.sig {
        Sig::Speech | Sig::Bursty | Sig::Loud | Sig::StereoMix => {
            signals::speech_like(n, ch, fs, cfg.seed)
        }
        Sig::Music => signals::music_like(n, ch, fs, cfg.seed),
        Sig::Noise => signals::noise(n, ch, 0.3, cfg.seed),
        Sig::Tone(f) => (0..n * ch)
            .map(|j| {
                let t = (j / ch) as f64 / fs as f64;
                let amp = 0.02 + 0.9 * (0.5 + 0.5 * (t * 0.7).sin());
                (amp * (2.0 * core::f64::consts::PI * f as f64 * t).sin()) as f32
            })
            .collect(),
        Sig::Garbage => {
            let mut rng = Rng::new(cfg.seed ^ 0x6A6A);
            let seg_len = fs as usize / 5;
            (0..n * ch)
                .map(|j| {
                    let i = j / ch;
                    let t = i as f32 / fs as f32;
                    match (i / seg_len) % 6 {
                        0 => 3.0 * rng.f32_sym(),
                        1 => {
                            if (i / 37).is_multiple_of(2) {
                                1.0
                            } else {
                                -1.0
                            }
                        }
                        2 => 0.9 * (2.0 * core::f32::consts::PI * 997.0 * t).sin(),
                        3 => 0.5,
                        4 => match rng.range_i32(0, 40) {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            _ => 0.1 * rng.f32_sym(),
                        },
                        _ => 1e-7 * rng.f32_sym(),
                    }
                })
                .collect()
        }
    };
    match cfg.sig {
        Sig::Bursty => {
            let mut rng = Rng::new(cfg.seed ^ 0x55);
            let seg_len = fs as usize * 7 / 10;
            for i in 0..n {
                let seg = i / seg_len;
                if seg % 2 == 1 {
                    for c in 0..ch {
                        x[i * ch + c] = if seg % 4 == 3 {
                            0.0003 * rng.f32_sym()
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        Sig::Loud => {
            for v in &mut x {
                *v *= 40.0;
            }
        }
        Sig::StereoMix if ch == 2 => {
            let music = signals::music_like(n, 1, fs, cfg.seed ^ 7);
            let seg_len = fs as usize / 3;
            for i in 0..n {
                match (i / seg_len) % 4 {
                    0 => x[i * 2 + 1] = x[i * 2],
                    1 => x[i * 2 + 1] = 0.5 * x[i * 2] + 0.5 * music[i],
                    2 => x[i * 2 + 1] = x[i * 2] + 0.00002 * music[i],
                    _ => x[i * 2 + 1] = -0.7 * x[i * 2] + 0.2 * music[i],
                }
            }
        }
        _ => {}
    }
    x
}

/// Coverage counters (sanity check that the streams exercise the interesting paths).
#[derive(Debug, Default, Clone, Copy)]
struct Cov {
    packets: usize,
    bytes_zero: usize,
    lbrr: usize,
    stereo: usize,
    mid_only: usize,
    voiced: usize,
    unvoiced: usize,
    fs: [usize; 3],
    prefill: usize,
    dtx: usize,
}

impl Cov {
    fn add(&mut self, o: &Self) {
        self.packets += o.packets;
        self.bytes_zero += o.bytes_zero;
        self.lbrr += o.lbrr;
        self.stereo += o.stereo;
        self.mid_only += o.mid_only;
        self.voiced += o.voiced;
        self.unvoiced += o.unvoiced;
        for i in 0..3 {
            self.fs[i] += o.fs[i];
        }
        self.prefill += o.prefill;
        self.dtx += o.dtx;
    }
}

struct Harness {
    c: c::SilkEnc,
    r: Box<SilkEncoder>,
    rbuf: Vec<u8>,
    /// Storage size and saved state of the Rust range encoder of the current packet (it is
    /// rebuilt over `rbuf` for every call, so a packet can span several `silk_Encode` calls).
    bytes: usize,
    snap: Option<EcEncSnapshot>,
    rdec: Box<SilkDecoder>,
    cdec: cdec::SilkDec,
    label: String,
    calls: usize,
}

impl Harness {
    fn new(channels: i32, label: &str) -> Self {
        let mut c = c::SilkEnc::new();
        let (cret, cst) = c.init(channels);
        let mut r = Box::new(SilkEncoder::new());
        let mut st = SilkEncControlStruct::default();
        let rret = r.init(channels, &mut st);
        assert_eq!(rret, cret, "{label}: init ret");
        assert_eq!(ctl_to_c(&st), cst, "{label}: init status");
        let mut h = Self {
            c,
            r,
            rbuf: vec![0u8; c::EC_BUF],
            bytes: 0,
            snap: None,
            rdec: Box::new(SilkDecoder::new()),
            cdec: cdec::SilkDec::new(),
            label: label.to_string(),
            calls: 0,
        };
        h.check_state("after init");
        h
    }

    fn reinit(&mut self, channels: i32) {
        let (cret, cst) = self.c.init(channels);
        let mut st = SilkEncControlStruct::default();
        let rret = self.r.init(channels, &mut st);
        assert_eq!(rret, cret, "{}: reinit ret", self.label);
        assert_eq!(ctl_to_c(&st), cst, "{}: reinit status", self.label);
        self.check_state("after reinit");
    }

    fn check_state(&mut self, what: &str) {
        let r = dump_state(&self.r);
        let c = self.c.dump();
        compare_dumps(
            &format!("{} [{what}, call {}]", self.label, self.calls),
            &r,
            &c,
        );
    }

    /// Starts a packet: both range encoders over zeroed buffers with `bytes` of storage.
    fn begin_packet(&mut self, bytes: usize) {
        self.c.ec_init(bytes);
        self.rbuf.fill(0);
        self.bytes = bytes;
        self.snap = Some(EcEnc::new(&mut self.rbuf[..bytes]).snapshot());
    }

    /// One `silk_Encode` call on both sides. Prefill calls pass a NULL range coder (C) / a
    /// dummy one (Rust), like the Opus encoder; other calls continue the current packet.
    /// Returns nBytesOut.
    fn call(
        &mut self,
        ctl: &mut SilkEncControlStruct,
        pcm: &[f32],
        n: i32,
        prefill: i32,
        activity: i32,
    ) -> i32 {
        self.calls += 1;
        let what = format!(
            "{} call {} (prefill {prefill}, {n} samples, ctl {:?})",
            self.label, self.calls, ctl
        );
        let mut cctl = ctl_to_c(ctl);
        let null_ec = prefill != 0;
        let nb_in = if null_ec { 0 } else { self.bytes as i32 };
        let (cret, cnb) = self
            .c
            .encode(&mut cctl, pcm, n, nb_in, prefill, activity, null_ec);

        let mut rnb = nb_in;
        let rret;
        if null_ec {
            let mut empty: [u8; 0] = [];
            let mut dummy = EcEnc::new(&mut empty);
            let before = ec_dump(&dummy);
            rret = self
                .r
                .silk_encode(ctl, pcm, n, &mut dummy, &mut rnb, prefill, activity);
            assert_eq!(
                ec_dump(&dummy),
                before,
                "{what}: prefill used the range coder"
            );
        } else {
            let mut enc = EcEnc::new(&mut self.rbuf[..self.bytes]);
            enc.restore(self.snap.as_ref().expect("packet started"));
            rret = self
                .r
                .silk_encode(ctl, pcm, n, &mut enc, &mut rnb, prefill, activity);
            self.snap = Some(enc.snapshot());
            assert_eq!(
                ec_dump(&enc),
                self.c.ec_state(),
                "{what}: range coder state"
            );
        }
        assert_eq!(rret, cret, "{what}: return code");
        assert_eq!(rnb, cnb, "{what}: nBytesOut");
        assert_eq!(ctl_to_c(ctl), cctl, "{what}: control struct");
        self.check_state("after encode");
        rnb
    }

    /// Finishes the packet (`ec_enc_done` on both sides) and compares the payload buffers.
    fn end_packet(&mut self) -> Vec<u8> {
        let mut enc = EcEnc::new(&mut self.rbuf[..self.bytes]);
        enc.restore(self.snap.as_ref().expect("packet started"));
        assert_eq!(
            ec_dump(&enc),
            self.c.ec_state(),
            "{}: range coder",
            self.label
        );
        enc.done();
        self.c.ec_done();
        assert_eq!(
            ec_dump(&enc),
            self.c.ec_state(),
            "{}: range coder after done (final range)",
            self.label
        );
        let cb = self.c.ec_buf(self.bytes);
        assert_slice_eq(
            &format!("{}: payload", self.label),
            &self.rbuf[..self.bytes],
            &cb,
        );
        self.snap = None;
        cb
    }

    /// Decodes a packet with the Rust and C SILK decoders and compares the output.
    fn decode(&mut self, ctl: &SilkEncControlStruct, payload: &[u8]) {
        let frames = (ctl.payload_size_ms / 20).max(1);
        let mut dctl = SilkDecControlStruct {
            n_channels_api: ctl.n_channels_api,
            n_channels_internal: ctl.n_channels_internal,
            api_sample_rate: ctl.api_sample_rate.min(48000),
            internal_sample_rate: ctl.internal_sample_rate,
            payload_size_ms: ctl.payload_size_ms,
            prev_pitch_lag: 0,
            enable_deep_plc: 0,
        };
        let mut rd = EcDec::new(payload);
        self.cdec.ec_init(payload);
        let mut rout = vec![0f32; 960 * 2];
        let mut cout = vec![0f32; 960 * 2];
        for f in 0..frames {
            let mut cctl = [
                dctl.n_channels_api,
                dctl.n_channels_internal,
                dctl.api_sample_rate,
                dctl.internal_sample_rate,
                dctl.payload_size_ms,
                dctl.prev_pitch_lag,
                dctl.enable_deep_plc,
            ];
            let mut rn = 0;
            let rret =
                self.rdec
                    .silk_decode(&mut dctl, 0, i32::from(f == 0), &mut rd, &mut rout, &mut rn);
            let (cret, cn) = self.cdec.decode(&mut cctl, 0, i32::from(f == 0), &mut cout);
            assert_eq!(rret, cret, "{}: decode ret", self.label);
            assert_eq!(rn, cn, "{}: decode n", self.label);
            let k = (rn * dctl.n_channels_api) as usize;
            opusorus_conformance::assert_bits_eq_f32(
                &format!("{}: decoded pcm", self.label),
                &rout[..k],
                &cout[..k],
            );
        }
    }
}

const RATE_SETS: [(i32, i32, i32); 7] = [
    (16000, 8000, 16000),
    (8000, 8000, 8000),
    (12000, 8000, 12000),
    (16000, 16000, 16000),
    (12000, 12000, 12000),
    (16000, 8000, 8000),
    (16000, 12000, 12000),
];

#[allow(
    clippy::too_many_lines,
    reason = "test driver mirrors the Opus encoder's use of silk_Encode"
)]
fn run(cfg: &Cfg) -> Cov {
    let label = format!("{cfg:?}");
    let mut h = Harness::new(cfg.api_ch, &label);
    let mut rng = Rng::new(cfg.seed ^ 0xC0FFEE);
    let ch = cfg.api_ch as usize;
    let total_ms = 60 * (cfg.n_packets + 2) + 10;
    // One signal per API rate (the API rate may change mid-stream), indexed by time in ms.
    let api_rates: Vec<i32> = if cfg.api_switch {
        vec![8000, 12000, 16000, 24000, 48000]
    } else {
        vec![cfg.api_fs]
    };
    let pcms: Vec<Vec<f32>> = api_rates
        .iter()
        .map(|&fs| gen_signal(cfg, fs, (fs as usize / 1000) * total_ms))
        .collect();
    let mut rate_ix = 0usize;
    let mut pos_ms = 0usize;
    let mut cov = Cov::default();

    let mut ctl = SilkEncControlStruct {
        n_channels_api: cfg.api_ch,
        n_channels_internal: cfg.int_ch,
        api_sample_rate: api_rates[0],
        max_internal_sample_rate: cfg.rates.0,
        min_internal_sample_rate: cfg.rates.1,
        desired_internal_sample_rate: cfg.rates.2,
        payload_size_ms: cfg.packet_ms,
        bit_rate: cfg.bitrate,
        packet_loss_percentage: cfg.loss,
        complexity: cfg.complexity,
        use_in_band_fec: i32::from(cfg.fec),
        lbrr_coded: i32::from(cfg.fec && cfg.loss > 0),
        use_dtx: i32::from(cfg.dtx),
        use_cbr: i32::from(cfg.cbr),
        reduced_dependency: i32::from(cfg.reduced_dependency),
        ..Default::default()
    };
    let mut int_ch = cfg.int_ch;

    for k in 0..cfg.n_packets {
        if cfg.vary && rng.range_i32(0, 5) == 0 {
            ctl.bit_rate = rng.range_i32(5000, 80000) * if int_ch == 2 { 2 } else { 1 };
            ctl.complexity = rng.range_i32(0, 10);
            let rs = RATE_SETS[rng.range_i32(0, RATE_SETS.len() as i32 - 1) as usize];
            ctl.max_internal_sample_rate = rs.0;
            ctl.min_internal_sample_rate = rs.1;
            ctl.desired_internal_sample_rate = rs.2;
            if !cfg.chunked {
                ctl.payload_size_ms = [10, 20, 40, 60][rng.range_i32(0, 3) as usize];
            }
            ctl.packet_loss_percentage = rng.range_i32(0, 30);
            ctl.use_in_band_fec = rng.range_i32(0, 1);
            ctl.lbrr_coded = ctl.use_in_band_fec & i32::from(ctl.packet_loss_percentage > 0);
            ctl.use_cbr = i32::from(rng.range_i32(0, 3) == 0);
            ctl.use_dtx = rng.range_i32(0, 1);
        }
        // API rate changes are not combined with internal channel switches: after an API rate
        // change in mono, the (uncontrolled) side-channel resampler still runs at the old rate
        // on the first mono frame (`nPrevChannelsInternal == 2`) and C writes out of bounds;
        // the Opus encoder never changes the API rate.
        if cfg.api_switch && !cfg.ch_switch && rng.range_i32(0, 7) == 0 {
            rate_ix = rng.range_i32(0, api_rates.len() as i32 - 1) as usize;
            ctl.api_sample_rate = api_rates[rate_ix];
        }
        if cfg.ch_switch && cfg.api_ch == 2 && rng.range_i32(0, 9) == 0 {
            if int_ch == 2 {
                // Opus sets toMono for one frame before switching to mono
                if ctl.to_mono == 0 {
                    ctl.to_mono = 1;
                } else {
                    ctl.to_mono = 0;
                    int_ch = 1;
                }
            } else {
                int_ch = 2;
            }
        } else if ctl.to_mono == 1 && rng.range_i32(0, 1) == 0 {
            ctl.to_mono = 0;
            int_ch = 1;
        }
        ctl.n_channels_internal = int_ch;
        ctl.opus_can_switch = ctl.switch_ready;
        let activity = match rng.range_i32(0, 5) {
            0 => 0,
            1 => 1,
            _ => -1,
        };
        let fs_khz = (ctl.api_sample_rate / 1000) as usize;
        let pcm = &pcms[rate_ix];

        // Prefill (10 ms of audio, NULL range coder), as the Opus encoder does on mode switches
        if cfg.prefill_every > 0 && k % cfg.prefill_every == cfg.prefill_every - 1 {
            let n10 = fs_khz * 10;
            let pf = 1 + (k / cfg.prefill_every) as i32 % 2;
            let s = pos_ms * fs_khz;
            h.call(
                &mut ctl,
                &pcm[s * ch..(s + n10) * ch],
                n10 as i32,
                pf,
                activity,
            );
            cov.prefill += 1;
        }

        // Payload budget: CBR -> exact size, VBR -> cap (or 1275)
        let target = (ctl.bit_rate * ctl.payload_size_ms / 8000).max(3) as usize;
        let max_bytes = if ctl.use_cbr != 0 {
            target + 1
        } else if cfg.cap > 0 {
            (cfg.cap as usize).max(target / 2 + 2)
        } else if cfg.vary && rng.range_i32(0, 4) == 0 {
            target * 3 / 4 + 2
        } else {
            1275
        };
        let max_bytes = max_bytes.min(1275);
        ctl.max_bits = (max_bytes as i32 - 1) * 8;
        h.begin_packet(max_bytes - 1);
        // Whole packet in one call, or (chunked) in 10 ms calls
        let pms = ctl.payload_size_ms as usize;
        let step = if cfg.chunked { 10 } else { pms };
        let mut nb = 0;
        let mut t = 0;
        while t < pms {
            // API rate change between the calls of one packet (control_encoder's
            // "controlled_since_last_payload" path re-initializes only the resamplers)
            if t > 0 && cfg.api_switch && !cfg.ch_switch && rng.range_i32(0, 3) == 0 {
                rate_ix = rng.range_i32(0, api_rates.len() as i32 - 1) as usize;
                ctl.api_sample_rate = api_rates[rate_ix];
            }
            let fs_khz = (ctl.api_sample_rate / 1000) as usize;
            let pcm = &pcms[rate_ix];
            let s = (pos_ms + t) * fs_khz;
            let n = step * fs_khz;
            nb = h.call(&mut ctl, &pcm[s * ch..(s + n) * ch], n as i32, 0, activity);
            t += step;
        }
        let payload = h.end_packet();
        pos_ms += pms;

        cov.packets += 1;
        if nb == 0 {
            cov.bytes_zero += 1;
        }
        let s0 = &h.r.state_fxx[0].s_cmn;
        if s0.lbrr_flag != 0 {
            cov.lbrr += 1;
        }
        if s0.in_dtx != 0 {
            cov.dtx += 1;
        }
        if int_ch == 2 {
            cov.stereo += 1;
            if h.r.s_stereo.mid_only_flags[0] != 0 {
                cov.mid_only += 1;
            }
        }
        match s0.indices.signal_type {
            2 => cov.voiced += 1,
            1 => cov.unvoiced += 1,
            _ => {}
        }
        cov.fs[match s0.fs_khz {
            8 => 0,
            12 => 1,
            _ => 2,
        }] += 1;
        // (The decoders are only fed streams with a fixed API rate, as in Opus: the C SILK
        // decoder does not support API rate changes mid-stream.)
        if cfg.decode && !cfg.api_switch && nb > 0 {
            let c2 = ctl;
            h.decode(&c2, &payload);
        }
        // Occasionally re-init mid stream (silk_InitEncoder as on OPUS_RESET_STATE)
        if cfg.vary && rng.range_i32(0, 60) == 0 {
            h.reinit(cfg.api_ch);
            int_ch = cfg.api_ch.min(int_ch);
            ctl.to_mono = 0;
        }
    }
    cov
}

// ---------------------------------------------------------------------------------------------
// Full encoder tests
// ---------------------------------------------------------------------------------------------

#[test]
fn encoder_sizes_and_init() {
    let [c1, c2] = c::encoder_sizes();
    assert!(c1 > 0 && c2 > c1);
    assert_eq!(c::ctl_size(), 26 * 4);
    let mut st = SilkEncControlStruct::default();
    for ch in [1, 2] {
        let mut h = Harness::new(ch, "init");
        h.reinit(ch);
        let mut e = SilkEncoder::new();
        assert_eq!(e.init(ch, &mut st), 0);
    }
    let mut s = SilkEncoderStateFlp::new();
    assert_eq!(silk_init_encoder(&mut s), 0);
    assert_eq!(s.s_cmn.first_frame_after_reset, 1);
    let mut sz = 0;
    opusorus::silk::encoder::silk_get_encoder_size(&mut sz, 2);
    let mut sz1 = 0;
    opusorus::silk::encoder::silk_get_encoder_size(&mut sz1, 1);
    assert!(sz > sz1 && sz1 > 0);
}

#[test]
fn mono_all_rates_packet_sizes() {
    let mut cov = Cov::default();
    let mut seed = 100;
    for &api_fs in &[8000, 12000, 16000, 24000, 48000] {
        for &pms in &[10, 20, 40, 60] {
            for (i, &rates) in RATE_SETS.iter().enumerate() {
                if !(i + pms as usize / 10).is_multiple_of(3) && i != 0 {
                    continue;
                }
                seed += 1;
                let mut cfg = Cfg::new(api_fs, 1, pms, 12000 + 4000 * i as i32, (seed % 11) as i32);
                cfg.rates = rates;
                cfg.seed = seed;
                cfg.n_packets = 4000 / pms as usize;
                cfg.sig = [Sig::Speech, Sig::Music, Sig::Noise][seed as usize % 3];
                cfg.decode = seed % 4 == 0;
                cov.add(&run(&cfg));
            }
        }
    }
    eprintln!("mono coverage: {cov:?}");
    assert!(cov.voiced > 100 && cov.unvoiced > 50);
    assert!(cov.fs.iter().all(|&n| n > 50), "{cov:?}");
}

#[test]
fn complexities_and_bitrates() {
    let mut cov = Cov::default();
    for cx in 0..=10 {
        for (j, &br) in [5000, 9000, 16000, 32000, 80000].iter().enumerate() {
            let mut cfg = Cfg::new(48000, 1, [20, 40, 10, 60, 20][j], br, cx);
            cfg.seed = 200 + (cx * 7 + j as i32) as u64;
            cfg.n_packets = 100;
            cfg.sig = if j % 2 == 0 { Sig::Speech } else { Sig::Music };
            cov.add(&run(&cfg));
        }
    }
    eprintln!("complexity coverage: {cov:?}");
}

#[test]
fn cbr_and_tight_vbr_caps() {
    let mut cov = Cov::default();
    for (i, &br) in [6000, 10000, 20000, 40000, 64000].iter().enumerate() {
        for &pms in &[10, 20, 60] {
            let mut cfg = Cfg::new(16000, 1, pms, br, [2, 5, 10][i % 3]);
            cfg.cbr = true;
            cfg.seed = 300 + i as u64 * 10 + pms as u64;
            cfg.n_packets = 120;
            cfg.sig = [Sig::Speech, Sig::Noise, Sig::Music, Sig::Loud, Sig::Speech][i];
            cov.add(&run(&cfg));
            // VBR with a cap well below what the signal wants
            let mut cfg = Cfg::new(24000, 1, pms, br, 9);
            cfg.cap = (br * pms / 8000 / 2).max(4);
            cfg.seed = 400 + i as u64 * 10 + pms as u64;
            cfg.n_packets = 120;
            cfg.sig = [Sig::Noise, Sig::Music, Sig::Loud, Sig::Speech, Sig::Noise][i];
            cov.add(&run(&cfg));
        }
    }
    eprintln!("cbr coverage: {cov:?}");
}

#[test]
fn fec_dtx_prefill_reduced_dependency() {
    let mut cov = Cov::default();
    for i in 0..12u64 {
        let mut cfg = Cfg::new(
            [8000, 16000, 48000][i as usize % 3],
            1,
            [20, 40, 60, 10][i as usize % 4],
            14000 + 3000 * i as i32,
            (i % 11) as i32,
        );
        cfg.fec = true;
        cfg.loss = [5, 10, 20, 30][i as usize % 4];
        cfg.dtx = i % 2 == 0;
        cfg.sig = if i % 3 == 0 { Sig::Speech } else { Sig::Bursty };
        cfg.prefill_every = [0, 7, 11][i as usize % 3];
        cfg.reduced_dependency = i % 5 == 0;
        cfg.seed = 500 + i;
        cfg.n_packets = 250;
        cfg.decode = i % 2 == 1;
        cov.add(&run(&cfg));
    }
    eprintln!("fec/dtx coverage: {cov:?}");
    assert!(cov.lbrr > 20, "{cov:?}");
    assert!(cov.bytes_zero > 5, "{cov:?}");
    assert!(cov.prefill > 10, "{cov:?}");
}

#[test]
fn stereo_streams() {
    let mut cov = Cov::default();
    for i in 0..12u64 {
        let mut cfg = Cfg::new(
            [16000, 24000, 48000, 12000][i as usize % 4],
            2,
            [20, 40, 60, 10][i as usize % 4],
            [12000, 24000, 40000, 60000][i as usize % 4],
            (i * 3 % 11) as i32,
        );
        cfg.sig = [Sig::StereoMix, Sig::Speech, Sig::Music, Sig::Bursty][i as usize % 4];
        cfg.ch_switch = i % 3 != 0;
        cfg.fec = i % 4 == 1;
        cfg.loss = 10;
        cfg.dtx = i % 4 == 3;
        cfg.cbr = i % 5 == 2;
        cfg.prefill_every = if i % 6 == 5 { 9 } else { 0 };
        cfg.seed = 600 + i;
        cfg.n_packets = 250;
        cfg.decode = i % 3 == 1;
        cov.add(&run(&cfg));
    }
    eprintln!("stereo coverage: {cov:?}");
    assert!(cov.stereo > 200 && cov.mid_only > 5, "{cov:?}");
}

#[test]
fn stereo_api_mono_internal_and_low_rates() {
    let mut cov = Cov::default();
    for i in 0..8u64 {
        let mut cfg = Cfg::new(
            [48000, 16000, 8000, 24000][i as usize % 4],
            2,
            [20, 60, 10, 40][i as usize % 4],
            [6000, 8000, 11000, 14000][i as usize % 4],
            10 - i as i32,
        );
        cfg.int_ch = if i % 2 == 0 { 1 } else { 2 };
        cfg.ch_switch = i % 2 == 1;
        cfg.sig = [Sig::StereoMix, Sig::Speech][i as usize % 2];
        cfg.seed = 700 + i;
        cfg.n_packets = 200;
        cov.add(&run(&cfg));
    }
    eprintln!("stereo/mono coverage: {cov:?}");
}

#[test]
fn randomized_parameter_changes() {
    let mut cov = Cov::default();
    for i in 0..10u64 {
        let api_ch = 1 + (i % 2) as i32;
        let mut cfg = Cfg::new(
            [8000, 12000, 16000, 24000, 48000][i as usize % 5],
            api_ch,
            20,
            20000 * api_ch,
            5,
        );
        cfg.vary = true;
        cfg.ch_switch = api_ch == 2;
        cfg.sig = [
            Sig::Speech,
            Sig::Music,
            Sig::Bursty,
            Sig::Noise,
            Sig::StereoMix,
        ][i as usize % 5];
        cfg.prefill_every = if i % 3 == 0 { 13 } else { 0 };
        cfg.seed = 800 + i;
        cfg.n_packets = 400;
        cfg.decode = i % 3 == 2;
        cov.add(&run(&cfg));
    }
    eprintln!("randomized coverage: {cov:?}");
    assert!(cov.fs.iter().all(|&n| n > 30), "{cov:?}");
}

#[test]
fn chunked_calls_and_api_rate_switches() {
    let mut cov = Cov::default();
    for i in 0..10u64 {
        let api_ch = 1 + (i % 2) as i32;
        let mut cfg = Cfg::new(
            48000,
            api_ch,
            [20, 10][i as usize % 2],
            18000 * api_ch,
            (i * 5 % 11) as i32,
        );
        cfg.chunked = i < 6;
        cfg.api_switch = true;
        cfg.vary = i % 3 == 0;
        cfg.ch_switch = api_ch == 2;
        cfg.fec = i % 4 == 1;
        cfg.loss = 15;
        cfg.dtx = i % 4 == 2;
        cfg.sig = [
            Sig::Speech,
            Sig::Music,
            Sig::Bursty,
            Sig::StereoMix,
            Sig::Noise,
        ][i as usize % 5];
        cfg.prefill_every = if i % 5 == 4 { 6 } else { 0 };
        cfg.seed = 1000 + i;
        cfg.n_packets = 250;
        cfg.decode = i % 4 == 3;
        cov.add(&run(&cfg));
    }
    eprintln!("chunked/api-switch coverage: {cov:?}");
}

#[test]
fn garbage_and_extreme_input() {
    let mut cov = Cov::default();
    for i in 0..8u64 {
        let api_ch = 1 + (i % 2) as i32;
        let mut cfg = Cfg::new(
            [48000, 16000, 8000, 24000][i as usize % 4],
            api_ch,
            [20, 60, 10, 40][i as usize % 4],
            [8000, 20000, 40000, 70000][i as usize % 4] * api_ch,
            (i * 3 % 11) as i32,
        );
        cfg.sig = Sig::Garbage;
        cfg.cbr = i % 3 == 0;
        cfg.fec = i % 3 == 1;
        cfg.loss = 20;
        cfg.dtx = i % 2 == 0;
        cfg.ch_switch = api_ch == 2;
        cfg.seed = 1200 + i;
        cfg.n_packets = 120;
        cfg.decode = true;
        cov.add(&run(&cfg));
    }
    eprintln!("garbage coverage: {cov:?}");
}

#[test]
fn pure_tones() {
    let mut cov = Cov::default();
    let freqs = [
        50u32, 97, 150, 220, 441, 1000, 1500, 2900, 3700, 5100, 6800, 7900,
    ];
    for (i, &f) in freqs.iter().enumerate() {
        let mut cfg = Cfg::new(
            [16000, 48000, 24000][i % 3],
            1,
            [20, 10, 40, 60][i % 4],
            [8000, 16000, 32000, 64000][i % 4],
            [10, 5, 2, 0, 7, 3][i % 6],
        );
        cfg.sig = Sig::Tone(f);
        cfg.rates = RATE_SETS[i % RATE_SETS.len()];
        cfg.cbr = i % 3 == 1;
        cfg.seed = 1300 + i as u64;
        cfg.n_packets = 100;
        cov.add(&run(&cfg));
    }
    eprintln!("tone coverage: {cov:?}");
}

#[test]
fn long_streams() {
    let mut cov = Cov::default();
    for i in 0..4u64 {
        let api_ch = 1 + (i / 2) as i32;
        let mut cfg = Cfg::new(
            [16000, 48000][i as usize % 2],
            api_ch,
            20,
            24000 * api_ch,
            [10, 6][i as usize % 2],
        );
        cfg.sig = [Sig::Speech, Sig::Music, Sig::StereoMix, Sig::Bursty][i as usize];
        cfg.fec = true;
        cfg.loss = 8;
        cfg.dtx = i % 2 == 1;
        cfg.seed = 1100 + i;
        cfg.n_packets = 1000;
        cov.add(&run(&cfg));
    }
    eprintln!("long stream coverage: {cov:?}");
}

#[cfg(feature = "qext")]
#[test]
fn qext_96k_api() {
    let mut cov = Cov::default();
    for i in 0..4u64 {
        let mut cfg = Cfg::new(
            96000,
            1 + (i % 2) as i32,
            [20, 10, 40, 60][i as usize],
            24000,
            8,
        );
        cfg.rates = (16000, 16000, 16000);
        cfg.seed = 900 + i;
        cfg.n_packets = 150;
        cov.add(&run(&cfg));
    }
    eprintln!("96k coverage: {cov:?}");
}

/// Invalid inputs: the C oracle aborts (hardened `celt_assert`), so only the Rust error codes
/// are checked, and that the encoder still works afterwards.
#[test]
fn invalid_input_rust_only() {
    use opusorus::silk::errors::*;
    let mut e = SilkEncoder::new();
    let mut st = SilkEncControlStruct::default();
    assert_eq!(e.init(1, &mut st), 0);
    let base = SilkEncControlStruct {
        n_channels_api: 1,
        n_channels_internal: 1,
        api_sample_rate: 16000,
        max_internal_sample_rate: 16000,
        min_internal_sample_rate: 8000,
        desired_internal_sample_rate: 16000,
        payload_size_ms: 20,
        bit_rate: 20000,
        complexity: 5,
        max_bits: 8000,
        ..Default::default()
    };
    let pcm = vec![0.01f32; 2000];
    let mut buf = vec![0u8; 500];
    let mut run = |ctl: SilkEncControlStruct, n: i32, prefill: i32| {
        let mut c = ctl;
        let mut enc = EcEnc::new(&mut buf);
        let mut nb = 499;
        e.silk_encode(&mut c, &pcm, n, &mut enc, &mut nb, prefill, -1)
    };
    // With the `assertions` feature (ENABLE_ASSERTIONS) every error path below fails a
    // `celt_assert( 0 )` and panics, like C's abort (checked on a fresh encoder each time).
    let mut check = |ctl: SilkEncControlStruct, n: i32, prefill: i32, want: i32| {
        if cfg!(feature = "assertions") {
            let r = std::panic::catch_unwind(|| {
                let mut e = SilkEncoder::new();
                let mut st = SilkEncControlStruct::default();
                assert_eq!(e.init(1, &mut st), 0);
                let mut buf = vec![0u8; 500];
                let mut c = ctl;
                let mut enc = EcEnc::new(&mut buf);
                let mut nb = 499;
                e.silk_encode(&mut c, &pcm, n, &mut enc, &mut nb, prefill, -1)
            });
            assert!(r.is_err(), "{ctl:?}: no assertion failure (result {r:?})");
        } else {
            assert_eq!(run(ctl, n, prefill), want, "{ctl:?}");
        }
    };
    // not a multiple of 10 ms
    check(base, 170, 0, SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES);
    // more than one packet
    check(base, 640, 0, SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES);
    // prefill must be 10 ms
    check(base, 320, 1, SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES);
    // bad packet size / rates / complexity
    let mut c = base;
    c.payload_size_ms = 30;
    check(c, 160, 0, SILK_ENC_PACKET_SIZE_NOT_SUPPORTED);
    let mut c = base;
    c.api_sample_rate = 11025;
    check(c, 441, 0, SILK_ENC_FS_NOT_SUPPORTED);
    // 44.1 kHz passes check_control_input but has no resampler: silk_resampler_init returns -1
    // (C asserts there)
    let mut c = base;
    c.api_sample_rate = 44100;
    check(c, 441, 0, -1);
    let mut c = base;
    c.max_internal_sample_rate = 8000;
    check(c, 320, 0, SILK_ENC_FS_NOT_SUPPORTED);
    let mut c = base;
    c.packet_loss_percentage = 101;
    check(c, 320, 0, SILK_ENC_INVALID_LOSS_RATE);
    // (nChannelsAPI > 2 is not testable: C indexes state_Fxx[ nChannelsAPI - 1 ] before
    // validating it)
    let mut c = base;
    c.use_dtx = 2;
    check(c, 320, 0, SILK_ENC_INVALID_DTX_SETTING);
    let mut c = base;
    c.complexity = 11;
    check(c, 320, 0, SILK_ENC_INVALID_COMPLEXITY_SETTING);
    // still works
    assert_eq!(run(base, 320, 0), 0);
}

/// Exhaustive random sweep (all parameters randomized per stream and per packet):
/// `cargo test -p opusorus-conformance --test silk_encoder_flp -- --ignored sweep`.
#[test]
#[ignore = "exhaustive sweep, about a minute"]
fn exhaustive_random_sweep() {
    let cov = random_sweep(0x5EED, 400);
    eprintln!("sweep coverage: {cov:?}");
}

/// A smaller random sweep in the default run.
#[test]
fn random_sweep_small() {
    let cov = random_sweep(0xBEEF, 50);
    eprintln!("small sweep coverage: {cov:?}");
    assert!(cov.lbrr > 0 && cov.stereo > 0 && cov.prefill > 0, "{cov:?}");
}

fn random_sweep(seed: u64, streams: u64) -> Cov {
    let mut rng = Rng::new(seed);
    let mut cov = Cov::default();
    for i in 0..streams {
        let api_ch = rng.range_i32(1, 2);
        let api_fs = [8000, 12000, 16000, 24000, 48000][rng.range_i32(0, 4) as usize];
        let mut cfg = Cfg::new(
            api_fs,
            api_ch,
            [10, 20, 40, 60][rng.range_i32(0, 3) as usize],
            rng.range_i32(5000, 80000) * api_ch,
            rng.range_i32(0, 10),
        );
        cfg.int_ch = rng.range_i32(1, api_ch);
        cfg.rates = RATE_SETS[rng.range_i32(0, RATE_SETS.len() as i32 - 1) as usize];
        cfg.sig = match rng.range_i32(0, 7) {
            0 => Sig::Speech,
            1 => Sig::Music,
            2 => Sig::Noise,
            3 => Sig::Bursty,
            4 => Sig::Loud,
            5 => Sig::StereoMix,
            6 => Sig::Garbage,
            _ => Sig::Tone(rng.range_i32(40, 7900) as u32),
        };
        cfg.cbr = rng.range_i32(0, 3) == 0;
        cfg.cap = if rng.range_i32(0, 3) == 0 {
            rng.range_i32(4, 200)
        } else {
            0
        };
        cfg.fec = rng.range_i32(0, 1) == 1;
        cfg.loss = rng.range_i32(0, 40);
        cfg.dtx = rng.range_i32(0, 1) == 1;
        cfg.vary = rng.range_i32(0, 1) == 1;
        cfg.ch_switch = api_ch == 2 && rng.range_i32(0, 1) == 1;
        cfg.reduced_dependency = rng.range_i32(0, 4) == 0;
        cfg.prefill_every = [0, 0, 5, 17][rng.range_i32(0, 3) as usize];
        cfg.chunked = cfg.packet_ms <= 20 && rng.range_i32(0, 3) == 0;
        cfg.api_switch = rng.range_i32(0, 5) == 0;
        cfg.decode = rng.range_i32(0, 3) == 0;
        cfg.seed = seed ^ (5000 + i);
        cfg.n_packets = 150;
        cov.add(&run(&cfg));
    }
    cov
}

/// Rough speed comparison (not a correctness test): `cargo test --release -p
/// opusorus-conformance --test silk_encoder_flp -- --ignored --nocapture perf`.
#[test]
#[ignore = "timing only"]
fn perf_compare() {
    for (api_fs, ch, cx) in [
        (16000, 1, 10),
        (48000, 1, 10),
        (48000, 2, 10),
        (16000, 1, 2),
    ] {
        let mut cfg = Cfg::new(api_fs, ch, 20, 24000 * ch, cx);
        cfg.sig = Sig::Speech;
        let n_packets = 1500;
        let n = (api_fs / 50) as usize;
        let pcm = gen_signal(&cfg, api_fs, n * n_packets);
        let ctl0 = SilkEncControlStruct {
            n_channels_api: ch,
            n_channels_internal: ch,
            api_sample_rate: api_fs,
            max_internal_sample_rate: 16000,
            min_internal_sample_rate: 8000,
            desired_internal_sample_rate: 16000,
            payload_size_ms: 20,
            bit_rate: cfg.bitrate,
            complexity: cx,
            max_bits: 1274 * 8,
            ..Default::default()
        };
        let chu = ch as usize;
        let mut ce = c::SilkEnc::new();
        ce.init(ch);
        let t0 = std::time::Instant::now();
        let mut cctl = ctl_to_c(&ctl0);
        for k in 0..n_packets {
            ce.ec_init(1274);
            ce.encode(
                &mut cctl,
                &pcm[k * n * chu..(k + 1) * n * chu],
                n as i32,
                1274,
                0,
                -1,
                false,
            );
        }
        let tc = t0.elapsed();
        let mut re = Box::new(SilkEncoder::new());
        let mut st = SilkEncControlStruct::default();
        re.init(ch, &mut st);
        let mut buf = vec![0u8; 1274];
        let mut ctl = ctl0;
        let t0 = std::time::Instant::now();
        for k in 0..n_packets {
            let mut enc = EcEnc::new(&mut buf);
            let mut nb = 1274;
            re.silk_encode(
                &mut ctl,
                &pcm[k * n * chu..(k + 1) * n * chu],
                n as i32,
                &mut enc,
                &mut nb,
                0,
                -1,
            );
        }
        let tr = t0.elapsed();
        eprintln!(
            "api {api_fs} ch {ch} cx {cx}: C {:.1} ms, Rust {:.1} ms ({:.2}x) for {} s of audio",
            tc.as_secs_f64() * 1e3,
            tr.as_secs_f64() * 1e3,
            tr.as_secs_f64() / tc.as_secs_f64(),
            n_packets / 50
        );
    }
}

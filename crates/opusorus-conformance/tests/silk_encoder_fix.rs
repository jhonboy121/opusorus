//! Differential tests for unit `silk_encoder_fix`: the fixed-point SILK encoder
//! (`silk/fixed/*.c`, `silk/enc_API.c`, `silk/init_encoder.c`, `silk/control_codec.c` in a
//! `FIXED_POINT` build) vs the fixed-point C oracle (features `fixed-point` / `fixed-res24`,
//! with or without `qext`). Every comparison is exact.
//!
//! * Leaf DSP functions (sine window, autocorrelation, Burg, Schur / Schur64, k2a / k2a_Q16,
//!   vector ops, correlation matrix / vector, residual energies, correlation regularization,
//!   warped autocorrelation, LTP analysis filter, LTP analysis, LPC analysis, the pitch
//!   analyser and its static stage-3 helpers, the static warped-coefficient helpers of the
//!   noise shaping analysis): realistic inputs (the signals the encoder feeds them, from
//!   near-silence to full scale), within the domain where the C code has no signed overflow.
//! * Full encoder: long multi-packet streams through `silk_Encode` on both sides with identical
//!   control parameters (API rates 8..48 kHz (+96 kHz with qext), NB/MB/WB internal rates with
//!   switching, 10/20/40/60 ms, complexity 0..10, 5..80 kb/s, CBR/VBR with tight caps, FEC,
//!   DTX on silence, mono/stereo incl. mono<->stereo switches and toMono, reducedDependency,
//!   prefill, API rate switches): return codes, nBytesOut, the control struct, the range
//!   encoder state, the finished payload bytes, final range, and the complete encoder state
//!   after every call. Some payloads are also decoded with the Rust and C SILK decoders.
//!
//! Run with `cargo test -p opusorus-conformance --features fixed-point --test
//! silk_encoder_fix` (also `fixed-res24`, and either with `qext`).

#![cfg(feature = "fixed-point")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "test code: failures should panic; index loops mirror the C layouts"
)]

use opusorus::celt::arch::OpusRes;
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::{EcEnc, EcEncSnapshot};
use opusorus::silk::decoder::SilkDecoder;
use opusorus::silk::encoder::{SilkEncoder, silk_init_encoder};
use opusorus::silk::fixed::{self as fix, SilkEncoderStateFix};
use opusorus::silk::resampler::{ResamplerFunction, SilkResamplerState};
use opusorus::silk::sigproc::silk_bwexpander_32;
use opusorus::silk::structs::{SideInfoIndices, SilkDecControlStruct, SilkEncControlStruct};
use opusorus_conformance::{Rng, assert_slice_eq, signals};
use opusorus_oracle::silk_decoder as cdec;
use opusorus_oracle::silk_encoder_fix as c;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Float PCM in [-1, 1] (or beyond, saturated) to `opus_res` of the build.
#[cfg(not(feature = "fixed-res24"))]
fn to_res(x: f32) -> OpusRes {
    (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Float PCM in [-1, 1] (or beyond) to `opus_res` of the build (24-bit resolution, bounded so
/// that the stereo downmix sum and `RES2INT16`'s rounding add cannot overflow).
#[cfg(feature = "fixed-res24")]
fn to_res(x: f32) -> OpusRes {
    (x as f64 * 8_388_608.0)
        .round()
        .clamp(-((1i32 << 29) as f64), (1i32 << 29) as f64) as i32
}

/// Float signal to `opus_int16` with gain `amp` (saturating).
fn to_i16(x: &[f32], amp: f32) -> Vec<i16> {
    x.iter()
        .map(|&v| (v * amp).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// A realistic int16 test signal of `n` samples: speech / music / noise / resonant AR /
/// pulse trains, with an amplitude from near-silence to full scale.
fn test_signal(rng: &mut Rng, n: usize, fs: u32) -> Vec<i16> {
    let seed = rng.next_u64();
    let amp = match rng.range_i32(0, 7) {
        0 => 3.0,
        1 => 60.0,
        2 => 800.0,
        3 => 32767.0,
        _ => 12000.0,
    };
    let x: Vec<f32> = match rng.range_i32(0, 4) {
        0 => signals::speech_like(n, 1, fs, seed),
        1 => signals::music_like(n, 1, fs, seed),
        2 => signals::noise(n, 1, 0.5, seed),
        3 => ar_signal(rng, n),
        _ => {
            let period = rng.range_i32(20, 280) as usize;
            let mut v = signals::noise(n, 1, 0.05, seed);
            for i in (0..n).step_by(period) {
                v[i] += 0.9;
            }
            v
        }
    };
    to_i16(&x, amp)
}

/// AR(2) resonance filtered noise in [-1, 1], plus a pitch pulse train sometimes.
fn ar_signal(rng: &mut Rng, n: usize) -> Vec<f32> {
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
        *v /= m;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Leaf functions
// ---------------------------------------------------------------------------------------------

#[test]
fn sine_window_scale_inner_prod() {
    let mut rng = Rng::new(1);
    for it in 0..600 {
        let length = 16 + 4 * rng.range_i32(0, 26) as usize;
        let x = test_signal(&mut rng, length, 16000);
        for win_type in [1, 2] {
            let mut r = vec![0i16; length];
            let mut cc = vec![0i16; length];
            fix::silk_apply_sine_window(&mut r, &x, win_type, length);
            c::apply_sine_window(&mut cc, &x, win_type, length);
            assert_slice_eq(
                &format!("sine window {it} type {win_type} len {length}"),
                &r,
                &cc,
            );
        }
        // scale_copy_vector16 (gains up to the 16-bit inverse gains of find_pred_coefs)
        let n = rng.range_i32(1, 120) as usize;
        let gain = rng.range_i32(100, 32767);
        let mut r = vec![0i16; n];
        let mut cc = vec![0i16; n];
        fix::silk_scale_copy_vector16(&mut r, &x, gain, n.min(length));
        c::scale_copy_vector16(&mut cc, &x, gain, n.min(length));
        assert_slice_eq("scale_copy_vector16", &r, &cc);
        // scale_vector32_Q26_lshift_18
        let mut d: Vec<i32> = (0..n).map(|_| rng.range_i32(-(1 << 24), 1 << 24)).collect();
        let mut dc = d.clone();
        let g = rng.range_i32(-(1 << 26), 1 << 26);
        fix::silk_scale_vector32_q26_lshift_18(&mut d, g, n);
        c::scale_vector32_q26_lshift_18(&mut dc, g, n);
        assert_slice_eq("scale_vector32_Q26_lshift_18", &d, &dc);
        // inner_prod_aligned (no-overflow domain)
        let len = rng.range_i32(1, 320) as usize;
        let a = to_i16(&signals::noise(len, 1, 1.0, it), 2000.0);
        let b = to_i16(&signals::noise(len, 1, 1.0, it + 7), 2000.0);
        assert_eq!(
            fix::silk_inner_prod_aligned(&a, &b, len),
            c::inner_prod_aligned(&a, &b, len),
            "inner_prod_aligned"
        );
    }
}

#[test]
fn autocorr_schur_k2a() {
    let mut rng = Rng::new(2);
    for it in 0..1500 {
        // pitch LPC windows (<= 384) and noise shaping windows (<= 240)
        let size = [96, 144, 192, 224, 288, 384, 120, 180, 240][it % 9];
        let x = test_signal(&mut rng, size, 16000);
        let count = rng.range_i32(2, 25) as usize;
        let mut r = vec![0i32; 25];
        let mut cc = vec![0i32; 25];
        let rs = fix::silk_autocorr(&mut r, &x, size, count);
        let cs = c::autocorr(&mut cc, &x, size, count);
        assert_eq!(rs, cs, "autocorr scale {it}");
        assert_slice_eq(&format!("autocorr {it}"), &r, &cc);

        let order = count - 1;
        let mut ac = r.clone();
        // white noise as in find_pitch_lags / noise_shape_analysis
        ac[0] = ac[0] + (ac[0] >> 10) + 1;
        if order == 0 {
            continue;
        }
        // schur (Q15) + k2a
        let mut rc_r = [0i16; 24];
        let mut rc_c = [0i16; 24];
        let nr = fix::silk_schur(&mut rc_r, &ac, order);
        let nc = c::schur(&mut rc_c, &ac, order);
        assert_eq!(nr, nc, "schur nrg {it}");
        assert_slice_eq("schur rc", &rc_r, &rc_c);
        let mut a_r = [0i32; 24];
        let mut a_c = [0i32; 24];
        fix::silk_k2a(&mut a_r, &rc_r, order);
        c::k2a(&mut a_c, &rc_c, order);
        assert_slice_eq("k2a", &a_r, &a_c);
        // schur64 (Q16) + k2a_Q16
        let mut rc_r = [0i32; 24];
        let mut rc_c = [0i32; 24];
        let nr = fix::silk_schur64(&mut rc_r, &ac, order);
        let nc = c::schur64(&mut rc_c, &ac, order);
        assert_eq!(nr, nc, "schur64 nrg {it}");
        assert_slice_eq("schur64 rc", &rc_r, &rc_c);
        let mut a_r = [0i32; 24];
        let mut a_c = [0i32; 24];
        fix::silk_k2a_q16(&mut a_r, &rc_r, order);
        c::k2a_q16(&mut a_c, &rc_c, order);
        assert_slice_eq("k2a_Q16", &a_r, &a_c);
    }
    // schur64 invalid input
    let mut rc = [7i32; 4];
    assert_eq!(fix::silk_schur64(&mut rc, &[0, 1, 2, 3, 4], 4), 0);
    assert_eq!(rc, [0; 4]);
}

#[test]
fn burg_modified() {
    let mut rng = Rng::new(3);
    let mut branches = [0usize; 3];
    for it in 0..3000 {
        let d = [10usize, 16, 6, 12, 24][it % 5];
        let sub = [40usize, 60, 80, 30][rng.range_i32(0, 3) as usize] + d;
        let nb_subfr = if sub * 4 <= 384 && rng.range_i32(0, 2) > 0 {
            4
        } else {
            2
        };
        let x = test_signal(&mut rng, sub * nb_subfr, 16000);
        let min_inv_gain = match rng.range_i32(0, 3) {
            0 => 10737418, // 1/100 (after reset)
            1 => 107374,   // 1/1e4
            2 => rng.range_i32(1 << 16, 1 << 28),
            _ => rng.range_i32(1000, 1 << 20),
        };
        let mut ar = [0i32; 24];
        let mut ac = [0i32; 24];
        let r = fix::silk_burg_modified(&mut ar, &x, min_inv_gain, sub, nb_subfr, d);
        let cc = c::burg_modified(&mut ac, &x, min_inv_gain, sub, nb_subfr, d);
        assert_eq!(r, cc, "burg nrg/Q {it} (d {d}, sub {sub}, nb {nb_subfr})");
        assert_slice_eq(&format!("burg A {it}"), &ar[..d], &ac[..d]);
        branches[if -r.1 > 0 {
            0
        } else if -r.1 > -2 {
            1
        } else {
            2
        }] += 1;
    }
    eprintln!("burg rshifts branches (>0, -1..0, <=-2): {branches:?}");
    assert!(branches.iter().all(|&n| n > 50), "{branches:?}");
}

#[test]
fn corr_matrix_vector_ltp() {
    let mut rng = Rng::new(4);
    for it in 0..1500 {
        let order = [5usize, 5, 5, 2, 10, 16][it % 6];
        let l = [40usize, 60, 80, 30][rng.range_i32(0, 3) as usize];
        let x = test_signal(&mut rng, l + order - 1 + l, 16000);
        let mut xr = vec![0i32; order * order];
        let mut xc = vec![0i32; order * order];
        let r = fix::silk_corr_matrix_fix(&x, l, order, &mut xr);
        let cc = c::corr_matrix(&x, l, order, &mut xc);
        assert_eq!(r, cc, "corrMatrix nrg/rshifts {it}");
        assert_slice_eq(&format!("corrMatrix {it}"), &xr, &xc);
        let t = &x[order - 1..];
        // (rshifts == 0 only when the energy fits: the C inner product has no headroom)
        for rshifts in [r.1, r.1 + 1] {
            let mut vr = vec![0i32; order];
            let mut vc = vec![0i32; order];
            fix::silk_corr_vector_fix(&x, t, l, order, &mut vr, rshifts);
            c::corr_vector(&x, t, l, order, &mut vc, rshifts);
            assert_slice_eq(&format!("corrVector {it} rshifts {rshifts}"), &vr, &vc);
        }
        if order <= 16 && order > 0 {
            // residual_energy16_covar with realistic predictors
            let cq = rng.range_i32(8, 15);
            let cvec: Vec<i16> = (0..order)
                .map(|_| (rng.f32_sym() * 0.9 * (1 << cq) as f32) as i16)
                .collect();
            let mut xv = vec![0i32; order];
            fix::silk_corr_vector_fix(&x, t, l, order, &mut xv, r.1);
            let wxx = r.0;
            assert_eq!(
                fix::silk_residual_energy16_covar_fix(&cvec, &xr, &xv, wxx, order, cq),
                c::residual_energy16_covar(&cvec, &xr, &xv, wxx, order, cq),
                "residual_energy16_covar {it}"
            );
            // regularize
            let mut mr = xr.clone();
            let mut mc = xr.clone();
            let mut vr = xv.clone();
            let mut vc = xv.clone();
            let noise = rng.range_i32(0, 1000);
            fix::silk_regularize_correlations_fix(&mut mr, &mut vr, noise, order);
            c::regularize_correlations(&mut mc, &mut vc, noise, order);
            assert_slice_eq("regularize XX", &mr, &mc);
            assert_slice_eq("regularize xx", &vr, &vc);
        }
    }
    // LTP analysis + LTP analysis filter on LPC-residual-like signals
    for it in 0..1500 {
        let nb_subfr = if it % 3 == 0 { 2 } else { 4 };
        let sub = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let max_lag = sub / 5 * 18;
        let off = max_lag + 16 + 2;
        let x = test_signal(&mut rng, off + nb_subfr * sub + 16, 16000);
        let mut lag = [0i32; 4];
        for k in 0..nb_subfr {
            lag[k] = rng.range_i32((sub / 5 * 2) as i32, max_lag as i32);
        }
        let (xr, vr) = {
            let mut xx = [0i32; 100];
            let mut xv = [0i32; 20];
            fix::silk_find_ltp_fix(&mut xx, &mut xv, &x, off, &lag, sub, nb_subfr);
            (xx, xv)
        };
        let (xc, vc) = c::find_ltp(&x, off, &lag, sub, nb_subfr);
        assert_slice_eq(&format!("find_LTP XX {it}"), &xr, &xc);
        assert_slice_eq(&format!("find_LTP xX {it}"), &vr, &vc);

        let pre = [10usize, 16][it % 2];
        let mut b = [0i16; 20];
        for v in &mut b {
            *v = (rng.f32_sym() * 12000.0) as i16;
        }
        let mut inv = [0i32; 4];
        for v in &mut inv {
            *v = rng.range_i32(100, 32767);
        }
        let mut rr = vec![0i16; nb_subfr * (sub + pre)];
        let mut rc = vec![0i16; nb_subfr * (sub + pre)];
        fix::silk_ltp_analysis_filter_fix(
            &mut rr,
            &x,
            off - pre,
            &b,
            &lag,
            &inv,
            sub,
            nb_subfr,
            pre,
        );
        c::ltp_analysis_filter(&mut rc, &x, off - pre, &b, &lag, &inv, sub, nb_subfr, pre);
        assert_slice_eq(&format!("LTP_analysis_filter {it}"), &rr, &rc);
    }
}

#[test]
fn warped_autocorrelation_and_coef_limiters() {
    let mut rng = Rng::new(5);
    let mut limited = 0;
    for it in 0..1500 {
        let length = [120usize, 180, 240, 72, 150][it % 5];
        let order = 2 * rng.range_i32(1, 12) as usize;
        let x = test_signal(&mut rng, length, 16000);
        let mut xw = vec![0i16; length];
        let slope = (length - 48) / 2 / 4 * 4;
        fix::silk_apply_sine_window(&mut xw, &x, 1, slope.clamp(16, 120));
        let warping = rng.range_i32(0, 20000);
        let mut r = [0i32; 25];
        let mut cc = [0i32; 25];
        let rs = fix::silk_warped_autocorrelation_fix(&mut r, &xw, warping, length, order);
        let cs = c::warped_autocorrelation(&mut cc, &xw, warping, length, order);
        assert_eq!(rs, cs, "warped autocorr scale {it}");
        assert_slice_eq(&format!("warped autocorr {it}"), &r, &cc);

        // noise_shape_analysis chain: white noise, schur64, k2a_Q16, bwexpander, then the
        // warped gain / coefficient limiter
        r[0] += ((((r[0] >> 4) as i64 * 31) >> 16) as i32).max(1);
        let mut rc = [0i32; 24];
        fix::silk_schur64(&mut rc, &r, order);
        let mut ar = [0i32; 24];
        fix::silk_k2a_q16(&mut ar, &rc, order);
        assert_eq!(
            fix::warped_gain(&ar, warping, order),
            c::warped_gain(&ar, warping, order),
            "warped_gain {it}"
        );
        silk_bwexpander_32(&mut ar, order, rng.range_i32(50000, 65536));
        let mut a2 = ar;
        let limit = [65_520_000, rng.range_i32(1 << 22, 1 << 26)][it % 2];
        fix::limit_warped_coefs(&mut ar, warping, limit, order);
        c::limit_warped_coefs(&mut a2, warping, limit, order);
        assert_slice_eq(&format!("limit_warped_coefs {it}"), &ar, &a2);
        if it % 2 == 1 {
            limited += 1;
        }
    }
    assert!(limited > 0);
}

#[test]
fn residual_energy_and_find_lpc() {
    let mut rng = Rng::new(6);
    let mut interp = 0;
    for it in 0..800 {
        let order = [10usize, 16][it % 2];
        let sub = [40usize, 60, 80][rng.range_i32(0, 2) as usize];
        let nb_subfr = if it % 5 == 0 { 2 } else { 4 };
        let x = test_signal(&mut rng, nb_subfr * (sub + order), 16000);
        // A_Q12 from the Burg analysis of the signal itself (stable) or random small values
        let mut a = [[0i16; 16]; 2];
        for h in 0..2 {
            for i in 0..order {
                a[h][i] = (rng.f32_sym() * 1500.0 / (i + 1) as f32) as i16;
            }
        }
        let mut gains = [0i32; 4];
        for g in &mut gains {
            *g = rng.range_i32(1 << 10, 1 << 24);
        }
        let mut nr = [0i32; 4];
        let mut qr = [0i32; 4];
        fix::silk_residual_energy_fix(&mut nr, &mut qr, &x, &a, &gains, sub, nb_subfr, order);
        let (nc, qc) = c::residual_energy(&x, &a, &gains, sub, nb_subfr, order);
        assert_slice_eq(
            &format!("residual_energy nrgs {it}"),
            &nr[..nb_subfr],
            &nc[..nb_subfr],
        );
        assert_slice_eq(
            &format!("residual_energy Q {it}"),
            &qr[..nb_subfr],
            &qc[..nb_subfr],
        );

        // find_LPC with and without NLSF interpolation
        let mut prev = [0i16; 16];
        for (i, p) in prev.iter_mut().enumerate().take(order) {
            *p = ((i as i32 + 1) * 32767 / (order as i32 + 1)) as i16
                + (rng.f32_sym() * 400.0) as i16;
        }
        let use_interp = it % 3 != 0;
        let first = it % 7 == 0;
        let min_inv = [10737418, 107374, rng.range_i32(100_000, 10_000_000)][it % 3];
        let mut st = opusorus::silk::structs::SilkEncoderState::new();
        st.predict_lpc_order = order as i32;
        st.subfr_length = sub as i32;
        st.nb_subfr = nb_subfr as i32;
        st.use_interpolated_nlsfs = i32::from(use_interp);
        st.first_frame_after_reset = i32::from(first);
        st.prev_nlsfq_q15 = prev;
        let mut nlsf_r = [0i16; 16];
        fix::silk_find_lpc_fix(&mut st, &mut nlsf_r, &x, min_inv);
        let (nlsf_c, ic) = c::find_lpc(order, sub, nb_subfr, use_interp, first, &prev, &x, min_inv);
        assert_eq!(
            st.indices.nlsf_interp_coef_q2 as i32, ic,
            "find_LPC interp {it}"
        );
        assert_slice_eq(
            &format!("find_LPC NLSF {it}"),
            &nlsf_r[..order],
            &nlsf_c[..order],
        );
        if ic != 4 {
            interp += 1;
        }
    }
    assert!(interp > 20, "NLSF interpolation chosen only {interp} times");
}

/// A pitch-analysis input frame: `(20 + nb_subfr * 5) * fs_khz` samples of an LPC residual
/// like signal (voiced pulse trains with jitter, noise, silence).
fn pitch_frame(rng: &mut Rng, fs_khz: i32, nb_subfr: i32, kind: i32) -> Vec<i16> {
    let n = ((20 + nb_subfr * 5) * fs_khz) as usize;
    let amp = [2.0f32, 40.0, 1500.0, 30000.0][rng.range_i32(0, 3) as usize];
    let mut x = vec![0f32; n];
    match kind % 4 {
        0 | 1 => {
            let mut period = rng.range_i32(2 * fs_khz, 18 * fs_khz) as f32;
            let mut t = 0f32;
            let noise = signals::noise(n, 1, 0.1, rng.next_u64());
            for i in 0..n {
                x[i] = noise[i];
                if (i as f32) >= t {
                    x[i] += 1.0;
                    t += period;
                    period *= 1.0 + 0.01 * rng.f32_sym();
                }
            }
        }
        2 => x = signals::speech_like(n, 1, fs_khz as u32 * 1000, rng.next_u64()),
        _ => x = signals::noise(n, 1, 0.5, rng.next_u64()),
    }
    to_i16(&x, amp)
}

#[test]
fn pitch_analysis_core() {
    let mut rng = Rng::new(7);
    let mut voiced = 0;
    for it in 0..2500 {
        let fs_khz = [8, 12, 16][it % 3];
        let nb_subfr = if it % 4 == 0 { 2 } else { 4 };
        let complexity = rng.range_i32(0, 2);
        let frame = pitch_frame(&mut rng, fs_khz, nb_subfr, it as i32);
        let prev_lag = if rng.range_i32(0, 1) == 0 {
            0
        } else {
            rng.range_i32(2 * fs_khz, 18 * fs_khz)
        };
        let corr_in = rng.range_i32(0, 1 << 15);
        let thres1 = [52429, 49807, 48497, 47186, 45875][rng.range_i32(0, 4) as usize];
        let thres2 = rng.range_i32(2000, 6000);
        let mut pitch = [0i32; 4];
        let mut li = 0i16;
        let mut ci = 0i8;
        let mut corr = corr_in;
        let r = fix::silk_pitch_analysis_core(
            &frame, &mut pitch, &mut li, &mut ci, &mut corr, prev_lag, thres1, thres2, fs_khz,
            complexity, nb_subfr,
        );
        let cc = c::pitch_analysis_core(
            &frame, corr_in, prev_lag, thres1, thres2, fs_khz, complexity, nb_subfr,
        );
        assert_eq!(
            (r, pitch, li, ci, corr),
            cc,
            "pitch_analysis_core {it} (fs {fs_khz}, nb {nb_subfr}, cx {complexity})"
        );
        if r == 0 {
            voiced += 1;
        }

        // stage-3 helpers (on a frame with the headroom the analyser gives them)
        let peak = frame.iter().fold(0i32, |m, &v| m.max((v as i32).abs()));
        let frame: Vec<i16> = if peak > 4000 {
            frame.iter().map(|&v| v >> 3).collect()
        } else {
            frame
        };
        if fs_khz > 8 {
            let sf = (5 * fs_khz) as usize;
            let start_lag = rng.range_i32(2 * fs_khz, 18 * fs_khz - 12);
            if start_lag + 12 > 4 * sf as i32 {
                continue;
            }
            let mut rc = [[0i32; 5]; 4 * 34];
            fix::silk_p_ana_calc_corr_st3(
                &mut rc, &frame, start_lag, sf as i32, nb_subfr, complexity,
            );
            let ccorr = c::calc_corr_st3(&frame, start_lag, sf, nb_subfr, complexity);
            assert_slice_eq("calc_corr_st3", rc.as_flattened(), &ccorr);
            let mut re = [[0i32; 5]; 4 * 34];
            fix::silk_p_ana_calc_energy_st3(
                &mut re, &frame, start_lag, sf as i32, nb_subfr, complexity,
            );
            let cen = c::calc_energy_st3(&frame, start_lag, sf, nb_subfr, complexity);
            assert_slice_eq("calc_energy_st3", re.as_flattened(), &cen);
        }
    }
    assert!(voiced > 500, "too few voiced frames: {voiced}");
}

// ---------------------------------------------------------------------------------------------
// Full encoder: state dump (same order as oracle_sefx_dump in csrc/silk_encoder_fix.c)
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

fn dump_channel(e: &SilkEncoderStateFix, v: &mut Vec<i32>) {
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
    v.extend([
        e.s_shape.last_gain_index as i32,
        e.s_shape.harm_boost_smth_q16,
        e.s_shape.harm_shape_gain_smth_q16,
        e.s_shape.tilt_smth_q16,
    ]);
    v.extend(e.x_buf.iter().map(|&x| x as i32));
    v.push(e.ltp_corr_q15);
    v.push(e.res_nrg_smth);
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
    /// Garbage input: full-scale noise and square waves, pure tones, DC, NaN / infinities (as
    /// saturated / zero samples).
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

fn gen_signal(cfg: &Cfg, fs: i32, n: usize) -> Vec<OpusRes> {
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
                        _ => 1e-5 * rng.f32_sym(),
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
                    2 => x[i * 2 + 1] = x[i * 2] + 0.0002 * music[i],
                    _ => x[i * 2 + 1] = -0.7 * x[i * 2] + 0.2 * music[i],
                }
            }
        }
        _ => {}
    }
    x.iter().map(|&v| to_res(v)).collect()
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
        pcm: &[OpusRes],
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
        let mut rout: Vec<OpusRes> = vec![Default::default(); 960 * 2];
        let mut cout: Vec<OpusRes> = vec![Default::default(); 960 * 2];
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
            assert_slice_eq(
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
    let pcms: Vec<Vec<OpusRes>> = api_rates
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
    assert_eq!(c::res_size() as usize, size_of::<OpusRes>());
    let mut st = SilkEncControlStruct::default();
    for ch in [1, 2] {
        let mut h = Harness::new(ch, "init");
        h.reinit(ch);
        let mut e = SilkEncoder::new();
        assert_eq!(e.init(ch, &mut st), 0);
    }
    let mut s = SilkEncoderStateFix::new();
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
        cfg.sig = [Sig::StereoMix, Sig::Loud][i as usize % 2];
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

/// Exhaustive random sweep (all parameters randomized per stream and per packet):
/// `cargo test -p opusorus-conformance --features fixed-point --test silk_encoder_fix --
/// --ignored sweep`.
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
/// opusorus-conformance --features fixed-point --test silk_encoder_fix -- --ignored
/// --nocapture perf`.
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

//! Differential tests for unit `dnn_osce` (dnn/osce.c, osce_features.c: LACE, NoLACE and the
//! BBWENet bandwidth extension) vs the C oracle built with `ENABLE_OSCE` + `ENABLE_OSCE_BWE`.
//!
//! Run with `cargo test -p opusorus-conformance --features osce --test dnn_osce` (needs the
//! model data from `scripts/fetch_dnn_models.sh`). Every comparison is bit-exact.
//!
//! * Realistic streams: speech-like signals are encoded with the C Opus encoder
//!   (RESTRICTED_SILK) and decoded by a C SILK decoder whose OSCE calls are logged with their
//!   inputs and outputs (`opusorus_oracle::dnn_osce::CapDecoder`). The Rust side replays every
//!   call (`osce_reset`, `osce_enhance_frame`, `osce_bwe_reset`, `osce_bwe`,
//!   `osce_bwe_cross_fade_10ms`, decoder init/reset) with the same inputs and must produce the
//!   same outputs; the full OSCE / BWE state of both channels is compared after every decode
//!   call.
//! * Direct calls with synthetic inputs for every internal function (feature extraction, the
//!   LACE / NoLACE / BBWENet networks, resamplers, cross-fades).
//!
//! The C oracle uses its compiled-in model tables; the Rust side loads a weight blob the oracle
//! serializes from those same tables.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![cfg(feature = "osce")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use std::sync::OnceLock;

use opusorus::dnn::nndsp::{AdaCombState, AdaConvState, AdaShapeState};
use opusorus::dnn::osce::{self as o, OsceModel, SilkOsceBweStruct, SilkOsceStruct};
use opusorus::dnn::osce_features::{self as of, Filterbank, OsceDecInfo};
use opusorus::silk::structs::SilkDecoderControl;
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::api::Encoder;
use opusorus_oracle::dnn_core::{MODEL_BBWENET, MODEL_LACE, MODEL_NOLACE, write_blob};
use opusorus_oracle::dnn_osce as c;
use opusorus_oracle::sys;

// ---------------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------------

fn blob() -> &'static [u8] {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    B.get_or_init(|| {
        let mut b = write_blob(MODEL_LACE);
        b.extend(write_blob(MODEL_NOLACE));
        b.extend(write_blob(MODEL_BBWENET));
        b
    })
}

/// Rust model loaded from the blob (`loaded` = true) and an unloaded copy.
fn models() -> &'static (OsceModel, OsceModel) {
    static M: OnceLock<(OsceModel, OsceModel)> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = OsceModel::default();
        o::osce_load_models(&mut m, Some(blob())).expect("blob loads");
        m.loaded = true;
        let mut u = m.clone();
        u.loaded = false;
        (m, u)
    })
}

fn model(loaded: bool) -> &'static OsceModel {
    let (m, u) = models();
    if loaded { m } else { u }
}

fn c_model() -> c::Model {
    let (m, ret) = c::Model::new(Some(blob()));
    assert_eq!(ret, 0);
    m
}

// ---------------------------------------------------------------------------------------------
// State dumps in the oc_dump_osce / oc_dump_bwe format
// ---------------------------------------------------------------------------------------------

fn put_f(v: &mut Vec<u32>, x: &[f32]) {
    v.extend(x.iter().map(|f| f.to_bits()));
}

fn put_conv(v: &mut Vec<u32>, s: &AdaConvState) {
    put_f(v, &s.history);
    put_f(v, &s.last_kernel);
    put_f(v, &[s.last_gain]);
}

fn put_comb(v: &mut Vec<u32>, s: &AdaCombState) {
    put_f(v, &s.history);
    put_f(v, &s.last_kernel);
    put_f(v, &[s.last_global_gain]);
    v.push(s.last_pitch_lag as u32);
}

fn put_shape(v: &mut Vec<u32>, s: &AdaShapeState) {
    put_f(v, &s.conv_alpha1f_state);
    put_f(v, &s.conv_alpha1t_state);
    put_f(v, &s.conv_alpha2_state);
    put_f(v, &s.interpolate_state);
}

fn dump_osce(s: &SilkOsceStruct) -> Vec<u32> {
    let mut v = Vec::new();
    let f = &s.features;
    v.push(s.method as u32);
    put_f(&mut v, &[f.numbits_smooth]);
    v.push(f.pitch_hangover_count as u32);
    v.push(f.last_lag as u32);
    v.push(f.last_type as u32);
    v.push(f.reset as u32);
    put_f(&mut v, &f.signal_history);
    if s.method == o::OSCE_METHOD_LACE {
        let l = &s.state.lace;
        put_f(&mut v, &l.feature_net_conv2_state);
        put_f(&mut v, &l.feature_net_gru_state);
        put_comb(&mut v, &l.cf1_state);
        put_comb(&mut v, &l.cf2_state);
        put_conv(&mut v, &l.af1_state);
        put_f(&mut v, &[l.preemph_mem, l.deemph_mem]);
    } else if s.method == o::OSCE_METHOD_NOLACE {
        let n = &s.state.nolace;
        put_f(&mut v, &n.feature_net_conv2_state);
        put_f(&mut v, &n.feature_net_gru_state);
        put_f(&mut v, &n.post_cf1_state);
        put_f(&mut v, &n.post_cf2_state);
        put_f(&mut v, &n.post_af1_state);
        put_f(&mut v, &n.post_af2_state);
        put_f(&mut v, &n.post_af3_state);
        put_comb(&mut v, &n.cf1_state);
        put_comb(&mut v, &n.cf2_state);
        put_conv(&mut v, &n.af1_state);
        put_conv(&mut v, &n.af2_state);
        put_conv(&mut v, &n.af3_state);
        put_conv(&mut v, &n.af4_state);
        put_shape(&mut v, &n.tdshape1_state);
        put_shape(&mut v, &n.tdshape2_state);
        put_shape(&mut v, &n.tdshape3_state);
        put_f(&mut v, &[n.preemph_mem, n.deemph_mem]);
    }
    v
}

fn dump_bwe(b: &SilkOsceBweStruct) -> Vec<u32> {
    let mut v = Vec::new();
    let s = &b.state.bbwenet;
    put_f(&mut v, &b.features.signal_history);
    put_f(&mut v, &b.features.last_spec);
    put_f(&mut v, &s.feature_net_conv1_state);
    put_f(&mut v, &s.feature_net_conv2_state);
    put_f(&mut v, &s.feature_net_gru_state);
    v.extend(s.outbut_buffer.iter().map(|&x| i32::from(x) as u32));
    put_conv(&mut v, &s.af1_state);
    put_conv(&mut v, &s.af2_state);
    put_conv(&mut v, &s.af3_state);
    put_shape(&mut v, &s.tdshape1_state);
    put_shape(&mut v, &s.tdshape2_state);
    for r in &s.resampler_state {
        put_f(&mut v, &r.upsamp_buffer[0]);
        put_f(&mut v, &r.upsamp_buffer[1]);
        put_f(&mut v, &r.interpol_buffer);
    }
    v
}

#[track_caller]
fn assert_dump_eq(what: &str, rust: &[u32], c: &[u32]) {
    assert_eq!(rust.len(), c.len(), "{what}: dump length");
    if let Some(i) = rust.iter().zip(c).position(|(a, b)| a != b) {
        panic!(
            "{what}: state word {i} differs: rust={:#010x} ({}) c={:#010x} ({})",
            rust[i],
            f32::from_bits(rust[i]),
            c[i],
            f32::from_bits(c[i])
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------------------------

fn ctrl_to_rust(c: &c::Ctrl) -> SilkDecoderControl {
    let mut pred = [[0i16; 16]; 2];
    pred[0].copy_from_slice(&c.pred_coef_q12[..16]);
    pred[1].copy_from_slice(&c.pred_coef_q12[16..]);
    SilkDecoderControl {
        pitch_l: c.pitch_l,
        gains_q16: c.gains_q16,
        pred_coef_q12: pred,
        ltp_coef_q14: c.ltp_coef_q14,
        ..SilkDecoderControl::default()
    }
}

const fn info_to_rust(i: &c::DecInfo) -> OsceDecInfo {
    OsceDecInfo {
        fs_khz: i[0],
        nb_subfr: i[1],
        lpc_order: i[2],
        signal_type: i[3],
    }
}

const fn event_ctrl(e: &c::Event) -> c::Ctrl {
    c::Ctrl {
        pitch_l: e.pitch_l,
        gains_q16: e.gains_q16,
        pred_coef_q12: e.pred_coef_q12,
        ltp_coef_q14: e.ltp_coef_q14,
    }
}

// ---------------------------------------------------------------------------------------------
// Synthetic inputs
// ---------------------------------------------------------------------------------------------

/// A plausible random SILK decoder control (stable-ish LPC, SILK-range gains / LTP / lags).
fn random_ctrl(rng: &mut Rng, fs_khz: i32, lpc_order: usize) -> c::Ctrl {
    let mut ctrl = c::Ctrl::default();
    for k in 0..4 {
        ctrl.pitch_l[k] = rng.range_i32(2 * fs_khz, 18 * fs_khz);
        ctrl.gains_q16[k] = rng.range_i32(1 << 10, 1 << 24);
    }
    for half in 0..2 {
        // Decaying coefficients keep the synthesis filter well-behaved.
        let mut scale = 3000.0f32;
        for i in 0..lpc_order {
            ctrl.pred_coef_q12[half * 16 + i] = (scale * rng.f32_sym()) as i16;
            scale *= 0.8;
        }
    }
    for v in &mut ctrl.ltp_coef_q14 {
        *v = rng.range_i32(-4000, 12000) as i16;
    }
    ctrl
}

/// Speech-like 16 kHz signal (plus occasional clipping-range bursts).
fn speech16(n: usize, seed: u64) -> Vec<i16> {
    let x = signals::speech_like(n, 1, 16000, seed);
    let mut y = signals::to_i16(&x);
    let mut rng = Rng::new(seed ^ 0x77);
    for chunk in y.chunks_mut(997) {
        if rng.range_i32(0, 9) == 0 {
            for v in chunk.iter_mut().take(40) {
                *v = if rng.range_i32(0, 1) == 0 {
                    32767
                } else {
                    -32768
                };
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------------------------
// Tests: model / helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn load_models() {
    let (m, _) = models();
    let mut cm = c_model();
    assert_bits_eq_f32("lace window", &m.lace.window, &cm.window(0));
    assert_bits_eq_f32("nolace window", &m.nolace.window, &cm.window(1));
    assert_bits_eq_f32("bbwenet window16", &m.bbwenet.window16, &cm.window(2));
    assert_bits_eq_f32("bbwenet window32", &m.bbwenet.window32, &cm.window(3));
    assert_bits_eq_f32("bbwenet window48", &m.bbwenet.window48, &cm.window(4));
    // Compiled-in C tables load too.
    let (_, ret) = c::Model::new(None);
    assert_eq!(ret, 0);

    // No data: USE_WEIGHTS_FILE behaviour.
    let mut r = OsceModel::default();
    assert!(o::osce_load_models(&mut r, None).is_err());
    assert!(o::osce_load_models(&mut r, Some(&[])).is_err());
    // A blob without BBWENet (like upstream write_lpcnet_weights output) fails in both.
    let mut partial = write_blob(MODEL_LACE);
    partial.extend(write_blob(MODEL_NOLACE));
    assert!(o::osce_load_models(&mut r, Some(&partial)).is_err());
    let (_, ret) = c::Model::new(Some(&partial));
    assert_eq!(ret, -1);
    // Missing NoLACE.
    let mut partial = write_blob(MODEL_LACE);
    partial.extend(write_blob(MODEL_BBWENET));
    assert!(o::osce_load_models(&mut r, Some(&partial)).is_err());
    let (_, ret) = c::Model::new(Some(&partial));
    assert_eq!(ret, -1);
    // Garbage (C would crash on the NULL list; Rust reports an error).
    assert!(o::osce_load_models(&mut r, Some(&[1, 2, 3, 4, 5])).is_err());
    // A successful load binds every layer like C (checked through outputs in the other tests).
    let mut r = OsceModel::default();
    o::osce_load_models(&mut r, Some(blob())).unwrap();
    assert_eq!(r.lace, m.lace);
    assert_eq!(r.nolace, m.nolace);
    assert_eq!(r.bbwenet, m.bbwenet);
}

#[test]
fn numbits_embedding() {
    let mut rng = Rng::new(1);
    let lo = (50f64).ln() as f32;
    let hi = (650f64).ln() as f32;
    let mut vals: Vec<f32> = vec![
        0.0, 1.0, 10.0, 49.0, 50.0, 51.0, 200.0, 649.0, 650.0, 651.0, 5000.0,
    ];
    for _ in 0..2000 {
        vals.push(rng.range_i32(0, 2000) as f32 + 0.37 * rng.f32_sym());
    }
    for &nb in &vals {
        for nolace in [false, true] {
            for (logscale, mn, mx) in [(true, lo, hi), (false, 50.0, 650.0), (false, -3.0, 2.0)] {
                let mut e = [0f32; 8];
                if nolace {
                    o::compute_nolace_numbits_embedding(&mut e, nb, 8, mn, mx, logscale);
                } else {
                    o::compute_lace_numbits_embedding(&mut e, nb, 8, mn, mx, logscale);
                }
                let ce = c::numbits_embedding(nolace, nb, mn, mx, logscale);
                assert_bits_eq_f32(&format!("numbits {nb} {nolace} {logscale}"), &e, &ce);
            }
        }
    }
}

#[test]
fn spectral_helpers() {
    let mut rng = Rng::new(2);
    let banks = [Filterbank::Clean, Filterbank::Noisy, Filterbank::Bwe];
    for it in 0..400 {
        let x: Vec<f32> = (0..320)
            .map(|_| rng.f32_sym() * if it % 3 == 0 { 1e-3 } else { 1.0 })
            .collect();
        // Filterbanks on non-negative spectra.
        let spec: Vec<f32> = x[..161].iter().map(|v| v.abs() * 10.0).collect();
        for (b, &bank) in banks.iter().enumerate() {
            let mut out = [0f32; 64];
            of::apply_filterbank(&mut out, &spec, bank);
            let cv = c::apply_filterbank(b as i32, &spec);
            assert_bits_eq_f32("filterbank", &out[..cv.len()], &cv);
        }
        let mut ms = [0f32; 161];
        of::mag_spec_320_onesided(&mut ms, &x);
        assert_bits_eq_f32("mag_spec", &ms, &c::mag_spec(&x));

        let mut cep = [0f32; 18];
        of::calculate_cepstrum(&mut cep, &x);
        assert_bits_eq_f32("cepstrum", &cep, &c::cepstrum(&x));

        let order = if it % 2 == 0 { 16 } else { 10 };
        let ctrl = random_ctrl(&mut rng, 16, order);
        let a = &ctrl.pred_coef_q12[..16];
        let mut ls = [0f32; 64];
        of::calculate_log_spectrum_from_lpc(&mut ls, a, order);
        assert_bits_eq_f32("log spectrum", &ls, &c::log_spectrum_from_lpc(a, order));

        let buf: Vec<f32> = (0..700).map(|_| rng.f32_sym()).collect();
        let lag = if it % 5 == 0 {
            7
        } else {
            rng.range_i32(32, 288)
        };
        let pos = 350 + 80 * rng.range_i32(0, 3) as usize;
        let mut ac = [0f32; 5];
        of::calculate_acorr(&mut ac, &buf, pos, lag);
        assert_bits_eq_f32("acorr", &ac, &c::acorr(&buf, pos, lag));
    }
    // Silence (log of the 1e-9 floors, zero correlation).
    let z = [0f32; 700];
    let mut cep = [0f32; 18];
    of::calculate_cepstrum(&mut cep, &z);
    assert_bits_eq_f32("cepstrum silence", &cep, &c::cepstrum(&z));
    let mut ac = [0f32; 5];
    of::calculate_acorr(&mut ac, &z, 350, 100);
    assert_bits_eq_f32("acorr silence", &ac, &c::acorr(&z, 350, 100));
}

#[test]
fn pitch_postprocessing() {
    let mut rng = Rng::new(3);
    let mut cs = c::DecState::new();
    let mut rs = SilkOsceStruct::default();
    for _ in 0..5000 {
        let lag = rng.range_i32(32, 288);
        let t = rng.range_i32(0, 2);
        let r = of::pitch_postprocessing(&mut rs.features, lag, t);
        assert_eq!(r, cs.pitch_postprocessing(lag, t));
    }
    assert_dump_eq("state", &dump_osce(&rs), &cs.dump());
}

#[test]
fn resamplers_and_activation() {
    let mut rng = Rng::new(4);
    let mut rs = o::ResampState::default();
    let mut cst: c::ResampState = [0.0; 14];
    for it in 0..300 {
        let n = 2 * rng.range_i32(1, 150) as usize;
        let x: Vec<f32> = (0..n).map(|_| rng.f32_sym()).collect();
        let mut out = vec![0f32; 2 * n];
        o::upsamp_2x(&mut rs, &mut out, &x, n);
        assert_bits_eq_f32("upsamp_2x", &out, &c::upsamp_2x(&mut cst, &x));
        let mut out = vec![0f32; 3 * n / 2];
        o::interpol_3_2(&mut rs, &mut out, &x, n);
        assert_bits_eq_f32("interpol_3_2", &out, &c::interpol_3_2(&mut cst, &x));
        let mut st = Vec::new();
        put_f(&mut st, &rs.upsamp_buffer[0]);
        put_f(&mut st, &rs.upsamp_buffer[1]);
        put_f(&mut st, &rs.interpol_buffer);
        let cst_bits: Vec<u32> = cst.iter().map(|f| f.to_bits()).collect();
        assert_dump_eq("resamp state", &st, &cst_bits);

        let len = rng.range_i32(1, 480) as usize;
        let mut v: Vec<f32> = (0..len)
            .map(|i| {
                if it % 7 == 0 && i % 3 == 0 {
                    0.0
                } else {
                    rng.f32_sym() * [1e-7, 1e-3, 1.0, 30.0][i % 4]
                }
            })
            .collect();
        let mut cv = v.clone();
        o::apply_valin_activation(&mut v, len);
        c::valin_activation(&mut cv);
        assert_bits_eq_f32("valin", &v, &cv);
    }
}

#[test]
fn cross_fades() {
    let mut rng = Rng::new(5);
    for it in 0..500 {
        let mut e: Vec<f32> = (0..320).map(|_| rng.f32_sym()).collect();
        let x: Vec<f32> = (0..320).map(|_| rng.f32_sym()).collect();
        let mut ce = e.clone();
        of::osce_cross_fade_10ms(&mut e, &x, 320);
        c::cross_fade_10ms(&mut ce, &x);
        assert_bits_eq_f32("cross_fade", &e, &ce);

        let ext = [i16::MIN, i16::MAX, -1, 0, 1];
        let mut fi: Vec<i16> = (0..480)
            .map(|i| if it % 4 == 0 { ext[i % 5] } else { rng.i16() })
            .collect();
        let fo: Vec<i16> = (0..480)
            .map(|i| {
                if it % 4 == 1 {
                    ext[(i + 2) % 5]
                } else {
                    rng.i16()
                }
            })
            .collect();
        let mut cfi = fi.clone();
        of::osce_bwe_cross_fade_10ms(&mut fi, &fo, 480);
        c::bwe_cross_fade_10ms(&mut cfi, &fo);
        assert_slice_eq("bwe_cross_fade", &fi, &cfi);
    }
}

// ---------------------------------------------------------------------------------------------
// Tests: features, networks, enhancement and BWE with direct calls
// ---------------------------------------------------------------------------------------------

/// Runs `osce_calculate_features` over a synthetic stream in both implementations and returns
/// the per-frame (features, numbits, periods, signal) for reuse.
type FrameInputs = (Vec<f32>, [f32; 2], [i32; 4], Vec<i16>);

fn feature_stream(frames: usize, seed: u64) -> Vec<FrameInputs> {
    let mut rng = Rng::new(seed);
    let sig = speech16(frames * 320, seed);
    let mut cs = c::DecState::new();
    let mut rs = SilkOsceStruct::default();
    let mut out = Vec::new();
    for f in 0..frames {
        let nb_subfr = if f % 11 == 5 { 2 } else { 4 };
        let lpc_order = if f % 7 == 3 { 10 } else { 16 };
        let signal_type = rng.range_i32(0, 2);
        let info: c::DecInfo = [16, nb_subfr, lpc_order, signal_type];
        let ctrl = random_ctrl(&mut rng, 16, lpc_order as usize);
        let xq = &sig[f * 320..f * 320 + nb_subfr as usize * 80];
        let num_bits = rng.range_i32(0, 1400);
        let (cf, cn, cp) = cs.features(&info, &ctrl, xq, num_bits);
        let mut feats = vec![0f32; 4 * o::OSCE_FEATURE_DIM];
        let mut numbits = [0f32; 2];
        let mut periods = [0i32; 4];
        of::osce_calculate_features(
            &mut rs.features,
            info_to_rust(&info),
            &ctrl_to_rust(&ctrl),
            &mut feats,
            &mut numbits,
            &mut periods,
            xq,
            num_bits,
        );
        let n = nb_subfr as usize * o::OSCE_FEATURE_DIM;
        assert_bits_eq_f32(&format!("features frame {f}"), &feats[..n], &cf);
        assert_bits_eq_f32("numbits", &numbits, &cn);
        assert_eq!(
            periods[..nb_subfr as usize],
            cp[..nb_subfr as usize],
            "periods"
        );
        assert_dump_eq(
            &format!("feature state frame {f}"),
            &dump_osce(&rs),
            &cs.dump(),
        );
        if nb_subfr == 4 {
            out.push((feats, numbits, periods, xq.to_vec()));
        }
    }
    out
}

#[test]
fn features_stream() {
    let v = feature_stream(400, 6);
    assert!(v.len() > 300);
}

#[test]
fn lace_nolace_networks() {
    let inputs = feature_stream(120, 7);
    let mut cm = c_model();
    for nolace in [false, true] {
        let method = if nolace {
            o::OSCE_METHOD_NOLACE
        } else {
            o::OSCE_METHOD_LACE
        };
        let mut cs = c::DecState::new();
        cs.reset(method);
        let mut rs = SilkOsceStruct::default();
        o::osce_reset(&mut rs, method);
        let m = model(true);
        for (f, (feats, numbits, periods, xq)) in inputs.iter().enumerate() {
            // Feature net alone on even frames, full frame processing on odd ones (both
            // advance the same state).
            if f % 4 == 0 {
                let cv = if nolace {
                    cs.nolace_feature_net(&mut cm, feats, numbits, periods)
                } else {
                    cs.lace_feature_net(&mut cm, feats, numbits, periods)
                };
                let mut out = vec![0f32; cv.len()];
                if nolace {
                    o::nolace_feature_net(
                        &m.nolace,
                        &mut rs.state.nolace,
                        &mut out,
                        feats,
                        numbits,
                        periods,
                    );
                } else {
                    o::lace_feature_net(
                        &m.lace,
                        &mut rs.state.lace,
                        &mut out,
                        feats,
                        numbits,
                        periods,
                    );
                }
                assert_bits_eq_f32(&format!("feature net {nolace} frame {f}"), &out, &cv);
            } else {
                let x: Vec<f32> = xq.iter().map(|&v| v as f32 * (1.0 / 32768.0)).collect();
                let cv = cs.process_frame(nolace, &mut cm, &x, feats, numbits, periods);
                let mut out = vec![0f32; 320];
                if nolace {
                    o::nolace_process_20ms_frame(
                        &m.nolace,
                        &mut rs.state.nolace,
                        &mut out,
                        &x,
                        feats,
                        numbits,
                        periods,
                    );
                } else {
                    o::lace_process_20ms_frame(
                        &m.lace,
                        &mut rs.state.lace,
                        &mut out,
                        &x,
                        feats,
                        numbits,
                        periods,
                    );
                }
                assert_bits_eq_f32(&format!("process frame {nolace} frame {f}"), &out, &cv);
            }
            assert_dump_eq(
                &format!("state {nolace} frame {f}"),
                &dump_osce(&rs),
                &cs.dump(),
            );
        }
    }
}

#[test]
fn enhance_frame_direct() {
    let mut rng = Rng::new(8);
    let sig = speech16(300 * 320, 8);
    let mut cm = c_model();
    let mut cm_unloaded = c_model();
    cm_unloaded.set_loaded(false);
    let mut cs = c::DecState::new();
    let mut rs = SilkOsceStruct::default();
    // Starts from the zeroed state (method NONE, reset 0) like a fresh silk_decoder_state.
    for f in 0..300 {
        if f % 37 == 0 {
            let method = rng.range_i32(0, 2);
            cs.reset(method);
            o::osce_reset(&mut rs, method);
        }
        let (fs_khz, nb_subfr) = match f % 23 {
            7 => (8, 4),
            13 => (12, 4),
            17 => (16, 2),
            _ => (16, 4),
        };
        let lpc_order = if fs_khz == 16 { 16 } else { 10 };
        let info: c::DecInfo = [fs_khz, nb_subfr, lpc_order, rng.range_i32(0, 2)];
        let ctrl = random_ctrl(&mut rng, fs_khz, lpc_order as usize);
        let mut xq = sig[f * 320..(f + 1) * 320].to_vec();
        let mut cxq = xq.clone();
        let num_bits = rng.range_i32(20, 1200);
        let loaded = f % 50 < 45;
        let cmod = if loaded { &mut cm } else { &mut cm_unloaded };
        cs.enhance(cmod, &info, &ctrl, &mut cxq, num_bits);
        o::osce_enhance_frame(
            model(loaded),
            &mut rs,
            info_to_rust(&info),
            &ctrl_to_rust(&ctrl),
            &mut xq,
            num_bits,
        );
        assert_slice_eq(&format!("enhance frame {f}"), &xq, &cxq);
        assert_dump_eq(&format!("enhance state {f}"), &dump_osce(&rs), &cs.dump());
    }
}

#[test]
fn bwe_direct() {
    let mut rng = Rng::new(9);
    let sig = speech16(400 * 320, 9);
    let mut cm = c_model();
    let m = model(true);

    // osce_bwe_calculate_features / bbwe_feature_net / bbwenet_process_frames / osce_bwe on
    // separate states, each fed the same stream.
    let mut c_feat = c::BweState::new();
    let mut r_feat = SilkOsceBweStruct::default();
    let mut c_net = c::BweState::new();
    let mut r_net = SilkOsceBweStruct::default();
    let mut c_run = c::BweState::new();
    let mut r_run = SilkOsceBweStruct::default();
    c_net.reset();
    o::osce_bwe_reset(&mut r_net);
    let mut pos = 0;
    let mut f = 0;
    while pos + 320 <= sig.len() {
        let len = if rng.range_i32(0, 3) == 0 { 160 } else { 320 };
        let xq = &sig[pos..pos + len];
        pos += len;
        let nf = len / 160;
        if f % 97 == 50 {
            // Mid-stream reset of the run state (and a zeroed-state start before the first).
            c_run.reset();
            o::osce_bwe_reset(&mut r_run);
        }

        let cf = c_feat.features(xq);
        let mut rf = vec![0f32; 2 * o::OSCE_BWE_FEATURE_DIM];
        of::osce_bwe_calculate_features(&mut r_feat.features, &mut rf, xq, len);
        assert_bits_eq_f32(&format!("bwe features {f}"), &rf[..cf.len()], &cf);

        if f % 2 == 0 {
            let cv = c_net.feature_net(&mut cm, &cf, nf);
            let mut out = vec![0f32; cv.len()];
            o::bbwe_feature_net(&m.bbwenet, &mut r_net.state.bbwenet, &mut out, &cf, nf);
            assert_bits_eq_f32(&format!("bwe feature net {f}"), &out, &cv);
        } else {
            let x: Vec<f32> = xq.iter().map(|&v| v as f32 * (1.0 / 32768.0)).collect();
            let cv = c_net.process_frames(&mut cm, &x, &cf, nf);
            let mut out = vec![0f32; cv.len()];
            o::bbwenet_process_frames(&m.bbwenet, &mut r_net.state.bbwenet, &mut out, &x, &cf, nf);
            assert_bits_eq_f32(&format!("bwe process frames {f}"), &out, &cv);
        }
        assert_dump_eq(
            &format!("bwe net state {f}"),
            &dump_bwe(&r_net),
            &c_net.dump(),
        );

        let cv = c_run.run(&mut cm, xq);
        let mut out = vec![0i16; 3 * len];
        o::osce_bwe(m, &mut r_run, &mut out, xq, len);
        assert_slice_eq(&format!("osce_bwe {f}"), &out, &cv);
        assert_dump_eq(
            &format!("osce_bwe state {f}"),
            &dump_bwe(&r_run),
            &c_run.dump(),
        );
        f += 1;
    }
}

// ---------------------------------------------------------------------------------------------
// Tests: realistic SILK streams through the capturing C decoder
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct StreamCfg {
    channels: i32,
    bitrate: i32,
    max_bw: i32,
    frame_ms: i32,
    fec: bool,
    /// Random bitrate / bandwidth changes.
    vary: bool,
    n_packets: usize,
    seed: u64,
}

fn encode_stream(cfg: &StreamCfg) -> Vec<Vec<u8>> {
    let mut enc =
        Encoder::new(48000, cfg.channels, sys::OPUS_APPLICATION_RESTRICTED_SILK).expect("encoder");
    enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, cfg.max_bw)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, i32::from(cfg.fec))
        .unwrap();
    enc.ctl_set(
        sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
        if cfg.fec { 20 } else { 0 },
    )
    .unwrap();
    enc.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, 10).unwrap();
    let frame = (48 * cfg.frame_ms) as usize;
    let ch = cfg.channels as usize;
    let x = signals::speech_like(frame * cfg.n_packets, ch, 48000, cfg.seed);
    let pcm = signals::to_i16(&x);
    let mut rng = Rng::new(cfg.seed ^ 0xABCD);
    let mut out = Vec::with_capacity(cfg.n_packets);
    let mut buf = vec![0u8; 1500];
    for k in 0..cfg.n_packets {
        if cfg.vary && rng.range_i32(0, 9) == 0 {
            let br = [6000, 9000, 12000, 16000, 24000, 32000][rng.range_i32(0, 5) as usize];
            enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, br * cfg.channels)
                .unwrap();
            let bw = [
                sys::OPUS_BANDWIDTH_NARROWBAND,
                sys::OPUS_BANDWIDTH_MEDIUMBAND,
                sys::OPUS_BANDWIDTH_WIDEBAND,
                sys::OPUS_BANDWIDTH_WIDEBAND,
            ][rng.range_i32(0, 3) as usize];
            enc.ctl_set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw)
                .unwrap();
        }
        let n = enc
            .encode(&pcm[k * frame * ch..(k + 1) * frame * ch], frame, &mut buf)
            .expect("encode");
        out.push(buf[..n].to_vec());
    }
    out
}

/// Rust mirror of the OSCE parts of the two SILK decoder channels, driven by the C events.
struct Replay {
    osce: [SilkOsceStruct; 2],
    bwe: [SilkOsceBweStruct; 2],
    label: String,
    enhanced: usize,
    enhanced_by_method: [usize; 3],
    /// Enhanced frames whose output differs from the input.
    changed: usize,
    bwe_calls: usize,
    xfades: usize,
    resets: usize,
}

impl Replay {
    fn new(label: &str) -> Self {
        Self {
            osce: [SilkOsceStruct::default(), SilkOsceStruct::default()],
            bwe: [SilkOsceBweStruct::default(), SilkOsceBweStruct::default()],
            label: label.to_string(),
            enhanced: 0,
            enhanced_by_method: [0; 3],
            changed: 0,
            bwe_calls: 0,
            xfades: 0,
            resets: 0,
        }
    }

    fn apply(&mut self, events: &[c::Event], what: &str) {
        for (i, e) in events.iter().enumerate() {
            let tag = format!("{} {what} event {i} (kind {})", self.label, e.kind);
            let ch = e.ch;
            match e.kind {
                c::EV_INIT => {
                    let ch = ch as usize;
                    self.osce[ch] = SilkOsceStruct::default();
                    self.bwe[ch] = SilkOsceBweStruct::default();
                    o::osce_reset(&mut self.osce[ch], o::OSCE_DEFAULT_METHOD);
                }
                c::EV_RESETDEC => {
                    o::osce_reset(&mut self.osce[ch as usize], o::OSCE_DEFAULT_METHOD);
                }
                c::EV_RESET => {
                    self.resets += 1;
                    o::osce_reset(&mut self.osce[ch as usize], e.method);
                }
                c::EV_ENHANCE => {
                    let ch = ch as usize;
                    assert_eq!(self.osce[ch].method, e.method, "{tag}: method");
                    let info: c::DecInfo = [e.fs_khz, e.nb_subfr, e.lpc_order, e.signal_type];
                    let len = e.len as usize;
                    let mut xq = [0i16; 320];
                    xq[..len].copy_from_slice(&e.input[..len]);
                    o::osce_enhance_frame(
                        model(e.model_loaded != 0),
                        &mut self.osce[ch],
                        info_to_rust(&info),
                        &ctrl_to_rust(&event_ctrl(e)),
                        &mut xq[..len],
                        e.num_bits,
                    );
                    assert_slice_eq(&tag, &xq[..len], &e.output[..len]);
                    if e.fs_khz == 16 && e.nb_subfr == 4 {
                        self.enhanced += 1;
                        let m = if e.model_loaded != 0 { e.method } else { 0 };
                        self.enhanced_by_method[m as usize] += 1;
                        if m != 0 && e.input[..len] != e.output[..len] {
                            self.changed += 1;
                        }
                    }
                }
                c::EV_BWE_RESET => o::osce_bwe_reset(&mut self.bwe[ch as usize]),
                c::EV_BWE => {
                    let len = e.len as usize;
                    let mut out = [0i16; 960];
                    o::osce_bwe(
                        model(e.model_loaded != 0),
                        &mut self.bwe[ch as usize],
                        &mut out[..3 * len],
                        &e.input[..len],
                        len,
                    );
                    assert_slice_eq(&tag, &out[..3 * len], &e.output[..3 * len]);
                    self.bwe_calls += 1;
                }
                c::EV_BWE_XFADE => {
                    let mut x = e.input;
                    of::osce_bwe_cross_fade_10ms(&mut x, &e.input2, e.len as usize);
                    assert_slice_eq(&tag, &x[..480], &e.output[..480]);
                    self.xfades += 1;
                }
                k => panic!("{tag}: unknown event kind {k}"),
            }
        }
    }

    fn check_state(&self, cd: &mut c::CapDecoder, what: &str) {
        for ch in 0..2 {
            assert_dump_eq(
                &format!("{} {what}: osce state ch {ch}", self.label),
                &dump_osce(&self.osce[ch]),
                &cd.dump_osce(ch),
            );
            assert_dump_eq(
                &format!("{} {what}: bwe state ch {ch}", self.label),
                &dump_bwe(&self.bwe[ch]),
                &cd.dump_bwe(ch),
            );
        }
    }
}

/// How the Opus-level decoder settings evolve over the stream.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Fixed decoder complexity (osce_method NONE / LACE / NoLACE) and API rate.
    Fixed {
        complexity: i32,
        api_fs: i32,
        bwe: bool,
    },
    /// Random complexity / BWE toggles per packet, occasional CELT->SILK transitions and
    /// decoder resets (API 48 kHz).
    Switching,
}

/// Mirrors the opus_decoder.c DecControl setup for a SILK-only packet (or PLC).
const fn osce_method_for(complexity: i32) -> i32 {
    if complexity >= 7 {
        o::OSCE_METHOD_NOLACE
    } else if complexity >= 6 {
        o::OSCE_METHOD_LACE
    } else {
        o::OSCE_METHOD_NONE
    }
}

#[derive(Default, Debug)]
struct Cov {
    enhanced: usize,
    by_method: [usize; 3],
    changed: usize,
    bwe_calls: usize,
    xfades: usize,
    resets: usize,
}

fn run_stream(cfg: &StreamCfg, mode: Mode, loss_pct: i32) -> Cov {
    let pkts = encode_stream(cfg);
    let label = format!("{cfg:?} {mode:?} loss {loss_pct}");
    let mut cd = c::CapDecoder::new();
    let mut rp = Replay::new(&label);
    let init = cd.take_init_events();
    assert_eq!(init.len(), 2, "silk_InitDecoder initializes both channels");
    rp.apply(&init, "init");
    rp.check_state(&mut cd, "init");
    assert!(cd.loaded());

    let mut rng = Rng::new(cfg.seed ^ 0x5151);
    let mut last_ms = cfg.frame_ms;
    let (mut complexity, mut api_fs, mut bwe) = match mode {
        Mode::Fixed {
            complexity,
            api_fs,
            bwe,
        } => (complexity, api_fs, bwe),
        Mode::Switching => (7, 48000, true),
    };
    let mut ctrl = c::CapCtrl {
        n_channels_api: cfg.channels,
        n_channels_internal: cfg.channels,
        api_sample_rate: api_fs,
        internal_sample_rate: 16000,
        payload_size_ms: cfg.frame_ms,
        ..c::CapCtrl::default()
    };
    for (k, p) in pkts.iter().enumerate() {
        if matches!(mode, Mode::Switching) && k % 20 == 19 {
            // Cycle through the complexities (NONE / BWE-only / LACE / NoLACE).
            complexity = [6, 0, 10, 4, 7, 5, 6, 7][(k / 20) % 8];
        }
        if matches!(mode, Mode::Switching) && rng.range_i32(0, 11) == 0 {
            match rng.range_i32(0, 2) {
                0 => bwe = !bwe,
                1 => {
                    // CELT->SILK transition as seen by the SILK decoder.
                    cd.set_prev_ext_mode(o::OSCE_MODE_CELT_ONLY);
                }
                _ => {
                    let r = cd.reset();
                    assert_eq!(r, 0);
                    rp.apply(&cd.events(), "reset");
                    rp.check_state(&mut cd, "reset");
                }
            }
            api_fs = 48000;
        }
        let lost = rng.range_i32(0, 99) < loss_pct;
        let data = if lost || p.len() <= 1 {
            None
        } else {
            Some(&p[..])
        };
        let fec = lost && cfg.fec && k + 1 < pkts.len() && pkts[k + 1].len() > 1;
        let (payload, lost_flag) = match (data, fec) {
            (Some(d), _) => (Some(d), 0),
            (None, true) => (Some(&pkts[k + 1][..]), 2),
            (None, false) => (None, 1),
        };
        if let Some(d) = payload {
            let toc = d[0];
            let config = i32::from(toc >> 3);
            assert!(config < 12, "not a SILK-only packet");
            assert_eq!(toc & 3, 0, "expected a code-0 packet");
            last_ms = [10, 20, 40, 60][(config & 3) as usize];
            ctrl.n_channels_internal = if toc & 4 != 0 { 2 } else { 1 };
            ctrl.internal_sample_rate = [8000, 12000, 16000][(config >> 2) as usize];
        }
        ctrl.api_sample_rate = api_fs;
        ctrl.payload_size_ms = last_ms;
        ctrl.enable_deep_plc = i32::from(complexity >= 5);
        ctrl.osce_method = osce_method_for(complexity);
        ctrl.enable_osce_bwe = i32::from(bwe);
        ctrl.osce_extended_mode =
            if complexity >= 4 && bwe && api_fs == 48000 && ctrl.internal_sample_rate == 16000 {
                o::OSCE_MODE_SILK_BBWE
            } else {
                o::OSCE_MODE_SILK_ONLY
            };

        let body: &[u8] = payload.map_or(&[], |d| &d[1..]);
        cd.ec_init(body);
        let frame_size = last_ms * api_fs / 1000;
        let mut decoded = 0;
        while decoded < frame_size {
            let (ret, n) = cd.decode(&ctrl, lost_flag, i32::from(decoded == 0));
            rp.apply(&cd.events(), &format!("packet {k} at {decoded}"));
            rp.check_state(&mut cd, &format!("packet {k} at {decoded}"));
            if ret != 0 || n <= 0 {
                break;
            }
            decoded += n;
        }
    }
    let cov = Cov {
        enhanced: rp.enhanced,
        by_method: rp.enhanced_by_method,
        changed: rp.changed,
        bwe_calls: rp.bwe_calls,
        xfades: rp.xfades,
        resets: rp.resets,
    };
    // Coverage summary (visible with --nocapture).
    #[expect(clippy::print_stderr, reason = "test diagnostics")]
    {
        eprintln!("{label}: {cov:?}");
    }
    cov
}

const fn scfg(
    channels: i32,
    bitrate: i32,
    max_bw: i32,
    frame_ms: i32,
    n_packets: usize,
    seed: u64,
) -> StreamCfg {
    StreamCfg {
        channels,
        bitrate,
        max_bw,
        frame_ms,
        fec: false,
        vary: false,
        n_packets,
        seed,
    }
}

#[test]
fn stream_wb_nolace() {
    // 10 s of 20 ms WB speech at three bitrates, NoLACE (complexity 7), 16 kHz output.
    for (br, seed) in [(12000, 11), (20000, 13), (32000, 12)] {
        let cfg = scfg(1, br, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 500, seed);
        let cov = run_stream(
            &cfg,
            Mode::Fixed {
                complexity: 7,
                api_fs: 16000,
                bwe: false,
            },
            0,
        );
        assert!(cov.by_method[2] > 490 && cov.changed > 400, "{cov:?}");
    }
}

#[test]
fn stream_wb_lace() {
    // 6 s each, 20/40/60 ms packets, LACE (complexity 6), 48 kHz output without BWE.
    for (br, ms, seed) in [(14000, 20, 21), (24000, 40, 22), (16000, 60, 23)] {
        let cfg = scfg(
            1,
            br,
            sys::OPUS_BANDWIDTH_WIDEBAND,
            ms,
            6000 / ms as usize,
            seed,
        );
        let cov = run_stream(
            &cfg,
            Mode::Fixed {
                complexity: 6,
                api_fs: 48000,
                bwe: false,
            },
            0,
        );
        assert!(cov.by_method[1] > 250 && cov.changed > 200, "{cov:?}");
    }
}

#[test]
fn stream_nb_mb_and_10ms() {
    // NB / MB internal rates (no enhancement: the state is reset every frame) and 10 ms WB
    // frames (nb_subfr 2: reset too), with NoLACE requested.
    for (bw, ms, seed) in [
        (sys::OPUS_BANDWIDTH_NARROWBAND, 20, 31),
        (sys::OPUS_BANDWIDTH_MEDIUMBAND, 20, 32),
        (sys::OPUS_BANDWIDTH_WIDEBAND, 10, 33),
    ] {
        let cfg = scfg(1, 12000, bw, ms, 100, seed);
        let cov = run_stream(
            &cfg,
            Mode::Fixed {
                complexity: 10,
                api_fs: 16000,
                bwe: false,
            },
            0,
        );
        assert_eq!(cov.enhanced, 0, "{cov:?}");
    }
}

#[test]
fn stream_bwe() {
    // WB -> 48 kHz extension with NoLACE, LACE and no enhancement.
    for (cx, seed) in [(7, 41), (6, 42), (4, 43)] {
        let cfg = scfg(1, 16000, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 300, seed);
        let cov = run_stream(
            &cfg,
            Mode::Fixed {
                complexity: cx,
                api_fs: 48000,
                bwe: true,
            },
            0,
        );
        assert!(cov.bwe_calls >= 299, "{cov:?}");
    }
}

#[test]
fn stream_loss_fec_and_switching() {
    // Losses (PLC resets OSCE), FEC (LBRR frames are enhanced too), bitrate / bandwidth
    // changes, method / BWE toggles, CELT->SILK transitions and decoder resets.
    let mut cfg = scfg(1, 16000, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 750, 51);
    cfg.fec = true;
    cfg.vary = true;
    let cov = run_stream(&cfg, Mode::Switching, 10);
    assert!(
        cov.xfades > 0 && cov.resets > 0 && cov.bwe_calls > 0,
        "{cov:?}"
    );
    assert!(cov.by_method.iter().all(|&n| n > 0), "{cov:?}");

    let mut cfg = scfg(1, 24000, sys::OPUS_BANDWIDTH_WIDEBAND, 40, 200, 52);
    cfg.vary = true;
    let cov = run_stream(&cfg, Mode::Switching, 5);
    assert!(cov.enhanced > 0, "{cov:?}");
}

#[test]
fn stream_stereo() {
    // Stereo SILK: both channels (mid / side) are enhanced and extended independently.
    let cfg = scfg(2, 32000, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 300, 61);
    let cov = run_stream(
        &cfg,
        Mode::Fixed {
            complexity: 7,
            api_fs: 48000,
            bwe: true,
        },
        0,
    );
    assert!(cov.enhanced > 300 && cov.bwe_calls > 0, "{cov:?}");
    let mut cfg = scfg(2, 20000, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 300, 62);
    cfg.vary = true;
    let cov = run_stream(&cfg, Mode::Switching, 8);
    assert!(cov.enhanced > 0, "{cov:?}");
}

#[test]
fn unloaded_model_stream() {
    // silk_LoadOSCEModels failing (e.g. USE_WEIGHTS_FILE without a blob): no enhancement, the
    // features and the reset / cross-fade logic still run.
    let cfg = scfg(1, 16000, sys::OPUS_BANDWIDTH_WIDEBAND, 20, 60, 71);
    let pkts = encode_stream(&cfg);
    let mut cd = c::CapDecoder::new();
    let mut rp = Replay::new("unloaded");
    rp.apply(&cd.take_init_events(), "init");
    cd.set_loaded(false);
    let ctrl = c::CapCtrl {
        n_channels_api: 1,
        n_channels_internal: 1,
        api_sample_rate: 16000,
        internal_sample_rate: 16000,
        payload_size_ms: 20,
        enable_deep_plc: 1,
        osce_method: o::OSCE_METHOD_NOLACE,
        enable_osce_bwe: 0,
        osce_extended_mode: o::OSCE_MODE_SILK_ONLY,
    };
    for (k, p) in pkts.iter().enumerate() {
        cd.ec_init(&p[1..]);
        let (ret, _) = cd.decode(&ctrl, 0, 1);
        assert_eq!(ret, 0);
        rp.apply(&cd.events(), &format!("packet {k}"));
        rp.check_state(&mut cd, &format!("packet {k}"));
    }
    assert!(rp.enhanced_by_method[0] >= 59);
    // Loading from a blob through silk_LoadOSCEModels in C.
    assert_eq!(cd.load(Some(blob())), 0);
    assert!(cd.loaded());
}

#[test]
#[ignore = "long sweep (minutes): cargo test --features osce --test dnn_osce -- --ignored"]
fn long_sweep() {
    let mut seed = 1000;
    for channels in [1, 2] {
        for br in [8000, 12000, 16000, 24000, 40000] {
            for ms in [10, 20, 40, 60] {
                for (mode, loss) in [
                    (
                        Mode::Fixed {
                            complexity: 7,
                            api_fs: 48000,
                            bwe: true,
                        },
                        0,
                    ),
                    (
                        Mode::Fixed {
                            complexity: 6,
                            api_fs: 16000,
                            bwe: false,
                        },
                        3,
                    ),
                    (Mode::Switching, 10),
                ] {
                    seed += 1;
                    let mut cfg = scfg(
                        channels,
                        br * channels,
                        sys::OPUS_BANDWIDTH_WIDEBAND,
                        ms,
                        30_000 / ms as usize,
                        seed,
                    );
                    cfg.fec = seed % 2 == 0;
                    cfg.vary = seed % 3 == 0;
                    run_stream(&cfg, mode, loss);
                }
            }
        }
    }
}

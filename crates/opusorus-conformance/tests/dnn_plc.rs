//! Differential tests for unit `dnn_plc` (dnn/fargan.c, dnn/lpcnet_plc.c and the generated
//! `init_fargan` / `init_plcmodel` bindings) vs the C oracle built with deep PLC.
//!
//! Run with `cargo test -p opusorus-conformance --features deep-plc --test dnn_plc` (needs the
//! model data from `scripts/fetch_dnn_models.sh`). Every comparison is bit-exact.
//!
//! The C oracle uses its compiled-in model tables; the Rust side parses a weight blob that the
//! oracle serializes from those same tables (pitch DNN + PLC + FARGAN, the models
//! `lpcnet_plc_load_model` binds). Features come from the (bit-exact) Rust LPCNet feature
//! extractor run on speech-like and other 16 kHz signals.
#![cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::dnn::fargan::{self as fg, FarganState};
use opusorus::dnn::lpcnet_enc as le;
use opusorus::dnn::lpcnet_plc::{self as lp, LpcnetPlcState};
use opusorus::dnn::nnet::LinearLayer;
use opusorus::dnn::parse_lpcnet_weights as pw;
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::dnn_core as dc;
use opusorus_oracle::dnn_plc as c;

// ---------------------------------------------------------------------------------------------
// Helpers

/// Weight blob with every model `lpcnet_plc_load_model` binds (pitch DNN, PLC, FARGAN).
fn blob_all() -> Vec<u8> {
    let mut b = dc::write_blob(dc::MODEL_PITCHDNN);
    b.extend(dc::write_blob(dc::MODEL_PLC));
    b.extend(dc::write_blob(dc::MODEL_FARGAN));
    b
}

fn rust_fargan(blob: &[u8]) -> Box<FarganState> {
    let mut st = FarganState::new();
    st.load_model(blob).expect("fargan model");
    st
}

fn rust_plc(blob: &[u8]) -> Box<LpcnetPlcState> {
    let mut st = LpcnetPlcState::new();
    assert!(!st.loaded);
    st.load_model(blob).expect("plc models");
    assert!(st.loaded);
    st
}

fn fargan_floats(st: &FarganState) -> Vec<f32> {
    let mut v = Vec::with_capacity(c::FARGAN_STATE_LEN);
    v.push(st.deemph_mem);
    v.extend_from_slice(&st.pitch_buf);
    v.extend_from_slice(&st.cond_conv1_state);
    v.extend_from_slice(&st.fwc0_mem);
    v.extend_from_slice(&st.gru1_state);
    v.extend_from_slice(&st.gru2_state);
    v.extend_from_slice(&st.gru3_state);
    v
}

fn fargan_ints(st: &FarganState) -> [i32; 2] {
    [i32::from(st.cont_initialized), st.last_period]
}

fn set_rust_fargan_state(st: &mut FarganState, f: &[f32], ints: [i32; 2]) {
    assert_eq!(f.len(), c::FARGAN_STATE_LEN);
    let mut pos = 0;
    let mut take = |dst: &mut [f32]| {
        dst.copy_from_slice(&f[pos..pos + dst.len()]);
        pos += dst.len();
    };
    let mut dm = [0f32; 1];
    take(&mut dm);
    st.deemph_mem = dm[0];
    take(&mut st.pitch_buf);
    take(&mut st.cond_conv1_state);
    take(&mut st.fwc0_mem);
    take(&mut st.gru1_state);
    take(&mut st.gru2_state);
    take(&mut st.gru3_state);
    st.cont_initialized = ints[0] != 0;
    st.last_period = ints[1];
}

#[track_caller]
fn check_fargan(what: &str, r: &FarganState, h: &c::Fargan) {
    let (cf, ci) = h.state();
    assert_bits_eq_f32(&format!("{what}: fargan state"), &fargan_floats(r), &cf);
    assert_eq!(fargan_ints(r), ci, "{what}: fargan ints");
}

fn enc_floats(st: &le::LpcnetEncState, v: &mut Vec<f32>) {
    v.extend_from_slice(&st.analysis_mem);
    v.push(st.mem_preemph);
    v.extend(st.prev_if.iter().flat_map(|c| [c.r, c.i]));
    v.extend_from_slice(&st.if_features);
    v.extend_from_slice(&st.xcorr_features);
    v.push(st.dnn_pitch);
    v.extend_from_slice(&st.pitch_mem);
    v.push(st.pitch_filt);
    v.extend_from_slice(&st.exc_buf);
    v.extend_from_slice(&st.lp_buf);
    v.extend_from_slice(&st.lp_mem);
    v.extend_from_slice(&st.lpc);
    v.extend_from_slice(&st.features);
    v.extend_from_slice(&st.sig_mem);
    v.extend_from_slice(&st.burg_cepstrum);
    v.extend_from_slice(&st.pitchdnn.gru_state);
    v.extend_from_slice(&st.pitchdnn.xcorr_mem1);
    v.extend_from_slice(&st.pitchdnn.xcorr_mem2);
}

fn plc_floats(st: &LpcnetPlcState) -> Vec<f32> {
    let mut v = Vec::with_capacity(c::PLC_STATE_LEN);
    for f in &st.fec {
        v.extend_from_slice(f);
    }
    v.extend_from_slice(&st.pcm);
    v.extend_from_slice(&st.features);
    v.extend_from_slice(&st.cont_features);
    for n in [&st.plc_net, &st.plc_bak[0], &st.plc_bak[1]] {
        v.extend_from_slice(&n.gru1_state);
        v.extend_from_slice(&n.gru2_state);
    }
    v.extend(fargan_floats(&st.fargan));
    enc_floats(&st.enc, &mut v);
    v
}

fn plc_ints(st: &LpcnetPlcState) -> Vec<i32> {
    vec![
        i32::from(st.loaded),
        st.analysis_gap,
        st.fec_read_pos,
        st.fec_fill_pos,
        st.fec_skip,
        st.analysis_pos,
        st.predict_pos,
        st.blend,
        st.loss_count,
        i32::from(st.fargan.cont_initialized),
        st.fargan.last_period,
    ]
}

#[track_caller]
fn check_plc(what: &str, r: &LpcnetPlcState, h: &c::Plc) {
    let (cf, ci) = h.state();
    assert_slice_eq(&format!("{what}: plc ints"), &plc_ints(r), &ci);
    assert_bits_eq_f32(&format!("{what}: plc state"), &plc_floats(r), &cf);
}

/// 16 kHz test signals, int16.
fn signal16(kind: usize, n: usize, seed: u64) -> Vec<i16> {
    let s = match kind % 6 {
        0 | 3 => signals::speech_like(n, 1, 16000, seed),
        1 => signals::music_like(n, 1, 16000, seed),
        2 => signals::noise(n, 1, 0.3, seed),
        4 => signals::noise(n, 1, 1e-4, seed),
        _ => (0..n)
            .map(|i| {
                // Loud chirp with silence gaps.
                let t = i as f64 / 16000.0;
                if (i / 4000) % 3 == 2 {
                    0.0
                } else {
                    (0.9 * (2.0 * core::f64::consts::PI * (100.0 + 600.0 * t) * t).sin()) as f32
                }
            })
            .collect(),
    };
    signals::to_i16(&s)
}

/// LPCNet features (`NB_TOTAL_FEATURES` per 160-sample frame) of a signal.
fn features_of(pcm: &[i16], blob: &[u8]) -> Vec<[f32; le::NB_TOTAL_FEATURES]> {
    let mut enc = le::LpcnetEncState::new();
    enc.load_model(blob).expect("pitch model");
    pcm.as_chunks::<160>()
        .0
        .iter()
        .map(|fr| {
            let mut f = [0f32; le::NB_TOTAL_FEATURES];
            le::lpcnet_compute_single_frame_features(&mut enc, fr, &mut f);
            f
        })
        .collect()
}

fn rand_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| amp * rng.f32_sym()).collect()
}

// ---------------------------------------------------------------------------------------------
// Sizes and model binding

#[test]
fn sizes_match_c() {
    assert_eq!(
        c::c_sizes(),
        [
            fg::FARGAN_COND_SIZE,
            fg::COND_NET_FDENSE2_OUT_SIZE,
            fg::SIG_NET_INPUT_SIZE,
            lp::PLC_BUF_SIZE,
            le::PLC_MAX_FEC,
        ]
    );
    assert_eq!(lp::PLC_INPUT_SIZE, c::PLC_INPUT_SIZE);
    assert_eq!(fg::FARGAN_FRAME_SIZE, c::FRAME);
}

/// (field, [bias, subias, weights, float_weights, weights_idx, diag, scale], nb_in, nb_out)
/// of every `linear_init` line of a generated `init_*` function.
type Spec = (String, [Option<String>; 7], usize, usize);

fn init_specs(file: &str) -> Vec<Spec> {
    let path = format!(
        "{}/../../vendor/libopus/dnn/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("model data (scripts/fetch_dnn_models.sh)");
    let start = text.rfind("\nint init_").expect("init function");
    let mut out = Vec::new();
    for line in text[start..].lines() {
        let Some(p) = line.find("linear_init(&model->") else {
            continue;
        };
        let rest = &line[p + "linear_init(&model->".len()..];
        let (field, rest) = rest.split_once(',').expect("field");
        let args = rest.trim_start().strip_prefix("arrays,").expect("arrays");
        let args = &args[..args.rfind("))").expect("call end")];
        let toks: Vec<&str> = args.split(',').map(str::trim).collect();
        assert_eq!(toks.len(), 9, "{line}");
        let names = core::array::from_fn(|i| {
            (toks[i] != "NULL").then(|| toks[i].trim_matches('"').to_string())
        });
        out.push((
            field.to_string(),
            names,
            toks[7].parse().expect("nb_in"),
            toks[8].parse().expect("nb_out"),
        ));
    }
    assert!(!out.is_empty());
    out
}

fn fargan_layer<'a>(m: &'a fg::Fargan, field: &str) -> &'a LinearLayer {
    match field {
        "cond_net_pembed" => &m.cond_net_pembed,
        "cond_net_fdense1" => &m.cond_net_fdense1,
        "cond_net_fconv1" => &m.cond_net_fconv1,
        "cond_net_fdense2" => &m.cond_net_fdense2,
        "sig_net_cond_gain_dense" => &m.sig_net_cond_gain_dense,
        "sig_net_fwc0_conv" => &m.sig_net_fwc0_conv,
        "sig_net_fwc0_glu_gate" => &m.sig_net_fwc0_glu_gate,
        "sig_net_gru1_input" => &m.sig_net_gru1_input,
        "sig_net_gru1_recurrent" => &m.sig_net_gru1_recurrent,
        "sig_net_gru2_input" => &m.sig_net_gru2_input,
        "sig_net_gru2_recurrent" => &m.sig_net_gru2_recurrent,
        "sig_net_gru3_input" => &m.sig_net_gru3_input,
        "sig_net_gru3_recurrent" => &m.sig_net_gru3_recurrent,
        "sig_net_gru1_glu_gate" => &m.sig_net_gru1_glu_gate,
        "sig_net_gru2_glu_gate" => &m.sig_net_gru2_glu_gate,
        "sig_net_gru3_glu_gate" => &m.sig_net_gru3_glu_gate,
        "sig_net_skip_glu_gate" => &m.sig_net_skip_glu_gate,
        "sig_net_skip_dense" => &m.sig_net_skip_dense,
        "sig_net_sig_dense_out" => &m.sig_net_sig_dense_out,
        "sig_net_gain_dense_out" => &m.sig_net_gain_dense_out,
        _ => panic!("unknown FARGAN field {field}"),
    }
}

fn plc_layer<'a>(m: &'a lp::PlcModel, field: &str) -> &'a LinearLayer {
    match field {
        "plc_dense_in" => &m.plc_dense_in,
        "plc_dense_out" => &m.plc_dense_out,
        "plc_gru1_input" => &m.plc_gru1_input,
        "plc_gru1_recurrent" => &m.plc_gru1_recurrent,
        "plc_gru2_input" => &m.plc_gru2_input,
        "plc_gru2_recurrent" => &m.plc_gru2_recurrent,
        _ => panic!("unknown PLC field {field}"),
    }
}

/// Compares every bound array of a Rust layer with the C one bound by the generated line.
fn compare_layer(what: &str, r: &LinearLayer, model: i32, s: &Spec) {
    let n = |i: usize| s.1[i].as_deref();
    let (h, ret) = dc::Linear::init(model, n(0), n(1), n(2), n(3), n(4), n(5), n(6), s.2, s.3);
    assert_eq!(ret, 0, "{what}: C linear_init");
    assert_eq!(h.dims(), (r.nb_inputs, r.nb_outputs), "{what}: dims");
    let f32_fields: [(i32, &Option<Vec<f32>>); 5] = [
        (0, &r.bias),
        (1, &r.subias),
        (3, &r.float_weights),
        (5, &r.diag),
        (6, &r.scale),
    ];
    for (f, v) in f32_fields {
        let cv = h.field_f32(f, v.as_ref().map_or(0, Vec::len));
        assert_eq!(cv.is_some(), v.is_some(), "{what}: field {f} presence");
        if let (Some(cv), Some(v)) = (cv, v) {
            assert_bits_eq_f32(&format!("{what}: field {f}"), v, &cv);
        }
    }
    let cw = h.field_i8(r.weights.as_ref().map_or(0, Vec::len));
    assert_eq!(cw.is_some(), r.weights.is_some(), "{what}: weights");
    if let (Some(cw), Some(w)) = (cw, &r.weights) {
        assert_slice_eq(&format!("{what}: weights"), w, &cw);
    }
    assert!(r.weights_idx.is_none() && r.diag.is_none(), "{what}");
}

#[test]
fn model_binding_matches_generated_init() {
    let blob = blob_all();
    let list = pw::parse_weights(&blob).unwrap();
    let fargan = fg::init_fargan(&list).expect("init_fargan");
    let specs = init_specs("fargan_data.c");
    assert_eq!(specs.len(), 20);
    for s in &specs {
        compare_layer(&s.0, fargan_layer(&fargan, &s.0), dc::MODEL_FARGAN, s);
    }
    let plc = lp::init_plcmodel(&list).expect("init_plcmodel");
    let specs = init_specs("plc_data.c");
    assert_eq!(specs.len(), 6);
    for s in &specs {
        compare_layer(&s.0, plc_layer(&plc, &s.0), dc::MODEL_PLC, s);
    }
    // The loaders bind the same models.
    let st = rust_plc(&blob);
    assert_eq!(st.model, plc);
    assert_eq!(st.fargan.model, fargan);
}

#[test]
fn load_model_errors() {
    let pitch = dc::write_blob(dc::MODEL_PITCHDNN);
    let plc = dc::write_blob(dc::MODEL_PLC);
    let fargan = dc::write_blob(dc::MODEL_FARGAN);
    // Missing arrays.
    let list = pw::parse_weights(&pitch).unwrap();
    assert!(fg::init_fargan(&list).is_err());
    assert!(lp::init_plcmodel(&list).is_err());
    let mut f = FarganState::new();
    assert!(f.load_model(&plc).is_err());
    assert!(f.load_model(&fargan).is_ok());
    // A corrupted blob is rejected (C would crash on it) and keeps the model.
    let m = f.model.clone();
    assert!(f.load_model(&fargan[..fargan.len() - 3]).is_err());
    assert_eq!(f.model, m);

    // lpcnet_plc_load_model: each model is required; `loaded` only when all bind.
    let mut st = LpcnetPlcState::new();
    let mut no_fargan = pitch.clone();
    no_fargan.extend_from_slice(&plc);
    assert!(st.load_model(&no_fargan).is_err());
    assert!(!st.loaded);
    // As in C, the PLC model and the pitch DNN were already replaced.
    assert_eq!(
        st.model,
        lp::init_plcmodel(&pw::parse_weights(&plc).unwrap()).unwrap()
    );
    let mut no_pitch = plc.clone();
    no_pitch.extend_from_slice(&fargan);
    assert!(st.load_model(&no_pitch).is_err());
    assert!(!st.loaded);
    assert!(st.load_model(&blob_all()).is_ok());
    assert!(st.loaded);
    // init / reset keep the models and `loaded` (the compiled-in-table equivalent).
    st.lpcnet_plc_init();
    assert!(st.loaded);
    assert_eq!(
        st.fargan.model,
        fg::init_fargan(&pw::parse_weights(&fargan).unwrap()).unwrap()
    );
    lp::lpcnet_plc_reset(&mut st);
    assert!(st.loaded);

    // C lpcnet_plc_load_model / fargan_load_model on the same blob give the same states.
    let blob = blob_all();
    let mut h = c::Plc::new();
    assert_eq!(h.load_model(&blob), 0);
    let mut hf = c::Fargan::new();
    assert_eq!(hf.load_model(&blob), 0);
    hf.init();
    let r = rust_plc(&blob);
    check_plc("after C load_model", &r, &h);
    check_fargan("after C fargan_init", &rust_fargan(&blob), &hf);
}

// ---------------------------------------------------------------------------------------------
// FARGAN

/// Periods reachable from the pitch feature (`0..=INT_MAX`: the float-to-int cast saturates
/// on the AArch64 oracle and in Rust).
const PERIODS: [i32; 15] = [
    0,
    1,
    2,
    3,
    20,
    31,
    32,
    33,
    100,
    200,
    254,
    255,
    256,
    300,
    i32::MAX,
];

#[test]
fn fargan_static_helpers() {
    let blob = blob_all();
    let mut rng = Rng::new(0xFA26);
    let mut r = rust_fargan(&blob);
    let mut h = c::Fargan::new();
    for trial in 0..400 {
        // Random state (tanh-range GRU states, audio-range buffers).
        let mut sf = rand_vec(&mut rng, c::FARGAN_STATE_LEN, 1.0);
        let amp = [0.01f32, 0.3, 1.0, 3.0][trial % 4];
        for v in &mut sf[1..257] {
            *v *= amp;
        }
        let ints = [1, PERIODS[trial % PERIODS.len()]];
        set_rust_fargan_state(&mut r, &sf, ints);
        h.set_state(&sf, ints);
        check_fargan("set_state", &r, &h);

        let period = if trial % 3 == 0 {
            PERIODS[rng.range_i32(0, PERIODS.len() as i32 - 1) as usize]
        } else {
            rng.range_i32(0, 400)
        };
        let feat = rand_vec(&mut rng, le::NB_FEATURES, 2.0);
        let mut rc = [0f32; fg::COND_NET_FDENSE2_OUT_SIZE];
        fg::compute_fargan_cond(&mut r, &mut rc, &feat, period);
        let cc = h.compute_cond(&feat, period);
        assert_bits_eq_f32(&format!("cond #{trial} p={period}"), &rc, &cc);
        check_fargan(&format!("cond #{trial}"), &r, &h);

        for sub in 0..4 {
            let cond = &rc[sub * fg::FARGAN_COND_SIZE..(sub + 1) * fg::FARGAN_COND_SIZE];
            let p = PERIODS[(trial + sub) % PERIODS.len()];
            let mut rp = [0f32; fg::FARGAN_SUBFRAME_SIZE];
            fg::run_fargan_subframe(&mut r, &mut rp, cond, p);
            let cp = h.run_subframe(cond, p);
            assert_bits_eq_f32(&format!("subframe #{trial}/{sub} p={p}"), &rp, &cp);
            check_fargan(&format!("subframe #{trial}/{sub}"), &r, &h);
        }

        let mut rp = rand_vec(&mut rng, fg::FARGAN_SUBFRAME_SIZE, 1.0);
        let mut cp = rp.clone();
        let mut rm = rng.f32_sym();
        let mut cm = rm;
        fg::fargan_deemphasis(&mut rp, &mut rm);
        c::fargan_deemphasis(&mut cp, &mut cm);
        assert_bits_eq_f32("deemphasis", &rp, &cp);
        assert_eq!(rm.to_bits(), cm.to_bits());
    }
}

/// Feature perturbations for the FARGAN streams.
fn perturb(kind: usize, frame: usize, f: &mut [f32]) {
    match kind {
        // Real features.
        0 => {}
        // Pitch feature sweeps through the whole period range, including periods of 0
        // (feature > 7.5) and above INT_MAX (saturated).
        1 => {
            let sweep = [-1.5f32, 0.0, 1.4, 3.0, 7.0, 7.6, 12.0, -3.0, -40.0, -1e6];
            f[18] = sweep[frame % sweep.len()];
        }
        // Loud / attenuated energy.
        2 => f[0] += if frame % 7 < 3 { 3.0 } else { -6.0 },
        _ => {}
    }
}

#[test]
fn fargan_streams() {
    let blob = blob_all();
    let mut rng = Rng::new(0xFA);
    for stream in 0..9 {
        let frames = 120;
        let sig = signal16(stream, frames * 160, rng.next_u64());
        let feats = features_of(&sig, &blob);
        let kind = stream % 3;
        let mut r = rust_fargan(&blob);
        let mut h = c::Fargan::new();
        check_fargan("init", &r, &h);
        // Prime from frames 0..5 (as lpcnet_plc_conceal does with its history).
        let prime = |start: usize, r: &mut FarganState, h: &mut c::Fargan| {
            let pcm0: Vec<f32> = sig[(start + 3) * 160..(start + 5) * 160]
                .iter()
                .map(|&v| f32::from(v) / 32768.0)
                .collect();
            let mut f0 = Vec::new();
            for (i, f) in feats[start..start + 5].iter().enumerate() {
                let mut f = *f;
                perturb(kind, start + i, &mut f);
                f0.extend_from_slice(&f[..le::NB_FEATURES]);
            }
            fg::fargan_cont(r, &pcm0, &f0);
            h.cont(&pcm0, &f0);
        };
        prime(0, &mut r, &mut h);
        check_fargan(&format!("cont {stream}"), &r, &h);
        #[expect(
            clippy::needless_range_loop,
            reason = "frame index drives several things"
        )]
        for fr in 5..frames {
            if fr % 40 == 0 {
                // Re-prime mid-stream, sometimes after fargan_init.
                if stream % 2 == 1 {
                    r.fargan_init();
                    h.init();
                    check_fargan("fargan_init", &r, &h);
                }
                prime(fr - 5, &mut r, &mut h);
                check_fargan(&format!("re-cont {stream}/{fr}"), &r, &h);
            }
            let mut f = feats[fr];
            perturb(kind, fr, &mut f);
            if fr % 2 == 0 {
                let mut rp = [0f32; fg::FARGAN_FRAME_SIZE];
                fg::fargan_synthesize(&mut r, &mut rp, &f);
                let cp = h.synthesize(&f);
                assert_bits_eq_f32(&format!("synth {stream}/{fr}"), &rp, &cp);
            } else {
                let mut rp = [0i16; fg::FARGAN_FRAME_SIZE];
                fg::fargan_synthesize_int(&mut r, &mut rp, &f);
                let cp = h.synthesize_int(&f);
                assert_slice_eq(&format!("synth_int {stream}/{fr}"), &rp, &cp);
            }
            check_fargan(&format!("synth {stream}/{fr}"), &r, &h);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LPCNet PLC

/// Loss patterns: returns whether frame `i` is lost.
const fn lost(pattern: usize, i: usize, rng: &mut Rng) -> bool {
    match pattern {
        // Random 15 % losses.
        0 => rng.next_u32() % 100 < 15,
        // Bursts of 1..=6 every 13 frames.
        1 => i % 13 < 1 + (i / 13) % 6,
        // Long bursts (attenuation, loss_count >= 10).
        2 => (i % 60) >= 20 && (i % 60) < 50,
        // Loss from the very start (no history), then alternate.
        3 => i < 4 || (i > 30 && i.is_multiple_of(2)),
        // Very long loss (loss_count up to ~100).
        4 => 25 <= i && i < 140,
        // Sparse single losses right after short good runs.
        _ => i % 5 == 4 || i.is_multiple_of(17),
    }
}

#[test]
fn plc_streams() {
    let blob = blob_all();
    let mut rng = Rng::new(0x91C);
    let (mut concealed, mut fec_frames, mut attenuated, mut energy) = (0usize, 0usize, 0, 0f64);
    for stream in 0..24 {
        let frames = 300;
        let sig = signal16(stream, frames * 160, rng.next_u64());
        let feats = features_of(&sig, &blob);
        let pattern = stream % 6;
        // FEC (DRED-like) feature queue before loss bursts in half of the streams.
        let use_fec = stream % 12 >= 6;
        let mut r = rust_plc(&blob);
        let mut h = c::Plc::new();
        check_plc("init", &r, &h);
        let mut prev_lost = false;
        for fr in 0..frames {
            if fr == 100 && stream % 4 == 1 {
                lp::lpcnet_plc_reset(&mut r);
                h.reset();
                check_plc("reset", &r, &h);
            }
            let is_lost = lost(pattern, fr, &mut rng);
            if is_lost {
                if use_fec && !prev_lost {
                    // As opus_decoder.c does with DRED: clear, then queue features for the
                    // upcoming frames, some skipped (NULL).
                    lp::lpcnet_plc_fec_clear(&mut r);
                    h.fec_clear();
                    let n = rng.range_i32(0, 12) as usize;
                    for k in 0..n {
                        if rng.next_u32().is_multiple_of(5) {
                            lp::lpcnet_plc_fec_add(&mut r, None);
                            h.fec_add(None);
                        } else {
                            let f = &feats[(fr + k).min(frames - 1)][..le::NB_FEATURES];
                            lp::lpcnet_plc_fec_add(&mut r, Some(f));
                            h.fec_add(Some(f));
                        }
                    }
                    check_plc(&format!("fec_add {stream}/{fr}"), &r, &h);
                }
                let mut rp = [0i16; 160];
                lp::lpcnet_plc_conceal(&mut r, &mut rp);
                let (ret, cp) = h.conceal();
                assert_eq!(ret, 0);
                assert_slice_eq(&format!("conceal {stream}/{fr}"), &rp, &cp);
                concealed += 1;
                fec_frames += usize::from(r.loss_count == 0);
                attenuated += usize::from(r.loss_count >= 10);
                energy += rp.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            } else {
                let pcm = &sig[fr * 160..(fr + 1) * 160];
                lp::lpcnet_plc_update(&mut r, pcm);
                assert_eq!(h.update(pcm), 0);
            }
            check_plc(&format!("stream {stream} frame {fr}"), &r, &h);
            prev_lost = is_lost;
        }
    }
    // The streams exercise real concealment, the FEC path and the long-loss attenuation.
    assert!(
        concealed > 1500 && fec_frames > 100 && attenuated > 300,
        "{concealed} {fec_frames} {attenuated}"
    );
    assert!(
        energy / (concealed as f64 * 160.0) > 1e4,
        "concealment is not silent"
    );
}

#[test]
fn plc_fec_queue_limits() {
    // Fill the FEC queue completely, mix skips, and conceal through all of it.
    let blob = blob_all();
    let sig = signal16(0, 40 * 160, 5);
    let feats = features_of(&sig, &blob);
    let mut r = rust_plc(&blob);
    let mut h = c::Plc::new();
    for fr in 0..10 {
        let pcm = &sig[fr * 160..(fr + 1) * 160];
        lp::lpcnet_plc_update(&mut r, pcm);
        h.update(pcm);
    }
    lp::lpcnet_plc_fec_clear(&mut r);
    h.fec_clear();
    for _ in 0..3 {
        lp::lpcnet_plc_fec_add(&mut r, None);
        h.fec_add(None);
    }
    for k in 0..le::PLC_MAX_FEC {
        let f = &feats[k % feats.len()][..le::NB_FEATURES];
        lp::lpcnet_plc_fec_add(&mut r, Some(f));
        h.fec_add(Some(f));
    }
    check_plc("full queue", &r, &h);
    for fr in 0..le::PLC_MAX_FEC + 10 {
        let mut rp = [0i16; 160];
        lp::lpcnet_plc_conceal(&mut r, &mut rp);
        let (_, cp) = h.conceal();
        assert_slice_eq(&format!("conceal {fr}"), &rp, &cp);
        check_plc(&format!("conceal {fr}"), &r, &h);
    }
}

#[test]
fn plc_static_helpers() {
    let blob = blob_all();
    let mut rng = Rng::new(0x5EC);
    let sig = signal16(0, 60 * 160, 11);
    let feats = features_of(&sig, &blob);
    let mut r = rust_plc(&blob);
    let mut h = c::Plc::new();
    for fr in 0..20 {
        let pcm = &sig[fr * 160..(fr + 1) * 160];
        lp::lpcnet_plc_update(&mut r, pcm);
        h.update(pcm);
    }
    for t in 0..300 {
        match t % 4 {
            0 => {
                let amp = [0.5f32, 1.0, 4.0][t % 3];
                let input = rand_vec(&mut rng, lp::PLC_INPUT_SIZE, amp);
                let mut ro = [0f32; le::NB_FEATURES];
                lp::compute_plc_pred(&mut r, &mut ro, &input);
                let co = h.compute_plc_pred(&input);
                assert_bits_eq_f32(&format!("compute_plc_pred #{t}"), &ro, &co);
            }
            1 => {
                if t % 8 == 1 {
                    lp::lpcnet_plc_fec_clear(&mut r);
                    h.fec_clear();
                    for k in 0..rng.range_i32(0, 4) as usize {
                        if k == 1 {
                            lp::lpcnet_plc_fec_add(&mut r, None);
                            h.fec_add(None);
                        }
                        let f = &feats[(t + k) % feats.len()][..le::NB_FEATURES];
                        lp::lpcnet_plc_fec_add(&mut r, Some(f));
                        h.fec_add(Some(f));
                    }
                }
                let init: [f32; le::NB_FEATURES] = core::array::from_fn(|_| rng.f32_sym());
                let mut ro = init;
                let rret = lp::get_fec_or_pred(&mut r, &mut ro);
                let (cret, co) = h.get_fec_or_pred(&init);
                assert_eq!(i32::from(rret), cret, "get_fec_or_pred #{t}");
                assert_bits_eq_f32(&format!("get_fec_or_pred #{t}"), &ro, &co);
            }
            2 => {
                let f = rand_vec(&mut rng, le::NB_FEATURES, 3.0);
                lp::queue_features(&mut r, &f);
                h.queue_features(&f);
            }
            _ => {
                let fr = 20 + t % 40;
                let pcm = &sig[fr * 160..(fr + 1) * 160];
                lp::lpcnet_plc_update(&mut r, pcm);
                h.update(pcm);
            }
        }
        check_plc(&format!("helper #{t}"), &r, &h);
    }
}

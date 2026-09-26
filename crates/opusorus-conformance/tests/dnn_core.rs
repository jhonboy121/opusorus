//! Differential tests for unit `dnn_core` (dnn/nnet.c, nnet_arch.h, vec.h, common.h, kiss99.c,
//! parse_lpcnet_weights.c, write_lpcnet_weights.c, burg.c, freq.c, lpcnet_tables.c,
//! pitchdnn.c, lpcnet_enc.c, nndsp.c) vs the C oracle built with the DNN features.
//!
//! Run with `cargo test -p opusorus-conformance --features deep-plc,osce --test dnn_core`
//! (needs the model data from `scripts/fetch_dnn_models.sh`). Every comparison is bit-exact.
//!
//! The C oracle uses its compiled-in model tables; the Rust side parses weight blobs that the
//! oracle serializes from those same tables (write_lpcnet_weights.c format), and every parsed
//! array and every bound layer is compared with the C one.
#![cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::celt::static_modes::KissFftCpx;
use opusorus::dnn::{
    burg, common, freq, kiss99, lpcnet_enc as le, lpcnet_tables, nnet, nnet_arch,
    parse_lpcnet_weights as pw, pitchdnn, vec as dv,
};
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::dnn_core as c;

const ACTIVATIONS: [i32; 7] = [
    nnet::ACTIVATION_LINEAR,
    nnet::ACTIVATION_SIGMOID,
    nnet::ACTIVATION_TANH,
    nnet::ACTIVATION_RELU,
    nnet::ACTIVATION_SOFTMAX,
    nnet::ACTIVATION_SWISH,
    nnet::ACTIVATION_EXP,
];

/// (oracle model id, generated data file).
fn models() -> Vec<(i32, &'static str)> {
    let all = [
        (c::MODEL_PITCHDNN, "pitchdnn_data.c"),
        (c::MODEL_PLC, "plc_data.c"),
        (c::MODEL_FARGAN, "fargan_data.c"),
        (c::MODEL_RDOVAE_ENC, "dred_rdovae_enc_data.c"),
        (c::MODEL_RDOVAE_DEC, "dred_rdovae_dec_data.c"),
        (c::MODEL_LACE, "lace_data.c"),
        (c::MODEL_NOLACE, "nolace_data.c"),
        (c::MODEL_BBWENET, "bbwenet_data.c"),
    ];
    all.into_iter()
        .filter(|(m, _)| c::model_count(*m).is_some())
        .collect()
}

fn rand_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| amp * rng.f32_sym()).collect()
}

fn cpx_to_flat(x: &[KissFftCpx]) -> Vec<f32> {
    x.iter().flat_map(|c| [c.r, c.i]).collect()
}

// ---------------------------------------------------------------------------------------------
// Layer specs parsed from the generated init_* functions.

#[derive(Debug, Clone)]
struct LinearSpec {
    field: String,
    names: [Option<String>; 7],
    nb_in: usize,
    nb_out: usize,
}

#[derive(Debug, Clone)]
struct Conv2dSpec {
    field: String,
    bias: Option<String>,
    weights: Option<String>,
    dims: [usize; 4],
}

fn arg(tok: &str) -> Option<String> {
    let t = tok.trim();
    if t == "NULL" {
        None
    } else {
        Some(t.trim_matches('"').to_string())
    }
}

/// Parses the `linear_init` / `conv2d_init` calls of the `init_*` function of a generated
/// `*_data.c` file.
fn init_specs(file: &str) -> (Vec<LinearSpec>, Vec<Conv2dSpec>) {
    let path = format!(
        "{}/../../vendor/libopus/dnn/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("model data (scripts/fetch_dnn_models.sh)");
    let start = text.rfind("\nint init_").expect("init function");
    let mut lin = Vec::new();
    let mut conv = Vec::new();
    for line in text[start..].lines() {
        let Some(p) = line.find("_init(&model->") else {
            continue;
        };
        let kind = &line[..p];
        let rest = &line[p + "_init(&model->".len()..];
        let (field, rest) = rest.split_once(',').expect("field");
        let args = rest
            .trim_start()
            .strip_prefix("arrays,")
            .expect("arrays arg");
        let args = &args[..args.rfind("))").expect("call end")];
        let toks: Vec<&str> = args.split(',').collect();
        if kind.ends_with("linear") {
            assert_eq!(toks.len(), 9, "{line}");
            lin.push(LinearSpec {
                field: field.to_string(),
                names: core::array::from_fn(|i| arg(toks[i])),
                nb_in: toks[7].trim().parse().expect("nb_in"),
                nb_out: toks[8].trim().parse().expect("nb_out"),
            });
        } else {
            assert!(kind.ends_with("conv2d"), "{line}");
            assert_eq!(toks.len(), 6, "{line}");
            conv.push(Conv2dSpec {
                field: field.to_string(),
                bias: arg(toks[0]),
                weights: arg(toks[1]),
                dims: core::array::from_fn(|i| toks[2 + i].trim().parse().expect("dim")),
            });
        }
    }
    assert!(!lin.is_empty(), "{file}");
    (lin, conv)
}

fn rust_linear(
    arrays: &[pw::WeightArray<'_>],
    s: &LinearSpec,
    nb_in: usize,
    nb_out: usize,
) -> opusorus::Result<nnet::LinearLayer> {
    let n = |i: usize| s.names[i].as_deref();
    pw::linear_init(
        arrays,
        n(0),
        n(1),
        n(2),
        n(3),
        n(4),
        n(5),
        n(6),
        nb_in,
        nb_out,
    )
}

fn c_linear(model: i32, s: &LinearSpec, nb_in: usize, nb_out: usize) -> (c::Linear, i32) {
    let n = |i: usize| s.names[i].as_deref();
    c::Linear::init(
        model,
        n(0),
        n(1),
        n(2),
        n(3),
        n(4),
        n(5),
        n(6),
        nb_in,
        nb_out,
    )
}

/// Compares every bound array of a Rust layer with the C one.
fn compare_layer(what: &str, r: &nnet::LinearLayer, h: &c::Linear) {
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
    assert_eq!(
        cw.is_some(),
        r.weights.is_some(),
        "{what}: weights presence"
    );
    if let (Some(cw), Some(w)) = (cw, &r.weights) {
        assert_slice_eq(&format!("{what}: weights"), w, &cw);
    }
    let ci = h.field_i32(4, r.weights_idx.as_ref().map_or(0, Vec::len));
    assert_eq!(
        ci.is_some(),
        r.weights_idx.is_some(),
        "{what}: idx presence"
    );
    if let (Some(ci), Some(idx)) = (ci, &r.weights_idx) {
        assert_slice_eq(&format!("{what}: idx"), idx, &ci);
    }
}

/// compute_linear + compute_generic_dense (every activation) on random inputs.
fn check_linear_compute(
    what: &str,
    rng: &mut Rng,
    r: &nnet::LinearLayer,
    h: &c::Linear,
    trials: usize,
) {
    for t in 0..trials {
        // Mostly the tanh range; sometimes beyond +-1 (int8 input quantization wraps).
        let amp = [1.0f32, 0.5, 1.6, 4.0, 1e-3][t % 5];
        let input = rand_vec(rng, r.nb_inputs, amp);
        let mut ro = vec![0f32; r.nb_outputs];
        let mut co = vec![0f32; r.nb_outputs];
        nnet::compute_linear(r, &mut ro, &input);
        h.compute_linear(&mut co, &input);
        assert_bits_eq_f32(&format!("{what}: compute_linear #{t}"), &ro, &co);
        let act = ACTIVATIONS[t % ACTIVATIONS.len()];
        nnet::compute_generic_dense(r, &mut ro, &input, act);
        h.compute_generic_dense(&mut co, &input, act);
        assert_bits_eq_f32(&format!("{what}: dense act {act} #{t}"), &ro, &co);
    }
}

// ---------------------------------------------------------------------------------------------
// Tables, blobs, parsing

#[test]
fn lpcnet_tables_match() {
    let t = c::tables();
    let k = &lpcnet_tables::KFFT;
    assert_eq!(k.nfft, t.nfft);
    assert_eq!(k.shift, t.shift);
    assert_eq!(k.scale.to_bits(), t.scale.to_bits());
    assert_eq!(k.factors, t.factors);
    assert_slice_eq("bitrev", &k.bitrev, &t.bitrev);
    assert_bits_eq_f32("twiddles", &cpx_to_flat(&k.twiddles), &t.twiddles);
    assert_bits_eq_f32("half_window", &lpcnet_tables::HALF_WINDOW, &t.half_window);
    assert_bits_eq_f32("dct_table", &lpcnet_tables::DCT_TABLE, &t.dct_table);
}

#[test]
fn weight_blobs_roundtrip_every_model() {
    // deep PLC models always; DRED and OSCE models with their features.
    let expected =
        3 + if cfg!(feature = "dred") { 2 } else { 0 } + if cfg!(feature = "osce") { 3 } else { 0 };
    assert_eq!(models().len(), expected);
    for (model, file) in models() {
        let table = c::model_arrays(model);
        let blob = c::write_blob(model);
        let parsed = pw::parse_weights(&blob).expect("valid blob");
        assert_eq!(parsed.len(), table.len(), "{file}: array count");
        for (p, t) in parsed.iter().zip(&table) {
            assert_eq!(p.name, t.name.as_bytes(), "{file}: name");
            assert_eq!((p.type_, p.size), (t.type_, t.size), "{file}: {}", t.name);
            assert_eq!(p.data, &t.data[..], "{file}: {} data", t.name);
        }
        // C parse_weights agrees on counts and offsets.
        let cp = c::parse_weights(&blob).expect("C accepts the blob");
        assert_eq!(cp.len(), parsed.len());
        let base = blob.as_ptr() as usize;
        for (p, ci) in parsed.iter().zip(&cp) {
            assert_eq!(
                [
                    p.type_,
                    p.size,
                    (p.data.as_ptr() as usize - base) as i32,
                    (p.name.as_ptr() as usize - base) as i32
                ],
                *ci,
                "{file}"
            );
        }
        // The Rust writer reproduces the C blob byte for byte.
        let mut out = Vec::new();
        pw::write_weights(&parsed, &mut out);
        assert!(out == blob, "{file}: write_weights differs");
    }
}

/// Mutates a blob and checks the Rust parser accepts/rejects exactly as C does.
#[test]
fn parse_weights_corrupted_blobs() {
    let blob = c::write_blob(c::MODEL_PITCHDNN);
    let good = pw::parse_weights(&blob).expect("valid");
    let offsets: Vec<usize> = good
        .iter()
        .map(|a| a.data.as_ptr() as usize - blob.as_ptr() as usize - 64)
        .collect();
    let mut rng = Rng::new(0x000D_1B0B);
    let check = |b: &[u8], what: &str| {
        let r = pw::parse_weights(b);
        let cr = c::parse_weights(b);
        assert_eq!(r.is_ok(), cr.is_some(), "{what}");
        if let (Ok(r), Some(cr)) = (r, cr) {
            assert_eq!(r.len(), cr.len(), "{what}");
            for (a, ci) in r.iter().zip(&cr) {
                let base = b.as_ptr() as usize;
                assert_eq!(
                    [
                        a.type_,
                        a.size,
                        (a.data.as_ptr() as usize - base) as i32,
                        (a.name.as_ptr() as usize - base) as i32
                    ],
                    *ci,
                    "{what}"
                );
            }
        }
    };
    check(&[], "empty");
    for len in [1, 63, 64, 65, 127, 128, blob.len() - 1, blob.len() - 64] {
        check(&blob[..len], &format!("truncated {len}"));
    }
    for trial in 0..3000 {
        let mut b = blob.clone();
        let rec = offsets[rng.range_i32(0, offsets.len() as i32 - 1) as usize];
        match trial % 8 {
            0 => {
                // block_size
                let v = [0i32, 1, -1, -64, 63, 64, 128, 1 << 30, i32::MIN]
                    [rng.range_i32(0, 8) as usize];
                b[rec + 16..rec + 20].copy_from_slice(&v.to_le_bytes());
            }
            1 => {
                // size
                let v = [0i32, -1, 1, 3, 64, 65, i32::MAX][rng.range_i32(0, 6) as usize];
                b[rec + 12..rec + 16].copy_from_slice(&v.to_le_bytes());
            }
            2 => b[rec + 63] = rng.next_u32() as u8 | 1, // unterminated name
            3 => {
                // type / version / head (not checked)
                let o = rng.range_i32(0, 11) as usize;
                b[rec + o] ^= 1 << rng.range_i32(0, 7);
            }
            4 => {
                let n = rng.range_i32(0, blob.len() as i32 - 1) as usize;
                b.truncate(n);
            }
            5 => {
                // random byte flips anywhere
                for _ in 0..rng.range_i32(1, 8) {
                    let o = rng.range_i32(0, b.len() as i32 - 1) as usize;
                    b[o] = rng.next_u32() as u8;
                }
            }
            6 => {
                // shrink size below block_size (valid)
                let size = i32::from_le_bytes(b[rec + 12..rec + 16].try_into().unwrap());
                let v = rng.range_i32(0, size);
                b[rec + 12..rec + 16].copy_from_slice(&v.to_le_bytes());
            }
            _ => {
                // append garbage
                let n = rng.range_i32(1, 200) as usize;
                let mut tail = vec![0u8; n];
                rng.fill_bytes(&mut tail);
                b.extend_from_slice(&tail);
            }
        }
        check(&b, &format!("mutation #{trial}"));
    }
}

/// Binds every layer of every model with Rust `linear_init`/`conv2d_init` from the blob and
/// with C from the compiled-in tables; compares all bound arrays and the layer outputs.
#[test]
fn real_model_layers() {
    let mut rng = Rng::new(0x001A_7E75);
    for (model, file) in models() {
        let blob = c::write_blob(model);
        let arrays = pw::parse_weights(&blob).expect("valid blob");
        let (lin, conv) = init_specs(file);
        let mut layers = Vec::new();
        for s in &lin {
            let what = format!("{file}:{}", s.field);
            let r = rust_linear(&arrays, s, s.nb_in, s.nb_out).expect(&what);
            let (h, ret) = c_linear(model, s, s.nb_in, s.nb_out);
            assert_eq!(ret, 0, "{what}");
            compare_layer(&what, &r, &h);
            check_linear_compute(&what, &mut rng, &r, &h, 21);
            // Wrong dimensions fail (or succeed) identically.
            for (di, dout) in [(0i64, 8i64), (0, -8), (4, 0), (-4, 0), (1, 1)] {
                let (ni, no) = (s.nb_in as i64 + di, s.nb_out as i64 + dout);
                if ni <= 0 || no <= 0 {
                    continue;
                }
                let r2 = rust_linear(&arrays, s, ni as usize, no as usize);
                let (_h2, ret2) = c_linear(model, s, ni as usize, no as usize);
                assert_eq!(r2.is_ok(), ret2 == 0, "{what}: dims {ni}x{no}");
            }
            layers.push((s.clone(), r, h));
        }
        for s in &conv {
            let what = format!("{file}:{}", s.field);
            let [i, o, kt, kh] = s.dims;
            let r = pw::conv2d_init(
                &arrays,
                s.bias.as_deref(),
                s.weights.as_deref(),
                i,
                o,
                kt,
                kh,
            )
            .expect(&what);
            let (h, ret) =
                c::Conv2d::init(model, s.bias.as_deref(), s.weights.as_deref(), i, o, kt, kh);
            assert_eq!(ret, 0, "{what}");
            assert_eq!(r.bias.is_some(), h.field(0).is_some());
            if let Some(b) = &r.bias {
                assert_bits_eq_f32(&what, b, &h.field(0).unwrap());
            }
            assert_bits_eq_f32(
                &what,
                r.float_weights.as_ref().unwrap(),
                &h.field(1).unwrap(),
            );
            let r_bad = pw::conv2d_init(
                &arrays,
                s.bias.as_deref(),
                s.weights.as_deref(),
                i + 1,
                o,
                kt,
                kh,
            );
            let (_hb, ret_bad) = c::Conv2d::init(
                model,
                s.bias.as_deref(),
                s.weights.as_deref(),
                i + 1,
                o,
                kt,
                kh,
            );
            assert_eq!(r_bad.is_ok(), ret_bad == 0, "{what}: bad dims");
            check_conv2d(&what, &mut rng, &r, &h, 40);
        }
        // Missing names.
        let s = &lin[0];
        let mut bad = s.clone();
        bad.names[0] = Some("no_such_array".into());
        assert!(rust_linear(&arrays, &bad, s.nb_in, s.nb_out).is_err());
        assert_ne!(c_linear(model, &bad, s.nb_in, s.nb_out).1, 0);

        // GRUs: *_input / *_recurrent pairs.
        for (si, ri, hi) in &layers {
            let Some(prefix) = si.field.strip_suffix("_input") else {
                continue;
            };
            let Some((_, rr, hr)) = layers
                .iter()
                .find(|(s, _, _)| s.field == format!("{prefix}_recurrent"))
            else {
                continue;
            };
            if rr.nb_outputs != 3 * rr.nb_inputs || ri.nb_outputs != rr.nb_outputs {
                continue;
            }
            let what = format!("{file}:{prefix} gru");
            let mut rs = rand_vec(&mut rng, rr.nb_inputs, 0.9);
            let mut cs = rs.clone();
            for step in 0..50 {
                let input = rand_vec(&mut rng, ri.nb_inputs, 1.0);
                nnet::compute_generic_gru(ri, rr, &mut rs, &input);
                c::compute_generic_gru(hi, hr, &mut cs, &input);
                assert_bits_eq_f32(&format!("{what} step {step}"), &rs, &cs);
            }
        }
        // GLUs.
        for (s, r, h) in &layers {
            if !s.field.contains("glu") || r.nb_inputs != r.nb_outputs {
                continue;
            }
            let what = format!("{file}:{} glu", s.field);
            let input = rand_vec(&mut rng, r.nb_inputs, 1.0);
            let mut ro = vec![0f32; r.nb_outputs];
            let mut co = vec![0f32; r.nb_outputs];
            nnet::compute_glu(r, &mut ro, &input);
            h.compute_glu(&mut co, Some(&input));
            assert_bits_eq_f32(&what, &ro, &co);
            let mut ri = input.clone();
            let mut cin = input.clone();
            nnet::compute_glu_inplace(r, &mut ri);
            h.compute_glu(&mut cin, None);
            assert_bits_eq_f32(&format!("{what} in place"), &ri, &cin);
        }
        // Conv1d on the conv layers (input size: nb_in/ksize for ksize 2..4, else nb_in).
        for (s, r, h) in &layers {
            if !s.field.contains("conv") || r.nb_inputs > 1024 {
                continue;
            }
            let input_size = [3, 2, 4]
                .iter()
                .find(|&&k| r.nb_inputs % k == 0)
                .map_or(r.nb_inputs, |&k| r.nb_inputs / k);
            check_conv1d(
                &format!("{file}:{} conv1d", s.field),
                &mut rng,
                r,
                h,
                input_size,
                30,
            );
        }
    }
}

fn check_conv1d(
    what: &str,
    rng: &mut Rng,
    r: &nnet::LinearLayer,
    h: &c::Linear,
    input_size: usize,
    steps: usize,
) {
    let hist = r.nb_inputs - input_size;
    let mut rm = rand_vec(rng, hist, 0.5);
    let mut cm = rm.clone();
    for step in 0..steps {
        let act = ACTIVATIONS[step % ACTIVATIONS.len()];
        let input = rand_vec(rng, input_size, 1.0);
        let mut ro = vec![0f32; r.nb_outputs];
        let mut co = vec![0f32; r.nb_outputs];
        nnet::compute_generic_conv1d(r, &mut ro, &mut rm, &input, input_size, act);
        h.compute_generic_conv1d(&mut co, &mut cm, &input, input_size, act);
        assert_bits_eq_f32(&format!("{what} out step {step}"), &ro, &co);
        assert_bits_eq_f32(&format!("{what} mem step {step}"), &rm, &cm);
    }
}

fn check_conv1d_dilation(
    what: &str,
    rng: &mut Rng,
    r: &nnet::LinearLayer,
    h: &c::Linear,
    input_size: usize,
    dilation: usize,
    steps: usize,
) {
    let ksize = r.nb_inputs / input_size;
    let n = if dilation == 1 {
        r.nb_inputs - input_size
    } else {
        input_size * dilation * (ksize - 1)
    };
    let mut rm = rand_vec(rng, n, 0.5);
    let mut cm = rm.clone();
    for step in 0..steps {
        let act = ACTIVATIONS[step % ACTIVATIONS.len()];
        let input = rand_vec(rng, input_size, 1.0);
        let mut ro = vec![0f32; r.nb_outputs];
        let mut co = vec![0f32; r.nb_outputs];
        nnet::compute_generic_conv1d_dilation(
            r, &mut ro, &mut rm, &input, input_size, dilation, act,
        );
        h.compute_generic_conv1d_dilation(&mut co, &mut cm, &input, input_size, dilation, act);
        assert_bits_eq_f32(&format!("{what} out step {step}"), &ro, &co);
        assert_bits_eq_f32(&format!("{what} mem step {step}"), &rm, &cm);
    }
}

fn check_conv2d(what: &str, rng: &mut Rng, r: &nnet::Conv2dLayer, h: &c::Conv2d, steps: usize) {
    let height = [224usize, 17, 1, 64][rng.range_i32(0, 3) as usize];
    let hstride = height + rng.range_i32(0, 3) as usize;
    let ts = r.in_channels * (height + r.kheight - 1);
    if r.ktime * ts > nnet_arch::MAX_CONV2D_INPUTS {
        return;
    }
    let mut rm = rand_vec(rng, (r.ktime - 1) * ts, 0.5);
    let mut cm = rm.clone();
    let mut scratch = vec![0f32; nnet_arch::MAX_CONV2D_INPUTS];
    let out_len = (r.out_channels - 1) * hstride + height;
    for step in 0..steps {
        let act = ACTIVATIONS[step % ACTIVATIONS.len()];
        let input = rand_vec(rng, ts, 1.0);
        let fill = rng.f32_sym();
        let mut ro = vec![fill; out_len];
        let mut co = ro.clone();
        nnet::compute_conv2d(
            r,
            &mut ro,
            &mut rm,
            &input,
            height,
            hstride,
            act,
            &mut scratch,
        );
        h.compute_conv2d(&mut co, &mut cm, &input, height, hstride, act);
        assert_bits_eq_f32(&format!("{what} out step {step} h={height}"), &ro, &co);
        assert_bits_eq_f32(&format!("{what} mem step {step}"), &rm, &cm);
    }
}

// ---------------------------------------------------------------------------------------------
// Random layers

fn rand_sparse_idx(rng: &mut Rng, nb_in: usize, nb_out: usize) -> (Vec<i32>, usize) {
    let mut idx = Vec::new();
    let mut total = 0;
    for _ in 0..nb_out / 8 {
        let cols: Vec<i32> = (0..nb_in / 4)
            .filter(|_| !rng.next_u32().is_multiple_of(3))
            .map(|c| 4 * c as i32)
            .collect();
        idx.push(cols.len() as i32);
        idx.extend_from_slice(&cols);
        total += cols.len();
    }
    (idx, total)
}

#[derive(Debug)]
struct RandLayer {
    bias: Option<Vec<f32>>,
    subias: Option<Vec<f32>>,
    weights: Option<Vec<i8>>,
    float_weights: Option<Vec<f32>>,
    idx: Option<Vec<i32>>,
    diag: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
    nb_in: usize,
    nb_out: usize,
}

impl RandLayer {
    /// kind: 0 float dense, 1 float sparse, 2 int8 dense, 3 int8 sparse, 4 no weights,
    /// 5 int8 + float dense (the debug-float layout: float weights win).
    fn new(rng: &mut Rng, kind: u32, nb_in: usize, nb_out: usize, diag: bool) -> Self {
        let (idx, nw) = if kind == 1 || kind == 3 {
            let (i, t) = rand_sparse_idx(rng, nb_in, nb_out);
            (Some(i), 32 * t)
        } else {
            (None, nb_in * nb_out)
        };
        let int8 = kind == 2 || kind == 3 || kind == 5;
        let float_w = kind == 0 || kind == 1 || kind == 5;
        let scale_amp = 1.0 / (128.0 * (nb_in as f32).sqrt());
        Self {
            bias: (!rng.next_u32().is_multiple_of(4)).then(|| rand_vec(rng, nb_out, 0.5)),
            subias: rng
                .next_u32()
                .is_multiple_of(2)
                .then(|| rand_vec(rng, nb_out, 0.5)),
            weights: int8.then(|| (0..nw).map(|_| rng.next_u32() as i8).collect()),
            float_weights: float_w.then(|| rand_vec(rng, nw, 1.0 / (nb_in as f32).sqrt())),
            idx,
            diag: diag.then(|| rand_vec(rng, nb_out, 0.5)),
            scale: int8.then(|| {
                (0..nb_out)
                    .map(|_| scale_amp * (1.0 + rng.f32_sym().abs()))
                    .collect()
            }),
            nb_in,
            nb_out,
        }
    }

    fn rust(&self) -> nnet::LinearLayer {
        nnet::LinearLayer {
            bias: self.bias.clone(),
            subias: self.subias.clone(),
            weights: self.weights.clone(),
            float_weights: self.float_weights.clone(),
            weights_idx: self.idx.clone(),
            diag: self.diag.clone(),
            scale: self.scale.clone(),
            nb_inputs: self.nb_in,
            nb_outputs: self.nb_out,
        }
    }

    fn c(&self) -> c::Linear {
        c::Linear::new(
            c::LinearArrays {
                bias: self.bias.as_deref(),
                subias: self.subias.as_deref(),
                weights: self.weights.as_deref(),
                float_weights: self.float_weights.as_deref(),
                weights_idx: self.idx.as_deref(),
                diag: self.diag.as_deref(),
                scale: self.scale.as_deref(),
            },
            self.nb_in,
            self.nb_out,
        )
    }
}

#[test]
fn random_linear_layers() {
    let mut rng = Rng::new(0x005E_ED11);
    for trial in 0..1500 {
        let kind = (trial % 6) as u32;
        let nb_in = 4 * rng.range_i32(1, 96) as usize;
        let nb_out = match kind {
            1..=3 | 5 => 8 * rng.range_i32(1, 32) as usize,
            _ => rng.range_i32(1, 200) as usize,
        };
        let l = RandLayer::new(&mut rng, kind, nb_in, nb_out, false);
        let (r, h) = (l.rust(), l.c());
        check_linear_compute(
            &format!("random layer #{trial} kind {kind} {nb_in}x{nb_out}"),
            &mut rng,
            &r,
            &h,
            3,
        );
    }
    // GRU with and without diag (recurrent diag is the GRU-only path).
    for trial in 0..200 {
        let n = [8usize, 16, 24, 64, 96, 192][trial % 6];
        let nin = 4 * rng.range_i32(1, 64) as usize;
        let ki = [0u32, 1, 2, 3][rng.range_i32(0, 3) as usize];
        let kr = [0u32, 2][rng.range_i32(0, 1) as usize];
        let li = RandLayer::new(&mut rng, ki, nin, 3 * n, false);
        let lr = RandLayer::new(&mut rng, kr, n, 3 * n, trial % 2 == 0);
        let (ri, hi, rr, hr) = (li.rust(), li.c(), lr.rust(), lr.c());
        let mut rs = rand_vec(&mut rng, n, 1.0);
        let mut cs = rs.clone();
        for step in 0..20 {
            let input = rand_vec(&mut rng, nin, 1.0);
            nnet::compute_generic_gru(&ri, &rr, &mut rs, &input);
            c::compute_generic_gru(&hi, &hr, &mut cs, &input);
            assert_bits_eq_f32(&format!("gru #{trial} step {step}"), &rs, &cs);
        }
    }
    // GLU, conv1d, conv1d with dilation.
    for trial in 0..300 {
        let kind = [0u32, 2, 1, 3][trial % 4];
        let n = 8 * rng.range_i32(1, 40) as usize;
        let l = RandLayer::new(&mut rng, kind, n, n, false);
        let (r, h) = (l.rust(), l.c());
        let input = rand_vec(&mut rng, n, 1.2);
        let mut ro = vec![0f32; n];
        let mut co = vec![0f32; n];
        nnet::compute_glu(&r, &mut ro, &input);
        h.compute_glu(&mut co, Some(&input));
        assert_bits_eq_f32(&format!("glu #{trial}"), &ro, &co);
        let mut ri = input.clone();
        nnet::compute_glu_inplace(&r, &mut ri);
        let mut cin = input.clone();
        h.compute_glu(&mut cin, None);
        assert_bits_eq_f32(&format!("glu in place #{trial}"), &ri, &cin);

        let input_size = 4 * rng.range_i32(1, 32) as usize;
        let ksize = rng.range_i32(1, 4) as usize;
        let nb_out = 8 * rng.range_i32(1, 16) as usize;
        let lc = RandLayer::new(&mut rng, kind, input_size * ksize, nb_out, false);
        let (r, h) = (lc.rust(), lc.c());
        check_conv1d(&format!("conv1d #{trial}"), &mut rng, &r, &h, input_size, 5);
        if ksize >= 2 {
            for dilation in 1..=3 {
                check_conv1d_dilation(
                    &format!("conv1d dil {dilation} #{trial}"),
                    &mut rng,
                    &r,
                    &h,
                    input_size,
                    dilation,
                    5,
                );
            }
        }
    }
}

#[test]
fn random_conv2d_layers() {
    let mut rng = Rng::new(0xC0_2D);
    for trial in 0..400 {
        let in_ch = rng.range_i32(1, 4) as usize;
        let out_ch = rng.range_i32(1, 5) as usize;
        let (kt, kh) = if trial % 2 == 0 {
            (3, 3)
        } else {
            (rng.range_i32(1, 4) as usize, rng.range_i32(1, 5) as usize)
        };
        let w = rand_vec(&mut rng, in_ch * out_ch * kt * kh, 0.5);
        let bias = (trial % 3 != 0).then(|| rand_vec(&mut rng, out_ch, 0.3));
        let r = nnet::Conv2dLayer {
            bias: bias.clone(),
            float_weights: Some(w.clone()),
            in_channels: in_ch,
            out_channels: out_ch,
            ktime: kt,
            kheight: kh,
        };
        let h = c::Conv2d::new(bias.as_deref(), &w, in_ch, out_ch, kt, kh);
        check_conv2d(
            &format!("conv2d #{trial} {in_ch}->{out_ch} {kt}x{kh}"),
            &mut rng,
            &r,
            &h,
            4,
        );

        // The two kernels directly.
        let height = rng.range_i32(1, 60) as usize;
        let hstride = height + rng.range_i32(0, 2) as usize;
        let input = rand_vec(&mut rng, kt * in_ch * (height + kh - 1), 1.0);
        let mut ro = vec![0.5f32; (out_ch - 1) * hstride + height];
        let mut co = ro.clone();
        nnet_arch::conv2d_float(&mut ro, &w, in_ch, out_ch, kt, kh, &input, height, hstride);
        c::conv2d_float(&mut co, &w, in_ch, out_ch, kt, kh, &input, height, hstride);
        assert_bits_eq_f32(&format!("conv2d_float #{trial}"), &ro, &co);
        if kt == 3 && kh == 3 {
            nnet_arch::conv2d_3x3_float(&mut ro, &w, in_ch, out_ch, &input, height, hstride);
            c::conv2d_3x3_float(&mut co, &w, in_ch, out_ch, &input, height, hstride);
            assert_bits_eq_f32(&format!("conv2d_3x3_float #{trial}"), &ro, &co);
        }
    }
}

#[test]
fn vec_kernels() {
    let mut rng = Rng::new(0x7EC);
    for trial in 0..1000 {
        let rows = match trial % 3 {
            0 => 16 * rng.range_i32(1, 12) as usize,
            1 => 8 * rng.range_i32(1, 25) as usize,
            _ => rng.range_i32(1, 150) as usize,
        };
        let cols = rng.range_i32(0, 200) as usize;
        let stride = rows + rng.range_i32(0, 5) as usize;
        let w = rand_vec(&mut rng, cols * stride + rows, 1.0);
        let x = rand_vec(&mut rng, cols, 1.0);
        let mut ro = vec![7.0f32; rows];
        let mut co = ro.clone();
        dv::sgemv(&mut ro, &w, rows, cols, stride, &x);
        c::sgemv(&mut co, &w, rows, cols, stride, &x);
        assert_bits_eq_f32(&format!("sgemv #{trial} {rows}x{cols}"), &ro, &co);

        let rows = 8 * rng.range_i32(1, 20) as usize;
        let cols = 4 * rng.range_i32(1, 100) as usize;
        let amp = [1.0f32, 3.0, 0.01][trial % 3];
        let x = rand_vec(&mut rng, cols, amp);
        let (idx, blocks) = rand_sparse_idx(&mut rng, cols, rows);
        let wf = rand_vec(&mut rng, 32 * blocks, 1.0);
        let wi: Vec<i8> = (0..rows * cols).map(|_| rng.next_u32() as i8).collect();
        let scale = rand_vec(&mut rng, rows, 0.01);
        let mut ro = vec![0f32; rows];
        let mut co = vec![0f32; rows];
        dv::sparse_sgemv8x4(&mut ro, &wf, &idx, rows, &x);
        c::sparse_sgemv8x4(&mut co, &wf, &idx, rows, &x);
        assert_bits_eq_f32(&format!("sparse_sgemv8x4 #{trial}"), &ro, &co);
        dv::cgemv8x4(&mut ro, &wi, &scale, rows, cols, &x);
        c::cgemv8x4(&mut co, &wi, &scale, rows, cols, &x);
        assert_bits_eq_f32(&format!("cgemv8x4 #{trial}"), &ro, &co);
        dv::sparse_cgemv8x4(&mut ro, &wi, &idx, &scale, rows, cols, &x);
        c::sparse_cgemv8x4(&mut co, &wi, &idx, &scale, rows, cols, &x);
        assert_bits_eq_f32(&format!("sparse_cgemv8x4 #{trial}"), &ro, &co);
    }
}

/// Inputs for the scalar approximations: dense sweep, random, and special values.
fn scalar_inputs(rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (-4000..=4000).map(|i| i as f32 / 200.0).collect();
    v.extend((0..20000).map(|_| rng.f32_sym() * 10f32.powi(rng.range_i32(-8, 4))));
    v.extend([
        0.0,
        -0.0,
        1e-45,
        -1e-45,
        1e-38,
        f32::MIN_POSITIVE,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -36.0,
        -35.0,
        -34.7,
        88.0,
        89.0,
        200.0,
        -200.0,
    ]);
    v
}

#[test]
fn activations_and_scalar_approximations() {
    let mut rng = Rng::new(0xAC7);
    let xs = scalar_inputs(&mut rng);
    type ScalarFn = fn(f32) -> f32;
    let rust_fns: [(i32, ScalarFn); 8] = [
        (0, dv::tanh_approx),
        (1, dv::sigmoid_approx),
        (2, dv::lpcnet_exp),
        (3, dv::lpcnet_exp2),
        (4, common::log2_approx),
        (5, common::log_approx),
        (6, common::ulaw2lin),
        (7, nnet_arch::relu),
    ];
    for (which, f) in rust_fns {
        // lpcnet_exp2 of huge values relies on an out-of-range float->int conversion in C;
        // keep its inputs within the int range.
        let inputs: Vec<f32> = if which == 2 || which == 3 {
            xs.iter().copied().filter(|x| x.abs() < 1e9).collect()
        } else {
            xs.clone()
        };
        let r: Vec<f32> = inputs.iter().map(|&x| f(x)).collect();
        assert_bits_eq_f32(
            &format!("scalar fn {which}"),
            &r,
            &c::scalar(which, &inputs),
        );
    }
    let lin: Vec<f32> = (0..30000)
        .map(|i| (i as f32 - 15000.0) * 2.5 + rng.f32_sym())
        .collect();
    let r: Vec<i32> = lin.iter().map(|&x| common::lin2ulaw(x)).collect();
    assert_slice_eq("lin2ulaw", &r, &c::lin2ulaw(&lin));
    let u: Vec<f32> = (0..=255).map(|i| i as f32).collect();
    let r: Vec<f32> = u.iter().map(|&x| common::ulaw2lin(x)).collect();
    assert_bits_eq_f32("ulaw2lin", &r, &c::scalar(6, &u));

    // compute_activation (out of place and in place) on every activation.
    let finite: Vec<f32> = xs.iter().copied().filter(|x| x.abs() < 1e9).collect();
    for chunk in finite.chunks(4096) {
        let n = chunk.len();
        for act in ACTIVATIONS {
            let mut ro = vec![0f32; n];
            let mut co = vec![0f32; n];
            nnet_arch::compute_activation(&mut ro, chunk, n, act);
            c::compute_activation(&mut co, Some(chunk), n, act);
            assert_bits_eq_f32(&format!("activation {act}"), &ro, &co);
            let mut ri = chunk.to_vec();
            let mut cin = chunk.to_vec();
            nnet::compute_activation_inplace(&mut ri, n, act);
            c::compute_activation(&mut cin, None, n, act);
            assert_bits_eq_f32(&format!("activation {act} in place"), &ri, &cin);
        }
        let mut ro = vec![0f32; n];
        let mut co = vec![0f32; n];
        nnet_arch::vec_swish(&mut ro, chunk, n);
        c::vec_swish(&mut co, chunk, n);
        assert_bits_eq_f32("vec_swish", &ro, &co);
    }
}

#[test]
fn kiss99_streams() {
    let mut rng = Rng::new(0x1155);
    for len in 0..40 {
        let mut seed = vec![0u8; len];
        rng.fill_bytes(&mut seed);
        let (cst, cout) = c::kiss99(&seed, 2000);
        let mut k = kiss99::Kiss99Ctx::default();
        k.kiss99_srand(&seed);
        assert_eq!([k.z, k.w, k.jsr, k.jcong], cst, "srand len {len}");
        let r: Vec<u32> = (0..2000).map(|_| kiss99::kiss99_rand(&mut k)).collect();
        assert_slice_eq(&format!("kiss99 len {len}"), &r, &cout);
    }
    // Seeds hitting the short-cycle fix-ups (z/w/jsr forced to 0 or the bad values).
    for seed in [
        [0x75u8, 0x88, 0x5f, 0x0],
        [0xff, 0xff, 0xff, 0xff],
        [0, 0, 0, 0],
    ] {
        let (cst, _) = c::kiss99(&seed[..3], 1);
        let mut k = kiss99::Kiss99Ctx::default();
        kiss99::kiss99_srand(&mut k, &seed[..3]);
        assert_eq!([k.z, k.w, k.jsr, k.jcong], cst);
    }
}

// ---------------------------------------------------------------------------------------------
// burg / freq

/// 16 kHz test signals at int16 scale.
fn signal16(kind: usize, n: usize, seed: u64) -> Vec<f32> {
    let s = match kind % 6 {
        0 => signals::speech_like(n, 1, 16000, seed),
        1 => signals::music_like(n, 1, 16000, seed),
        2 => signals::noise(n, 1, 0.3, seed),
        3 => signals::noise(n, 1, 1e-4, seed),
        4 => vec![0.0; n],
        _ => (0..n)
            .map(|i| if (i / 50) % 2 == 0 { 0.9 } else { -0.9 })
            .collect(),
    };
    s.iter().map(|v| v * 32768.0).collect()
}

#[test]
fn burg_analysis() {
    let mut rng = Rng::new(0xB0_46);
    for trial in 0..3000 {
        let n = rng.range_i32(0, 300) as usize;
        let x = rand_vec(&mut rng, n, 1000.0);
        let y = rand_vec(&mut rng, n, 1000.0);
        assert_eq!(
            burg::silk_energy_flp(&x, n).to_bits(),
            c::silk_energy_flp(&x).to_bits()
        );
        assert_eq!(
            burg::silk_inner_product_flp(&x, &y, n).to_bits(),
            c::silk_inner_product_flp(&x, &y).to_bits()
        );

        let d = rng.range_i32(1, 16) as usize;
        let nb_subfr = rng.range_i32(1, 4) as usize;
        let subfr = rng.range_i32(d as i32 + 1, (384 / nb_subfr) as i32) as usize;
        let sig = signal16(trial, subfr * nb_subfr, rng.next_u64());
        let min_inv_gain = [1e-3f32, 1e-2, 0.1, 0.5, 1e-4][trial % 5];
        let mut ra = vec![0f32; d];
        let mut ca = vec![0f32; d];
        let re = burg::silk_burg_analysis(&mut ra, &sig, min_inv_gain, subfr, nb_subfr, d);
        let ce = c::silk_burg_analysis(&mut ca, &sig, min_inv_gain, subfr, nb_subfr, d);
        assert_eq!(re.to_bits(), ce.to_bits(), "burg energy #{trial}");
        assert_bits_eq_f32(&format!("burg A #{trial}"), &ra, &ca);
    }
}

#[test]
fn freq_helpers() {
    let mut rng = Rng::new(0xF4E0);
    for trial in 0..1500 {
        let x = signal16(trial, 320, rng.next_u64());
        // apply_window / forward / inverse transform.
        let mut rw = x.clone();
        let mut cw = x.clone();
        freq::apply_window(&mut rw);
        c::apply_window(&mut cw);
        assert_bits_eq_f32("apply_window", &rw, &cw);
        let mut rx = vec![KissFftCpx::default(); freq::FREQ_SIZE];
        freq::forward_transform(&mut rx, &rw);
        let mut cx = vec![0f32; 2 * freq::FREQ_SIZE];
        c::forward_transform(&mut cx, &cw);
        assert_bits_eq_f32("forward_transform", &cpx_to_flat(&rx), &cx);
        let mut ri = vec![0f32; freq::WINDOW_SIZE];
        let mut cinv = vec![0f32; freq::WINDOW_SIZE];
        freq::inverse_transform(&mut ri, &rx);
        c::inverse_transform(&mut cinv, &cx);
        assert_bits_eq_f32("inverse_transform", &ri, &cinv);
        // Band energies.
        let mut re = [0f32; freq::NB_BANDS];
        freq::lpcn_compute_band_energy(&mut re, &rx);
        assert_bits_eq_f32("band_energy", &re, &c::lpcn_compute_band_energy(&cx));
        freq::compute_band_energy_inverse(&mut re, &rx);
        assert_bits_eq_f32(
            "band_energy_inverse",
            &re,
            &c::compute_band_energy_inverse(&cx),
        );
        // interp_band_gain (the last bin is never written by the loop).
        let band_e = rand_vec(&mut rng, freq::NB_BANDS, 10.0);
        let mut rg = vec![0f32; freq::FREQ_SIZE];
        let mut cg = vec![0f32; freq::FREQ_SIZE];
        freq::interp_band_gain(&mut rg, &band_e);
        c::interp_band_gain(&mut cg, &band_e);
        assert_bits_eq_f32("interp_band_gain", &rg[..160], &cg[..160]);
        // dct / idct.
        let v = rand_vec(&mut rng, freq::NB_BANDS, 5.0);
        let mut ro = [0f32; freq::NB_BANDS];
        let mut co = [0f32; freq::NB_BANDS];
        freq::dct(&mut ro, &v);
        c::dct(&mut co, &v);
        assert_bits_eq_f32("dct", &ro, &co);
        freq::idct(&mut ro, &v);
        c::idct(&mut co, &v);
        assert_bits_eq_f32("idct", &ro, &co);
        // lpcn_lpc on an autocorrelation.
        let mut ac = [0f32; 17];
        for (k, a) in ac.iter_mut().enumerate() {
            *a = x[..320 - k].iter().zip(&x[k..]).map(|(a, b)| a * b).sum();
        }
        if trial % 7 == 0 {
            ac = [0.0; 17];
        }
        let (mut rl, mut rr, mut cl, mut cr) = ([0f32; 16], [0f32; 16], [0f32; 16], [0f32; 16]);
        let e1 = freq::lpcn_lpc(&mut rl, &mut rr, &ac, 16);
        let e2 = c::lpcn_lpc(&mut cl, &mut cr, &ac, 16);
        assert_eq!(e1.to_bits(), e2.to_bits(), "lpcn_lpc error #{trial}");
        assert_bits_eq_f32("lpcn_lpc lpc", &rl, &cl);
        assert_bits_eq_f32("lpcn_lpc rc", &rr, &cr);
        // lpc_from_bands / lpc_from_cepstrum / lpc_weighting.
        let ex: Vec<f32> = re.iter().map(|e| e.abs() + 1e-3).collect();
        let (mut rl, mut cl) = ([0f32; 16], [0f32; 16]);
        let e1 = freq::lpc_from_bands(&mut rl, &ex);
        let e2 = c::lpc_from_bands(&mut cl, &ex);
        assert_eq!(e1.to_bits(), e2.to_bits(), "lpc_from_bands #{trial}");
        assert_bits_eq_f32("lpc_from_bands", &rl, &cl);
        let ceps = rand_vec(&mut rng, freq::NB_BANDS, 3.0);
        let e1 = freq::lpc_from_cepstrum(&mut rl, &ceps);
        let e2 = c::lpc_from_cepstrum(&mut cl, &ceps);
        assert_eq!(e1.to_bits(), e2.to_bits(), "lpc_from_cepstrum #{trial}");
        assert_bits_eq_f32("lpc_from_cepstrum", &rl, &cl);
        let gamma = 0.8 + 0.2 * rng.f32_sym();
        freq::lpc_weighting(&mut rl, gamma);
        c::lpc_weighting(&mut cl, gamma);
        assert_bits_eq_f32("lpc_weighting", &rl, &cl);
        // Burg cepstrum.
        let mut rc = [0f32; 2 * freq::NB_BANDS];
        let mut cc = [0f32; 2 * freq::NB_BANDS];
        freq::burg_cepstral_analysis(&mut rc, &x[..160]);
        c::burg_cepstral_analysis(&mut cc, &x[..160]);
        assert_bits_eq_f32(&format!("burg_cepstral_analysis #{trial}"), &rc, &cc);
        let len = rng.range_i32(20, 160) as usize;
        let order = rng.range_i32(1, 16) as usize;
        freq::compute_burg_cepstrum(&x, &mut rc, len, order);
        c::compute_burg_cepstrum(&x, &mut cc, len, order);
        assert_bits_eq_f32(
            &format!("compute_burg_cepstrum #{trial}"),
            &rc[..18],
            &cc[..18],
        );
    }
}

// ---------------------------------------------------------------------------------------------
// pitchdnn / lpcnet_enc

fn pitch_blob() -> Vec<u8> {
    c::write_blob(c::MODEL_PITCHDNN)
}

fn flatten_pitchdnn(st: &pitchdnn::PitchDnnState) -> Vec<f32> {
    let mut v = st.gru_state.to_vec();
    v.extend_from_slice(&st.xcorr_mem1);
    v.extend_from_slice(&st.xcorr_mem2);
    v
}

fn flatten_enc_state(st: &le::LpcnetEncState) -> Vec<f32> {
    let mut v = Vec::with_capacity(c::LPCNET_ENC_STATE_LEN);
    v.extend_from_slice(&st.analysis_mem);
    v.push(st.mem_preemph);
    v.extend(cpx_to_flat(&st.prev_if));
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
    v.extend(flatten_pitchdnn(&st.pitchdnn));
    v
}

#[test]
fn pitchdnn_streams() {
    let blob = pitch_blob();
    let mut rng = Rng::new(0x9D);
    for stream in 0..8 {
        let mut r = pitchdnn::PitchDnnState::new();
        r.load_model(&blob).expect("pitch model");
        let mut h = c::PitchDnn::new();
        let mut if_f = rand_vec(&mut rng, le::PITCH_IF_FEATURES, 1.0);
        let mut xc = rand_vec(&mut rng, pitchdnn::NB_XCORR_FEATURES, 1.0);
        for frame in 0..300 {
            // Slowly varying features (state evolves like on real speech).
            for v in if_f.iter_mut().chain(xc.iter_mut()) {
                *v = (*v * 0.8 + 0.2 * rng.f32_sym()).clamp(-1.0, 1.0);
            }
            if stream == 3 && frame % 10 == 0 {
                xc.iter_mut().for_each(|v| *v = 0.0);
            }
            let rp = pitchdnn::compute_pitchdnn(&mut r, &if_f, &xc);
            let cp = h.compute(&if_f, &xc);
            assert_eq!(
                rp.to_bits(),
                cp.to_bits(),
                "pitch stream {stream} frame {frame}"
            );
            assert_bits_eq_f32(
                &format!("pitchdnn state {stream}/{frame}"),
                &flatten_pitchdnn(&r),
                &h.state(),
            );
        }
    }
    // A corrupted blob is rejected (C would crash on it).
    let mut r = pitchdnn::PitchDnnState::new();
    assert!(r.load_model(&blob[..blob.len() - 1]).is_err());
    assert!(r.load_model(&c::write_blob(c::MODEL_PLC)).is_err());
}

#[test]
fn lpcnet_feature_streams() {
    let blob = pitch_blob();
    let mut rng = Rng::new(0x1EC);
    for stream in 0..12 {
        let frames = 500;
        let sig = signal16(stream, frames * 160, rng.next_u64());
        let mut r = le::LpcnetEncState::new();
        r.load_model(&blob).expect("pitch model");
        let mut h = c::LpcnetEnc::new();
        assert_bits_eq_f32("initial state", &flatten_enc_state(&r), &h.state());
        let use_int = stream % 2 == 0;
        for f in 0..frames {
            let frame = &sig[f * 160..(f + 1) * 160];
            let mut rf = [0f32; le::NB_TOTAL_FEATURES];
            let cf = if use_int {
                let pcm: Vec<i16> = frame
                    .iter()
                    .map(|&v| v.clamp(-32768.0, 32767.0) as i16)
                    .collect();
                le::lpcnet_compute_single_frame_features(&mut r, &pcm, &mut rf);
                h.features(&pcm)
            } else {
                le::lpcnet_compute_single_frame_features_float(&mut r, frame, &mut rf);
                h.features_float(frame)
            };
            assert_bits_eq_f32(&format!("features stream {stream} frame {f}"), &rf, &cf);
            assert_bits_eq_f32(
                &format!("enc state stream {stream} frame {f}"),
                &flatten_enc_state(&r),
                &h.state(),
            );
        }
        // compute_frame_features directly and re-init.
        let x = signal16(stream + 1, 160, 7);
        le::compute_frame_features(&mut r, &x);
        h.compute_frame_features(&x);
        assert_bits_eq_f32("compute_frame_features", &flatten_enc_state(&r), &h.state());
        r.lpcnet_encoder_init();
        let fresh = c::LpcnetEnc::new();
        assert_bits_eq_f32("re-init", &flatten_enc_state(&r), &fresh.state());
    }
}

#[test]
fn lpcnet_enc_helpers() {
    let blob = pitch_blob();
    let mut rng = Rng::new(0x4E1);
    let mut r = le::LpcnetEncState::new();
    r.load_model(&blob).expect("pitch model");
    let mut h = c::LpcnetEnc::new();
    for trial in 0..300 {
        let x = signal16(trial, 160, rng.next_u64());
        let mut rx = vec![KissFftCpx::default(); freq::FREQ_SIZE];
        let mut rex = [0f32; freq::NB_BANDS];
        le::frame_analysis(&mut r, &mut rx, &mut rex, &x);
        let (cx, cex) = h.frame_analysis(&x);
        assert_bits_eq_f32("frame_analysis X", &cpx_to_flat(&rx), &cx);
        assert_bits_eq_f32("frame_analysis Ex", &rex, &cex);
        assert_bits_eq_f32("frame_analysis state", &flatten_enc_state(&r), &h.state());

        let b = [rng.f32_sym(), rng.f32_sym()];
        let a = [rng.f32_sym() * 1.5, rng.f32_sym() * 0.7];
        let (mut rm, mut cm) = ([0.1f32, -0.2], [0.1f32, -0.2]);
        let mut ry = vec![0f32; 160];
        let mut cy = vec![0f32; 160];
        le::biquad(&mut ry, &mut rm, &x, &b, &a, 160);
        c::biquad(&mut cy, &mut cm, Some(&x), &b, &a, 160);
        assert_bits_eq_f32("biquad", &ry, &cy);
        assert_bits_eq_f32("biquad mem", &rm, &cm);
        le::biquad_inplace(&mut ry, &mut rm, &b, &a, 160);
        c::biquad(&mut cy, &mut cm, None, &b, &a, 160);
        assert_bits_eq_f32("biquad in place", &ry, &cy);
        assert_bits_eq_f32("biquad in place mem", &rm, &cm);

        let coef = 0.85 + 0.1 * rng.f32_sym();
        let mut rmem = rng.f32_sym();
        let mut cmem = rmem;
        le::preemphasis(&mut ry, &mut rmem, &x, coef, 160);
        c::preemphasis(&mut cy, &mut cmem, Some(&x), coef, 160);
        assert_bits_eq_f32("preemphasis", &ry, &cy);
        assert_eq!(rmem.to_bits(), cmem.to_bits());
        le::preemphasis_inplace(&mut ry, &mut rmem, coef, 160);
        c::preemphasis(&mut cy, &mut cmem, None, coef, 160);
        assert_bits_eq_f32("preemphasis in place", &ry, &cy);
        assert_eq!(rmem.to_bits(), cmem.to_bits());
    }
}

// ---------------------------------------------------------------------------------------------
// nndsp (osce)

#[cfg(feature = "osce")]
mod nndsp_tests {
    use super::*;
    use opusorus::dnn::nndsp as nd;

    fn flatten_conv(s: &nd::AdaConvState) -> Vec<f32> {
        let mut v = s.history.to_vec();
        v.extend_from_slice(&s.last_kernel);
        v.push(s.last_gain);
        v
    }

    fn flatten_comb(s: &nd::AdaCombState) -> (Vec<f32>, i32) {
        let mut v = s.history.to_vec();
        v.extend_from_slice(&s.last_kernel);
        v.push(s.last_global_gain);
        (v, s.last_pitch_lag)
    }

    fn flatten_shape(s: &nd::AdaShapeState) -> Vec<f32> {
        let mut v = s.conv_alpha1f_state.to_vec();
        v.extend_from_slice(&s.conv_alpha1t_state);
        v.extend_from_slice(&s.conv_alpha2_state);
        v.push(s.interpolate_state[0]);
        v
    }

    /// Named layers of an OSCE model: (Rust from blob, C from the compiled-in table).
    struct Model {
        model: i32,
        specs: Vec<LinearSpec>,
        blob: Vec<u8>,
    }

    impl Model {
        fn new(model: i32, file: &str) -> Self {
            Self {
                model,
                specs: init_specs(file).0,
                blob: c::write_blob(model),
            }
        }
        fn layer(&self, field: &str) -> (nnet::LinearLayer, c::Linear) {
            let s = self
                .specs
                .iter()
                .find(|s| s.field == field)
                .unwrap_or_else(|| panic!("{field}"));
            let arrays = pw::parse_weights(&self.blob).unwrap();
            let r = rust_linear(&arrays, s, s.nb_in, s.nb_out).unwrap();
            let (h, ret) = c_linear(self.model, s, s.nb_in, s.nb_out);
            assert_eq!(ret, 0);
            (r, h)
        }
    }

    #[test]
    fn overlap_window_and_helpers() {
        for n in [1usize, 2, 3, 40, 80, 120, 160, 240] {
            let mut w = vec![0f32; n];
            nd::compute_overlap_window(&mut w, n);
            assert_bits_eq_f32(
                &format!("overlap window {n}"),
                &w,
                &c::compute_overlap_window(n),
            );
        }
        let mut rng = Rng::new(0x5CA1E);
        for trial in 0..200 {
            let in_ch = rng.range_i32(1, 3) as usize;
            let out_ch = rng.range_i32(1, 3) as usize;
            let ks = rng.range_i32(1, 32) as usize;
            let mut rk = rand_vec(&mut rng, in_ch * out_ch * ks, 2.0);
            if trial % 11 == 0 {
                rk.iter_mut().for_each(|v| *v = 0.0);
            }
            let mut ck = rk.clone();
            let mut gains = rand_vec(&mut rng, out_ch, 1.0);
            let mut cg = gains.clone();
            let (a, b) = (rng.f32_sym() * 2.0, rng.f32_sym());
            nd::transform_gains(&mut gains, out_ch, a, b);
            c::transform_gains(&mut cg, a, b);
            assert_bits_eq_f32("transform_gains", &gains, &cg);
            nd::scale_kernel(&mut rk, in_ch, out_ch, ks, &gains);
            c::scale_kernel(&mut ck, in_ch, out_ch, ks, &cg);
            assert_bits_eq_f32(&format!("scale_kernel #{trial}"), &rk, &ck);
        }
    }

    /// Speech-like frames scaled like the OSCE signal path (+-1).
    fn frames(n: usize, seed: u64) -> Vec<f32> {
        signals::speech_like(n, 1, 16000, seed)
    }

    #[test]
    fn adaconv_real_and_random() {
        let mut rng = Rng::new(0x000A_DAC0);
        // (model, kernel, gain, feature_dim, in_ch, out_ch)
        let lace = Model::new(c::MODEL_LACE, "lace_data.c");
        let nolace = Model::new(c::MODEL_NOLACE, "nolace_data.c");
        let cases: Vec<(&Model, &str, &str, usize, usize, usize)> = vec![
            (&lace, "lace_af1_kernel", "lace_af1_gain", 128, 1, 1),
            (&nolace, "nolace_af1_kernel", "nolace_af1_gain", 160, 1, 2),
            (&nolace, "nolace_af2_kernel", "nolace_af2_gain", 160, 2, 2),
            (&nolace, "nolace_af3_kernel", "nolace_af3_gain", 160, 2, 2),
            (&nolace, "nolace_af4_kernel", "nolace_af4_gain", 160, 2, 1),
        ];
        for (ci_, (m, k, g, fd, in_ch, out_ch)) in cases.iter().enumerate() {
            let (rk, hk) = m.layer(k);
            let (rg, hg) = m.layer(g);
            let p = c::AdaConvParams {
                feature_dim: *fd,
                frame_size: 80,
                overlap_size: 40,
                in_channels: *in_ch,
                out_channels: *out_ch,
                kernel_size: 16,
                left_padding: 15,
                filter_gain_a: 1.381551,
                filter_gain_b: 0.0,
                shape_gain: 1.0,
            };
            run_adaconv(
                &format!("adaconv {k}"),
                &mut rng,
                &rk,
                &hk,
                &rg,
                &hg,
                &p,
                200 + ci_,
            );
        }
        // Random layers and shapes.
        for trial in 0..200 {
            let in_ch = rng.range_i32(1, 3) as usize;
            let out_ch = rng.range_i32(1, 3) as usize;
            let ks = rng.range_i32(2, 16) as usize;
            let frame_size = 4 * rng.range_i32(ks as i32 / 4 + 1, 60) as usize;
            // overlap >= 1: celt_pitch_xcorr requires max_pitch > 0.
            let overlap = rng.range_i32(1, (frame_size as i32).min(120)) as usize;
            let fd = 4 * rng.range_i32(1, 40) as usize;
            let kind = if (in_ch * out_ch * ks).is_multiple_of(8) {
                [0u32, 2][trial % 2]
            } else {
                0
            };
            let lk = RandLayer::new(&mut rng, kind, fd, in_ch * out_ch * ks, false);
            let lg = RandLayer::new(&mut rng, 0, fd, out_ch, false);
            let p = c::AdaConvParams {
                feature_dim: fd,
                frame_size,
                overlap_size: overlap,
                in_channels: in_ch,
                out_channels: out_ch,
                kernel_size: ks,
                left_padding: ks - 1,
                filter_gain_a: rng.f32_sym() * 2.0,
                filter_gain_b: rng.f32_sym() * 0.5,
                shape_gain: 1.0,
            };
            if 1 + (in_ch - 1) * (frame_size + ks) + frame_size + 35 > 3 * 272 {
                continue;
            }
            run_adaconv(
                &format!("adaconv random #{trial}"),
                &mut rng,
                &lk.rust(),
                &lk.c(),
                &lg.rust(),
                &lg.c(),
                &p,
                12,
            );
        }
    }

    #[expect(clippy::too_many_arguments, reason = "test helper mirrors the C call")]
    fn run_adaconv(
        what: &str,
        rng: &mut Rng,
        rk: &nnet::LinearLayer,
        hk: &c::Linear,
        rg: &nnet::LinearLayer,
        hg: &c::Linear,
        p: &c::AdaConvParams,
        frames_n: usize,
    ) {
        let mut window = vec![0f32; p.overlap_size];
        nd::compute_overlap_window(&mut window, p.overlap_size);
        let mut rs = nd::AdaConvState::default();
        let mut hs = c::AdaConv::new();
        let n = p.frame_size;
        let sig = frames(frames_n * n * p.in_channels, rng.next_u64());
        for f in 0..frames_n {
            let x = &sig[f * n * p.in_channels..(f + 1) * n * p.in_channels];
            let features = rand_vec(rng, p.feature_dim, 1.0);
            let inplace = f % 2 == 1 && p.in_channels == p.out_channels;
            let mut ro = vec![0f32; p.out_channels.max(p.in_channels) * n];
            let mut co = ro.clone();
            if inplace {
                ro[..x.len()].copy_from_slice(x);
                co[..x.len()].copy_from_slice(x);
                nd::adaconv_process_frame(
                    &mut rs,
                    &mut ro,
                    None,
                    &features,
                    rk,
                    rg,
                    p.feature_dim,
                    n,
                    p.overlap_size,
                    p.in_channels,
                    p.out_channels,
                    p.kernel_size,
                    p.left_padding,
                    p.filter_gain_a,
                    p.filter_gain_b,
                    p.shape_gain,
                    &window,
                );
                hs.process(&mut co, None, &features, hk, hg, p, &window);
            } else {
                nd::adaconv_process_frame(
                    &mut rs,
                    &mut ro,
                    Some(x),
                    &features,
                    rk,
                    rg,
                    p.feature_dim,
                    n,
                    p.overlap_size,
                    p.in_channels,
                    p.out_channels,
                    p.kernel_size,
                    p.left_padding,
                    p.filter_gain_a,
                    p.filter_gain_b,
                    p.shape_gain,
                    &window,
                );
                hs.process(&mut co, Some(x), &features, hk, hg, p, &window);
            }
            assert_bits_eq_f32(&format!("{what} out frame {f}"), &ro, &co);
            assert_bits_eq_f32(
                &format!("{what} state frame {f}"),
                &flatten_conv(&rs),
                &hs.state(),
            );
        }
    }

    #[test]
    fn adacomb_real_and_random() {
        let mut rng = Rng::new(0xC0_4B);
        let lace = Model::new(c::MODEL_LACE, "lace_data.c");
        let nolace = Model::new(c::MODEL_NOLACE, "nolace_data.c");
        for (m, prefix, fd) in [
            (&lace, "lace_cf1", 128usize),
            (&lace, "lace_cf2", 128),
            (&nolace, "nolace_cf1", 160),
            (&nolace, "nolace_cf2", 160),
        ] {
            let (rk, hk) = m.layer(&format!("{prefix}_kernel"));
            let (rg, hg) = m.layer(&format!("{prefix}_gain"));
            let (rgg, hgg) = m.layer(&format!("{prefix}_global_gain"));
            let p = c::AdaCombParams {
                feature_dim: fd,
                frame_size: 80,
                overlap_size: 40,
                kernel_size: 16,
                left_padding: 8,
                filter_gain_a: 0.690776,
                filter_gain_b: 0.0,
                log_gain_limit: 1.151293,
            };
            run_adacomb(
                prefix,
                &mut rng,
                [&rk, &rg, &rgg],
                [&hk, &hg, &hgg],
                &p,
                300,
            );
        }
        for trial in 0..200 {
            let ks = rng.range_i32(1, 16) as usize;
            let frame_size = rng.range_i32(ks as i32, 80) as usize;
            let overlap = rng.range_i32(1, (frame_size as i32).min(40)) as usize;
            let fd = 4 * rng.range_i32(1, 40) as usize;
            let kind = [0u32, 2][trial % 2];
            let nb_k = if kind == 2 { 8 * ks.div_ceil(8) } else { ks };
            if nb_k != ks {
                continue;
            }
            let lk = RandLayer::new(&mut rng, kind, fd, ks, false);
            let lg = RandLayer::new(&mut rng, 0, fd, 1, false);
            let lgg = RandLayer::new(&mut rng, 0, fd, 1, false);
            let p = c::AdaCombParams {
                feature_dim: fd,
                frame_size,
                overlap_size: overlap,
                kernel_size: ks,
                left_padding: rng.range_i32(0, ks as i32 - 1) as usize,
                filter_gain_a: rng.f32_sym(),
                filter_gain_b: rng.f32_sym() * 0.3,
                log_gain_limit: rng.f32_sym() * 2.0,
            };
            run_adacomb(
                &format!("adacomb random #{trial}"),
                &mut rng,
                [&lk.rust(), &lg.rust(), &lgg.rust()],
                [&lk.c(), &lg.c(), &lgg.c()],
                &p,
                10,
            );
        }
    }

    fn run_adacomb(
        what: &str,
        rng: &mut Rng,
        r: [&nnet::LinearLayer; 3],
        h: [&c::Linear; 3],
        p: &c::AdaCombParams,
        frames_n: usize,
    ) {
        let mut window = vec![0f32; p.overlap_size];
        nd::compute_overlap_window(&mut window, p.overlap_size);
        let mut rs = nd::AdaCombState::default();
        let mut hs = c::AdaComb::new();
        let n = p.frame_size;
        let sig = frames(frames_n * n, rng.next_u64());
        let mut lag = rng.range_i32(32, 300) as usize;
        for f in 0..frames_n {
            let x = &sig[f * n..(f + 1) * n];
            let features = rand_vec(rng, p.feature_dim, 1.0);
            if f % 4 == 0 {
                // Real OSCE lags are >= 32; small lags would make C read past its buffer.
                lag =
                    rng.range_i32(32, 300 + p.kernel_size as i32 - p.left_padding as i32) as usize;
            }
            let mut ro = vec![0f32; n];
            let mut co = vec![0f32; n];
            if f % 2 == 1 {
                ro.copy_from_slice(x);
                co.copy_from_slice(x);
                nd::adacomb_process_frame(
                    &mut rs,
                    &mut ro,
                    None,
                    &features,
                    r[0],
                    r[1],
                    r[2],
                    lag as i32,
                    p.feature_dim,
                    n,
                    p.overlap_size,
                    p.kernel_size,
                    p.left_padding,
                    p.filter_gain_a,
                    p.filter_gain_b,
                    p.log_gain_limit,
                    &window,
                );
                hs.process(&mut co, None, &features, h[0], h[1], h[2], lag, p, &window);
            } else {
                nd::adacomb_process_frame(
                    &mut rs,
                    &mut ro,
                    Some(x),
                    &features,
                    r[0],
                    r[1],
                    r[2],
                    lag as i32,
                    p.feature_dim,
                    n,
                    p.overlap_size,
                    p.kernel_size,
                    p.left_padding,
                    p.filter_gain_a,
                    p.filter_gain_b,
                    p.log_gain_limit,
                    &window,
                );
                hs.process(
                    &mut co,
                    Some(x),
                    &features,
                    h[0],
                    h[1],
                    h[2],
                    lag,
                    p,
                    &window,
                );
            }
            assert_bits_eq_f32(&format!("{what} out frame {f}"), &ro, &co);
            let (cst, clag) = hs.state();
            let (rst, rlag) = flatten_comb(&rs);
            assert_bits_eq_f32(&format!("{what} state frame {f}"), &rst, &cst);
            assert_eq!(rlag, clag);
        }
    }

    #[test]
    fn adashape_real_and_random() {
        let mut rng = Rng::new(0x5_4A9E);
        let nolace = Model::new(c::MODEL_NOLACE, "nolace_data.c");
        for k in 1..=3 {
            let (r1f, h1f) = nolace.layer(&format!("nolace_tdshape{k}_alpha1_f"));
            let (r1t, h1t) = nolace.layer(&format!("nolace_tdshape{k}_alpha1_t"));
            let (r2, h2) = nolace.layer(&format!("nolace_tdshape{k}_alpha2"));
            run_adashape(
                &format!("tdshape{k}"),
                &mut rng,
                [&r1f, &r1t, &r2],
                [&h1f, &h1t, &h2],
                160,
                80,
                4,
                1,
                300,
            );
        }
        for trial in 0..200 {
            let avg = [1usize, 2, 4, 5][trial % 4];
            let interp = [1usize, 2, 4][trial % 3];
            let frame_size =
                avg * interp * rng.range_i32(1, (240 / (avg * interp)) as i32) as usize;
            let tenv = frame_size / avg;
            let hidden = frame_size / interp;
            let fd = rng.range_i32(1, 200) as usize;
            if fd + tenv + 1 >= 512 {
                continue;
            }
            let k1 = rng.range_i32(1, 2) as usize;
            let k2 = rng.range_i32(1, 2) as usize;
            if fd * k1 > 512 || (tenv + 1) * k1 > 512 || hidden * k2 > 240 {
                continue;
            }
            let l1f = RandLayer::new(&mut rng, 0, fd * k1, hidden, false);
            let l1t = RandLayer::new(&mut rng, 0, (tenv + 1) * k1, hidden, false);
            let l2 = RandLayer::new(&mut rng, 0, hidden * k2, hidden, false);
            run_adashape(
                &format!("adashape random #{trial}"),
                &mut rng,
                [&l1f.rust(), &l1t.rust(), &l2.rust()],
                [&l1f.c(), &l1t.c(), &l2.c()],
                fd,
                frame_size,
                avg,
                interp,
                8,
            );
        }
    }

    #[expect(clippy::too_many_arguments, reason = "test helper mirrors the C call")]
    fn run_adashape(
        what: &str,
        rng: &mut Rng,
        r: [&nnet::LinearLayer; 3],
        h: [&c::Linear; 3],
        fd: usize,
        n: usize,
        avg: usize,
        interp: usize,
        frames_n: usize,
    ) {
        let mut rs = nd::AdaShapeState::default();
        let mut hs = c::AdaShape::new();
        let sig = frames(frames_n * n, rng.next_u64());
        for f in 0..frames_n {
            let x = &sig[f * n..(f + 1) * n];
            let features = rand_vec(rng, fd, 1.0);
            let mut ro = vec![0f32; n];
            let mut co = vec![0f32; n];
            if f % 2 == 1 {
                ro.copy_from_slice(x);
                co.copy_from_slice(x);
                nd::adashape_process_frame(
                    &mut rs, &mut ro, None, &features, r[0], r[1], r[2], fd, n, avg, interp,
                );
                hs.process(
                    &mut co, None, &features, h[0], h[1], h[2], fd, n, avg, interp,
                );
            } else {
                nd::adashape_process_frame(
                    &mut rs,
                    &mut ro,
                    Some(x),
                    &features,
                    r[0],
                    r[1],
                    r[2],
                    fd,
                    n,
                    avg,
                    interp,
                );
                hs.process(
                    &mut co,
                    Some(x),
                    &features,
                    h[0],
                    h[1],
                    h[2],
                    fd,
                    n,
                    avg,
                    interp,
                );
            }
            assert_bits_eq_f32(&format!("{what} out frame {f}"), &ro, &co);
            assert_bits_eq_f32(
                &format!("{what} state frame {f}"),
                &flatten_shape(&rs),
                &hs.state(),
            );
        }
    }
}

#[cfg(feature = "dred")]
#[test]
fn dred_dilated_conv_layers() {
    let mut rng = Rng::new(0xD4ED);
    let blob = c::write_blob(c::MODEL_RDOVAE_ENC);
    let arrays = pw::parse_weights(&blob).unwrap();
    let (lin, _) = init_specs("dred_rdovae_enc_data.c");
    for s in lin
        .iter()
        .filter(|s| s.field.starts_with("enc_conv") && !s.field.contains("dense"))
    {
        let r = rust_linear(&arrays, s, s.nb_in, s.nb_out).unwrap();
        let (h, ret) = c_linear(c::MODEL_RDOVAE_ENC, s, s.nb_in, s.nb_out);
        assert_eq!(ret, 0);
        let input_size = r.nb_inputs / 2;
        check_conv1d_dilation(
            &format!("{} dilation 2", s.field),
            &mut rng,
            &r,
            &h,
            input_size,
            2,
            10,
        );
    }
}

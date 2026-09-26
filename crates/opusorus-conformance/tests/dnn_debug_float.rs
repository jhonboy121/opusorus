//! libopus `--enable-dnn-debug-float` (`DISABLE_DEBUG_FLOAT` undefined) vs the oracle built the
//! same way (feature `dnn-debug-float` with `deep-plc` / `dred` / `osce`).
//!
//! Upstream's switch only changes the generated model tables: every int8-quantized layer keeps
//! a float copy of its weights (`<layer>_weights_float`), and `compute_linear` prefers the float
//! weights when a layer has them. The weight blob carries whatever arrays the tables have, so
//! the Rust port computes in float whenever its blob holds the copies:
//!
//! * the oracle's tables really carry the float copies (the define is off), and the blob
//!   serialized from them binds them into every quantized Rust layer;
//! * quantized layers (dense and sparse) compute bit-exactly like C with the float weights,
//!   and differently from their int8 path;
//! * end to end, deep PLC and OSCE decoding with the debug-float models is bit-exact with the
//!   oracle and differs from decoding with the default (int8) models;
//! * `scripts/gen_dnn_blob.sh --debug-float` produces exactly the oracle's tables (the blob to
//!   embed with `dnn-weights-embedded` + `dnn-debug-float`; skipped if not generated).
//!
//! Every other DNN differential suite (`dnn_core`, `dnn_plc`, `dnn_dred`, `dnn_osce`,
//! `dnn_integration`, the `opus_demo` comparison in `vectors`) also runs in this configuration
//! (`just test-options`): they bind the oracle's tables, so they then check the float path.

#![cfg(all(
    feature = "dnn-debug-float",
    any(feature = "deep-plc", feature = "dred", feature = "osce")
))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test code: failures should panic; notes go to stderr"
)]

use opusorus::dnn::{nnet, parse_lpcnet_weights as pw};
use opusorus_conformance::{Rng, assert_bits_eq_f32, signals};
use opusorus_oracle::{api, dnn_core as c, sys};
use std::path::{Path, PathBuf};

/// The oracle models compiled in for the enabled features.
fn models() -> Vec<i32> {
    [
        c::MODEL_PITCHDNN,
        c::MODEL_PLC,
        c::MODEL_FARGAN,
        c::MODEL_RDOVAE_ENC,
        c::MODEL_RDOVAE_DEC,
        c::MODEL_LACE,
        c::MODEL_NOLACE,
        c::MODEL_BBWENET,
    ]
    .into_iter()
    .filter(|&m| c::model_count(m).is_some())
    .collect()
}

/// The oracle's tables as one blob (`gen_dnn_blob.sh` order for a full build).
fn oracle_blob() -> Vec<u8> {
    models().iter().flat_map(|&m| c::write_blob(m)).collect()
}

/// `blob` without the float copies of the int8 layers: the default (`DISABLE_DEBUG_FLOAT`)
/// tables.
fn strip_float_copies(blob: &[u8]) -> Vec<u8> {
    let arrays = pw::parse_weights(blob).unwrap();
    let int8: Vec<Vec<u8>> = arrays
        .iter()
        .filter_map(|a| a.name.strip_suffix(b"_int8").map(<[u8]>::to_vec))
        .collect();
    let kept: Vec<pw::WeightArray<'_>> = arrays
        .iter()
        .filter(|a| {
            !a.name
                .strip_suffix(b"_float")
                .is_some_and(|p| int8.iter().any(|q| q == p))
        })
        .copied()
        .collect();
    let mut out = Vec::new();
    pw::write_weights(&kept, &mut out);
    out
}

#[test]
fn oracle_tables_carry_float_copies() {
    let mut total = 0;
    for m in models() {
        let arrays = c::model_arrays(m);
        for (i, a) in arrays.iter().enumerate() {
            let Some(prefix) = a.name.strip_suffix("_int8") else {
                continue;
            };
            // The generated tables list the float copy right after the int8 array.
            let f = &arrays[i + 1];
            assert_eq!(f.name, format!("{prefix}_float"), "model {m}");
            assert_eq!(f.type_, nnet::WEIGHT_TYPE_FLOAT);
            assert_eq!(f.size, 4 * a.size, "{}", f.name);
            // The copy holds the (unquantized) weights the int8 values approximate.
            let fw: Vec<f32> = f
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            assert!(fw.iter().all(|v| v.is_finite() && v.abs() < 128.0));
            total += 1;
        }
    }
    assert!(total > 20, "{total} quantized layers");
}

/// (model, bias/subias/weights/float/idx/scale prefix, nb_in, nb_out) of quantized layers from
/// the generated `init_*` functions.
fn quantized_layers() -> Vec<(i32, &'static str, bool, usize, usize)> {
    let mut v = vec![
        (c::MODEL_PLC, "plc_gru1_input", false, 128, 576),
        (c::MODEL_PLC, "plc_gru1_recurrent", false, 192, 576),
        (c::MODEL_PLC, "plc_gru2_input", false, 192, 576),
        (c::MODEL_PLC, "plc_gru2_recurrent", false, 192, 576),
    ];
    if cfg!(feature = "dred") {
        v.push((c::MODEL_RDOVAE_ENC, "gdense1", true, 544, 128));
        v.push((c::MODEL_RDOVAE_ENC, "enc_conv_dense2", true, 192, 64));
    }
    v
}

#[test]
fn quantized_layers_compute_in_float() {
    let mut rng = Rng::new(0xDEB6_F10A);
    for (model, p, sparse, nb_in, nb_out) in quantized_layers() {
        let blob = c::write_blob(model);
        let arrays = pw::parse_weights(&blob).unwrap();
        let names = [
            format!("{p}_bias"),
            format!("{p}_subias"),
            format!("{p}_weights_int8"),
            format!("{p}_weights_float"),
            format!("{p}_weights_idx"),
            format!("{p}_scale"),
        ];
        let idx = sparse.then_some(names[4].as_str());
        let r = pw::linear_init(
            &arrays,
            Some(&names[0]),
            Some(&names[1]),
            Some(&names[2]),
            Some(&names[3]),
            idx,
            None,
            Some(&names[5]),
            nb_in,
            nb_out,
        )
        .unwrap();
        assert!(r.float_weights.is_some() && r.weights.is_some(), "{p}");
        let (h, ret) = c::Linear::init(
            model,
            Some(&names[0]),
            Some(&names[1]),
            Some(&names[2]),
            Some(&names[3]),
            idx,
            None,
            Some(&names[5]),
            nb_in,
            nb_out,
        );
        assert_eq!(ret, 0, "{p}");
        let mut int8_only = r.clone();
        int8_only.float_weights = None;
        let mut differs = false;
        for t in 0..50 {
            let input: Vec<f32> = (0..nb_in).map(|_| rng.f32_sym()).collect();
            let mut ro = vec![0f32; nb_out];
            let mut co = vec![0f32; nb_out];
            let mut qo = vec![0f32; nb_out];
            nnet::compute_linear(&r, &mut ro, &input);
            h.compute_linear(&mut co, &input);
            assert_bits_eq_f32(&format!("{p} #{t}"), &ro, &co);
            nnet::compute_linear(&int8_only, &mut qo, &input);
            differs |= ro.iter().zip(&qo).any(|(a, b)| a.to_bits() != b.to_bits());
        }
        assert!(differs, "{p}: the float path gives the int8 result");
    }
}

/// A SILK/hybrid stream from the oracle encoder.
fn c_stream(fs: i32, channels: i32, bitrate: i32, seed: u64) -> Vec<Vec<u8>> {
    let mut e = api::Encoder::new(fs, channels, sys::OPUS_APPLICATION_VOIP).unwrap();
    e.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
    e.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, 1).unwrap();
    e.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20)
        .unwrap();
    let frame = (fs / 50) as usize;
    let x = signals::to_i16(&signals::speech_like(
        frame * 150,
        channels as usize,
        fs as u32,
        seed,
    ));
    x.chunks_exact(frame * channels as usize)
        .map(|f| {
            let mut out = vec![0u8; 1500];
            let n = e.encode(f, frame, &mut out).unwrap();
            out.truncate(n);
            out
        })
        .collect()
}

/// Decodes `packets` (every 7th and 8th lost) with the Rust decoder and `blob`'s models.
fn rust_decode(fs: i32, ch: i32, cplx: i32, blob: &[u8], packets: &[Vec<u8>]) -> Vec<i16> {
    let mut d = opusorus::Decoder::new(fs, ch).unwrap();
    d.set_dnn_blob(blob).unwrap();
    d.set_complexity(cplx).unwrap();
    let frame = (fs / 50) as usize;
    let mut all = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let mut pcm = vec![0i16; frame * ch as usize];
        let data = if i % 8 >= 6 { None } else { Some(&p[..]) };
        let n = d.decode(data, &mut pcm, frame, false).unwrap();
        all.extend_from_slice(&pcm[..n * ch as usize]);
    }
    all
}

/// The same with the oracle decoder (compiled-in debug-float models).
fn c_decode(fs: i32, ch: i32, cplx: i32, packets: &[Vec<u8>]) -> Vec<i16> {
    let mut d = api::Decoder::new(fs, ch).unwrap();
    d.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, cplx).unwrap();
    let frame = (fs / 50) as usize;
    let mut all = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let mut pcm = vec![0i16; frame * ch as usize];
        let data = if i % 8 >= 6 { None } else { Some(&p[..]) };
        let n = d.decode(data, &mut pcm, frame, false).unwrap();
        all.extend_from_slice(&pcm[..n * ch as usize]);
    }
    all
}

#[test]
fn decoding_matches_oracle() {
    let blob = oracle_blob();
    let int8_blob = strip_float_copies(&blob);
    assert!(int8_blob.len() < blob.len());
    // Deep PLC (complexity 5 and 10); OSCE LACE (6) and NoLACE (7) on 16 kHz SILK.
    let mut cases = vec![
        (16000, 1, 5, 16000),
        (48000, 2, 10, 32000),
        (24000, 1, 10, 20000),
    ];
    if cfg!(feature = "osce") {
        cases.extend([(16000, 1, 6, 12000), (16000, 1, 7, 16000)]);
    }
    for (i, &(fs, ch, cplx, br)) in cases.iter().enumerate() {
        let packets = c_stream(fs, ch, br, 40 + i as u64);
        let what = format!("{fs} Hz x{ch}, complexity {cplx}");
        let r = rust_decode(fs, ch, cplx, &blob, &packets);
        let h = c_decode(fs, ch, cplx, &packets);
        assert_eq!(r.len(), h.len(), "{what}");
        if let Some(k) = r.iter().zip(&h).position(|(a, b)| a != b) {
            panic!("{what}: sample {k} differs: {} vs {}", r[k], h[k]);
        }
        let q = rust_decode(fs, ch, cplx, &int8_blob, &packets);
        assert_ne!(q, r, "{what}: the debug-float models change nothing");
    }
}

#[test]
fn gen_dnn_blob_debug_float_matches_oracle() {
    if !(cfg!(feature = "dred") && cfg!(feature = "osce")) {
        eprintln!("NOTE: needs dred + osce (the script writes every model); skipping");
        return;
    }
    let path = match std::env::var_os("OPUSORUS_DNN_DEBUG_FLOAT_BLOB") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/dnn/weights_blob_debug_float.bin"),
    };
    let Ok(file) = std::fs::read(&path) else {
        eprintln!(
            "NOTE: {} not found (run scripts/gen_dnn_blob.sh --debug-float); skipping",
            path.display()
        );
        return;
    };
    // gen_dnn_blob.sh order: pitchdnn, fargan, plcmodel, rdovaeenc, rdovaedec, lace, nolace,
    // bbwenet.
    let order = [
        c::MODEL_PITCHDNN,
        c::MODEL_FARGAN,
        c::MODEL_PLC,
        c::MODEL_RDOVAE_ENC,
        c::MODEL_RDOVAE_DEC,
        c::MODEL_LACE,
        c::MODEL_NOLACE,
        c::MODEL_BBWENET,
    ];
    let oracle: Vec<u8> = order.iter().flat_map(|&m| c::write_blob(m)).collect();
    assert!(
        file == oracle,
        "{} differs from the oracle's debug-float tables",
        path.display()
    );
    // And the default blob is exactly these tables without the float copies.
    let default = path.with_file_name("weights_blob.bin");
    if let Ok(d) = std::fs::read(&default) {
        assert!(d == strip_float_copies(&oracle), "{}", default.display());
    }
}

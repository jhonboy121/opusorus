//! Differential tests for unit `dnn_dred` (dnn/dred_rdovae_enc.c, dred_rdovae_dec.c,
//! dred_rdovae_stats_data.c, dred_coding.c, dred_encoder.c, dred_decoder.c) vs the C oracle
//! built with `--enable-dred`.
//!
//! Run with `cargo test -p opusorus-conformance --features dred --test dnn_dred` (and
//! `--features dred,qext` for the 96 kHz path); needs the model data from
//! `scripts/fetch_dnn_models.sh`. Every comparison is bit-exact.
//!
//! The C oracle uses its compiled-in model tables; the Rust side loads weight blobs that the
//! oracle serializes from those same tables.
#![cfg(feature = "dred")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::celt::entcode::EcState;
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::dnn::dred_coding as dc;
use opusorus::dnn::dred_decoder as dd;
use opusorus::dnn::dred_encoder as de;
use opusorus::dnn::dred_rdovae as rv;
use opusorus::dnn::lpcnet_enc as le;
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::dnn_core as core_c;
use opusorus_oracle::dnn_dred as c;

// ---------------------------------------------------------------------------------------------
// helpers

fn enc_blob() -> Vec<u8> {
    let mut b = core_c::write_blob(core_c::MODEL_RDOVAE_ENC);
    b.extend(core_c::write_blob(core_c::MODEL_PITCHDNN));
    b
}

fn dec_model() -> rv::RdovaeDec {
    rv::dred_rdovae_dec_load_model(&core_c::write_blob(core_c::MODEL_RDOVAE_DEC))
        .expect("rdovae dec model")
}

fn enc_model() -> rv::RdovaeEnc {
    rv::dred_rdovae_enc_load_model(&core_c::write_blob(core_c::MODEL_RDOVAE_ENC))
        .expect("rdovae enc model")
}

fn rand_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| amp * rng.f32_sym()).collect()
}

fn flatten_rdovae_enc(st: &rv::RdovaeEncState) -> Vec<f32> {
    let mut v = Vec::new();
    for s in [
        &st.gru1_state,
        &st.gru2_state,
        &st.gru3_state,
        &st.gru4_state,
        &st.gru5_state,
    ] {
        v.extend_from_slice(s);
    }
    v.extend_from_slice(&st.conv1_state);
    for s in [
        &st.conv2_state,
        &st.conv3_state,
        &st.conv4_state,
        &st.conv5_state,
    ] {
        v.extend_from_slice(s);
    }
    v
}

fn flatten_rdovae_dec(st: &rv::RdovaeDecState) -> Vec<f32> {
    let mut v = Vec::new();
    for s in [
        &st.gru1_state,
        &st.gru2_state,
        &st.gru3_state,
        &st.gru4_state,
        &st.gru5_state,
    ] {
        v.extend_from_slice(s);
    }
    for s in [
        &st.conv1_state,
        &st.conv2_state,
        &st.conv3_state,
        &st.conv4_state,
        &st.conv5_state,
    ] {
        v.extend_from_slice(s);
    }
    v
}

/// Same order as the oracle's `oracle_dc_lpcnet_enc_state`.
fn flatten_lpcnet_enc(st: &le::LpcnetEncState) -> Vec<f32> {
    let mut v = Vec::new();
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
    v
}

fn enc_floats(e: &de::DredEnc) -> Vec<f32> {
    let mut v = Vec::with_capacity(c::DRED_ENC_FLOATS_LEN);
    v.extend_from_slice(&e.input_buffer);
    v.extend_from_slice(&e.latents_buffer);
    v.extend_from_slice(&e.state_buffer);
    v.extend_from_slice(&e.resample_mem);
    v.extend(flatten_rdovae_enc(&e.rdovae_enc));
    v.extend(flatten_lpcnet_enc(&e.lpcnet_enc_state));
    v
}

fn enc_ints(e: &de::DredEnc) -> c::DredEncInts {
    c::DredEncInts {
        input_buffer_fill: e.input_buffer_fill,
        dred_offset: e.dred_offset,
        latent_offset: e.latent_offset,
        last_extra_dred_offset: e.last_extra_dred_offset,
        latents_buffer_fill: e.latents_buffer_fill,
        loaded: i32::from(e.loaded),
        fs: e.fs,
        channels: e.channels,
        rdovae_initialized: e.rdovae_enc.initialized,
    }
}

#[track_caller]
fn compare_enc(what: &str, r: &de::DredEnc, h: &c::DredEnc) {
    assert_eq!(enc_ints(r), h.ints(), "{what}: ints");
    assert_bits_eq_f32(what, &enc_floats(r), &h.floats());
}

/// A new Rust encoder with the model loaded.
fn rust_enc(fs: i32, ch: i32, blob: &[u8]) -> Box<de::DredEnc> {
    let mut r = de::DredEnc::new(fs, ch);
    assert!(!r.loaded);
    r.load_model(blob).expect("dred enc model");
    r
}

/// Valid random (q0, dQ, qmax) as the Opus encoder would pass them.
const fn rand_q(rng: &mut Rng) -> (i32, i32, i32) {
    let q0 = rng.range_i32(0, 15);
    let dq = rng.range_i32(0, 7);
    let qmax = if q0 < 14 && dq > 0 {
        rng.range_i32(q0 + 1, 15)
    } else {
        rng.range_i32(q0, 15)
    };
    (q0, dq, qmax)
}

/// Voice-activity history as kept by the Opus encoder (newest first, 2.5 ms resolution), plus
/// 8 trailing bytes standing in for the C struct fields C reads past the array.
struct Activity {
    mem: Vec<u8>,
    talking: bool,
}

impl Activity {
    fn new() -> Self {
        Self {
            mem: vec![0; de::DRED_ACTIVITY_MEM_SIZE + 8],
            talking: false,
        }
    }
    fn push_frame(&mut self, rng: &mut Rng, n400: usize, mode: u32) {
        let n = de::DRED_ACTIVITY_MEM_SIZE;
        self.mem.copy_within(0..n - n400, n400);
        // Talk spurts / silences (Markov chain), or always on / always off.
        self.talking = match mode {
            0 => true,
            1 => false,
            _ => {
                if rng.range_i32(0, 99) < 12 {
                    !self.talking
                } else {
                    self.talking
                }
            }
        };
        let v = u8::from(self.talking);
        self.mem[..n400].fill(v);
    }
}

fn signal(kind: u64, n: usize, ch: usize, fs: u32, seed: u64) -> Vec<f32> {
    match kind % 6 {
        0 | 1 => signals::speech_like(n, ch, fs, seed),
        2 => signals::music_like(n, ch, fs, seed),
        3 => signals::noise(n, ch, 0.2, seed),
        // Clipping speech (FLOAT2INT16 saturation) with silent gaps.
        4 => signals::speech_like(n, ch, fs, seed)
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if (i / (ch * fs as usize / 4)) % 3 == 2 {
                    0.0
                } else {
                    v * 3.0
                }
            })
            .collect(),
        _ => vec![0.0; n * ch],
    }
}

// ---------------------------------------------------------------------------------------------
// constants / tables / quantizer

#[test]
fn constants_tables_and_quantizer() {
    let cst = c::constants();
    let rust: Vec<i32> = vec![
        rv::DRED_NUM_FEATURES as i32,
        rv::DRED_LATENT_DIM as i32,
        rv::DRED_STATE_DIM as i32,
        rv::DRED_PADDED_LATENT_DIM as i32,
        rv::DRED_PADDED_STATE_DIM as i32,
        rv::DRED_NUM_QUANTIZATION_LEVELS as i32,
        rv::DRED_MAX_RNN_NEURONS as i32,
        rv::DRED_MAX_CONV_INPUTS as i32,
        rv::DRED_ENC_MAX_RNN_NEURONS as i32,
        rv::DRED_ENC_MAX_CONV_INPUTS as i32,
        rv::DRED_DEC_MAX_RNN_NEURONS as i32,
        dc::DRED_EXTENSION_ID,
        dc::DRED_EXPERIMENTAL_VERSION,
        dc::DRED_EXPERIMENTAL_BYTES,
        dc::DRED_MIN_BYTES,
        dc::DRED_SILK_ENCODER_DELAY,
        dc::DRED_FRAME_SIZE as i32,
        dc::DRED_DFRAME_SIZE as i32,
        dc::DRED_MAX_DATA_SIZE as i32,
        dc::DRED_ENC_Q0,
        dc::DRED_ENC_Q1,
        dc::DRED_MAX_LATENTS as i32,
        dc::DRED_NUM_REDUNDANCY_FRAMES as i32,
        dc::DRED_MAX_FRAMES as i32,
        de::RESAMPLING_ORDER as i32,
        de::MAX_DOWNMIX_BUFFER as i32,
        dd::DRED_FEC_FEATURES_SIZE as i32,
        dd::DRED_LATENTS_SIZE as i32,
    ];
    assert_slice_eq("constants", &rust, &cst);

    let tables: [&[u8]; 8] = [
        &rv::DRED_LATENT_QUANT_SCALES_Q8,
        &rv::DRED_LATENT_DEAD_ZONE_Q8,
        &rv::DRED_LATENT_R_Q8,
        &rv::DRED_LATENT_P0_Q8,
        &rv::DRED_STATE_QUANT_SCALES_Q8,
        &rv::DRED_STATE_DEAD_ZONE_Q8,
        &rv::DRED_STATE_R_Q8,
        &rv::DRED_STATE_P0_Q8,
    ];
    for (i, t) in tables.iter().enumerate() {
        assert_slice_eq(&format!("stats table {i}"), t, &c::stats(i as i32));
    }

    for q0 in -2..20 {
        for dq in 0..8 {
            for qmax in -2..20 {
                for i in 0..60 {
                    assert_eq!(
                        dc::compute_quantizer(q0, dq, qmax, i),
                        c::compute_quantizer(q0, dq, qmax, i),
                        "compute_quantizer({q0}, {dq}, {qmax}, {i})"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// latent entropy coding

#[test]
fn encode_decode_latents() {
    let mut rng = Rng::new(0xD4ED);
    for iter in 0..3000 {
        let state = iter % 2 == 0;
        let (dim, levels) = if state {
            (rv::DRED_STATE_DIM, 16)
        } else {
            (rv::DRED_LATENT_DIM, 16)
        };
        let q = rng.range_i32(0, levels - 1) as usize;
        let off = q * dim;
        let (scale, dzone, r, p0): (&[u8], &[u8], &[u8], &[u8]) = if state {
            (
                &rv::DRED_STATE_QUANT_SCALES_Q8[off..],
                &rv::DRED_STATE_DEAD_ZONE_Q8[off..],
                &rv::DRED_STATE_R_Q8[off..],
                &rv::DRED_STATE_P0_Q8[off..],
            )
        } else {
            (
                &rv::DRED_LATENT_QUANT_SCALES_Q8[off..],
                &rv::DRED_LATENT_DEAD_ZONE_Q8[off..],
                &rv::DRED_LATENT_R_Q8[off..],
                &rv::DRED_LATENT_P0_Q8[off..],
            )
        };
        // Random tables too (exercise r == 0 / p0 == 255 / scale == 0 paths).
        // p0 == 0 is excluded: it makes the sign ICDF start at 32768, which zeroes the range
        // and hangs the C range coder (real tables have p0 >= 1).
        let rt: Vec<Vec<u8>> = (0..4)
            .map(|t| {
                (0..dim)
                    .map(|_| match rng.range_i32(0, 9) {
                        0 if t != 3 => 0,
                        1 => 255,
                        _ if t == 3 => rng.range_i32(1, 255) as u8,
                        _ => rng.next_u32() as u8,
                    })
                    .collect()
            })
            .collect();
        let use_rand = iter % 5 == 4;
        let (scale, dzone, r, p0) = if use_rand {
            (&rt[0][..], &rt[1][..], &rt[2][..], &rt[3][..])
        } else {
            (scale, dzone, r, p0)
        };
        let amp = [0.01f32, 0.3, 1.0, 4.0, 40.0][rng.range_i32(0, 4) as usize];
        let x = rand_vec(&mut rng, dim, amp);
        let max_bytes = [2usize, 8, 30, 200][rng.range_i32(0, 3) as usize];

        let ce = c::encode_latents(max_bytes, &x, scale, dzone, r, p0, dim);
        let mut buf = vec![0u8; max_bytes];
        let (tell, rng_final, err) = {
            let mut enc = EcEnc::new(&mut buf);
            de::dred_encode_latents(&mut enc, &x, scale, dzone, r, p0, dim);
            let t = enc.tell();
            enc.done();
            (t, enc.rng(), enc.get_error())
        };
        assert_eq!(
            (tell, rng_final, err),
            (ce.tell, ce.rng, ce.error),
            "encode_latents {iter}"
        );
        assert_slice_eq(&format!("encode_latents bytes {iter}"), &buf, &ce.buf);

        // Decode the C bytes (and a corrupted copy) on both sides.
        for pass in 0..2 {
            let mut bytes = ce.buf.clone();
            if pass == 1 {
                rng.fill_bytes(&mut bytes);
            }
            let (cx, ctell, crng) = c::decode_latents(&bytes, scale, r, p0, dim);
            let mut dec = EcDec::new(&bytes);
            let mut rx = vec![0f32; dim];
            dd::dred_decode_latents(&mut dec, &mut rx, scale, r, p0, dim);
            assert_eq!(
                (dec.tell(), dec.rng()),
                (ctell, crng),
                "decode_latents {iter}/{pass}"
            );
            assert_bits_eq_f32(&format!("decode_latents {iter}/{pass}"), &rx, &cx);
        }
    }
}

#[test]
fn voice_active() {
    let mut rng = Rng::new(0xAC7);
    for _ in 0..2000 {
        let mut mem = vec![0u8; de::DRED_ACTIVITY_MEM_SIZE + 8];
        for v in &mut mem {
            *v = match rng.range_i32(0, 20) {
                0 => 1,
                1 => 2,
                _ => 0,
            };
        }
        for off in 0..=51 {
            assert_eq!(
                de::dred_voice_active(&mem, off),
                c::voice_active(&mem, off),
                "voice_active {off}"
            );
        }
        // Past the 416-byte array Rust treats missing entries as inactive (C reads the
        // following struct fields; zero here).
        mem[de::DRED_ACTIVITY_MEM_SIZE..].fill(0);
        assert_eq!(
            de::dred_voice_active(&mem[..de::DRED_ACTIVITY_MEM_SIZE], 51),
            c::voice_active(&mem, 51)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// RDOVAE

#[test]
fn rdovae_encoder_streams() {
    let model = enc_model();
    let mut rng = Rng::new(0x5D0E);
    let pitch = core_c::write_blob(core_c::MODEL_PITCHDNN);
    for stream in 0..6u64 {
        let mut r = rv::RdovaeEncState::default();
        let mut h = c::RdovaeEnc::new();
        // Realistic LPCNet features from 16 kHz signals (and random ones for two streams).
        let frames = 250;
        let sig16: Vec<f32> = signal(stream, frames * 320, 1, 16000, rng.next_u64())
            .iter()
            .map(|v| v * 32768.0)
            .collect();
        let mut lpc = le::LpcnetEncState::new();
        lpc.load_model(&pitch).expect("pitch model");
        for f in 0..frames {
            let mut input = [0f32; 2 * rv::DRED_NUM_FEATURES];
            if stream >= 4 {
                let amp = if stream == 4 { 1.0 } else { 20.0 };
                input.copy_from_slice(&rand_vec(&mut rng, 40, amp));
            } else {
                let mut feat = [0f32; 72];
                le::lpcnet_compute_single_frame_features_float(
                    &mut lpc,
                    &sig16[f * 320..],
                    &mut feat,
                );
                le::lpcnet_compute_single_frame_features_float(
                    &mut lpc,
                    &sig16[f * 320 + 160..],
                    &mut feat[36..],
                );
                input[..20].copy_from_slice(&feat[..20]);
                input[20..].copy_from_slice(&feat[36..56]);
            }
            let mut lat = [0f32; rv::DRED_LATENT_DIM];
            let mut st = [0f32; rv::DRED_STATE_DIM];
            rv::dred_rdovae_encode_dframe(&mut r, &model, &mut lat, &mut st, &input);
            let (cl, cs) = h.encode_dframe(&input);
            assert_bits_eq_f32(&format!("latents {stream}/{f}"), &lat, &cl);
            assert_bits_eq_f32(&format!("state {stream}/{f}"), &st, &cs);
            let (ci, cst) = h.state();
            assert_eq!(r.initialized, ci);
            assert_bits_eq_f32(
                &format!("enc state {stream}/{f}"),
                &flatten_rdovae_enc(&r),
                &cst,
            );
        }
    }
}

#[test]
fn rdovae_decoder_streams() {
    let model = dec_model();
    let emodel = enc_model();
    let mut rng = Rng::new(0x5DEC);
    let h_all = c::RdovaeDec::new();
    for stream in 0..6u64 {
        let mut r = rv::RdovaeDecState::default();
        let mut h = c::RdovaeDec::new();
        // Latents from the encoder on smooth random features (like real DRED latents), with
        // their quantizer level appended.
        let mut es = rv::RdovaeEncState::default();
        let mut feat = rand_vec(&mut rng, 40, 1.0);
        for f in 0..200 {
            for v in &mut feat {
                *v = 0.9 * *v + 0.3 * rng.f32_sym();
            }
            let mut lat = [0f32; rv::DRED_LATENT_DIM + 1];
            let mut st = [0f32; rv::DRED_STATE_DIM];
            rv::dred_rdovae_encode_dframe(&mut es, &emodel, &mut lat, &mut st, &feat);
            lat[rv::DRED_LATENT_DIM] = rng.range_i32(0, 15) as f32 * 0.125 - 1.0;
            if stream == 5 {
                // Garbage latents.
                lat.copy_from_slice(&rand_vec(&mut rng, 26, 30.0));
            }
            // Re-initialise now and then without clearing the conv memories (C quirk).
            if f % 50 == 0 {
                rv::dred_rdovae_dec_init_states(&mut r, &model, &st);
                h.init_states(&st);
            }
            let mut q = [0f32; rv::DEC_OUTPUT_OUT_SIZE];
            rv::dred_rdovae_decode_qframe(&mut r, &model, &mut q, &lat);
            let cq = h.decode_qframe(&lat);
            assert_bits_eq_f32(&format!("qframe {stream}/{f}"), &q, &cq);
            let (ci, cst) = h.state();
            assert_eq!(r.initialized, ci);
            assert_bits_eq_f32(
                &format!("dec state {stream}/{f}"),
                &flatten_rdovae_dec(&r),
                &cst,
            );
        }
        // decode_all on random latents / state for every count.
        let state = rand_vec(&mut rng, rv::DRED_STATE_DIM, 2.0);
        let latents = rand_vec(&mut rng, dd::DRED_LATENTS_SIZE, 3.0);
        for nb in 0..=26 {
            let mut f = vec![0f32; 80 * nb];
            rv::dred_rdovae_decode_all(&model, &mut f, &state, &latents, nb);
            assert_bits_eq_f32(
                &format!("decode_all {stream}/{nb}"),
                &f,
                &h_all.decode_all(&state, &latents, nb),
            );
        }
    }
    // Corrupted blobs are rejected.
    let blob = core_c::write_blob(core_c::MODEL_RDOVAE_DEC);
    assert!(rv::dred_rdovae_dec_load_model(&blob[..blob.len() - 3]).is_err());
    assert!(rv::dred_rdovae_dec_load_model(&core_c::write_blob(core_c::MODEL_PITCHDNN)).is_err());
    assert!(rv::dred_rdovae_enc_load_model(&blob).is_err());
}

// ---------------------------------------------------------------------------------------------
// DRED encoder

#[test]
fn filter_df2t() {
    let mut rng = Rng::new(0xDF27);
    for iter in 0..400 {
        let order = if iter % 4 == 0 {
            rng.range_i32(1, 8) as usize
        } else {
            8
        };
        let b0 = rng.f32_sym() * 0.1;
        let b = rand_vec(&mut rng, 8, 0.1);
        let a = rand_vec(&mut rng, 8, 0.2);
        let mut rmem = rand_vec(&mut rng, 9, 1.0);
        let mut cmem = rmem.clone();
        for chunk in 0..5 {
            let len = rng.range_i32(0, 400) as usize;
            let x = rand_vec(&mut rng, len, 30000.0);
            let inplace = (iter + chunk) % 2 == 0;
            let cout = c::filter_df2t(&x, b0, &b, &a, order, &mut cmem, inplace, len);
            let mut rout = vec![0f32; len];
            if inplace {
                rout.copy_from_slice(&x);
                de::filter_df2t_inplace(&mut rout, len, b0, &b, &a, order, &mut rmem);
            } else {
                de::filter_df2t(&x, &mut rout, len, b0, &b, &a, order, &mut rmem);
            }
            assert_bits_eq_f32(&format!("filter_df2t {iter}/{chunk}"), &rout, &cout);
            assert_bits_eq_f32(&format!("filter_df2t mem {iter}/{chunk}"), &rmem, &cmem);
        }
    }
}

fn rates() -> Vec<i32> {
    let mut v = vec![8000, 12000, 16000, 24000, 48000];
    if cfg!(feature = "qext") {
        v.push(96000);
    }
    v
}

#[test]
fn convert_to_16k() {
    let blob = enc_blob();
    let mut rng = Rng::new(0x16C);
    for fs in rates() {
        for ch in 1..=2 {
            let mut r = rust_enc(fs, ch, &blob);
            let mut h = c::DredEnc::new(fs, ch);
            let sig = signal(
                rng.next_u64(),
                fs as usize * 2,
                ch as usize,
                fs as u32,
                rng.next_u64(),
            );
            let mut pos = 0usize;
            let mut k = 0;
            // 2.5..20 ms chunks at 16 kHz granularity.
            while pos + (fs as usize / 50) * ch as usize <= sig.len() {
                let out_len = [40usize, 80, 160, 320][k % 4];
                k += 1;
                let in_len = out_len * fs as usize / 16000;
                let x = &sig[pos..pos + in_len * ch as usize];
                let cout = h.convert_to_16k(x, in_len, out_len);
                let mut rout = vec![0f32; out_len];
                r.dred_convert_to_16k(x, in_len, &mut rout, out_len);
                assert_bits_eq_f32(&format!("convert_to_16k {fs}/{ch}/{k}"), &rout, &cout);
                pos += in_len * ch as usize;
            }
            compare_enc(&format!("convert state {fs}/{ch}"), &r, &h);
        }
    }
}

/// Frame sizes (in 2.5 ms units) valid for Opus at every rate.
const FRAME_UNITS: [i32; 6] = [1, 2, 4, 8, 16, 24];

struct EncCase {
    fs: i32,
    ch: i32,
    units: i32,
    extra_delay: i32,
    kind: u64,
    activity_mode: u32,
    frames: usize,
}

fn run_encoder_case(case: &EncCase, blob: &[u8], rng: &mut Rng, payloads: &mut Vec<Vec<u8>>) {
    let EncCase {
        fs,
        ch,
        units,
        extra_delay,
        kind,
        activity_mode,
        frames,
    } = *case;
    let tag = format!("fs {fs} ch {ch} units {units} delay {extra_delay} kind {kind}");
    let mut r = rust_enc(fs, ch, blob);
    let mut h = c::DredEnc::new(fs, ch);
    assert!(h.loaded());
    compare_enc(&format!("{tag} init"), &r, &h);
    let frame_size = units * fs / 400;
    let fsz = frame_size as usize * ch as usize;
    let sig = signal(
        kind,
        frames * frame_size as usize + frame_size as usize,
        ch as usize,
        fs as u32,
        rng.next_u64(),
    );
    let mut act = Activity::new();
    for f in 0..frames {
        let pcm = &sig[f * fsz..];
        de::dred_compute_latents(&mut r, pcm, frame_size, extra_delay);
        h.compute_latents(pcm, frame_size, extra_delay);
        compare_enc(&format!("{tag} frame {f}"), &r, &h);
        act.push_frame(rng, units as usize, activity_mode);
        // Randomize the 8 bytes C reads past activity_mem (struct fields in the encoder); Rust
        // gets them too to stay in sync.
        if f % 7 == 0 {
            for v in &mut act.mem[de::DRED_ACTIVITY_MEM_SIZE..] {
                *v = u8::from(rng.range_i32(0, 3) == 0);
            }
        }
        // Encode the payload like the Opus encoder would, with a couple of parameter sets.
        for _ in 0..2 {
            let (q0, dq, qmax) = rand_q(rng);
            let max_chunks = rng.range_i32(1, 26);
            let max_bytes = [
                rng.range_i32(1, 12),
                rng.range_i32(12, 80),
                rng.range_i32(80, 1000),
            ][rng.range_i32(0, 2) as usize];
            let buf_len = max_bytes as usize + rng.range_i32(0, 4) as usize;
            let (cret, cbuf) =
                h.encode_silk_frame(buf_len, max_chunks, max_bytes, q0, dq, qmax, &act.mem);
            let mut rbuf = vec![0u8; buf_len];
            let rret = de::dred_encode_silk_frame(
                &mut r, &mut rbuf, max_chunks, max_bytes, q0, dq, qmax, &act.mem,
            );
            let ptag = format!(
                "{tag} frame {f} q0 {q0} dq {dq} qmax {qmax} chunks {max_chunks} bytes {max_bytes}"
            );
            assert_eq!(rret, cret, "{ptag}: ret");
            assert_slice_eq(&format!("{ptag}: buf"), &rbuf, &cbuf);
            assert_eq!(enc_ints(&r), h.ints(), "{ptag}: ints");
            if rret > 0 && payloads.len() < 400 {
                payloads.push(cbuf[..rret as usize].to_vec());
            }
        }
    }
    // Reset and continue a few frames.
    r.dred_encoder_reset();
    h.reset();
    compare_enc(&format!("{tag} reset"), &r, &h);
    for f in 0..8 {
        let pcm = &sig[f * fsz..];
        de::dred_compute_latents(&mut r, pcm, frame_size, extra_delay);
        h.compute_latents(pcm, frame_size, extra_delay);
        compare_enc(&format!("{tag} after reset {f}"), &r, &h);
    }
}

fn check_payload_decode(payloads: &[Vec<u8>], rng: &mut Rng) {
    let model = dec_model();
    let h = c::RdovaeDec::new();
    for (k, p) in payloads.iter().enumerate() {
        let min_ff = [104, rng.range_i32(-3, 110), 2][k % 3];
        let frame_off = rng.range_i32(-4, 24);
        let cd = c::ec_decode(p, min_ff, frame_off);
        let mut rd = dd::OpusDred::new();
        let ret = dd::dred_ec_decode(&mut rd, p, min_ff, frame_off);
        let tag = format!("payload {k} ({} bytes)", p.len());
        assert_eq!(
            (ret, rd.nb_latents, rd.process_stage, rd.dred_offset),
            (cd.ret, cd.nb_latents, cd.process_stage, cd.dred_offset),
            "{tag}"
        );
        assert_bits_eq_f32(&format!("{tag} state"), &rd.state, &cd.state);
        assert_bits_eq_f32(&format!("{tag} latents"), &rd.latents, &cd.latents);
        // Stage 2: RDOVAE decoding to features.
        let nb = rd.nb_latents as usize;
        rv::dred_rdovae_decode_all(&model, &mut rd.fec_features, &rd.state, &rd.latents, nb);
        let cf = h.decode_all(&cd.state, &cd.latents, nb);
        assert_bits_eq_f32(&format!("{tag} features"), &rd.fec_features[..80 * nb], &cf);
    }
}

#[test]
fn dred_encoder_streams() {
    let blob = enc_blob();
    let mut rng = Rng::new(0xD8ED);
    let mut payloads = Vec::new();
    let mut n = 0u64;
    for fs in rates() {
        for ch in 1..=2 {
            // Every Opus frame size (2.5 ms .. 60 ms) per (rate, channels).
            for (j, &units) in FRAME_UNITS.iter().enumerate() {
                // Opus passes total_buffer = Fs/400 (or 0 in restricted low-delay mode).
                let extra_delay = if (n + j as u64).is_multiple_of(3) {
                    0
                } else {
                    fs / 400
                };
                let case = EncCase {
                    fs,
                    ch,
                    units,
                    extra_delay,
                    kind: n + j as u64,
                    activity_mode: ((n + j as u64) % 4) as u32,
                    // ~1.5 s of audio.
                    frames: 1200 / units as usize,
                };
                run_encoder_case(&case, &blob, &mut rng, &mut payloads);
            }
            n += 1;
        }
    }
    assert!(payloads.len() > 50, "only {} payloads", payloads.len());
    check_payload_decode(&payloads, &mut rng);
}

#[test]
fn dred_encoder_long_speech() {
    // 48 kHz mono 20 ms frames (the common Opus VoIP setup), 30 s.
    let blob = enc_blob();
    let mut rng = Rng::new(0x10E6);
    let mut payloads = Vec::new();
    let case = EncCase {
        fs: 48000,
        ch: 1,
        units: 8,
        extra_delay: 120,
        kind: 0,
        activity_mode: 2,
        frames: 1500,
    };
    run_encoder_case(&case, &blob, &mut rng, &mut payloads);
    check_payload_decode(&payloads, &mut rng);
}

#[test]
fn encode_silk_frame_params() {
    // The payload coder on arbitrary latent buffers / offsets, all quantizer settings.
    let blob = enc_blob();
    let mut rng = Rng::new(0x51CF);
    let mut r = rust_enc(16000, 1, &blob);
    let mut h = c::DredEnc::new(16000, 1);
    let mut payloads = Vec::new();
    for iter in 0..2500 {
        let amp = [0.1f32, 1.0, 3.0, 50.0][iter % 4];
        let lat = rand_vec(&mut rng, c::MAX_FRAMES * c::LATENT_DIM, amp);
        let st = rand_vec(&mut rng, c::MAX_FRAMES * c::STATE_DIM, amp);
        let fill = rng.range_i32(0, 52);
        let latent_offset = rng.range_i32(0, fill);
        let dred_offset = rng.range_i32(-8, 16);
        let last_extra = [0, rng.range_i32(0, 10), rng.range_i32(0, fill)][iter % 3];
        r.latents_buffer.copy_from_slice(&lat);
        r.state_buffer.copy_from_slice(&st);
        r.latents_buffer_fill = fill;
        r.latent_offset = latent_offset;
        r.dred_offset = dred_offset;
        r.last_extra_dred_offset = last_extra;
        h.set(&lat, &st, fill, dred_offset, latent_offset, last_extra);
        let mut act = Activity::new();
        let mode = (iter % 5) as u32;
        for _ in 0..rng.range_i32(1, 60) {
            let n400 = rng.range_i32(1, 24) as usize;
            act.push_frame(&mut rng, n400, mode.min(2));
        }
        if mode == 4 {
            for v in &mut act.mem {
                *v = u8::from(rng.range_i32(0, 3) == 0);
            }
        }
        let (q0, dq, qmax) = rand_q(&mut rng);
        let max_chunks = rng.range_i32(0, 30);
        let max_bytes = rng.range_i32(0, 400);
        let buf_len = max_bytes as usize;
        let (cret, cbuf) =
            h.encode_silk_frame(buf_len, max_chunks, max_bytes, q0, dq, qmax, &act.mem);
        let mut rbuf = vec![0u8; buf_len];
        let rret = de::dred_encode_silk_frame(
            &mut r, &mut rbuf, max_chunks, max_bytes, q0, dq, qmax, &act.mem,
        );
        let tag = format!(
            "iter {iter} q0 {q0} dq {dq} qmax {qmax} chunks {max_chunks} bytes {max_bytes}"
        );
        assert_eq!(rret, cret, "{tag}");
        assert_slice_eq(&tag, &rbuf, &cbuf);
        assert_eq!(
            enc_ints(&r).last_extra_dred_offset,
            h.ints().last_extra_dred_offset
        );
        if rret > 0 && iter % 4 == 0 {
            payloads.push(cbuf[..rret as usize].to_vec());
        }
    }
    assert!(payloads.len() > 100);
    check_payload_decode(&payloads, &mut rng);
}

#[test]
fn dred_process_frame_direct() {
    // dred_process_frame on arbitrary 16 kHz input buffers.
    let blob = enc_blob();
    let mut rng = Rng::new(0xF4A3);
    let mut r = rust_enc(16000, 1, &blob);
    let mut h = c::DredEnc::new(16000, 1);
    for f in 0..120 {
        let input: Vec<f32> = if f % 3 == 0 {
            rand_vec(&mut rng, c::INPUT_BUFFER_SIZE, 20000.0)
        } else {
            signal(f as u64, c::INPUT_BUFFER_SIZE, 1, 16000, rng.next_u64())
                .iter()
                .map(|v| v * 32768.0)
                .collect()
        };
        let fill = rng.range_i32(320, 639);
        r.input_buffer.copy_from_slice(&input);
        r.input_buffer_fill = fill;
        h.set_input(&input, fill);
        de::dred_process_frame(&mut r);
        h.process_frame();
        compare_enc(&format!("process_frame {f}"), &r, &h);
    }
}

#[test]
fn dred_ec_decode_garbage() {
    let mut rng = Rng::new(0x6A4B);
    let model = dec_model();
    let h = c::RdovaeDec::new();
    for iter in 0..4000 {
        let len = [
            0usize,
            1,
            3,
            8,
            rng.range_i32(0, 40) as usize,
            rng.range_i32(0, 300) as usize,
        ][iter % 6];
        let mut bytes = vec![0u8; len];
        rng.fill_bytes(&mut bytes);
        if iter % 7 == 0 {
            bytes.iter_mut().for_each(|b| *b &= 0x0F);
        }
        let min_ff = rng.range_i32(-10, 120);
        let frame_off = rng.range_i32(-100, 100);
        let cd = c::ec_decode(&bytes, min_ff, frame_off);
        let mut rd = dd::OpusDred::default();
        let ret = dd::dred_ec_decode(&mut rd, &bytes, min_ff, frame_off);
        assert_eq!(
            (ret, rd.nb_latents, rd.process_stage, rd.dred_offset),
            (cd.ret, cd.nb_latents, cd.process_stage, cd.dred_offset),
            "garbage {iter}"
        );
        assert_bits_eq_f32(&format!("garbage state {iter}"), &rd.state, &cd.state);
        assert_bits_eq_f32(&format!("garbage latents {iter}"), &rd.latents, &cd.latents);
        if iter % 20 == 0 {
            let nb = rd.nb_latents as usize;
            rv::dred_rdovae_decode_all(&model, &mut rd.fec_features, &rd.state, &rd.latents, nb);
            assert_bits_eq_f32(
                &format!("garbage features {iter}"),
                &rd.fec_features[..80 * nb],
                &h.decode_all(&cd.state, &cd.latents, nb),
            );
        }
    }
}

#[test]
fn encoder_model_loading() {
    let blob = enc_blob();
    let mut h = c::DredEnc::new(48000, 1);
    assert_eq!(h.load_model(&blob), 0);
    // RDOVAE only (no pitch DNN): both fail (C returns OPUS_BAD_ARG = -1).
    let rdo = core_c::write_blob(core_c::MODEL_RDOVAE_ENC);
    assert_eq!(h.load_model(&rdo), -1);
    let mut r = de::DredEnc::new(48000, 1);
    assert!(r.load_model(&rdo).is_err());
    assert!(!r.loaded);
    // Pitch only: RDOVAE binding fails.
    assert_eq!(
        h.load_model(&core_c::write_blob(core_c::MODEL_PITCHDNN)),
        -1
    );
    assert!(
        r.load_model(&core_c::write_blob(core_c::MODEL_PITCHDNN))
            .is_err()
    );
    // Truncated blob: C would crash, Rust rejects it.
    assert!(r.load_model(&blob[..blob.len() - 10]).is_err());
    assert!(!r.loaded);
    r.load_model(&blob).expect("full blob");
    assert!(r.loaded);
    // Re-init clears `loaded` (USE_WEIGHTS_FILE semantics) but keeps the Fs/channels update.
    r.dred_encoder_init(16000, 2);
    assert!(!r.loaded);
    assert_eq!((r.fs, r.channels), (16000, 2));
}

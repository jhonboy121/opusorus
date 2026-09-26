//! Public-API overflow hardening, Rust-only (so it also runs on 32-bit targets such as
//! wasm32-wasip1, where the fixed-point build uses libopus' `OPUS_FAST_INT64 = 0` multiply
//! forms): configurations on which libopus overflows signed integers (undefined behaviour in
//! C, wrapping in practice) must not panic in the overflow-checked test profile. The bit-exact
//! comparison of the same cases with the C oracle is in
//! `opusorus-conformance/tests/api_overflow.rs`.
//!
//! * `lfe_stereo_encoder`: a stereo encoder with `OPUS_SET_LFE(1)` (in the fixed-point build the
//!   mid/side energies of `stereo_itheta` overflow).
//! * `decoder_max_gain`: `OPUS_SET_GAIN` at ±128 dB (the 24-bit fixed-point gain product
//!   `MULT32_32_Q16` exceeds 32 bits).
//! * `random_public_api`: random encoders (rates, channels, applications, CTLs, frame sizes,
//!   input formats, full-scale and beyond-full-scale signals) and decoders (gain, output
//!   formats, corrupted packets, PLC, FEC); `OPUSORUS_API_SWEEP=<n>` / `OPUS_TEST_FULL` scale it.
//!   On wasm32 it found the decoder gain overflow of the 24-bit fixed-point build (the 32-bit
//!   `MULT32_32_Q16` form at `OPUS_SET_GAIN` near +128 dB).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::FastRand;
use opusorus::encoder::request as req;
use opusorus::{Application, Decoder, Encoder};

/// A test signal in `[-amp, amp]`: `kind` 0 = noise, 1 = square wave, 2 = low sine, 3 = DC,
/// 4 = alternating samples, 5 = sparse impulses.
fn signal(rng: &mut FastRand, kind: u32, n: usize, ch: usize, amp: f32) -> Vec<f32> {
    let period = 2 + (rng.next() % 300) as usize;
    (0..n * ch)
        .map(|k| {
            let (i, c) = (k / ch, k % ch);
            let v = match kind {
                0 => (rng.next() >> 8) as f32 / 8_388_608.0 - 1.0,
                1 => {
                    if (i / (period + c)).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                2 => (i as f32 * 0.005).sin(),
                3 => 1.0,
                4 => {
                    if i.is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => {
                    if rng.next().is_multiple_of(64) {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            amp * v
        })
        .collect()
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    x.iter()
        .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

fn to_i24(x: &[f32]) -> Vec<i32> {
    x.iter()
        .map(|&v| (v * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32)
        .collect()
}

/// Encodes `pcm` in `fmt` (0: int16, 1: int24, 2: float).
fn encode(enc: &mut Encoder, fmt: u32, pcm: &[f32], n: usize, out: &mut [u8]) -> Option<usize> {
    let r = match fmt {
        0 => enc.encode(&to_i16(pcm), n, out),
        1 => enc.encode24(&to_i24(pcm), n, out),
        _ => enc.encode_float(pcm, n, out),
    };
    r.ok()
}

#[test]
fn lfe_stereo_encoder() {
    let mut rng = FastRand::new(0x1fe);
    let mut packets = 0;
    for kind in 0..6 {
        for &bitrate in &[16_000, 64_000, 256_000] {
            for fmt in 0..3 {
                let mut enc = Encoder::new(48000, 2, Application::Audio).unwrap();
                enc.ctl_set(req::OPUS_SET_LFE_REQUEST, 1).unwrap();
                enc.ctl_set(req::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
                let mut dec = Decoder::new(48000, 2).unwrap();
                let pcm = signal(&mut rng, kind, 960 * 10, 2, 1.0);
                for f in pcm.chunks(960 * 2) {
                    let mut buf = [0u8; 1500];
                    let len = encode(&mut enc, fmt, f, 960, &mut buf).unwrap();
                    let mut out = [0i16; 960 * 2];
                    dec.decode(Some(&buf[..len]), &mut out, 960, false).unwrap();
                    assert_eq!(enc.final_range(), dec.final_range());
                    packets += 1;
                }
            }
        }
    }
    assert_eq!(packets, 6 * 3 * 3 * 10);
}

/// Decoders at the extreme `OPUS_SET_GAIN` values (±128 dB) on loud packets, every output
/// format (the 24-bit fixed-point gain product exceeds 32 bits).
#[test]
fn decoder_max_gain() {
    let mut rng = FastRand::new(0x9a1);
    let mut enc = Encoder::new(48000, 2, Application::Audio).unwrap();
    enc.ctl_set(req::OPUS_SET_BITRATE_REQUEST, 128_000).unwrap();
    let pcm = signal(&mut rng, 0, 960 * 8, 2, 1.0);
    let mut packets = Vec::new();
    for f in pcm.chunks(960 * 2) {
        let mut buf = [0u8; 1500];
        let len = encode(&mut enc, 2, f, 960, &mut buf).unwrap();
        packets.push(buf[..len].to_vec());
    }
    for gain in [32767, -32768] {
        let mut dec = Decoder::new(48000, 2).unwrap();
        dec.set_gain(gain).unwrap();
        for p in &packets {
            let mut a = [0i16; 1920];
            let mut b = [0i32; 1920];
            let mut c = [0f32; 1920];
            assert_eq!(dec.decode(Some(p), &mut a, 960, false).unwrap(), 960);
            assert_eq!(dec.decode24(Some(p), &mut b, 960, false).unwrap(), 960);
            assert_eq!(dec.decode_float(Some(p), &mut c, 960, false).unwrap(), 960);
        }
    }
}

/// A random encoder CTL.
const fn random_ctl(rng: &mut FastRand) -> (i32, i32) {
    let v = rng.next();
    match rng.next() % 15 {
        0 => (
            req::OPUS_SET_BITRATE_REQUEST,
            rng.sample(&[500, 6000, 16000, 32000, 64000, 128_000, 510_000, -1000, -1]),
        ),
        1 => (req::OPUS_SET_COMPLEXITY_REQUEST, (v % 11) as i32),
        2 => (req::OPUS_SET_VBR_REQUEST, (v % 2) as i32),
        3 => (req::OPUS_SET_VBR_CONSTRAINT_REQUEST, (v % 2) as i32),
        4 => (
            req::OPUS_SET_FORCE_CHANNELS_REQUEST,
            rng.sample(&[-1000, 1, 2]),
        ),
        5 => (
            req::OPUS_SET_BANDWIDTH_REQUEST,
            rng.sample(&[-1000, 1101, 1102, 1103, 1104, 1105]),
        ),
        6 => (
            req::OPUS_SET_SIGNAL_REQUEST,
            rng.sample(&[-1000, 3001, 3002]),
        ),
        7 => (req::OPUS_SET_INBAND_FEC_REQUEST, (v % 3) as i32),
        8 => (req::OPUS_SET_PACKET_LOSS_PERC_REQUEST, (v % 101) as i32),
        9 => (req::OPUS_SET_DTX_REQUEST, (v % 2) as i32),
        10 => (req::OPUS_SET_LSB_DEPTH_REQUEST, 8 + (v % 17) as i32),
        11 => (req::OPUS_SET_PREDICTION_DISABLED_REQUEST, (v % 2) as i32),
        12 => (req::OPUS_SET_LFE_REQUEST, (v % 2) as i32),
        13 => (
            req::OPUS_SET_FORCE_MODE_REQUEST,
            rng.sample(&[-1000, 1000, 1001, 1002]),
        ),
        _ => (
            req::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            (v % 2) as i32,
        ),
    }
}

#[test]
fn random_public_api() {
    // `OPUSORUS_API_SWEEP=<n>` multiplies the case count (as the C-comparing sweep).
    let scale: usize = match std::env::var("OPUSORUS_API_SWEEP") {
        Ok(s) => s.trim().parse().expect("OPUSORUS_API_SWEEP: number"),
        Err(_) => 1,
    };
    let cases = scale * if common::full_length() { 4000 } else { 300 };
    let mut rng = FastRand::new(common::seed());
    let mut rates = vec![8000, 12000, 16000, 24000, 48000];
    if cfg!(feature = "qext") {
        rates.push(96000);
    }
    let apps = [
        Application::Voip,
        Application::Audio,
        Application::RestrictedLowDelay,
        Application::RestrictedSilk,
        Application::RestrictedCelt,
    ];
    let mut encoded = 0;
    let mut ctls_ok = 0;
    for _ in 0..cases {
        let fs = rng.sample(&rates);
        let ch = 1 + (rng.next() % 2) as i32;
        let chu = ch as usize;
        let mut enc = Encoder::new(fs, ch, rng.sample(&apps)).unwrap();
        for _ in 0..rng.next() % 8 {
            let (r, v) = random_ctl(&mut rng);
            // Some combinations are rejected (e.g. two forced channels on a mono encoder).
            ctls_ok += usize::from(enc.ctl_set(r, v).is_ok());
        }
        let mut dec = Decoder::new(rng.sample(&rates), 1 + (rng.next() % 2) as i32).unwrap();
        let gain = rng.sample(&[0, -32768, 32767, 1536, -1536]);
        dec.set_gain(gain).unwrap();
        let dfs = dec.sample_rate() as usize;
        let dch = dec.channels();
        let n = fs as usize * rng.sample(&[1, 2, 4, 8, 16, 24]) / 400;
        let fmt = rng.next() % 3;
        let amp = rng.sample(&[1.0, 1.0, 0.5, 3.0, 1e-4]);
        let kind = rng.next() % 6;
        let pcm = signal(&mut rng, kind, n * 8, chu, amp);
        for f in pcm.chunks(n * chu) {
            let mut buf = [0u8; 1500];
            let Some(len) = encode(&mut enc, fmt, f, n, &mut buf) else {
                continue;
            };
            encoded += 1;
            let mut pkt = buf[..len].to_vec();
            if rng.next().is_multiple_of(5) && !pkt.is_empty() {
                let k = (rng.next() as usize) % pkt.len();
                pkt[k] ^= 1 << (rng.next() % 8);
            }
            let max = dfs * 120 / 1000;
            let data = if rng.next().is_multiple_of(8) {
                None
            } else {
                Some(&pkt[..])
            };
            let frame = if data.is_none() { dfs / 100 } else { max };
            let fec = data.is_some() && rng.next().is_multiple_of(6);
            // Errors (corrupted packets) are fine; panics are not.
            let ok = match rng.next() % 3 {
                0 => {
                    let mut out = vec![0i16; max * dch];
                    dec.decode(data, &mut out, frame, fec).is_ok()
                }
                1 => {
                    let mut out = vec![0i32; max * dch];
                    dec.decode24(data, &mut out, frame, fec).is_ok()
                }
                _ => {
                    let mut out = vec![0f32; max * dch];
                    dec.decode_float(data, &mut out, frame, fec).is_ok()
                }
            };
            assert!(ok || data.is_some());
        }
    }
    assert!(encoded > cases && ctls_ok > cases);
}

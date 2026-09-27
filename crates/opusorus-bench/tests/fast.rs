//! Acceptance tests of the `fast` feature (PLAN D-032): opusorus with the non-bit-exact kernels
//! must stay as close to the reference (the scalar C oracle) as libopus' own optimized build.
//!
//! * Decoding: the same packets (from the oracle's encoder) decoded by opusorus `fast`, the
//!   oracle and the optimized C build; the SNR of opusorus against the oracle must be at least
//!   [`CLOSE_DB`] or at most [`MARGIN_DB`] below the optimized build's. Covers every codec
//!   configuration, classic PLC under 20 % loss and, with `dnn`, the deep PLC and LACE/NoLACE.
//! * Encoding: the packets of each implementation decoded by the oracle's decoder; the
//!   delay-aligned SNR against the input of opusorus `fast` must be within [`ENCODE_MARGIN_DB`]
//!   of the lower of the two C encoders' (encoder decisions may differ, quality may not).
//!
//! Run with `cargo test -p opusorus-bench --features fast[,dnn,qext] --test fast -- --nocapture`
//! to see the table.
#![cfg(feature = "fast")]
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    reason = "test code: failures should panic; the SNR table goes to stdout"
)]

use opusorus_bench::{
    AnyDecoder, CodecConfig, Impl, MAX_PACKET, Sample, codec_configs, encode_all,
};
use opusorus_oracle::{api, sys};

/// Deviations this small are accepted regardless of the optimized build.
const CLOSE_DB: f64 = 90.0;
/// How much further from the reference than the optimized C build opusorus may be.
const MARGIN_DB: f64 = 6.0;
/// Encoder quality margin against the C encoders.
const ENCODE_MARGIN_DB: f64 = 1.0;

/// Signal-to-difference ratio in dB of `b` against `a`.
fn snr_db(a: &[f32], b: &[f32]) -> f64 {
    let (mut s, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        s += x * x;
        n += (x - y) * (x - y);
    }
    if n == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / n).log10()
    }
}

/// Decodes `packets` with `imp` at decoder `complexity`, losing every 5th packet if `lossy`.
fn decode(
    imp: Impl,
    cfg: &CodecConfig,
    packets: &[Vec<u8>],
    complexity: i32,
    lossy: bool,
) -> Vec<f32> {
    let mut dec = AnyDecoder::new(imp, cfg.fs, cfg.channels).unwrap();
    #[cfg(feature = "dnn")]
    if let AnyDecoder::Rust(d) = &mut dec {
        d.set_dnn_blob(opusorus_oracle::dnn_integration::decoder_blob())
            .unwrap();
    }
    dec.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, complexity)
        .unwrap();
    let fsz = cfg.frame_size();
    let mut pcm: Vec<Sample> = vec![0.0; fsz * cfg.channels as usize];
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let lost = lossy && i % 5 == 4;
        let n = dec
            .decode_opt((!lost).then_some(p.as_slice()), &mut pcm, fsz)
            .unwrap();
        out.extend_from_slice(&pcm[..n * cfg.channels as usize]);
    }
    out
}

/// Checks the decoding criterion and prints a table row.
fn check_decode(what: &str, rust: f64, opt: f64) {
    println!("{what:<48} rust {rust:>7.1} dB   c_opt {opt:>7.1} dB");
    assert!(
        rust >= CLOSE_DB || rust >= opt - MARGIN_DB,
        "{what}: opusorus fast is {rust:.1} dB from the reference, the optimized C build {opt:.1} dB"
    );
}

fn decode_case(cfg: &CodecConfig, complexity: i32, lossy: bool) {
    let packets = encode_all(Impl::CScalar, cfg, 10).unwrap();
    let reference = decode(Impl::CScalar, cfg, &packets, complexity, lossy);
    let rust = decode(Impl::Rust, cfg, &packets, complexity, lossy);
    let opt = decode(Impl::COpt, cfg, &packets, complexity, lossy);
    let what = format!(
        "decode {} cx{complexity}{}",
        cfg.id,
        if lossy { " loss20" } else { "" }
    );
    check_decode(&what, snr_db(&reference, &rust), snr_db(&reference, &opt));
}

#[test]
fn decoding_close_to_reference() {
    for cfg in codec_configs() {
        decode_case(&cfg, 10, false);
        // Classic PLC (complexity < 5 disables the deep PLC).
        decode_case(&cfg, 0, true);
    }
}

#[cfg(feature = "dnn")]
#[test]
fn dnn_decoding_close_to_reference() {
    let cfgs = codec_configs();
    for id in ["silk_16k_mono_16k_voip", "hybrid_48k_mono_32k"] {
        let cfg = cfgs.iter().find(|c| c.id == id).unwrap();
        // Deep PLC under loss; LACE and NoLACE (SILK only) without.
        decode_case(cfg, 5, true);
        decode_case(cfg, 6, false);
        decode_case(cfg, 7, false);
    }
}

/// Delay-aligned SNR of the oracle's decoding of `packets` against the input.
fn encode_quality(cfg: &CodecConfig, packets: &[Vec<u8>], lookahead: usize) -> f64 {
    let out = decode(Impl::CScalar, cfg, packets, 10, false);
    let input = cfg.samples();
    let d = lookahead * cfg.channels as usize;
    snr_db(&input[..input.len() - d], &out[d..])
}

#[test]
fn encoding_quality_matches_c() {
    for cfg in codec_configs() {
        let lookahead = {
            let mut e = api::Encoder::new(cfg.fs, cfg.channels, cfg.application).unwrap();
            e.ctl_get(sys::OPUS_GET_LOOKAHEAD_REQUEST).unwrap() as usize
        };
        for cx in [10, 5] {
            let q = |imp| {
                let packets = encode_all(imp, &cfg, cx).unwrap();
                assert!(
                    packets
                        .iter()
                        .all(|p| !p.is_empty() && p.len() <= MAX_PACKET)
                );
                encode_quality(&cfg, &packets, lookahead)
            };
            let (rust, scalar, opt) = (q(Impl::Rust), q(Impl::CScalar), q(Impl::COpt));
            println!(
                "encode {:<30} cx{cx:<2}  rust {rust:>6.2} dB   c_scalar {scalar:>6.2} dB   c_opt {opt:>6.2} dB",
                cfg.id
            );
            assert!(
                rust >= scalar.min(opt) - ENCODE_MARGIN_DB,
                "encode {} cx{cx}: opusorus fast {rust:.2} dB vs C {scalar:.2} / {opt:.2} dB",
                cfg.id
            );
        }
    }
}

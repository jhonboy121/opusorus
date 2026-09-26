//! Encoder robustness: arbitrary PCM (synthesized or raw bit patterns, including NaN/Inf
//! floats and out-of-range 24-bit values) in all three input formats, any frame size and output
//! buffer size, and arbitrary CTL sequences. Must never panic; every packet must be within the
//! buffer, parse, have the expected duration and decode.
//!
//! Input: see `opusorus_fuzz::enc`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::encoder::request::{OPUS_GET_EXPERT_FRAME_DURATION_REQUEST, OPUS_RESET_STATE};
use opusorus::{Decoder, Encoder, Error, packet};
use opusorus_fuzz::enc::{self, EncOp, Synth, rust_encode};
use opusorus_fuzz::{Reader, code};

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let h = enc::read_header(&mut r);
    let Ok(mut e) = Encoder::new_raw(h.fs, h.channels, h.app) else {
        assert!(
            !enc::APPS[..5].contains(&h.app),
            "valid configuration rejected"
        );
        return;
    };
    let mut d = Decoder::new(h.fs, h.channels).expect("valid configuration");
    let ch = h.channels as usize;
    let mut synth = Synth::new();
    while let Some(op) = enc::read_op(&mut r, &h, &mut synth) {
        match op {
            EncOp::Frame {
                pcm,
                frame_size,
                max_bytes,
            } => {
                let mut out = vec![0u8; max_bytes];
                match rust_encode(&mut e, &pcm, frame_size, &mut out) {
                    Ok(n) => {
                        assert!(
                            n >= 1 && n <= max_bytes,
                            "packet length {n} (max {max_bytes})"
                        );
                        let p = &out[..n];
                        let parsed = packet::parse(p).expect("encoder output parses");
                        assert!(parsed.nb_frames >= 1);
                        let dur = packet::get_nb_samples(p, h.fs).expect("valid duration");
                        let dur = dur as usize;
                        if e.ctl_get(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST) == Ok(5000) {
                            assert_eq!(dur, frame_size, "packet duration vs frame size");
                        } else {
                            assert!(dur <= frame_size, "packet longer than the input");
                        }
                        let mut pcm_out = vec![0.0f32; dur * ch];
                        assert_eq!(
                            d.decode_float(Some(p), &mut pcm_out, dur, false),
                            Ok(dur),
                            "encoder output decodes"
                        );
                    }
                    Err(err) => assert!(
                        err == Error::BadArg.code() || err == Error::BufferTooSmall.code(),
                        "encode error {err}"
                    ),
                }
            }
            EncOp::Ctl { req, value } => {
                if let Err(err) = code(e.ctl_set(req, value)) {
                    assert!(
                        err == Error::BadArg.code() || err == Error::Unimplemented.code(),
                        "ctl_set({req}, {value}) -> {err}"
                    );
                }
                if req == OPUS_RESET_STATE {
                    d.reset();
                }
            }
        }
        assert_eq!(e.sample_rate(), h.fs);
        assert!(e.lookahead() > 0);
        assert!((0..=10).contains(&e.complexity()));
    }
});

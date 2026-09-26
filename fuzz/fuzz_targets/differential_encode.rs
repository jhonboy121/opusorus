//! Differential encoding: an opusorus `Encoder` and a C libopus `OpusEncoder` get the same
//! configuration, the same CTL changes (bitrate, bandwidth, VBR, complexity, FEC, DTX, forced
//! mode/channels, signal, LSB depth, expert frame duration, prediction, phase inversion, LFE,
//! application, QEXT, resets, invalid values) and the same PCM (synthesized or raw, as 16-bit,
//! 24-bit or float, any valid or invalid frame size, any output-buffer size). Every return
//! code, packet byte, the final range and every GET CTL must be identical.
//!
//! Input: see `opusorus_fuzz::enc`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::Encoder;
use opusorus_fuzz::enc::{self, ENC_GETS, EncOp, Synth, c_encode, rust_encode};
use opusorus_fuzz::{Reader, assert_slice_eq, code};
use opusorus_oracle::api as capi;

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let h = enc::read_header(&mut r);
    let (mut re, mut ce) = match (
        code(Encoder::new_raw(h.fs, h.channels, h.app)),
        capi::Encoder::new(h.fs, h.channels, h.app),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "create error");
            return;
        }
        (a, b) => panic!("create mismatch: {:?} vs {:?}", a.err(), b.err()),
    };
    let mut synth = Synth::new();
    let mut i = 0;
    while let Some(op) = enc::read_op(&mut r, &h, &mut synth) {
        let what = format!("op {i}");
        match op {
            EncOp::Frame {
                pcm,
                frame_size,
                max_bytes,
            } => {
                let mut a = vec![0xA5u8; max_bytes];
                let mut b = a.clone();
                let rr = rust_encode(&mut re, &pcm, frame_size, &mut a);
                let cr = c_encode(&mut ce, &pcm, frame_size, &mut b);
                assert_eq!(
                    rr, cr,
                    "{what}: encode(n {frame_size}, max {max_bytes}) return"
                );
                // Bytes past the packet are scratch space for both encoders (unspecified).
                if let Ok(n) = rr {
                    assert_slice_eq(&format!("{what}: packet"), &a[..n], &b[..n]);
                    assert_eq!(re.final_range(), ce.final_range().expect("C GET"), "{what}");
                }
            }
            EncOp::Ctl { req, value } => {
                assert_eq!(
                    code(re.ctl_set(req, value)),
                    ce.ctl_set(req, value),
                    "{what}: ctl_set({req}, {value})"
                );
            }
        }
        for req in ENC_GETS {
            assert_eq!(
                code(re.ctl_get(req)),
                ce.ctl_get(req),
                "{what}: ctl_get({req})"
            );
        }
        i += 1;
    }
});

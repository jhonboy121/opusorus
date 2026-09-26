//! Encoder -> decoder invariants (the properties libopus' own test_opus_encode checks, and a
//! few more):
//!
//! * every packet the encoder produces decodes, at the encoder's rate and channel count and
//!   at another rate / channel count, to exactly its duration;
//! * `get_nb_samples` / `get_nb_frames` / `get_samples_per_frame` agree with each other, with
//!   the decoder and with the frame size;
//! * the decoder's final range equals the encoder's (also for the second decoder);
//! * FEC decoding of the packet (as the redundancy for a lost previous frame) succeeds;
//! * padding the packet, re-packetizing it, or (without extensions) unpadding it yields a
//!   packet that decodes bit-identically on an identical decoder.
//!
//! Input: `alt:u8` (second decoder: rate and channels) followed by the input of
//! `opusorus_fuzz::enc`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::{Decoder, Encoder, Error, Repacketizer, extensions, packet, repacketizer};
use opusorus_fuzz::enc::{self, EncOp, Synth, rust_encode};
use opusorus_fuzz::{Reader, assert_f32_bits_eq, pick_fs};

/// A transformed but equivalent version of packet `p` (number `k` of the stream).
fn transform(p: &[u8], k: usize) -> Vec<u8> {
    let parsed = packet::parse(p).expect("encoder output parses");
    let n_ext = extensions::count(parsed.padding, parsed.nb_frames as i32);
    match k % 3 {
        0 => {
            let extra = 1 + (k * 37) % 300;
            let mut buf = p.to_vec();
            buf.resize(p.len() + extra, 0);
            repacketizer::packet_pad(&mut buf, p.len(), p.len() + extra).expect("pad succeeds");
            buf
        }
        1 => {
            let mut rp = Repacketizer::new();
            rp.cat(p).expect("cat of an encoder packet");
            let mut buf = vec![0u8; p.len() + 64];
            let n = rp.out(&mut buf).expect("repacketizer out");
            buf.truncate(n);
            buf
        }
        _ if n_ext == 0 => {
            let mut buf = p.to_vec();
            let n = repacketizer::packet_unpad(&mut buf).expect("unpad succeeds");
            assert!(n <= p.len());
            buf.truncate(n);
            buf
        }
        _ => p.to_vec(),
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let alt = r.u8();
    let h = enc::read_header(&mut r);
    let Ok(mut e) = Encoder::new_raw(h.fs, h.channels, h.app) else {
        return;
    };
    let ch = h.channels as usize;
    let fs2 = pick_fs(alt);
    let ch2 = 1 + usize::from((alt >> 4) & 1);
    let mut d1 = Decoder::new(h.fs, h.channels).expect("valid configuration");
    let mut d2 = Decoder::new(fs2, ch2 as i32).expect("valid configuration");
    let mut d3 = Decoder::new(h.fs, h.channels).expect("valid configuration");
    let mut d4 = Decoder::new(h.fs, h.channels).expect("valid configuration");
    let mut synth = Synth::new();
    let mut k = 0usize;
    while let Some(op) = enc::read_op(&mut r, &h, &mut synth) {
        let (pcm, frame_size, max_bytes) = match op {
            EncOp::Frame {
                pcm,
                frame_size,
                max_bytes,
            } => (pcm, frame_size, max_bytes),
            EncOp::Ctl { req, value } => {
                if let Err(err) = e.ctl_set(req, value) {
                    assert!(matches!(err, Error::BadArg | Error::Unimplemented));
                }
                continue;
            }
        };
        let mut out = vec![0u8; max_bytes];
        let Ok(n) = rust_encode(&mut e, &pcm, frame_size, &mut out) else {
            continue;
        };
        let p = &out[..n];
        let what = format!("packet {k} ({n} bytes, toc {:#04x})", p[0]);
        let enc_rng = e.final_range();

        // Duration bookkeeping.
        let dur = packet::get_nb_samples(p, h.fs).expect("valid duration");
        let nf = packet::get_nb_frames(p).expect("frame count");
        let spf = packet::get_samples_per_frame(p, h.fs).expect("samples per frame");
        assert_eq!(
            dur,
            nf * spf,
            "{what}: nb_samples = nb_frames * samples_per_frame"
        );
        assert!(dur as usize <= frame_size, "{what}: longer than the input");
        let dur = dur as usize;
        assert_eq!(d1.nb_samples(p), Ok(dur), "{what}: Decoder::nb_samples");

        // Reference decode.
        let mut out1 = vec![0.0f32; dur * ch];
        assert_eq!(
            d1.decode_float(Some(p), &mut out1, dur, false),
            Ok(dur),
            "{what}: decode"
        );
        assert_eq!(d1.final_range(), enc_rng, "{what}: final range enc vs dec");

        // Another rate / channel count.
        let dur2 = dur * fs2 as usize / h.fs as usize;
        let mut out2 = vec![0i16; dur2 * ch2];
        assert_eq!(
            d2.decode(Some(p), &mut out2, dur2, false),
            Ok(dur2),
            "{what}: decode at {fs2} Hz x{ch2}"
        );
        assert_eq!(
            d2.final_range(),
            enc_rng,
            "{what}: final range, second decoder"
        );

        // FEC for a lost previous frame, then the packet itself.
        let mut out3 = vec![0.0f32; dur * ch];
        assert_eq!(
            d3.decode_float(Some(p), &mut out3, dur, true),
            Ok(dur),
            "{what}: FEC decode"
        );
        assert_eq!(
            d3.decode_float(Some(p), &mut out3, dur, false),
            Ok(dur),
            "{what}: decode after FEC"
        );

        // Equivalent packets decode identically.
        let q = transform(p, k);
        let mut out4 = vec![0.0f32; dur * ch];
        assert_eq!(
            d4.decode_float(Some(&q), &mut out4, dur, false),
            Ok(dur),
            "{what}: transformed decode"
        );
        assert_f32_bits_eq(&format!("{what}: transformed packet output"), &out4, &out1);
        k += 1;
    }
});

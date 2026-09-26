//! Decoder robustness through the public API: arbitrary packet sequences with random sampling
//! rate, channel count, output format, FEC, frame sizes (including invalid ones), too-short
//! output buffers and CTL calls (gain, complexity, phase inversion, ignore-extensions, reset).
//! Must never panic, and must uphold the API contract:
//!
//! * a normal decode returns exactly the packet duration; PLC / FEC return `frame_size`;
//! * `last_packet_duration` matches the returned count;
//! * float output is finite;
//! * CTL errors are only `BadArg` / `Unimplemented`.
//!
//! Input: `fs_sel:u8 ch_sel:u8` then the operation stream of `opusorus_fuzz::dec`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::{Decoder, Error, packet};
use opusorus_fuzz::dec::{self, DecOp, Fmt, MAX_FRAME, frame_size};
use opusorus_fuzz::{Reader, pick_fs};

fn exercise_packet(p: &[u8], fs: i32) {
    // Only "no panic" here; the parser is checked against C in packet_parse_extensions.
    if let Ok(parsed) = packet::parse(p) {
        let frames = parsed.frames();
        assert_eq!(frames.len(), parsed.nb_frames);
        assert_eq!(packet::get_nb_frames(p), Ok(parsed.nb_frames as i32));
    }
    if let Ok(n) = packet::get_nb_samples(p, fs) {
        // 0 for a code-3 packet declaring no frames (as in C).
        assert!(n >= 0 && n <= fs / 25 * 3, "nb_samples {n} out of range");
    }
    if packet::has_lbrr(p) == Ok(true) {
        assert_ne!(
            packet::toc_mode(p[0]),
            packet::MODE_CELT_ONLY,
            "LBRR in a CELT packet"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let ch = 1 + usize::from(r.u8() & 1);
    let mut d = Decoder::new(fs, ch as i32).expect("valid configuration");
    while let Some(op) = dec::read_op(&mut r) {
        match op {
            DecOp::Decode {
                data,
                fmt,
                fec,
                fsel,
                buf,
            } => {
                if let Some(p) = data {
                    exercise_packet(p, fs);
                }
                let n = frame_size(fsel, fs, data);
                if fec == 2 {
                    let mut pcm = vec![0.0f32; n.clamp(1, MAX_FRAME) as usize * ch];
                    assert_eq!(
                        d.opus_decode_float(data, &mut pcm, n, 2),
                        Err(Error::BadArg),
                        "decode_fec=2 must be rejected"
                    );
                    continue;
                }
                let fec = fec == 1;
                // The public API takes usize: negative sizes become huge (and must fail).
                let n_arg = n as isize as usize;
                let need = n.clamp(0, MAX_FRAME) as usize * ch;
                let len = match buf {
                    0 | 1 => need,
                    2 => need.saturating_sub(1),
                    _ => need / 2,
                };
                let res = match fmt {
                    Fmt::I16 => d.decode(data, &mut vec![0; len], n_arg, fec),
                    Fmt::I24 => d.decode24(data, &mut vec![0; len], n_arg, fec),
                    Fmt::F32 => {
                        let mut pcm = vec![0.0f32; len];
                        let res = d.decode_float(data, &mut pcm, n_arg, fec);
                        if let Ok(k) = res {
                            let bad = pcm[..k * ch].iter().position(|v| !v.is_finite());
                            assert!(bad.is_none(), "non-finite output at {bad:?}");
                        }
                        res
                    }
                };
                let Ok(k) = res else {
                    continue;
                };
                assert!(n > 0 && k <= n as usize, "returned {k} for frame_size {n}");
                match data {
                    Some(p) if !p.is_empty() && !fec => {
                        let expect = packet::get_nb_samples(p, fs).expect("decoded packet");
                        assert_eq!(k, expect as usize, "decode count vs packet duration");
                        assert_eq!(d.nb_samples(p), Ok(k), "Decoder::nb_samples");
                    }
                    _ => assert_eq!(k, n as usize, "PLC/FEC count"),
                }
                assert_eq!(d.last_packet_duration(), k, "last_packet_duration");
            }
            DecOp::Ctl { req, value } => {
                if let Err(e) = d.ctl_set(req, value) {
                    assert!(
                        matches!(e, Error::BadArg | Error::Unimplemented),
                        "ctl_set({req}, {value}) -> {e:?}"
                    );
                }
                assert!((-32768..=32767).contains(&d.gain()));
                assert!((0..=10).contains(&d.complexity()));
            }
            DecOp::NbSamples(p) => {
                exercise_packet(p, fs);
                if let Ok(k) = d.nb_samples(p) {
                    assert!(k <= (fs / 25 * 3) as usize);
                }
            }
        }
        assert_eq!(d.sample_rate(), fs);
        assert!(d.pitch() >= 0);
    }
});

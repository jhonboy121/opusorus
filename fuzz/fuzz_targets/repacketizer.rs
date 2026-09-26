//! Repacketizer and padding on arbitrary packets, differential against C libopus, plus
//! structural invariants on every output:
//!
//! * `cat` / `get_nb_frames` / `out_range` / `out` / `init` return the same codes and bytes as C;
//!   a successful output re-parses into exactly the requested frames (byte-identical);
//! * `packet_pad` / `packet_unpad` / `multistream_packet_pad` / `multistream_packet_unpad`
//!   match C byte for byte; padding keeps the frames and yields exactly `new_len` bytes;
//!   unpadding keeps the frames.
//!
//! Input, repeated until the end:
//!
//! ```text
//! tag:u8  tag % 8:
//!   0..=2  cat        len:u16 packet[len]
//!   3      out_range  begin:i8 end:i8 maxlen:u16
//!   4      out        maxlen:u16  (the whole range)
//!   5      init
//!   6      pad+unpad  extra:u16 len:u16 packet[len]
//!   7      ms pad+unpad  nb_streams:u8 extra:u16 len:u16 packet[len]
//! ```

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::{Repacketizer, packet, repacketizer as rp};
use opusorus_fuzz::{Reader, assert_slice_eq, code};
use opusorus_oracle::api as capi;

fn frames_of(p: &[u8]) -> Vec<Vec<u8>> {
    packet::parse(p)
        .expect("valid packet")
        .frames()
        .iter()
        .map(|f| f.to_vec())
        .collect()
}

fn check_pad(p: &[u8], extra: usize) {
    let new_len = p.len() + extra;
    let mut a = p.to_vec();
    a.resize(new_len.max(1), 0);
    let mut b = a.clone();
    let rr = code(rp::packet_pad(&mut a, p.len(), new_len));
    let cr = if p.is_empty() {
        // C returns OPUS_BAD_ARG for len < 1 before touching the buffer.
        Err(opusorus_oracle::sys::OPUS_BAD_ARG)
    } else {
        capi::packet_pad(&mut b, p.len(), new_len)
    };
    assert_eq!(rr, cr, "packet_pad({}, {new_len})", p.len());
    if rr.is_ok() {
        assert_slice_eq("padded packet", &a[..new_len], &b[..new_len]);
        // With new_len == len the packet is returned untouched (and never parsed).
        if extra > 0 {
            let padded = &a[..new_len];
            assert_eq!(
                frames_of(padded),
                frames_of(p),
                "padding changed the frames"
            );
            let parsed = packet::parse(padded).expect("padded packet parses");
            let used: usize = parsed.frames().iter().map(|f| f.len()).sum::<usize>();
            assert!(used <= new_len);
        }
    }
    // Unpad whatever we have (padded or not).
    let len = if rr.is_ok() { new_len } else { p.len() };
    let mut a2 = a[..len].to_vec();
    let mut b2 = a2.clone();
    let ru = code(rp::packet_unpad(&mut a2));
    let cu = if a2.is_empty() {
        Err(opusorus_oracle::sys::OPUS_BAD_ARG)
    } else {
        capi::packet_unpad(&mut b2)
    };
    assert_eq!(ru, cu, "packet_unpad(len {len})");
    if let Ok(n) = ru {
        assert_slice_eq("unpadded packet", &a2[..n], &b2[..n]);
        assert_eq!(
            frames_of(&a2[..n]),
            frames_of(&a[..len]),
            "unpadding changed the frames"
        );
        assert!(packet::parse(&a2[..n]).expect("parses").padding.is_empty());
    }
}

fn check_ms_pad(p: &[u8], extra: usize, nb_streams: i32) {
    let new_len = p.len() + extra;
    let mut a = p.to_vec();
    a.resize(new_len.max(1), 0);
    let mut b = a.clone();
    let rr = code(rp::multistream_packet_pad(
        &mut a,
        p.len(),
        new_len,
        nb_streams,
    ));
    let cr = if p.is_empty() {
        Err(opusorus_oracle::sys::OPUS_BAD_ARG)
    } else {
        capi::multistream_packet_pad(&mut b, p.len(), new_len, nb_streams)
    };
    assert_eq!(
        rr,
        cr,
        "multistream_packet_pad({}, {new_len}, {nb_streams})",
        p.len()
    );
    if rr.is_ok() {
        assert_slice_eq("ms padded packet", &a[..new_len], &b[..new_len]);
    }
    let len = if rr.is_ok() { new_len } else { p.len() };
    let mut a2 = a[..len].to_vec();
    let mut b2 = a2.clone();
    let ru = code(rp::multistream_packet_unpad(&mut a2, nb_streams));
    let cu = if a2.is_empty() {
        Err(opusorus_oracle::sys::OPUS_BAD_ARG)
    } else {
        capi::multistream_packet_unpad(&mut b2, nb_streams)
    };
    assert_eq!(ru, cu, "multistream_packet_unpad(len {len}, {nb_streams})");
    if let Ok(n) = ru {
        assert_slice_eq("ms unpadded packet", &a2[..n], &b2[..n]);
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let mut rrp = Repacketizer::new();
    let mut crp = capi::Repacketizer::new().expect("C allocation");
    // Frames currently held (Rust view), for the structural checks.
    let mut held: Vec<Vec<u8>> = Vec::new();
    while !r.is_empty() {
        match r.u8() % 8 {
            0..=2 => {
                let p = r.blob();
                let rr = code(rrp.cat(p));
                let cr = if p.is_empty() {
                    Err(opusorus_oracle::sys::OPUS_INVALID_PACKET)
                } else {
                    crp.cat(p)
                };
                assert_eq!(rr, cr, "cat(len {})", p.len());
                if rr.is_ok() {
                    held.extend(frames_of(p));
                }
            }
            t @ (3 | 4) => {
                let (begin, end) = if t == 3 {
                    (i32::from(r.u8() as i8), i32::from(r.u8() as i8))
                } else {
                    (0, rrp.nb_frames())
                };
                let maxlen = usize::from(r.u16());
                let mut a = vec![0xEEu8; maxlen];
                let mut b = a.clone();
                let (rr, cr) = if t == 3 {
                    (
                        code(rrp.out_range(begin, end, &mut a)),
                        crp.out_range(begin, end, &mut b),
                    )
                } else {
                    (code(rrp.out(&mut a)), crp.out(&mut b))
                };
                assert_eq!(rr, cr, "out_range({begin}, {end}, {maxlen})");
                if let Ok(n) = rr {
                    assert_slice_eq("out_range bytes", &a[..n], &b[..n]);
                    let out = frames_of(&a[..n]);
                    let want = &held[begin as usize..end as usize];
                    assert_eq!(out.as_slice(), want, "out_range frames");
                }
            }
            5 => {
                rrp.init();
                crp.init();
                held.clear();
            }
            6 => {
                let extra = usize::from(r.u16() % 2048);
                check_pad(r.blob(), extra);
            }
            _ => {
                let nb_streams = i32::from(r.u8() % 12) - 1;
                let extra = usize::from(r.u16() % 2048);
                check_ms_pad(r.blob(), extra, nb_streams);
            }
        }
        assert_eq!(rrp.nb_frames() as usize, crp.nb_frames(), "nb_frames");
        assert_eq!(rrp.nb_frames() as usize, held.len(), "held frames");
    }
});

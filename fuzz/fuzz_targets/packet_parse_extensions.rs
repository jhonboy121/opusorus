//! Packet parsing, TOC helpers, extension parsing / iteration / generation and the soft
//! clipper on arbitrary input, differential against C libopus, plus round trips:
//!
//! * `parse_impl` (normal and self-delimited), `parse`, every TOC helper and `has_lbrr` match C;
//!   parsed frames lie inside the packet;
//! * the extensions in a parsed packet's padding (and arbitrary extension data) give the same
//!   `count` / `count_ext` / `parse` / `parse_ext` / iterator results as C;
//! * `generate` (measure and write, with and without padding) matches C byte for byte on
//!   arbitrary extension lists, and parsing the output gives back the same extensions;
//!   re-generating parsed extensions and parsing again is the identity (as a multiset);
//! * `pcm_soft_clip` matches C bit for bit (including NaN / huge inputs).
//!
//! Input: `mode:u8`, then per mode (`mode % 4`):
//!
//! ```text
//! 0  packet                          packet = rest
//! 1  extensions  nbf:u8 nops:u8 ops[nops]:(u8 u8)   data = rest
//! 2  generate    nbf:u8 flags:u8 len:u16 n:u8 {id:u8 frame:u8 len:u8 payload[len]}*n
//! 3  soft clip   ch:u8 n:u8 mem:(f32 f32) samples = rest (f32 LE)
//! ```

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::extensions::{self, Extension, ExtensionIterator};
use opusorus::packet;
use opusorus_fuzz::{Reader, assert_f32_bits_eq, assert_slice_eq, code};
use opusorus_oracle::api;
use opusorus_oracle::opus_packet::{self as c, Ext, IterOp};

const fn flat(r: Result<i32, i32>) -> i32 {
    match r {
        Ok(v) | Err(v) => v,
    }
}

fn offset_of(base: &[u8], s: &[u8]) -> usize {
    s.as_ptr() as usize - base.as_ptr() as usize
}

fn ext_to_c(base: &[u8], e: &Extension<'_>) -> Ext {
    Ext {
        id: e.id,
        frame: e.frame,
        off: offset_of(base, e.data) as i32,
        len: e.data.len() as i32,
    }
}

fn check_packet(data: &[u8]) {
    for sd in [false, true] {
        let r = code(packet::parse_impl(data, sd));
        let cc = c::parse_impl(data, sd);
        match (&r, &cc) {
            (Err(a), Err(b)) => assert_eq!(a, b, "parse_impl err sd={sd}"),
            (Ok(p), Ok(q)) => {
                assert_eq!(p.toc, q.toc);
                let frames: Vec<(usize, usize)> = p
                    .frames()
                    .iter()
                    .map(|f| (offset_of(data, f), f.len()))
                    .collect();
                assert_eq!(frames, q.frames, "frames sd={sd}");
                for &(o, l) in &frames {
                    assert!(o + l <= data.len(), "frame outside the packet");
                }
                assert_eq!(p.payload_offset, q.payload_offset, "payload_offset");
                assert_eq!(p.packet_offset, q.packet_offset, "packet_offset");
                assert!(p.packet_offset <= data.len());
                assert_eq!(
                    (offset_of(data, p.padding), p.padding.len()),
                    q.padding,
                    "padding"
                );
                // The extensions in the padding, as the decoder / repacketizer see them.
                check_ext_parsers(p.padding, p.nb_frames as i32);
                regenerate(p.padding, p.nb_frames as i32);
            }
            _ => panic!("parse_impl mismatch sd={sd}: rust={r:?} c={cc:?}"),
        }
        if !sd {
            let pub_r = code(packet::parse(data)).map(|p| p.nb_frames);
            let api_r = api::packet_parse(data).map(|p| p.frames.len());
            assert_eq!(pub_r, api_r, "parse");
        }
    }
    // TOC helpers.
    assert_eq!(
        code(packet::get_nb_frames(data)),
        api::packet_get_nb_frames(data).map(|n| n as i32),
        "get_nb_frames"
    );
    for fs in [8000, 12000, 16000, 24000, 48000, 96000] {
        assert_eq!(
            code(packet::get_nb_samples(data, fs)),
            api::packet_get_nb_samples(data, fs).map(|n| n as i32),
            "get_nb_samples fs={fs}"
        );
    }
    if data.is_empty() {
        assert_eq!(code(packet::get_bandwidth(data)), Err(-1));
        return;
    }
    assert_eq!(
        packet::get_bandwidth(data).map(opusorus::Bandwidth::to_raw),
        Ok(api::packet_get_bandwidth(data)),
        "get_bandwidth"
    );
    assert_eq!(
        packet::get_nb_channels(data),
        Ok(api::packet_get_nb_channels(data)),
        "get_nb_channels"
    );
    for fs in [8000, 12000, 16000, 24000, 48000] {
        assert_eq!(
            packet::get_samples_per_frame(data, fs),
            Ok(api::packet_get_samples_per_frame(data, fs)),
            "get_samples_per_frame"
        );
    }
    assert_eq!(
        code(packet::has_lbrr(data)).map(i32::from),
        api::packet_has_lbrr(data).map(|n| n as i32),
        "has_lbrr"
    );
}

fn check_ext_parsers(data: &[u8], nb_frames: i32) {
    let count = extensions::count(data, nb_frames);
    assert_eq!(
        count,
        c::ext_count(data, nb_frames),
        "count nbf={nb_frames}"
    );
    let full = count.max(0) as usize;
    // The C wrapper supports at most 4096 output slots.
    for cap in [0usize, full.saturating_sub(1), full, full + 3].map(|c| c.min(4096)) {
        let mut out = vec![Extension::default(); cap];
        let r = code(extensions::parse(data, &mut out, nb_frames));
        let (cret, cn, cexts) = c::ext_parse(data, nb_frames, cap);
        match r {
            Ok(n) => {
                assert_eq!(cret, 0, "parse ret cap={cap}");
                assert_eq!(n as i32, cn, "parse count cap={cap}");
                let rexts: Vec<Ext> = out[..n].iter().map(|e| ext_to_c(data, e)).collect();
                assert_eq!(rexts, cexts, "parse cap={cap}");
            }
            Err(e) => assert_eq!(e, cret, "parse err cap={cap}"),
        }
    }
    if (0..=48).contains(&nb_frames) {
        let mut per = vec![0i32; nb_frames as usize];
        let n = extensions::count_ext(data, &mut per);
        let (cn, cper) = c::ext_count_ext(data, nb_frames);
        assert_eq!(n, cn, "count_ext");
        assert_eq!(per, cper, "count_ext per frame");
        for cap in [full, full + 1, full.saturating_sub(1)].map(|c| c.min(4096)) {
            let mut out = vec![Extension::default(); cap];
            let r = code(extensions::parse_ext(data, &mut out, &per));
            let (cret, cn, cexts) = c::ext_parse_ext(data, &cper, cap);
            match r {
                Ok(n) => {
                    assert_eq!(cret, 0, "parse_ext ret");
                    assert_eq!(n as i32, cn, "parse_ext count");
                    let rexts: Vec<Ext> = out[..n].iter().map(|e| ext_to_c(data, e)).collect();
                    assert_eq!(rexts, cexts, "parse_ext");
                }
                Err(e) => assert_eq!(e, cret, "parse_ext err"),
            }
        }
    }
}

fn check_iter_ops(data: &[u8], nb_frames: i32, ops: &[IterOp]) {
    let cres = c::ext_iter_ops(data, nb_frames, ops);
    let mut it = ExtensionIterator::new(data, nb_frames);
    for (op, (cret, cext)) in ops.iter().zip(cres) {
        let r = match *op {
            IterOp::Next => it.next_extension(),
            IterOp::Find(id) => it.find(id),
            IterOp::Reset => {
                it.reset();
                Ok(None)
            }
            IterOp::SetFrameMax(v) => {
                it.set_frame_max(v);
                Ok(None)
            }
        };
        match r {
            Ok(Some(e)) => {
                assert_eq!(cret, 1, "iter op {op:?}");
                assert_eq!(ext_to_c(data, &e), cext, "iter op {op:?}");
            }
            Ok(None) => assert_eq!(cret, 0, "iter op {op:?}"),
            Err(e) => assert_eq!(e.code(), cret, "iter op {op:?}"),
        }
    }
}

/// Sorted (frame, id, payload) multiset.
fn multiset(exts: &[Extension<'_>]) -> Vec<(i32, i32, Vec<u8>)> {
    let mut v: Vec<_> = exts
        .iter()
        .map(|e| (e.frame, e.id, e.data.to_vec()))
        .collect();
    v.sort();
    v
}

/// Parses the extensions of `data` and, if that works, re-generates and re-parses them.
fn regenerate(data: &[u8], nb_frames: i32) {
    if !(1..=48).contains(&nb_frames) {
        return;
    }
    let n = extensions::count(data, nb_frames).max(0) as usize;
    let mut exts = vec![Extension::default(); n];
    let Ok(n) = extensions::parse(data, &mut exts, nb_frames) else {
        return;
    };
    let exts = &exts[..n];
    let size = extensions::generated_size(65536, exts, nb_frames, false)
        .expect("parsed extensions can be generated");
    let mut out = vec![0u8; size];
    assert_eq!(
        extensions::generate(&mut out, exts, nb_frames, false),
        Ok(size)
    );
    let mut back = vec![Extension::default(); n];
    assert_eq!(
        extensions::parse(&out, &mut back, nb_frames),
        Ok(n),
        "re-parse of generated extensions"
    );
    assert_eq!(multiset(&back), multiset(exts), "generate/parse round trip");
}

fn check_generate(r: &mut Reader<'_>) {
    let nb_frames = i32::from(r.u8() % 50);
    let flags = r.u8();
    let pad = flags & 1 != 0;
    let len = i32::from(r.u16() % 4000);
    let n = usize::from(r.u8() % 64);
    // Payload bytes live in one buffer so C gets (offset, len) pairs.
    let mut buf = Vec::new();
    let mut exts = Vec::with_capacity(n);
    for _ in 0..n {
        let id = i32::from(r.u8());
        let frame = i32::from(r.u8() as i8);
        let plen = usize::from(r.u8());
        let payload = r.bytes(plen);
        exts.push(Ext {
            id,
            frame,
            off: buf.len() as i32,
            len: payload.len() as i32,
        });
        buf.extend_from_slice(payload);
    }
    let rexts: Vec<Extension<'_>> = exts
        .iter()
        .map(|e| Extension {
            id: e.id,
            frame: e.frame,
            data: &buf[e.off as usize..(e.off + e.len) as usize],
        })
        .collect();
    // Measure only.
    let rr = code(extensions::generate_impl(None, len, &rexts, nb_frames, pad));
    let cr = c::ext_generate(None, len, &exts, &buf, nb_frames, pad);
    assert_eq!(flat(rr), cr, "generate(NULL) len={len}");
    // Write.
    let init = flags >> 1;
    let mut out_r = vec![init; len as usize];
    let mut out_c = out_r.clone();
    let rr = code(extensions::generate(&mut out_r, &rexts, nb_frames, pad)).map(|n| n as i32);
    let cr = c::ext_generate(Some(&mut out_c), len, &exts, &buf, nb_frames, pad);
    assert_eq!(flat(rr), cr, "generate len={len}");
    assert_slice_eq("generate bytes", &out_r, &out_c);
    let sz = code(extensions::generated_size(
        len as usize,
        &rexts,
        nb_frames,
        pad,
    ));
    assert_eq!(flat(sz.map(|n| n as i32)), cr, "generated_size");
    if cr >= 0 && (1..=48).contains(&nb_frames) {
        let data = &out_c[..cr as usize];
        assert_eq!(
            extensions::count(data, nb_frames),
            n as i32,
            "round-trip count"
        );
        let mut got = vec![Extension::default(); n];
        let k = extensions::parse(data, &mut got, nb_frames).expect("round-trip parse");
        assert_eq!(k, n);
        assert_eq!(multiset(&got), multiset(&rexts), "round trip");
        check_ext_parsers(data, nb_frames);
    }
}

fn check_soft_clip(r: &mut Reader<'_>) {
    let ch = usize::from(r.u8() % 3);
    let n = usize::from(r.u8());
    let mut mem_r = [f32::from_bits(r.u32()), f32::from_bits(r.u32())];
    let mut mem_c = mem_r;
    let mut x: Vec<f32> = (0..n * ch).map(|_| f32::from_bits(r.u32())).collect();
    // Keep most samples in a sane range so the clipper's interesting branches are reached.
    for v in x.iter_mut().step_by(2) {
        if v.is_finite() {
            *v = v.clamp(-4.0, 4.0);
        }
    }
    let mut xc = x.clone();
    packet::pcm_soft_clip(&mut x, n, ch, &mut mem_r);
    api::pcm_soft_clip(&mut xc, n, ch, &mut mem_c);
    assert_f32_bits_eq("soft_clip x", &x, &xc);
    assert_f32_bits_eq("soft_clip mem", &mem_r, &mem_c);
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    match r.u8() % 4 {
        0 => {
            let rest = r.bytes(usize::MAX);
            check_packet(rest);
        }
        1 => {
            let nb_frames = i32::from(r.u8() % 49);
            let nops = usize::from(r.u8() % 48);
            let ops: Vec<IterOp> = (0..nops)
                .map(|_| {
                    let (a, b) = (r.u8(), r.u8());
                    match a % 8 {
                        0 => IterOp::Find(i32::from(b)),
                        1 => IterOp::Reset,
                        2 => IterOp::SetFrameMax(i32::from(b) % (nb_frames + 1)),
                        _ => IterOp::Next,
                    }
                })
                .collect();
            let rest = r.bytes(usize::MAX);
            check_ext_parsers(rest, nb_frames);
            check_iter_ops(rest, nb_frames, &ops);
            regenerate(rest, nb_frames);
        }
        2 => check_generate(&mut r),
        _ => check_soft_clip(&mut r),
    }
});

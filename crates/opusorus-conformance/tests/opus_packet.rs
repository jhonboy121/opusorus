//! Differential tests for the `opus_packet` unit: packet parsing / TOC helpers / soft clip
//! (`src/opus.c`), extensions (`src/extensions.c`), repacketizer and pad/unpad
//! (`src/repacketizer.c`), mapping matrix (`src/mapping_matrix.c`), analysis MLP
//! (`src/mlp.c`) and multistream layout helpers (`src/opus_multistream.c`) vs the C oracle.

use opusorus::extensions::{self, Extension, ExtensionIterator};
use opusorus::mapping_matrix as mm;
use opusorus::mlp;
use opusorus::multistream::{self, ChannelLayout};
use opusorus::packet;
use opusorus::repacketizer::{self, Repacketizer};
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq};
use opusorus_oracle::api;
use opusorus_oracle::opus_packet::{self as c, Ext, IterOp};

// ------------------------------------------------------------------------------- helpers

fn code<T>(r: opusorus::Result<T>) -> Result<T, i32> {
    r.map_err(opusorus::Error::code)
}

const fn flat(r: Result<i32, i32>) -> i32 {
    match r {
        Ok(v) | Err(v) => v,
    }
}

fn offset_of(base: &[u8], s: &[u8]) -> usize {
    s.as_ptr() as usize - base.as_ptr() as usize
}

fn exts_to_rust<'a>(exts: &[Ext], buf: &'a [u8]) -> Vec<Extension<'a>> {
    exts.iter()
        .map(|e| Extension {
            id: e.id,
            frame: e.frame,
            data: &buf[e.off as usize..(e.off + e.len) as usize],
        })
        .collect()
}

fn ext_to_c(base: &[u8], e: &Extension<'_>) -> Ext {
    Ext {
        id: e.id,
        frame: e.frame,
        off: offset_of(base, e.data) as i32,
        len: e.data.len() as i32,
    }
}

/// Random size for a frame: mostly small, sometimes large (2-byte size codes).
fn rand_size(rng: &mut Rng, max: i32) -> i32 {
    match rng.range_i32(0, 9) {
        0 => 0,
        1..=6 => rng.range_i32(0, 40.min(max)),
        7 => rng.range_i32(245.min(max), 260.min(max)),
        _ => rng.range_i32(0, max),
    }
}

/// Random extension list for a packet of `nb_frames` frames; payloads point into `buf`
/// (random bytes). Tends to produce repeatable patterns across frames.
fn rand_exts(rng: &mut Rng, nb_frames: i32, buf_len: usize, allow_bad: bool) -> Vec<Ext> {
    let mut exts = Vec::new();
    if nb_frames <= 0 {
        return exts;
    }
    let pattern_len = rng.range_i32(0, 4) as usize;
    let pattern: Vec<(i32, i32)> = (0..pattern_len)
        .map(|_| {
            let id = if rng.range_i32(0, 1) == 0 {
                rng.range_i32(3, 31)
            } else {
                rng.range_i32(32, 127)
            };
            let len = if id < 32 {
                rng.range_i32(0, 1)
            } else {
                rand_len(rng)
            };
            (id, len)
        })
        .collect();
    let mode = rng.range_i32(0, 3);
    let n = rng.range_i32(0, 24);
    let mut push = |rng: &mut Rng, id: i32, frame: i32, len: i32| {
        let len = len.min(buf_len as i32);
        let off = rng.range_i32(0, buf_len as i32 - len);
        exts.push(Ext {
            id,
            frame,
            off,
            len,
        });
    };
    match mode {
        // Same pattern in every frame (fully repeatable).
        0 => {
            for f in 0..nb_frames {
                for &(id, len) in &pattern {
                    let len = if id >= 32 && rng.range_i32(0, 3) == 0 {
                        rand_len(rng)
                    } else {
                        len
                    };
                    push(rng, id, f, len);
                }
            }
        }
        // Pattern in a suffix of frames plus random extras, in frame order.
        1 => {
            let start = rng.range_i32(0, nb_frames - 1);
            for f in 0..nb_frames {
                if f >= start {
                    for &(id, len) in &pattern {
                        push(rng, id, f, len);
                    }
                }
                for _ in 0..rng.range_i32(0, 2) {
                    let id = rng.range_i32(3, 127);
                    let len = if id < 32 {
                        rng.range_i32(0, 1)
                    } else {
                        rand_len(rng)
                    };
                    push(rng, id, f, len);
                }
            }
        }
        // Completely random order.
        _ => {
            for _ in 0..n {
                let id = rng.range_i32(3, 127);
                let len = if id < 32 {
                    rng.range_i32(0, 1)
                } else {
                    rand_len(rng)
                };
                let f = rng.range_i32(0, nb_frames - 1);
                push(rng, id, f, len);
            }
        }
    }
    exts.truncate(200);
    if allow_bad && !exts.is_empty() && rng.range_i32(0, 9) == 0 {
        let i = rng.range_i32(0, exts.len() as i32 - 1) as usize;
        match rng.range_i32(0, 3) {
            0 => exts[i].id = rng.range_i32(-2, 2),
            1 => exts[i].id = rng.range_i32(128, 300),
            2 => exts[i].frame = rng.range_i32(nb_frames, nb_frames + 3),
            _ => {
                exts[i].id = rng.range_i32(3, 31);
                exts[i].off = 0;
                exts[i].len = 2.min(buf_len as i32);
            }
        }
    }
    exts
}

const fn rand_len(rng: &mut Rng) -> i32 {
    match rng.range_i32(0, 9) {
        0 => 0,
        1..=6 => rng.range_i32(0, 12),
        7 => rng.range_i32(250, 520),
        _ => rng.range_i32(0, 300),
    }
}

/// Builds a random packet with TOC `toc`. Returns the packet and its frame count.
fn build_packet(rng: &mut Rng, toc: u8, sd: bool, ext_buf: &[u8]) -> (Vec<u8>, i32) {
    let spf = packet::toc_samples_per_frame(toc, 48000);
    let (count, cbr) = match toc & 3 {
        0 => (1, true),
        1 => (2, true),
        2 => (2, false),
        _ => {
            let maxc = (5760 / spf).min(48);
            let count = if rng.range_i32(0, 4) == 0 {
                rng.range_i32(1, maxc)
            } else {
                rng.range_i32(1, maxc.min(6))
            };
            (count, rng.range_i32(0, 1) == 0)
        }
    };
    let maxsz = if count > 6 { 60 } else { 1275 };
    let mut sizes: Vec<i32> = (0..count).map(|_| rand_size(rng, maxsz)).collect();
    if cbr {
        let s = sizes[0];
        sizes.iter_mut().for_each(|x| *x = s);
    }
    let mut padding: Vec<u8> = Vec::new();
    let mut has_pad = false;
    if toc & 3 == 3 && rng.range_i32(0, 1) == 0 {
        has_pad = true;
        match rng.range_i32(0, 4) {
            0 => {}
            1 => {
                padding = vec![0u8; rng.range_i32(0, 600) as usize];
            }
            2 => {
                padding = vec![0u8; rng.range_i32(0, 40) as usize];
                rng.fill_bytes(&mut padding);
            }
            _ => {
                let exts = rand_exts(rng, count, ext_buf.len(), false);
                let rexts = exts_to_rust(&exts, ext_buf);
                let mut out = vec![0u8; 4000];
                if let Ok(n) = extensions::generate(&mut out, &rexts, count, false) {
                    out.truncate(n);
                    padding = out;
                    if rng.range_i32(0, 3) == 0 {
                        // trailing zeros / garbage
                        padding.extend(std::iter::repeat_n(0u8, rng.range_i32(0, 5) as usize));
                    }
                }
            }
        }
    }
    let mut p = vec![toc];
    match toc & 3 {
        2 => push_size(&mut p, sizes[0]),
        3 => {
            let ch = count as u8 | if cbr { 0 } else { 0x80 } | if has_pad { 0x40 } else { 0 };
            p.push(ch);
            if has_pad {
                let mut n = padding.len();
                while n >= 255 {
                    p.push(255);
                    n -= 254;
                }
                p.push(n as u8);
            }
            if !cbr {
                for &s in &sizes[..count as usize - 1] {
                    push_size(&mut p, s);
                }
            }
        }
        _ => {}
    }
    if sd {
        push_size(&mut p, sizes[count as usize - 1]);
    }
    for &s in &sizes {
        let mut f = vec![0u8; s as usize];
        rng.fill_bytes(&mut f);
        p.extend_from_slice(&f);
    }
    p.extend_from_slice(&padding);
    (p, count)
}

fn push_size(p: &mut Vec<u8>, s: i32) {
    let mut b = [0u8; 2];
    let n = packet::encode_size(s, &mut b);
    p.extend_from_slice(&b[..n]);
}

/// Randomly corrupts a packet (sometimes).
fn mutate(rng: &mut Rng, p: &mut Vec<u8>) {
    match rng.range_i32(0, 9) {
        0 if !p.is_empty() => {
            let i = rng.range_i32(0, p.len() as i32 - 1) as usize;
            p[i] = rng.next_u32() as u8;
        }
        1 if !p.is_empty() => {
            let n = rng.range_i32(0, p.len() as i32 - 1) as usize;
            p.truncate(n);
        }
        2 => {
            for _ in 0..rng.range_i32(1, 4) {
                p.push(rng.next_u32() as u8);
            }
        }
        _ => {}
    }
}

fn ext_buf(rng: &mut Rng) -> Vec<u8> {
    let mut b = vec![0u8; 1024];
    rng.fill_bytes(&mut b);
    b
}

// ------------------------------------------------------------------------------- opus.c

fn check_parse(data: &[u8]) {
    for sd in [false, true] {
        let r = code(packet::parse_impl(data, sd));
        let cc = c::parse_impl(data, sd);
        match (&r, &cc) {
            (Err(a), Err(b)) => assert_eq!(a, b, "parse_impl err sd={sd} {data:02x?}"),
            (Ok(p), Ok(q)) => {
                assert_eq!(p.toc, q.toc);
                let frames: Vec<(usize, usize)> = p
                    .frames()
                    .iter()
                    .map(|f| (offset_of(data, f), f.len()))
                    .collect();
                assert_eq!(frames, q.frames, "frames sd={sd} {data:02x?}");
                assert_eq!(p.payload_offset, q.payload_offset);
                assert_eq!(p.packet_offset, q.packet_offset);
                assert_eq!((offset_of(data, p.padding), p.padding.len()), q.padding);
            }
            _ => panic!("parse_impl mismatch sd={sd}: rust={r:?} c={cc:?} data={data:02x?}"),
        }
        if !sd {
            let pub_r = code(packet::parse(data)).map(|p| p.nb_frames);
            let api_r = api::packet_parse(data).map(|p| p.frames.len());
            assert_eq!(pub_r, api_r);
        }
    }
}

fn check_toc_helpers(data: &[u8]) {
    assert_eq!(
        code(packet::get_nb_frames(data)),
        api::packet_get_nb_frames(data).map(|n| n as i32)
    );
    for fs in [8000, 12000, 16000, 24000, 48000, 96000] {
        assert_eq!(
            code(packet::get_nb_samples(data, fs)),
            api::packet_get_nb_samples(data, fs).map(|n| n as i32),
            "nb_samples fs={fs} {data:02x?}"
        );
    }
    if data.is_empty() {
        assert_eq!(code(packet::get_bandwidth(data)), Err(-1));
        return;
    }
    assert_eq!(
        packet::get_bandwidth(data).map(opusorus::Bandwidth::to_raw),
        Ok(api::packet_get_bandwidth(data))
    );
    assert_eq!(
        packet::get_nb_channels(data),
        Ok(api::packet_get_nb_channels(data))
    );
    for fs in [8000, 12000, 16000, 24000, 48000] {
        assert_eq!(
            packet::get_samples_per_frame(data, fs),
            Ok(api::packet_get_samples_per_frame(data, fs))
        );
    }
    assert_eq!(
        code(packet::has_lbrr(data)).map(i32::from),
        api::packet_has_lbrr(data).map(|n| n as i32),
        "has_lbrr {data:02x?}"
    );
}

#[test]
fn parse_and_toc_random_bytes() {
    let mut rng = Rng::new(0x5041_434b);
    for _ in 0..20000 {
        let n = rng.range_i32(0, 40) as usize;
        let mut p = vec![0u8; n];
        rng.fill_bytes(&mut p);
        check_parse(&p);
        check_toc_helpers(&p);
    }
    // All TOC bytes with a few bodies.
    for toc in 0..=255u8 {
        for body in [
            &[][..],
            &[0u8][..],
            &[1, 2, 3, 4][..],
            &[0x83, 0xff, 5, 6, 7][..],
        ] {
            let mut p = vec![toc];
            p.extend_from_slice(body);
            check_parse(&p);
            check_toc_helpers(&p);
        }
    }
}

#[test]
fn parse_and_toc_structured() {
    let mut rng = Rng::new(0x1234_5678);
    let eb = ext_buf(&mut rng);
    let mut ok = [0; 2];
    for _ in 0..60000 {
        let toc = rng.next_u32() as u8;
        let sd = rng.range_i32(0, 1) == 0;
        let (mut p, _) = build_packet(&mut rng, toc, sd, &eb);
        mutate(&mut rng, &mut p);
        check_parse(&p);
        check_toc_helpers(&p);
        if packet::parse_impl(&p, sd).is_ok() {
            ok[usize::from(sd)] += 1;
        }
    }
    eprintln!("parse ok: normal={} self-delimited={}", ok[0], ok[1]);
    assert!(ok[0] > 15000 && ok[1] > 15000);
}

#[test]
fn encode_size_and_align() {
    for s in 0..=1275 {
        let mut b = [0u8; 2];
        let n = packet::encode_size(s, &mut b);
        assert_eq!(&b[..n], &c::encode_size(s)[..], "size {s}");
    }
    for i in (-100..300).chain([65004, 65536, 1 << 20, i32::MAX - 16, -(1 << 20)]) {
        assert_eq!(packet::align(i), c::align(i), "align {i}");
    }
}

#[test]
fn soft_clip() {
    let mut rng = Rng::new(0xc11f);
    for iter in 0..3000 {
        let ch = rng.range_i32(1, 4) as usize;
        let n = match rng.range_i32(0, 3) {
            0 => rng.range_i32(1, 8),
            _ => rng.range_i32(1, 960),
        } as usize;
        let amp = [0.5f32, 1.0, 1.5, 2.5, 4.0][rng.range_i32(0, 4) as usize];
        let mut mem_r: Vec<f32> = (0..ch)
            .map(|_| {
                if rng.range_i32(0, 2) == 0 {
                    0.0
                } else {
                    0.3 * rng.f32_sym()
                }
            })
            .collect();
        let mut mem_c = mem_r.clone();
        // Several consecutive frames to exercise the carried state.
        for _ in 0..3 {
            let mut x: Vec<f32> = (0..n * ch)
                .map(|i| {
                    let t = i as f32 / (ch as f32 * 37.0);
                    amp * (0.7 * (t * 3.1).sin() + 0.3 * rng.f32_sym())
                })
                .collect();
            if iter % 17 == 0 && !x.is_empty() {
                // edge values
                let k = rng.range_i32(0, x.len() as i32 - 1) as usize;
                x[k] = [1.0, -1.0, 2.0, -2.0, 1e30, -1e30, 0.0][rng.range_i32(0, 6) as usize];
            }
            let mut xr = x.clone();
            let mut xc = x.clone();
            if iter % 2 == 0 {
                packet::pcm_soft_clip(&mut xr, n, ch, &mut mem_r);
                api::pcm_soft_clip(&mut xc, n, ch, &mut mem_c);
            } else {
                packet::opus_pcm_soft_clip_impl(&mut xr, n as i32, ch as i32, &mut mem_r);
                c::soft_clip_impl(&mut xc, n as i32, ch as i32, &mut mem_c);
            }
            assert_bits_eq_f32("soft_clip x", &xr, &xc);
            assert_bits_eq_f32("soft_clip mem", &mem_r, &mem_c);
        }
    }
    // Degenerate arguments are no-ops.
    let mut x = [3.0f32; 4];
    let mut m = [0.0f32; 2];
    packet::opus_pcm_soft_clip_impl(&mut x, 0, 2, &mut m);
    packet::opus_pcm_soft_clip_impl(&mut x, 2, 0, &mut m);
    assert_eq!(x, [3.0; 4]);
}

// ------------------------------------------------------------------------------- extensions.c

fn check_ext_parsers(data: &[u8], nb_frames: i32) {
    // count
    assert_eq!(
        extensions::count(data, nb_frames),
        c::ext_count(data, nb_frames),
        "count nb_frames={nb_frames} {data:02x?}"
    );
    // parse with several capacities
    let full = c::ext_count(data, nb_frames).max(0) as usize;
    for cap in [0usize, full.saturating_sub(1), full, full + 3] {
        let mut out = vec![Extension::default(); cap];
        let r = code(extensions::parse(data, &mut out, nb_frames));
        let (cret, cn, cexts) = c::ext_parse(data, nb_frames, cap);
        match r {
            Ok(n) => {
                assert_eq!(cret, 0, "parse ret cap={cap} {data:02x?}");
                assert_eq!(n as i32, cn);
                let rexts: Vec<Ext> = out[..n].iter().map(|e| ext_to_c(data, e)).collect();
                assert_eq!(rexts, cexts);
            }
            Err(e) => assert_eq!(e, cret, "parse err cap={cap} {data:02x?}"),
        }
    }
    if (0..=48).contains(&nb_frames) {
        let mut per = vec![0i32; nb_frames as usize];
        let n = extensions::count_ext(data, &mut per);
        let (cn, cper) = c::ext_count_ext(data, nb_frames);
        assert_eq!(n, cn);
        assert_eq!(per, cper);
        for cap in [full, full + 1, full.saturating_sub(1)] {
            let mut out = vec![Extension::default(); cap];
            let r = code(extensions::parse_ext(data, &mut out, &per));
            let (cret, cn, cexts) = c::ext_parse_ext(data, &cper, cap);
            match r {
                Ok(n) => {
                    assert_eq!(cret, 0);
                    assert_eq!(n as i32, cn);
                    let rexts: Vec<Ext> = out[..n].iter().map(|e| ext_to_c(data, e)).collect();
                    assert_eq!(rexts, cexts, "parse_ext {data:02x?}");
                }
                Err(e) => assert_eq!(e, cret, "parse_ext err {data:02x?}"),
            }
        }
    }
}

fn check_iter_ops(rng: &mut Rng, data: &[u8], nb_frames: i32) {
    let nops = rng.range_i32(1, 40) as usize;
    let ops: Vec<IterOp> = (0..nops)
        .map(|_| match rng.range_i32(0, 9) {
            0 => IterOp::Find(rng.range_i32(3, 127)),
            1 => IterOp::Reset,
            2 => IterOp::SetFrameMax(rng.range_i32(0, nb_frames.max(0))),
            _ => IterOp::Next,
        })
        .collect();
    let cres = c::ext_iter_ops(data, nb_frames, &ops);
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
                assert_eq!(cret, 1, "iter op {op:?} {data:02x?}");
                assert_eq!(ext_to_c(data, &e), cext, "iter op {op:?} {data:02x?}");
            }
            Ok(None) => assert_eq!(cret, 0, "iter op {op:?} {data:02x?}"),
            Err(e) => assert_eq!(e.code(), cret, "iter op {op:?} {data:02x?}"),
        }
    }
    // Iterator adaptor yields the same sequence as repeated next_extension().
    let mut a = ExtensionIterator::new(data, nb_frames);
    let mut seq = Vec::new();
    loop {
        match a.next_extension() {
            Ok(Some(e)) => seq.push(Ok(e)),
            Ok(None) => break,
            Err(e) => {
                seq.push(Err(e));
                break;
            }
        }
    }
    let seq2: Vec<_> = ExtensionIterator::new(data, nb_frames).collect();
    assert_eq!(seq, seq2);
}

#[test]
fn extensions_generate_and_parse() {
    let mut rng = Rng::new(0xe47e);
    let eb = ext_buf(&mut rng);
    let (mut ok, mut multi, mut padded, mut repeat_l0) = (0, 0, 0, 0);
    for _ in 0..20000 {
        let nb_frames = if rng.range_i32(0, 9) == 0 {
            rng.range_i32(0, 50) // negative nb_frames is UB in C (memset with a huge size)
        } else {
            let hi = if rng.range_i32(0, 3) == 0 { 48 } else { 12 };
            rng.range_i32(1, hi)
        };
        let exts = rand_exts(&mut rng, nb_frames.clamp(0, 48), eb.len(), true);
        let rexts = exts_to_rust(&exts, &eb);
        let pad = rng.range_i32(0, 3) == 0;
        let len = match rng.range_i32(0, 3) {
            0 => rng.range_i32(0, 64),
            _ => rng.range_i32(0, 3000),
        };
        // measure only
        let rr = code(extensions::generate_impl(None, len, &rexts, nb_frames, pad));
        let cr = c::ext_generate(None, len, &exts, &eb, nb_frames, pad);
        assert_eq!(flat(rr), cr, "generate(NULL) len={len} exts={exts:?}");
        // write
        let init = rng.next_u32() as u8;
        let mut out_r = vec![init; len as usize];
        let mut out_c = vec![init; len as usize];
        let rr = code(extensions::generate(&mut out_r, &rexts, nb_frames, pad)).map(|n| n as i32);
        let cr = c::ext_generate(Some(&mut out_c), len, &exts, &eb, nb_frames, pad);
        assert_eq!(flat(rr), cr, "generate len={len} exts={exts:?}");
        assert_slice_eq("generate bytes", &out_r, &out_c);
        let sz = code(extensions::generated_size(
            len as usize,
            &rexts,
            nb_frames,
            pad,
        ));
        assert_eq!(flat(sz.map(|n| n as i32)), cr);

        if cr >= 0 && (0..=48).contains(&nb_frames) {
            ok += 1;
            if nb_frames >= 2 && exts.len() >= 2 * nb_frames as usize {
                multi += 1;
            }
            if pad && cr as usize == out_c.len() && !exts.is_empty() {
                padded += 1;
            }
            // Repeat indicator with L=0 (0x04) followed by more data.
            if out_c[..cr as usize].windows(2).any(|w| w[0] == 0x04) {
                repeat_l0 += 1;
            }
            let mut data = out_c[..cr as usize].to_vec();
            // Round trip: every generated extension is parsed back.
            if nb_frames > 0 {
                assert_eq!(extensions::count(&data, nb_frames), exts.len() as i32);
                let mut got = vec![Extension::default(); exts.len()];
                let n = extensions::parse(&data, &mut got, nb_frames).expect("roundtrip");
                assert_eq!(n, exts.len());
                let mut a: Vec<_> = got
                    .iter()
                    .map(|e| (e.frame, e.id, e.data.to_vec()))
                    .collect();
                let mut b: Vec<_> = rexts
                    .iter()
                    .map(|e| (e.frame, e.id, e.data.to_vec()))
                    .collect();
                a.sort();
                b.sort();
                assert_eq!(a, b, "roundtrip {exts:?}");
            }
            let nbf = if rng.range_i32(0, 4) == 0 {
                rng.range_i32(0, 48)
            } else {
                nb_frames
            };
            check_ext_parsers(&data, nbf);
            check_iter_ops(&mut rng, &data, nbf);
            mutate(&mut rng, &mut data);
            check_ext_parsers(&data, nbf);
            check_iter_ops(&mut rng, &data, nbf);
        }
    }
    eprintln!("extensions: ok={ok} multi={multi} padded={padded} repeat_l0={repeat_l0}");
    assert!(ok > 8000 && multi > 1500 && padded > 800 && repeat_l0 > 400);
}

#[test]
fn extensions_random_bytes() {
    let mut rng = Rng::new(0xbad_e47);
    for _ in 0..20000 {
        let n = rng.range_i32(0, 48) as usize;
        let mut data = vec![0u8; n];
        rng.fill_bytes(&mut data);
        // Bias towards small IDs (separators, repeats, padding).
        for b in data.iter_mut() {
            if rng.range_i32(0, 3) == 0 {
                *b = rng.range_i32(0, 9) as u8;
            }
        }
        let nb_frames = rng.range_i32(0, 48);
        check_ext_parsers(&data, nb_frames);
        check_iter_ops(&mut rng, &data, nb_frames);
    }
}

// ------------------------------------------------------------------------------- repacketizer.c

#[test]
fn repacketizer_random() {
    let mut rng = Rng::new(0x4e9a_c4e7);
    let eb = ext_buf(&mut rng);
    let (mut ok, mut ok_ext, mut ok_pad_ext, mut ok_sd, mut padded_src) = (0, 0, 0, 0, 0);
    for it in 0..30000 {
        let toc_base = (rng.next_u32() as u8) & 0xFC;
        let npk = rng.range_i32(1, 6) as usize;
        let cat_sd = rng.range_i32(0, 3) == 0;
        let packets: Vec<Vec<u8>> = (0..npk)
            .map(|_| {
                let toc = if rng.range_i32(0, 19) == 0 {
                    rng.next_u32() as u8
                } else {
                    toc_base | rng.range_i32(0, 3) as u8
                };
                let (mut p, _) = build_packet(&mut rng, toc, cat_sd, &eb);
                if rng.range_i32(0, 9) == 0 {
                    mutate(&mut rng, &mut p);
                }
                p
            })
            .collect();
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();

        let mut rp = Repacketizer::new();
        let mut cat_rets = Vec::new();
        for p in &refs {
            cat_rets.push(flat(code(rp.cat_impl(p, cat_sd)).map(|()| 0)));
        }
        let nb = rp.nb_frames();
        let (begin, end) = if nb > 0 && rng.range_i32(0, 9) != 0 {
            let b = rng.range_i32(0, nb - 1);
            (b, rng.range_i32(b + 1, nb))
        } else {
            (rng.range_i32(-1, nb + 1), rng.range_i32(-1, nb + 1))
        };
        let count = (end - begin).max(1);
        let exts = if rng.range_i32(0, 2) == 0 {
            rand_exts(&mut rng, count, eb.len(), true)
        } else {
            Vec::new()
        };
        let rexts = exts_to_rust(&exts, &eb);
        let sd = rng.range_i32(0, 3) == 0;
        let pad = rng.range_i32(0, 2) == 0;
        let maxlen = match rng.range_i32(0, 4) {
            0 => rng.range_i32(0, 16),
            1 => rng.range_i32(0, 300),
            _ => rng.range_i32(0, 6000),
        } as usize;
        let init = rng.next_u32() as u8;
        let mut out_r = vec![init; maxlen];
        let mut out_c = vec![init; maxlen];
        let rret = code(rp.out_range_impl(begin, end, &mut out_r, sd, pad, &rexts));
        let cres = c::rp_run(&refs, cat_sd, begin, end, &mut out_c, sd, pad, &exts, &eb);
        assert_eq!(cat_rets, cres.cat_rets, "cat rets it={it}");
        assert_eq!(nb, cres.nb_frames);
        assert_eq!(
            flat(rret),
            cres.ret,
            "out_range_impl it={it} begin={begin} end={end} sd={sd} pad={pad} maxlen={maxlen}"
        );
        assert_slice_eq("out_range_impl bytes", &out_r, &out_c);
        if rret.is_ok() {
            ok += 1;
            let src_ext = (begin.max(0)..end.min(nb))
                .any(|i| extensions::count(rp_padding(&refs, i), 48) > 0);
            if !exts.is_empty() || src_ext {
                ok_ext += 1;
                if pad {
                    ok_pad_ext += 1;
                }
            }
            if src_ext {
                padded_src += 1;
            }
            if sd {
                ok_sd += 1;
            }
        }

        // Public API: out_range / out through the C repacketizer.
        if !cat_sd {
            let mut crp = api::Repacketizer::new().expect("alloc");
            for (p, &want) in refs.iter().zip(&cat_rets) {
                assert_eq!(flat(crp.cat(p).map(|()| 0)), want);
            }
            let mut o1 = vec![0u8; maxlen];
            let mut o2 = vec![0u8; maxlen];
            let r1 = code(rp.out_range(begin, end, &mut o1));
            let r2 = crp.out_range(begin, end, &mut o2);
            assert_eq!(r1, r2);
            assert_slice_eq("out_range bytes", &o1, &o2);
            let r1 = code(rp.out(&mut o1));
            let r2 = crp.out(&mut o2);
            assert_eq!(r1, r2);
            assert_slice_eq("out bytes", &o1, &o2);
        }
        // A successfully produced packet must parse again (checked below).
        if let Ok(n) = rret
            && !sd
        {
            assert!(packet::parse(&out_r[..n as usize]).is_ok());
        }
    }
    eprintln!(
        "repacketizer: ok={ok} ext={ok_ext} pad+ext={ok_pad_ext} src_ext={padded_src} sd={ok_sd}"
    );
    assert!(ok > 7500 && ok_ext > 1500 && ok_pad_ext > 500 && padded_src > 250 && ok_sd > 1500);
}

/// Padding of the packet that contributed frame `i` (approximation for coverage stats: the
/// padding of the `i`-th packet if it exists).
fn rp_padding<'a>(refs: &[&'a [u8]], i: i32) -> &'a [u8] {
    refs.get(i as usize)
        .and_then(|p| packet::parse(p).ok())
        .map_or(&[], |p| p.padding)
}

#[test]
fn pad_unpad() {
    let mut rng = Rng::new(0x9ad);
    let eb = ext_buf(&mut rng);
    let (mut pad_ok, mut pad_ext_ok, mut unpad_ok) = (0, 0, 0);
    for _ in 0..30000 {
        let toc = rng.next_u32() as u8;
        let (mut p, _) = build_packet(&mut rng, toc, false, &eb);
        if rng.range_i32(0, 5) == 0 {
            mutate(&mut rng, &mut p);
        }
        let len = p.len();
        // opus_packet_pad
        let new_len = match rng.range_i32(0, 5) {
            0 => rng.range_i32(0, len as i32 + 2) as usize,
            _ => len + rng.range_i32(0, 800) as usize,
        };
        let cap = new_len.max(len);
        let mut br = vec![0xAAu8; cap];
        br[..len].copy_from_slice(&p);
        let mut bc = br.clone();
        let rr = code(repacketizer::packet_pad(&mut br, len, new_len));
        let cr = api::packet_pad(&mut bc, len, new_len);
        assert_eq!(rr, cr, "pad {len}->{new_len} {p:02x?}");
        if rr.is_ok() && new_len > len {
            pad_ok += 1;
        }
        assert_slice_eq("pad bytes", &br, &bc);

        // opus_packet_pad_impl with extensions (pad on/off).
        let count = packet::get_nb_frames(&p).unwrap_or(1).clamp(1, 48);
        let exts = if rng.range_i32(0, 1) == 0 {
            rand_exts(&mut rng, count, eb.len(), true)
        } else {
            Vec::new()
        };
        let rexts = exts_to_rust(&exts, &eb);
        let padf = rng.range_i32(0, 1) == 0;
        let mut br = vec![0x55u8; cap];
        br[..len].copy_from_slice(&p);
        let mut bc = br.clone();
        let rr = code(repacketizer::opus_packet_pad_impl(
            &mut br,
            len as i32,
            new_len as i32,
            padf,
            &rexts,
        ));
        if new_len >= len {
            // The C shim passes new_len = slice length.
            let cr = c::pad_impl(&mut bc[..new_len], len as i32, padf, &exts, &eb);
            assert_eq!(flat(rr), cr, "pad_impl {len}->{new_len}");
            if cr > 0 && !exts.is_empty() {
                pad_ext_ok += 1;
            }
            assert_slice_eq("pad_impl bytes", &br, &bc);
        } else {
            assert_eq!(rr, Err(-1));
        }

        // opus_packet_unpad (on the original and on the padded packet)
        for src in [p.clone(), bc[..new_len.max(len)].to_vec()] {
            let mut ur = src.clone();
            let mut uc = src.clone();
            let rr = code(repacketizer::packet_unpad(&mut ur));
            let cr = api::packet_unpad(&mut uc);
            assert_eq!(rr, cr, "unpad {src:02x?}");
            if rr.is_ok() {
                unpad_ok += 1;
            }
            assert_slice_eq("unpad bytes", &ur, &uc);
        }
    }
    assert_eq!(code(repacketizer::packet_unpad(&mut [])), Err(-1));
    eprintln!("pad ok={pad_ok} pad_impl+ext ok={pad_ext_ok} unpad ok={unpad_ok}");
    assert!(pad_ok > 12000 && pad_ext_ok > 2500 && unpad_ok > 25000);
}

#[test]
fn multistream_pad_unpad() {
    let mut rng = Rng::new(0x3570);
    let eb = ext_buf(&mut rng);
    let (mut pad_ok, mut unpad_ok) = (0, 0);
    for _ in 0..20000 {
        let nb_streams = rng.range_i32(1, 5);
        let mut ms = Vec::new();
        for s in 0..nb_streams {
            let toc = rng.next_u32() as u8;
            let (p, _) = build_packet(&mut rng, toc, s != nb_streams - 1, &eb);
            ms.extend_from_slice(&p);
        }
        if rng.range_i32(0, 5) == 0 {
            mutate(&mut rng, &mut ms);
        }
        let streams_arg = if rng.range_i32(0, 9) == 0 {
            rng.range_i32(0, 6)
        } else {
            nb_streams
        };
        let len = ms.len();
        let new_len = match rng.range_i32(0, 5) {
            0 => rng.range_i32(0, len as i32 + 2) as usize,
            _ => len + rng.range_i32(0, 600) as usize,
        };
        let cap = new_len.max(len);
        let mut br = vec![0x11u8; cap];
        br[..len].copy_from_slice(&ms);
        let mut bc = br.clone();
        let rr = code(repacketizer::multistream_packet_pad(
            &mut br,
            len,
            new_len,
            streams_arg,
        ));
        let cr = api::multistream_packet_pad(&mut bc, len, new_len, streams_arg);
        assert_eq!(rr, cr, "ms pad");
        if rr.is_ok() && new_len > len {
            pad_ok += 1;
        }
        assert_slice_eq("ms pad bytes", &br, &bc);

        for src in [ms.clone(), bc[..cap].to_vec()] {
            let mut ur = src.clone();
            let mut uc = src.clone();
            let rr = code(repacketizer::multistream_packet_unpad(&mut ur, streams_arg));
            let cr = api::multistream_packet_unpad(&mut uc, streams_arg);
            assert_eq!(rr, cr, "ms unpad");
            if rr.is_ok() {
                unpad_ok += 1;
            }
            // Only the produced prefix is specified; beyond it C leaves stale input bytes,
            // which the port reproduces as well.
            assert_slice_eq("ms unpad bytes", &ur, &uc);
        }
    }
    eprintln!("multistream pad ok={pad_ok} unpad ok={unpad_ok}");
    assert!(pad_ok > 8000 && unpad_ok > 20000);
}

// ------------------------------------------------------------------------------- mapping_matrix.c

#[test]
fn mapping_matrix_static_and_size() {
    let statics: [(mm::MappingMatrixHeader, &[i16]); 10] = [
        (
            mm::MAPPING_MATRIX_FOA_MIXING,
            &mm::MAPPING_MATRIX_FOA_MIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_SOA_MIXING,
            &mm::MAPPING_MATRIX_SOA_MIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_TOA_MIXING,
            &mm::MAPPING_MATRIX_TOA_MIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_FOURTHOA_MIXING,
            &mm::MAPPING_MATRIX_FOURTHOA_MIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_FIFTHOA_MIXING,
            &mm::MAPPING_MATRIX_FIFTHOA_MIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_FOA_DEMIXING,
            &mm::MAPPING_MATRIX_FOA_DEMIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_SOA_DEMIXING,
            &mm::MAPPING_MATRIX_SOA_DEMIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_TOA_DEMIXING,
            &mm::MAPPING_MATRIX_TOA_DEMIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_FOURTHOA_DEMIXING,
            &mm::MAPPING_MATRIX_FOURTHOA_DEMIXING_DATA,
        ),
        (
            mm::MAPPING_MATRIX_FIFTHOA_DEMIXING,
            &mm::MAPPING_MATRIX_FIFTHOA_DEMIXING_DATA,
        ),
    ];
    for (i, (h, d)) in statics.iter().enumerate() {
        let (r, cc, g, data) = c::mm_static(i as i32);
        assert_eq!((h.rows, h.cols, h.gain), (r, cc, g), "static header {i}");
        assert_slice_eq("static data", d, &data);
        let m = mm::MappingMatrix::new(h.rows, h.cols, h.gain, d);
        assert_slice_eq("init data", m.get_data(), &data);
    }
    for rows in -3..=260 {
        for cols in [-2, 0, 1, 2, 6, 38, 100, 127, 128, 200, 254, 255, 256] {
            assert_eq!(
                mm::mapping_matrix_get_size(rows, cols),
                c::mm_get_size(rows, cols),
                "get_size {rows}x{cols}"
            );
        }
    }
}

#[test]
fn mapping_matrix_multiply() {
    let mut rng = Rng::new(0x3a7);
    for it in 0..3000 {
        let rows = rng.range_i32(1, 20) as usize;
        let cols = rng.range_i32(1, 20) as usize;
        let data: Vec<i16> = (0..rows * cols)
            .map(|_| match rng.range_i32(0, 5) {
                0 => [0, 32767, -32768, 16384][rng.range_i32(0, 3) as usize],
                _ => rng.i16(),
            })
            .collect();
        let m = mm::MappingMatrix::new(rows as i32, cols as i32, 0, &data);
        let cm = c::Mm {
            rows: rows as i32,
            cols: cols as i32,
            data: &data,
        };
        let frame_size = rng.range_i32(0, 120) as usize;
        let amp = [0.1f32, 1.0, 1.9, 200.0][rng.range_i32(0, 3) as usize];

        // channel_in: input_rows <= cols, output_row < rows, output stride output_rows.
        let input_rows = rng.range_i32(1, cols as i32) as usize;
        let output_rows = rng.range_i32(1, rows as i32) as usize;
        let output_row = rng.range_i32(0, rows as i32 - 1) as usize;
        let olen = output_rows * frame_size.max(1);
        let fin: Vec<f32> = (0..input_rows * frame_size)
            .map(|_| amp * rng.f32_sym())
            .collect();
        let mut or = vec![0.0f32; olen];
        let mut oc = vec![0.0f32; olen];
        mm::mapping_matrix_multiply_channel_in_float(
            &m,
            &fin,
            input_rows,
            &mut or,
            output_row,
            output_rows,
            frame_size,
        );
        c::mm_in_float(
            cm,
            &fin,
            input_rows,
            &mut oc,
            output_row,
            output_rows,
            frame_size,
        );
        assert_bits_eq_f32("in_float", &or, &oc);

        let sin: Vec<i16> = (0..input_rows * frame_size).map(|_| rng.i16()).collect();
        let mut or = vec![0.0f32; olen];
        let mut oc = vec![0.0f32; olen];
        mm::mapping_matrix_multiply_channel_in_short(
            &m,
            &sin,
            input_rows,
            &mut or,
            output_row,
            output_rows,
            frame_size,
        );
        c::mm_in_short(
            cm,
            &sin,
            input_rows,
            &mut oc,
            output_row,
            output_rows,
            frame_size,
        );
        assert_bits_eq_f32("in_short", &or, &oc);

        let iin: Vec<i32> = (0..input_rows * frame_size)
            .map(|_| rng.range_i32(-(1 << 23), (1 << 23) - 1))
            .collect();
        let mut or = vec![0.0f32; olen];
        let mut oc = vec![0.0f32; olen];
        mm::mapping_matrix_multiply_channel_in_int24(
            &m,
            &iin,
            input_rows,
            &mut or,
            output_row,
            output_rows,
            frame_size,
        );
        c::mm_in_int24(
            cm,
            &iin,
            input_rows,
            &mut oc,
            output_row,
            output_rows,
            frame_size,
        );
        assert_bits_eq_f32("in_int24", &or, &oc);

        // channel_out: input stride input_rows (<= cols), input_row < cols,
        // output_rows <= rows, accumulate into output.
        let input_rows = rng.range_i32(1, cols as i32) as usize;
        let input_row = rng.range_i32(0, cols as i32 - 1) as usize;
        let output_rows = rng.range_i32(1, rows as i32) as usize;
        let ilen = input_rows * frame_size.max(1);
        let fin: Vec<f32> = (0..ilen).map(|_| amp * rng.f32_sym()).collect();
        let init: Vec<f32> = (0..output_rows * frame_size)
            .map(|_| rng.f32_sym())
            .collect();
        let mut or = init.clone();
        let mut oc = init;
        mm::mapping_matrix_multiply_channel_out_float(
            &m,
            &fin,
            input_row,
            input_rows,
            &mut or,
            output_rows,
            frame_size,
        );
        c::mm_out_float(
            cm,
            &fin,
            input_row,
            input_rows,
            &mut oc,
            output_rows,
            frame_size,
        );
        assert_bits_eq_f32("out_float", &or, &oc);

        let init: Vec<i16> = (0..output_rows * frame_size)
            .map(|_| {
                if it % 3 == 0 {
                    rng.i16()
                } else {
                    rng.i16() / 64
                }
            })
            .collect();
        let mut or = init.clone();
        let mut oc = init;
        mm::mapping_matrix_multiply_channel_out_short(
            &m,
            &fin,
            input_row,
            input_rows,
            &mut or,
            output_rows,
            frame_size,
        );
        c::mm_out_short(
            cm,
            &fin,
            input_row,
            input_rows,
            &mut oc,
            output_rows,
            frame_size,
        );
        assert_slice_eq("out_short", &or, &oc);

        let init: Vec<i32> = (0..output_rows * frame_size)
            .map(|_| {
                if it % 3 == 0 {
                    rng.next_u32() as i32
                } else {
                    rng.range_i32(-9999, 9999)
                }
            })
            .collect();
        let mut or = init.clone();
        let mut oc = init;
        mm::mapping_matrix_multiply_channel_out_int24(
            &m,
            &fin,
            input_row,
            input_rows,
            &mut or,
            output_rows,
            frame_size,
        );
        c::mm_out_int24(
            cm,
            &fin,
            input_row,
            input_rows,
            &mut oc,
            output_rows,
            frame_size,
        );
        assert_slice_eq("out_int24", &or, &oc);
    }
}

// ------------------------------------------------------------------------------- mlp.c

#[test]
fn mlp_activations() {
    let mut rng = Rng::new(0x71a5);
    let mut xs: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        8.0,
        -8.0,
        1e-30,
        1e10,
        -1e10,
        1e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    for _ in 0..200_000 {
        let scale = [0.01f32, 1.0, 5.0, 30.0, 1e3][rng.range_i32(0, 4) as usize];
        xs.push(scale * rng.f32_sym());
    }
    for _ in 0..20000 {
        xs.push(f32::from_bits(rng.next_u32()));
    }
    for &x in &xs {
        assert_eq!(
            mlp::tansig_approx(x).to_bits(),
            c::mlp_tansig(x).to_bits(),
            "tansig({x})"
        );
        assert_eq!(
            mlp::sigmoid_approx(x).to_bits(),
            c::mlp_sigmoid(x).to_bits(),
            "sigmoid({x})"
        );
    }
}

fn rand_i8s(rng: &mut Rng, n: usize) -> Vec<i8> {
    (0..n).map(|_| rng.next_u32() as i8).collect()
}

#[test]
fn mlp_layers() {
    let mut rng = Rng::new(0x6e0);
    for _ in 0..3000 {
        // Dense layer.
        let nin = rng.range_i32(1, 40) as usize;
        let nn = rng.range_i32(1, 40) as usize;
        let bias = rand_i8s(&mut rng, nn);
        let w = rand_i8s(&mut rng, nin * nn);
        let sigmoid = rng.range_i32(0, 1) == 1;
        let amp = [0.1f32, 1.0, 4.0][rng.range_i32(0, 2) as usize];
        let input: Vec<f32> = (0..nin).map(|_| amp * rng.f32_sym()).collect();
        let layer = mlp::AnalysisDenseLayer {
            bias: &bias,
            input_weights: &w,
            nb_inputs: nin as i32,
            nb_neurons: nn as i32,
            sigmoid,
        };
        let mut or = vec![0.0f32; nn];
        let mut oc = vec![0.0f32; nn];
        mlp::analysis_compute_dense(&layer, &mut or, &input);
        c::mlp_dense(&bias, &w, nin, nn, sigmoid, &mut oc, &input);
        assert_bits_eq_f32("dense", &or, &oc);

        // GRU layer, several steps.
        let nin = rng.range_i32(1, 40) as usize;
        let nn = rng.range_i32(1, 32) as usize;
        let bias = rand_i8s(&mut rng, 3 * nn);
        let w = rand_i8s(&mut rng, 3 * nin * nn);
        let rw = rand_i8s(&mut rng, 3 * nn * nn);
        let gru = mlp::AnalysisGRULayer {
            bias: &bias,
            input_weights: &w,
            recurrent_weights: &rw,
            nb_inputs: nin as i32,
            nb_neurons: nn as i32,
        };
        let mut sr: Vec<f32> = (0..nn).map(|_| rng.f32_sym()).collect();
        let mut sc = sr.clone();
        for _ in 0..4 {
            let input: Vec<f32> = (0..nin).map(|_| amp * rng.f32_sym()).collect();
            mlp::analysis_compute_gru(&gru, &mut sr, &input);
            c::mlp_gru(&bias, &w, &rw, nin, nn, &mut sc, &input);
            assert_bits_eq_f32("gru", &sr, &sc);
        }
    }
}

#[test]
fn mlp_builtin_layers() {
    let mut rng = Rng::new(0xb1);
    let mut state_r = [0.0f32; 24];
    let mut state_c = [0.0f32; 24];
    for i in 0..5000 {
        let amp = [0.1f32, 1.0, 3.0, 20.0][rng.range_i32(0, 3) as usize];
        let feat: Vec<f32> = (0..25).map(|_| amp * rng.f32_sym()).collect();
        let mut l0r = [0.0f32; 32];
        let mut l0c = [0.0f32; 32];
        mlp::analysis_compute_dense(&mlp::LAYER0, &mut l0r, &feat);
        c::mlp_builtin(0, &mut l0c, &feat);
        assert_bits_eq_f32("layer0", &l0r, &l0c);
        if i % 100 == 0 {
            state_r = [0.0; 24];
            state_c = [0.0; 24];
        }
        mlp::analysis_compute_gru(&mlp::LAYER1, &mut state_r, &l0r);
        c::mlp_builtin(1, &mut state_c, &l0c);
        assert_bits_eq_f32("layer1", &state_r, &state_c);
        let mut pr = [0.0f32; 2];
        let mut pc = [0.0f32; 2];
        mlp::analysis_compute_dense(&mlp::LAYER2, &mut pr, &state_r);
        c::mlp_builtin(2, &mut pc, &state_c);
        assert_bits_eq_f32("layer2", &pr, &pc);
    }
}

// ------------------------------------------------------------------------------- opus_multistream.c

#[test]
fn multistream_layout() {
    let mut rng = Rng::new(0x1a70);
    for _ in 0..20000 {
        let nb_streams = rng.range_i32(0, 260);
        let nb_coupled = rng.range_i32(0, nb_streams.min(140));
        let nb_channels = rng.range_i32(0, 256);
        let mut mapping = [0u8; 256];
        let maxc = (nb_streams + nb_coupled).clamp(1, 256);
        for m in mapping.iter_mut() {
            *m = match rng.range_i32(0, 9) {
                0 => 255,
                1 => rng.next_u32() as u8,
                _ => rng.range_i32(0, maxc - 1) as u8,
            };
        }
        let layout = ChannelLayout {
            nb_channels,
            nb_streams,
            nb_coupled_streams: nb_coupled,
            mapping,
        };
        assert_eq!(
            multistream::validate_layout(&layout),
            c::ms_validate_layout(nb_channels, nb_streams, nb_coupled, &mapping)
        );
        for _ in 0..4 {
            let stream_id = rng.range_i32(0, nb_streams.max(1));
            let mut prev = [-1, -1, -1];
            for _ in 0..3 {
                for (which, pv) in prev.iter_mut().enumerate() {
                    let r = match which {
                        0 => multistream::get_left_channel(&layout, stream_id, *pv),
                        1 => multistream::get_right_channel(&layout, stream_id, *pv),
                        _ => multistream::get_mono_channel(&layout, stream_id, *pv),
                    };
                    let cr = c::ms_get_channel(
                        which as i32,
                        nb_channels,
                        nb_streams,
                        nb_coupled,
                        &mapping,
                        stream_id,
                        *pv,
                    );
                    assert_eq!(r, cr, "get_channel which={which} stream={stream_id}");
                    *pv = r;
                }
            }
        }
    }
}

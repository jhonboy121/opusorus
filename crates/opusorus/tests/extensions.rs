//! Port of libopus `tests/test_opus_extensions.c`: generating and parsing packet padding
//! extensions (including the "Repeat These Extensions" mechanism), random-input parsing with
//! round trips, and the repacketizer's extension merging, through the public
//! [`opusorus::extensions`] API.
//!
//! Adaptations:
//! * `opus_extension_data` with a negative `len`: the payload is a slice.
//! * The C `*nb_extensions` in/out capacity is the output slice length; after an
//!   `OPUS_INVALID_PACKET` result C stores the number of extensions extracted so far, which the
//!   port recomputes with [`extensions::count`].
//! * `test_random_extensions_parse` runs 100 M iterations in C; the default test runs 2 M
//!   (the full count is the `#[ignore]`d `random_extensions_parse_full`, or set
//!   `OPUS_TEST_FULL`). Output slices are sized to what a call can write (behaviour-identical
//!   to the C capacity).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::{FastRand, offset_in};
use opusorus::extensions::{self, Extension, count, count_ext, generate, parse, parse_ext};
use opusorus::packet::parse_impl;
use opusorus::{Error, Repacketizer};

const fn ext(id: i32, frame: i32, data: &[u8]) -> Extension<'_> {
    Extension { id, frame, data }
}

/// The extensions most tests use.
const EXT4: [Extension<'static>; 4] = [
    ext(3, 0, b"a"),
    ext(32, 10, b"DRED"),
    ext(33, 1, b"NOT DRED"),
    ext(4, 4, b""),
];

/// Port of `test_extensions_generate_success`.
#[test]
fn generate_success() {
    let mut packet = [0u8; 32];
    let result = generate(&mut packet[..23 + 4], &EXT4, 11, true);
    assert_eq!(result, Ok(23 + 4), "expected length 23+4");
    let mut p = &packet[..];

    // expect padding
    assert_eq!(p[..4], [1, 1, 1, 1], "expected padding");
    p = &p[4..];

    // extension ID=3
    assert_eq!(p[0] >> 1, 3, "expected extension id 3");
    // For extension IDs 1 through 31, L=0 means that no data follows the extension, whereas
    // L=1 means that exactly one byte of extension data follows.
    assert_eq!(p[0] & 0x01, 1, "expected L-bit set");
    // content
    assert_eq!(p[1], b'a', "expected extension content");
    p = &p[2..];

    // next byte should increment the frame count, ID=1, L=0
    assert_eq!(p[0], 0x02, "bad frame separator");
    p = &p[1..];
    // extension ID=33
    assert_eq!(p[0] >> 1, 33, "expected extension id 33");
    // For IDs 32 to 127, L=0 signals that the extension data takes up the rest of the
    // padding, and L=1 signals that a length indicator follows.
    assert_eq!(p[0] & 0x01, 1, "expected L-bit set");
    // content
    assert_eq!(usize::from(p[1]), EXT4[2].data.len(), "expected length");
    p = &p[2..];
    assert_eq!(
        &p[..EXT4[2].data.len()],
        EXT4[2].data,
        "expected extension content"
    );
    p = &p[EXT4[2].data.len()..];

    // advance to frame 4, increment by 3
    // next byte should increment the frame count, ID=1, L=1
    assert_eq!(p[0], 0x03, "bad frame separator");
    assert_eq!(p[1], 0x03, "bad frame increment");
    p = &p[2..];
    // extension ID=4
    assert_eq!(p[0] >> 1, 4, "expected extension id 4");
    assert_eq!(p[0] & 0x01, 0, "expected L-bit unset");
    p = &p[1..];

    // advance to frame 10, increment by 6
    // next byte should increment the frame count, ID=1, L=1
    assert_eq!(p[0], 0x03, "bad frame separator");
    assert_eq!(p[1], 0x06, "bad frame increment");
    p = &p[2..];
    // extension ID=32
    assert_eq!(p[0] >> 1, 32, "expected extension id 32");
    // For IDs 32 to 127, L=0 signals that the extension data takes up the rest of the padding
    assert_eq!(p[0] & 0x01, 0, "expected L-bit unset");
    p = &p[1..];
    assert_eq!(
        &p[..EXT4[1].data.len()],
        EXT4[1].data,
        "expected extension content"
    );
}

/// Port of `test_extensions_generate_zero`.
#[test]
fn generate_zero() {
    let mut packet = [0u8; 32];
    // zero length packet, zero extensions
    assert_eq!(generate(&mut packet[..0], &[], 0, true), Ok(0));
    assert_eq!(extensions::generated_size(0, &[], 0, true), Ok(0));
}

/// Port of `test_extensions_generate_no_padding`.
#[test]
fn generate_no_padding() {
    let mut packet = [0u8; 32];
    assert_eq!(generate(&mut packet, &EXT4, 11, false), Ok(23));
    assert_eq!(extensions::generated_size(32, &EXT4, 11, false), Ok(23));
}

/// Port of `test_extensions_generate_fail`.
#[test]
fn generate_fail() {
    let mut packet = [0u8; 100];

    // buffer too small: this failure can occur at lots of points, so iterate to check as many
    // as possible
    for len in 0..23 {
        packet[len..].fill(0xFE);
        assert_eq!(
            generate(&mut packet[..len], &EXT4, 11, true),
            Err(Error::BufferTooSmall),
            "len {len}"
        );
        assert!(
            packet[len..].iter().all(|&b| b == 0xFE),
            "expected 0xFE padding to be undisturbed"
        );
    }

    let cases: [(&[Extension<'_>], i32, &str); 6] = [
        (&[ext(256, 0, b"a")], 11, "invalid id"),
        (&[ext(2, 0, b"a")], 11, "invalid id"),
        (&[ext(33, 11, b"a")], 49, "frame count too big"),
        (&[ext(33, -1, b"a")], 11, "frame index too small"),
        (&[ext(33, 11, b"a")], 11, "frame index too big"),
        (
            &[ext(3, 0, b"abcd")],
            1,
            "size too big for extension IDs 1 through 31",
        ),
    ];
    for (e, nb_frames, what) in cases {
        assert_eq!(
            generate(&mut packet, e, nb_frames, true),
            Err(Error::BadArg),
            "{what}"
        );
    }
    // Skipped: negative sizes for extension IDs 1..31 and 32..127 (the payload is a slice).
}

/// Port of `test_extensions_parse_success`.
#[test]
fn parse_success() {
    let mut ext_out = [Extension::default(); 10];
    let mut packet = [0u8; 32];
    let nb_frames = 11;
    let len = generate(&mut packet, &EXT4, 11, true).unwrap();
    assert_eq!(len, 32, "expected length 32");
    assert_eq!(count(&packet[..len], nb_frames), 4);
    let nb_ext = parse(&packet[..len], &mut ext_out, nb_frames).unwrap();
    assert_eq!(nb_ext, 4, "expected 4 extensions");

    assert_eq!(
        (ext_out[0].id, ext_out[0].frame, ext_out[0].data),
        (3, 0, EXT4[0].data)
    );
    assert_eq!(
        (ext_out[1].id, ext_out[1].frame, ext_out[1].data),
        (33, 1, EXT4[2].data)
    );
    assert_eq!(
        (ext_out[2].id, ext_out[2].frame, ext_out[2].data.len()),
        (4, 4, 0)
    );
    assert_eq!(
        (ext_out[3].id, ext_out[3].frame, ext_out[3].data),
        (32, 10, EXT4[1].data)
    );
}

/// Port of `test_extensions_parse_zero`.
#[test]
fn parse_zero() {
    let e = [ext(32, 1, b"DRED")];
    let mut packet = [0u8; 32];
    let len = generate(&mut packet, &e, 2, true).unwrap();
    assert_eq!(len, 32, "expected length 32");
    assert_eq!(
        parse(&packet[..len], &mut [], 2),
        Err(Error::BufferTooSmall)
    );
}

/// `opus_packet_extensions_parse` with the C in/out `nb_ext`: returns the result and updates
/// `nb_ext` as C does (the count extracted before an `OPUS_INVALID_PACKET`).
fn parse_c(data: &[u8], nb_ext: &mut usize, nb_frames: i32) -> Result<(), Error> {
    let mut out = vec![Extension::default(); *nb_ext];
    match parse(data, &mut out, nb_frames) {
        Ok(n) => {
            *nb_ext = n;
            Ok(())
        }
        Err(Error::InvalidPacket) => {
            *nb_ext = count(data, nb_frames) as usize;
            Err(Error::InvalidPacket)
        }
        Err(e) => Err(e),
    }
}

/// Port of `test_extensions_parse_fail`.
#[test]
fn parse_fail() {
    let e: [Extension<'static>; 7] = [
        ext(3, 0, b"a"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
        ext(32, 10, b"DRED"),
        ext(32, 9, b"DRED"),
        ext(4, 9, b"b"),
        ext(4, 10, b"c"),
    ];
    let mut packet = [0u8; 32];

    // create invalid length
    let len = generate(&mut packet, &e[..4], 11, false).unwrap();
    packet[4] = 255;
    let mut nb_ext = 10;
    let nb_frames = 11;
    assert_eq!(
        parse_c(&packet[..len], &mut nb_ext, nb_frames),
        Err(Error::InvalidPacket)
    );
    // note, opus_packet_extensions_count stops at the invalid frame increment and tells us
    // that we have 1 extension
    assert_eq!(count(&packet[..len], nb_frames), 1);

    // create invalid frame increment
    nb_ext = 10;
    let len = generate(&mut packet, &e[..4], 11, false).unwrap();
    // first by reducing the number of frames
    assert_eq!(
        parse_c(&packet[..len], &mut nb_ext, 5),
        Err(Error::InvalidPacket)
    );
    // note, opus_packet_extensions_count stops at the invalid frame increment and tells us
    // that we have 3 extensions
    assert_eq!(count(&packet[..len], 5), 3);

    // then by increasing the increment
    packet[14] = 255;
    assert_eq!(
        parse_c(&packet[..len], &mut nb_ext, nb_frames),
        Err(Error::InvalidPacket)
    );
    // note, opus_packet_extensions_count stops at the invalid frame increment and tells us
    // that we have 2 extensions
    assert_eq!(count(&packet[..len], nb_frames), 2);

    // not enough space
    nb_ext = 1;
    let len = generate(&mut packet, &e[..4], 11, false).unwrap();
    assert_eq!(
        parse_c(&packet[..len], &mut nb_ext, nb_frames),
        Err(Error::BufferTooSmall)
    );

    // create repeated L=0 long extension without enough room for all of the short extension
    // payloads that need to follow it
    let len = generate(&mut packet, &e, 11, false).unwrap() - 5;
    nb_ext = 10;
    assert_eq!(
        parse_c(&packet[..len], &mut nb_ext, nb_frames),
        Err(Error::InvalidPacket)
    );
    // note, opus_packet_extensions_count stops at the invalid long extension and tells us that
    // we have 5 extensions
    assert_eq!(count(&packet[..len], nb_frames), 5);

    // overflow for long extension length (about 8 MB)
    const LENSIZE: usize = (1usize << 31) / 255 + 1;
    let mut buf = vec![0xFFu8; LENSIZE + 1];
    buf[0] = (33 << 1) | 1;
    buf[LENSIZE] = 0xFE;
    assert_eq!(parse_c(&buf, &mut nb_ext, 1), Err(Error::InvalidPacket));
}

/// Port of `check_ext_data`: the same extensions with the same data appear in both `ext_in`
/// and `ext_out`, with `ext_out` in frame order and the order within a frame matching
/// `ext_in`.
fn check_ext_data(ext_in: &[Extension<'_>], ext_out: &[Extension<'_>], nb_ext: usize) {
    let mut prev_frame = -1;
    let mut j = 0;
    for out in &ext_out[..nb_ext] {
        assert!(
            out.frame >= prev_frame,
            "expected parsed extensions to be returned in frame order"
        );
        if out.frame > prev_frame {
            j = 0;
        }
        while j < nb_ext && ext_in[j].frame != out.frame {
            j += 1;
        }
        assert!(j < nb_ext, "expected enough extensions matching this frame");
        assert_eq!(ext_in[j].id, out.id, "expected extension IDs to match");
        assert_eq!(
            ext_in[j].data.len(),
            out.data.len(),
            "expected extension lengths to match"
        );
        assert_eq!(ext_in[j].data, out.data, "expected extension data to match");
        prev_frame = out.frame;
        j += 1;
    }
}

const NB_EXT: usize = 13;

/// `count_ext` + `parse_ext` + `check_ext_data` of `test_extensions_repeating`.
fn check_repeating(packet: &[u8], e: &[Extension<'_>], nb_ext: usize) {
    let mut nb_frame_exts = [0i32; 48];
    let result = count_ext(packet, &mut nb_frame_exts[..3]);
    assert_eq!(result as usize, nb_ext, "expected extension count to match");
    let mut ext_out = [Extension::default(); NB_EXT];
    let nb_ext_out = parse_ext(packet, &mut ext_out, &nb_frame_exts[..3])
        .expect("expected extension parsing to succeed");
    assert_eq!(nb_ext_out, nb_ext, "expected extension count to match");
    check_ext_data(e, &ext_out, nb_ext);
}

/// Port of `test_extensions_repeating`.
#[test]
fn repeating() {
    let e: [Extension<'static>; NB_EXT] = [
        ext(3, 0, b"a"),
        ext(3, 1, b"b"),
        ext(3, 2, b"c"),
        ext(4, 0, b"d"),
        ext(4, 1, b""),
        ext(4, 2, b""),
        ext(32, 2, b"DRED2"),
        ext(32, 1, b"DRED"),
        ext(5, 1, b""),
        ext(5, 2, b""),
        ext(6, 2, b"f"),
        ext(6, 1, b"e"),
        ext(32, 2, b"DREDthree"),
    ];
    let encoded_len: [usize; NB_EXT + 1] = [
        0, 2,
        // nb_ext = 2: don't try to repeat if the same extension is not used in every frame
        5, // nb_ext = 3: do repeat if the same extension is used in every frame
        5, 7, 9,
        // nb_ext = 6: do not repeat short extensions if the lengths do not match ... but do
        // repeat after the first frame when the lengths do match.
        10,
        // nb_ext = 7: code repeated extensions with L=0 to skip a frame separator because they
        // are all short extensions.
        16,
        // nb_ext = 8: repeat multiple extensions in the same frame. code the last repeated
        // extension with L=0 if it is a long extension.
        21, 23,
        // nb_ext = 10: code the last repeated long extension with L=0 even if it is followed
        // by repeated short extensions with L=0.
        22,
        // nb_ext = 11: don't use L=0 to skip a frame separator if repeats end on a short
        // extension if there was a preceding L=0 long extension.
        26,
        // nb_ext = 12: code the last repeated long extension with L=0 even if it is followed
        // by repeated short extensions with L=1.
        25,
        // nb_ext = 13: don't use L=0 to skip a frame separator if repeats end on a short
        // extension if there was a preceding L=1 long extension.
        37,
    ];
    for nb_ext in 0..=NB_EXT {
        let mut packet = [0u8; 64];
        let mut len = generate(&mut packet, &e[..nb_ext], 3, false).unwrap();
        assert_eq!(
            len, encoded_len[nb_ext],
            "expected extension encoding length to match (nb_ext={nb_ext})"
        );
        check_repeating(&packet[..len], &e, nb_ext);
        // Special case some modifications to test things our generator will never produce.
        match nb_ext {
            6 => {
                // allow a repeat in the last frame, as well as trailing junk after an L=0
                // repeat that MUST be ignored.
                packet[len] = 2 << 1;
                packet[len + 1] = 3 << 1;
                len += 2;
            }
            8 => {
                // insert padding before the repeat indicator
                packet.copy_within(15..len, 16);
                packet[15] = 0x1;
                len += 1;
                // don't repeat the padding and continue decoding last extension with L=0
            }
            10 => {
                // use a frame separator with an increment of 0 as padding. This should _not_
                // change which extensions get repeated.
                packet.copy_within(15..len, 17);
                packet[15] = 0x3;
                packet[16] = 0;
                len += 2;
            }
            13 => {
                // use L=0 to skip a frame separator if a repeat has no extensions at all
                packet[26] = 2 << 1;
            }
            _ => continue,
        }
        check_repeating(&packet[..len], &e, nb_ext);
        if nb_ext == 8 {
            // allow multiple repeat indicators in the same frame
            packet.copy_within(9..len, 10);
            packet[9] = (2 << 1) | 1;
            len += 1;
            // even when there are no new extensions to repeat
            packet.copy_within(5..len, 6);
            packet[5] = (2 << 1) | 1;
            len += 1;
        } else {
            continue;
        }
        check_repeating(&packet[..len], &e, nb_ext);
    }
}

const MAX_EXTENSION_SIZE: usize = 200;
/// The worst case is a 48-frame packet filled entirely with 0-byte short extensions followed
/// by an RTE.
const MAX_NB_EXTENSIONS: u32 = ((MAX_EXTENSION_SIZE - 1) * 48) as u32;

/// Port of `test_random_extensions_parse` with `iterations` iterations (C: 100 M).
fn random_extensions_parse(iterations: u64, rng: &mut FastRand) {
    let mut payload = [0u8; MAX_EXTENSION_SIZE];
    let mut payload2 = [0u8; MAX_EXTENSION_SIZE + 1];
    let mut nb_frame_exts = [0i32; 48];
    let mut ok = 0u64;
    for _ in 0..iterations {
        let len = (rng.next() % (MAX_EXTENSION_SIZE as u32 + 1)) as usize;
        for b in &mut payload[..len] {
            *b = (rng.next() & 0xFF) as u8;
        }
        let capacity = (rng.next() % (MAX_NB_EXTENSIONS + 1)) as usize;
        // The success rate is near 60.4% almost independent of the number of frames we
        // choose, so pick it randomly.
        let nb_frames = (rng.next() % 48) as i32 + 1;
        let payload = &payload[..len];
        // `count` stops at the first malformed extension, like `parse`; a capacity beyond
        // `total + 1` behaves exactly like the C capacity.
        let total = count(payload, nb_frames) as usize;
        let mut ext_out = vec![Extension::default(); capacity.min(total + 1)];
        let result = parse(payload, &mut ext_out, nb_frames);
        // The extensions C reports (`nb_ext`) for each outcome.
        let nb_ext = match result {
            Ok(n) => {
                assert_eq!(n, total);
                n
            }
            Err(Error::BufferTooSmall) => {
                assert!(capacity <= total);
                capacity
            }
            Err(Error::InvalidPacket) => total,
            Err(e) => {
                panic!("expected OPUS_OK, OPUS_BUFFER_TOO_SMALL or OPUS_INVALID_PACKET, got {e:?}")
            }
        };
        // Even if parsing fails, check that the extensions that got extracted make sense.
        for x in &ext_out[..nb_ext] {
            assert!(
                x.frame >= 0 && x.frame < nb_frames,
                "expected frame between 0 and nb_frames-1"
            );
            assert!((2..=127).contains(&x.id), "expected id between 2 and 127");
            // expected data to be within packet
            let _offset = offset_in(payload, x.data);
        }
        // If parsing succeeds, check to see if we can round-trip these extensions.
        if result.is_ok() {
            ok += 1;
            let len2 = generate(&mut payload2, &ext_out[..nb_ext], nb_frames, false)
                .expect("expected extension generation to succeed");
            let nbf = nb_frames as usize;
            let result = count_ext(&payload2[..len2], &mut nb_frame_exts[..nbf]);
            assert_eq!(result as usize, nb_ext, "expected extension count to match");
            let mut ext_out2 = vec![Extension::default(); nb_ext + 1];
            let nb_ext_out = parse_ext(&payload2[..len2], &mut ext_out2, &nb_frame_exts[..nbf])
                .expect("expected extension parsing to succeed");
            assert_eq!(nb_ext, nb_ext_out, "expected extension count to match");
            check_ext_data(&ext_out, &ext_out2, nb_ext);
        }
    }
    // Roughly 60% of random paddings parse (see the C comment above).
    assert!(ok * 2 > iterations, "{ok} of {iterations} parsed");
}

/// The seeded generator of `main` (`fast_rand() % 65535` is printed with the seed), advanced
/// past the deterministic tests (which draw nothing).
fn main_rng() -> FastRand {
    let mut rng = FastRand::new(common::seed());
    rng.next();
    rng
}

#[test]
fn random_extensions() {
    let n = if common::full_length() {
        100_000_000
    } else {
        2_000_000
    };
    random_extensions_parse(n, &mut main_rng());
}

#[test]
#[ignore = "full libopus iteration count (100 M); run with --ignored"]
fn random_extensions_parse_full() {
    random_extensions_parse(100_000_000, &mut main_rng());
}

/// Port of `test_opus_repacketizer_out_range_impl`.
#[test]
fn repacketizer_out_range_impl() {
    let e = [ext(33, 0, b"abcdefg"), ext(100, 0, b"uvwxyz")];
    let mut packet = [0u8; 1024];
    let mut packet_out = [0u8; 1024];

    // Hybrid Packet with 20 msec frames, Code 3
    packet[0] = (15 << 3) | 3;
    // Code 3, padding bit set, 1 frame
    packet[1] = (1 << 6) | 1;
    packet[2] = 0;
    packet[3] = 0;

    // generate 2 extensions, id 33 and 100
    let len = generate(&mut packet[4..], &e, 1, false).unwrap();
    // update the padding length
    packet[2] = len as u8;
    let first = packet;
    // for the middle frame, no padding, no extensions
    packet[1] = 1;
    let middle = packet;
    // switch back to extensions for the last frame extensions
    packet[1] = (1 << 6) | 1;
    let last = packet;

    // concatenate 3 frames
    let mut rp = Repacketizer::new();
    rp.cat(&first[..4 + len]).unwrap();
    rp.cat(&middle[..4]).unwrap();
    rp.cat(&last[..4 + len]).unwrap();

    assert_eq!(rp.nb_frames(), 3, "Expected 3 frames");
    let res = rp
        .out_range_impl(0, 3, &mut packet_out, false, false, &[])
        .unwrap();
    assert!(res > 0, "expected valid packet length");

    // now verify that we have the expected extensions
    let parsed = parse_impl(&packet_out[..res as usize], false).unwrap();
    let mut ext_out = [Extension::default(); 10];
    let nb_ext = parse(parsed.padding, &mut ext_out, 3).unwrap();
    assert_eq!(nb_ext, 4, "Expected 4 extensions");
    let mut first_count = 0;
    let mut second_count = 0;
    for (i, x) in ext_out[..nb_ext].iter().enumerate() {
        if x.id == 33 {
            assert_eq!(x.data, e[0].data);
            first_count += 1;
        } else if x.id == 100 {
            assert_eq!(x.data, e[1].data);
            second_count += 1;
        }
        assert_eq!(x.frame, if i < 2 { 0 } else { 2 });
    }
    assert_eq!(first_count, 2);
    assert_eq!(second_count, 2);
}

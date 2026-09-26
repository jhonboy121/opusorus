//! Port of libopus `tests/test_opus_api.c`: the API invariants of the decoder, multistream
//! decoder, packet parser, encoder and repacketizer (sane options are accepted, insane options
//! are rejected, nothing blows up), through the public opusorus API.
//!
//! Adaptations (C API misuse that cannot be expressed in Rust):
//! * NULL out-pointers for `*_GET_*` CTLs (`OPUS_BAD_ARG` in C): Rust getters return values.
//! * Negative packet lengths / frame sizes (`OPUS_BAD_ARG` in C): lengths are slice lengths and
//!   frame sizes `usize`.
//! * `*_create(..., NULL)` without an error pointer: Rust constructors always return a
//!   `Result`, so the call with and without the error pointer is the same call.
//! * `opus_*_init` on uninitialized `malloc`ed memory: `init` is called on a live object, and
//!   additionally checked to leave it unchanged on error.
//! * The `memcmp` of a decoder before/after `OPUS_RESET_STATE`: the state is opaque, so the
//!   observable state (last packet duration, bandwidth) is compared instead.
//! * `test_malloc_fail` (glibc `__malloc_hook`): allocation failure aborts in Rust; the C test
//!   also skips it on every modern libc.
//! * The integer applications (`OPUS_AUTO`, `OPUS_UNIMPLEMENTED`) passed as `application`:
//!   [`Application::from_raw`] rejects them with `BadArg` before an encoder can be created.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::{FastRand, GlibcRand, debruijn2, offset_in};
use opusorus::constants::raw::*;
use opusorus::decoder::{
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_GAIN_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_PITCH_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST,
    OPUS_RESET_STATE, OPUS_SET_GAIN_REQUEST,
};
use opusorus::encoder::encoder_get_size;
use opusorus::encoder::request as req;
use opusorus::packet;
use opusorus::repacketizer::{
    Repacketizer, multistream_packet_pad, multistream_packet_unpad, packet_pad, packet_unpad,
};
use opusorus::{Application, Bandwidth, Bitrate, Decoder, Encoder, Error, FrameSize, MsDecoder};

/// `OPUS_UNIMPLEMENTED` used as a CTL request number.
const OPUS_UNIMPLEMENTED: i32 = -5;

const OPUS_RATES: [i32; 5] = [48000, 24000, 16000, 12000, 8000];

/// Whether `fs` is a supported API sampling rate.
const fn valid_rate(fs: i32) -> bool {
    matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000) || (cfg!(feature = "qext") && fs == 96000)
}

/// The sampling rates of the "unsupported sample rates" loops: `i` in `-7..=96000` with
/// `-5`, `-6`, `-7` replaced by `-8000`, `INT32_MAX`, `INT32_MIN`.
fn rate_sweep() -> impl Iterator<Item = i32> {
    (-7..=96000).map(|i| match i {
        -5 => -8000,
        -6 => i32::MAX,
        -7 => i32::MIN,
        _ => i,
    })
}

#[test]
fn common_helpers_match_c() {
    // glibc rand() without srand().
    let mut g = GlibcRand::new(1);
    assert_eq!(g.next(), 1_804_289_383);
    assert_eq!(g.next(), 846_930_886);
    assert_eq!(g.next(), 1_681_692_777);
    // fast_rand from seed 0 is stuck at 0 (test_opus_projection relies on it).
    let mut r = FastRand::new(0);
    assert_eq!(r.next(), 0);
    // A De Bruijn sequence contains every pair exactly once (cyclically).
    for k in [6usize, 64] {
        let s = debruijn2(k);
        let mut seen = vec![0u32; k * k];
        for i in 0..k * k {
            seen[usize::from(s[i]) * k + usize::from(s[(i + 1) % (k * k)])] += 1;
        }
        assert!(seen.iter().all(|&n| n == 1), "k={k}");
    }
}

/// `main`: version string and `opus_strerror`.
#[test]
fn strerror() {
    for code in [-1, -2, -3, -4, -5, -6, -7, -32768] {
        let e = Error::from_code(code).expect("negative codes are errors");
        assert!(!e.as_str().is_empty());
        assert_eq!(e.to_string(), e.as_str());
    }
    assert_eq!(Error::from_code(-32768), Some(Error::InternalError));
    assert_eq!(Error::from_code(32767), None);
    assert_eq!(Error::from_code(0), None);
    for e in [
        Error::BadArg,
        Error::BufferTooSmall,
        Error::InternalError,
        Error::InvalidPacket,
        Error::Unimplemented,
        Error::InvalidState,
        Error::AllocFail,
    ] {
        assert_eq!(Error::from_code(e.code()), Some(e));
    }
}

/// Port of `test_dec_api`.
#[test]
fn dec_api() {
    let mut sbuf = [0i16; 960 * 2];
    #[cfg(not(feature = "disable-float-api"))]
    let mut fbuf = [0f32; 960 * 2];
    let mut packet = [0u8; 1276];

    for c in 0..4 {
        let i = Decoder::get_size(c);
        if c == 1 || c == 2 {
            assert!(i > 2048 && i <= 1 << 18, "opus_decoder_get_size({c})={i}");
        } else {
            assert_eq!(i, 0);
        }
    }

    // Test with unsupported sample rates.
    let mut live = Decoder::new(48000, 2).unwrap();
    let live_rate = live.sample_rate();
    for c in 0..4 {
        for fs in rate_sweep() {
            if valid_rate(fs) && (c == 1 || c == 2) {
                continue;
            }
            assert_eq!(Decoder::new(fs, c).err(), Some(Error::BadArg));
            assert_eq!(live.init(fs, c), Err(Error::BadArg));
        }
    }
    assert_eq!(live.sample_rate(), live_rate);
    assert_eq!(live.channels(), 2);

    let mut dec = Decoder::new(48000, 2).unwrap();

    // OPUS_GET_FINAL_RANGE
    let _range: u32 = dec.final_range();
    assert!(dec.ctl_get(OPUS_GET_FINAL_RANGE_REQUEST).is_ok());

    // OPUS_UNIMPLEMENTED
    assert_eq!(
        dec.ctl_set(OPUS_UNIMPLEMENTED, 0),
        Err(Error::Unimplemented)
    );
    assert_eq!(dec.ctl_get(OPUS_UNIMPLEMENTED), Err(Error::Unimplemented));

    // OPUS_GET_BANDWIDTH
    assert_eq!(dec.bandwidth(), None);
    assert_eq!(dec.ctl_get(OPUS_GET_BANDWIDTH_REQUEST), Ok(0));

    // OPUS_GET_SAMPLE_RATE
    assert_eq!(dec.sample_rate(), 48000);
    assert_eq!(dec.ctl_get(OPUS_GET_SAMPLE_RATE_REQUEST), Ok(48000));

    // GET_PITCH has different execution paths depending on the previously decoded frame.
    let pitch_ok = |p: i32| (-1..=0).contains(&p);
    assert!(pitch_ok(dec.pitch()));
    packet[0] = 63 << 2;
    packet[1] = 0;
    packet[2] = 0;
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 960, false),
        Ok(960)
    );
    assert!(pitch_ok(dec.pitch()));
    assert!(pitch_ok(dec.ctl_get(OPUS_GET_PITCH_REQUEST).unwrap()));
    packet[0] = 1;
    assert_eq!(
        dec.decode(Some(&packet[..1]), &mut sbuf, 960, false),
        Ok(960)
    );
    assert!(pitch_ok(dec.pitch()));

    // OPUS_GET_LAST_PACKET_DURATION
    assert_eq!(dec.last_packet_duration(), 960);
    assert_eq!(dec.ctl_get(OPUS_GET_LAST_PACKET_DURATION_REQUEST), Ok(960));

    // OPUS_SET_GAIN / OPUS_GET_GAIN
    assert_eq!(dec.gain(), 0);
    assert_eq!(dec.set_gain(-32769), Err(Error::BadArg));
    assert_eq!(dec.set_gain(32768), Err(Error::BadArg));
    assert_eq!(
        dec.ctl_set(OPUS_SET_GAIN_REQUEST, -32769),
        Err(Error::BadArg)
    );
    assert_eq!(
        dec.ctl_set(OPUS_SET_GAIN_REQUEST, 32768),
        Err(Error::BadArg)
    );
    assert_eq!(dec.set_gain(-15), Ok(()));
    assert_eq!(dec.gain(), -15);
    assert_eq!(dec.ctl_get(OPUS_GET_GAIN_REQUEST), Ok(-15));

    // Reset the decoder: the state must change (C compares the raw memory).
    let before = (dec.last_packet_duration(), dec.bandwidth());
    assert_eq!(dec.ctl_set(OPUS_RESET_STATE, 0), Ok(()));
    let after = (dec.last_packet_duration(), dec.bandwidth());
    assert_ne!(before, after);
    assert_eq!(after, (0, None));
    // The settings survive the reset.
    assert_eq!(dec.gain(), -15);
    dec.reset();

    packet[0] = 0;
    assert_eq!(dec.nb_samples(&packet[..1]), Ok(480));
    assert_eq!(packet::get_nb_samples(&packet[..1], 48000), Ok(480));
    assert_eq!(packet::get_nb_samples(&packet[..1], 96000), Ok(960));
    assert_eq!(packet::get_nb_samples(&packet[..1], 32000), Ok(320));
    assert_eq!(packet::get_nb_samples(&packet[..1], 8000), Ok(80));
    packet[0] = 3;
    assert_eq!(
        packet::get_nb_samples(&packet[..1], 24000),
        Err(Error::InvalidPacket)
    );
    packet[0] = (63 << 2) | 3;
    packet[1] = 63;
    assert_eq!(
        packet::get_nb_samples(&packet[..0], 24000),
        Err(Error::BadArg)
    );
    assert_eq!(
        packet::get_nb_samples(&packet[..2], 48000),
        Err(Error::InvalidPacket)
    );
    assert_eq!(dec.nb_samples(&packet[..2]), Err(Error::InvalidPacket));

    // opus_packet_get_nb_frames()
    assert_eq!(packet::get_nb_frames(&packet[..0]), Err(Error::BadArg));
    for i in 0..256 {
        let l1res = [Ok(1), Ok(2), Ok(2), Err(Error::InvalidPacket)];
        packet[0] = i as u8;
        assert_eq!(packet::get_nb_frames(&packet[..1]), l1res[i & 3]);
        for j in 0..256 {
            packet[1] = j as u8;
            let expected = if i & 3 != 3 { l1res[i & 3] } else { Ok(j & 63) };
            assert_eq!(packet::get_nb_frames(&packet[..2]), expected);
        }
    }

    // opus_packet_get_bandwidth()
    for i in 0..256i32 {
        packet[0] = i as u8;
        let bw = i >> 4;
        let bw = OPUS_BANDWIDTH_NARROWBAND
            + (((((bw & 7) * 9) & (63 - (bw & 8))) + 2 + 12 * i32::from((bw & 8) != 0)) >> 4);
        assert_eq!(
            packet::get_bandwidth(&packet[..1]).map(Bandwidth::to_raw),
            Ok(bw)
        );
        assert_eq!(packet::toc_bandwidth(packet[0]).to_raw(), bw);
    }

    // opus_packet_get_samples_per_frame()
    for i in 0..256i32 {
        packet[0] = i as u8;
        let fp3s = i >> 3;
        let fp3s = (((((3 - (fp3s & 3)) * 13) & 119) + 9) >> 2)
            * (i32::from(fp3s > 13) * (3 - i32::from((fp3s & 3) == 3)) + 1)
            * 25;
        for rate in OPUS_RATES {
            assert_eq!(
                packet::get_samples_per_frame(&packet[..1], rate),
                Ok(rate * 3 / fp3s)
            );
        }
    }

    packet[0] = (63 << 2) + 3;
    packet[1] = 49;
    packet[2..51].fill(0);
    assert_eq!(
        dec.decode(Some(&packet[..51]), &mut sbuf, 960, false),
        Err(Error::InvalidPacket)
    );
    packet[0] = 63 << 2;
    packet[1] = 0;
    packet[2] = 0;
    // Skipped: `opus_decode(dec, packet, -1, ...)` (negative length) cannot be expressed.
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 60, false),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 480, false),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 960, false),
        Ok(960)
    );
    #[cfg(not(feature = "disable-float-api"))]
    assert_eq!(
        dec.decode_float(Some(&packet[..3]), &mut fbuf, 960, false),
        Ok(960)
    );
    let mut ibuf = [0i32; 960 * 2];
    assert_eq!(
        dec.decode24(Some(&packet[..3]), &mut ibuf, 960, false),
        Ok(960)
    );
}

/// Port of `test_msdec_api`.
#[test]
fn msdec_api() {
    let mut sbuf = [0i16; 960 * 2];
    #[cfg(not(feature = "disable-float-api"))]
    let mut fbuf = [0f32; 960 * 2];
    let mut packet = [0u8; 1276];
    let mut mapping = [0u8; 256];
    mapping[0] = 0;
    mapping[1] = 1;

    for a in -1..4 {
        for b in -1..4 {
            let i = MsDecoder::get_size(a, b);
            if a > 0 && b <= a && b >= 0 {
                assert!(
                    i > 2048 && i <= (1 << 18) * a as usize,
                    "opus_multistream_decoder_get_size({a},{b})={i}"
                );
            } else {
                assert_eq!(i, 0, "opus_multistream_decoder_get_size({a},{b})");
            }
        }
    }

    // Test with unsupported sample rates.
    let mut live = MsDecoder::new(48000, 1, 1, 0, &mapping).unwrap();
    for c in 1..3 {
        for fs in rate_sweep() {
            if valid_rate(fs) {
                continue;
            }
            assert_eq!(
                MsDecoder::new(fs, c, 1, c - 1, &mapping).err(),
                Some(Error::BadArg)
            );
            assert_eq!(live.init(fs, c, 1, c - 1, &mapping), Err(Error::BadArg));
        }
    }
    assert_eq!(live.channels(), 1);
    assert_eq!(live.sample_rate(), 48000);

    // C runs this block twice (with and without an error pointer); it is the same call in Rust.
    {
        mapping[0] = 0;
        mapping[1] = 1;
        assert_eq!(
            MsDecoder::new(48000, 2, 1, 0, &mapping).err(),
            Some(Error::BadArg)
        );

        mapping[0] = 0;
        mapping[1] = 0;
        assert!(MsDecoder::new(48000, 2, 1, 0, &mapping).is_ok());

        let mut dec = MsDecoder::new(48000, 1, 4, 1, &mapping).unwrap();
        assert_eq!(dec.init(48000, 1, 0, 0, &mapping), Err(Error::BadArg));
        assert_eq!(dec.init(48000, 1, 1, -1, &mapping), Err(Error::BadArg));
        drop(dec);

        assert!(MsDecoder::new(48000, 2, 1, 1, &mapping).is_ok());
        for (c, s, cs) in [
            (255, 255, 1),
            (-1, 1, 1),
            (0, 1, 1),
            (1, -1, 2),
            (1, -1, -1),
            (256, 255, 1),
            (256, 255, 0),
        ] {
            assert_eq!(
                MsDecoder::new(48000, c, s, cs, &mapping).err(),
                Some(Error::BadArg),
                "channels={c} streams={s} coupled={cs}"
            );
        }

        mapping[..3].copy_from_slice(&[255, 1, 2]);
        assert_eq!(
            MsDecoder::new(48000, 3, 2, 0, &mapping).err(),
            Some(Error::BadArg)
        );

        mapping[..3].copy_from_slice(&[0, 0, 0]);
        assert!(MsDecoder::new(48000, 3, 2, 1, &mapping).is_ok());

        mapping[..5].copy_from_slice(&[0, 255, 1, 2, 3]);
        assert_eq!(
            MsDecoder::new(48001, 5, 4, 1, &mapping).err(),
            Some(Error::BadArg)
        );
    }

    mapping[..4].copy_from_slice(&[0, 255, 1, 2]);
    let mut dec = MsDecoder::new(48000, 4, 2, 1, &mapping).unwrap();

    // OPUS_GET_FINAL_RANGE
    let _range: u32 = dec.final_range();
    assert!(dec.ctl_get(OPUS_GET_FINAL_RANGE_REQUEST).is_ok());

    // OPUS_MULTISTREAM_GET_DECODER_STATE
    assert_eq!(dec.decoder_state(-1).err(), Some(Error::BadArg));
    assert!(dec.decoder_state(1).is_ok());
    assert_eq!(dec.decoder_state(2).err(), Some(Error::BadArg));
    assert!(dec.decoder_state(0).is_ok());

    for j in 0..2 {
        assert_eq!(dec.decoder_state(j).unwrap().gain(), 0);
    }
    assert_eq!(dec.set_gain(15), Ok(()));
    for j in 0..2 {
        let od = dec.decoder_state(j).unwrap();
        assert_eq!(od.gain(), 15);
        assert_eq!(od.ctl_get(OPUS_GET_GAIN_REQUEST), Ok(15));
    }

    // OPUS_GET_BANDWIDTH
    assert_eq!(dec.bandwidth(), None);
    assert_eq!(dec.ctl_get(OPUS_GET_BANDWIDTH_REQUEST), Ok(0));

    // OPUS_UNIMPLEMENTED
    assert_eq!(
        dec.ctl_set(OPUS_UNIMPLEMENTED, 0),
        Err(Error::Unimplemented)
    );
    assert_eq!(dec.ctl_get(OPUS_UNIMPLEMENTED), Err(Error::Unimplemented));

    // (OPUS_GET_PITCH is "currently unimplemented for multistream" and disabled in C.)

    // Reset the decoder.
    assert_eq!(dec.ctl_set(OPUS_RESET_STATE, 0), Ok(()));
    dec.reset();
    drop(dec);

    let mut dec = MsDecoder::new(48000, 2, 1, 1, &mapping).unwrap();
    packet[0] = (63 << 2) + 3;
    packet[1] = 49;
    packet[2..51].fill(0);
    assert_eq!(
        dec.decode(Some(&packet[..51]), &mut sbuf, 960, false),
        Err(Error::InvalidPacket)
    );
    packet[0] = 63 << 2;
    packet[1] = 0;
    packet[2] = 0;
    // Skipped: negative length and negative frame size (-960) cannot be expressed.
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 60, false),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 480, false),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 960, false),
        Ok(960)
    );
    #[cfg(not(feature = "disable-float-api"))]
    assert_eq!(
        dec.decode_float(Some(&packet[..3]), &mut fbuf, 960, false),
        Ok(960)
    );
    // A zero frame size is the closest Rust equivalent of the C negative one.
    assert_eq!(
        dec.decode(Some(&packet[..3]), &mut sbuf, 0, false),
        Err(Error::BadArg)
    );
}

/// Checks the result of a parse call: `Ok` with `nb_frames` frames or the given error.
fn parse(data: &[u8]) -> Result<packet::ParsedPacket<'_>, Error> {
    packet::parse(data)
}

/// The frame sizes of a parsed packet (the C `size` array).
fn sizes(p: &packet::ParsedPacket<'_>) -> Vec<usize> {
    p.frames().iter().map(|f| f.len()).collect()
}

/// Checks that the frames of `p` are contiguous in `buf` (`frames[j]==frames[j-1]+size[j-1]`).
fn assert_contiguous(buf: &[u8], p: &packet::ParsedPacket<'_>) {
    let f = p.frames();
    for jj in 1..f.len() {
        assert_eq!(
            offset_in(buf, f[jj]),
            offset_in(buf, f[jj - 1]) + f[jj - 1].len()
        );
    }
}

/// The largest length any parse call below uses (the C test passes lengths beyond its
/// 1276-byte buffer, relying on the parser not reading the payload).
const PARSE_BUF: usize = 70000;

/// Port of `test_parse`, part 1: code 0, 1 and 2 packets.
#[test]
fn parse_code_0_1_2() {
    let mut packet = vec![0u8; PARSE_BUF];
    // Skipped: `opus_packet_parse(packet,1,&toc,frames,0,&payload_offset)` (NULL size array).

    // code 0
    for i in 0..64u8 {
        packet[0] = i << 2;
        let p = parse(&packet[..4]).unwrap();
        assert_eq!(p.nb_frames, 1);
        assert_eq!(p.frames[0].len(), 3);
        assert_eq!(offset_in(&packet, p.frames[0]), 1);
        assert_eq!(p.payload_offset, 1);
    }

    // code 1, two frames of the same size
    for i in 0..64u8 {
        packet[0] = (i << 2) + 1;
        for jj in 0..=1275 * 2 + 3 {
            let r = parse(&packet[..jj]);
            if (jj & 1) == 1 && jj <= 2551 {
                // Must pass if payload length even (packet length odd) and size<=2551, must
                // fail otherwise.
                let p = r.unwrap();
                assert_eq!(p.nb_frames, 2);
                assert_eq!(sizes(&p), [(jj - 1) >> 1; 2]);
                assert_eq!(offset_in(&packet, p.frames[0]), 1);
                assert_contiguous(&packet, &p);
                assert_eq!(p.toc >> 2, i);
            } else {
                assert_eq!(r.err(), Some(Error::InvalidPacket), "jj={jj}");
            }
        }
    }

    for i in 0..64u8 {
        // code 2, length code overflow
        packet[0] = (i << 2) + 2;
        assert_eq!(parse(&packet[..1]).err(), Some(Error::InvalidPacket));
        packet[1] = 252;
        assert_eq!(parse(&packet[..2]).err(), Some(Error::InvalidPacket));
        for j in 0..1275usize {
            if j < 252 {
                packet[1] = j as u8;
            } else {
                packet[1] = (252 + (j & 3)) as u8;
                packet[2] = ((j - 252) >> 2) as u8;
            }
            let hdr = if j < 252 { 2 } else { 3 };
            // Code 2, one too short
            assert_eq!(
                parse(&packet[..j + hdr - 1]).err(),
                Some(Error::InvalidPacket)
            );
            // Code 2, one too long
            assert_eq!(
                parse(&packet[..j + hdr + 1276]).err(),
                Some(Error::InvalidPacket)
            );
            // Code 2, second zero
            let p = parse(&packet[..j + hdr]).unwrap();
            assert_eq!(p.nb_frames, 2);
            assert_eq!(sizes(&p), [j, 0]);
            assert_contiguous(&packet, &p);
            assert_eq!(p.toc >> 2, i);
            // Code 2, normal
            let p = parse(&packet[..(j << 1) + 4]).unwrap();
            assert_eq!(p.nb_frames, 2);
            assert_eq!(sizes(&p), [j, (j << 1) + 3 - j - (hdr - 1)]);
            assert_contiguous(&packet, &p);
            assert_eq!(p.toc >> 2, i);
        }
    }
}

/// Port of `test_parse`, part 2: code 3 m-truncation, invalid m, m=1 CBR and the m>1 CBR sweep
/// (for every frame count, every packet size up to `(m+2)*1275`: ~100 M parses).
#[test]
fn parse_code3_cbr() {
    let mut packet = vec![0u8; PARSE_BUF];

    // code 3, length code overflow
    for i in 0..64u8 {
        packet[0] = (i << 2) + 3;
        assert_eq!(parse(&packet[..1]).err(), Some(Error::InvalidPacket));
    }

    // code 3, m is zero or 49-63
    for i in 0..64u8 {
        packet[0] = (i << 2) + 3;
        for jj in 49..=64u8 {
            for flags in [0u8, 128, 64, 128 + 64] {
                // CBR/VBR, with/without padding
                packet[1] = flags + (jj & 63);
                assert_eq!(parse(&packet[..1275]).err(), Some(Error::InvalidPacket));
            }
        }
    }

    // code 3, m is one, cbr
    for i in 0..64u8 {
        packet[0] = (i << 2) + 3;
        packet[1] = 1;
        for j in 0..1276 {
            let p = parse(&packet[..j + 2]).unwrap();
            assert_eq!(p.nb_frames, 1);
            assert_eq!(p.frames[0].len(), j);
            assert_eq!(p.toc >> 2, i);
        }
        assert_eq!(parse(&packet[..1276 + 2]).err(), Some(Error::InvalidPacket));
    }

    // code 3, m>1 CBR
    for i in 0..64u8 {
        packet[0] = (i << 2) + 3;
        let frame_samp = packet::get_samples_per_frame(&packet[..1], 48000).unwrap() as usize;
        for j in 2..49usize {
            packet[1] = j as u8;
            for sz in 2..(j + 2) * 1275 {
                let r = parse(&packet[..sz]);
                // Must be <=120ms, must be evenly divisible, can't have frames>1275 bytes
                if frame_samp * j <= 5760 && (sz - 2) % j == 0 && (sz - 2) / j < 1276 {
                    let p = r.unwrap();
                    assert_eq!(p.nb_frames, j);
                    assert_contiguous(&packet, &p);
                    assert_eq!(p.toc >> 2, i);
                } else {
                    assert_eq!(r.err(), Some(Error::InvalidPacket));
                }
            }
        }
        // Super jumbo packets
        let m = 5760 / frame_samp;
        packet[1] = m as u8;
        let p = parse(&packet[..1275 * m + 2]).unwrap();
        assert_eq!(p.nb_frames, m);
        assert!(sizes(&p).iter().all(|&s| s == 1275));
    }
}

/// Port of `test_parse`, part 3: code 3 VBR and padding.
#[test]
fn parse_code3_vbr_padding() {
    let mut packet = vec![0u8; PARSE_BUF];

    for i in 0..64usize {
        // Code 3 VBR, m one
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = 128 + 1;
        let frame_samp = packet::get_samples_per_frame(&packet[..1], 48000).unwrap() as usize;
        for jj in 0..1276 {
            let p = parse(&packet[..2 + jj]).unwrap();
            assert_eq!(p.nb_frames, 1);
            assert_eq!(p.frames[0].len(), jj);
            assert_eq!(usize::from(p.toc >> 2), i);
        }
        assert_eq!(parse(&packet[..2 + 1276]).err(), Some(Error::InvalidPacket));
        for j in 2..49usize {
            packet[1] = (128 + j) as u8;
            // Length code overflow
            assert_eq!(
                parse(&packet[..2 + j - 2]).err(),
                Some(Error::InvalidPacket)
            );
            packet[2] = 252;
            packet[3] = 0;
            packet[4..2 + j].fill(0);
            assert_eq!(parse(&packet[..2 + j]).err(), Some(Error::InvalidPacket));
            // One byte too short
            packet[2..2 + j].fill(0);
            assert_eq!(
                parse(&packet[..2 + j - 2]).err(),
                Some(Error::InvalidPacket)
            );
            // One byte too short thanks to length coding
            packet[2] = 252;
            packet[3] = 0;
            packet[4..2 + j].fill(0);
            assert_eq!(
                parse(&packet[..2 + j + 252 - 1]).err(),
                Some(Error::InvalidPacket)
            );
            // Most expensive way of coding zeros
            packet[2..2 + j].fill(0);
            let r = parse(&packet[..2 + j - 1]);
            if frame_samp * j <= 5760 {
                let p = r.unwrap();
                assert_eq!(p.nb_frames, j);
                assert!(sizes(&p).iter().all(|&s| s == 0));
                assert_eq!(usize::from(p.toc >> 2), i);
            } else {
                assert_eq!(r.err(), Some(Error::InvalidPacket));
            }
            // Quasi-CBR use of mode 3
            for tsz in [50usize, 201, 403, 700, 1472, 5110, 20400, 61298] {
                let mut pos = 0;
                let as_ = (tsz + i - j - 2) / j;
                for _ in 0..j - 1 {
                    if as_ < 252 {
                        packet[2 + pos] = as_ as u8;
                        pos += 1;
                    } else {
                        packet[2 + pos] = (252 + (as_ & 3)) as u8;
                        packet[3 + pos] = ((as_ - 252) >> 2) as u8;
                        pos += 2;
                    }
                }
                let r = parse(&packet[..tsz + i]);
                let last = (tsz + i) as isize - 2 - pos as isize - (as_ * (j - 1)) as isize;
                if frame_samp * j <= 5760 && as_ < 1276 && last < 1276 {
                    let p = r.unwrap();
                    assert_eq!(p.nb_frames, j);
                    let s = sizes(&p);
                    assert!(s[..j - 1].iter().all(|&v| v == as_));
                    assert_eq!(s[j - 1] as isize, last);
                    assert_eq!(usize::from(p.toc >> 2), i);
                } else {
                    assert_eq!(r.err(), Some(Error::InvalidPacket));
                }
            }
        }
    }

    for i in 0..64usize {
        packet[0] = ((i << 2) + 3) as u8;
        // Padding
        packet[1] = 128 + 1 + 64;
        // Overflow the length coding
        packet[2..127].fill(255);
        assert_eq!(parse(&packet[..127]).err(), Some(Error::InvalidPacket));

        for (sz, tsz) in [0usize, 72, 512, 1275].into_iter().enumerate() {
            let mut jj = sz;
            while jj < 65025 {
                let mut pos = 0;
                while pos < jj / 254 {
                    packet[2 + pos] = 255;
                    pos += 1;
                }
                packet[2 + pos] = (jj % 254) as u8;
                pos += 1;
                if sz == 0 && i == 63 {
                    // Code more padding than there is room in the packet
                    assert_eq!(
                        parse(&packet[..2 + jj + pos - 1]).err(),
                        Some(Error::InvalidPacket)
                    );
                }
                let r = parse(&packet[..2 + jj + tsz + i + pos]);
                if tsz + i < 1276 {
                    let p = r.unwrap();
                    assert_eq!(p.nb_frames, 1);
                    assert_eq!(p.frames[0].len(), tsz + i);
                    assert_eq!(p.padding.len(), jj);
                    assert_eq!(usize::from(p.toc >> 2), i);
                } else {
                    assert_eq!(r.err(), Some(Error::InvalidPacket));
                }
                jj += 11;
            }
        }
    }
}

/// `CHECK_SETGET`: set-must-fail twice, set-must-pass + get-and-compare twice, through the
/// numeric CTL interface.
fn check_setget(enc: &mut Encoder, set: i32, get: i32, bad: [i32; 2], good: [i32; 2]) {
    for b in bad {
        assert!(enc.ctl_set(set, b).is_err(), "set {set} {b} must fail");
    }
    for g in good {
        assert_eq!(enc.ctl_set(set, g), Ok(()), "set {set} {g}");
        assert_eq!(enc.ctl_get(get), Ok(g), "get {get}");
    }
}

/// Port of `test_enc_api`.
#[test]
fn enc_api() {
    let mut packet = [0u8; 1276];
    let sbuf = [0i16; 960 * 2];
    #[cfg(not(feature = "disable-float-api"))]
    let fbuf = [0f32; 960 * 2];

    for c in 0..4 {
        let i = encoder_get_size(c);
        if c == 1 || c == 2 {
            assert!(i > 2048 && i <= 1 << 18, "opus_encoder_get_size({c})={i}");
        } else {
            assert_eq!(i, 0);
        }
    }

    // Test with unsupported sample rates, channel counts.
    let mut live = Encoder::new(48000, 2, Application::Voip).unwrap();
    for c in 0..4 {
        for fs in rate_sweep() {
            if valid_rate(fs) && (c == 1 || c == 2) {
                continue;
            }
            assert_eq!(
                Encoder::new(fs, c, Application::Voip).err(),
                Some(Error::BadArg)
            );
            assert_eq!(live.init(fs, c, Application::Voip), Err(Error::BadArg));
        }
    }
    assert_eq!((live.sample_rate(), live.channels()), (48000, 2));

    // OPUS_AUTO is not an application.
    assert_eq!(Application::from_raw(OPUS_AUTO), Err(Error::BadArg));

    assert!(Encoder::new(48000, 2, Application::Voip).is_ok());

    let enc = Encoder::new(48000, 2, Application::RestrictedLowDelay).unwrap();
    assert!(enc.lookahead() <= 32766);
    assert_eq!(enc.ctl_get(req::OPUS_GET_LOOKAHEAD_REQUEST), Ok(120));

    let enc = Encoder::new(48000, 2, Application::Audio).unwrap();
    assert!(enc.lookahead() <= 32766);

    let mut enc = Encoder::new(48000, 2, Application::Voip).unwrap();

    // OPUS_GET_LOOKAHEAD
    let i = enc.ctl_get(req::OPUS_GET_LOOKAHEAD_REQUEST).unwrap();
    assert!((0..=32766).contains(&i));
    assert_eq!(i as usize, enc.lookahead());

    // OPUS_GET_SAMPLE_RATE
    assert_eq!(enc.sample_rate(), 48000);
    assert_eq!(enc.ctl_get(req::OPUS_GET_SAMPLE_RATE_REQUEST), Ok(48000));

    // OPUS_UNIMPLEMENTED
    assert_eq!(
        enc.ctl_set(OPUS_UNIMPLEMENTED, 0),
        Err(Error::Unimplemented)
    );
    assert_eq!(enc.ctl_get(OPUS_UNIMPLEMENTED), Err(Error::Unimplemented));

    check_setget(
        &mut enc,
        req::OPUS_SET_APPLICATION_REQUEST,
        req::OPUS_GET_APPLICATION_REQUEST,
        [-1, OPUS_AUTO],
        [OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY],
    );
    assert_eq!(enc.application(), Application::RestrictedLowDelay);

    assert_eq!(
        enc.ctl_set(req::OPUS_SET_BITRATE_REQUEST, 1_073_741_832),
        Ok(())
    );
    let i = enc.ctl_get(req::OPUS_GET_BITRATE_REQUEST).unwrap();
    assert!((256_000..=1_700_000).contains(&i), "bitrate {i}");
    assert_eq!(enc.bitrate(), i);
    check_setget(
        &mut enc,
        req::OPUS_SET_BITRATE_REQUEST,
        req::OPUS_GET_BITRATE_REQUEST,
        [-12345, 0],
        [500, 256_000],
    );
    assert_eq!(enc.set_bitrate(Bitrate::Bits(0)), Err(Error::BadArg));
    assert_eq!(enc.set_bitrate(Bitrate::Bits(256_000)), Ok(()));

    check_setget(
        &mut enc,
        req::OPUS_SET_FORCE_CHANNELS_REQUEST,
        req::OPUS_GET_FORCE_CHANNELS_REQUEST,
        [-1, 3],
        [1, OPUS_AUTO],
    );
    assert_eq!(enc.set_force_channels(Some(3)), Err(Error::BadArg));
    assert_eq!(enc.force_channels(), None);

    for (bw, ok) in [
        (-2, false),
        (OPUS_BANDWIDTH_FULLBAND + 1, false),
        (OPUS_BANDWIDTH_NARROWBAND, true),
        (OPUS_BANDWIDTH_FULLBAND, true),
        (OPUS_BANDWIDTH_WIDEBAND, true),
        (OPUS_BANDWIDTH_MEDIUMBAND, true),
    ] {
        assert_eq!(
            enc.ctl_set(req::OPUS_SET_BANDWIDTH_REQUEST, bw).is_ok(),
            ok,
            "OPUS_SET_BANDWIDTH({bw})"
        );
    }
    // We don't test if the bandwidth has actually changed, because the change may be delayed
    // until the encoder is advanced.
    let i = enc.ctl_get(req::OPUS_GET_BANDWIDTH_REQUEST).unwrap();
    assert!(
        [
            OPUS_BANDWIDTH_NARROWBAND,
            OPUS_BANDWIDTH_MEDIUMBAND,
            OPUS_BANDWIDTH_WIDEBAND,
            OPUS_BANDWIDTH_FULLBAND,
            OPUS_AUTO
        ]
        .contains(&i)
    );
    assert_eq!(
        enc.ctl_set(req::OPUS_SET_BANDWIDTH_REQUEST, OPUS_AUTO),
        Ok(())
    );
    enc.set_bandwidth(None);

    for (bw, ok) in [
        (-2, false),
        (OPUS_BANDWIDTH_FULLBAND + 1, false),
        (OPUS_BANDWIDTH_NARROWBAND, true),
        (OPUS_BANDWIDTH_FULLBAND, true),
        (OPUS_BANDWIDTH_WIDEBAND, true),
        (OPUS_BANDWIDTH_MEDIUMBAND, true),
    ] {
        assert_eq!(
            enc.ctl_set(req::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw).is_ok(),
            ok,
            "OPUS_SET_MAX_BANDWIDTH({bw})"
        );
    }
    let i = enc.ctl_get(req::OPUS_GET_MAX_BANDWIDTH_REQUEST).unwrap();
    assert!(
        [
            OPUS_BANDWIDTH_NARROWBAND,
            OPUS_BANDWIDTH_MEDIUMBAND,
            OPUS_BANDWIDTH_WIDEBAND,
            OPUS_BANDWIDTH_FULLBAND
        ]
        .contains(&i)
    );
    assert_eq!(enc.max_bandwidth(), Bandwidth::Mediumband);

    check_setget(
        &mut enc,
        req::OPUS_SET_DTX_REQUEST,
        req::OPUS_GET_DTX_REQUEST,
        [-1, 2],
        [1, 0],
    );
    check_setget(
        &mut enc,
        req::OPUS_SET_COMPLEXITY_REQUEST,
        req::OPUS_GET_COMPLEXITY_REQUEST,
        [-1, 11],
        [0, 10],
    );
    assert_eq!(enc.set_complexity(11), Err(Error::BadArg));
    assert_eq!(enc.complexity(), 10);
    check_setget(
        &mut enc,
        req::OPUS_SET_INBAND_FEC_REQUEST,
        req::OPUS_GET_INBAND_FEC_REQUEST,
        [-1, 3],
        [1, 0],
    );
    check_setget(
        &mut enc,
        req::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
        req::OPUS_GET_PACKET_LOSS_PERC_REQUEST,
        [-1, 101],
        [100, 0],
    );
    check_setget(
        &mut enc,
        req::OPUS_SET_VBR_REQUEST,
        req::OPUS_GET_VBR_REQUEST,
        [-1, 2],
        [1, 0],
    );
    // (The OPUS_SET_VOICE_RATIO check is commented out in C.)
    check_setget(
        &mut enc,
        req::OPUS_SET_VBR_CONSTRAINT_REQUEST,
        req::OPUS_GET_VBR_CONSTRAINT_REQUEST,
        [-1, 2],
        [1, 0],
    );
    check_setget(
        &mut enc,
        req::OPUS_SET_SIGNAL_REQUEST,
        req::OPUS_GET_SIGNAL_REQUEST,
        [-12345, 0x7FFF_FFFF],
        [OPUS_SIGNAL_MUSIC, OPUS_AUTO],
    );
    check_setget(
        &mut enc,
        req::OPUS_SET_LSB_DEPTH_REQUEST,
        req::OPUS_GET_LSB_DEPTH_REQUEST,
        [7, 25],
        [16, 24],
    );

    assert_eq!(
        enc.ctl_get(req::OPUS_GET_PREDICTION_DISABLED_REQUEST),
        Ok(0)
    );
    assert!(!enc.prediction_disabled());
    check_setget(
        &mut enc,
        req::OPUS_SET_PREDICTION_DISABLED_REQUEST,
        req::OPUS_GET_PREDICTION_DISABLED_REQUEST,
        [-1, 2],
        [1, 0],
    );

    for fs in [
        FrameSize::Ms2_5,
        FrameSize::Ms5,
        FrameSize::Ms10,
        FrameSize::Ms20,
        FrameSize::Ms40,
        FrameSize::Ms60,
        FrameSize::Ms80,
        FrameSize::Ms100,
        FrameSize::Ms120,
    ] {
        assert_eq!(
            enc.ctl_set(req::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, fs.to_raw()),
            Ok(())
        );
        assert_eq!(enc.expert_frame_duration(), fs);
    }
    check_setget(
        &mut enc,
        req::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
        req::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST,
        [0, -1],
        [OPUS_FRAMESIZE_60_MS, OPUS_FRAMESIZE_ARG],
    );
    // (OPUS_SET_FORCE_MODE is not a public API; the encoder tests use it.)

    // OPUS_GET_FINAL_RANGE
    let _range: u32 = enc.final_range();
    assert!(enc.ctl_get(req::OPUS_GET_FINAL_RANGE_REQUEST).is_ok());

    // Reset the encoder
    assert_eq!(enc.ctl_set(req::OPUS_RESET_STATE, 0), Ok(()));
    enc.reset();

    let len = packet.len();
    let i = enc.encode(&sbuf, 960, &mut packet).unwrap();
    assert!((1..=len).contains(&i));
    #[cfg(not(feature = "disable-float-api"))]
    {
        let i = enc.encode_float(&fbuf, 960, &mut packet).unwrap();
        assert!((1..=len).contains(&i));
    }
    let ibuf = [0i32; 960 * 2];
    let i = enc.encode24(&ibuf, 960, &mut packet).unwrap();
    assert!((1..=len).contains(&i));
}

/// `max_out` of `test_repacketizer_api`.
const MAX_OUT: usize = 1276 * 48 + 48 * 2 + 2;

/// The pad/unpad round trips `test_repacketizer_api` runs on every output packet of length
/// `len` (in `po`, which holds `MAX_OUT + 256` bytes), plus the `opus_repacketizer_out`
/// buffer-size checks.
fn pad_roundtrips(rp: &Repacketizer<'_>, po: &mut [u8], len: usize) {
    assert_eq!(rp.out(&mut po[..len]), Ok(len));
    assert_eq!(packet_unpad(&mut po[..len]), Ok(len));
    assert_eq!(packet_pad(po, len, len + 1), Ok(()));
    assert_eq!(packet_pad(po, len + 1, len + 256), Ok(()));
    assert_eq!(packet_unpad(&mut po[..len + 256]), Ok(len));
    assert_eq!(multistream_packet_unpad(&mut po[..len], 1), Ok(len));
    assert_eq!(multistream_packet_pad(po, len, len + 1, 1), Ok(()));
    assert_eq!(multistream_packet_pad(po, len + 1, len + 256, 1), Ok(()));
    assert_eq!(multistream_packet_unpad(&mut po[..len + 256], 1), Ok(len));
    assert_eq!(rp.out(&mut po[..len - 1]), Err(Error::BufferTooSmall));
    if len > 1 {
        assert_eq!(rp.out(&mut po[..1]), Err(Error::BufferTooSmall));
    }
    assert_eq!(rp.out(&mut po[..0]), Err(Error::BufferTooSmall));
}

/// Port of `test_repacketizer_api`.
///
/// The C repacketizer keeps pointers into `packet` while the test rewrites it; with Rust
/// borrows every packet a repacketizer holds is an immutable copy (same bytes).
#[test]
fn repacketizer_api() {
    let mut packet = vec![0u8; MAX_OUT];
    let mut po = vec![0u8; MAX_OUT + 256];

    let mut rp = Repacketizer::new();
    assert_eq!(rp.nb_frames(), 0);
    rp.init();
    assert_eq!(rp.nb_frames(), 0);

    // Length overflows
    {
        let p = |b0: u8, b1: u8| {
            let mut v = vec![0u8; 256];
            v[0] = b0;
            v[1] = b1;
            v
        };
        let zero = p(0, 0);
        let odd = p(1, 0);
        let c2 = p(2, 0);
        let c3 = p(3, 0);
        let c2b = p(2, 255);
        let c2c = p(2, 250);
        let c3m0 = p(3, 0);
        let c3m49 = p(3, 49);
        let ok = p(0, 49);
        let toc = p(1 << 2, 49);
        let mut rp = Repacketizer::new();
        // Zero len
        assert_eq!(rp.cat(&zero[..0]), Err(Error::InvalidPacket));
        // Odd payload code 1
        assert_eq!(rp.cat(&odd[..2]), Err(Error::InvalidPacket));
        // Code 2 overflow one
        assert_eq!(rp.cat(&c2[..1]), Err(Error::InvalidPacket));
        // Code 3 no count
        assert_eq!(rp.cat(&c3[..1]), Err(Error::InvalidPacket));
        // Code 2 overflow two
        assert_eq!(rp.cat(&c2b[..2]), Err(Error::InvalidPacket));
        // Code 2 overflow three
        assert_eq!(rp.cat(&c2c[..251]), Err(Error::InvalidPacket));
        // Code 3 m=0
        assert_eq!(rp.cat(&c3m0[..2]), Err(Error::InvalidPacket));
        // Code 3 m=49
        assert_eq!(rp.cat(&c3m49[..100]), Err(Error::InvalidPacket));
        assert_eq!(rp.cat(&ok[..3]), Ok(()));
        // Change in TOC
        assert_eq!(rp.cat(&toc[..3]), Err(Error::InvalidPacket));
    }
    packet[0] = 0;
    packet[1] = 49;

    // Code 0,1,3 CBR -> Code 0,1,3 CBR
    for j in 0..32u8 {
        // TOC types, test half with stereo
        packet[0] = ((j << 1) + (j & 1)) << 2;
        let maxi = 960 / packet::get_samples_per_frame(&packet[..1], 8000).unwrap() as usize;
        for i in 1..=maxi {
            // Number of CBR frames in the input packets
            packet[0] = ((j << 1) + (j & 1)) << 2;
            if i > 1 {
                packet[0] += if i == 2 { 1 } else { 3 };
            }
            packet[1] = if i > 2 { i as u8 } else { 0 };
            let maxp =
                960 / (i * packet::get_samples_per_frame(&packet[..1], 8000).unwrap() as usize);
            let pk = packet.clone();
            let hdr = if i > 2 { 2 } else { 1 };
            let mut k = 0;
            while k <= 1275 + 75 {
                // Payload size; only testing CBR here, payload must be a multiple of the count.
                if k % i != 0 {
                    k += 3;
                    continue;
                }
                let mut rp = Repacketizer::new();
                for cnt in 0..maxp + 2 {
                    if cnt > 0 {
                        let ret = rp.cat(&pk[..k + hdr]);
                        if cnt <= maxp && k <= 1275 * i {
                            assert_eq!(ret, Ok(()));
                        } else {
                            assert_eq!(ret, Err(Error::InvalidPacket));
                        }
                    }
                    let rcnt = if k <= 1275 * i { cnt.min(maxp) } else { 0 };
                    assert_eq!(rp.nb_frames() as usize, rcnt * i);
                    let ret = rp.out_range(0, (rcnt * i) as i32, &mut po[..MAX_OUT]);
                    if rcnt > 0 {
                        let len = k * rcnt + if rcnt * i > 2 { 2 } else { 1 };
                        assert_eq!(ret, Ok(len));
                        if rcnt * i < 2 {
                            assert_eq!(po[0] & 3, 0); // Code 0
                        }
                        if rcnt * i == 2 {
                            assert_eq!(po[0] & 3, 1); // Code 1
                        }
                        if rcnt * i > 2 {
                            // Code 3 CBR
                            assert_eq!(po[0] & 3, 3);
                            assert_eq!(usize::from(po[1]), rcnt * i);
                        }
                        pad_roundtrips(&rp, &mut po, len);
                    } else {
                        // M must not be 0
                        assert_eq!(ret, Err(Error::BadArg));
                    }
                }
                k += 3;
            }
        }
    }

    // Change in input count code, CBR out
    {
        let mut a = packet.clone();
        a[0] = 0;
        let mut b = a.clone();
        b[0] += 1;
        let mut rp = Repacketizer::new();
        assert_eq!(rp.cat(&a[..5]), Ok(()));
        assert_eq!(rp.cat(&b[..9]), Ok(()));
        let i = rp.out(&mut po[..MAX_OUT]).unwrap();
        assert_eq!(i, 4 + 8 + 2);
        assert_eq!(po[0] & 3, 3);
        assert_eq!(po[1] & 63, 3);
        assert_eq!(po[1] >> 7, 0);
        assert_eq!(rp.out_range(0, 1, &mut po[..MAX_OUT]), Ok(5));
        assert_eq!(po[0] & 3, 0);
        assert_eq!(rp.out_range(1, 2, &mut po[..MAX_OUT]), Ok(5));
        assert_eq!(po[0] & 3, 0);
        packet.copy_from_slice(&b);
    }

    // Change in input count code, VBR out
    {
        let mut a = packet.clone();
        a[0] = 1;
        let mut b = a.clone();
        b[0] = 0;
        let mut rp = Repacketizer::new();
        assert_eq!(rp.cat(&a[..9]), Ok(()));
        assert_eq!(rp.cat(&b[..3]), Ok(()));
        let i = rp.out(&mut po[..MAX_OUT]).unwrap();
        assert_eq!(i, 2 + 8 + 2 + 2);
        assert_eq!(po[0] & 3, 3);
        assert_eq!(po[1] & 63, 3);
        assert_eq!(po[1] >> 7, 1);
        packet.copy_from_slice(&b);
    }

    // VBR in, VBR out
    {
        packet[0] = 2;
        packet[1] = 4;
        let a = packet.clone();
        let mut rp = Repacketizer::new();
        assert_eq!(rp.cat(&a[..8]), Ok(()));
        assert_eq!(rp.cat(&a[..8]), Ok(()));
        let i = rp.out(&mut po[..MAX_OUT]).unwrap();
        assert_eq!(i, 2 + 1 + 1 + 1 + 4 + 2 + 4 + 2);
        assert_eq!(po[0] & 3, 3);
        assert_eq!(po[1] & 63, 4);
        assert_eq!(po[1] >> 7, 1);
    }

    // VBR in, CBR out
    {
        packet[0] = 2;
        packet[1] = 4;
        let a = packet.clone();
        let mut rp = Repacketizer::new();
        assert_eq!(rp.cat(&a[..10]), Ok(()));
        assert_eq!(rp.cat(&a[..10]), Ok(()));
        let i = rp.out(&mut po[..MAX_OUT]).unwrap();
        assert_eq!(i, 2 + 4 + 4 + 4 + 4);
        assert_eq!(po[0] & 3, 3);
        assert_eq!(po[1] & 63, 4);
        assert_eq!(po[1] >> 7, 0);
    }

    // Count 0 in, VBR out
    for j in 0..32u8 {
        // TOC types, test half with stereo
        packet[0] = ((j << 1) + (j & 1)) << 2;
        let maxi = 960 / packet::get_samples_per_frame(&packet[..1], 8000).unwrap() as usize;
        let mut sum = 0;
        let mut rcnt = 0;
        let pk = packet.clone();
        let mut rp = Repacketizer::new();
        for i in 1..=maxi + 2 {
            let ret = rp.cat(&pk[..i]);
            if rcnt < maxi {
                assert_eq!(ret, Ok(()));
                rcnt += 1;
                sum += i - 1;
            } else {
                assert_eq!(ret, Err(Error::InvalidPacket));
            }
            let len = sum
                + if rcnt < 2 {
                    1
                } else if rcnt < 3 {
                    2
                } else {
                    2 + rcnt - 1
                };
            assert_eq!(rp.out(&mut po[..MAX_OUT]), Ok(len));
            if rcnt > 2 {
                assert_eq!(usize::from(po[1] & 63), rcnt);
            }
            if rcnt == 2 {
                assert_eq!(po[0] & 3, 2);
            }
            if rcnt == 1 {
                assert_eq!(po[0] & 3, 0);
            }
            pad_roundtrips(&rp, &mut po, len);
        }
    }

    po[0] = b'O';
    po[1] = b'p';
    assert_eq!(packet_pad(&mut po, 4, 4), Ok(()));
    assert_eq!(multistream_packet_pad(&mut po, 4, 4, 1), Ok(()));
    assert_eq!(packet_pad(&mut po, 4, 5), Err(Error::InvalidPacket));
    assert_eq!(
        multistream_packet_pad(&mut po, 4, 5, 1),
        Err(Error::InvalidPacket)
    );
    assert_eq!(packet_pad(&mut po, 0, 5), Err(Error::BadArg));
    assert_eq!(multistream_packet_pad(&mut po, 0, 5, 1), Err(Error::BadArg));
    assert_eq!(packet_unpad(&mut po[..0]), Err(Error::BadArg));
    assert_eq!(
        multistream_packet_unpad(&mut po[..0], 1),
        Err(Error::BadArg)
    );
    assert_eq!(packet_unpad(&mut po[..4]), Err(Error::InvalidPacket));
    assert_eq!(
        multistream_packet_unpad(&mut po[..4], 1),
        Err(Error::InvalidPacket)
    );
    po[0] = 0;
    po[1] = 0;
    po[2] = 0;
    assert_eq!(packet_pad(&mut po, 5, 4), Err(Error::BadArg));
    assert_eq!(multistream_packet_pad(&mut po, 5, 4, 1), Err(Error::BadArg));
    // Rust-only: a buffer shorter than the padded length is rejected, not overrun.
    assert_eq!(packet_pad(&mut po[..4], 3, 5), Err(Error::BadArg));
}

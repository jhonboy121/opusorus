//! Port of libopus `tests/test_opus_decode.c`: decoder behaviour on every TOC, PLC, FEC and
//! fuzzed packets across all sampling rates and channel counts, with the libopus final-range
//! checksums, plus `opus_pcm_soft_clip`.
//!
//! The random draws use the libopus `fast_rand` with a fixed seed (override with `SEED`, as the
//! C program accepts). Setting `TEST_OPUS_NOFUZZ` skips the fuzzing part, as in C.
//!
//! Adaptations (C API misuse that cannot be expressed in Rust):
//! * Negative packet lengths (`-1`, `INT_MIN`) and negative frame sizes: lengths are slice
//!   lengths, frame sizes `usize`. A NULL packet with a nonzero length is `None` (the random
//!   lengths are still drawn so the random sequence matches C).
//! * "Crazy FEC values" (`decode_fec` of -1 or 2): `decode_fec` is a `bool`.
//! * `opus_pcm_soft_clip` with a negative frame size or channel count: both are `usize`.
//! * The C test copies decoders with `memcpy`; the Rust decoder is `Clone`.
//! * `t=rand()&3` uses the C library `rand()`, reproduced with glibc's generator.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::{FastRand, GlibcRand, debruijn2};
use opusorus::packet::{self, pcm_soft_clip};
use opusorus::{Decoder, Error};

const MAX_PACKET: usize = 1500;
const MAX_FRAME_SAMP: usize = 5760;
const FSV: [i32; 5] = [48000, 24000, 16000, 12000, 8000];
/// Guard value written around the output buffer.
const GUARD: i16 = 32749;

/// `OPUS_SET_COMPLEXITY(fast_rand()%11)`, only compiled in with the DNN features (as in C).
#[cfg(any(feature = "osce", feature = "deep-plc"))]
fn random_complexity(dec: &mut Decoder, rng: &mut FastRand) {
    dec.set_complexity((rng.next() % 11) as i32).unwrap();
}

#[cfg(not(any(feature = "osce", feature = "deep-plc")))]
const fn random_complexity(_dec: &mut Decoder, _rng: &mut FastRand) {}

/// `true` unless the decode returned a positive sample count (C `if(out_samples>0)test_failed()`).
const fn not_positive(r: Result<usize, Error>) -> bool {
    !matches!(r, Ok(n) if n > 0)
}

/// Port of `test_decoder_code0`.
#[allow(
    clippy::too_many_lines,
    reason = "straight port of one C test function; splitting it would obscure the correspondence"
)]
fn test_decoder_code0(no_fuzz: bool, rng: &mut FastRand) {
    let mut packet = vec![0u8; MAX_PACKET];
    let mut outbuf_int = vec![GUARD; (MAX_FRAME_SAMP + 16) * 2];
    let out_range = 8 * 2..(8 + MAX_FRAME_SAMP) * 2;

    let mut dec: Vec<Decoder> = Vec::with_capacity(10);
    for t in 0..5 * 2 {
        let fs = FSV[t >> 1];
        let c = (t & 1) as i32 + 1;
        let d = Decoder::new(fs, c).unwrap();
        // The opus state structures contain no pointers and can be freely copied.
        let copy = d.clone();
        drop(d);
        dec.push(copy);
    }

    for t in 0..5 * 2 {
        let factor = (48000 / FSV[t >> 1]) as usize;
        let outbuf = &mut outbuf_int[out_range.clone()];
        for fec in [false, true] {
            random_complexity(&mut dec[t], rng);
            // Test PLC on a fresh decoder
            assert_eq!(
                dec[t].decode(None, outbuf, 120 / factor, fec),
                Ok(120 / factor)
            );
            assert_eq!(dec[t].last_packet_duration(), 120 / factor);

            // Test on a size which isn't a multiple of 2.5ms
            assert_eq!(
                dec[t].decode(None, outbuf, 120 / factor + 2, fec),
                Err(Error::BadArg)
            );

            // Test null pointer input (C lengths -1, 1, 10, fast_rand(): all a lost packet).
            for _len in [-1i64, 1, 10, i64::from(rng.next())] {
                assert_eq!(
                    dec[t].decode(None, outbuf, 120 / factor, fec),
                    Ok(120 / factor)
                );
            }
            assert_eq!(dec[t].last_packet_duration(), 120 / factor);

            // Zero lengths
            assert_eq!(
                dec[t].decode(Some(&packet[..0]), outbuf, 120 / factor, fec),
                Ok(120 / factor)
            );

            // Zero buffer
            outbuf[0] = GUARD;
            assert!(not_positive(dec[t].decode(
                Some(&packet[..0]),
                outbuf,
                0,
                fec
            )));
            assert!(not_positive(dec[t].decode(
                Some(&packet[..0]),
                &mut [],
                0,
                fec
            )));
            assert_eq!(outbuf[0], GUARD);

            // Skipped: invalid (negative) lengths and crazy FEC values (see the module docs).

            // Reset the decoder
            dec[t].reset();
        }
    }

    // Count code 0 tests
    for i in 0..64u8 {
        let mut expected = [0usize; 5 * 2];
        packet[0] = i << 2;
        packet[1] = 255;
        packet[2] = 255;
        assert_eq!(
            packet::get_nb_channels(&packet[..3]),
            Ok(i32::from(i & 1) + 1)
        );

        for t in 0..5 * 2 {
            expected[t] = dec[t].nb_samples(&packet[..1]).unwrap();
            assert!(expected[t] <= 2880);
        }

        let outbuf = &mut outbuf_int[out_range.clone()];
        for j in 0..=255u8 {
            packet[1] = j;
            let mut dec_final_range2 = 0;
            for t in 0..5 * 2 {
                random_complexity(&mut dec[t], rng);
                let out_samples = dec[t].decode(Some(&packet[..3]), outbuf, MAX_FRAME_SAMP, false);
                assert_eq!(out_samples, Ok(expected[t]));
                assert_eq!(dec[t].last_packet_duration(), expected[t]);
                let dec_final_range1 = dec[t].final_range();
                if t == 0 {
                    dec_final_range2 = dec_final_range1;
                } else {
                    assert_eq!(dec_final_range1, dec_final_range2);
                }
            }
        }

        for t in 0..5 * 2 {
            let factor = (48000 / FSV[t >> 1]) as usize;
            // The PLC is run for 6 frames in order to get better PLC coverage.
            for _ in 0..6 {
                random_complexity(&mut dec[t], rng);
                assert_eq!(
                    dec[t].decode(None, outbuf, expected[t], false),
                    Ok(expected[t])
                );
                assert_eq!(dec[t].last_packet_duration(), expected[t]);
            }
            // Run the PLC once at 2.5ms, as a simulation of someone trying to do small drift
            // corrections.
            if expected[t] != 120 / factor {
                assert_eq!(
                    dec[t].decode(None, outbuf, 120 / factor, false),
                    Ok(120 / factor)
                );
                assert_eq!(dec[t].last_packet_duration(), 120 / factor);
            }
            assert!(not_positive(dec[t].decode(
                Some(&packet[..2]),
                outbuf,
                expected[t] - 1,
                false
            )));
        }
    }

    if no_fuzz {
        // Skipping many tests which fuzz the decoder as requested.
        check_guards(&outbuf_int);
        return;
    }

    {
        // We only test a subset of the modes here simply because the longer durations end up
        // taking a long time.
        const CMODES: [u8; 4] = [16, 20, 24, 28];
        const CRES: [u32; 4] = [116_290_185, 2_172_123_586, 2_172_123_586, 2_172_123_586];
        const LRES: [u32; 3] = [3_285_687_739, 1_481_572_662, 694_350_475];
        const LMODES: [u8; 3] = [0, 4, 8];
        let outbuf = &mut outbuf_int[out_range.clone()];

        let mode = (rng.next() % 4) as usize;
        packet[0] = CMODES[mode] << 3;
        let mut dec_final_acc = 0u32;
        let t = (rng.next() % 10) as usize;
        for i in 0..65536u32 {
            let factor = (48000 / FSV[t >> 1]) as usize;
            packet[1] = (i >> 8) as u8;
            packet[2] = (i & 255) as u8;
            packet[3] = 255;
            let out_samples = dec[t].decode(Some(&packet[..4]), outbuf, MAX_FRAME_SAMP, false);
            assert_eq!(out_samples, Ok(120 / factor));
            dec_final_acc = dec_final_acc.wrapping_add(dec[t].final_range());
        }
        assert_eq!(
            dec_final_acc, CRES[mode],
            "dec[{t}] all 3-byte prefix for length 4, mode {}",
            CMODES[mode]
        );

        let mode = (rng.next() % 3) as usize;
        packet[0] = LMODES[mode] << 3;
        let mut dec_final_acc = 0u32;
        let t = (rng.next() % 10) as usize;
        for i in 0..65536u32 {
            let factor = (48000 / FSV[t >> 1]) as usize;
            packet[1] = (i >> 8) as u8;
            packet[2] = (i & 255) as u8;
            packet[3] = 255;
            let out_samples = dec[t].decode(Some(&packet[..4]), outbuf, MAX_FRAME_SAMP, false);
            assert_eq!(out_samples, Ok(480 / factor));
            dec_final_acc = dec_final_acc.wrapping_add(dec[t].final_range());
        }
        assert_eq!(
            dec_final_acc, LRES[mode],
            "dec[{t}] all 3-byte prefix for length 4, mode {}",
            LMODES[mode]
        );
    }

    let skip = (rng.next() % 7) as usize;
    for i in 0..64u8 {
        let mut expected = [0usize; 5 * 2];
        packet[0] = i << 2;
        for t in 0..5 * 2 {
            expected[t] = dec[t].nb_samples(&packet[..1]).unwrap();
        }
        let outbuf = &mut outbuf_int[out_range.clone()];
        for j in (2 + skip..1275).step_by(4) {
            for jj in 0..j {
                packet[jj + 1] = (rng.next() & 255) as u8;
            }
            let mut dec_final_range2 = 0;
            for t in 0..5 * 2 {
                random_complexity(&mut dec[t], rng);
                let out_samples =
                    dec[t].decode(Some(&packet[..j + 1]), outbuf, MAX_FRAME_SAMP, false);
                assert_eq!(out_samples, Ok(expected[t]));
                let dec_final_range1 = dec[t].final_range();
                if t == 0 {
                    dec_final_range2 = dec_final_range1;
                } else {
                    assert_eq!(dec_final_range1, dec_final_range2);
                }
            }
        }
    }
    // random packets, all modes (64), every 8th size from 2+skip bytes to maximum OK.

    let modes = debruijn2(64);
    let plen = ((rng.next() % 18 + 3) * 8) as usize + skip + 3;
    for &mode in &modes[..4096] {
        let mut expected = [0usize; 5 * 2];
        packet[0] = mode << 2;
        for t in 0..5 * 2 {
            expected[t] = dec[t].nb_samples(&packet[..plen]).unwrap();
        }
        for j in 0..plen {
            packet[j + 1] = ((rng.next() | rng.next()) & 255) as u8;
        }
        let outbuf = &mut outbuf_int[out_range.clone()];
        let mut decbak = dec[0].clone();
        assert_eq!(
            decbak.decode(Some(&packet[..plen + 1]), outbuf, expected[0], true),
            Ok(expected[0])
        );
        let mut decbak = dec[0].clone();
        assert!(decbak.decode(None, outbuf, MAX_FRAME_SAMP, true).unwrap() >= 20);
        let mut decbak = dec[0].clone();
        assert!(decbak.decode(None, outbuf, MAX_FRAME_SAMP, false).unwrap() >= 20);
        for t in 0..5 * 2 {
            random_complexity(&mut dec[t], rng);
            let out_samples =
                dec[t].decode(Some(&packet[..plen + 1]), outbuf, MAX_FRAME_SAMP, false);
            assert_eq!(out_samples, Ok(expected[t]));
            // (C compares a stale `dec_final_range1` here, which trivially passes.)
            assert_eq!(dec[t].last_packet_duration(), expected[t]);
        }
    }
    // random packets, all mode pairs (4096), plen+1 bytes/frame OK.

    let plen = ((rng.next() % 18 + 3) * 8) as usize + skip + 3;
    // `t=rand()&3` with the (unseeded) C library rand().
    let t = (GlibcRand::new(1).next() & 3) as usize;
    for &mode in &modes[..4096] {
        packet[0] = mode << 2;
        let expected = dec[t].nb_samples(&packet[..plen]).unwrap();
        let outbuf = &mut outbuf_int[out_range.clone()];
        for _count in 0..10 {
            random_complexity(&mut dec[t], rng);
            for j in 0..plen {
                packet[j + 1] = ((rng.next() | rng.next()) & 255) as u8;
            }
            let out_samples =
                dec[t].decode(Some(&packet[..plen + 1]), outbuf, MAX_FRAME_SAMP, false);
            assert_eq!(out_samples, Ok(expected));
        }
    }
    // random packets, all mode pairs (4096)*10, plen+1 bytes/frame OK.

    {
        const TMODES: [u8; 1] = [25 << 2];
        const TSEEDS: [u32; 1] = [140_441];
        const TLEN: [usize; 1] = [157];
        const TRET: [usize; 1] = [480];
        let t = (rng.next() & 1) as usize;
        let outbuf = &mut outbuf_int[out_range.clone()];
        for i in 0..1 {
            packet[0] = TMODES[i];
            *rng = FastRand::new(TSEEDS[i]);
            for b in &mut packet[1..TLEN[i]] {
                *b = (rng.next() & 255) as u8;
            }
            let out_samples =
                dec[t].decode(Some(&packet[..TLEN[i]]), outbuf, MAX_FRAME_SAMP, false);
            assert_eq!(out_samples, Ok(TRET[i]));
        }
        // pre-selected random packets OK.
    }

    check_guards(&outbuf_int);
}

/// The guard samples around the output buffer must be untouched.
fn check_guards(outbuf_int: &[i16]) {
    assert!(outbuf_int[..8 * 2].iter().all(|&v| v == GUARD));
    assert!(
        outbuf_int[(8 + MAX_FRAME_SAMP) * 2..(16 + MAX_FRAME_SAMP) * 2]
            .iter()
            .all(|&v| v == GUARD)
    );
}

/// `main` of `test_opus_decode.c`.
#[test]
fn decoder_code0() {
    let mut rng = FastRand::new(common::seed());
    // `main` prints `fast_rand() % 65535` with the seed.
    rng.next();
    let no_fuzz = std::env::var_os("TEST_OPUS_NOFUZZ").is_some();
    test_decoder_code0(no_fuzz, &mut rng);
}

/// The ramp of `test_soft_clip`: `(j&255)*(1/32.f)-4.f`.
fn fill_ramp(x: &mut [f32]) {
    for (j, v) in x.iter_mut().enumerate() {
        *v = (j & 255) as f32 * (1.0 / 32.0f32) - 4.0;
    }
}

/// Port of `test_soft_clip`.
#[test]
fn soft_clip() {
    let mut x = [0f32; 1024];
    let mut s = [0f32; 8];
    for i in 0..1024 {
        fill_ramp(&mut x);
        pcm_soft_clip(&mut x[i..], 1024 - i, 1, &mut s);
        assert!(x[i..].iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }
    for i in 1..9 {
        fill_ramp(&mut x);
        pcm_soft_clip(&mut x, 1024 / i, i, &mut s);
        assert!(
            x[..(1024 / i) * i]
                .iter()
                .all(|&v| (-1.0..=1.0).contains(&v))
        );
    }
    // Degenerate calls do nothing (C: zero sizes and NULL pointers; negative sizes cannot be
    // expressed).
    let before = x;
    pcm_soft_clip(&mut x, 0, 1, &mut s);
    pcm_soft_clip(&mut x, 1, 0, &mut s);
    pcm_soft_clip(&mut x, 1, 1, &mut []);
    pcm_soft_clip(&mut [], 1, 1, &mut s);
    assert_eq!(x, before);
}

//! Port of libopus `tests/test_opus_encode.c` (`run_test1` and `fuzz_encoder_settings`; the
//! `regression_test` of `opus_encode_regressions.c` is in `regressions.rs`): encode + decode in
//! every mode, bitrate and frame size with forced modes, VBR/CVBR/CBR, FEC, DTX, packet
//! padding/unpadding, multistream, frame-size switching and packet fuzzing, checking the
//! encoder/decoder final range after every packet; then random encoder settings.
//!
//! The random draws use the libopus `fast_rand` with a fixed seed (override with `SEED`).
//! Setting `TEST_OPUS_NOFUZZ` disables the packet corruption and the settings fuzzing, as in C.
//!
//! Length: libopus encodes `SAMPLES = 48000*30` samples in `run_test1` (plus 4x that in the
//! frame-size switching loop) and fuzzes 5 encoders x 40 settings. The default test runs a
//! 1/5 scale version (`SAMPLES = 48000*6`, 3 x 10 settings) to stay within the test time
//! budget; the full-length versions are the `#[ignore]`d `*_full` tests (or set
//! `OPUS_TEST_FULL`).
//!
//! Adaptations:
//! * `OPUS_UNIMPLEMENTED` as an application cannot reach the constructor
//!   ([`Application::from_raw`] rejects it).
//! * `ENABLE_DRED`: the DRED decoder API (`opus_dred_parse` / `opus_decoder_dred_decode`) is
//!   not part of the Rust API; only `OPUS_SET_DRED_DURATION` is exercised (with `dred`).
//! * C leaves the evaluation order of two `fast_rand()` calls in one expression unspecified;
//!   the port draws them left to right.
//! * The C test copies the encoder with `memcpy`; the Rust encoder is `Clone`.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::{FastRand, debruijn2};
use opusorus::constants::raw::*;
use opusorus::encoder::request::OPUS_SET_FORCE_MODE_REQUEST;
use opusorus::packet::{self, MODE_SILK_ONLY};
use opusorus::repacketizer::{
    multistream_packet_pad, multistream_packet_unpad, packet_pad, packet_unpad,
};
use opusorus::{
    Application, Bandwidth, Bitrate, Decoder, Encoder, Error, FrameSize, MsDecoder, MsEncoder,
    Signal,
};

const MAX_PACKET: usize = 1500;
const MAX_FRAME_SAMP: usize = 5760;

/// Test length (`SAMPLES` of the C test) and settings-fuzzing size.
#[derive(Debug, Clone, Copy)]
struct Scale {
    samples: usize,
    fuzz_encoders: usize,
    fuzz_settings: usize,
}

const FULL: Scale = Scale {
    samples: 48000 * 30,
    fuzz_encoders: 5,
    fuzz_settings: 40,
};
const REDUCED: Scale = Scale {
    samples: 48000 * 6,
    fuzz_encoders: 3,
    fuzz_settings: 10,
};

fn scale() -> Scale {
    if common::full_length() { FULL } else { REDUCED }
}

/// Port of `generate_music`: `len` stereo frames of pseudo-music (60 ms of silence first).
fn generate_music(buf: &mut [i16], len: usize, rng: &mut FastRand) {
    let (mut a1, mut b1, mut a2, mut b2) = (0i32, 0i32, 0i32, 0i32);
    let (mut c1, mut c2, mut d1, mut d2) = (0i32, 0i32, 0i32, 0i32);
    let mut j: i32 = 0;
    // 60ms silence
    buf[..2880 * 2].fill(0);
    for i in 2880..len {
        let mut v1 =
            ((((j * ((j >> 12) ^ ((j >> 10 | j >> 12) & 26 & j >> 7))) & 128) + 128) << 15) as u32;
        let mut v2 = v1;
        // C: `v1+=r&65535; v1-=r>>16;` in unsigned arithmetic, stored back to int.
        let r = rng.next();
        v1 = v1.wrapping_add(r & 65535).wrapping_sub(r >> 16);
        let r = rng.next();
        v2 = v2.wrapping_add(r & 65535).wrapping_sub(r >> 16);
        let (v1, v2) = (v1 as i32, v2 as i32);
        b1 = v1 - a1 + ((b1 * 61 + 32) >> 6);
        a1 = v1;
        b2 = v2 - a2 + ((b2 * 61 + 32) >> 6);
        a2 = v2;
        c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
        d1 = b1;
        c2 = (30 * (c2 + b2 + d2) + 32) >> 6;
        d2 = b2;
        let v1 = (c1 + 128) >> 8;
        let v2 = (c2 + 128) >> 8;
        buf[i * 2] = v1.clamp(-32768, 32767) as i16;
        buf[i * 2 + 1] = v2.clamp(-32768, 32767) as i16;
        if i % 6 == 0 {
            j += 1;
        }
    }
}

/// Port of `get_frame_size_enum`.
fn get_frame_size_enum(frame_size: i32, sampling_rate: i32) -> FrameSize {
    let table = [
        (sampling_rate / 400, FrameSize::Ms2_5),
        (sampling_rate / 200, FrameSize::Ms5),
        (sampling_rate / 100, FrameSize::Ms10),
        (sampling_rate / 50, FrameSize::Ms20),
        (sampling_rate / 25, FrameSize::Ms40),
        (3 * sampling_rate / 50, FrameSize::Ms60),
        (4 * sampling_rate / 50, FrameSize::Ms80),
        (5 * sampling_rate / 50, FrameSize::Ms100),
        (6 * sampling_rate / 50, FrameSize::Ms120),
    ];
    match table.iter().find(|(n, _)| *n == frame_size) {
        Some(&(_, e)) => e,
        None => panic!("frame size {frame_size} at {sampling_rate} Hz"),
    }
}

/// Port of `test_encode` (the settings-fuzzing encode/decode loop).
fn test_encode(
    enc: &mut Encoder,
    channels: usize,
    frame_size: usize,
    dec: &mut Decoder,
    ssamples: usize,
    rng: &mut FastRand,
) {
    let mut packet = [0u8; MAX_PACKET + 257];
    // Generate input data
    let mut inbuf = vec![0i16; ssamples];
    generate_music(&mut inbuf, ssamples / 2, rng);
    // Allocate memory for output data
    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * 3];

    // Encode data, then decode for sanity check
    let mut samp_count = 0usize;
    loop {
        let len = enc
            .encode(
                &inbuf[samp_count * channels..],
                frame_size,
                &mut packet[..MAX_PACKET],
            )
            .unwrap();
        assert!(len <= MAX_PACKET);
        // ENABLE_DRED: opus_dred_parse / opus_decoder_dred_decode are not part of the Rust API.
        let out_samples = dec
            .decode(Some(&packet[..len]), &mut outbuf, MAX_FRAME_SAMP, false)
            .unwrap();
        assert_eq!(out_samples, frame_size);
        samp_count += frame_size;
        if samp_count as isize >= (ssamples / 2) as isize - MAX_FRAME_SAMP as isize {
            break;
        }
    }
}

/// Port of `fuzz_encoder_settings`.
fn fuzz_encoder_settings(
    num_encoders: usize,
    num_setting_changes: usize,
    ssamples: usize,
    rng: &mut FastRand,
) {
    // Parameters to fuzz. Some values are duplicated to increase their probability of being
    // tested.
    let sampling_rates = [8000, 12000, 16000, 24000, 48000];
    let channels = [1, 2];
    let applications = [
        Application::Audio,
        Application::Voip,
        Application::RestrictedLowDelay,
    ];
    let bitrates = [
        6000,
        12000,
        16000,
        24000,
        32000,
        48000,
        64000,
        96000,
        510_000,
        OPUS_AUTO,
        OPUS_BITRATE_MAX,
    ];
    let force_channels = [OPUS_AUTO, OPUS_AUTO, 1, 2];
    let use_vbr = [0, 1, 1];
    let vbr_constraints = [0, 1, 1];
    let complexities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max_bandwidths = [
        Bandwidth::Narrowband,
        Bandwidth::Mediumband,
        Bandwidth::Wideband,
        Bandwidth::Superwideband,
        Bandwidth::Fullband,
        Bandwidth::Fullband,
    ];
    let signals = [Signal::Auto, Signal::Auto, Signal::Voice, Signal::Music];
    let inband_fecs = [0, 0, 1];
    let packet_loss_perc = [0, 1, 2, 5];
    let lsb_depths = [8, 24];
    let prediction_disabled = [0, 0, 1];
    let use_dtx = [0, 1];
    let frame_sizes_ms_x2 = [5, 10, 20, 40, 80, 120, 160, 200, 240]; // x2 to avoid 2.5 ms

    for _ in 0..num_encoders {
        let sampling_rate = rng.sample(&sampling_rates);
        let num_channels = rng.sample(&channels);
        let application = rng.sample(&applications);

        let mut dec = Decoder::new(sampling_rate, num_channels).unwrap();
        let mut enc = Encoder::new(sampling_rate, num_channels, application).unwrap();

        for _ in 0..num_setting_changes {
            let bitrate = rng.sample(&bitrates);
            let force_channel = rng.sample(&force_channels);
            let vbr = rng.sample(&use_vbr);
            let vbr_constraint = rng.sample(&vbr_constraints);
            let complexity = rng.sample(&complexities);
            let max_bw = rng.sample(&max_bandwidths);
            let sig = rng.sample(&signals);
            let inband_fec = rng.sample(&inband_fecs);
            let pkt_loss = rng.sample(&packet_loss_perc);
            let lsb_depth = rng.sample(&lsb_depths);
            let pred_disabled = rng.sample(&prediction_disabled);
            let dtx = rng.sample(&use_dtx);
            let frame_size_ms_x2 = rng.sample(&frame_sizes_ms_x2);
            let frame_size = frame_size_ms_x2 * sampling_rate / 2000;
            let frame_size_enum = get_frame_size_enum(frame_size, sampling_rate);
            let force_channel = force_channel.min(num_channels);

            enc.set_bitrate(Bitrate::from_raw(bitrate)).unwrap();
            enc.set_force_channels(if force_channel == OPUS_AUTO {
                None
            } else {
                Some(force_channel as u8)
            })
            .unwrap();
            enc.set_vbr(vbr != 0);
            enc.set_vbr_constraint(vbr_constraint != 0);
            enc.set_complexity(complexity).unwrap();
            enc.set_max_bandwidth(max_bw);
            enc.set_signal(sig);
            enc.set_inband_fec(inband_fec).unwrap();
            enc.set_packet_loss_perc(pkt_loss).unwrap();
            enc.set_lsb_depth(lsb_depth).unwrap();
            enc.set_prediction_disabled(pred_disabled != 0);
            enc.set_dtx(dtx != 0);
            enc.set_expert_frame_duration(frame_size_enum);
            #[cfg(feature = "dred")]
            enc.set_dred_duration((rng.next() % 101) as i32).unwrap();
            #[cfg(feature = "qext")]
            enc.set_qext(!rng.next().is_multiple_of(2));
            test_encode(
                &mut enc,
                num_channels as usize,
                frame_size as usize,
                &mut dec,
                ssamples,
                rng,
            );
        }
    }
}

/// `opus_decode` arguments of a possibly lost packet (`len>0?packet:NULL`).
fn maybe(packet: &[u8], len: usize) -> Option<&[u8]> {
    if len > 0 { Some(&packet[..len]) } else { None }
}

/// Port of `run_test1`.
#[allow(
    clippy::too_many_lines,
    reason = "straight port of one C test function; splitting it would obscure the correspondence"
)]
fn run_test1(no_fuzz: bool, samples: usize, rng: &mut FastRand) {
    let ssamples = samples / 3;
    let fsizes = [960 * 3, 960 * 2, 120, 240, 480, 960];
    let mut mapping = [0u8; 256];
    mapping[..3].copy_from_slice(&[0, 1, 255]);
    let mut packet = [0u8; MAX_PACKET + 257];

    // Encode+Decode tests.
    let enc = Encoder::new(48000, 2, Application::Voip).unwrap();

    // C runs these twice (with and without an error pointer); it is the same call in Rust.
    assert_eq!(Application::from_raw(-5), Err(Error::BadArg)); // OPUS_UNIMPLEMENTED
    for (fs, ch, s, cs) in [
        (8000, 0, 1, 0),
        (44100, 2, 2, 0),
        (8000, 2, 2, 3),
        (8000, 2, -1, 0),
        (8000, 256, 2, 0),
    ] {
        assert_eq!(
            MsEncoder::new(fs, ch, s, cs, &mapping, Application::Voip).err(),
            Some(Error::BadArg),
            "fs={fs} channels={ch} streams={s} coupled={cs}"
        );
    }

    let mut ms_enc = MsEncoder::new(8000, 2, 2, 0, &mapping, Application::Audio).unwrap();

    // Some multistream encoder API tests
    let _bitrate: i32 = ms_enc.bitrate();
    let i = ms_enc.lsb_depth();
    assert!(i >= 16);
    {
        let tmp_enc = ms_enc.encoder_state(1).unwrap();
        assert_eq!(tmp_enc.lsb_depth(), i);
        assert_eq!(ms_enc.encoder_state(2).err(), Some(Error::BadArg));
    }

    let mut dec = Decoder::new(48000, 2).unwrap();
    let mut ms_dec = MsDecoder::new(48000, 2, 2, 0, &mapping).unwrap();
    let mut ms_dec_err = MsDecoder::new(48000, 3, 2, 0, &mapping).unwrap();

    let mut dec_err: Vec<Decoder> = vec![dec.clone()];
    for (fs, c) in [
        (48000, 1),
        (24000, 2),
        (24000, 1),
        (16000, 2),
        (16000, 1),
        (12000, 2),
        (12000, 1),
        (8000, 2),
        (8000, 1),
    ] {
        dec_err.push(Decoder::new(fs, c).unwrap());
    }

    // The opus state structures contain no pointers and can be freely copied.
    let mut enc = {
        let copy = enc.clone();
        drop(enc);
        copy
    };

    let mut inbuf = vec![0i16; samples * 2];
    let mut outbuf = vec![0i16; samples * 2];
    let mut out2buf = vec![0i16; MAX_FRAME_SAMP * 3];
    generate_music(&mut inbuf, samples, rng);

    enc.set_bandwidth(None);
    assert_eq!(
        enc.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, -2),
        Err(Error::BadArg)
    );
    assert_eq!(
        enc.encode(&inbuf, 500, &mut packet[..MAX_PACKET]),
        Err(Error::BadArg)
    );

    for rc in 0..3 {
        enc.set_vbr(rc < 2);
        enc.set_vbr_constraint(rc == 1);
        enc.set_vbr_constraint(rc == 1);
        enc.set_inband_fec(i32::from(rc == 0)).unwrap();
        #[cfg(feature = "dred")]
        enc.set_dred_duration((rng.next() % 101) as i32).unwrap();
        for j in 0..13 {
            const MODES: [i32; 13] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
            const RATES: [u32; 13] = [
                6000, 12000, 48000, 16000, 32000, 48000, 64000, 512_000, 13000, 24000, 48000,
                64000, 96000,
            ];
            const FRAME: [usize; 13] = [
                960 * 2,
                960,
                480,
                960,
                960,
                960,
                480,
                960 * 3,
                960 * 3,
                960,
                480,
                240,
                120,
            ];
            let rate = (RATES[j] + rng.next() % RATES[j]) as i32;
            let mut count = 0;
            let mut i = 0usize;
            loop {
                let frame_size = FRAME[j];
                if rng.next() & 255 == 0 {
                    enc.reset();
                    dec.reset();
                    if rng.next() & 1 != 0 {
                        dec_err[(rng.next() & 1) as usize].reset();
                    }
                }
                if rng.next() & 127 == 0 {
                    dec_err[(rng.next() & 1) as usize].reset();
                }
                if rng.next().is_multiple_of(10) {
                    let complex = (rng.next() % 11) as i32;
                    enc.set_complexity(complex).unwrap();
                }
                if rng.next().is_multiple_of(50) {
                    dec.reset();
                }
                enc.set_inband_fec(i32::from(rc == 0)).unwrap();
                enc.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK_ONLY + MODES[j])
                    .unwrap();
                enc.set_dtx(rng.next() & 1 != 0);
                enc.set_bitrate(Bitrate::Bits(rate)).unwrap();
                enc.set_force_channels(Some(if RATES[j] >= 64000 { 2 } else { 1 }))
                    .unwrap();
                enc.set_complexity((count >> 2) % 11).unwrap();
                let a = rng.next() & 15;
                let b = rng.next() % 15;
                enc.set_packet_loss_perc((a & b) as i32).unwrap();
                let mut bw = match MODES[j] {
                    0 => OPUS_BANDWIDTH_NARROWBAND + (rng.next() % 3) as i32,
                    1 => OPUS_BANDWIDTH_SUPERWIDEBAND + (rng.next() & 1) as i32,
                    _ => OPUS_BANDWIDTH_NARROWBAND + (rng.next() % 5) as i32,
                };
                if MODES[j] == 2 && bw == OPUS_BANDWIDTH_MEDIUMBAND {
                    bw += 3;
                }
                enc.set_bandwidth(Some(Bandwidth::from_raw(bw).unwrap()));
                #[cfg(feature = "qext")]
                let qext = {
                    let q = !rng.next().is_multiple_of(2);
                    enc.set_qext(q);
                    q
                };
                #[cfg(not(feature = "qext"))]
                let qext = false;
                let mut len = enc
                    .encode(&inbuf[i << 1..], frame_size, &mut packet[..MAX_PACKET])
                    .unwrap();
                assert!(len <= MAX_PACKET);
                let enc_final_range = enc.final_range();
                if rng.next() & 3 == 0 {
                    packet_pad(&mut packet, len, len + 1).unwrap();
                    len += 1;
                }
                if rng.next() & 7 == 0 {
                    packet_pad(&mut packet, len, len + 256).unwrap();
                    len += 256;
                }
                let mut unpad = false;
                if rng.next() & 3 == 0 {
                    unpad = true;
                    len = packet_unpad(&mut packet[..len]).unwrap();
                    assert!(len >= 1);
                }
                let out_samples = dec
                    .decode(
                        Some(&packet[..len]),
                        &mut outbuf[i << 1..],
                        MAX_FRAME_SAMP,
                        false,
                    )
                    .unwrap();
                assert_eq!(out_samples, frame_size);
                let dec_final_range = dec.final_range();
                assert!(
                    enc_final_range == dec_final_range || (unpad && qext),
                    "final range mismatch: mode {} rate {rate} rc {rc}",
                    MODES[j]
                );
                // LBRR decode
                let fec = rng.next() & 3 != 0;
                let out_samples = dec_err[0]
                    .decode(Some(&packet[..len]), &mut out2buf, frame_size, fec)
                    .unwrap();
                assert_eq!(out_samples, frame_size);
                let lost = rng.next() & 3 == 0;
                let fec = rng.next() & 7 != 0;
                let out_samples = dec_err[1]
                    .decode(
                        maybe(&packet, if lost { 0 } else { len }),
                        &mut out2buf,
                        MAX_FRAME_SAMP,
                        fec,
                    )
                    .unwrap();
                assert!(out_samples >= 120);
                i += frame_size;
                count += 1;
                if i as isize >= ssamples as isize - MAX_FRAME_SAMP as isize {
                    break;
                }
            }
        }
    }

    enc.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, OPUS_AUTO).unwrap();
    enc.set_force_channels(None).unwrap();
    enc.set_inband_fec(0).unwrap();
    enc.set_dtx(false);

    for rc in 0..3 {
        ms_enc.set_vbr(rc < 2);
        ms_enc.set_vbr_constraint(rc == 1);
        ms_enc.set_vbr_constraint(rc == 1);
        ms_enc.set_inband_fec(i32::from(rc == 0)).unwrap();
        for j in 0..16 {
            const MODES: [i32; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2];
            const RATES: [u32; 16] = [
                4000, 12000, 32000, 8000, 16000, 32000, 48000, 88000, 4000, 12000, 32000, 8000,
                16000, 32000, 48000, 288_000,
            ];
            const FRAME: [usize; 16] = [
                160, 160, 80, 160, 160, 80, 40, 20, 160, 160, 80, 160, 160, 80, 40, 20,
            ];
            ms_enc.set_inband_fec(i32::from(rc == 0 && j == 1)).unwrap();
            ms_enc
                .ctl_set(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK_ONLY + MODES[j])
                .unwrap();
            let rate = (RATES[j] + rng.next() % RATES[j]) as i32;
            ms_enc.set_dtx(rng.next() & 1 != 0);
            ms_enc.set_bitrate(Bitrate::Bits(rate)).unwrap();
            let mut count = 0;
            let mut i = 0usize;
            loop {
                let pred = ms_enc.prediction_disabled();
                ms_enc.set_prediction_disabled((rng.next() & 15) < if pred { 11 } else { 4 });
                let frame_size = FRAME[j];
                ms_enc.set_complexity((count >> 2) % 11).unwrap();
                let a = rng.next() & 15;
                let b = rng.next() % 15;
                ms_enc.set_packet_loss_perc((a & b) as i32).unwrap();
                if rng.next() & 255 == 0 {
                    ms_enc.reset();
                    ms_dec.reset();
                    if rng.next() & 3 != 0 {
                        ms_dec_err.reset();
                    }
                }
                if rng.next() & 255 == 0 {
                    ms_dec_err.reset();
                }
                #[cfg(feature = "qext")]
                let qext = {
                    let q = !rng.next().is_multiple_of(2);
                    ms_enc.set_qext(q);
                    q
                };
                #[cfg(not(feature = "qext"))]
                let qext = false;
                let mut len = ms_enc
                    .encode(&inbuf[i << 1..], frame_size, &mut packet[..MAX_PACKET])
                    .unwrap();
                assert!(len <= MAX_PACKET);
                let enc_final_range = ms_enc.final_range();
                if rng.next() & 3 == 0 {
                    multistream_packet_pad(&mut packet, len, len + 1, 2).unwrap();
                    len += 1;
                }
                if rng.next() & 7 == 0 {
                    multistream_packet_pad(&mut packet, len, len + 256, 2).unwrap();
                    len += 256;
                }
                let mut unpad = false;
                if rng.next() & 3 == 0 {
                    len = multistream_packet_unpad(&mut packet[..len], 2).unwrap();
                    assert!(len >= 1);
                    unpad = true;
                }
                let out_samples = ms_dec
                    .decode(Some(&packet[..len]), &mut out2buf, MAX_FRAME_SAMP, false)
                    .unwrap();
                assert_eq!(out_samples, frame_size * 6);
                let dec_final_range = ms_dec.final_range();
                assert!(
                    enc_final_range == dec_final_range || (unpad && qext),
                    "MS final range mismatch: mode {} rate {rate} rc {rc}",
                    MODES[j]
                );
                // LBRR decode
                let loss = rng.next() & 63 == 0;
                let fec = rng.next() & 3 != 0;
                let out_samples = ms_dec_err
                    .decode(
                        maybe(&packet, if loss { 0 } else { len }),
                        &mut out2buf,
                        frame_size * 6,
                        fec,
                    )
                    .unwrap();
                assert_eq!(out_samples, frame_size * 6);
                i += frame_size;
                count += 1;
                if i as isize >= (ssamples / 12) as isize - MAX_FRAME_SAMP as isize {
                    break;
                }
            }
        }
    }

    let mut bitrate_bps: u32 = 512_000;
    let mut fsize = (rng.next() % 31) as usize;
    let mut fswitch: u32 = 100;

    let db62 = debruijn2(6);
    let mut count = 0;
    let mut i = 0usize;
    loop {
        let frame_size = fsizes[usize::from(db62[fsize])];
        let offset = i % (samples - MAX_FRAME_SAMP);

        enc.set_bitrate(Bitrate::Bits(bitrate_bps as i32)).unwrap();

        let mut len = enc
            .encode(&inbuf[offset << 1..], frame_size, &mut packet[..MAX_PACKET])
            .unwrap();
        assert!(len <= MAX_PACKET);
        count += 1;

        let enc_final_range = enc.final_range();

        let out_samples = dec
            .decode(
                Some(&packet[..len]),
                &mut outbuf[offset << 1..],
                MAX_FRAME_SAMP,
                false,
            )
            .unwrap();
        assert_eq!(out_samples, frame_size);

        // compare final range encoder rng values of encoder and decoder
        assert_eq!(dec.final_range(), enc_final_range);

        // We fuzz the packet, but take care not to only corrupt the payload. Corrupted headers
        // are tested elsewhere and we need to actually run the decoders in order to compare
        // them.
        let payload_offset = packet::parse(&packet[..len]).unwrap().payload_offset;
        if rng.next() & 1023 == 0 {
            len = 0;
        }
        // (No-op when the packet was dropped: `len == 0`.)
        for b in packet.iter_mut().take(len).skip(payload_offset) {
            for jj in 0..8 {
                let flip = !no_fuzz && (rng.next() & 1023) == 0;
                *b ^= u8::from(flip) << jj;
            }
        }
        let out_samples = dec_err[0]
            .decode(maybe(&packet, len), &mut out2buf, MAX_FRAME_SAMP, false)
            .unwrap();
        assert!(out_samples <= MAX_FRAME_SAMP);
        if len > 0 {
            assert_eq!(out_samples, frame_size); // FIXME use lastframe
        }

        let dec_final_range = dec_err[0].final_range();

        // randomly select one of the decoders to compare with
        let dec2 = (rng.next() % 9 + 1) as usize;
        let out_samples = dec_err[dec2]
            .decode(maybe(&packet, len), &mut out2buf, MAX_FRAME_SAMP, false)
            .unwrap();
        assert!(out_samples <= MAX_FRAME_SAMP); // FIXME, use factor, lastframe for loss

        let dec_final_range2 = dec_err[dec2].final_range();
        if len > 0 {
            assert_eq!(dec_final_range, dec_final_range2);
        }

        fswitch -= 1;
        if fswitch < 1 {
            fsize = (fsize + 1) % 36;
            let new_size = fsizes[usize::from(db62[fsize])] as u32;
            if new_size == 960 || new_size == 480 {
                fswitch = 2880 / new_size * (rng.next() % 19 + 1);
            } else {
                fswitch = (rng.next() % (2880 / new_size)) + 1;
            }
        }
        bitrate_bps = ((rng.next() % 508_000 + 4000) + bitrate_bps) >> 1;
        i += frame_size;
        if i >= samples * 4 {
            break;
        }
    }
    assert!(count > 0);
    // All framesize pairs switching encode, `count` frames OK.

    enc.reset();
    drop(enc);
    ms_enc.reset();
    drop(ms_enc);
    dec.reset();
    drop(dec);
    ms_dec.reset();
}

fn run(scale: Scale) {
    let mut rng = FastRand::new(common::seed());
    // `main` prints `fast_rand() % 65535` with the seed.
    rng.next();
    // (`regression_test()` runs here in C; see regressions.rs.)
    let no_fuzz = std::env::var_os("TEST_OPUS_NOFUZZ").is_some();
    run_test1(no_fuzz, scale.samples, &mut rng);
    // Fuzz encoder settings online
    if !no_fuzz {
        fuzz_encoder_settings(
            scale.fuzz_encoders,
            scale.fuzz_settings,
            scale.samples / 3,
            &mut rng,
        );
    }
}

#[test]
fn encode() {
    run(scale());
}

#[test]
#[ignore = "full libopus length (minutes); run with --ignored"]
fn encode_full() {
    run(FULL);
}

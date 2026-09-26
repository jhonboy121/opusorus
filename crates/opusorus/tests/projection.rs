//! Port of libopus `tests/test_opus_projection.c`: projection (ambisonics) encoder/decoder
//! creation for every channel count, and an encode/decode round trip.
//!
//! `test_simple_matrix` exercises the internal mapping-matrix routines and lives in
//! `crates/opusorus-conformance/tests/libopus_unit.rs` (it needs the `internals` feature).
//!
//! Adaptation: the decode output buffer holds `MAX_FRAME_SAMPLES` (the `frame_size` argument)
//! rather than `BUFFER_SIZE` samples per channel, see `test_encode_decode`.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod common;

use common::FastRand;
use opusorus::{Application, Bitrate, ProjectionDecoder, ProjectionEncoder};

const BUFFER_SIZE: usize = 960;
const MAX_DATA_BYTES: usize = 32768;
const MAX_FRAME_SAMPLES: usize = 5760;

/// `OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE` + `OPUS_PROJECTION_GET_DEMIXING_MATRIX`.
fn demixing_matrix(enc: &ProjectionEncoder) -> Vec<u8> {
    let matrix_size = enc.demixing_matrix_size();
    assert_ne!(matrix_size, 0);
    let mut matrix = vec![0u8; matrix_size];
    enc.write_demixing_matrix(&mut matrix).unwrap();
    assert_eq!(matrix, enc.demixing_matrix());
    matrix
}

/// Port of `test_creation_arguments`.
fn test_creation_arguments(channels: i32, mapping_family: i32) {
    let fs = 48000;
    let application = Application::Audio;

    let order_plus_one = f64::from(channels as f32).sqrt().floor() as i32;
    let nondiegetic_channels = channels - order_plus_one * order_plus_one;

    let st_enc = ProjectionEncoder::new_ambisonics(fs, channels, mapping_family, application);
    let enc_ok = st_enc.is_ok();
    let mut dec_ok = false;
    if let Ok(st_enc) = st_enc {
        let matrix = demixing_matrix(&st_enc);
        let (streams, coupled_streams) = (st_enc.streams(), st_enc.coupled_streams());
        drop(st_enc);
        let st_dec = ProjectionDecoder::new(fs, channels, streams, coupled_streams, &matrix);
        dec_ok = st_dec.is_ok();
    }

    let is_channels_valid = (2..=6).contains(&order_plus_one)
        && (nondiegetic_channels == 0 || nondiegetic_channels == 2);
    let is_projection_valid = enc_ok && dec_ok;
    assert_eq!(
        is_channels_valid, is_projection_valid,
        "Channels: {channels}, Family: {mapping_family}, Order+1: {order_plus_one}, \
         Non-diegetic Channels: {nondiegetic_channels}"
    );
}

/// Port of the projection test's `generate_music` (per-channel filters).
fn generate_music(buf: &mut [i16], len: usize, channels: usize, rng: &mut FastRand) {
    let mut a = vec![0i32; channels];
    let mut b = vec![0i32; channels];
    let mut c = vec![0i32; channels];
    let mut d = vec![0i32; channels];
    let mut j: i32 = 0;

    for i in 0..len {
        for k in 0..channels {
            let v = ((((j * ((j >> 12) ^ ((j >> 10 | j >> 12) & 26 & j >> 7))) & 128) + 128) << 15)
                as u32;
            // C: `v+=r&65535; v-=r>>16;` in unsigned arithmetic, stored back to int.
            let r = rng.next();
            let v = v.wrapping_add(r & 65535).wrapping_sub(r >> 16) as i32;
            b[k] = v - a[k] + ((b[k] * 61 + 32) >> 6);
            a[k] = v;
            c[k] = (30 * (c[k] + b[k] + d[k]) + 32) >> 6;
            d[k] = b[k];
            let v = (c[k] + 128) >> 8;
            buf[i * channels + k] = v.clamp(-32768, 32767) as i16;
            if i % 6 == 0 {
                j += 1;
            }
        }
    }
}

/// Port of `test_encode_decode`.
fn test_encode_decode(bitrate: i32, channels: i32, mapping_family: i32) {
    let fs = 48000;
    let application = Application::Audio;
    let ch = channels as usize;
    let mut buffer_in = vec![0i16; BUFFER_SIZE * ch];
    // C allocates `BUFFER_SIZE * channels` and decodes with `MAX_FRAME_SAMPLES`; the Rust
    // multistream/projection decoders require room for `frame_size * channels` samples even
    // when the packet is shorter (a Rust-only bounds guard), so the buffer holds 120 ms.
    let mut buffer_out = vec![0i16; MAX_FRAME_SAMPLES * ch];
    let mut data = vec![0u8; MAX_DATA_BYTES];

    let mut st_enc =
        match ProjectionEncoder::new_ambisonics(fs, channels, mapping_family, application) {
            Ok(e) => e,
            Err(e) => panic!(
                "Couldn't create encoder with {channels} channels and mapping family \
                 {mapping_family}: {e}"
            ),
        };
    let (streams, coupled) = (st_enc.streams(), st_enc.coupled_streams());

    st_enc
        .ms_mut()
        .set_bitrate(Bitrate::Bits(bitrate * 1000 * (streams + coupled)))
        .unwrap();

    let matrix = demixing_matrix(&st_enc);
    let mut st_dec = ProjectionDecoder::new(fs, channels, streams, coupled, &matrix).unwrap();

    // The C test never seeds `fast_rand` (static `Rz = Rw = 0`).
    let mut rng = FastRand::new(0);
    generate_music(&mut buffer_in, BUFFER_SIZE, ch, &mut rng);

    let len = st_enc.encode(&buffer_in, BUFFER_SIZE, &mut data).unwrap();
    assert!(len <= MAX_DATA_BYTES);

    let out_samples = st_dec
        .decode(
            Some(&data[..len]),
            &mut buffer_out,
            MAX_FRAME_SAMPLES,
            false,
        )
        .unwrap();
    assert_eq!(out_samples, BUFFER_SIZE);
}

/// `main` of `test_opus_projection.c` (without `test_simple_matrix`).
#[test]
fn creation_arguments() {
    // Test full range of channels in creation arguments.
    for i in 0..255 {
        test_creation_arguments(i, 3);
    }
}

#[test]
fn encode_decode() {
    // Test encode/decode pipeline.
    test_encode_decode(64 * 18, 18, 3);
}

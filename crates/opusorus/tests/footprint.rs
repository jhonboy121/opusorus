//! Checks the memory footprint reported by [`Decoder::get_size`] and [`MsDecoder::get_size`]
//! against the memory decoders actually allocate: many decoders are created and fed CELT,
//! hybrid and SILK packets and losses through the int16 API (so every lazily allocated buffer
//! grows, including the DNN states when models are loaded), and the growth of the process'
//! data segment (`VmData`, Linux) per decoder is compared with `get_size`.
//!
//! This is the only test of its binary: other tests allocating concurrently would distort
//! the measurement.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

use opusorus::encoder::request as req;
use opusorus::{Application, Decoder, Encoder, MsDecoder};

/// Decoders per measurement (fewer with the DNN models loaded: their states are large and
/// slow to run in debug builds).
fn count() -> usize {
    if dnn_loaded() { 200 } else { 1000 }
}

/// Decoding rate: the highest supported one, which `get_size` assumes.
const FS: i32 = if cfg!(feature = "qext") { 96000 } else { 48000 };
/// Samples per channel of a 20 ms frame at [`FS`].
const FRAME: usize = FS as usize / 50;

/// The process' data segment size (`VmData`) in bytes.
fn vm_data() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status
        .lines()
        .find(|l| l.starts_with("VmData:"))
        .expect("VmData in /proc/self/status");
    let kb: usize = line
        .trim_start_matches("VmData:")
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .unwrap();
    kb * 1024
}

/// A 48 kHz stereo test signal (two voice-like tone mixes with noise).
fn signal(frames: usize) -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    (0..frames * 960)
        .flat_map(|i| {
            let t = i as f32 / 48000.0;
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let noise = ((seed >> 16) as f32 / 32768.0 - 1.0) * 0.05;
            let env = 0.5 + 0.5 * (2.0 * core::f32::consts::PI * 3.0 * t).sin();
            let l = env * 0.3 * (2.0 * core::f32::consts::PI * 220.0 * t).sin() + noise;
            let r = env * 0.3 * (2.0 * core::f32::consts::PI * 330.0 * t).sin() - noise;
            [l, r]
        })
        .collect()
}

/// 20 ms packets of the Rust encoder: CELT (stereo, 128 kb/s), hybrid (SWB, 40 kb/s), SILK
/// (WB, 24 kb/s, 16 kHz internal as needed by OSCE and the BWE) and, with QEXT, Opus HD.
fn packets() -> Vec<Vec<u8>> {
    let pcm = signal(5);
    let mut out = Vec::new();
    for (app, bitrate, bandwidth) in [
        (Application::Audio, 128_000, OPUS_BANDWIDTH_FULLBAND),
        (Application::Voip, 40_000, OPUS_BANDWIDTH_SUPERWIDEBAND),
        (Application::Voip, 24_000, OPUS_BANDWIDTH_WIDEBAND),
    ] {
        let mut enc = Encoder::new(48000, 2, app).unwrap();
        enc.ctl_set(req::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
        enc.ctl_set(req::OPUS_SET_BANDWIDTH_REQUEST, bandwidth)
            .unwrap();
        for f in pcm.chunks(2 * 960) {
            let mut buf = [0u8; 1500];
            let n = enc.encode_float(f, 960, &mut buf).unwrap();
            out.push(buf[..n].to_vec());
        }
    }
    // Opus HD: 96 kHz CELT with the QEXT extension (its bands grow the CELT band scratch).
    #[cfg(feature = "qext")]
    {
        let mut enc = Encoder::new(96000, 2, Application::Audio).unwrap();
        enc.ctl_set(req::OPUS_SET_BITRATE_REQUEST, 256_000).unwrap();
        enc.ctl_set(req::OPUS_SET_QEXT_REQUEST, 1).unwrap();
        // The 48 kHz signal at 96 kHz (sample repetition).
        let pcm96: Vec<f32> = pcm
            .chunks(2)
            .flat_map(|lr| [lr[0], lr[1], lr[0], lr[1]])
            .collect();
        for f in pcm96.chunks(2 * 1920) {
            let mut buf = [0u8; 1500];
            let n = enc.encode_float(f, 1920, &mut buf).unwrap();
            out.push(buf[..n].to_vec());
        }
    }
    out
}

const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

/// Feeds `packets` (every fifth one lost) to a decoder through `decode`.
fn run(packets: &[Vec<u8>], mut decode: impl FnMut(Option<&[u8]>)) {
    for (i, p) in packets.iter().enumerate() {
        decode(if i % 5 == 3 { None } else { Some(p) });
    }
}

/// Enables every DNN path of a decoder with loaded models (deep PLC, NoLACE, BWE).
#[allow(unused_variables, reason = "only used with the DNN features")]
fn enable_dnn(dec: &mut Decoder) {
    #[cfg(any(feature = "deep-plc", feature = "osce"))]
    {
        dec.set_complexity(10).unwrap();
        #[cfg(feature = "osce")]
        dec.set_osce_bwe(1).unwrap();
    }
}

/// Whether the decoders have DNN models (so the DNN states `get_size` counts are allocated).
fn dnn_loaded() -> bool {
    #[cfg(any(feature = "deep-plc", feature = "osce"))]
    {
        let (plc, osce) = Decoder::new(48000, 1).unwrap().dnn_loaded();
        plc || osce
    }
    #[cfg(not(any(feature = "deep-plc", feature = "osce")))]
    false
}

/// Bytes allocated per object by [`count`] calls of `make` (the objects are leaked), plus the
/// inline size of `T` (its slots are allocated before the measurement).
fn measure<T>(mut make: impl FnMut() -> T) -> usize {
    // Warm-up: shared model caches, allocator state.
    drop(make());
    let count = count();
    let mut objs: Vec<T> = Vec::with_capacity(count);
    let before = vm_data();
    for _ in 0..count {
        objs.push(make());
    }
    let after = vm_data();
    // Kept alive: freed memory would be reused by the next measurement.
    core::mem::forget(objs);
    (after - before) / count + size_of::<T>()
}

#[test]
fn decoder_footprint_matches_get_size() {
    let packets = packets();
    // Without every DNN state allocated `get_size` is only an upper bound.
    let exact = !cfg!(any(feature = "deep-plc", feature = "osce")) || dnn_loaded();
    // `get_size` covers the 96 kHz buffers of QEXT and the OSCE BWE state together, but the
    // BWE only runs at 48 kHz: at 96 kHz its two states (~35 KB) are not allocated.
    let slack = |m: usize| {
        if cfg!(all(feature = "qext", feature = "osce")) {
            m / 5
        } else {
            m / 20
        }
    };
    let check = |what: &str, measured: usize, reported: usize| {
        println!("{what}: measured {measured} B, get_size {reported} B");
        assert!(
            measured <= reported + reported / 20 + 1024,
            "{what}: get_size {reported} < measured {measured}"
        );
        if exact {
            assert!(
                reported <= measured + slack(measured) + 1024,
                "{what}: get_size {reported} > measured {measured}"
            );
        }
    };
    for ch in [1usize, 2] {
        let measured = measure(|| {
            let mut dec = Decoder::new(FS, ch as i32).unwrap();
            enable_dnn(&mut dec);
            let mut pcm = vec![0i16; FRAME * ch];
            run(&packets, |p| {
                dec.decode(p, &mut pcm, FRAME, false).unwrap();
            });
            dec
        });
        check(
            &format!("Decoder ({ch} ch)"),
            measured,
            Decoder::get_size(ch as i32),
        );
    }
    for (streams, coupled) in [(1, 0), (1, 1)] {
        let ch = streams + coupled;
        let measured = measure(|| {
            let mut dec = MsDecoder::new(FS, ch, streams, coupled, &[0, 1]).unwrap();
            enable_dnn(dec.decoder_state(0).unwrap());
            let mut pcm = vec![0i16; FRAME * ch as usize];
            run(&packets, |p| {
                dec.decode(p, &mut pcm, FRAME, false).unwrap();
            });
            dec
        });
        check(
            &format!("MsDecoder ({streams} streams, {coupled} coupled)"),
            measured,
            MsDecoder::get_size(streams, coupled),
        );
    }
}

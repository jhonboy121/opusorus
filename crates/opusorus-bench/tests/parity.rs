//! Sanity checks for the benchmark workloads: all three implementations do the same work.
//!
//! * Rust and the scalar C oracle are bit-exact (packets, decoded PCM, kernel outputs).
//! * The optimized C build decodes the same packets to nearly the same audio, produces valid
//!   packets of the expected mode, and gives identical integer results (range coder,
//!   resampler) and near-identical float kernel results.
//! * Every configuration really runs in the intended coding mode.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]

use opus_sys_optimized::{Micro, Resampler, Variant};
use opusorus::celt::kiss_fft::opus_fft;
use opusorus::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
use opusorus::celt::static_modes::KissFftCpx;
use opusorus::silk::resampler::{SilkResamplerState, silk_resampler};
use opusorus_bench::micro::{EC_BUF, MicroBench, ec_ops, ec_roundtrip_rust, mode48};
use opusorus_bench::{
    AnyDecoder, AnyMsDecoder, CodecBench, ENCODE_COMPLEXITIES, FRAMES, Impl, Mode, SURROUND,
    codec_configs, decode_packets, encode_all, report_rows, runner_for_group, surround_encode_all,
};
use opusorus_conformance::{Rng, assert_bits_eq_f32, signals};

/// Signal-to-difference ratio in dB of `b` against reference `a`.
fn snr_db(a: &[f32], b: &[f32]) -> f64 {
    let (mut s, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        s += f64::from(*x) * f64::from(*x);
        n += f64::from(x - y) * f64::from(x - y);
    }
    if n == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / n).log10()
    }
}

#[test]
fn encoders_match_and_use_intended_mode() {
    for cfg in codec_configs() {
        for cx in ENCODE_COMPLEXITIES {
            let rust = encode_all(Impl::Rust, &cfg, cx).unwrap();
            let scalar = encode_all(Impl::CScalar, &cfg, cx).unwrap();
            let opt = encode_all(Impl::COpt, &cfg, cx).unwrap();
            assert_eq!(rust.len(), FRAMES);
            assert_eq!(rust, scalar, "{} cx{cx}: rust vs c_scalar packets", cfg.id);
            for (imp, packets) in [("rust", &rust), ("c_opt", &opt)] {
                for (i, p) in packets.iter().enumerate() {
                    assert!(
                        p.len() > 1,
                        "{} cx{cx} {imp}: frame {i} is DTX/empty",
                        cfg.id
                    );
                    assert_eq!(
                        Mode::of_toc(p[0]),
                        cfg.expect_mode,
                        "{} cx{cx} {imp}: frame {i} mode",
                        cfg.id
                    );
                }
            }
            // Rate sanity: VBR average between 50 % (speech pauses) and 130 % of the target.
            let bytes: usize = rust.iter().map(Vec::len).sum();
            let kbps = bytes as f64 * 8.0 / (FRAMES as f64 * 0.02) / 1000.0;
            let target = f64::from(cfg.bitrate) / 1000.0;
            assert!(
                kbps > 0.5 * target && kbps < 1.3 * target,
                "{} cx{cx}: {kbps:.1} kbit/s vs target {target}",
                cfg.id
            );
        }
    }
}

#[test]
fn decoders_match() {
    for cfg in codec_configs() {
        let packets = decode_packets(&cfg).unwrap();
        let fsz = cfg.frame_size();
        let n = fsz * cfg.channels as usize;
        let mut outs = Vec::new();
        for imp in Impl::ALL {
            let mut dec = AnyDecoder::new(imp, cfg.fs, cfg.channels).unwrap();
            let mut pcm = vec![0f32; n];
            let mut all = Vec::with_capacity(n * packets.len());
            for p in &packets {
                assert_eq!(dec.decode(p, &mut pcm, fsz).unwrap(), fsz);
                all.extend_from_slice(&pcm);
            }
            outs.push(all);
        }
        assert_bits_eq_f32(cfg.id, &outs[0], &outs[1]);
        let snr = snr_db(&outs[1], &outs[2]);
        assert!(snr > 60.0, "{}: c_opt vs c_scalar SNR {snr:.1} dB", cfg.id);
    }
}

#[test]
fn surround_matches() {
    for cx in ENCODE_COMPLEXITIES {
        let (layout_r, rust) = surround_encode_all(Impl::Rust, &SURROUND, cx).unwrap();
        let (layout_s, scalar) = surround_encode_all(Impl::CScalar, &SURROUND, cx).unwrap();
        let (layout_o, opt) = surround_encode_all(Impl::COpt, &SURROUND, cx).unwrap();
        assert_eq!(layout_r, layout_s);
        assert_eq!(layout_r, layout_o);
        assert_eq!(
            (layout_r.0, layout_r.1),
            (4, 2),
            "5.1 = 4 streams, 2 coupled"
        );
        assert_eq!(rust, scalar, "cx{cx}: surround packets");
        assert_eq!(opt.len(), FRAMES);
    }
    let (layout, packets) = surround_encode_all(Impl::Rust, &SURROUND, 10).unwrap();
    let fsz = SURROUND.frame_size();
    let mut outs = Vec::new();
    for imp in Impl::ALL {
        let mut dec = AnyMsDecoder::new(imp, &SURROUND, &layout).unwrap();
        let mut pcm = vec![0f32; fsz * 6];
        let mut all = Vec::new();
        for p in &packets {
            assert_eq!(dec.decode(p, &mut pcm, fsz).unwrap(), fsz);
            all.extend_from_slice(&pcm);
        }
        outs.push(all);
    }
    assert_bits_eq_f32("surround decode", &outs[0], &outs[1]);
    let snr = snr_db(&outs[1], &outs[2]);
    assert!(snr > 60.0, "surround c_opt SNR {snr:.1} dB");
}

#[test]
fn mdct_and_fft_match() {
    let mode = mode48().unwrap();
    let n2 = mode.mdct.n as usize / 2;
    let ov = mode.overlap as usize;
    let scalar = Micro::new(Variant::Scalar);
    let opt = Micro::new(Variant::Optimized);
    assert_eq!(scalar.arch(), 0, "the oracle has no RTCD");
    assert_eq!((scalar.mdct_n(), scalar.overlap()), (2 * n2, ov));
    assert_eq!((opt.mdct_n(), opt.overlap()), (2 * n2, ov));

    let x = signals::music_like(n2 + ov, 1, 48000, 7);
    for shift in 0..=3 {
        let m2 = n2 >> shift;
        let mut r = vec![0f32; m2];
        let mut s = vec![0f32; m2];
        let mut o = vec![0f32; m2];
        clt_mdct_forward(
            &mode.mdct,
            &x[..m2 + ov],
            &mut r,
            &mode.window,
            ov,
            shift,
            1,
        );
        scalar.mdct_forward(&x[..m2 + ov], &mut s, shift);
        opt.mdct_forward(&x[..m2 + ov], &mut o, shift);
        assert_bits_eq_f32(&format!("mdct fwd shift {shift}"), &r, &s);
        assert!(snr_db(&s, &o) > 100.0, "mdct fwd c_opt");

        let coefs: Vec<f32> = r.iter().map(|v| v * 0.5).collect();
        let mut rb = vec![0.1f32; m2 + ov];
        let mut sb = rb.clone();
        let mut ob = rb.clone();
        clt_mdct_backward(&mode.mdct, &coefs, &mut rb, &mode.window, ov, shift, 1);
        scalar.mdct_backward(&coefs, &mut sb, shift);
        opt.mdct_backward(&coefs, &mut ob, shift);
        assert_bits_eq_f32(&format!("mdct bwd shift {shift}"), &rb, &sb);
        assert!(snr_db(&sb, &ob) > 100.0, "mdct bwd c_opt");
    }

    for idx in 0..4 {
        let st = &*mode.mdct.kfft[idx];
        let nfft = st.nfft as usize;
        assert_eq!(nfft, 480 >> idx);
        assert_eq!(scalar.fft_nfft(idx), nfft);
        let mut rng = Rng::new(idx as u64 + 1);
        let fin: Vec<f32> = (0..2 * nfft).map(|_| rng.f32_sym()).collect();
        let cin: Vec<KissFftCpx> = fin
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| KissFftCpx { r: c[0], i: c[1] })
            .collect();
        let mut cout = vec![KissFftCpx::default(); nfft];
        opus_fft(st, &cin, &mut cout);
        let r: Vec<f32> = cout.iter().flat_map(|c| [c.r, c.i]).collect();
        let mut s = vec![0f32; 2 * nfft];
        let mut o = vec![0f32; 2 * nfft];
        scalar.fft(idx, &fin, &mut s);
        opt.fft(idx, &fin, &mut o);
        assert_bits_eq_f32(&format!("fft {nfft}"), &r, &s);
        assert!(snr_db(&s, &o) > 100.0, "fft c_opt");
    }
}

#[test]
fn range_coder_matches() {
    let ops = ec_ops();
    assert_eq!(ops.len(), 1000);
    let mut buf = vec![0u8; EC_BUF];
    let r = ec_roundtrip_rust(&ops, &mut buf);
    let rust_bytes = buf.clone();
    let mut sbuf = vec![0u8; EC_BUF];
    let s = Micro::new(Variant::Scalar).ec_roundtrip(&ops, &mut sbuf);
    let mut obuf = vec![0u8; EC_BUF];
    let o = Micro::new(Variant::Optimized).ec_roundtrip(&ops, &mut obuf);
    assert_eq!(r, s);
    assert_eq!(r, o);
    assert_eq!(rust_bytes, sbuf);
    assert_eq!(rust_bytes, obuf);
    // The op list must fit (no coder error) and round-trip: re-derive the decoded values.
    let used = rust_bytes.iter().rposition(|&b| b != 0).unwrap();
    assert!(used > 1000 && used < EC_BUF, "{used} bytes");
}

#[test]
fn resampler_matches() {
    let x = signals::to_i16(&signals::speech_like(960 * 10, 1, 48000, 3));
    let mut rs = SilkResamplerState::new(48000, 16000, true).unwrap();
    let mut cs = Resampler::new(Variant::Scalar, 48000, 16000, true).unwrap();
    let mut co = Resampler::new(Variant::Optimized, 48000, 16000, true).unwrap();
    for block in x.as_chunks::<960>().0 {
        let mut r = vec![0i16; 320];
        let mut s = vec![0i16; 320];
        let mut o = vec![0i16; 320];
        silk_resampler(&mut rs, &mut r, block, 960);
        cs.run(&mut s, block);
        co.run(&mut o, block);
        assert_eq!(r, s);
        assert_eq!(r, o);
    }
    // (Unsupported rate pairs are not probed: ENABLE_HARDENING turns them into an abort.)
}

#[test]
fn every_runner_runs() {
    let rows = report_rows();
    let expected = CodecBench::all().len() + MicroBench::ALL.len();
    assert_eq!(rows.len(), expected);
    for row in &rows {
        for imp in Impl::ALL {
            let mut run = runner_for_group(&row.group, imp).unwrap();
            // Wrap around the prepared inputs at least once.
            for _ in 0..FRAMES + 3 {
                run().unwrap();
            }
        }
    }
    assert!(runner_for_group("nope", Impl::Rust).is_err());
}

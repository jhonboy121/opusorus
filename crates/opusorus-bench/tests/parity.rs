//! Sanity checks for the benchmark workloads: all three implementations do the same work.
//!
//! * Rust and the scalar C oracle are bit-exact (packets, decoded PCM, kernel outputs), in the
//!   float build and in the fixed-point builds (`fixed-point`, `fixed-res24`: native 16/24-bit
//!   PCM, `i32` kernels).
//! * The optimized C build decodes the same packets to nearly the same audio, produces valid
//!   packets of the expected mode, and gives identical integer results (range coder,
//!   resampler) and near-identical float kernel results (fixed-point kernels: identical).
//! * Every configuration really runs in the intended coding mode.

use opus_sys_optimized::{KissFftScalar, Micro, Resampler, Variant};
use opusorus::celt::kiss_fft::opus_fft;
use opusorus::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
use opusorus::celt::static_modes::KissFftCpx;
use opusorus::silk::resampler::{SilkResamplerState, silk_resampler};
use opusorus_bench::micro::{EC_BUF, MicroBench, ec_ops, ec_roundtrip_rust, mode48, to_kfft};
use opusorus_bench::{
    AnyDecoder, AnyMsDecoder, BUILD, CodecBench, ENCODE_COMPLEXITIES, FRAMES, Impl, Mode, SURROUND,
    Sample, codec_configs, decode_packets, encode_all, report_rows, runner_for_group,
    surround_encode_all,
};
use opusorus_conformance::{Rng, signals};

/// Signal-to-difference ratio in dB of `b` against reference `a`.
fn snr_db<T: Copy + Into<f64>>(a: &[T], b: &[T]) -> f64 {
    let (mut s, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y): (f64, f64) = (x.into(), y.into());
        s += x * x;
        n += (x - y) * (x - y);
    }
    if n == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / n).log10()
    }
}

/// Bit-level identity of PCM / kernel data (floats compared by bit pattern).
trait Bits: Copy + core::fmt::Debug {
    fn bits(self) -> u64;
}
impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl Bits for i16 {
    fn bits(self) -> u64 {
        u64::from(self as u16)
    }
}
impl Bits for i32 {
    fn bits(self) -> u64 {
        u64::from(self as u32)
    }
}

/// Asserts `a` and `b` are bit-identical, reporting the first mismatch.
#[track_caller]
fn assert_bits_eq<T: Bits>(what: &str, a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x.bits() != y.bits()) {
        panic!("{what}: first mismatch at {i}: {:?} vs {:?}", a[i], b[i]);
    }
}

/// Minimum SNR of the optimized C kernels against the scalar C ones: FMA contraction and NEON
/// reorder float sums; the fixed-point kernels are plain integer C and must be identical.
const KERNEL_SNR_DB: f64 = if cfg!(feature = "fixed-point") {
    f64::INFINITY
} else {
    100.0
};

#[test]
fn encoders_match_and_use_intended_mode() {
    for cfg in codec_configs() {
        for cx in ENCODE_COMPLEXITIES {
            let rust = encode_all(Impl::Rust, &cfg, cx).unwrap();
            let scalar = encode_all(Impl::CScalar, &cfg, cx).unwrap();
            let opt = encode_all(Impl::COpt, &cfg, cx).unwrap();
            assert_eq!(rust.len(), FRAMES);
            assert_eq!(rust, scalar, "{} cx{cx}: rust vs c_scalar packets", cfg.id);
            eprintln!(
                "{BUILD} {} cx{cx}: c_opt packets {} c_scalar",
                cfg.id,
                if opt == scalar { "==" } else { "!=" }
            );
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
            let mut pcm = vec![Sample::default(); n];
            let mut all = Vec::with_capacity(n * packets.len());
            for p in &packets {
                assert_eq!(dec.decode(p, &mut pcm, fsz).unwrap(), fsz);
                all.extend_from_slice(&pcm);
            }
            outs.push(all);
        }
        assert_bits_eq(cfg.id, &outs[0], &outs[1]);
        let snr = snr_db(&outs[1], &outs[2]);
        eprintln!(
            "{BUILD} {}: c_opt vs c_scalar decode SNR {snr:.1} dB",
            cfg.id
        );
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
        eprintln!(
            "{BUILD} surround cx{cx}: c_opt packets {} c_scalar",
            if opt == scalar { "==" } else { "!=" }
        );
    }
    let (layout, packets) = surround_encode_all(Impl::Rust, &SURROUND, 10).unwrap();
    let fsz = SURROUND.frame_size();
    let mut outs = Vec::new();
    for imp in Impl::ALL {
        let mut dec = AnyMsDecoder::new(imp, &SURROUND, &layout).unwrap();
        let mut pcm = vec![Sample::default(); fsz * 6];
        let mut all = Vec::new();
        for p in &packets {
            assert_eq!(dec.decode(p, &mut pcm, fsz).unwrap(), fsz);
            all.extend_from_slice(&pcm);
        }
        outs.push(all);
    }
    assert_bits_eq("surround decode", &outs[0], &outs[1]);
    let snr = snr_db(&outs[1], &outs[2]);
    eprintln!("{BUILD} surround: c_opt vs c_scalar decode SNR {snr:.1} dB");
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

    let x: Vec<KissFftScalar> = signals::music_like(n2 + ov, 1, 48000, 7)
        .into_iter()
        .map(to_kfft)
        .collect();
    for shift in 0..=3 {
        let m2 = n2 >> shift;
        let mut r = vec![KissFftScalar::default(); m2];
        let mut s = r.clone();
        let mut o = r.clone();
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
        assert_bits_eq(&format!("mdct fwd shift {shift}"), &r, &s);
        assert!(snr_db(&s, &o) >= KERNEL_SNR_DB, "mdct fwd c_opt");

        #[cfg(not(feature = "fixed-point"))]
        let coefs: Vec<KissFftScalar> = r.iter().map(|v| v * 0.5).collect();
        #[cfg(feature = "fixed-point")]
        let coefs: Vec<KissFftScalar> = r.iter().map(|v| v / 2).collect();
        let fill = to_kfft(0.1);
        let mut rb = vec![fill; m2 + ov];
        let mut sb = rb.clone();
        let mut ob = rb.clone();
        clt_mdct_backward(&mode.mdct, &coefs, &mut rb, &mode.window, ov, shift, 1);
        scalar.mdct_backward(&coefs, &mut sb, shift);
        opt.mdct_backward(&coefs, &mut ob, shift);
        assert_bits_eq(&format!("mdct bwd shift {shift}"), &rb, &sb);
        assert!(snr_db(&sb, &ob) >= KERNEL_SNR_DB, "mdct bwd c_opt");
    }

    for idx in 0..4 {
        let st = &*mode.mdct.kfft[idx];
        let nfft = st.nfft as usize;
        assert_eq!(nfft, 480 >> idx);
        assert_eq!(scalar.fft_nfft(idx), nfft);
        let mut rng = Rng::new(idx as u64 + 1);
        let fin: Vec<KissFftScalar> = (0..2 * nfft).map(|_| to_kfft(rng.f32_sym())).collect();
        let cin: Vec<KissFftCpx> = fin
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| KissFftCpx { r: c[0], i: c[1] })
            .collect();
        let mut cout = vec![KissFftCpx::default(); nfft];
        opus_fft(st, &cin, &mut cout);
        let r: Vec<KissFftScalar> = cout.iter().flat_map(|c| [c.r, c.i]).collect();
        let mut s = vec![KissFftScalar::default(); 2 * nfft];
        let mut o = s.clone();
        scalar.fft(idx, &fin, &mut s);
        opt.fft(idx, &fin, &mut o);
        assert_bits_eq(&format!("fft {nfft}"), &r, &s);
        assert!(snr_db(&s, &o) >= KERNEL_SNR_DB, "fft c_opt");
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

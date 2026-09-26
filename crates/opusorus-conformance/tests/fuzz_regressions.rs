//! Regression tests for port bugs found by the fuzz targets (`fuzz/`), each checked against the
//! C oracle.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    reason = "test code: failures should panic"
)]

/// `differential_custom` (custom-modes + QEXT): creating an Opus Custom encoder for a 96 kHz
/// custom mode with a 1.25 ms short MDCT (96000 / 120) panicked in debug builds. The port precomputes the QEXT mode at init, and `compute_qext_mode`'s assertion (C
/// `celt_assert(0)`: no QEXT band layout for such a mode) fired there, although C only calls
/// `compute_qext_mode` for frames that carry QEXT data. Encoding without QEXT must work and
/// match C.
#[cfg(all(feature = "custom-modes", feature = "qext"))]
#[test]
fn custom_96k_mode_without_qext_layout() {
    use opusorus::celt::celt_encoder::CeltEncoder;
    use opusorus::celt::modes::opus_custom_mode_create_custom;
    use opusorus_oracle::celt_encoder::CeltEnc;

    for (fs, frame) in [(96000, 120)] {
        for channels in 1..=2 {
            let what = format!("custom mode ({fs}, {frame}) x{channels}");
            let mode = opus_custom_mode_create_custom(fs, frame).unwrap();
            // Neither QEXT layout (2.5 ms or 1.875 ms short MDCT) fits the mode.
            let short = mode.short_mdct_size * 48000;
            assert!(short != 120 * fs && short != 90 * fs, "{what}: layout");
            let mut r = CeltEncoder::opus_custom_encoder_create(mode, channels).unwrap();
            let mut c = CeltEnc::new_custom(fs, frame, channels).unwrap();
            let ch = channels as usize;
            let n = frame as usize;
            for k in 0..20 {
                let pcm: Vec<f32> = (0..n * ch)
                    .map(|i| {
                        let t = (k * n + i / ch) as f32 / fs as f32;
                        0.4 * (2.0 * core::f32::consts::PI * 997.0 * t).sin()
                    })
                    .collect();
                let nb = 60 + 7 * k as i32;
                let mut a = vec![0u8; nb as usize];
                let mut b = a.clone();
                #[cfg(not(feature = "disable-float-api"))]
                let (rr, cr) = (
                    r.opus_custom_encode_float(&pcm, frame, &mut a, nb),
                    c.custom_encode_float(&pcm, ch, frame, &mut b, nb),
                );
                // DISABLE_FLOAT_API: the same signal as 24-bit PCM.
                #[cfg(feature = "disable-float-api")]
                let (rr, cr) = {
                    let p24: Vec<i32> = pcm.iter().map(|&v| (v * 8_388_607.0) as i32).collect();
                    (
                        r.opus_custom_encode24(&p24, frame, &mut a, nb),
                        c.custom_encode24(&p24, ch, frame, &mut b, nb),
                    )
                };
                assert_eq!(rr, cr, "{what} frame {k}: return");
                assert!(rr > 0, "{what} frame {k}: {rr}");
                assert_eq!(
                    a[..rr as usize],
                    b[..rr as usize],
                    "{what} frame {k}: bytes"
                );
                let (ret, rng) = c.ctl_get(4031);
                assert_eq!(ret, 0);
                assert_eq!(r.final_range(), rng as u32, "{what} frame {k}: final range");
            }
        }
    }
}

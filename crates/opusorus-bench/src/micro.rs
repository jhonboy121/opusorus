//! Micro-benchmark kernels: CELT MDCT (forward/backward), 480-point FFT, range coder and the
//! SILK 48 kHz -> 16 kHz resampler, for all three implementations.
//!
//! Each kernel call corresponds to the work done once per 20 ms frame per channel in the codec
//! (one 1920-point MDCT, one 480-point FFT, one frame of resampling), except the range coder,
//! whose op list ([`ec_ops`]) is a fixed mix of about one high-rate frame's worth of symbols.
//!
//! In fixed-point builds the MDCT/FFT data is `i32` (`kiss_fft_scalar`, `celt_sig` scale
//! `SIG_SHIFT` = 12: full scale ±2^27); the range coder and resampler are the same code in
//! every build.

use opus_sys_optimized::{self as copt, Micro, Variant};
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::kiss_fft::opus_fft;
use opusorus::celt::laplace::{ec_laplace_decode, ec_laplace_encode};
use opusorus::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
use opusorus::celt::modes::opus_custom_mode_create;
use opusorus::celt::static_modes::{CeltMode, KissFftCpx, KissFftScalar};
use opusorus::silk::resampler::{SilkResamplerState, silk_resampler};
use opusorus_conformance::{Rng, signals};

use crate::{BenchError, GROUP_PREFIX, Impl, Result, Runner};

/// Converts a float kernel sample (nominal range ±1) to `kiss_fft_scalar`: unchanged in the
/// float build.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub const fn to_kfft(v: f32) -> KissFftScalar {
    v
}

/// Converts a float kernel sample (nominal range ±1) to `kiss_fft_scalar`: `celt_sig` Q27
/// (`SIG_SHIFT` 12 on top of 16-bit full scale) in fixed-point builds.
#[cfg(feature = "fixed-point")]
#[must_use]
pub fn to_kfft(v: f32) -> KissFftScalar {
    (f64::from(v) * f64::from(1u32 << 27)).round() as i32
}

/// The 48 kHz static CELT mode.
///
/// # Errors
/// Never in practice (the static mode always exists); propagated for form.
pub fn mode48() -> Result<&'static CeltMode> {
    opus_custom_mode_create(48000, 960).map_err(|e| BenchError(format!("mode: {e:?}")))
}

/// ICDF table of range coder op kind 1 (same as `csrc/bench_shim.c`).
pub const BENCH_ICDF: [u8; 8] = [250, 200, 150, 100, 60, 30, 10, 0];

/// Range coder buffer size.
pub const EC_BUF: usize = 4096;

/// Deterministic range coder op list (see `csrc/bench_shim.c` for the op kinds): 1000 ops
/// mixing equiprobable symbols, ICDF symbols, `bit_logp` flags, `enc_uint`, raw bits and
/// Laplace-coded energies, about 1.6 kB of output.
#[must_use]
pub fn ec_ops() -> Vec<[u32; 4]> {
    let mut rng = Rng::new(0x00EC_0DE5);
    (0..1000)
        .map(|_| match rng.range_i32(0, 5) {
            0 => {
                let ft = rng.range_i32(2, 300) as u32;
                [0, rng.next_u32() % ft, ft, 0]
            }
            1 => [1, rng.range_i32(0, 7) as u32, 0, 0],
            2 => {
                let logp = rng.range_i32(1, 15) as u32;
                // Mostly the likely value, as in CELT's flags.
                [2, u32::from(rng.range_i32(0, 7) == 0), logp, 0]
            }
            3 => {
                let ft = rng.range_i32(2, 1 << 20) as u32;
                [3, rng.next_u32() % ft, ft, 0]
            }
            4 => {
                let bits = rng.range_i32(1, 16) as u32;
                [4, rng.next_u32() & ((1 << bits) - 1), bits, 0]
            }
            _ => {
                let v = rng.range_i32(-6, 6);
                [5, v as u32, 72 << 7, 127 << 6]
            }
        })
        .collect()
}

/// Rust counterpart of `bench_ec_roundtrip` in `csrc/bench_shim.c` (same checksum).
#[must_use]
pub fn ec_roundtrip_rust(ops: &[[u32; 4]], buf: &mut [u8]) -> u32 {
    let mut enc = EcEnc::new(buf);
    for o in ops {
        match o[0] {
            0 => enc.encode(o[1], o[1] + 1, o[2]),
            1 => enc.enc_icdf(o[1] as usize, &BENCH_ICDF, 8),
            2 => enc.enc_bit_logp(o[1] != 0, o[2]),
            3 => enc.enc_uint(o[1], o[2]),
            4 => enc.enc_bits(o[1], o[2]),
            _ => {
                let mut v = o[1] as i32;
                ec_laplace_encode(&mut enc, &mut v, o[2], o[3] as i32);
            }
        }
    }
    enc.done();
    let mut sum = enc.rng ^ enc.error as u32;
    let mut dec = EcDec::new(buf);
    for o in ops {
        let v = match o[0] {
            0 => {
                let v = dec.decode(o[2]);
                dec.update(v, v + 1, o[2]);
                v
            }
            1 => dec.dec_icdf(&BENCH_ICDF, 8) as u32,
            2 => u32::from(dec.dec_bit_logp(o[2])),
            3 => dec.dec_uint(o[2]),
            4 => dec.dec_bits(o[2]),
            _ => ec_laplace_decode(&mut dec, o[2], o[3] as i32) as u32,
        };
        sum = sum.wrapping_mul(31).wrapping_add(v);
    }
    sum ^ dec.rng
}

/// Micro benchmark kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroBench {
    /// `clt_mdct_forward`, N = 1920 (one 20 ms long block), 48 kHz mode window.
    MdctForward,
    /// `clt_mdct_backward`, N = 1920.
    MdctBackward,
    /// `opus_fft` with the 480-point state of the 48 kHz mode.
    Fft480,
    /// Range coder encode + decode of [`ec_ops`].
    RangeCoder,
    /// `silk_resampler` 48 kHz -> 16 kHz (encoder direction), 20 ms of input per call.
    Resample48To16,
}

impl MicroBench {
    /// All micro benchmarks.
    pub const ALL: [Self; 5] = [
        Self::MdctForward,
        Self::MdctBackward,
        Self::Fft480,
        Self::RangeCoder,
        Self::Resample48To16,
    ];

    /// Criterion group name (with the build's [`GROUP_PREFIX`]).
    #[must_use]
    pub fn group(self) -> String {
        let name = match self {
            Self::MdctForward => "micro_mdct_forward_1920",
            Self::MdctBackward => "micro_mdct_backward_1920",
            Self::Fft480 => "micro_fft_480",
            Self::RangeCoder => "micro_range_coder_1000ops",
            Self::Resample48To16 => "micro_resampler_48k_to_16k",
        };
        format!("{GROUP_PREFIX}{name}")
    }

    /// Report label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MdctForward => "MDCT forward N=1920",
            Self::MdctBackward => "MDCT backward N=1920",
            Self::Fft480 => "FFT 480",
            Self::RangeCoder => "range coder enc+dec, 1000 ops",
            Self::Resample48To16 => "SILK resampler 48k->16k",
        }
    }

    /// Whether one call corresponds to one 20 ms frame of one channel (so a realtime factor is
    /// meaningful).
    #[must_use]
    pub const fn per_frame(self) -> bool {
        !matches!(self, Self::RangeCoder)
    }

    /// Builds the per-call runner for `imp`.
    ///
    /// # Errors
    /// If setup fails (unsupported resampler ratio, missing mode).
    pub fn runner(self, imp: Impl) -> Result<Runner> {
        let variant = match imp {
            Impl::Rust => None,
            Impl::CScalar => Some(Variant::Scalar),
            Impl::COpt => Some(Variant::Optimized),
        };
        match self {
            Self::MdctForward | Self::MdctBackward => mdct_runner(self, variant),
            Self::Fft480 => fft_runner(variant),
            Self::RangeCoder => {
                let ops = ec_ops();
                let mut buf = vec![0u8; EC_BUF];
                Ok(match variant {
                    None => Box::new(move || {
                        core::hint::black_box(ec_roundtrip_rust(
                            core::hint::black_box(&ops),
                            &mut buf,
                        ));
                        Ok(())
                    }),
                    Some(v) => {
                        let m = Micro::new(v);
                        Box::new(move || {
                            core::hint::black_box(
                                m.ec_roundtrip(core::hint::black_box(&ops), &mut buf),
                            );
                            Ok(())
                        })
                    }
                })
            }
            Self::Resample48To16 => resampler_runner(variant),
        }
    }
}

/// Input for the MDCT/FFT kernels: 8 blocks of music-like samples, times `scale`, to cycle
/// through.
fn kernel_input(len: usize, scale: f32) -> Vec<Vec<KissFftScalar>> {
    let x = signals::music_like(len * 8, 1, 48000, 0x3D17);
    x.chunks_exact(len)
        .map(|b| b.iter().map(|&v| to_kfft(v * scale)).collect())
        .collect()
}

fn mdct_runner(kind: MicroBench, variant: Option<Variant>) -> Result<Runner> {
    let mode = mode48()?;
    let n2 = mode.mdct.n as usize / 2;
    let overlap = mode.overlap as usize;
    let forward = kind == MicroBench::MdctForward;
    // Forward: N/2 + overlap time samples in, N/2 coefficients out. Backward: N/2 coefficients
    // in, N/2 + overlap samples out (scaled like real MDCT coefficients).
    let in_len = if forward { n2 + overlap } else { n2 };
    let scale = if forward { 1.0 } else { 0.05 };
    let inputs = kernel_input(in_len, scale);
    let mut out = vec![KissFftScalar::default(); n2 + overlap];
    let mut i = 0;
    Ok(match variant {
        None => Box::new(move || {
            let x = core::hint::black_box(&inputs[i]);
            i = (i + 1) % inputs.len();
            if forward {
                clt_mdct_forward(&mode.mdct, x, &mut out, &mode.window, overlap, 0, 1);
            } else {
                clt_mdct_backward(&mode.mdct, x, &mut out, &mode.window, overlap, 0, 1);
            }
            core::hint::black_box(&out);
            Ok(())
        }),
        Some(v) => {
            let m = Micro::new(v);
            Box::new(move || {
                let x = core::hint::black_box(&inputs[i]);
                i = (i + 1) % inputs.len();
                if forward {
                    m.mdct_forward(x, &mut out, 0);
                } else {
                    m.mdct_backward(x, &mut out, 0);
                }
                core::hint::black_box(&out);
                Ok(())
            })
        }
    })
}

fn fft_runner(variant: Option<Variant>) -> Result<Runner> {
    let mode = mode48()?;
    let st = &*mode.mdct.kfft[0];
    let nfft = st.nfft as usize;
    let inputs = kernel_input(2 * nfft, 1.0);
    let mut i = 0;
    Ok(match variant {
        None => {
            let cpx: Vec<Vec<KissFftCpx>> = inputs
                .iter()
                .map(|b| {
                    b.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| KissFftCpx { r: c[0], i: c[1] })
                        .collect()
                })
                .collect();
            let mut out = vec![KissFftCpx::default(); nfft];
            Box::new(move || {
                let x = core::hint::black_box(&cpx[i]);
                i = (i + 1) % cpx.len();
                opus_fft(st, x, &mut out);
                core::hint::black_box(&out);
                Ok(())
            })
        }
        Some(v) => {
            let m = Micro::new(v);
            let mut out = vec![KissFftScalar::default(); 2 * nfft];
            Box::new(move || {
                let x = core::hint::black_box(&inputs[i]);
                i = (i + 1) % inputs.len();
                m.fft(0, x, &mut out);
                core::hint::black_box(&out);
                Ok(())
            })
        }
    })
}

/// 20 ms blocks of 48 kHz speech-like `i16` input for the resampler.
fn resampler_input() -> Vec<Vec<i16>> {
    let x = signals::to_i16(&signals::speech_like(960 * 16, 1, 48000, 0x4816));
    x.as_chunks::<960>().0.iter().map(|c| c.to_vec()).collect()
}

fn resampler_runner(variant: Option<Variant>) -> Result<Runner> {
    let inputs = resampler_input();
    let mut out = vec![0i16; 320];
    let mut i = 0;
    Ok(match variant {
        None => {
            let mut st = SilkResamplerState::new(48000, 16000, true)
                .map_err(|e| BenchError(format!("rust resampler: {e:?}")))?;
            Box::new(move || {
                let x = core::hint::black_box(&inputs[i]);
                i = (i + 1) % inputs.len();
                silk_resampler(&mut st, &mut out, x, x.len() as i32);
                core::hint::black_box(&out);
                Ok(())
            })
        }
        Some(v) => {
            let mut st = copt::Resampler::new(v, 48000, 16000, true)
                .ok_or_else(|| BenchError("C resampler init failed".to_owned()))?;
            Box::new(move || {
                let x = core::hint::black_box(&inputs[i]);
                i = (i + 1) % inputs.len();
                st.run(&mut out, x);
                core::hint::black_box(&out);
                Ok(())
            })
        }
    })
}

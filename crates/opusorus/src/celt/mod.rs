//! CELT layer (port of `celt/`).

pub mod arch;
pub mod bands;
#[allow(clippy::module_inception, reason = "mirrors libopus celt/celt.c")]
pub mod celt;
pub mod celt_decoder;
pub mod celt_encoder;
pub mod celt_lpc;
pub mod cwrs;
pub mod entcode;
pub mod entdec;
pub mod entenc;
pub mod kiss_fft;
pub mod laplace;
pub mod mathops;
pub mod mdct;
#[cfg(feature = "qext")]
pub mod mini_kfft;
pub mod modes;
pub mod pitch;
pub mod quant_bands;
pub mod rate;
pub mod static_modes;
pub mod vq;

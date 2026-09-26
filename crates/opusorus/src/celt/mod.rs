//! CELT layer (port of `celt/`).
//!
//! Fixed-point builds (feature `fixed-point`) only compile the modules already converted to the
//! fixed-point types; the others are gated with `#[cfg(not(feature = "fixed-point"))]` until
//! their unit converts them (see `docs/FIXED_POINT.md`).

pub mod arch;
#[cfg(not(feature = "fixed-point"))]
pub mod bands;
#[cfg(not(feature = "fixed-point"))]
#[allow(clippy::module_inception, reason = "mirrors libopus celt/celt.c")]
pub mod celt;
#[cfg(not(feature = "fixed-point"))]
pub mod celt_decoder;
#[cfg(not(feature = "fixed-point"))]
pub mod celt_encoder;
#[cfg(not(feature = "fixed-point"))]
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
#[cfg(not(feature = "fixed-point"))]
pub mod pitch;
#[cfg(not(feature = "fixed-point"))]
pub mod quant_bands;
pub mod rate;
pub mod static_modes;
#[cfg(feature = "fixed-point")]
pub mod static_modes_fixed;
#[cfg(not(feature = "fixed-point"))]
pub mod vq;

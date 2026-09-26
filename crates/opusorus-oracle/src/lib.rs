//! # opusorus-oracle
//!
//! Test-only bindings to the vendored C libopus v1.6.1 (see `build.rs` for build flags). The
//! Rust port is verified against this library.
//!
//! * [`sys`] / [`api`]: public libopus API (raw + safe RAII wrappers).
//! * One module per port unit exposing *safe* wrappers over internal C functions, typically via
//!   small C shims in `csrc/<unit>.c` (compiled automatically) so tests never touch C structs.

pub mod api;
pub mod sys;

pub mod foundation;

// Fixed-point oracle only (`// oracle-build: fixed`).
#[cfg(feature = "fixed-point")]
pub mod fixed_foundation;
#[cfg(feature = "fixed-point")]
pub mod silk_encoder_fix;

// Units whose shims compile in both oracles (`// oracle-build: any`).
pub mod analysis;
pub mod celt_bands;
pub mod celt_decoder;
pub mod celt_encoder;
pub mod celt_fft;
pub mod celt_modes;
pub mod celt_pitch_lpc;
pub mod opus_encoder;
pub mod opus_packet;
pub mod silk_common;
pub mod silk_decoder;
pub mod silk_encoder_common;
pub mod silk_resampler;

// Float-only units: their shims (no `// oracle-build:` marker) are not compiled into a fixed-point
// oracle. A unit converted to fixed point marks its shim `any`/`fixed` and lifts the cfg here.
#[cfg(not(feature = "fixed-point"))]
pub mod dnn_core;
#[cfg(not(feature = "fixed-point"))]
pub mod dnn_dred;
#[cfg(not(feature = "fixed-point"))]
pub mod dnn_integration;
#[cfg(not(feature = "fixed-point"))]
pub mod dnn_osce;
#[cfg(not(feature = "fixed-point"))]
pub mod dnn_plc;
pub mod opus_decoder;
#[cfg(not(feature = "fixed-point"))]
pub mod silk_encoder_flp;
#[cfg(not(feature = "fixed-point"))]
pub mod tools_compare;

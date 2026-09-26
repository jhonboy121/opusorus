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

pub mod analysis;
pub mod celt_bands;
pub mod celt_decoder;
pub mod celt_encoder;
pub mod celt_fft;
pub mod celt_modes;
pub mod celt_pitch_lpc;
pub mod dnn_core;
pub mod dnn_dred;
pub mod dnn_osce;
pub mod dnn_plc;
pub mod opus_decoder;
pub mod opus_encoder;
pub mod opus_packet;
pub mod silk_common;
pub mod silk_decoder;
pub mod silk_encoder_common;
pub mod silk_encoder_flp;
pub mod silk_resampler;
pub mod tools_compare;

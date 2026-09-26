//! Port of libopus `dnn/` (deep PLC, DRED, OSCE).
//!
//! Unit `dnn_core`: the shared neural-network runtime and signal processing used by the DNN
//! features:
//! * [`nnet`] / [`nnet_arch`] / [`vec`]: layer types and the generic (non-SIMD) C kernels,
//! * [`parse_lpcnet_weights`]: the libopus weight-blob format and layer binding
//!   (`linear_init` / `conv2d_init`), plus the blob writer,
//! * [`common`], [`kiss99`], [`tansig_table`]: small helpers,
//! * [`burg`], [`freq`], [`lpcnet_tables`]: LPCNet analysis DSP,
//! * [`pitchdnn`], [`lpcnet_enc`]: the pitch DNN and the LPCNet feature extraction,
//! * [`nndsp`]: the adaptive conv/comb/shape layers of OSCE.
//!
//! Model weights are parsed from a libopus weight blob (upstream `USE_WEIGHTS_FILE` /
//! `OPUS_SET_DNN_BLOB` path, PLAN D-015); with the `dnn-weights-embedded` feature a blob is
//! compiled in and bound at initialization (`embedded`).
//!
//! With the `lossgen` feature the layer runtime also backs [`lossgen`], the packet loss
//! generator of `opus_demo -sim_loss` (its small model is compiled in).

// The layer runtime and weight-blob format (also all the `lossgen` model needs).
pub mod nnet;
pub mod nnet_arch;
pub mod parse_lpcnet_weights;
pub mod tansig_table;
pub mod vec;

// Shared by the DNN features (compiled out in a `lossgen`-only build, which may be
// fixed-point).
#[cfg(feature = "deep-plc")]
pub mod burg;
#[cfg(feature = "deep-plc")]
pub mod common;
#[cfg(all(feature = "deep-plc", feature = "dnn-weights-embedded"))]
pub mod embedded;
#[cfg(feature = "deep-plc")]
pub mod freq;
#[cfg(feature = "deep-plc")]
pub mod kiss99;
#[cfg(feature = "deep-plc")]
pub mod lpcnet_enc;
#[cfg(feature = "deep-plc")]
pub mod lpcnet_tables;
#[cfg(feature = "deep-plc")]
pub mod nndsp;
#[cfg(feature = "deep-plc")]
pub mod pitchdnn;

// Packet loss generator (`--enable-lossgen`).
#[cfg(feature = "lossgen")]
pub mod lossgen;
#[cfg(feature = "lossgen")]
pub mod lossgen_data;

// Deep PLC (unit `dnn_plc`).
#[cfg(feature = "deep-plc")]
pub mod fargan;
#[cfg(feature = "deep-plc")]
pub mod lpcnet_plc;
// DRED (unit `dnn_dred`).
#[cfg(feature = "dred")]
pub mod dred_coding;
#[cfg(feature = "dred")]
pub mod dred_decoder;
#[cfg(feature = "dred")]
pub mod dred_encoder;
#[cfg(feature = "dred")]
pub mod dred_rdovae;
// OSCE (unit `dnn_osce`).
#[cfg(feature = "osce")]
pub mod osce;
#[cfg(feature = "osce")]
pub mod osce_features;

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
//! Model weights are never compiled in: they are parsed from a libopus weight blob
//! (upstream `USE_WEIGHTS_FILE` / `OPUS_SET_DNN_BLOB` path, PLAN D-015).

pub mod burg;
pub mod common;
pub mod freq;
pub mod kiss99;
pub mod lpcnet_enc;
pub mod lpcnet_tables;
pub mod nndsp;
pub mod nnet;
pub mod nnet_arch;
pub mod parse_lpcnet_weights;
pub mod pitchdnn;
pub mod tansig_table;
pub mod vec;

//! SILK layer (port of `silk/`).
//!
//! The SILK decoder and the encoder's shared integer code are identical in the float and
//! fixed-point builds of libopus. Only the encoder analysis differs: `silk/float/*` (+ the float
//! encoder super-struct/API in `encoder.rs`) in the float build, `silk/fixed/*` in the
//! fixed-point build (feature `fixed-point`, see `docs/FIXED_POINT.md`); `encoder.rs` selects
//! the analysis of the build.

pub mod cng;
pub mod coding;
pub mod decoder;
pub mod define;
pub mod encoder;
pub mod encoder_common;
pub mod errors;
#[cfg(feature = "fixed-point")]
pub mod fixed;
#[cfg(not(feature = "fixed-point"))]
pub mod float;
#[cfg(feature = "fixed-point-debug")]
mod macro_debug;
pub mod macros;
pub mod nlsf;
pub mod nsq;
pub mod nsq_del_dec;
pub mod plc;
pub mod resampler;
pub mod sigproc;
pub mod structs;
pub mod tables;
pub mod tuning_parameters;
pub mod vad;

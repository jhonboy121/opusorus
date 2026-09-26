//! # opusorus
//!
//! A pure, safe-Rust port of the [Opus](https://opus-codec.org) audio codec, tracking
//! libopus v1.6.1. The float build of the codec is bit-exact with the C reference when both are
//! built for the same platform (see `docs/PORTING.md`).
//!
//! The crate is `no_std` + `alloc`; enable the default `std` feature to use the platform libm.

#![no_std]
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "numeric literals are copied verbatim from libopus so they round identically; \
              bit-exactness with the C reference depends on it"
)]
#![cfg_attr(
    not(feature = "internals"),
    allow(
        dead_code,
        reason = "port in progress: internal items are wired up incrementally"
    )
)]

#[allow(
    unused_extern_crates,
    reason = "alloc is used by modules still being ported"
)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod error;
pub use error::{Error, Result};

pub mod constants;
pub use constants::*;

#[doc(hidden)]
pub mod math;

#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod celt;
#[cfg(not(feature = "internals"))]
pub(crate) mod celt;

#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod silk;
#[cfg(not(feature = "internals"))]
pub(crate) mod silk;

#[cfg(all(
    feature = "internals",
    any(feature = "deep-plc", feature = "dred", feature = "osce")
))]
#[doc(hidden)]
pub mod dnn;
#[cfg(all(
    not(feature = "internals"),
    any(feature = "deep-plc", feature = "dred", feature = "osce")
))]
pub(crate) mod dnn;

// ---- Opus layer (src/*.c) ----
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod analysis;
#[cfg(not(feature = "internals"))]
pub(crate) mod analysis;
pub mod decoder;
pub mod encoder;
pub mod extensions;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod mapping_matrix;
#[cfg(not(feature = "internals"))]
pub(crate) mod mapping_matrix;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod mlp;
#[cfg(not(feature = "internals"))]
pub(crate) mod mlp;
pub mod ms_decoder;
pub mod ms_encoder;
pub mod multistream;
pub mod packet;
pub mod projection_decoder;
pub mod projection_encoder;
pub mod repacketizer;

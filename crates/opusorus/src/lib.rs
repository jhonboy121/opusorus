//! # opusorus
//!
//! A pure, safe-Rust port of the [Opus](https://opus-codec.org) audio codec, tracking
//! libopus v1.6.1. The float build of the codec is bit-exact with the C reference when both are
//! built for the same platform (see `docs/PORTING.md`).
//!
//! The crate is `no_std` + `alloc`; enable the default `std` feature to use the platform libm.
//!
//! # Fixed-point build (feature `fixed-point`) — NOT additive
//!
//! **`fixed-point` replaces the float implementation**, exactly like libopus'
//! `--enable-fixed-point`: the codec internals switch to the integer types of
//! `celt/arch.h` + `celt/fixed_generic.h` and the output becomes bit-exact with a *fixed-point*
//! libopus build instead of the float one. `fixed-res24` additionally selects the 24-bit
//! internal resolution (`ENABLE_RES24`). Do not enable these features in a library that other
//! crates depend on for float output: Cargo feature unification would switch them too. As with
//! upstream configure, they cannot be combined with `deep-plc`, `dred` or `osce`.
//!
//! The fixed-point port is **in progress** (see `docs/FIXED_POINT.md`): with `fixed-point`
//! enabled only the modules converted so far are compiled (range coder, CELT fixed-point
//! arithmetic and math, static modes, CWRS/Laplace, SILK shared integer code, packet
//! parsing/repacketizer). The encoder/decoder APIs are not available in that configuration yet.

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
#![cfg_attr(
    all(feature = "fixed-point", not(feature = "internals")),
    allow(
        unused_imports,
        reason = "fixed-point port in progress: converted modules have no users in the crate yet"
    )
)]

// Upstream configure: "--enable-fixed-point cannot be used with --enable-deep-plc, --enable-dred,
// and --enable-osce." (the DNN code exchanges float PCM/features with the codec).
#[cfg(all(
    feature = "fixed-point",
    any(feature = "deep-plc", feature = "dred", feature = "osce")
))]
compile_error!(
    "feature `fixed-point` cannot be combined with `deep-plc`, `dred` or `osce` (like libopus' \
     --enable-fixed-point); use explicit feature lists instead of --all-features"
);

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
    not(feature = "fixed-point"),
    any(feature = "deep-plc", feature = "dred", feature = "osce")
))]
#[doc(hidden)]
pub mod dnn;
#[cfg(all(
    not(feature = "internals"),
    not(feature = "fixed-point"),
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
#[cfg(all(not(feature = "fixed-point"), feature = "dred"))]
pub mod dred;
#[cfg(not(feature = "fixed-point"))]
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
#[cfg(not(feature = "fixed-point"))]
pub mod ms_encoder;
pub mod multistream;
pub mod packet;
pub mod projection_decoder;
#[cfg(not(feature = "fixed-point"))]
pub mod projection_encoder;
pub mod repacketizer;

pub use decoder::Decoder;
#[cfg(not(feature = "fixed-point"))]
pub use encoder::Encoder;
pub use ms_decoder::MsDecoder;
#[cfg(not(feature = "fixed-point"))]
pub use ms_encoder::MsEncoder;
pub use packet::ParsedPacket;
pub use projection_decoder::ProjectionDecoder;
#[cfg(not(feature = "fixed-point"))]
pub use projection_encoder::ProjectionEncoder;
pub use repacketizer::Repacketizer;

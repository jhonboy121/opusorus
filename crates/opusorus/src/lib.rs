//! # opusorus
//!
//! A pure, safe-Rust port of the [Opus](https://opus-codec.org) audio codec, tracking
//! libopus v1.6.1. The float build of the codec is bit-exact with the C reference when both are
//! built for the same platform (see `docs/PORTING.md`).
//!
//! The crate is `no_std` + `alloc`; enable the default `std` feature to use the platform libm.
//!
//! # Fixed-point build (features `fixed-point`, `fixed-res24`) — NOT additive
//!
//! **`fixed-point` replaces the float implementation**, exactly like libopus'
//! `--enable-fixed-point`: the codec internals switch to the integer types of
//! `celt/arch.h` + `celt/fixed_generic.h` (and the `silk/fixed` encoder analysis), and the
//! output becomes bit-exact with a *fixed-point* libopus build instead of the float one.
//! `fixed-res24` additionally selects the 24-bit internal resolution (`ENABLE_RES24`, the
//! default of upstream's autotools fixed-point build); without it `opus_res` is 16-bit. Both
//! combine with `qext` and `custom-modes`. Do not enable these features in a library that other
//! crates depend on for float output: Cargo feature unification would switch them too. As with
//! upstream configure, they cannot be combined with `deep-plc`, `dred` or `osce`.
//!
//! The fixed-point build is complete: the same public API (encoders, decoders, multistream,
//! projection, repacketizer, extensions, the float API included) is available, verified
//! bit-exact against a fixed-point libopus 1.6.1 (16- and 24-bit resolution, with and without
//! QEXT and custom modes), and its decoder passes the RFC 8251 conformance vectors. Behaviour
//! that differs from the float build, as it does in libopus:
//!
//! * **No soft clipping.** [`Decoder::decode`] converts the integer output directly (the float
//!   build soft-clips before converting to 16 bits), and [`Decoder::decode_float`] is the
//!   integer output scaled by `RES2FLOAT` (16-bit: to `[-1, 1)`). In the 16-bit
//!   build [`Decoder::decode`] writes the decoder's samples directly and `decode24` /
//!   `decode_float` have 16-bit precision (so Opus HD / `qext` output keeps only 16 bits; use
//!   `fixed-res24` for it); in the 24-bit build [`Decoder::decode24`] is the direct one.
//!   [`packet::pcm_soft_clip`] is a float function and stays available.
//! * **Encoder analysis.** The tonality/music analysis only runs at complexity 10 (the float
//!   build runs it from complexity 7), and the encoder's integer front end (DC rejection,
//!   stereo width, gain fades) differs from the float one, so bitstreams differ from the float
//!   build's. [`Encoder::encode_float`] converts its input with `FLOAT2RES` (saturated to 16
//!   bits in the 16-bit build).
//! * **Integer input path.** [`Encoder::encode`] (16-bit build) and [`Encoder::encode24`]
//!   (24-bit build) pass the samples straight to the encoder, without the float build's early
//!   frame-size check: an invalid frame size is still rejected with [`Error::BadArg`], and the
//!   final range then reads 0, as in C.
//! * The (hidden) surround energy mask takes Q24 `celt_glog` values; the version string ends
//!   in `-fixed`; the DRED API does not exist (libopus returns `OPUS_UNIMPLEMENTED`).
//! * `OPUS_FAST_INT64`: 32-bit targets (armv7, wasm32, x86) use libopus' 32-bit forms of the
//!   32x32 multiplies, which round differently; the output is bit-exact with libopus built for
//!   the same target.
//!
//! Where libopus relies on signed overflow that is undefined behaviour in C (and wraps in
//! practice), such as a stereo encoder with `OPUS_SET_LFE(1)` in the fixed-point build, the
//! port wraps explicitly, so debug and release builds behave like libopus and do not panic.
//! See `docs/FIXED_POINT.md`.

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

pub use decoder::Decoder;
pub use encoder::Encoder;
pub use ms_decoder::MsDecoder;
pub use ms_encoder::MsEncoder;
pub use packet::ParsedPacket;
pub use projection_decoder::ProjectionDecoder;
pub use projection_encoder::ProjectionEncoder;
pub use repacketizer::Repacketizer;

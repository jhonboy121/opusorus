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
//!
//! # DNN build options (features `dnn-debug-float`, `osce-training-data`, `lossgen`)
//!
//! * `dnn-debug-float` (libopus `--enable-dnn-debug-float`): the int8-quantized DNN layers keep
//!   float copies of their weights and compute with them. Upstream's switch only changes the
//!   compiled-in model tables; weight blobs carry their own arrays (a blob with the float
//!   copies computes in float, with or without the feature, as in libopus' `USE_WEIGHTS_FILE`
//!   builds). With `dnn-weights-embedded` the embedded blob must match the feature (checked at
//!   compile time; `scripts/gen_dnn_blob.sh --debug-float` writes the debug-float blob).
//! * `osce-training-data` (libopus `--enable-osce-training-data`, implies `osce` and `std`):
//!   the encoder (`voip` application) and the OSCE enhancer append training data to files in
//!   the current working directory, with upstream's names and formats; see
//!   `osce_training_data` (a debugging/training aid, not for production builds).
//! * `lossgen` (libopus `--enable-lossgen`): `lossgen`, the generative packet loss model of
//!   `opus_demo -sim_loss` (upstream links it into its tools only). Its small model is compiled
//!   in; it works in every build, float or fixed-point, without the DNN features.
//!
//! # No float API (feature `disable-float-api`, requires `fixed-point`)
//!
//! libopus' `--disable-float-api` (`DISABLE_FLOAT_API`): the float entry points
//! (`Encoder::encode_float`, `Decoder::decode_float`, their multistream, projection and
//! Opus Custom counterparts, `packet::pcm_soft_clip`) do not exist, and neither does the
//! encoder's float tonality/music analysis (`src/analysis.c`, `mlp.c`): the encoder then makes
//! its mode, bandwidth, DTX and CELT allocation decisions without it, bit-exact with a libopus
//! built with the same option. Like upstream, where the float build does not compile with
//! `DISABLE_FLOAT_API`, the feature requires `fixed-point` (`compile_error!` otherwise).
//!
//! # Checking fixed-point arithmetic (feature `fixed-point-debug`, implies `fixed-point`)
//!
//! libopus' `--enable-fixed-point-debug` (`FIXED_DEBUG`): the fixed-point macros are the
//! checking versions of `celt/fixed_debug.h` and `silk/MacroDebug.h`, which verify their operand
//! and result ranges, report every violation with libopus' message (to standard error with
//! `std`, or to a handler installed with `fixed_debug::set_handler`) and continue, and count
//! operations (`fixed_debug::celt_mips`). The output is bit-exact with libopus built with
//! `FIXED_DEBUG` (which differs from a release build where operands are out of range). For
//! debugging only: it is several times slower. See the `fixed_debug` module.
//!
//! # Upstream build options (`float-approx`, `assertions`, `fuzzing`, `disable-rfc8251`)
//!
//! Each mirrors a libopus configure/meson option and is verified against the C library built
//! with the same define:
//!
//! * **`float-approx`** (`--enable-float-approx`, `FLOAT_APPROX`): the float build's
//!   `celt_log2`/`celt_exp2` become libopus' polynomial approximations (no libm call) and
//!   `celt_isnan` tests the IEEE 754 bits. Output changes accordingly (bit-exact with a
//!   `FLOAT_APPROX` libopus, which upstream's autotools enable by default on x86, ARM and
//!   AArch64; meson and CMake do not). No effect on the fixed-point build, as upstream.
//! * **`assertions`** (`--enable-assertions`, `ENABLE_ASSERTIONS`): the libopus internal checks
//!   (`celt_assert`, `celt_sig_assert`, `silk_assert`, which the port otherwise has as
//!   `debug_assert!`s) are `assert!`s in every build profile, and the checks that the port
//!   leaves out of debug builds (`celt_assert(0)` before error returns, the `compute_ebands`
//!   layout checks, the non-converging SILK limiters) are compiled in: a failed check panics
//!   where libopus aborts. In the fixed-point build this includes the `celt_sig_assert`s that
//!   libopus fails on its signed-overflow (C UB) paths, such as a stereo encoder with
//!   `OPUS_SET_LFE(1)`: such an encoder panics, as libopus aborts (without the feature, both
//!   wrap). For testing and debugging; the port's memory safety never depends on it.
//! * **`fuzzing`** (`--enable-fuzzing`, `FUZZING`): the encoder makes random decisions (mode,
//!   mono/stereo, transient, TF, spreading, tapset, trim, band skipping, silence,
//!   anti-collapse, QEXT depths) from `glibc_rand` (a public module with the feature), a
//!   process-wide reproduction of glibc's `rand()`, so its bitstreams are bit-exact with a
//!   `FUZZING` libopus on a glibc host given the same seed (see the module for the state
//!   semantics). **Not for production.**
//! * **`disable-rfc8251`** (`--disable-rfc8251`, `DISABLE_UPDATE_DRAFT`): the decoder (and the
//!   encoder's band quantization) without the RFC 8251 bitstream fixes: RFC 6716 band folding
//!   and mono decoders with stereo phase inversion enabled. Such a decoder passes the original
//!   RFC 6716 test vectors, not the RFC 8251 ones.

#![no_std]
#![cfg_attr(
    feature = "disable-float-api",
    allow(
        rustdoc::broken_intra_doc_links,
        reason = "the docs link the float API, which `disable-float-api` removes"
    )
)]
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

// Upstream's float build does not compile with DISABLE_FLOAT_API (`FLOAT2INT16`,
// `celt_float2int16` and `opus_pcm_soft_clip_impl` are compiled out but used by the float
// decoder); CMake only uses it with OPUS_FIXED_POINT.
#[cfg(all(feature = "disable-float-api", not(feature = "fixed-point")))]
compile_error!(
    "feature `disable-float-api` requires `fixed-point` (libopus' float build does not compile \
     with DISABLE_FLOAT_API)"
);

#[allow(
    unused_extern_crates,
    reason = "alloc is used by modules still being ported"
)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// Declares a function `const` except with `fixed-point-debug`, where it (transitively) calls
/// the checking fixed-point macros, which report and count and so cannot be `const`.
macro_rules! const_unless_fixed_debug {
    ($(#[$m:meta])* $v:vis const fn $name:ident $($rest:tt)*) => {
        #[cfg(not(feature = "fixed-point-debug"))]
        $(#[$m])* $v const fn $name $($rest)*
        #[cfg(feature = "fixed-point-debug")]
        $(#[$m])* $v fn $name $($rest)*
    };
}

// libopus assertion macros (`celt_assert!`, `celt_sig_assert!`, `silk_assert!`); must precede the
// modules that use them.
#[macro_use]
mod assertions;

mod error;
pub use error::{Error, Result};

#[cfg(feature = "fixed-point-debug")]
pub mod fixed_debug;

pub mod constants;
pub use constants::*;

#[doc(hidden)]
pub mod math;

// glibc `rand()` for the fuzzing build (libopus `FUZZING`); `opus_demo` uses its generator.
#[cfg(any(feature = "fuzzing", feature = "internals"))]
#[cfg_attr(not(feature = "fuzzing"), doc(hidden))]
pub mod glibc_rand;

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

// The DNN runtime: the DNN features (float build only), or just its core for the `lossgen`
// packet loss model (any build, like upstream's `--enable-lossgen`).
#[cfg(all(
    feature = "internals",
    any(
        all(
            not(feature = "fixed-point"),
            any(feature = "deep-plc", feature = "dred", feature = "osce")
        ),
        feature = "lossgen"
    )
))]
#[doc(hidden)]
pub mod dnn;
#[cfg(all(
    not(feature = "internals"),
    any(
        all(
            not(feature = "fixed-point"),
            any(feature = "deep-plc", feature = "dred", feature = "osce")
        ),
        feature = "lossgen"
    )
))]
pub(crate) mod dnn;
#[cfg(feature = "lossgen")]
pub use dnn::lossgen;

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
#[cfg(all(feature = "internals", not(feature = "disable-float-api")))]
#[doc(hidden)]
pub mod mlp;
#[cfg(all(not(feature = "internals"), not(feature = "disable-float-api")))]
pub(crate) mod mlp;
pub mod ms_decoder;
pub mod ms_encoder;
pub mod multistream;
#[cfg(feature = "osce-training-data")]
pub mod osce_training_data;
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

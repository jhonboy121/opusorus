//! # opusorus C ABI
//!
//! A drop-in replacement for the libopus C library, backed by the safe-Rust
//! [`opusorus`](https://docs.rs/opusorus) port of libopus 1.6.1. It builds `libopusorus.so`
//! (`.dylib`/`.dll`) and `libopusorus.a` exporting every function of libopus'
//! `opus.h`, `opus_multistream.h` and `opus_projection.h` (and `opus_custom.h` with the
//! `custom-modes` feature) with the same signatures and semantics; the unmodified upstream
//! headers are shipped in `include/`.
//!
//! ## Design
//!
//! * **State handles.** libopus states are flat C structs that callers may allocate themselves
//!   (`*_get_size` + `*_init`) and even copy with `memcpy`. Here the C memory only holds a
//!   32-byte header; the Rust objects live in a registry keyed by the header's id, and byte
//!   copies of a state become independent states on first use. See [`handle`] for the scheme
//!   and its (narrow) limitations. `*_get_size` still reports header + Rust footprint, so it is
//!   a meaningful memory estimate; `*_destroy` releases the block with C `free` like libopus.
//! * **CTLs.** Stable Rust cannot define C variadic functions: `csrc/ctl.c` implements the
//!   `opus_*_ctl(st, request, ...)` functions, reading the argument list according to the
//!   request, and calls the non-variadic `opusorus_ctl_*` functions of [`ctl`].
//! * **Panics** never unwind into C: every export runs inside `catch_unwind` and reports a
//!   caught panic as `OPUS_INTERNAL_ERROR` (or NULL / no-op). The workspace release profile
//!   uses `panic = "abort"`, where a panic aborts the process instead.
//! * **Pointer arguments** that libopus dereferences unconditionally are checked: NULL gives
//!   `OPUS_BAD_ARG` instead of a crash, and buffer lengths are validated before slices are
//!   formed.
//!
//! ## Differences from libopus
//!
//! * `OPUS_GET_*_STATE` hands out small per-stream handles (valid while the parent lives), not
//!   interior pointers of the parent block.
//! * The private `CELT_GET_MODE` request is not supported (`OPUS_UNIMPLEMENTED`).
//! * DRED (`opus_dred_*`): without the `dred` feature it behaves like a libopus build without
//!   `ENABLE_DRED`; with it, like an `ENABLE_DRED` build whose weights are loaded with
//!   `OPUS_SET_DNN_BLOB` (`USE_WEIGHTS_FILE`), or compiled in with `dnn-weights-embedded`
//!   (see [`dred`]). `OPUS_SET_DNN_BLOB` is accepted in both cases (libopus with compiled-in
//!   weights answers `OPUS_UNIMPLEMENTED`).
//! * `opus_get_version_string` reports `"libopus 1.6.1 (opusorus)"` (`"libopus
//!   1.6.1-fixed (opusorus)"` in fixed-point builds, with `-fuzzing` after the version in
//!   fuzzing builds).
//!
//! ## Fixed-point
//!
//! With the `fixed-point` / `fixed-res24` features the library is backed by the fixed-point
//! build of `opusorus` (docs/FIXED_POINT.md) and behaves like libopus configured with
//! `--enable-fixed-point` (`--enable-fixed-res24` / `ENABLE_RES24`): the same exports, with
//! the fixed-point `opus_res` semantics (`opus_encode`/`opus_decode` are the native 16-bit
//! paths, or `opus_encode24`/`opus_decode24` with `fixed-res24`; the float API converts with
//! `FLOAT2RES`/`RES2FLOAT`, and `opus_decode_float` does not soft-clip), `OPUS_SET_ENERGY_MASK`
//! takes Q24 `opus_int32` values, and the DRED API answers `OPUS_UNIMPLEMENTED` (the DNN
//! features cannot be combined with fixed-point, as upstream).

pub mod ctl;
pub mod decoder;
pub mod dred;
pub mod encoder;
pub mod handle;
pub mod multistream;
pub mod packet;
pub mod projection;
pub mod types;
mod util;

#[cfg(feature = "custom-modes")]
pub mod custom;
#[cfg(feature = "internal-api")]
pub mod internal;

use core::ffi::{c_char, c_int};

/// `opus_strerror` strings, NUL-terminated.
const ERROR_STRINGS: [&core::ffi::CStr; 8] = [
    c"success",
    c"invalid argument",
    c"buffer too small",
    c"internal error",
    c"corrupted stream",
    c"request not implemented",
    c"invalid state",
    c"memory allocation failed",
];

/// Converts an Opus error code into a human readable string (`opus_strerror`).
#[unsafe(no_mangle)]
pub const extern "C" fn opus_strerror(error: c_int) -> *const c_char {
    match error {
        -7..=0 => ERROR_STRINGS[error.unsigned_abs() as usize].as_ptr(),
        _ => c"unknown error".as_ptr(),
    }
}

/// The library version string (`opus_get_version_string`). Starts with `"libopus "` like
/// upstream, which applications use to detect the library, and contains `"-fixed"` in
/// fixed-point builds (upstream: "applications may rely on the presence of this substring").
#[unsafe(no_mangle)]
pub const extern "C" fn opus_get_version_string() -> *const c_char {
    #[cfg(all(feature = "fixed-point", not(feature = "fuzzing")))]
    let s = c"libopus 1.6.1-fixed (opusorus)";
    #[cfg(all(not(feature = "fixed-point"), not(feature = "fuzzing")))]
    let s = c"libopus 1.6.1 (opusorus)";
    // Fuzzing builds (`FUZZING`) append "-fuzzing" like upstream.
    #[cfg(all(feature = "fixed-point", feature = "fuzzing"))]
    let s = c"libopus 1.6.1-fixed-fuzzing (opusorus)";
    #[cfg(all(not(feature = "fixed-point"), feature = "fuzzing"))]
    let s = c"libopus 1.6.1-fuzzing (opusorus)";
    s.as_ptr()
}

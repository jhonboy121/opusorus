//! # opusorus-tools
//!
//! Library core of the command-line tools shipped with libopus v1.6.1, ported to safe Rust:
//!
//! * [`demo::opus_demo_main`] (`src/opus_demo.c`): the reference encoder/decoder front end
//!   (`opus_demo` binary), which the conformance procedure uses to decode the test vectors.
//!   With this crate's `qext` feature it accepts 96 kHz and `-qext`; with `osce`,
//!   `-enable_osce_bwe`; with `dred`, DRED decoding after packet losses (`deep-plc`, `dred`,
//!   `osce` take the weights from `weights_blob.bin`, or compiled in with
//!   `dnn-weights-embedded`).
//! * [`compare::opus_compare`] (`src/opus_compare.c`): the RFC 6716 / RFC 8251 conformance
//!   quality metric used with the decoder test vectors (`opus_compare` binary).
//! * `compare::qext_compare` (`src/qext_compare.c`, feature `qext`): the Opus HD (QEXT) quality
//!   metric (`qext_compare` binary). It needs the QEXT-only `opusorus::celt::mini_kfft` port, so
//!   it is behind this crate's `qext` feature (which enables `opusorus/qext`).
//!
//! With the `fixed-point` / `fixed-res24` features the tools use a fixed-point build of
//! `opusorus` (docs/FIXED_POINT.md), like upstream's tools in a `--enable-fixed-point` build:
//! `opus_demo` is then byte-identical to the C `opus_demo` linked with a fixed-point libopus
//! (the comparison tools are float code in either build).
//!
//! The computations are bit-exact with the C tools (same float operation order, same libm), and
//! the `*_main` functions reproduce the C command-line parsing and the exact text the C tools
//! print on stderr, so the binaries are drop-in replacements.

#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "numeric literals are copied verbatim from libopus so they round identically; \
              bit-exactness with the C tools depends on it"
)]

pub mod compare;
pub mod demo;

//! # opusorus-tools
//!
//! Library core of the command-line tools shipped with libopus v1.6.1, ported to safe Rust:
//!
//! * [`compare::opus_compare`] (`src/opus_compare.c`): the RFC 6716 / RFC 8251 conformance
//!   quality metric used with the decoder test vectors (`opus_compare` binary).
//! * `compare::qext_compare` (`src/qext_compare.c`, feature `qext`): the Opus HD (QEXT) quality
//!   metric (`qext_compare` binary). It needs the QEXT-only `opusorus::celt::mini_kfft` port, so
//!   it is behind this crate's `qext` feature (which enables `opusorus/qext`).
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

//! Shared harness code for the opusorus fuzz targets.
//!
//! Every target reads its input through a tiny hand-rolled byte format ([`Reader`]) rather than
//! `arbitrary`, so the seed-corpus generator (`examples/seed_corpus.rs`) can write structured
//! seeds (real packets from the C encoder, realistic encoder configurations) with the matching
//! [`Writer`]. Reads past the end of the input yield zeros, so every byte string is a valid
//! input.
//!
//! * [`dec`]: the decoder operation stream shared by `decode`, `differential_decode`,
//!   `multistream_decode` and `projection_decode`, plus the Rust / C decoder adapters and the
//!   differential driver.
//! * [`enc`]: the encoder operation stream (configuration, CTL changes, synthesized PCM) shared
//!   by `encode`, `differential_encode` and `roundtrip_invariants`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    reason = "fuzz harness: a panic (failed check) is exactly the signal libFuzzer records"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "byte-level input decoding deliberately truncates and reinterprets"
)]

pub mod dec;
pub mod enc;

use opusorus::Error;

/// Sampling rates the codec accepts (96 kHz only with `qext`).
#[cfg(feature = "qext")]
pub const FS_LIST: [i32; 6] = [8000, 12000, 16000, 24000, 48000, 96000];
/// Sampling rates the codec accepts (96 kHz only with `qext`).
#[cfg(not(feature = "qext"))]
pub const FS_LIST: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Picks a sampling rate from a selector byte.
#[must_use]
pub const fn pick_fs(sel: u8) -> i32 {
    FS_LIST[sel as usize % FS_LIST.len()]
}

/// Index of `fs` in [`FS_LIST`] (the selector byte the seed generator writes).
#[must_use]
pub fn fs_sel(fs: i32) -> u8 {
    FS_LIST
        .iter()
        .position(|&f| f == fs)
        .expect("supported rate") as u8
}

/// A Rust result in the C convention (negative libopus code on error).
pub fn code<T>(r: opusorus::Result<T>) -> Result<T, i32> {
    r.map_err(Error::code)
}

/// Sequential reader over the fuzz input. Reads past the end return zeros / empty slices.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Wraps `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    /// Whether all input has been consumed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }
    /// Bytes left.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    /// Next byte (0 past the end).
    pub fn u8(&mut self) -> u8 {
        match self.data.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => 0,
        }
    }
    /// Little-endian `u16`.
    pub fn u16(&mut self) -> u16 {
        u16::from_le_bytes([self.u8(), self.u8()])
    }
    /// Little-endian `u32`.
    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes([self.u8(), self.u8(), self.u8(), self.u8()])
    }
    /// Little-endian `i32`.
    pub fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
    /// Up to `n` bytes (fewer at the end of the input), borrowed from the input.
    pub fn bytes(&mut self, n: usize) -> &'a [u8] {
        let start = self.pos.min(self.data.len());
        let end = start.saturating_add(n).min(self.data.len());
        self.pos = end;
        &self.data[start..end]
    }
    /// A `u16`-length-prefixed byte string.
    pub fn blob(&mut self) -> &'a [u8] {
        let n = self.u16() as usize;
        self.bytes(n)
    }
}

/// Serializer matching [`Reader`] (used by the seed-corpus generator).
#[derive(Debug, Clone, Default)]
pub struct Writer(pub Vec<u8>);

impl Writer {
    /// Empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }
    /// Appends a byte.
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    /// Appends a little-endian `u16`.
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    /// Appends a little-endian `i32`.
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    /// Appends raw bytes.
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
    /// Appends a `u16`-length-prefixed byte string (truncated to 65535 bytes).
    pub fn blob(&mut self, b: &[u8]) -> &mut Self {
        let n = b.len().min(usize::from(u16::MAX));
        self.u16(n as u16);
        self.bytes(&b[..n])
    }
}

/// Deterministic xorshift64* PRNG for signal synthesis (no external dependency).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Creates a PRNG (seed 0 is remapped).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    /// Next 64 random bits.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform float in `[-1, 1)`.
    pub fn f32_sym(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 23) as f32 - 1.0
    }
    /// Uniform integer in `0..n` (`n > 0`).
    pub const fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Asserts two float slices are bit-identical, reporting the first difference.
#[track_caller]
pub fn assert_f32_bits_eq(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!("{what}: sample {i} differs: rust={} c={}", a[i], b[i]);
    }
}

/// Asserts two slices are equal, reporting the first difference.
#[track_caller]
pub fn assert_slice_eq<T: PartialEq + core::fmt::Debug>(what: &str, a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{what}: element {i} differs: rust={:?} c={:?}", a[i], b[i]);
    }
}

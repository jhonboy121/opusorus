//! Shared helpers for differential tests against the C oracle.

#![allow(
    clippy::missing_panics_doc,
    reason = "test helpers panic on mismatch by design"
)]

/// Deterministic xorshift64* PRNG (tests must be reproducible; no external RNG dependency).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Creates a PRNG from a seed (0 is remapped).
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
    /// Next 32 random bits.
    pub const fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    /// Uniform integer in `[lo, hi]` (inclusive).
    pub const fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi as i64 - lo as i64 + 1) as u64;
        (lo as i64 + (self.next_u64() % span) as i64) as i32
    }
    /// Uniform float in `[-1, 1)`.
    pub fn f32_sym(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
    /// Random `i16`.
    pub const fn i16(&mut self) -> i16 {
        self.next_u32() as i16
    }
    /// Fills a byte buffer.
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        for b in buf {
            *b = self.next_u32() as u8;
        }
    }
}

/// Test signal generators (interleaved, `channels` channels, `n` samples per channel).
pub mod signals {
    use super::Rng;

    /// Sum of sines with slowly varying amplitude plus a little noise, in `[-1, 1]`.
    #[must_use]
    pub fn music_like(n: usize, channels: usize, fs: u32, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let freqs = [
            110.0f64,
            220.0,
            440.0 * 1.5,
            1234.5,
            3001.0,
            7000.0,
            12000.0,
        ];
        let mut out = Vec::with_capacity(n * channels);
        for i in 0..n {
            let t = i as f64 / fs as f64;
            for c in 0..channels {
                let mut v = 0.0f64;
                for (k, f) in freqs.iter().enumerate() {
                    let amp = 0.08 * (1.0 + (t * (0.3 + k as f64 * 0.1) + c as f64).sin());
                    v +=
                        amp * (2.0 * core::f64::consts::PI * f * (1.0 + 0.01 * c as f64) * t).sin();
                }
                v += 0.01 * rng.f32_sym() as f64;
                out.push(v.clamp(-1.0, 1.0) as f32);
            }
        }
        out
    }

    /// Speech-like: pulse train through a crude formant filter with syllabic envelope.
    #[must_use]
    pub fn speech_like(n: usize, channels: usize, fs: u32, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(n * channels);
        let (mut y1, mut y2) = (0.0f64, 0.0f64);
        let mut phase = 0.0f64;
        for i in 0..n {
            let t = i as f64 / fs as f64;
            let f0 = 120.0 + 40.0 * (t * 3.0).sin();
            phase += f0 / fs as f64;
            let voiced = (t * 4.0).sin() > -0.3;
            let exc = if voiced {
                if phase >= 1.0 {
                    phase -= 1.0;
                    1.0
                } else {
                    0.0
                }
            } else {
                0.3 * rng.f32_sym() as f64
            };
            let r = 0.97f64;
            let w = 2.0 * core::f64::consts::PI * (500.0 + 300.0 * (t * 2.0).sin()) / fs as f64;
            let y = exc + 2.0 * r * w.cos() * y1 - r * r * y2;
            y2 = y1;
            y1 = y;
            let env = 0.5 * (1.0 + (t * 5.0).sin()).max(0.05);
            let v = (0.05 * y * env).clamp(-1.0, 1.0) as f32;
            for c in 0..channels {
                out.push(if c == 0 {
                    v
                } else {
                    0.8 * v + 0.01 * rng.f32_sym()
                });
            }
        }
        out
    }

    /// White noise at the given amplitude.
    #[must_use]
    pub fn noise(n: usize, channels: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n * channels).map(|_| amp * rng.f32_sym()).collect()
    }

    /// Converts float samples to `i16` (C `FLOAT2INT16`-like, with rounding and clipping).
    #[must_use]
    pub fn to_i16(x: &[f32]) -> Vec<i16> {
        x.iter()
            .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect()
    }
}

/// Asserts two slices are bit-identical, reporting the first mismatch with context.
#[track_caller]
pub fn assert_bits_eq_f32(what: &str, rust: &[f32], c: &[f32]) {
    assert_eq!(rust.len(), c.len(), "{what}: length mismatch");
    if let Some(i) = rust
        .iter()
        .zip(c)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        panic!(
            "{what}: first mismatch at {i}: rust={} ({:#010x}) c={} ({:#010x})",
            rust[i],
            rust[i].to_bits(),
            c[i],
            c[i].to_bits()
        );
    }
}

/// Asserts two slices are identical, reporting the first mismatch.
#[track_caller]
pub fn assert_slice_eq<T: PartialEq + core::fmt::Debug>(what: &str, rust: &[T], c: &[T]) {
    assert_eq!(rust.len(), c.len(), "{what}: length mismatch");
    if let Some(i) = rust.iter().zip(c).position(|(a, b)| a != b) {
        panic!(
            "{what}: first mismatch at {i}: rust={:?} c={:?}",
            rust[i], c[i]
        );
    }
}

/// `FLOAT2RES` for test signals: the library's (`opusorus::celt::arch::float2res`), or with
/// `disable-float-api` (where libopus and the port compile `FLOAT2INT16` / `FLOAT2INT24` out) a
/// copy of the `celt/float_cast.h` definition, as the oracle shims use (`oracle_float_cast.h`).
#[must_use]
#[cfg_attr(
    not(feature = "fixed-point"),
    expect(
        clippy::missing_const_for_fn,
        reason = "`FLOAT2RES` is only a const identity in the float build"
    )
)]
pub fn float2res(x: f32) -> opusorus::celt::arch::OpusRes {
    #[cfg(not(feature = "disable-float-api"))]
    return opusorus::celt::arch::float2res(x);
    #[cfg(all(feature = "disable-float-api", feature = "fixed-res24"))]
    {
        let x = x * (32768.0 * 256.0);
        let x = if x > -16_777_216.0 { x } else { -16_777_216.0 };
        let x = if x < 16_777_216.0 { x } else { 16_777_216.0 };
        opusorus::math::lrintf(x)
    }
    #[cfg(all(feature = "disable-float-api", not(feature = "fixed-res24")))]
    {
        let x = x * 32768.0;
        let x = if x > -32768.0 { x } else { -32768.0 };
        let x = if x < 32767.0 { x } else { 32767.0 };
        opusorus::math::lrintf(x) as i16
    }
}

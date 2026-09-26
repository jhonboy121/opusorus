//! The C library `rand()` of glibc, for the `fuzzing` feature (libopus `FUZZING`).
//!
//! A libopus build with `--enable-fuzzing` makes some encoder decisions at random with C
//! `rand()`: the CELT transient, TF resolution, spreading/tapset, allocation trim, band skipping,
//! silence and anti-collapse decisions, the QEXT extra-allocation depths, and the Opus layer's
//! mono/stereo and SILK/CELT mode decisions. To stay bit-exact with such a build on a glibc host
//! (Linux), the port draws those numbers from [`rand`], a reproduction of glibc's generator
//! (`random_r` with the default `TYPE_3` state: the additive lagged Fibonacci generator
//! `r[i] = r[i-3] + r[i-31]`, output `r >> 1`, [`RAND_MAX`] = 2^31 - 1).
//!
//! # State semantics
//!
//! C's `rand()` state is **one per process**, shared by every encoder (and everything else in the
//! program that calls `rand()`), in call order. Per-encoder state would not be equivalent: a
//! multistream or projection encoder runs several encoders from one call, `opus_demo` draws its
//! packet losses from the same generator, and two encoders used alternately interleave their
//! draws. So the port's generator is also **global**: one state per process (per copy of this
//! crate), shared by all threads and encoders, guarded by a spin lock (the crate is `no_std`).
//! Like C, it starts in the state of `srand(1)`; [`srand`] reseeds it with glibc's `srand`
//! semantics (seed 0 is seed 1). A Rust program reproduces a C program's encoder output when it
//! seeds the same way and calls [`rand`] wherever the C program calls `rand()` itself (the
//! `opus_demo` port does). The C library's own `rand()` is a separate generator: this crate
//! never calls or advances it. Hosts whose C library has another `rand()` (macOS, Windows,
//! musl) produce other random decisions in C; the port always uses glibc's.
//!
//! Without the `fuzzing` feature the encoder never calls [`rand`]; the module is then only
//! compiled for the `internals` feature ([`GlibcRand`] is used by the `opus_demo` port).

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// `RAND_MAX` of glibc.
pub const RAND_MAX: i32 = 0x7fff_ffff;

/// Whether this build is a fuzzing build (feature `fuzzing`, libopus `FUZZING`): the encoder
/// then draws from the global generator ([`rand`]).
pub const FUZZING: bool = cfg!(feature = "fuzzing");

/// One glibc `rand()` generator state (`random_r` with the default `TYPE_3` state).
///
/// [`GlibcRand::default`] is the state an unseeded C program starts with (`srand(1)`).
#[derive(Debug, Clone)]
pub struct GlibcRand {
    r: [u32; 31],
    f: usize,
    b: usize,
}

impl GlibcRand {
    /// The state after `srand(seed)` (glibc maps seed 0 to 1).
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        let mut r = [0u32; 31];
        let seed = if seed == 0 { 1 } else { seed };
        // glibc keeps the state as int32_t and the seed as a (possibly negative) int32_t.
        let mut word = seed as i32 as i64;
        r[0] = seed;
        let mut i = 1;
        while i < 31 {
            let hi = word / 127_773;
            let lo = word % 127_773;
            word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            r[i] = word as u32;
            i += 1;
        }
        let mut s = Self { r, f: 3, b: 0 };
        // srandom_r discards the first 10 * 31 outputs.
        let mut k = 0;
        while k < 310 {
            s.next_value();
            k += 1;
        }
        s
    }

    /// `rand()`: the next value in `0..=RAND_MAX`.
    pub const fn next_value(&mut self) -> i32 {
        let val = self.r[self.f].wrapping_add(self.r[self.b]);
        self.r[self.f] = val;
        self.f = if self.f == 30 { 0 } else { self.f + 1 };
        self.b = if self.b == 30 { 0 } else { self.b + 1 };
        (val >> 1) as i32
    }
}

impl Default for GlibcRand {
    fn default() -> Self {
        Self::new(1)
    }
}

/// The process-wide generator: the state words and indices are atomics accessed only while
/// `lock` is held (so the crate needs neither `std` nor `unsafe`).
struct Global {
    lock: AtomicBool,
    r: [AtomicU32; 31],
    f: AtomicUsize,
    b: AtomicUsize,
}

impl Global {
    const fn new(s: &GlibcRand) -> Self {
        let mut r = [const { AtomicU32::new(0) }; 31];
        let mut i = 0;
        while i < 31 {
            r[i] = AtomicU32::new(s.r[i]);
            i += 1;
        }
        Self {
            lock: AtomicBool::new(false),
            r,
            f: AtomicUsize::new(s.f),
            b: AtomicUsize::new(s.b),
        }
    }

    /// Runs `op` on a copy of the state under the lock and stores the result back.
    fn with<R>(&self, op: impl FnOnce(&mut GlibcRand) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        let mut s = GlibcRand {
            r: core::array::from_fn(|i| self.r[i].load(Ordering::Relaxed)),
            f: self.f.load(Ordering::Relaxed),
            b: self.b.load(Ordering::Relaxed),
        };
        let out = op(&mut s);
        for (a, v) in self.r.iter().zip(s.r) {
            a.store(v, Ordering::Relaxed);
        }
        self.f.store(s.f, Ordering::Relaxed);
        self.b.store(s.b, Ordering::Relaxed);
        self.lock.store(false, Ordering::Release);
        out
    }
}

static GLOBAL: Global = Global::new(&GlibcRand::new(1));

/// C `srand(seed)` on the process-wide generator (see the module docs).
pub fn srand(seed: u32) {
    let s = GlibcRand::new(seed);
    GLOBAL.with(|g| *g = s);
}

/// C `rand()` on the process-wide generator: the next value in `0..=RAND_MAX`.
#[must_use]
pub fn rand() -> i32 {
    GLOBAL.with(GlibcRand::next_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glibc_sequence() {
        // First outputs of glibc rand() in an unseeded program.
        let mut r = GlibcRand::default();
        let v: [i32; 5] = core::array::from_fn(|_| r.next_value());
        assert_eq!(
            v,
            [1804289383, 846930886, 1681692777, 1714636915, 1957747793]
        );
        // srand(0) is srand(1).
        assert_eq!(GlibcRand::new(0).next_value(), 1804289383);
        assert_eq!(GlibcRand::new(42).next_value(), 71876166);
        // A seed above INT_MAX is used as a negative int32_t.
        let mut big = GlibcRand::new(0x8000_0001);
        assert!((0..100).all(|_| (0..=RAND_MAX).contains(&big.next_value())));
    }

    // Without `fuzzing` nothing else in this test binary draws from the global generator
    // (the fuzzing differential tests check it against C's under a lock).
    #[cfg(not(feature = "fuzzing"))]
    #[test]
    fn global_generator() {
        srand(42);
        assert_eq!(rand(), 71876166);
        srand(1);
        let mut r = GlibcRand::default();
        for _ in 0..1000 {
            assert_eq!(rand(), r.next_value());
        }
    }
}

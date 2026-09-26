//! Helpers shared by the ports of the libopus test programs (`tests/test_opus_common.h`).
//!
//! Every test binary includes this module with `mod common;` and uses a subset of it.

#![allow(
    dead_code,
    reason = "each integration test binary uses a different subset of these helpers"
)]

/// Default seed of the fuzz-style loops. libopus seeds from `time()` (or the `SEED` environment
/// variable / first argument); the Rust ports are deterministic unless `SEED` is set.
pub const DEFAULT_SEED: u32 = 0x5EED_0105;

/// The seed for the random tests: the `SEED` environment variable if it is set (as the libopus
/// test programs accept), otherwise [`DEFAULT_SEED`].
#[must_use]
pub fn seed() -> u32 {
    match std::env::var("SEED") {
        Ok(s) => match s.trim().parse() {
            Ok(v) => v,
            Err(e) => panic!("invalid SEED {s:?}: {e}"),
        },
        Err(_) => DEFAULT_SEED,
    }
}

/// Whether the long (full-length, as in libopus) variants were requested through the
/// `OPUS_TEST_FULL` environment variable. The full variants are also available as `#[ignore]`d
/// tests; this only lets a single run switch every test to full length.
#[must_use]
pub fn full_length() -> bool {
    std::env::var_os("OPUS_TEST_FULL").is_some()
}

/// The multiply-with-carry generator of George Marsaglia used by the libopus tests
/// (`fast_rand` in `tests/test_opus_common.h`). `Rz` and `Rw` are the C globals.
#[derive(Debug, Clone)]
pub struct FastRand {
    pub rz: u32,
    pub rw: u32,
}

impl FastRand {
    /// `Rw=Rz=iseed`.
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self { rz: seed, rw: seed }
    }

    /// Port of `fast_rand`.
    pub const fn next(&mut self) -> u32 {
        self.rz = 36969u32
            .wrapping_mul(self.rz & 65535)
            .wrapping_add(self.rz >> 16);
        self.rw = 18000u32
            .wrapping_mul(self.rw & 65535)
            .wrapping_add(self.rw >> 16);
        (self.rz << 16).wrapping_add(self.rw)
    }

    /// `RAND_SAMPLE(a)`: `a[fast_rand() % len]`.
    pub const fn sample<T: Copy>(&mut self, a: &[T]) -> T {
        a[(self.next() % a.len() as u32) as usize]
    }
}

/// glibc's `rand()` (the `TYPE_3` additive feedback generator behind `random_r`), so the ports of
/// tests that call the C library `rand()` draw the same values as the C programs do on glibc.
#[derive(Debug, Clone)]
pub struct GlibcRand {
    r: [u32; 34],
    pos: usize,
}

impl GlibcRand {
    /// `RAND_MAX`.
    pub const RAND_MAX: u32 = 2_147_483_647;

    /// `srand(seed)` (an unseeded `rand()` behaves as `srand(1)`).
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut r = [0u32; 34];
        r[0] = if seed == 0 { 1 } else { seed };
        for i in 1..31 {
            // r[i] = (16807 * r[i-1]) % 2147483647, computed without overflow as glibc does.
            let prev = r[i - 1] as i32;
            let hi = prev / 127_773;
            let lo = prev % 127_773;
            let mut word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            r[i] = word as u32;
        }
        for i in 31..34 {
            r[i] = r[i - 31];
        }
        let mut g = Self { r, pos: 0 };
        for _ in 34..344 {
            g.step();
        }
        g
    }

    const fn step(&mut self) -> u32 {
        let v = self.r[(self.pos + 3) % 34].wrapping_add(self.r[(self.pos + 31) % 34]);
        self.r[self.pos] = v;
        self.pos = (self.pos + 1) % 34;
        v
    }

    /// `rand()`: a value in `0..=RAND_MAX`.
    pub const fn next(&mut self) -> i32 {
        (self.step() >> 1) as i32
    }
}

/// Port of `deb2_impl` (`tests/test_opus_common.h`).
fn deb2_impl(t: &mut [u8], res: &mut [u8], p: &mut usize, k: usize, x: usize, y: usize) {
    if x > 2 {
        if y < 3 {
            for i in 0..y {
                *p -= 1;
                res[*p] = t[i + 1];
            }
        }
    } else {
        t[x] = t[x - y];
        deb2_impl(t, res, p, k, x + 1, y);
        for i in usize::from(t[x - y]) + 1..k {
            t[x] = i as u8;
            deb2_impl(t, res, p, k, x + 1, x);
        }
    }
}

/// Port of `debruijn2`: a De Bruijn sequence (k,2) of length `k^2`.
#[must_use]
pub fn debruijn2(k: usize) -> Vec<u8> {
    let mut res = vec![0u8; k * k];
    let mut t = vec![0u8; k * 2];
    let mut p = k * k;
    deb2_impl(&mut t, &mut res, &mut p, k, 1, 1);
    res
}

/// Byte offset of `inner` (a subslice) within `outer`, for the C tests that compare pointers.
#[must_use]
pub fn offset_in(outer: &[u8], inner: &[u8]) -> usize {
    let o = outer.as_ptr() as usize;
    let i = inner.as_ptr() as usize;
    assert!(
        i >= o && i + inner.len() <= o + outer.len(),
        "not a subslice"
    );
    i - o
}

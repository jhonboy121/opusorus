//! Kernel benchmarks: CELT MDCT forward/backward (N = 1920), 480-point FFT, range coder and the
//! SILK 48 kHz -> 16 kHz resampler, for `rust`, `c_scalar` and `c_opt`.

#![cfg_attr(
    not(feature = "fixed-point"),
    expect(
        clippy::expect_used,
        reason = "benchmark harness: a failing setup or call aborts the run with a message"
    )
)]

#[cfg(not(feature = "fixed-point"))]
use core::time::Duration;

#[cfg(not(feature = "fixed-point"))]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(not(feature = "fixed-point"))]
use opusorus_bench::Impl;
#[cfg(not(feature = "fixed-point"))]
use opusorus_bench::micro::MicroBench;

#[cfg(not(feature = "fixed-point"))]
fn micro(c: &mut Criterion) {
    for bench in MicroBench::ALL {
        let mut g = c.benchmark_group(bench.group());
        for imp in Impl::ALL {
            let mut run = bench.runner(imp).expect("benchmark setup");
            g.bench_function(imp.id(), |b| b.iter(|| run().expect("kernel call")));
        }
        g.finish();
    }
}

#[cfg(not(feature = "fixed-point"))]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(50);
    targets = micro
}
#[cfg(not(feature = "fixed-point"))]
criterion_main!(benches);

/// Fixed-point builds of `opusorus` have no codec API yet (docs/FIXED_POINT.md).
#[cfg(feature = "fixed-point")]
fn main() {}

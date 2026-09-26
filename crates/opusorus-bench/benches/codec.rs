//! Codec benchmarks: one iteration = one 20 ms frame, for opusorus (`rust`), the scalar C
//! oracle (`c_scalar`) and the optimized upstream build (`c_opt`).
//!
//! Groups: `decode_<cfg>`, `encode_c<complexity>_<cfg>` for the single-stream configurations of
//! `opusorus_bench::codec_configs` and the 5.1 surround configuration. `scripts/bench_report.sh`
//! turns the results into a markdown table.

#![cfg_attr(
    not(feature = "fixed-point"),
    expect(
        clippy::expect_used,
        reason = "benchmark harness: a failing setup or frame aborts the run with a message"
    )
)]

#[cfg(not(feature = "fixed-point"))]
use core::time::Duration;

#[cfg(not(feature = "fixed-point"))]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(not(feature = "fixed-point"))]
use opusorus_bench::{CodecBench, Impl};

#[cfg(not(feature = "fixed-point"))]
fn codec(c: &mut Criterion) {
    for bench in CodecBench::all() {
        let mut g = c.benchmark_group(bench.group());
        for imp in Impl::ALL {
            let mut run = bench.runner(imp).expect("benchmark setup");
            g.bench_function(imp.id(), |b| b.iter(|| run().expect("frame")));
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
    targets = codec
}
#[cfg(not(feature = "fixed-point"))]
criterion_main!(benches);

/// Fixed-point builds of `opusorus` have no codec API yet (docs/FIXED_POINT.md).
#[cfg(feature = "fixed-point")]
fn main() {}

//! Runs one benchmark body in a loop, for profilers (`perf record`, `valgrind --tool=callgrind`).
//!
//! Usage: `bench_profile <group> [rust|c_scalar|c_opt] [<seconds> | <n>i]`, e.g.
//! `perf record -g target/release/bench_profile encode_c10_celt_48k_stereo_128k rust 10` or
//! `valgrind --tool=callgrind target/release/bench_profile decode_silk_16k_mono_16k_voip rust 2000i`
//! (a count with an `i` suffix runs exactly that many iterations, for deterministic instruction
//! counts). `bench_profile list` prints the group names. Prints iterations and ns per iteration.

use std::error::Error;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use opusorus_bench::{Impl, report_rows, runner_for_group};

/// How long to run.
enum Budget {
    Time(Duration),
    Iterations(u64),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = io::stdout();
    let mut w = out.lock();
    let Some(group) = args.first() else {
        return Err(
            "usage: bench_profile <group>|list [rust|c_scalar|c_opt] [<seconds>|<n>i]".into(),
        );
    };
    if group == "list" {
        for row in report_rows() {
            writeln!(w, "{}", row.group)?;
        }
        return Ok(());
    }
    let imp = match args.get(1) {
        Some(id) => Impl::from_id(id).ok_or_else(|| format!("unknown implementation {id:?}"))?,
        None => Impl::Rust,
    };
    let budget = match args.get(2) {
        Some(s) => match s.strip_suffix('i') {
            Some(n) => Budget::Iterations(n.parse()?),
            None => Budget::Time(Duration::from_secs_f64(s.parse()?)),
        },
        None => Budget::Time(Duration::from_secs(10)),
    };
    let mut run = runner_for_group(group, imp)?;
    let start = Instant::now();
    let mut iters = 0u64;
    match budget {
        Budget::Time(d) => {
            while start.elapsed() < d {
                for _ in 0..64 {
                    run()?;
                }
                iters += 64;
            }
        }
        Budget::Iterations(n) => {
            for _ in 0..n {
                run()?;
            }
            iters = n;
        }
    }
    let ns = start.elapsed().as_nanos() as f64 / iters.max(1) as f64;
    writeln!(
        w,
        "{group} {}: {iters} iterations, {ns:.0} ns/iteration",
        imp.id()
    )?;
    Ok(())
}

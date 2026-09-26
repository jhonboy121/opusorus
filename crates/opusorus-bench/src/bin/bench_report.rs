//! Prints a markdown table from the Criterion results of `cargo bench -p opusorus-bench`.
//!
//! Usage: `bench_report [CRITERION_DIR]` (default: `$CARGO_TARGET_DIR/criterion`, else the
//! workspace `target/criterion`). Run it through `scripts/bench_report.sh`, which runs the
//! benchmarks first.
//!
//! Columns: time per 20 ms frame (per call for kernels) of each implementation, realtime
//! factor (20 ms / time; how many times faster than realtime, single core) and the ratios
//! Rust / C-scalar and Rust / C-optimized (below 1.00 = Rust faster).

use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use opus_sys_optimized::{Micro, Variant};
use opusorus_bench::{Impl, report_rows};

type BoxResult<T> = Result<T, Box<dyn Error>>;

/// Point estimate (ns per iteration) Criterion reports: the linear-regression slope when it
/// was computed, otherwise the mean.
fn estimate(dir: &Path, group: &str, imp: Impl) -> BoxResult<Option<f64>> {
    let path = dir.join(group).join(imp.id()).join("new/estimates.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let pick = |k: &str| v.get(k).and_then(|e| e.get("point_estimate")?.as_f64());
    match pick("slope").or_else(|| pick("mean")) {
        Some(ns) => Ok(Some(ns)),
        None => Err(format!("{}: no slope/mean point estimate", path.display()).into()),
    }
}

fn default_dir() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(t) => PathBuf::from(t).join("criterion"),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion"),
    }
}

fn fmt_ns(ns: Option<f64>) -> String {
    match ns {
        Some(ns) if ns >= 1e6 => format!("{:.2} ms", ns / 1e6),
        Some(ns) if ns >= 1e4 => format!("{:.1} µs", ns / 1e3),
        Some(ns) if ns >= 1e3 => format!("{:.2} µs", ns / 1e3),
        Some(ns) => format!("{ns:.0} ns"),
        None => "–".to_owned(),
    }
}

fn fmt_rtf(ns: Option<f64>, per_frame: bool) -> String {
    match ns {
        Some(ns) if per_frame => format!("{:.0}×", 20e6 / ns),
        _ => "–".to_owned(),
    }
}

fn fmt_ratio(a: Option<f64>, b: Option<f64>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => {
            let r = a / b;
            // Flag rows where Rust is more than 10 % slower.
            if r > 1.10 {
                format!("**{r:.2}**")
            } else {
                format!("{r:.2}")
            }
        }
        _ => "–".to_owned(),
    }
}

fn cpu_model() -> BoxResult<String> {
    let info = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok("unknown".to_owned()),
        Err(e) => return Err(e.into()),
    };
    let field = info.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        let k = k.trim();
        (k == "model name" || k == "CPU part").then(|| format!("{k} {}", v.trim()))
    });
    Ok(match field {
        Some(f) => f,
        None => "unknown".to_owned(),
    })
}

fn main() -> BoxResult<()> {
    let dir = match std::env::args_os().nth(1) {
        Some(d) => PathBuf::from(d),
        None => default_dir(),
    };
    let out = io::stdout();
    let mut w = out.lock();

    writeln!(w, "## opusorus benchmarks\n")?;
    writeln!(
        w,
        "- host: {} {}, {}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        cpu_model()?
    )?;
    writeln!(
        w,
        "- c_scalar: opusorus-oracle (libopus 1.6.1, -O2 -ffp-contract=off, no intrinsics)"
    )?;
    writeln!(
        w,
        "- c_opt: libopus 1.6.1, default CMake Release (-O3, intrinsics, RTCD arch index {})",
        Micro::new(Variant::Optimized).arch()
    )?;
    writeln!(
        w,
        "- rust: opusorus, release profile (lto=fat, codegen-units=1), feature qext: {}",
        cfg!(feature = "qext")
    )?;
    writeln!(
        w,
        "- time = Criterion slope estimate per iteration (one 20 ms frame; kernels: one call); \
         RTF = 20 ms / time (single core); ratio < 1 means Rust is faster; **bold** = Rust \
         more than 10 % slower\n"
    )?;
    writeln!(
        w,
        "| benchmark | rust | c_scalar | c_opt | RTF rust | RTF c_scalar | RTF c_opt | \
         rust/c_scalar | rust/c_opt |"
    )?;
    writeln!(w, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")?;
    let mut missing = 0usize;
    for row in report_rows() {
        let r = estimate(&dir, &row.group, Impl::Rust)?;
        let s = estimate(&dir, &row.group, Impl::CScalar)?;
        let o = estimate(&dir, &row.group, Impl::COpt)?;
        if r.is_none() && s.is_none() && o.is_none() {
            missing += 1;
            continue;
        }
        writeln!(
            w,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.label,
            fmt_ns(r),
            fmt_ns(s),
            fmt_ns(o),
            fmt_rtf(r, row.per_frame),
            fmt_rtf(s, row.per_frame),
            fmt_rtf(o, row.per_frame),
            fmt_ratio(r, s),
            fmt_ratio(r, o),
        )?;
    }
    if missing > 0 {
        writeln!(
            w,
            "\n({missing} benchmark group(s) have no results in {}; run the benches first)",
            dir.display()
        )?;
    }
    Ok(())
}

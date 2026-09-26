#!/usr/bin/env bash
# Runs the opusorus benchmarks (Rust vs scalar C oracle vs optimized upstream C) in release
# mode and prints a markdown table (time per 20 ms frame, realtime factor, Rust/C ratios).
#
# Usage: scripts/bench_report.sh [--qext] [--quick] [--report-only] [-- <criterion args>]
#   --qext         also benchmark 96 kHz Opus HD (QEXT) encode/decode
#   --quick        shorter warm-up/measurement (noisier; for smoke runs)
#   --report-only  skip running the benches, only print the table from existing results
#   <criterion args> are passed to both bench binaries, e.g. a filter: -- decode_
#
# Needs cmake, ar, nm and objcopy (the optimized libopus is built by
# crates/opusorus-bench/opt-sys/build.rs). Results live in target/criterion.
set -euo pipefail
cd "$(dirname "$0")/.."

features=()
crit=()
run=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qext) features=(--features qext) ;;
        --quick) crit+=(--warm-up-time 0.5 --measurement-time 1 --sample-size 20) ;;
        --report-only) run=0 ;;
        --) shift; crit+=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ $run == 1 ]]; then
    cargo bench -p opusorus-bench "${features[@]}" --bench codec --bench micro -- "${crit[@]}" >&2
fi
cargo run -q --release -p opusorus-bench "${features[@]}" --bin bench_report

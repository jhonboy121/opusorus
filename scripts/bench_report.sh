#!/usr/bin/env bash
# Runs the opusorus benchmarks (Rust vs scalar C oracle vs optimized upstream C) in release
# mode and prints a markdown table (time per 20 ms frame, realtime factor, Rust/C ratios).
#
# Usage: scripts/bench_report.sh [--qext] [--dnn] [--fixed | --fixed-res24] [--runs N] [--quick]
#                                [--report-only] [-- <criterion args>]
#   --qext         also benchmark 96 kHz Opus HD (QEXT) encode/decode
#   --dnn          also benchmark the DNN features (deep PLC, LACE/NoLACE, DRED encoding;
#                  float only; needs scripts/fetch_dnn_models.sh)
#   --fixed        fixed-point builds (16-bit opus_res): opusorus `fixed-point` vs the fixed
#                  oracle vs upstream CMake with OPUS_FIXED_POINT=ON; native opus_encode /
#                  opus_decode PCM; Criterion groups prefixed `fixed_`
#   --fixed-res24  the same with 24-bit opus_res (`fixed-res24`, ENABLE_RES24; opus_encode24 /
#                  opus_decode24); groups prefixed `fixed24_`
#   --runs N       run the benches N times and report the fastest estimate of each cell
#                  (robust against other load on the machine; snapshots of every run are kept
#                  in target/criterion-runs/<features>/<run>)
#   --quick        shorter warm-up/measurement (noisier; for smoke runs)
#   --report-only  skip running the benches, only print the table from existing results
#   <criterion args> are passed to both bench binaries, e.g. a filter: -- decode_
#
# Needs cmake, ar, nm and objcopy (the optimized libopus is built by
# crates/opusorus-bench/opt-sys/build.rs). Results live in target/criterion; float and
# fixed-point results have distinct group names and do not overwrite each other.
set -euo pipefail
cd "$(dirname "$0")/.."

feats=()
crit=()
run=1
runs=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qext) feats+=(qext) ;;
        --dnn) feats+=(dnn) ;;
        --fixed) feats+=(fixed-point) ;;
        --fixed-res24) feats+=(fixed-res24) ;;
        --runs) runs=$2; shift ;;
        --quick) crit+=(--warm-up-time 0.5 --measurement-time 1 --sample-size 20) ;;
        --report-only) run=0 ;;
        --) shift; crit+=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

features=()
if [[ ${#feats[@]} -gt 0 ]]; then
    features=(--features "$(IFS=,; echo "${feats[*]}")")
fi

target=$(realpath -m "${CARGO_TARGET_DIR:-target}")
dirs=()
if [[ $runs -gt 1 ]]; then
    snap="$target/criterion-runs/$(IFS=_; echo "${feats[*]:-float}")"
    for ((i = 1; i <= runs; i++)); do
        if [[ $run == 1 ]]; then
            cargo bench -p opusorus-bench "${features[@]}" --bench codec --bench micro -- "${crit[@]}" >&2
            rm -rf "${snap:?}/$i"
            mkdir -p "$snap/$i"
            (cd "$target/criterion" && find . -path '*/new/estimates.json' -print0 |
                cpio -0pdm --quiet "$snap/$i") >&2
        fi
        dirs+=("$snap/$i")
    done
elif [[ $run == 1 ]]; then
    cargo bench -p opusorus-bench "${features[@]}" --bench codec --bench micro -- "${crit[@]}" >&2
fi
cargo run -q --release -p opusorus-bench "${features[@]}" --bin bench_report -- "${dirs[@]}"

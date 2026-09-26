#!/usr/bin/env bash
# Fuzzing campaign over every target (run from anywhere; needs the nightly toolchain and
# cargo-fuzz). Seeds the corpus first if it is missing.
#
#   fuzz/run_all.sh                            # default durations, 4 forked workers per target
#   SECS=60 LONG_SECS=60 FORK=2 fuzz/run_all.sh  # every target for 60 s with 2 workers
#   TARGETS="differential_decode decode" fuzz/run_all.sh
#   FEATURES=qext fuzz/run_all.sh              # QEXT build (target-qext/, corpus-qext/)
#   FEATURES=fixed-point fuzz/run_all.sh       # fixed-point build, 16-bit opus_res
#   FEATURES=fixed-res24,qext fuzz/run_all.sh  # fixed-point, 24-bit opus_res, QEXT
#   FEATURES=deep-plc,osce,dred fuzz/run_all.sh  # float + DNN: adds differential_dnn_decode
#   FEATURES=custom-modes fuzz/run_all.sh      # adds differential_custom
#
# `fixed-point` / `fixed-res24` replace the float codec on both sides (the port and the C
# oracle, docs/FIXED_POINT.md); they cannot be combined with the DNN features. Each feature set
# gets its own target directory and corpus (`target-<features>`, `corpus-<features>`); the
# corpus is seeded from the C encoder of the same configuration.
#
# Durations: LONG_SECS (default 600) for differential_decode, differential_encode and decode,
# SECS (default 240) for the others. Crashes are written to fuzz/artifacts/<target>/ (replay:
# `cargo +nightly fuzz run [--features ...] <target> artifacts/<target>/crash-...`) and stop
# that target's run (the campaign continues with the next target). Each run's output is kept in
# logs/<features>/<target>.log and a summary line (final coverage, features, corpus size, execs,
# crash count) is printed at the end.
set -uo pipefail
cd "$(dirname "$0")"

all="differential_decode decode differential_encode encode roundtrip_invariants differential_ms_encode repacketizer packet_parse_extensions multistream_decode projection_decode"
fork="${FORK:-4}"
features="${FEATURES:-}"
feat_args=()
target_dir=target
corpus=corpus
logs=logs/default
if [ -n "$features" ]; then
    feat_args=(--features "$features")
    target_dir="target-${features//,/-}"
    corpus="corpus-${features//,/-}"
    logs="logs/${features//,/-}"
fi
# Targets that need a feature (`required-features` in Cargo.toml).
case ",$features," in
    *,deep-plc,* | *,osce,* | *,dred,*) all="$all differential_dnn_decode" ;;
esac
case ",$features," in
    *,custom-modes,*) all="$all differential_custom" ;;
esac
targets="${TARGETS:-$all}"

if [ ! -d "$corpus" ]; then
    ./seed_corpus.sh "${feat_args[@]}" -- "$corpus" || exit 1
fi
mkdir -p "$logs"

# Build every target once, then run the binaries directly: sources edited while the campaign
# runs do not change what is being fuzzed.
cargo +nightly fuzz build "${feat_args[@]}" --target-dir "$target_dir" || exit 1
host=$(rustc +nightly -vV | sed -n 's/^host: //p')
bin_dir="$target_dir/$host/release"

summary=()
status=0
for t in $targets; do
    case "$t" in
        differential_decode | differential_encode | decode) secs="${LONG_SECS:-600}"; max_len=16384 ;;
        multistream_decode | projection_decode | differential_dnn_decode) secs="${SECS:-240}"; max_len=16384 ;;
        *) secs="${SECS:-240}"; max_len=4096 ;;
    esac
    echo "== $t (${secs}s, fork=$fork${features:+, features=$features})"
    mkdir -p "$corpus/$t" "artifacts/$t"
    log="$logs/$t.log"
    "$bin_dir/$t" -artifact_prefix="artifacts/$t/" -max_total_time="$secs" -fork="$fork" \
        -max_len="$max_len" -timeout=60 -rss_limit_mb=4096 -print_final_stats=1 \
        "$corpus/$t" > "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || status=1
    # Last fork-mode status line: "#<execs>: cov: <n> ft: <n> corp: <n> exec/s: <n> oom/timeout/crash: a/b/c ...".
    last=$(grep -E '^#[0-9]+: cov: ' "$log" | tail -n 1)
    summary+=("$(printf '%-24s rc=%d %s' "$t" "$rc" "$last")")
    tail -n 3 "$log"
done

echo
echo "== summary${features:+ ($features)}"
printf '%s\n' "${summary[@]}"
exit "$status"

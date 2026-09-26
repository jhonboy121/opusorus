#!/usr/bin/env bash
# Fuzzing campaign over every target (run from anywhere; needs the nightly toolchain and
# cargo-fuzz). Seeds the corpus first if it is missing.
#
#   fuzz/run_all.sh                 # default durations, 4 forked workers per target
#   SECS=60 FORK=2 fuzz/run_all.sh  # every target for 60 s with 2 workers
#   TARGETS="differential_decode decode" fuzz/run_all.sh
#   FEATURES=qext fuzz/run_all.sh   # QEXT build (separate target dir and corpus-qext/)
#
# Default durations: 10 min for differential_decode and decode, 4 min for the others.
# Crashes are written to fuzz/artifacts/<target>/ and stop that target's run.
set -euo pipefail
cd "$(dirname "$0")"

all="differential_decode decode differential_encode encode roundtrip_invariants differential_ms_encode repacketizer packet_parse_extensions multistream_decode projection_decode"
targets="${TARGETS:-$all}"
fork="${FORK:-4}"
features="${FEATURES:-}"
feat_args=()
target_dir=target
corpus=corpus
if [ -n "$features" ]; then
    feat_args=(--features "$features")
    target_dir="target-${features//,/-}"
    corpus="corpus-${features//,/-}"
fi

if [ ! -d "$corpus" ]; then
    ./seed_corpus.sh "${feat_args[@]}" -- "$corpus"
fi

for t in $targets; do
    case "$t" in
        differential_decode | decode) secs="${SECS:-600}"; max_len=16384 ;;
        multistream_decode | projection_decode) secs="${SECS:-240}"; max_len=16384 ;;
        *) secs="${SECS:-240}"; max_len=4096 ;;
    esac
    echo "== $t (${secs}s, fork=$fork${features:+, features=$features})"
    mkdir -p "$corpus/$t"
    cargo +nightly fuzz run "${feat_args[@]}" --target-dir "$target_dir" "$t" "$corpus/$t" -- \
        -max_total_time="$secs" -fork="$fork" -max_len="$max_len" -timeout=60 -rss_limit_mb=4096 \
        -print_final_stats=1
done

#!/usr/bin/env bash
# Generates the seed corpus (fuzz/corpus/<target>/seed-NNNN) from the C libopus encoder.
# Deterministic; runs on the stable toolchain. Extra arguments go to cargo (e.g.
# `--features qext` adds QEXT streams; the output directory can be given after `--`).
set -euo pipefail
cd "$(dirname "$0")"
cargo run --release --example seed_corpus "$@"

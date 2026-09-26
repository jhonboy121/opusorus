# opusorus task runner. `just --list` for recipes.

targets := "wasm32-unknown-unknown wasm32-wasip1 aarch64-linux-android armv7-linux-androideabi x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios x86_64-unknown-linux-gnu"

# `--all-features` is not a valid configuration: `fixed-point` replaces the float codec (not
# additive) and upstream refuses fixed-point together with the DNN features. These lists are the
# two "everything on" configurations (docs/FIXED_POINT.md).
float_all := "opusorus/internals,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-conformance/deep-plc,opusorus-conformance/dred,opusorus-conformance/osce,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/deep-plc,opusorus-capi/dred,opusorus-capi/osce,opusorus-capi/internal-api,opusorus-bench/qext,opusorus-tools/qext,opusorus-tools/osce"
fixed_all := "opusorus/internals,opusorus-conformance/fixed-res24,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-capi/fixed-res24,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/internal-api,opusorus-bench/fixed-res24,opusorus-bench/qext,opusorus-tools/qext"

default:
    @just --list

# Type-check the whole workspace.
check:
    cargo check --workspace --all-targets --features {{float_all}}
    cargo check --workspace --all-targets --features {{fixed_all}}

# Format check.
fmt:
    cargo fmt --all -- --check

# Clippy, warnings are errors.
clippy:
    cargo clippy --workspace --all-targets --features {{float_all}} -- -D warnings
    cargo clippy --workspace --all-targets --features {{fixed_all}} -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools --all-targets --features opusorus-conformance/fixed-point -- -D warnings
    cargo clippy -p opusorus --no-default-features -- -D warnings
    cargo clippy -p opusorus --no-default-features --features fixed-point -- -D warnings

# All tests (unit + differential vs C oracle + vectors): default, every float feature, and the
# fixed-point builds (16- and 24-bit resolution; only the converted modules are tested).
test:
    cargo test --workspace
    cargo test --workspace --features {{float_all}}
    cargo test -p opusorus -p opusorus-conformance --features opusorus-conformance/fixed-point
    cargo test -p opusorus -p opusorus-conformance --features opusorus-conformance/fixed-res24,opusorus-conformance/qext

# Build the library for every supported platform (+ no_std bare-metal).
cross:
    #!/usr/bin/env bash
    set -euo pipefail
    for t in {{targets}}; do
        echo "== $t"; cargo build -q -p opusorus --release --target "$t"
    done
    echo "== thumbv7em-none-eabihf (no_std)"; cargo build -q -p opusorus --release --no-default-features --target thumbv7em-none-eabihf
    echo "== wasm32-unknown-unknown (no_std)"; cargo build -q -p opusorus --release --no-default-features --target wasm32-unknown-unknown
    # Fixed-point build (reduced crate while being ported): a 32-bit target (OPUS_FAST_INT64 = 0
    # forms) and no_std.
    echo "== armv7-linux-androideabi (fixed-res24)"; cargo build -q -p opusorus --release --features fixed-res24 --target armv7-linux-androideabi
    echo "== thumbv7em-none-eabihf (no_std, fixed-point)"; cargo build -q -p opusorus --release --no-default-features --features fixed-point --target thumbv7em-none-eabihf

# Build the C-ABI static library for every mobile/desktop target. The C glue for variadic ctls is
# compiled with host clang in freestanding mode; final shared-library linking needs the platform
# SDK (NDK/Xcode), so only the static library is produced here.
cross-capi:
    #!/usr/bin/env bash
    set -euo pipefail
    for t in x86_64-unknown-linux-gnu aarch64-linux-android armv7-linux-androideabi x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios; do
        tu=${t//-/_}
        echo "== capi $t"
        env SDKROOT=/tmp "CC_$tu=clang" "CFLAGS_$tu=-ffreestanding" \
            cargo rustc -q -p opusorus-capi --lib --release --target "$t" --crate-type staticlib
    done

# Run the library's own tests under wasmtime (wasm32-wasip1).
test-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=." cargo test -p opusorus --target wasm32-wasip1

# Official RFC 6716/8251 decoder test vectors.
vectors:
    ./scripts/fetch_vectors.sh
    cargo test -p opusorus-conformance --release --test vectors -- --nocapture

# Benchmarks (Rust vs C oracle).
bench:
    cargo bench -p opusorus-bench

# Shared-library size comparison (Rust cdylib vs libopus.so).
size:
    ./scripts/size_report.sh

# Fuzz one target for a while: `just fuzz decode 60`.
fuzz target secs="60":
    cd fuzz && cargo +nightly fuzz run {{target}} -- -max_total_time={{secs}}

# Seed fuzz corpora from the C encoder (default + qext).
fuzz-seed:
    cd fuzz && ./seed_corpus.sh && ./seed_corpus.sh --features qext -- corpus-qext

# Full fuzz campaign (SECS=, FORK=, TARGETS=, FEATURES=qext knobs).
fuzz-all:
    fuzz/run_all.sh

# Everything CI runs.
ci: fmt clippy test cross cross-capi test-wasm

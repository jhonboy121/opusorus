# opusorus task runner. `just --list` for recipes.

targets := "wasm32-unknown-unknown wasm32-wasip1 aarch64-linux-android armv7-linux-androideabi x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios x86_64-unknown-linux-gnu"

default:
    @just --list

# Type-check the whole workspace.
check:
    cargo check --workspace --all-targets --all-features

# Format check.
fmt:
    cargo fmt --all -- --check

# Clippy, warnings are errors.
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo clippy -p opusorus --no-default-features -- -D warnings

# All tests (unit + differential vs C oracle + vectors), default and qext configs.
test:
    cargo test --workspace
    cargo test --workspace --all-features

# Build the library for every supported platform (+ no_std bare-metal).
cross:
    #!/usr/bin/env bash
    set -euo pipefail
    for t in {{targets}}; do
        echo "== $t"; cargo build -q -p opusorus --release --target "$t"
    done
    echo "== thumbv7em-none-eabihf (no_std)"; cargo build -q -p opusorus --release --no-default-features --target thumbv7em-none-eabihf
    echo "== wasm32-unknown-unknown (no_std)"; cargo build -q -p opusorus --release --no-default-features --target wasm32-unknown-unknown

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

# Everything CI runs.
ci: fmt clippy test cross test-wasm

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

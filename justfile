# opusorus task runner. `just --list` for recipes.

targets := "wasm32-unknown-unknown wasm32-wasip1 aarch64-linux-android armv7-linux-androideabi x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios x86_64-unknown-linux-gnu"

# Weight blob embedded by the `dnn-weights-embedded` feature; made by the `dnn-blob` recipe.
# Override with an absolute path in the environment.
export OPUSORUS_DNN_BLOB := env_var_or_default("OPUSORUS_DNN_BLOB", justfile_directory() + "/target/dnn/weights_blob.bin")
# The same with the float copies of the int8 layers (`dnn-debug-float`, `dnn-blob-debug-float`).
export OPUSORUS_DNN_DEBUG_FLOAT_BLOB := env_var_or_default("OPUSORUS_DNN_DEBUG_FLOAT_BLOB", justfile_directory() + "/target/dnn/weights_blob_debug_float.bin")

# `--all-features` is not a valid configuration: `fixed-point` replaces the float codec (not
# additive) and upstream refuses fixed-point together with the DNN features. These lists are the
# two "everything on" configurations (docs/FIXED_POINT.md). float_all excludes
# `dnn-weights-embedded` (covered by `test-dnn`) and `opusorus-capi/osce` (OSCE makes
# `Decoder::get_size` exceed the 256 KiB bound upstream's test_opus_api checks).
float_all := "opusorus/internals,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-conformance/deep-plc,opusorus-conformance/dred,opusorus-conformance/osce,opusorus-tools/dred,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/dred,opusorus-capi/osce,opusorus-capi/internal-api,opusorus-bench/qext"
fixed_all := "opusorus/internals,opusorus-conformance/fixed-res24,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-capi/fixed-res24,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/internal-api,opusorus-bench/fixed-res24,opusorus-bench/qext,opusorus-tools/qext"
# The upstream build options (docs/FEATURES.md O7-O10), each forwarded to the oracle.
options_all := "opusorus-conformance/float-approx,opusorus-conformance/assertions,opusorus-conformance/fuzzing,opusorus-conformance/disable-rfc8251"

# The fixed-point build without the float API (`disable-float-api`, libopus DISABLE_FLOAT_API)
# and with the checking fixed-point macros (`fixed-point-debug`, libopus FIXED_DEBUG), with
# everything else on (see `test-options`).
nofloat_all := "opusorus/internals,opusorus-conformance/fixed-res24,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-conformance/disable-float-api,opusorus-capi/fixed-res24,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/disable-float-api,opusorus-capi/internal-api,opusorus-tools/qext"
fixeddebug_all := "opusorus/internals,opusorus-conformance/fixed-point-debug,opusorus-conformance/fixed-res24,opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-capi/fixed-point-debug,opusorus-capi/fixed-res24,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-capi/internal-api,opusorus-tools/qext"

default:
    @just --list

# Type-check the whole workspace.
check: dnn-blob
    cargo check --workspace --all-targets --features {{float_all}}
    cargo check --workspace --all-targets --features {{fixed_all}}

# Rustdoc with warnings as errors, float/DNN/fixed configurations.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps --features qext,custom-modes,deep-plc,dred,osce
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps --features fixed-res24,qext,custom-modes
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps --features osce-training-data,dnn-debug-float,lossgen
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps --features fixed-res24,qext,custom-modes,disable-float-api,fixed-point-debug
    RUSTDOCFLAGS="-D warnings" cargo doc -p opusorus --no-deps --features float-approx,assertions,fuzzing,disable-rfc8251

# Format check.
fmt:
    cargo fmt --all -- --check

# Clippy, warnings are errors.
clippy: dnn-blob
    cargo clippy --workspace --all-targets --features {{float_all}} -- -D warnings
    cargo clippy --workspace --all-targets --features {{fixed_all}} -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools --all-targets --features opusorus-conformance/fixed-point -- -D warnings
    cargo clippy -p opusorus --no-default-features -- -D warnings
    cargo clippy -p opusorus --no-default-features --features fixed-point -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools --all-targets --features opusorus-conformance/qext,opusorus-conformance/dred,opusorus-conformance/osce-training-data,opusorus-conformance/dnn-debug-float,opusorus-conformance/lossgen -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools --all-targets --features opusorus-conformance/fixed-res24,opusorus-conformance/lossgen -- -D warnings
    cargo clippy -p opusorus --no-default-features --features lossgen -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools -p opusorus-capi --all-targets --features {{nofloat_all}} -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools -p opusorus-capi --all-targets --features {{fixeddebug_all}} -- -D warnings
    cargo clippy -p opusorus --no-default-features --features fixed-point-debug,disable-float-api -- -D warnings
    # Upstream build options (not in float_all/fixed_all: `fuzzing` changes every encoder output).
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools -p opusorus-capi -p opusorus-bench --all-targets --features {{options_all}},opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-conformance/deep-plc,opusorus-conformance/dred,opusorus-conformance/osce,opusorus-capi/float-approx,opusorus-capi/assertions,opusorus-capi/fuzzing,opusorus-capi/disable-rfc8251,opusorus-bench/float-approx -- -D warnings
    cargo clippy -p opusorus -p opusorus-oracle -p opusorus-conformance -p opusorus-tools --all-targets --features {{options_all}},opusorus-conformance/fixed-res24,opusorus-conformance/qext -- -D warnings
    cargo clippy -p opusorus --no-default-features --features float-approx,assertions,fuzzing,disable-rfc8251 -- -D warnings
    # `fast` (non-bit-exact kernels, PLAN D-032), with the DNN benchmarks.
    cargo clippy -p opusorus -p opusorus-bench -p opusorus-tools --all-targets --features opusorus/internals,opusorus/fast,opusorus-bench/dnn,opusorus-bench/qext,opusorus-tools/dred -- -D warnings
    cargo clippy -p opusorus --no-default-features --features fast,deep-plc -- -D warnings
    cargo clippy -p opusorus-bench --all-targets --features dnn -- -D warnings

# All tests (unit + differential vs C oracle + vectors): default, every float feature (DNN weights
# loaded at runtime; `test-dnn` covers compiled-in weights), and the fixed-point builds: the full
# differential suite in 16-bit and 24-bit + QEXT resolution, and the public-API tests (the Rust
# port of the libopus tests, examples), the libopus unit tests, the public-API overflow sweep
# and the opus_demo/conformance vectors in the other fixed configurations.
test: dnn-blob
    #!/usr/bin/env bash
    set -euxo pipefail
    cargo test --workspace
    cargo test --workspace --features {{float_all}}
    cargo test -p opusorus -p opusorus-tools -p opusorus-conformance --features opusorus-conformance/fixed-point
    cargo test -p opusorus -p opusorus-tools -p opusorus-conformance --features opusorus-conformance/fixed-res24,opusorus-conformance/qext
    for f in fixed-res24 fixed-point,qext fixed-point,custom-modes fixed-res24,qext,custom-modes; do
        cargo test -p opusorus --features "$f"
        cargo test -p opusorus-conformance --features "$f" --test vectors --test libopus_unit --test api_overflow
    done
    # C ABI harness (upstream C test suite + opus_demo vectors) and bench parity, fixed-point.
    cargo test -p opusorus-capi --features fixed-point,custom-modes
    cargo test -p opusorus-capi --features fixed-res24,qext,custom-modes
    cargo test -p opusorus-bench --features fixed-res24,qext --test parity

# The optional libopus build options (docs/FEATURES.md), each against the oracle built with the
# same define(s): `disable-float-api` (DISABLE_FLOAT_API; every fixed differential suite,
# the public-API tests and the upstream C suite through the C ABI, 16- and 24-bit + QEXT +
# custom modes) and `fixed-point-debug` (FIXED_DEBUG; the checking macros one by one, then the
# fixed differential suites, 16-bit and 24-bit + QEXT + custom modes).
test-options-fixed:
    #!/usr/bin/env bash
    set -euxo pipefail
    cargo test -p opusorus -p opusorus-tools -p opusorus-conformance --features opusorus-conformance/fixed-point,opusorus-conformance/disable-float-api
    cargo test -p opusorus -p opusorus-conformance --features fixed-res24,qext,custom-modes,disable-float-api
    cargo test -p opusorus-capi --features fixed-point,custom-modes,disable-float-api
    for f in fixed-point-debug fixed-point-debug,fixed-res24,qext,custom-modes fixed-point-debug,disable-float-api; do
        cargo test -p opusorus-conformance --features "$f" --test fixed_debug --test fixed_foundation --test celt_fft --test celt_modes --test celt_pitch_lpc_fixed --test celt_bands --test celt_decoder --test celt_encoder --test silk_common --test silk_decoder --test silk_encoder_common --test silk_encoder_fix --test opus_decoder --test opus_encoder --test api_overflow --test vectors
    done
    cargo test -p opusorus --features fixed-point-debug --lib

# Full-length libopus runs, exhaustive sweeps and timing tests, default and QEXT + custom-modes.
# All tests including the ignored long ones (OPUS_TEST_FULL=1, --include-ignored).
test-full:
    OPUS_TEST_FULL=1 cargo test --workspace -- --include-ignored
    OPUS_TEST_FULL=1 cargo test --workspace --features opusorus-conformance/qext,opusorus-conformance/custom-modes,opusorus-capi/qext,opusorus-capi/custom-modes,opusorus-bench/qext -- --include-ignored

# All conformance tests vs the oracle built with deep PLC + DRED + OSCE (+ QEXT), including the
# C opus_demo comparison with DRED/deep PLC/OSCE cases; then compiled-in weights
# (`dnn-weights-embedded`): library unit tests, the opus_demo comparison with the weights
# embedded in the Rust tool, and upstream's C tests (incl. test_opus_dred) against the C ABI.
# DNN tests (deep PLC, DRED, OSCE) vs the oracle, runtime-loaded and compiled-in weights.
test-dnn: dnn-blob
    cargo test -p opusorus-conformance --features qext,deep-plc,dred,osce,opusorus-tools/dred
    cargo test -p opusorus --features qext,dred,osce,dnn-weights-embedded --lib
    cargo test -p opusorus-conformance --features qext,dred,osce,opusorus-tools/dred,opusorus-tools/dnn-weights-embedded --test vectors
    cargo test -p opusorus-capi --features qext,dred

# The DNN build options of phase G vs the oracle built with the same defines: `lossgen`
# (loss model + `opus_demo -sim_loss` + `lossgen_demo`, float and fixed-point),
# `dnn-debug-float` (every DNN suite and the opus_demo comparison without DISABLE_DEBUG_FLOAT,
# runtime-loaded and compiled-in debug-float weights) and `osce-training-data` (its own test
# only: every SILK encode/decode of the other suites would write training files).
# Phase-G DNN options (lossgen, dnn-debug-float, osce-training-data) vs the matching oracle.
test-dnn-extras: dnn-blob dnn-blob-debug-float
    #!/usr/bin/env bash
    set -euxo pipefail
    cargo test -p opusorus --features lossgen --lib
    cargo test -p opusorus-tools --features lossgen
    cargo test -p opusorus-conformance --features lossgen --test lossgen --test vectors
    cargo test -p opusorus-conformance --features fixed-res24,lossgen --test lossgen --test vectors
    OPUSORUS_DNN_BLOB="$OPUSORUS_DNN_DEBUG_FLOAT_BLOB" cargo test -p opusorus-conformance --features qext,deep-plc,dred,osce,dnn-debug-float,lossgen,opusorus-tools/dred
    OPUSORUS_DNN_BLOB="$OPUSORUS_DNN_DEBUG_FLOAT_BLOB" cargo test -p opusorus --features qext,dred,osce,dnn-weights-embedded,dnn-debug-float --lib
    OPUSORUS_DNN_BLOB="$OPUSORUS_DNN_DEBUG_FLOAT_BLOB" cargo test -p opusorus-conformance --features qext,dred,osce,dnn-debug-float,opusorus-tools/dred,opusorus-tools/dnn-weights-embedded --test vectors
    cargo test -p opusorus-conformance --features osce-training-data --test osce_training_data
    cargo test -p opusorus-conformance --features osce-training-data,dred,qext --test osce_training_data
# `fast` (non-bit-exact kernels, PLAN D-032): the fast kernels against the reference kernels
# within error bounds (library unit tests), opusorus `fast` against the C oracle and the optimized
# C build (decoding incl. PLC/deep PLC/OSCE, encoding quality), and the RFC 8251 / Opus HD vectors
# through opus_compare / qext_compare. The bit-exact suites are not run with `fast`.
test-fast: dnn-blob
    #!/usr/bin/env bash
    set -euxo pipefail
    ./scripts/fetch_vectors.sh
    cargo test -p opusorus --features fast,qext,dred,osce --lib
    cargo test -p opusorus-bench --features fast,dnn,qext --test fast -- --nocapture
    cargo test -p opusorus-conformance --features opusorus/fast,qext --test vectors -- rfc8251_vectors opushd_vectors
    cargo test -p opusorus-conformance --features opusorus/fast,deep-plc,dred,osce,opusorus-tools/dred --test vectors -- rfc8251_vectors

# Upstream build options vs the oracle built with the same defines: float-approx (FLOAT_APPROX)
# and disable-rfc8251 (DISABLE_UPDATE_DRAFT, RFC 6716 vectors) over the whole differential
# suite, float/fixed-point/QEXT/custom modes (float-approx also with the DNN features);
# assertions (ENABLE_ASSERTIONS) likewise, plus the hard-check tests in release (no debug
# assertions); fuzzing (FUZZING) over its seeded suite and the decoder / opus_demo comparisons
# (the other encoder suites would race on the process-wide rand() state); the library tests,
# the bench parity (float-approx) and the C ABI harness.
test-options-float: dnn-blob
    #!/usr/bin/env bash
    set -euxo pipefail
    ./scripts/fetch_vectors.sh
    c=opusorus-conformance
    for f in float-approx float-approx,qext,custom-modes disable-rfc8251 disable-rfc8251,qext,custom-modes fixed-point,disable-rfc8251 fixed-res24,qext,disable-rfc8251 assertions assertions,qext,custom-modes fixed-point,custom-modes,assertions fixed-res24,qext,assertions; do
        cargo test -p $c --features "$f"
    done
    cargo test -p $c --features float-approx,qext,deep-plc,dred,osce,opusorus-tools/dred
    cargo test -p $c --features assertions,qext,deep-plc,dred,osce,opusorus-tools/dred
    cargo test --release -p $c --features assertions --test assertions
    cargo test --release -p $c --features fixed-point,assertions --test assertions
    for f in fuzzing fuzzing,qext fixed-point,fuzzing,custom-modes fixed-res24,qext,fuzzing; do
        cargo test -p $c --features "$f" --test fuzzing --test vectors --test opus_decoder --test celt_decoder
    done
    cargo test -p opusorus --features float-approx,assertions,fuzzing,disable-rfc8251,qext,custom-modes
    cargo test -p opusorus --features fixed-res24,assertions,fuzzing,disable-rfc8251
    cargo test -p opusorus --release --features assertions
    cargo test -p opusorus-tools --features opusorus/fuzzing
    cargo test -p opusorus-bench --features float-approx --test parity
    cargo test -p opusorus-capi --features float-approx,assertions,fuzzing
    cargo test -p opusorus-capi --features disable-rfc8251

# Every optional upstream build option (phase G) against the oracle built with the same defines.
test-options: test-options-float test-options-fixed test-dnn-extras

# Extract the DNN model data (vendor/libopus/dnn/*_data.c, gitignored) if missing.
dnn-models:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f vendor/libopus/dnn/fargan_data.c ] || ./scripts/fetch_dnn_models.sh

# Generate the DNN weight blob at $OPUSORUS_DNN_BLOB if missing.
dnn-blob: dnn-models
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f "$OPUSORUS_DNN_BLOB" ] || ./scripts/gen_dnn_blob.sh "$OPUSORUS_DNN_BLOB"

# Generate the debug-float DNN weight blob at $OPUSORUS_DNN_DEBUG_FLOAT_BLOB if missing.
dnn-blob-debug-float: dnn-models
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f "$OPUSORUS_DNN_DEBUG_FLOAT_BLOB" ] || ./scripts/gen_dnn_blob.sh --debug-float "$OPUSORUS_DNN_DEBUG_FLOAT_BLOB"

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

# Run the library's own tests under wasmtime (wasm32-wasip1), float and fixed-point (a 32-bit
# target: the OPUS_FAST_INT64 = 0 multiply forms).
test-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=." cargo test -p opusorus --target wasm32-wasip1
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=." cargo test -p opusorus --target wasm32-wasip1 --features fixed-point
    # SIMD128 build: the fearless_simd FFT/MDCT/DNN kernels must match the scalar code bit for bit.
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=." RUSTFLAGS="-Ctarget-feature=+simd128" cargo test -p opusorus --target wasm32-wasip1 --features qext,deep-plc --lib simd

# RFC 6716/8251 through the Rust opus_demo/opus_compare, the Opus HD (QEXT) vectors with
# qext_compare, the C opus_demo comparison (against the matching float or fixed-point C
# opus_demo), for the float and the fixed-point (16-bit, 24-bit, with QEXT) decoders, and the
# RFC 8251 vectors through upstream's C opus_demo linked to the C ABI library; the RFC 6716
# vectors with the pre-RFC 8251 decoder (`disable-rfc8251`).
# Official decoder test vectors (fetched if missing) and the opus_demo comparisons.
vectors:
    ./scripts/fetch_vectors.sh
    cargo test -p opusorus-conformance --test vectors -- --nocapture
    cargo test -p opusorus-conformance --features qext --test vectors -- --nocapture
    cargo test -p opusorus-conformance --features fixed-point --test vectors -- --nocapture
    cargo test -p opusorus-conformance --features fixed-res24,qext --test vectors -- --nocapture
    cargo test -p opusorus-conformance --features fixed-point,qext --test vectors -- --nocapture
    cargo test -p opusorus-conformance --features disable-rfc8251 --test vectors -- --nocapture
    cargo test -p opusorus-capi --test c_suite rfc8251_vectors -- --nocapture

# Benchmarks (Rust vs C oracle).
bench:
    cargo bench -p opusorus-bench

# Markdown benchmark report: float (default), `--fixed`, `--fixed-res24`, `--qext`, `--runs N`.
bench-report *args:
    scripts/bench_report.sh {{args}}

# Shared-library size comparison (Rust cdylib vs libopus.so).
size:
    ./scripts/size_report.sh

# Fuzz one target for a while: `just fuzz decode 60`.
fuzz target secs="60":
    mkdir -p fuzz/corpus/{{target}}
    cd fuzz && cargo +nightly fuzz run {{target}} corpus/{{target}} -- -max_total_time={{secs}}

# Seed fuzz corpora from the C encoder (default + qext).
fuzz-seed:
    cd fuzz && ./seed_corpus.sh && ./seed_corpus.sh --features qext -- corpus-qext

# Full fuzz campaign (SECS=, FORK=, TARGETS=, FEATURES=qext knobs).
fuzz-all:
    fuzz/run_all.sh
    FEATURES=fixed-point fuzz/run_all.sh
    FEATURES=fixed-res24,qext fuzz/run_all.sh

# Everything CI runs.
ci: fmt clippy doc test test-options cross cross-capi test-wasm

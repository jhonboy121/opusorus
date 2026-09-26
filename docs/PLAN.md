# Plan & decision log

## Goal
A complete, safe-Rust port of libopus v1.6.1 with every feature, verified bit-exact against the C
library, with conformance vectors, fuzzing, benchmarks and size comparisons.

## Phases

| Phase | Content | Exit criteria |
|---|---|---|
| 0 Foundation | workspace, oracle crate, conventions, range coder, math, SILK macros, docs | foundation tests bit-exact, all targets build |
| A Leaf DSP | kiss_fft/mdct, modes/tables/cwrs/laplace/rate, pitch/lpc, SILK tables/sigproc/NLSF, resampler, packet/extensions/repacketizer/mapping_matrix/mlp | per-unit differential tests bit-exact |
| B Mid-level | CELT bands/vq/quant/celt.c, SILK decoder, SILK encoder common (NSQ, VAD, ...), analysis | same |
| C Codecs | CELT decoder, CELT encoder, SILK FLP encoder + enc_API | same |
| D Opus API | Opus decoder/encoder, multistream, projection | encode bytes + decode samples bit-exact across configs |
| E Hardening | RFC vectors + opus_compare, ported libopus test suite, C ABI crate (+ run C tests against it), fuzzing, benches, size report, cross-platform CI | all green |
| F Optional features | QEXT (done alongside A–D), custom modes, fixed-point build (layers FX0–FX5, docs/FIXED_POINT.md), DNN (deep PLC, DRED, OSCE/BWE) | differential tests vs oracle built with the same options |

## Decisions

| ID | Date | Decision | Rationale |
|---|---|---|---|
| D-001 | 2026-09-25 | Oracle = vendored libopus **v1.6.1** (latest tag), compiled by `cc` in `opusorus-oracle`. | Pinned, reproducible, no system dependency; `cc` lets us pass exact flags. |
| D-002 | 2026-09-25 | Oracle flags: float, `OPUS_BUILD VAR_ARRAYS HAVE_LRINT(F) ENABLE_HARDENING`, **no intrinsics/RTCD**, `-ffp-contract=off`. | Scalar C with no FMA contraction has a deterministic op order that Rust can reproduce → bit-exact comparisons instead of tolerance-based ones. |
| D-003 | 2026-09-25 | Faithful function-by-function translation, C names kept. | Bit-exactness requires identical operation order; traceability to C for review. |
| D-004 | 2026-09-25 | Workspace: `opusorus` (lib, no_std+alloc, forbid unsafe), `opusorus-oracle` (test-only FFI, unsafe allowed), `opusorus-conformance` (differential tests), later `opusorus-capi`, `opusorus-tools`, `opusorus-bench`, `fuzz/`. | Keeps unsafe out of the product; C ABI shim needs raw pointers so it is isolated in its own crate with `// SAFETY:` comments. |
| D-005 | 2026-09-25 | `no_std` + `alloc`; `std` feature (default) uses platform libm via `crate::math`; without it the `libm` crate. | Bit-exactness with the oracle needs the same libm (glibc); `no_std` still supported. |
| D-006 | 2026-09-25 | Float build semantics ported first; CELT arithmetic written through `arch` macro-functions (`mult16_32_q15` etc.). | Mirrors C; leaves a seam for the fixed-point build (phase F). |
| D-007 | 2026-09-25 | Range coder split into `EcEnc` (&mut [u8]) / `EcDec` (&[u8]) + `EcCoder` enum for shared paths. | Rust borrow rules; decoding never needs a mutable packet copy. |
| D-008 | 2026-09-25 | Parallel porting by units with strict file ownership, each unit in its own git worktree, merged between layers. | Lets independent units progress concurrently without breaking each other's builds. |
| D-009 | 2026-09-25 | QEXT, custom modes: ported in the same pass as the code they live in, behind cargo features `qext`/`custom-modes`, tested against an oracle built with the same define. | The `#ifdef` blocks are interleaved with the base code; porting them later would mean re-reading everything. |
| D-010 | 2026-09-25 | DNN features (deep PLC, DRED, OSCE, BWE) ported last. | Off by default upstream and need downloaded model weights (~MBs); core codec first. |
| D-011 | 2026-09-25 | Fixed-point build ported as a later phase with its own oracle config (`FIXED_POINT`), not in the first pass. | Doubles the verification surface of CELT; float is the default build and what users need first. The oracle `fixed-point` feature was removed until then because `--all-features` would build a fixed oracle against float tests. |
| D-012 | 2026-09-25 | SILK `OPUS_FAST_INT64` 64-bit macro forms used on all targets. | Proven exactly equivalent to the 32-bit fallbacks (unit test), simpler and faster. |
| D-013 | 2026-09-25 | Clippy: `clippy::all` + `missing_const_for_fn`, `unwrap_used`, `expect_used`, `allow_attributes_without_reason` etc. at warn, CI with `-D warnings`. `excessive_precision`/`approx_constant` allowed crate-wide (verbatim C literals). | User requirements; literal-exactness needed for bit-exact output. |
| D-014 | 2026-09-25 | Commits: plain messages, no co-author/session trailers. | User instruction. |
| D-015 | 2026-09-26 | DNN weights: fetched by `scripts/fetch_dnn_models.sh` (upstream tarball + checksum), not committed. The Rust DNN features will embed a binary weight blob (produced with upstream `write_lpcnet_weights` format) via `include_bytes!` and parse it with the ported `parse_lpcnet_weights`; users can also supply a blob at runtime (`OPUS_SET_DNN_BLOB`). | 87 MB of generated C arrays would be unworkable as Rust source (compile time/size); the blob path is upstream-supported and bit-identical. |
| D-016 | 2026-09-26 | Opus HD (QEXT) conformance vectors from `media.xiph.org/opus/ietf/opushd_testvectors.tar.gz`, RFC 8251 vectors from opus-codec.org; both fetched by `scripts/fetch_vectors.sh`, gitignored. | Large (765 MB), public, versioned upstream. |
| D-017 | 2026-09-26 | C ABI: opaque handles are registry-backed (memcpy-safe) Rust objects; variadic `*_ctl` entry points are C glue (`csrc/ctl.c`) jumping via naked-function trampolines into non-variadic Rust exports. | Stable Rust cannot define C-variadic functions; glue keeps the exact `opus.h` ABI. |
| D-018 | 2026-09-26 | DNN weights are runtime blobs (like upstream `USE_WEIGHTS_FILE`); tests serialize the oracle's compiled-in weights to a blob. `osce` implies `deep-plc` (upstream coupling). | See D-015; keeps the crate small and compile times sane. |
| D-019 | 2026-09-26 | Cross builds of the C ABI crate use host clang (freestanding) for the ctl glue; only static libraries are verified without platform SDKs. | NDK/Xcode are not available on the build host; the Rust core is pure Rust and needs no C compiler. |
| D-020 | 2026-09-26 | Fixed point: cargo features `fixed-point` and `fixed-res24` (implies `fixed-point`) on `opusorus`, the oracle and the test/tool crates; **not additive** (replaces float, like upstream), incompatible with the DNN features (upstream configure refuses it: `compile_error!`). Float API kept on. `--all-features` is no longer a valid configuration: `just` uses explicit `float_all` / `fixed_all` feature lists. | Mirrors libopus; a runtime switch would double the code and could not be bit-exact with either build. |
| D-021 | 2026-09-26 | Fixed-point port is done incrementally: unconverted modules are `#[cfg(not(feature = "fixed-point"))]`-gated (reduced crate), shared code keeps one Rust name per C macro/function with per-build definitions (`arch/{float,fixed}.rs`, `mathops/{float,fixed}.rs`); shims declare `// oracle-build: float|fixed|any`. See docs/FIXED_POINT.md. | Lets FX units proceed in parallel without breaking either build. |
| D-022 | 2026-09-26 | Fixed macros mirror the *release* `fixed_generic.h` expansion exactly (arguments `impl Into<i32>` with C's casts, C result types, plain arithmetic where C overflow is UB), not `fixed_debug.h`'s narrower types. | Bit-exactness first; explicit `extract16` marks every implicit C narrowing. |
| D-023 | 2026-09-26 | `OPUS_FAST_INT64` follows the C per-target selection (64-bit forms on x86_64/64-bit/MIPS, 32-bit partial-product forms elsewhere); both forms are tested against C on the host. | `MULT32_32_Q31/P31/Q32` differ between the forms; matching libopus on each target is the contract. |
| D-024 | 2026-09-26 | opusorus is consumed from the Rust ecosystem only. `opusorus-capi` stays as a verification harness (runs upstream C tests / opus_demo against the port) but is not a shipped product: no no_std rewrite, no size work on the cdylib. Size/perf work targets the Rust crate. | User direction. |

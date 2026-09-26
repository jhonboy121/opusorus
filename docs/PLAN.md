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
| F Optional features | QEXT (done alongside A–D), custom modes, fixed-point build, DNN (deep PLC, DRED, OSCE/BWE) | differential tests vs oracle built with the same options |

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

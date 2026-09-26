# Status

_Last updated: 2026-09-26 (after layer E)_

## Summary
The complete float codec is ported and verified bit-exact against libopus 1.6.1: CELT, SILK, hybrid,
Opus encoder/decoder, multistream, projection (ambisonics), repacketizer, extensions, analysis,
QEXT (Opus HD, 96 kHz) and custom modes. DNN modules (deep PLC/FARGAN, DRED, OSCE/BWE) are ported
and bit-exact standalone; wiring them into the codec is in progress. The RFC 8251 and Opus HD
conformance vectors decode bit-identically to C and pass opus_compare/qext_compare.

Hardening phase (C ABI + upstream C test suite, Rust port of libopus tests, fuzzing, benchmarks,
opus_demo, DNN integration) in progress. Remaining after that: fixed-point build, performance pass.

| Area | State |
|---|---|
| Foundation (range coder, mathops, SILK macros, constants) | ✅ bit-exact vs libopus 1.6.1 |
| CELT (encoder, decoder, PLC, custom modes, QEXT) | ✅ bit-exact |
| SILK (FLP encoder, decoder, PLC, CNG, LBRR, resampler) | ✅ bit-exact |
| Opus layer (encoder, decoder, multistream, surround, projection, repacketizer, extensions, analysis) | ✅ bit-exact |
| QEXT (Opus HD) | ✅ bit-exact, Opus HD vectors pass |
| Custom modes | ✅ bit-exact |
| DNN: deep PLC / DRED / OSCE+BWE | ✅ integrated, bit-exact vs C built with the same features (weights via runtime blob) |
| Fixed-point build | 🟨 foundation done (fixed macros/mathops/static modes bit-exact vs a fixed oracle, gating in place); codec conversion planned in docs/FIXED_POINT.md |
| Conformance vectors (RFC 8251 all rates mono/stereo; Opus HD) | ✅ bit-exact vs C, opus_compare pass |
| Fuzzing | ✅ 11 cargo-fuzz targets (differential vs C + invariants), ~7.5M execs default+QEXT, 0 crashes/divergences |
| Benchmarks | ✅ see below (Rust ≈ scalar C; SILK decode 1.39×) |
| Shared library size | ✅ measured (Rust 773–1029 KiB vs C 323–578 KiB) |
| C ABI (`libopusorus` .so/.a, opus.h compatible) | ✅ upstream C test suite passes |
| opus_demo port | ✅ byte-identical to C over 102–121 invocations |
| Cross-platform build (wasm32 ×2, Android ×3, iOS ×3, linux x86_64/aarch64, thumbv7em no_std) | ✅ |
| wasm32-wasip1 tests under wasmtime | ✅ |
| clippy `-D warnings`, rustfmt | ✅ |

## Environment
Host: aarch64 Linux, rustc 1.98.1, clang 21 / gcc (oracle via `cc`).

## Benchmarks

Host: aarch64 Linux (Neoverse V2, 16 cores), rustc 1.98.1 release (fat LTO), criterion slope
estimates per 20 ms frame (codec) or per call (kernels). `c_scalar` = libopus 1.6.1 built like
the oracle (no SIMD, `-ffp-contract=off`, bit-identical output); `c_opt` = upstream CMake Release
(NEON intrinsics, RTCD). Ratio < 1 means Rust is faster. Reproduce with `scripts/bench_report.sh`.

| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
|---|---:|---:|---:|---:|---:|
| decode celt_48k_stereo_128k | 28.2 µs | 30.0 µs | 25.3 µs | 0.94 | 1.12 |
| decode hybrid_48k_mono_32k | 23.5 µs | 23.6 µs | 20.9 µs | 1.00 | 1.13 |
| decode silk_16k_mono_16k_voip | 8.77 µs | 6.33 µs | 5.90 µs | **1.39** | 1.49 |
| encode celt_48k_stereo_128k cx10 | 137.8 µs | 157.7 µs | 122.8 µs | 0.87 | 1.12 |
| encode celt_48k_stereo_128k cx5 | 112.6 µs | 114.5 µs | 95.6 µs | 0.98 | 1.18 |
| encode hybrid_48k_mono_32k cx10 | 239.1 µs | 242.7 µs | 179.8 µs | 0.98 | 1.33 |
| encode hybrid_48k_mono_32k cx5 | 140.7 µs | 139.7 µs | 120.7 µs | 1.01 | 1.17 |
| encode silk_16k_mono_16k_voip cx10 | 193.8 µs | 203.8 µs | 143.9 µs | 0.95 | 1.35 |
| encode silk_16k_mono_16k_voip cx5 | 99.5 µs | 105.3 µs | 88.2 µs | 0.95 | 1.13 |
| decode surround51_48k_256k | 87.6 µs | 94.5 µs | 78.3 µs | 0.93 | 1.12 |
| encode surround51_48k_256k cx10 | 460.1 µs | 486.0 µs | 396.6 µs | 0.95 | 1.16 |
| encode surround51_48k_256k cx5 | 398.9 µs | 407.5 µs | 323.7 µs | 0.98 | 1.23 |
| decode qext_96k_stereo_256k | 75.8 µs | 86.2 µs | 68.0 µs | 0.88 | 1.11 |
| encode qext_96k_stereo_256k cx10 | 290.6 µs | 298.2 µs | 244.6 µs | 0.97 | 1.19 |
| MDCT forward N=1920 | 3.91 µs | 3.79 µs | 2.90 µs | 1.03 | 1.35 |
| MDCT backward N=1920 | 3.55 µs | 3.38 µs | 2.59 µs | 1.05 | 1.37 |
| FFT 480 | 2.52 µs | 2.78 µs | 2.35 µs | 0.91 | 1.07 |
| range coder enc+dec (1000 ops) | 8.03 µs | 11.9 µs | 10.3 µs | 0.67 | 0.78 |
| SILK resampler 48k→16k | 5.12 µs | 5.33 µs | 5.50 µs | 0.96 | 0.93 |

DNN decode (20 s at complexity 10, 20 % loss, release): SILK WB 0.77× C time, CELT 48k 0.77× C.
`opus_demo` end-to-end (Rust vs C, byte-identical output): decode 12 RFC vectors 0.58 s vs 0.63 s;
encode 64 kb/s stereo 29 s input 0.41 s vs 0.45 s.

Findings: the Rust port matches or beats the scalar C build everywhere except SILK decode (1.39×);
vs NEON-optimized C it is 7–37 % slower (SIMD kernels). Performance pass: see TRACKER.

## Shared library sizes

`scripts/size_report.sh` (aarch64 Linux, stripped):

| Library | Config | Bytes | KiB |
|---|---|---:|---:|
| libopus 1.6.1 (C) | Release (-O3), default features | 592328 | 578.4 |
| libopus 1.6.1 (C) | MinSizeRel (-Os) | 330192 | 322.5 |
| libopus 1.6.1 (C) | Release, no intrinsics | 526752 | 514.4 |
| opusorus (Rust, C ABI) | release (opt-level 3, fat LTO) | 1053248 | 1028.6 |
| opusorus (Rust, C ABI) | release-small (opt-level s, fat LTO) | 791104 | 772.6 |

Of the release-small build ≈389 KB is codec code, ≈166 KB is Rust std (panic/backtrace/fmt),
≈16 KB the C ABI layer. Size reduction is tracked in the performance pass.

## Fuzzing

`fuzz/` (cargo-fuzz, own workspace). Targets: differential_decode, differential_encode,
differential_ms_encode (bit-exact vs C incl. full state dumps), decode, encode, roundtrip_invariants,
repacketizer, packet_parse_extensions, multistream_decode, projection_decode. ASan + debug assertions +
overflow checks. Campaign (2026-09-26): default build ≈6.9M execs (differential_decode 789k, decode
355k, repacketizer 4.9M, ...), QEXT build ≈1.0M execs; 0 crashes, 0 OOM, 0 timeouts, 0 divergences.
Mutation checks: injected one-character bugs were caught in 143 execs / ~1 min. Reproduce with
`fuzz/run_all.sh` (`SECS=`, `FORK=`, `TARGETS=`, `FEATURES=qext`).

## Conformance
- RFC 8251 vectors: bit-exact vs C at 8/12/16/24/48 kHz mono+stereo; average opus_compare quality
  48k 97.41 % mono / 99.70 % stereo, 24k 91.63/91.91, 16k 84.72/84.65, 12k 81.95/82.14, 8k 73.23/73.52
  (identical to C).
- Opus HD: all 12 RFC vectors pass at 96 kHz (qext_compare); qext_vector* bit-exact vs C.
- Upstream C test programs (test_opus_api/decode/encode+regressions/padding/projection/extensions/custom)
  all PASS linked against `libopusorus` (C ABI); upstream opus_demo + run_vectors 120/120 match.
- Rust port of the libopus test suite passes on host and under wasmtime (wasm32-wasip1).

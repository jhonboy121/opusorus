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
estimates per 20 ms frame (codec) or per call (kernels). `c_scalar` = libopus 1.6.1 built like the
oracle (no SIMD, `-ffp-contract=off`; bit-identical output to Rust); `c_opt` = upstream CMake
Release (NEON intrinsics, RTCD, FMA contraction allowed). Ratio < 1 means Rust is faster.
Reproduce with `scripts/bench_report.sh [--qext]`. Numbers after the performance pass (2026-09-26):

| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
|---|---:|---:|---:|---:|---:|
| decode celt_48k_stereo_128k | 28.4 µs | 32.9 µs | 28.2 µs | 0.86 | 1.01 |
| decode hybrid_48k_mono_32k | 19.7 µs | 23.8 µs | 21.2 µs | 0.83 | 0.93 |
| decode silk_16k_mono_16k_voip | 5.22 µs | 6.32 µs | 5.90 µs | 0.83 | 0.88 |
| decode qext_96k_stereo_256k | 66.0 µs | 86.3 µs | 67.8 µs | 0.76 | 0.97 |
| encode celt_48k_stereo_128k cx10 | 124.6 µs | 146.8 µs | 124.6 µs | 0.85 | 1.00 |
| encode celt_48k_stereo_128k cx5 | 100.9 µs | 116.1 µs | 97.0 µs | 0.87 | 1.04 |
| encode hybrid_48k_mono_32k cx10 | 186.0 µs | 243.1 µs | 180.1 µs | 0.77 | 1.03 |
| encode hybrid_48k_mono_32k cx5 | 125.5 µs | 139.9 µs | 121.0 µs | 0.90 | 1.04 |
| encode silk_16k_mono_16k_voip cx10 | 143.7 µs | 203.9 µs | 144.0 µs | 0.70 | 1.00 |
| encode silk_16k_mono_16k_voip cx5 | 86.6 µs | 105.2 µs | 88.2 µs | 0.82 | 0.98 |
| encode qext_96k_stereo_256k cx10 | 249.4 µs | 297.2 µs | 244.3 µs | 0.84 | 1.02 |
| encode qext_96k_stereo_256k cx5 | 224.5 µs | 270.0 µs | 221.1 µs | 0.83 | 1.02 |
| decode surround51_48k_256k | 87.7 µs | 100.9 µs | 84.8 µs | 0.87 | 1.03 |
| encode surround51_48k_256k cx10 | 415.6 µs | 490.3 µs | 400.9 µs | 0.85 | 1.04 |
| encode surround51_48k_256k cx5 | 360.6 µs | 410.9 µs | 328.2 µs | 0.88 | 1.10 |
| MDCT forward N=1920 | 3.42 µs | 3.69 µs | 2.92 µs | 0.93 | 1.17 |
| MDCT backward N=1920 | 3.25 µs | 3.36 µs | 2.58 µs | 0.97 | 1.26 |
| FFT 480 | 2.50 µs | 2.78 µs | 2.31 µs | 0.90 | 1.09 |
| range coder enc+dec (1000 ops) | 9.80 µs | 11.9 µs | 10.2 µs | 0.83 | 0.96 |
| SILK resampler 48k→16k | 5.14 µs | 5.29 µs | 5.50 µs | 0.97 | 0.93 |

Summary: Rust is faster than scalar C on every codec benchmark (0.70–0.90×) and on par with
NEON-optimized C (0.88–1.10×). The remaining MDCT/FFT gap to `c_opt` comes from FMA contraction and
NEON kernels that a bit-exact port cannot use for float code.
DNN decode (20 s at complexity 10, 20 % loss): SILK WB 0.77×, CELT 48k 0.77× C time.

## Decoder memory

`Decoder::get_size` / `MsDecoder::get_size` report the Rust footprint (struct + owned heap after
decoding 20 ms frames through the int16 API, with the DNN states of loaded models; shared DNN
weights excluded), checked against the allocations of real decoders (`crates/opusorus/tests/footprint.rs`).
Bytes, mono / stereo decoder (C: `opus_decoder_get_size`):

| Build | Rust before | Rust now | C |
|---|---:|---:|---:|
| default (48 kHz) | 87 031 / 122 682 | 36 450 / 49 058 | 18 468 / 27 236 |
| qext (96 kHz) | 152 274 / 221 480 | 53 146 / 78 266 | 27 252 / 44 692 |
| deep-plc + dred + osce | ≈3.9 MB per decoder with embedded weights | 171 578 / 201 586 | 191 580 / 200 348 |
| + qext | (399 035 / 468 314 without models) | 175 762 / 205 770 | 200 364 / 217 804 |

The C VLAs are stack arrays (the frame-sized CELT ones for up to 20 ms at 48 kHz, and 96 kHz with
QEXT); the int16/int24 `out` and multistream `buf` buffers grow on first use; the deep PLC, LACE/NoLACE
and BWE states are allocated when their models are loaded and used (C semantics unchanged); the
models of the embedded blob are bound once per process and shared (`Arc`). Without DNN models
loaded, a DNN build's decoder is 59 KB / 73 KB (48 kHz). Remaining gap without DNN: the CELT
`quant_all_bands` scratch (≈12 KB, C stack) and the 20 ms `out` buffer of the int16 API.

## Shared library sizes

`scripts/size_report.sh` (aarch64 Linux, stripped; section totals in parentheses because file sizes
are quantised by 64 KiB segment alignment on aarch64):

| Library | Config | Bytes | KiB |
|---|---|---:|---:|
| libopus 1.6.1 (C) | Release (-O3), default features | 592328 | 578.4 |
| libopus 1.6.1 (C) | MinSizeRel (-Os) | 330192 | 322.5 |
| libopus 1.6.1 (C) | Release, no intrinsics | 526752 | 514.4 |
| opusorus (Rust, C ABI) | release (opt-level 3, fat LTO) | 987712 (942801) | 964.6 |
| opusorus (Rust, C ABI) | release-small (opt-level s, fat LTO) | 791104 (733209) | 772.6 |

Breakdown of the Rust C ABI: ≈390 KB codec code, ≈150 KB Rust std panic/backtrace machinery
(gimli/addr2line/demangle), ≈90 KB panic location records (~1,400 sites), ≈47 KB `.eh_frame`,
≈16 KB C ABI glue. The C ABI is kept only as a verification harness (upstream C tests), not as a shipped product
(PLAN D-025), so its size is not optimised further.

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

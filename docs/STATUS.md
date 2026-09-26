# Status

_Last updated: 2026-09-26 (after phase G: all features)_

## Summary
The complete float codec is ported and verified bit-exact against libopus 1.6.1: CELT, SILK, hybrid,
Opus encoder/decoder, multistream, projection (ambisonics), repacketizer, extensions, analysis,
QEXT (Opus HD, 96 kHz), custom modes and the DNN features (deep PLC/FARGAN, DRED, OSCE/BWE). The
RFC 8251 and Opus HD conformance vectors decode bit-identically to C and pass
opus_compare/qext_compare. Every user-selectable upstream build option is available as a cargo
feature (phase G, PLAN D-026..D-030); see docs/FEATURES.md.

The fixed-point build (`fixed-point`, `fixed-res24`, libopus `--enable-fixed-point`) is complete:
the whole codec and the tools, bit-exact with a fixed-point libopus 1.6.1 in 16- and 24-bit
resolution, with and without QEXT and custom modes; its decoder passes the RFC 8251 vectors
(docs/FIXED_POINT.md).

| Area | State |
|---|---|
| Foundation (range coder, mathops, SILK macros, constants) | ✅ bit-exact vs libopus 1.6.1 |
| CELT (encoder, decoder, PLC, custom modes, QEXT) | ✅ bit-exact |
| SILK (FLP encoder, decoder, PLC, CNG, LBRR, resampler) | ✅ bit-exact |
| Opus layer (encoder, decoder, multistream, surround, projection, repacketizer, extensions, analysis) | ✅ bit-exact |
| QEXT (Opus HD) | ✅ bit-exact, Opus HD vectors pass |
| Custom modes | ✅ bit-exact |
| DNN: deep PLC / DRED / OSCE+BWE | ✅ integrated, bit-exact vs C built with the same features (weights via runtime blob) |
| Phase G DNN options: `dnn-debug-float`, `osce-training-data`, `lossgen` | ✅ each vs the oracle built with the same define (`just test-dnn-extras`): all DNN suites + opus_demo (161 cases) bit-exact with float-weight models (runtime and embedded blob); training files byte-identical to the C opus_demo's over 13 runs (14 with `dred`); loss model decisions + GRU states bit-exact over 91 seed/percentage streams, `opus_demo -sim_loss` and `lossgen_demo` identical to C (float and fixed-point) |
| Fixed-point build (`fixed-point`, `fixed-res24`, × `qext`, `custom-modes`) | ✅ bit-exact vs a fixed-point libopus oracle: all differential suites, libopus test-suite port (host + wasmtime), opus_demo vs the fixed C opus_demo, RFC 8251 vectors pass opus_compare (48 kHz quality 97.15/98.68 % 16-bit, 97.36/98.88 % 24-bit), Opus HD vectors pass qext_compare with `fixed-res24` |
| `disable-float-api` (libopus `DISABLE_FLOAT_API`, requires fixed point) | ✅ bit-exact vs the oracle built with `-DDISABLE_FLOAT_API`: every fixed differential suite (16-bit; 24-bit + QEXT + custom modes), libopus test-suite port, upstream C tests via the C ABI |
| `fixed-point-debug` (libopus `FIXED_DEBUG`) | ✅ every checking macro vs C (value, diagnostic, `celt_mips`); all fixed differential suites bit-exact vs the `-DFIXED_DEBUG` oracle (16-bit, 24-bit + QEXT + custom modes, with `disable-float-api`); same diagnostics as C on overflowing streams |
| Debug-build overflow hardening (public API) | ✅ C-UB overflows reachable from the API wrap like C (stereo LFE `stereo_itheta`; res24 decoder gain `MULT32_32_Q16`; CELT `alloc_trim_analysis`); public-API sweep 101 200 cases × 6 builds vs C and 40 000 Rust-only cases × 6 builds under wasmtime, 0 panics |
| Conformance vectors (RFC 8251 all rates mono/stereo; Opus HD) | ✅ bit-exact vs C, opus_compare pass |
| Fuzzing | ✅ 13 cargo-fuzz targets (differential vs C incl. DNN + Opus Custom, invariants): ~7.5M execs float/QEXT, ~11M execs fixed-point/res24; 1 port bug found+fixed (debug-assert in custom 96 kHz QEXT mode); upstream C bugs in docs/UPSTREAM_ISSUES.md |
| Benchmarks | ✅ Rust float 0.70–0.90× scalar C / 0.88–1.10× NEON C; fixed 0.68–0.94× / 0.89–1.10× |
| Shared library size | measured (information only: Rust-ecosystem use, PLAN D-025) |
| C ABI verification harness (`libopusorus`) | ✅ upstream C test suite passes, float + fixed-point (6 configs) |
| opus_demo port | ✅ byte-identical to C over 102–133 invocations (float and fixed-point C opus_demo) |
| Cross-platform build (wasm32 ×2, Android ×3, iOS ×3, linux x86_64/aarch64, thumbv7em no_std) | ✅ |
| wasm32-wasip1 tests under wasmtime | ✅ |
| clippy `-D warnings`, rustfmt, rustdoc `-D warnings` | ✅ |
| Upstream options `float-approx`, `assertions`, `fuzzing`, `disable-rfc8251` (phase G, PLAN D-030) | ✅ each vs the oracle built with the same define (`just test-options`): `float-approx` whole differential suite float/QEXT/custom/DNN + `celt_log2`/`celt_exp2`/`celt_isnan` on 44 M bit patterns; `assertions` whole suite float/fixed/QEXT/custom/DNN against an `ENABLE_ASSERTIONS` oracle, hard in release, each check kind fails on both sides (C abort in a child process); `fuzzing` seeded single-stream/multistream encoders (15 configs × 3 seeds, surround) + decoders + opus_demo vs C `opus_demo` bit-exact, float/fixed/res24/QEXT; `disable-rfc8251` whole suite float/fixed/QEXT, RFC 6716 vectors pass `run_vectors.sh` (48 kHz quality 97.43/99.66 % float; the RFC 8251 decoder fails them) |

## Verification run (2026-09-27, final: all features)

All pass: `just fmt`, `just clippy` (float, fixed-point, no_std, build-option feature sets),
`just doc` (rustdoc `-D warnings`, float/DNN/fixed/options), `just test` (float default + all float
features incl. DNN; fixed-point 16/24-bit × QEXT/custom modes incl. the upstream C suite through
the C ABI and bench parity), `just test-options` (every upstream build option vs an oracle built
with the same defines: 736 test binaries), `just cross` (wasm32 ×2, Android ×3, iOS ×3, host,
thumbv7em no_std; float + fixed), `just cross-capi`, `just test-wasm` (float + fixed under
wasmtime), `just test-dnn`, `just vectors` (RFC 8251/6716 + Opus HD, float + fixed).

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


## Benchmarks (fixed-point build)

`scripts/bench_report.sh --qext --fixed` / `--fixed-res24` (Rust fixed vs the fixed C oracle
(scalar) vs upstream CMake `OPUS_FIXED_POINT=ON` with NEON). Rust output is bit-exact with the
fixed oracle in every case.

**Fixed-point** (16-bit opus_res, opus_encode/opus_decode; c_opt = CMake OPUS_FIXED_POINT=ON, NEON, RTCD arch 3):
| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
|---|---:|---:|---:|---:|---:|
| decode celt_48k_stereo_128k | 41.7 µs | 45.3 µs | 39.8 µs | 0.92 | 1.05 |
| decode hybrid_48k_mono_32k | 23.9 µs | 27.0 µs | 24.3 µs | 0.89 | 0.99 |
| decode silk_16k_mono_16k_voip | 5.11 µs | 6.15 µs | 5.72 µs | 0.83 | 0.89 |
| decode qext_96k_stereo_256k | 93.8 µs | 107.4 µs | 89.3 µs | 0.87 | 1.05 |
| encode celt_48k_stereo_128k (cx10) | 126.1 µs | 164.5 µs | 127.1 µs | 0.77 | 0.99 |
| encode celt_48k_stereo_128k (cx5) | 95.4 µs | 130.8 µs | 95.0 µs | 0.73 | 1.00 |
| encode hybrid_48k_mono_32k (cx10) | 190.1 µs | 239.6 µs | 174.5 µs | 0.79 | 1.09 |
| encode hybrid_48k_mono_32k (cx5) | 122.5 µs | 136.9 µs | 114.6 µs | 0.89 | 1.07 |
| encode silk_16k_mono_16k_voip (cx10) | 148.5 µs | 198.3 µs | 142.0 µs | 0.75 | 1.05 |
| encode silk_16k_mono_16k_voip (cx5) | 86.2 µs | 101.8 µs | 86.8 µs | 0.85 | 0.99 |
| encode qext_96k_stereo_256k (cx10) | 247.3 µs | 346.4 µs | 242.3 µs | 0.71 | 1.02 |
| encode qext_96k_stereo_256k (cx5) | 211.9 µs | 310.3 µs | 209.6 µs | 0.68 | 1.01 |
| decode surround51_48k_256k | 121.9 µs | 129.3 µs | 111.0 µs | 0.94 | 1.10 |
| encode surround51_48k_256k (cx10) | 396.1 µs | 529.1 µs | 401.7 µs | 0.75 | 0.99 |
| encode surround51_48k_256k (cx5) | 333.8 µs | 453.5 µs | 329.2 µs | 0.74 | 1.01 |
| MDCT forward N=1920 | 6.39 µs | 5.96 µs | 5.75 µs | 1.07 | **1.11** |
| MDCT backward N=1920 | 6.32 µs | 6.28 µs | 5.23 µs | 1.01 | **1.21** |
| FFT 480 | 4.41 µs | 4.29 µs | 4.30 µs | 1.03 | 1.03 |
| range coder enc+dec, 1000 ops | 9.73 µs | 11.9 µs | 10.2 µs | 0.82 | 0.95 |
| SILK resampler 48k->16k | 5.14 µs | 5.32 µs | 5.48 µs | 0.97 | 0.94 |

**Fixed-res24** (24-bit opus_res, opus_encode24/opus_decode24; c_opt + -DENABLE_RES24):
| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
|---|---:|---:|---:|---:|---:|
| decode celt_48k_stereo_128k | 44.6 µs | 44.5 µs | 39.6 µs | 1.00 | **1.13** |
| decode hybrid_48k_mono_32k | 23.4 µs | 27.2 µs | 24.3 µs | 0.86 | 0.96 |
| decode silk_16k_mono_16k_voip | 5.06 µs | 6.27 µs | 5.84 µs | 0.81 | 0.87 |
| decode qext_96k_stereo_256k | 92.1 µs | 106.3 µs | 89.0 µs | 0.87 | 1.04 |
| encode celt_48k_stereo_128k (cx10) | 126.7 µs | 162.7 µs | 128.4 µs | 0.78 | 0.99 |
| encode celt_48k_stereo_128k (cx5) | 96.3 µs | 128.3 µs | 96.8 µs | 0.75 | 1.00 |
| encode hybrid_48k_mono_32k (cx10) | 190.4 µs | 238.7 µs | 175.2 µs | 0.80 | 1.09 |
| encode hybrid_48k_mono_32k (cx5) | 123.1 µs | 136.1 µs | 115.1 µs | 0.90 | 1.07 |
| encode silk_16k_mono_16k_voip (cx10) | 149.3 µs | 198.8 µs | 142.7 µs | 0.75 | 1.05 |
| encode silk_16k_mono_16k_voip (cx5) | 87.0 µs | 102.1 µs | 87.5 µs | 0.85 | 0.99 |
| encode qext_96k_stereo_256k (cx10) | 250.5 µs | 343.1 µs | 247.5 µs | 0.73 | 1.01 |
| encode qext_96k_stereo_256k (cx5) | 215.4 µs | 306.8 µs | 214.8 µs | 0.70 | 1.00 |
| decode surround51_48k_256k | 126.8 µs | 127.1 µs | 109.9 µs | 1.00 | **1.15** |
| encode surround51_48k_256k (cx10) | 397.7 µs | 523.1 µs | 406.2 µs | 0.76 | 0.98 |
| encode surround51_48k_256k (cx5) | 336.5 µs | 446.7 µs | 335.3 µs | 0.75 | 1.00 |
| MDCT forward N=1920 | 6.39 µs | 5.96 µs | 5.75 µs | 1.07 | **1.11** |
| MDCT backward N=1920 | 6.32 µs | 6.28 µs | 5.23 µs | 1.01 | **1.21** |
| FFT 480 | 4.42 µs | 4.29 µs | 4.30 µs | 1.03 | 1.03 |
| range coder enc+dec, 1000 ops | 9.84 µs | 11.9 µs | 10.2 µs | 0.83 | 0.96 |
| SILK resampler 48k->16k | 5.13 µs | 5.32 µs | 5.50 µs | 0.97 | 0.93 |


Summary: fixed-point codec benches run at 0.68–0.94× scalar C time (res24 0.70–1.00×) and
0.89–1.10× NEON C (res24 0.87–1.15×).

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

## Conformance (fixed-point)

- Upstream C test suite against fixed `libopusorus` (C ABI): test_opus_api/decode/encode(+regressions)/padding/projection/extensions/custom all PASS in fixed-point, fixed-res24 (+qext, +custom-modes); upstream opus_demo + opus_compare RFC 8251 vectors 120/120 in each.
- Rust fixed decoder RFC 8251 quality (mono/stereo, 48k): fixed-point 97.15/98.68 %, fixed-res24 97.36/98.88 % (float 97.41/99.70 %); all rates pass. Opus HD: fixed-res24+qext passes all 12 at 96 kHz; 16-bit fixed+qext is limited by its output rounding floor (same as C).
- Fixed-point fuzzing: ≈6.4M execs fixed-point + ≈4.9M fixed-res24+qext across 10 targets (differential vs fixed C), 0 findings in the port. See docs/UPSTREAM_ISSUES.md for upstream C bugs found.

## Conformance
- RFC 8251 vectors: bit-exact vs C at 8/12/16/24/48 kHz mono+stereo; average opus_compare quality
  48k 97.41 % mono / 99.70 % stereo, 24k 91.63/91.91, 16k 84.72/84.65, 12k 81.95/82.14, 8k 73.23/73.52
  (identical to C).
- Fixed-point decoder (bit-exact vs the fixed C decoder, all 12 vectors pass at every rate):
  `fixed-point` 48k 97.15/98.68, 24k 91.62/91.91, 16k 84.69/84.65, 12k 81.96/82.17, 8k 73.23/73.53;
  `fixed-res24` 48k 97.36/98.88, 24k 91.63/91.91, 16k 84.69/84.66, 12k 81.95/82.16, 8k 73.23/73.52
  (with `qext`: 97.19/98.90 and 97.39/99.13 at 48k). Opus HD at 96 kHz with `fixed-res24` + `qext`:
  all 12 RFC vectors pass qext_compare; the 16-bit `fixed-point` + `qext` output is limited to
  16-bit resolution (9 vectors at rms 0.13–0.29 LSB > 0.1, identical to the 16-bit fixed C
  opus_demo). See docs/FIXED_POINT.md.
- Opus HD: all 12 RFC vectors pass at 96 kHz (qext_compare); qext_vector* bit-exact vs C.
- Upstream C test programs (test_opus_api/decode/encode+regressions/padding/projection/extensions/custom)
  all PASS linked against `libopusorus` (C ABI); upstream opus_demo + run_vectors 120/120 match.
- Rust port of the libopus test suite passes on host and under wasmtime (wasm32-wasip1), in the
  float and the fixed-point builds (`fixed-point`, `fixed-res24`, × `qext`, `custom-modes`).

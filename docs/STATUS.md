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
| Benchmarks | ✅ bit-exact Rust float codec 0.70–0.90× scalar C / 0.87–1.08× NEON C, FFT/MDCT 0.85–1.00× NEON C (fearless_simd, D-031); fixed 0.65–0.92× / 0.89–1.07×, FFT/MDCT 0.70–0.74×; DNN 1.27–1.39× NEON C bit-exact, 0.99–1.05× with `fast` (D-032) |
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
Reproduce with `scripts/bench_report.sh --qext --dnn`. Numbers after the fearless_simd work
(2026-09-27, PLAN D-031; the build has the DNN features, so its SILK/hybrid decode rows include the
DNN feature analysis every frame, on both sides):

| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode celt_48k_stereo_128k | 27.7 µs | 32.9 µs | 28.3 µs | 0.84 | 0.98 |
| decode hybrid_48k_mono_32k | 30.3 µs | 37.4 µs | 33.3 µs | 0.81 | 0.91 |
| decode silk_16k_mono_16k_voip | 15.6 µs | 19.6 µs | 17.9 µs | 0.79 | 0.87 |
| decode qext_96k_stereo_256k | 63.1 µs | 86.2 µs | 68.2 µs | 0.73 | 0.93 |
| encode celt_48k_stereo_128k (cx10) | 122.3 µs | 146.8 µs | 124.4 µs | 0.83 | 0.98 |
| encode celt_48k_stereo_128k (cx5) | 99.7 µs | 116.1 µs | 97.2 µs | 0.86 | 1.03 |
| encode hybrid_48k_mono_32k (cx10) | 185.4 µs | 243.0 µs | 180.1 µs | 0.76 | 1.03 |
| encode hybrid_48k_mono_32k (cx5) | 126.0 µs | 139.9 µs | 120.9 µs | 0.90 | 1.04 |
| encode silk_16k_mono_16k_voip (cx10) | 143.4 µs | 204.0 µs | 144.0 µs | 0.70 | 1.00 |
| encode silk_16k_mono_16k_voip (cx5) | 87.3 µs | 105.3 µs | 88.3 µs | 0.83 | 0.99 |
| encode qext_96k_stereo_256k (cx10) | 243.4 µs | 297.1 µs | 244.4 µs | 0.82 | 1.00 |
| encode qext_96k_stereo_256k (cx5) | 219.2 µs | 270.0 µs | 221.2 µs | 0.81 | 0.99 |
| decode surround51_48k_256k | 85.7 µs | 101.0 µs | 84.7 µs | 0.85 | 1.01 |
| encode surround51_48k_256k (cx10) | 403.4 µs | 488.4 µs | 399.9 µs | 0.83 | 1.01 |
| encode surround51_48k_256k (cx5) | 352.9 µs | 409.4 µs | 327.4 µs | 0.86 | 1.08 |
| DNN deep PLC, 20 % loss, silk_16k_mono | 267.9 µs | 1.49 ms | 197.6 µs | 0.18 | **1.36** |
| DNN deep PLC, 20 % loss, hybrid_48k_mono | 282.6 µs | 1.51 ms | 212.9 µs | 0.19 | **1.33** |
| DNN LACE decode silk_16k_mono | 70.4 µs | 328.0 µs | 55.2 µs | 0.21 | **1.27** |
| DNN NoLACE decode silk_16k_mono | 238.8 µs | 1.25 ms | 172.1 µs | 0.19 | **1.39** |
| DNN DRED encode silk_16k_mono (1 s) | 195.7 µs | 395.7 µs | 189.5 µs | 0.49 | 1.03 |
| MDCT forward N=1920 | 2.69 µs | 3.70 µs | 2.93 µs | 0.73 | 0.92 |
| MDCT backward N=1920 | 2.57 µs | 3.36 µs | 2.59 µs | 0.77 | 1.00 |
| FFT 480 | 1.95 µs | 2.79 µs | 2.28 µs | 0.70 | 0.85 |
| range coder enc+dec, 1000 ops | 9.83 µs | 11.9 µs | 10.2 µs | 0.83 | 0.96 |
| SILK resampler 48k->16k | 5.17 µs | 5.33 µs | 5.49 µs | 0.97 | 0.94 |

Summary: bit-exact Rust is faster than scalar C everywhere (0.18–0.97×) and on par with or faster
than NEON-optimized C on the codec (0.87–1.08×) and the FFT/MDCT kernels (0.85–1.00×). The DNN
rows are the exception in the bit-exact build (1.27–1.39× NEON C, whose int8 products accumulate
in integers across blocks, which the reference's float accumulation cannot): the opt-in `fast`
feature (PLAN D-032) closes it:

| DNN benchmark | rust (bit-exact) | rust `fast` | c_opt | fast/c_opt |
|---|---:|---:|---:|---:|
| deep PLC, 20 % loss, silk_16k_mono | 267.9 µs | 198.1 µs | 197.6 µs | 1.00 |
| deep PLC, 20 % loss, hybrid_48k_mono | 282.6 µs | 211.8 µs | 212.9 µs | 0.99 |
| LACE decode silk_16k_mono | 70.4 µs | 56.7 µs | 55.2 µs | 1.03 |
| NoLACE decode silk_16k_mono | 238.8 µs | 181.5 µs | 172.1 µs | 1.05 |
| DRED encode silk_16k_mono (1 s) | 195.7 µs | 190.5 µs | 189.5 µs | 1.01 |

Before the fearless_simd work the bit-exact DNN rows were 1143 / 1156 / 250 / 975 / 282 µs
(4.5–5.8× NEON C) and the float FFT/MDCT 1.09–1.26× NEON C.

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
| --- | ---: | ---: | ---: | ---: | ---: |
| decode celt_48k_stereo_128k | 38.0 µs | 45.3 µs | 39.9 µs | 0.84 | 0.95 |
| decode hybrid_48k_mono_32k | 22.6 µs | 27.0 µs | 24.3 µs | 0.84 | 0.93 |
| decode silk_16k_mono_16k_voip | 5.08 µs | 6.15 µs | 5.70 µs | 0.83 | 0.89 |
| decode qext_96k_stereo_256k | 83.9 µs | 107.4 µs | 89.3 µs | 0.78 | 0.94 |
| encode celt_48k_stereo_128k (cx10) | 121.6 µs | 164.6 µs | 127.0 µs | 0.74 | 0.96 |
| encode celt_48k_stereo_128k (cx5) | 92.1 µs | 130.7 µs | 95.0 µs | 0.70 | 0.97 |
| encode hybrid_48k_mono_32k (cx10) | 187.1 µs | 239.2 µs | 174.5 µs | 0.78 | 1.07 |
| encode hybrid_48k_mono_32k (cx5) | 122.0 µs | 136.8 µs | 114.5 µs | 0.89 | 1.06 |
| encode silk_16k_mono_16k_voip (cx10) | 147.1 µs | 198.1 µs | 142.0 µs | 0.74 | 1.04 |
| encode silk_16k_mono_16k_voip (cx5) | 86.4 µs | 101.7 µs | 86.8 µs | 0.85 | 1.00 |
| encode qext_96k_stereo_256k (cx10) | 237.9 µs | 346.5 µs | 242.4 µs | 0.69 | 0.98 |
| encode qext_96k_stereo_256k (cx5) | 202.7 µs | 310.3 µs | 209.7 µs | 0.65 | 0.97 |
| decode surround51_48k_256k | 111.6 µs | 129.2 µs | 111.0 µs | 0.86 | 1.01 |
| encode surround51_48k_256k (cx10) | 364.6 µs | 529.0 µs | 401.3 µs | 0.69 | 0.91 |
| encode surround51_48k_256k (cx5) | 309.8 µs | 453.4 µs | 329.0 µs | 0.68 | 0.94 |
| MDCT forward N=1920 | 4.09 µs | 5.96 µs | 5.75 µs | 0.69 | 0.71 |
| MDCT backward N=1920 | 3.88 µs | 6.30 µs | 5.24 µs | 0.62 | 0.74 |
| FFT 480 | 3.03 µs | 4.29 µs | 4.30 µs | 0.71 | 0.70 |
| range coder enc+dec, 1000 ops | 9.85 µs | 11.9 µs | 10.2 µs | 0.83 | 0.97 |
| SILK resampler 48k->16k | 5.11 µs | 5.33 µs | 5.50 µs | 0.96 | 0.93 |

**Fixed-res24** (24-bit opus_res, opus_encode24/opus_decode24; c_opt + -DENABLE_RES24):
| benchmark | rust | c_scalar | c_opt | rust/c_scalar | rust/c_opt |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode celt_48k_stereo_128k | 41.0 µs | 44.5 µs | 39.5 µs | 0.92 | 1.04 |
| decode hybrid_48k_mono_32k | 21.9 µs | 27.2 µs | 24.3 µs | 0.80 | 0.90 |
| decode silk_16k_mono_16k_voip | 5.08 µs | 6.27 µs | 5.82 µs | 0.81 | 0.87 |
| decode qext_96k_stereo_256k | 82.2 µs | 106.5 µs | 89.0 µs | 0.77 | 0.92 |
| encode celt_48k_stereo_128k (cx10) | 121.9 µs | 162.9 µs | 128.4 µs | 0.75 | 0.95 |
| encode celt_48k_stereo_128k (cx5) | 92.8 µs | 128.3 µs | 96.8 µs | 0.72 | 0.96 |
| encode hybrid_48k_mono_32k (cx10) | 185.0 µs | 238.5 µs | 175.0 µs | 0.78 | 1.06 |
| encode hybrid_48k_mono_32k (cx5) | 122.1 µs | 136.2 µs | 115.3 µs | 0.90 | 1.06 |
| encode silk_16k_mono_16k_voip (cx10) | 145.4 µs | 198.8 µs | 142.7 µs | 0.73 | 1.02 |
| encode silk_16k_mono_16k_voip (cx5) | 86.8 µs | 102.3 µs | 87.7 µs | 0.85 | 0.99 |
| encode qext_96k_stereo_256k (cx10) | 240.0 µs | 343.1 µs | 246.6 µs | 0.70 | 0.97 |
| encode qext_96k_stereo_256k (cx5) | 204.5 µs | 306.8 µs | 214.1 µs | 0.67 | 0.96 |
| decode surround51_48k_256k | 115.7 µs | 127.3 µs | 109.8 µs | 0.91 | 1.05 |
| encode surround51_48k_256k (cx10) | 364.5 µs | 522.4 µs | 404.1 µs | 0.70 | 0.90 |
| encode surround51_48k_256k (cx5) | 311.4 µs | 446.1 µs | 333.5 µs | 0.70 | 0.93 |
| MDCT forward N=1920 | 4.09 µs | 5.97 µs | 5.76 µs | 0.68 | 0.71 |
| MDCT backward N=1920 | 3.89 µs | 6.29 µs | 5.24 µs | 0.62 | 0.74 |
| FFT 480 | 3.01 µs | 4.29 µs | 4.30 µs | 0.70 | 0.70 |
| range coder enc+dec, 1000 ops | 9.84 µs | 11.9 µs | 10.2 µs | 0.83 | 0.96 |
| SILK resampler 48k->16k | 5.13 µs | 5.33 µs | 5.50 µs | 0.96 | 0.93 |


Summary (after the fixed-point fearless_simd FFT/MDCT, 2026-09-27): fixed-point codec benches run at
0.65–0.92× scalar C time and 0.89–1.07× NEON C (res24 0.87–1.06×); the FFT/MDCT kernels at 0.70–0.74×
NEON C (before: 1.03–1.21×).

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

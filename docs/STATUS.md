# Status

_Last updated: 2026-09-26 (after layer D)_

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
| DNN: deep PLC / DRED / OSCE+BWE | 🟨 modules bit-exact; codec integration in progress |
| Fixed-point build | ⬜ planned (phase F) |
| Conformance vectors (RFC 8251 all rates mono/stereo; Opus HD) | ✅ bit-exact vs C, opus_compare pass |
| Fuzzing | ⬜ |
| Benchmarks | 🟨 early: decoder ≈0.86–0.94× C (scalar) time; full suite in progress |
| Shared library size | ⬜ |
| Cross-platform build (wasm32 ×2, Android ×3, iOS ×3, linux x86_64/aarch64, thumbv7em no_std) | ✅ |
| wasm32-wasip1 tests under wasmtime | ✅ |
| clippy `-D warnings`, rustfmt | ✅ |

## Environment
Host: aarch64 Linux, rustc 1.98.1, clang 21 / gcc (oracle via `cc`).

## Benchmarks
_Pending (phase E)._

## Shared library sizes
_Pending (phase E)._

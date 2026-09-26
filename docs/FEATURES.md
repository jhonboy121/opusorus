# Feature list (libopus v1.6.1 → opusorus)

Status legend: ⬜ not started · 🟨 in progress · ✅ ported & verified vs oracle · ➖ not applicable

## Core codec
| # | Feature | libopus source | Status |
|---|---|---|---|
| F1 | Range coder (encode/decode, raw bits, uint, icdf, patch, shrink) | celt/entenc.c, entdec.c, entcode.c | ✅ |
| F2 | CELT math helpers (float), log/exp/cos/atan approximations | celt/mathops.* | ✅ |
| F3 | KISS FFT + MDCT (forward/backward, all 4 shifts) | celt/kiss_fft.c, mdct.c | ✅ |
| F4 | CELT mode tables (48 kHz static mode), band layout, caps | celt/modes.c, static_modes_float.h | ✅ |
| F5 | PVQ codebook (cwrs), Laplace coder | celt/cwrs.c, laplace.c | ✅ |
| F6 | CELT bit allocation | celt/rate.c | ✅ |
| F7 | Pitch analysis, LPC, autocorrelation | celt/pitch.c, celt_lpc.c | ✅ |
| F8 | PVQ search/quantization, spreading, energy quantization | celt/vq.c, quant_bands.c | ✅ |
| F9 | Band quantization (quant_all_bands, stereo, theta RDO, TF) | celt/bands.c | ✅ |
| F10 | CELT decoder incl. PLC (pitch-based + noise), postfilter, deemphasis | celt/celt_decoder.c, celt.c | ✅ |
| F11 | CELT encoder incl. transient detection, TF analysis, dynalloc, prefilter, VBR | celt/celt_encoder.c | ✅ |
| F12 | SILK tables, NLSF codebooks, LPC utilities, sigproc | silk/*.c | ✅ |
| F13 | SILK resampler (all rate pairs, up2 HQ, IIR/FIR, down FIR) | silk/resampler*.c | ✅ |
| F14 | SILK decoder (mono/stereo, 10/20/40/60 ms, LBRR/FEC, PLC, CNG) | silk/dec_API.c ... | ✅ |
| F15 | SILK encoder (float), NSQ + delayed-decision NSQ, VAD, LBRR, DTX, stereo | silk/enc_API.c, silk/float/*.c | ✅ |
| F16 | Opus decoder (SILK/CELT/hybrid, transitions, redundancy, FEC, PLC, gain, soft clip, 16/24-bit/float output) | src/opus_decoder.c | ✅ |
| F17 | Opus encoder (mode/bandwidth decisions, hybrid, redundancy, FEC, DTX, VBR/CVBR/CBR, LSB depth, prediction/phase-inversion controls, 2.5–120 ms frames, multi-frame packing) | src/opus_encoder.c | ✅ |
| F18 | Music/speech analysis (tonality, MLP) | src/analysis.c, mlp.c, mlp_data.c | ✅ |
| F19 | Packet parsing & TOC helpers, soft clip | src/opus.c | ✅ |
| F20 | Repacketizer, packet pad/unpad (single and multistream) | src/repacketizer.c | ✅ |
| F21 | Packet extensions (parse, generate, count, iterate) + `IGNORE_EXTENSIONS` | src/extensions.c | ✅ |
| F22 | Multistream encoder/decoder, surround (families 0, 1, 255), channel mapping | src/opus_multistream*.c | ✅ |
| F23 | Projection (ambisonics, families 2 and 3), mapping matrices | src/opus_projection_*.c, mapping_matrix.c | ✅ |
| F24 | All encoder/decoder CTLs as typed methods | include/opus_defines.h | ✅ |
| F25 | Restricted SILK / restricted CELT applications (1.6) | src/opus_encoder.c | ✅ |

## Optional (compile-time) features
| # | Feature | libopus flag | Cargo feature | Status |
|---|---|---|---|---|
| O1 | QEXT: Opus HD / scalable quality extension (96 kHz, extra precision in extensions) | `--enable-qext` | `qext` | ✅ |
| O2 | Opus Custom modes (non-48k-family rates, custom frame sizes, opus_custom API) | `--enable-custom-modes` | `custom-modes` | ✅ |
| O3 | Fixed-point build (+ RES24) | `--enable-fixed-point` | `fixed-point` | ⬜ |
| O4 | Deep PLC (LPCNet/FARGAN-based concealment) | `--enable-deep-plc` | `deep-plc` | ✅ |
| O5 | DRED: Deep REDundancy (encoder + decoder, `opus_dred_*` API) | `--enable-dred` | `dred` | ✅ |
| O6 | OSCE: SILK speech enhancement (LACE/NoLACE) + BWE | `--enable-osce` | `osce` | ✅ |

## Tooling & quality
| # | Item | Status |
|---|---|---|
| T1 | Differential tests vs C oracle (per unit + full codec) | ✅ |
| T2 | RFC 6716/8251 test vectors + `opus_compare` port | ✅ |
| T3 | Ported libopus test suite (test_opus_api/decode/encode/padding/extensions/projection) | ✅ |
| T4 | C ABI (`libopusorus.so`, `opus.h` compatible) + C test programs run against it | ✅ |
| T5 | Fuzzing (cargo-fuzz): decoder, encoder, repacketizer, multistream, extensions, differential | ✅ |
| T6 | Benchmarks (criterion) Rust vs C, reported in STATUS | ✅ |
| T7 | Shared library size comparison, reported in STATUS | ✅ |
| T8 | Cross-platform builds: wasm32, Android (arm64/armv7/x86_64), iOS (+sim), host, no_std | ✅ (foundation) |
| T9 | `opus_demo`-compatible CLI (`opusorus-tools`) | ✅ |

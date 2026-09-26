# opusorus fuzzing

cargo-fuzz targets (libFuzzer, nightly toolchain). This directory is a standalone workspace,
excluded from the main one, so `cargo build --workspace` never needs nightly.

| Target | What it checks |
| --- | --- |
| `differential_decode` | Arbitrary packet / PLC / FEC / CTL sequences through `Decoder` **and** C `OpusDecoder`: identical return codes, bit-exact output (plus untouched buffer tail), GET CTLs, final range and Opus-level state after every call. |
| `decode` | Same op stream through the public Rust API only (including too-short buffers and `usize` frame sizes): no panics, API contract (sample counts, `last_packet_duration`, finite output, CTL error kinds). |
| `differential_encode` | Arbitrary configuration, CTL changes and PCM (synthesized or raw bit patterns; i16 / i24 / float; any frame and buffer size) through `Encoder` and C `OpusEncoder`: identical return codes, packet bytes, final range and all GET CTLs. |
| `encode` | Same input, Rust only: no panics; every packet fits, parses, has the right duration and decodes. |
| `roundtrip_invariants` | Encoder output decodes at the same and another rate / channel count to exactly its duration; `get_nb_samples` family consistent; decoder final range == encoder final range; FEC decode works; padded / repacketized / unpadded packets decode bit-identically. |
| `differential_ms_encode` | Surround multistream encoder (families 0, 1, 2, 255, invalid) and ambisonics projection encoder (family 3) vs C: layouts, demixing matrix, return codes, packet bytes, GET CTLs. |
| `repacketizer` | `cat` / `out_range` / `out` / `init`, `packet_pad` / `unpad` and the multistream variants vs C (codes and bytes); outputs re-parse into the expected frames. |
| `packet_parse_extensions` | `parse_impl` (both framings), TOC helpers, `has_lbrr`, extension count / parse / parse_ext / iterator / generate vs C; generate-parse round trips; `pcm_soft_clip` vs C. |
| `multistream_decode` | Arbitrary layouts + op stream through `MsDecoder` and C `OpusMSDecoder` (codes, samples, CTLs, per-stream state). |
| `projection_decode` | Arbitrary demixing matrices + op stream through `ProjectionDecoder` and C `OpusProjectionDecoder`. |
| `differential_dnn_decode` | (`deep-plc`, `osce`, `dred`) The decoder op stream with deep PLC / OSCE active (complexity 5-10, `OPUS_SET_OSCE_BWE`) through a `Decoder` loaded with the oracle's weights (the blob it serializes from its compiled-in tables) and the C DNN `OpusDecoder`: codes, bit-exact samples, GET CTLs and the DNN state (LPCNet PLC counters, FEC queue, `pcm`/`features` history, OSCE method / BWE mode). |
| `differential_custom` | (`custom-modes`) Opus Custom API vs C: mode creation (static / custom / invalid rates and frame sizes), `opus_custom_encode*` (i16 / i24 / float), `opus_custom_decode*` of the encoder output, of arbitrary packets and PLC, numeric encoder / decoder CTLs, start / end bands: codes, packet bytes, bit-exact samples, final ranges and GET CTLs. |

The input formats are documented in `src/dec.rs`, `src/enc.rs` and each target's header.

## Configurations

`qext`, `custom-modes`, `fixed-point`, `fixed-res24`, `deep-plc`, `osce` and `dred` forward to both
`opusorus` and `opusorus-oracle`, so the port and the C oracle are always built alike.

* **Fixed point** (`fixed-point`, `fixed-res24`, optionally `+qext` / `+custom-modes`): every
  target builds and runs against the fixed-point C oracle; the corpus is seeded with the fixed
  C encoder. The APIs are the same (16-bit / 24-bit / float entry points; the float API goes
  through `FLOAT2RES` / `RES2FLOAT`); the decoder state dump has no soft-clip memory (compared
  as zeros, like the oracle's dump).
* **DNN** (`deep-plc`, `osce`, `dred`; float only; needs `scripts/fetch_dnn_models.sh`): build
  with all three (`--features deep-plc,osce,dred`): the oracle's DRED bindings are compiled
  unconditionally but their C shims only with `dred`, and the coverage-instrumented build keeps
  the references alive. The C oracle has compiled-in weights, so the plain decoder targets
  diverge by design (their Rust decoder has no weights); use `differential_dnn_decode`, which
  loads the oracle's weights into the port.

Input classes that are undefined behaviour in C (and overflow-panic or fail a debug assertion
in the port's fuzz builds) are not generated; see `enc::avoid_known_ub`, `enc::raw_i24` and the
header of `differential_custom`:

* fixed point: `OPUS_SET_LFE(1)` on a stereo `Encoder` (overflow in `stereo_itheta`) is issued
  as `OPUS_SET_LFE(0)`; raw 24-bit input is kept within 24 bits (`INT24TORES` / `INT24TOSIG`
  and the analysis downmix overflow on larger values);
* custom API: invalid CTL combinations (channels above the encoder's, QEXT without signalling,
  start bands other than 0 / 17, end bands < 3, stereo with < 13 bands, bands beyond
  `effEBands` on stereo / decoders, NaN input, frame sizes <= 0) and several upstream
  out-of-bounds accesses of Opus Custom (+ QEXT) listed in the target: `special_hybrid_folding`
  with custom band layouts, `qext_eBands_180` modes, 240-sample overlaps at rates other than
  96 kHz, end bands 2 / 14 taken for the QEXT pass, a 2-byte signalled CBR packet, the
  `opus_fft_free` failure path.

## Commands

```sh
cd fuzz
./seed_corpus.sh                                   # corpus/<target>/ from the C encoder
./seed_corpus.sh --features qext -- corpus-qext    # QEXT seeds
./seed_corpus.sh --features fixed-point -- corpus-fixed-point   # seeds from the fixed C encoder
cargo +nightly fuzz run differential_decode corpus/differential_decode -- -max_total_time=600 -fork=4 -max_len=16384
cargo +nightly fuzz run --features fixed-res24,qext --target-dir target-fixed-res24-qext \
    differential_encode corpus-fixed-res24-qext/differential_encode -- -fork=3
./run_all.sh                                       # whole campaign (see the script for knobs)
FEATURES=qext ./run_all.sh                         # QEXT build, corpus-qext/, target-qext/
FEATURES=fixed-point ./run_all.sh                  # fixed point, 16-bit opus_res
FEATURES=fixed-res24,qext ./run_all.sh             # fixed point, 24-bit opus_res, QEXT
FEATURES=deep-plc,osce,dred TARGETS=differential_dnn_decode ./run_all.sh
FEATURES=custom-modes,qext TARGETS=differential_custom ./run_all.sh
```

`run_all.sh` builds all targets once, then runs each for `LONG_SECS` (differential decode /
encode, decode) or `SECS`, with `FORK` workers; logs go to `logs/<features>/<target>.log` and a
per-target summary (final `cov` / `ft` / `corp` / execs / crashes) is printed at the end.

Crashes land in `artifacts/<target>/`; replay one with
`cargo +nightly fuzz run <target> artifacts/<target>/crash-...`. Real bugs get a minimal fix
plus a regression test in `crates/opusorus-conformance/tests/fuzz_regressions.rs`.

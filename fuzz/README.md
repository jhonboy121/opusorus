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

The input formats are documented in `src/dec.rs`, `src/enc.rs` and each target's header.

## Commands

```sh
cd fuzz
./seed_corpus.sh                                   # corpus/<target>/ from the C encoder
./seed_corpus.sh --features qext -- corpus-qext    # QEXT seeds
cargo +nightly fuzz run differential_decode corpus/differential_decode -- -max_total_time=600 -fork=4 -max_len=16384
./run_all.sh                                       # whole campaign (see the script for knobs)
FEATURES=qext ./run_all.sh                         # QEXT build, corpus-qext/, target-qext/
```

Crashes land in `artifacts/<target>/`; replay one with
`cargo +nightly fuzz run <target> artifacts/<target>/crash-...`. Real bugs get a minimal fix
plus a regression test in `crates/opusorus-conformance/tests/fuzz_regressions.rs`.

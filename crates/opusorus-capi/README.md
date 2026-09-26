# opusorus-capi

A drop-in C ABI for libopus 1.6.1 backed by the safe-Rust `opusorus` port. It builds
`libopusorus.so` (`.dylib`/`.dll`) and `libopusorus.a`, exporting every function declared in
libopus' `opus.h`, `opus_multistream.h` and `opus_projection.h` (plus `opus_custom.h` with the
`custom-modes` feature) with the same signatures and semantics. The unmodified upstream headers
are in `include/`.

```sh
cargo build -p opusorus-capi --release            # target/release/libopusorus.{so,a}
crates/opusorus-capi/scripts/pkgconfig.sh         # target/release/pkgconfig/{opus,opusorus}.pc
PKG_CONFIG_PATH=target/release/pkgconfig pkg-config --cflags --libs opus
crates/opusorus-capi/scripts/pkgconfig.sh --prefix /usr/local   # install lib, headers, .pc
```

Features: `qext` (Opus HD, 96 kHz), `custom-modes` (`opus_custom.h`), `deep-plc` / `osce` /
`dred` (forwarded to `opusorus`; weights loaded with `OPUS_SET_DNN_BLOB` on encoders, decoders
and DRED decoders, like libopus `USE_WEIGHTS_FILE`), `dnn-weights-embedded` (compiled-in
weights: `OPUSORUS_DNN_BLOB=/abs/path/weights_blob.bin`, see `scripts/gen_dnn_blob.sh`),
`internal-api` (also exports the libopus-internal functions used by upstream's
`test_opus_extensions.c`).

## How it works

* **States.** A C state block (`opus_*_get_size()` bytes, from `*_create` or caller memory via
  `*_init`) holds a 32-byte header; the Rust object lives in a registry keyed by the header's
  id (`src/handle.rs`). Byte copies of a state (`memcpy`, as upstream's tests and some
  applications do) become independent states on first use; a moved state (copy, then scramble
  and destroy the old block) is adopted by its new block. `*_destroy` frees the block with C
  `free`, like libopus.
* **CTLs.** The variadic `opus_*_ctl` functions are C (`csrc/ctl.c`), reading the arguments by
  request type and calling non-variadic Rust functions (`src/ctl.rs`). Because a Rust `cdylib`
  only exports Rust-defined symbols, the exported CTL symbols are Rust naked functions that
  tail-jump to the C code (x86, x86-64, AArch64, ARM, RISC-V 64).
* **Safety.** Every export runs under `catch_unwind` (a caught panic returns
  `OPUS_INTERNAL_ERROR`; release builds use `panic = "abort"`). Pointers libopus would
  dereference unconditionally are NULL-checked (`OPUS_BAD_ARG`), and buffer sizes are
  validated before slices are formed.

## Differences from libopus

* A byte copy of a state is materialized on its first use, so it sees the source state as of
  that moment (the same as the copy-then-use pattern libopus users rely on). A copy whose source
  was destroyed before the copy was used returns `OPUS_INVALID_STATE`.
* `OPUS_MULTISTREAM_GET_ENCODER_STATE` / `_DECODER_STATE` return small per-stream handles valid
  while the parent lives, not interior pointers.
* The private `CELT_GET_MODE` request is not supported; `OPUS_SET_ENERGY_MASK` copies the mask.
* `opus_dred_*` behaves like a libopus build without `ENABLE_DRED` unless the `dred` feature is
  on. `OPUS_SET_DNN_BLOB` is also accepted with compiled-in weights (libopus: `OPUS_UNIMPLEMENTED`).
* Repacketizer: frame bytes are read at output time, as in C; the output buffer may overlap the
  input packets.

## Tests

`cargo test -p opusorus-capi [--features qext,custom-modes]` builds the library (nested
`cargo build`) and runs upstream's C programs against it: `test_opus_api`, `test_opus_decode`,
`test_opus_encode` (+ `opus_encode_regressions.c`), `test_opus_padding`, `test_opus_projection`,
`test_opus_extensions`, `test_opus_custom`, plus `tests/csrc/capi_extra.c`, a static-link run,
and the RFC 8251 vectors through upstream `opus_demo` / `opus_compare` at 8/12/16/24/48 kHz,
mono and stereo (skipped when `testdata/vectors/rfc8251` is absent).

With `--features dred` (or `deep-plc` / `osce`) the library is built with compiled-in weights
(`dnn-weights-embedded`; the blob is `$OPUSORUS_DNN_BLOB` or generated with
`scripts/gen_dnn_blob.sh`) and the C programs with the matching `ENABLE_*` defines, as upstream
tests a DNN build: `test_opus_dred` (10 M random DRED payloads through `opus_dred_parse` /
`opus_dred_process`) is added, `test_opus_encode` parses and decodes the DRED of its packets,
and `capi_extra.c` checks the DRED API (deferred processing, `OpusDRED` / decoder byte copies,
argument errors).

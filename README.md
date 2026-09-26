# opusorus

A complete, **pure safe-Rust** port of the [Opus](https://opus-codec.org) audio codec, tracking
**libopus v1.6.1**, verified **bit-exact** against the C reference.

* `#![forbid(unsafe_code)]`, edition 2024, `no_std` + `alloc` (optional `std` for the platform libm).
* Every module was ported function-by-function and differentially tested against libopus 1.6.1
  (compiled from `vendor/libopus` as an oracle): encoder packets, decoder PCM, range-coder state
  and internal state match bit-for-bit.
* Passes the RFC 6716/8251 conformance vectors (all rates, mono/stereo) and the Opus HD vectors,
  the upstream C test suite (through the C ABI), and ~7.5M differential fuzz executions.
* Performance on par with scalar libopus (see [docs/STATUS.md](docs/STATUS.md)).

## Features

| Feature | Cargo feature | Notes |
|---|---|---|
| SILK, CELT, hybrid encoder + decoder, all frame sizes 2.5–120 ms, FEC, DTX, PLC, VBR/CVBR/CBR | default | |
| Multistream, surround (families 0/1/255), projection / ambisonics (families 2/3) | default | |
| Repacketizer, packet padding, packet extensions, packet inspection helpers | default | |
| Opus HD / scalable quality extension (96 kHz) | `qext` | upstream `--enable-qext` |
| Opus Custom modes | `custom-modes` | upstream `--enable-custom-modes` |
| Deep PLC (FARGAN), DRED, OSCE (LACE/NoLACE) + BWE | `deep-plc`, `dred`, `osce` | weights loaded at runtime from a libopus weight blob (`scripts/fetch_dnn_models.sh`) |
| Fixed-point build | `fixed-point`, `fixed-res24` | in progress, see [docs/FIXED_POINT.md](docs/FIXED_POINT.md) |
| `std` platform libm (bit-exact with C on the same platform) | `std` (default) | without it: pure-Rust `libm`, `no_std` |

## Usage

```rust
use opusorus::{Application, Bitrate, Decoder, Encoder};

fn main() -> opusorus::Result<()> {
    let mut enc = Encoder::new(48000, 2, Application::Audio)?;
    enc.set_bitrate(Bitrate::Bits(96_000))?;
    let mut dec = Decoder::new(48000, 2)?;

    let pcm = vec![0i16; 960 * 2]; // 20 ms stereo
    let mut packet = [0u8; 1275];
    let len = enc.encode(&pcm, 960, &mut packet)?;

    let mut out = vec![0i16; 960 * 2];
    let n = dec.decode(Some(&packet[..len]), &mut out, 960, false)?;
    assert_eq!(n, 960);
    Ok(())
}
```

More in `crates/opusorus/examples/` (`roundtrip`, `encode_file`, `decode_file`).

## Workspace

| Crate | Purpose |
|---|---|
| `crates/opusorus` | the codec library |
| `crates/opusorus-capi` | drop-in C ABI (`libopusorus.so`/`.a`, `opus.h` compatible, pkg-config) |
| `crates/opusorus-tools` | `opus_demo`, `opus_compare`, `qext_compare` ports |
| `crates/opusorus-oracle` | test-only FFI to vendored libopus 1.6.1 (the oracle) |
| `crates/opusorus-conformance` | differential tests vs the oracle, conformance vectors |
| `crates/opusorus-bench` | criterion benchmarks: Rust vs C scalar vs C NEON |
| `fuzz/` | cargo-fuzz targets (differential and invariant) |

## Development

```sh
just check          # type-check everything
just test           # unit + differential + libopus-suite tests (default and all features)
just clippy         # clippy -D warnings
just cross          # wasm32, Android, iOS, host, no_std builds
just cross-capi     # C ABI static library for Android/iOS
just test-wasm      # library tests under wasmtime
just vectors        # RFC 8251 + Opus HD conformance vectors
just bench          # benchmarks
just size           # shared-library size comparison
just fuzz-all       # fuzz campaign
```

Docs: [feature list](docs/FEATURES.md) · [tracker](docs/TRACKER.md) · [status, benchmarks, sizes](docs/STATUS.md) ·
[plan & decision log](docs/PLAN.md) · [porting guide](docs/PORTING.md) · [internal API map](docs/INTERNAL_API.md).

## License

BSD-3-Clause, like libopus (this is a derivative work of libopus; see `vendor/libopus/COPYING`).

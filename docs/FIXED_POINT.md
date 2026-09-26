# Fixed-point build (`fixed-point`, `fixed-res24`)

libopus can be built as a fixed-point codec (`./configure --enable-fixed-point`, CMake
`OPUS_FIXED_POINT`, meson `-Dfixed-point=true`), optionally with a 24-bit internal resolution
(`ENABLE_RES24`: autotools' default for fixed-point builds, `--disable-fixed-res24` turns it off;
meson/CMake never define it). The port mirrors this with two cargo features on `opusorus`:

| Feature | libopus | Effect |
|---|---|---|
| `fixed-point` | `FIXED_POINT` | integer CELT (`opus_val16` = `i16`, `opus_val32`/`celt_sig`/`celt_norm`/`celt_glog` = `i32`, Q-format macros), `silk/fixed` encoder analysis, 16-bit `opus_res` |
| `fixed-res24` | `FIXED_POINT` + `ENABLE_RES24` | implies `fixed-point`; `opus_res` = `i32` with `RES_SHIFT` = 8 |
| + `qext` | `ENABLE_QEXT` | `celt_coef` becomes Q31 `i32` (Q15 `i16` otherwise) and the QEXT math |

**These features are not additive: they replace the float implementation**, exactly like the
configure switch (output becomes bit-exact with a fixed-point libopus instead of the float one).
The float API (`opus_encode_float`, ...) stays available, as in upstream's default
(`DISABLE_FLOAT_API` is not ported). Like upstream configure, `fixed-point` cannot be combined
with `deep-plc`, `dred` or `osce` (`compile_error!` / oracle build-script panic) — hence
`--all-features` is not a valid configuration any more; `just check|clippy|test` use the two
"everything on" feature lists `float_all` / `fixed_all` of the `justfile`.

The same features exist on `opusorus-oracle` (the C oracle then compiles `FIXED_POINT=1`,
`SILK_SOURCES_FIXED` + `-I silk/fixed` instead of the float SILK sources, keeps
`OPUS_SOURCES_FLOAT` (analysis + MLP) because the float API is on, and adds `ENABLE_RES24`) and
forwarding features on `opusorus-conformance`, `opusorus-tools`, `opusorus-capi` and
`opusorus-bench`.

## Status

The port is **complete** (FX0–FX5 done). With `fixed-point` / `fixed-res24` the whole codec
is the integer implementation: CELT, SILK (decoder and the `silk/fixed` encoder), the Opus
encoder/decoder, multistream, surround, projection, repacketizer, extensions, the float API and
the tools (`opus_demo`, `opus_compare`, `qext_compare`), with the same public API as the float
build. Verified bit-exact against the fixed oracle in `fixed-point`, `fixed-point` + `qext`,
`fixed-res24`, `fixed-res24` + `qext`, each also with `custom-modes`:

* every differential suite of `opusorus-conformance` (units, CELT/SILK/Opus encoder and
  decoder streams with full state dumps);
* the Rust port of the libopus test suite (`crates/opusorus/tests`: api, decode, encode,
  regressions, padding, extensions, projection, footprint) and the examples, on the host and
  under wasmtime (`wasm32-wasip1`, a 32-bit target with the `OPUS_FAST_INT64 = 0` multiplies);
  the libopus unit tests (`libopus_unit.rs`: celt/tests incl. the `FIXED_POINT` mathops tests,
  `test_simple_matrix`, `test_opus_custom`);
* `opus_demo` byte-identical to the C `opus_demo` linked with the fixed oracle (102
  invocations; 133 with QEXT, including the RFC vectors decoded at 96 kHz), same output files,
  text and exit status;
* the conformance procedure with the fixed decoder (below), and the public-API overflow sweep
  (`api_overflow.rs`).

### Conformance (fixed-point decoder)

RFC 8251 vectors (`tests/run_vectors.sh` procedure: `opus_demo -d <rate> <ch>` +
`opus_compare`), all 12 vectors pass at every rate, mono and stereo. Average quality (%, mono /
stereo; the float decoder's in the last row):

| Build | 48 kHz | 24 kHz | 16 kHz | 12 kHz | 8 kHz |
|---|---|---|---|---|---|
| `fixed-point` | 97.15 / 98.68 | 91.62 / 91.91 | 84.69 / 84.65 | 81.96 / 82.17 | 73.23 / 73.53 |
| `fixed-point` + `qext` | 97.19 / 98.90 | 91.62 / 91.90 | 84.73 / 84.63 | 81.96 / 82.15 | 73.23 / 73.52 |
| `fixed-res24` | 97.36 / 98.88 | 91.63 / 91.91 | 84.69 / 84.66 | 81.95 / 82.16 | 73.23 / 73.52 |
| `fixed-res24` + `qext` | 97.39 / 99.13 | 91.62 / 91.91 | 84.72 / 84.64 | 81.95 / 82.14 | 73.23 / 73.51 |
| float | 97.41 / 99.70 | 91.63 / 91.91 | 84.72 / 84.65 | 81.95 / 82.14 | 73.23 / 73.52 |

Opus HD (`tests/run_opushd_vectors.sh` procedure with `qext_compare`): with `fixed-res24` +
`qext` all 12 RFC vectors decoded at 96 kHz pass (thresholds 0.05 / 0.1 / 0.1). With the
16-bit `fixed-point` + `qext` build, 3 pass (the SILK-only 02–04) and 9 exceed the `rms`
threshold of 0.1 LSB with rms 0.13–0.29: the output has 16-bit resolution, and rounding to 16
bits leaves an rms error of about `1/sqrt(12)` = 0.289 LSB; the 16-bit fixed C `opus_demo` gives
the identical output (asserted). The published `qext_vector*` / `qext_vector*fuzz` files stop
with libopus 1.6.1's own range coder mismatch in every build (C and Rust alike, see
`scripts/run_vectors.sh`). `scripts/run_vectors.sh --fixed` / `--fixed-res24` (with `--hd`)
run upstream's scripts on the fixed-point tools; `--fixed` (all rates) and `--hd --fixed-res24`
(all rates, then the 12 RFC vectors at 96 kHz) give the numbers above, the latter then stopping
at the first `qext_vector` as with every libopus 1.6.1 build.

### Overflow hardening

A Rust library must not panic in debug builds on valid API input. Where libopus overflows
signed integers (undefined behaviour in C, two's-complement wrap in practice) on inputs the
public API accepts, the port makes exactly those operations wrapping (`*_ovflw` /
`wrapping_*`, with a comment citing the C UB), so debug == release == libopus:

* `celt/vq.rs:stereo_itheta` (fixed): a *stereo* encoder with `OPUS_SET_LFE(1)` (the CELT LFE
  band-energy clamp makes `normalise_bands` scale bands far above unit energy): `ADD32` /
  `SUB32` of the channels, the rounding add of `PSHR32` and the `MAC16_16` energy sums wrap.
  The wrapped (negative) energies then reach `celt_sqrt32` / `celt_atan2p_norm` /
  `frac_div32` / `celt_rcp` / `celt_ilog2` outside their `celt_sig_assert` preconditions
  (compiled out in release C): `celt_sqrt32_release` / `celt_atan2p_norm_release`
  (`mathops/fixed.rs`, const-generic `CHECKED = false` variants) compute what release C does,
  and are used only when a sum wrapped, so the checked versions keep their debug assertions.
* `celt/celt_encoder/fixed.rs:alloc_trim_analysis`: the spectral tilt sum (and its products)
  with up-sampled input and the end band above the input bandwidth at high rates (direct CELT
  use; the Opus encoder limits the end band).
* `decoder.rs` decode gain (`fixed-res24`): `MULT32_32_Q16(pcm, gain)` with `OPUS_SET_GAIN`
  near +128 dB does not fit the 32 bits the macro promises: the 64-bit form wraps in its
  `opus_int64` to `opus_val32` conversion (defined), the 32-bit form used on 32-bit targets
  (`OPUS_FAST_INT64 = 0`: armv7, wasm32, x86) overflows its `ADD32`s. `mult32_32_q16_ovflw`
  (`arch/fixed.rs`, both forms, not a libopus macro) wraps, giving the 64-bit form's result on
  every target.

Regression tests (`crates/opusorus-conformance/tests/api_overflow.rs`, every build): stereo LFE
encoders over full-scale signals, bitrates and input formats (1350 packets), direct CELT
encoders at 8–24 kHz with the full end band (1280 packets), decoders at the extreme gains
(972 frames), bit-exact with the C oracle; the two `mult32_32_q16_ovflw` forms agree on 2 M
random and edge inputs. `crates/opusorus/tests/overflow.rs` holds Rust-only versions (stereo
LFE, maximum gain, a random public-API sweep) that also run under wasmtime, i.e. with the
32-bit multiply forms, where the C oracle cannot run: that sweep found the decoder gain site;
after the fix 40 000 cases (`OPUS_TEST_FULL=1 OPUSORUS_API_SWEEP=10`) pass under wasmtime in
`fixed-point`, `fixed-res24`, `fixed-point` + `qext`, `fixed-res24` + `qext`,
`fixed-point` + `custom-modes` and float + `qext`.
`extreme_input_no_panic` feeds ±1e30, ±inf, NaN and ±f32::MAX float samples and full-scale
24-bit samples to every build. The
sweep there (`public_api_sweep`, `OPUSORUS_API_SWEEP=<scale>`) runs random public-API
configurations under `catch_unwind` in the overflow-checked test profile against C: encoders
(every rate, channel count, application, CTL incl. LFE/force-mode/expert frame duration, input
format, extreme signals: full-scale noise and square waves, DC, impulses, Nyquist, clipped float
input), decoders (random gain from -32768 to 32767, complexity, output format, corrupted and
random packets, PLC, FEC), surround multistream encoders (LFE streams) and projection
encoders/decoders. At scale 200 (101 200 cases, 0.54 M encode and 0.80 M decode calls per
build) in the float, `qext`, `fixed-point`, `fixed-res24`, `fixed-point` + `qext` and
`fixed-res24` + `qext` + `custom-modes` builds it finds no other site (before the fix it hit the
stereo LFE one in about 1.4 % of the fixed-point cases; the CELT site is not reachable through
the Opus API).

## Gating scheme

* **Rust modules** not converted yet carry `#[cfg(not(feature = "fixed-point"))]` on their
  declaration (`lib.rs`, `celt/mod.rs`, `silk/mod.rs`); `pub use` re-exports too. A unit that
  converts a module removes that attribute. The crate allows `dead_code`/`unused_imports` without
  `internals` in fixed builds, since converted modules may have no in-crate user yet.
* **Build-specific definitions** of one C name live side by side with the same Rust name:
  `celt/arch/{float,fixed}.rs` and `celt/mathops/{float,fixed}.rs` are selected by
  `celt/arch.rs` / `celt/mathops.rs` (`pub use float::*` or `pub use fixed::*`). Everything else
  uses `#[cfg(feature = "fixed-point")]` / `#[cfg(not(feature = "fixed-point"))]` on the items or
  statements that differ, mirroring the C `#ifdef FIXED_POINT` blocks. **Remove the
  `// FIXED_POINT: not ported (float build)` markers** when porting the branch.
* **Oracle shims** declare the oracle builds they compile in with a marker line in the C file:
  `// oracle-build: float` (the default when there is no marker), `fixed`, or `any`. The Rust
  binding module in `opusorus-oracle/src/lib.rs` gets the matching cfg (float-only modules are
  under `#[cfg(not(feature = "fixed-point"))]`). A shim that must differ per build uses
  `#ifdef FIXED_POINT` inside an `any` file (see `csrc/foundation.c`).
* **Conformance tests** of float-only units start with `#![cfg(not(feature = "fixed-point"))]`;
  fixed-point tests with `#![cfg(feature = "fixed-point")]`; shared ones have per-test cfgs.
  `opusorus/tests/*`, the examples and `opusorus-tools` run in every build (FX5).
* Fixed-point checks are run per package (a workspace-wide `--features
  opusorus-conformance/fixed-point` would unify the reduced `opusorus` into the C ABI crate
  without enabling its own `fixed-point` feature): `cargo clippy -p opusorus -p opusorus-oracle
  -p opusorus-conformance -p opusorus-tools --all-targets --features
  opusorus-conformance/fixed-point`, or workspace-wide with the `fixed_all` list.

## Porting conventions for fixed-point code

* **Typing** (module docs of `celt/arch/fixed.rs`): macro arguments are `impl Into<i32>` and every
  cast of the C expansion is reproduced (`MULT16_16` truncates its arguments to 16 bits, `ADD16`
  truncates the sum, `SHL32` shifts as unsigned, ...). Return types are the C expression types:
  `i32` unless the macro casts (`EXTRACT16`, `ADD16`, `SHL16`, `ROUND16`, `SROUND16`,
  `DIV32_16`, `SATURATE16`, `SAT16`, `QCONST16`, `SIG2WORD16` return `i16`). `SUB16`,
  `MULT16_16_Q15`, `SHR16`, `HALF16`, `NEG16`, `ABS16` return `i32`, as in C.
* **Implicit C narrowing** (an `int` stored in an `opus_val16` variable/array/parameter) is written
  `extract16(x)` in code shared with the float build (identity there) or `x as i16` in
  fixed-only code. Widening for a mixed-width `MIN16`/`MAX16`/`MIN32`/`MAX32` (generic over one
  operand type) is `extend32(x)`.
* **Undefined behaviour in C** (signed overflow in `ADD32`, `SUB32`, `MULT32_32_32`, `PSHR32`,
  `NEG32`, shifts ≥ 32) uses plain Rust operators so debug/test builds panic; `*_ovflw` macros and
  unsigned casts wrap (`wrapping_*`, `as`). A panic in a test therefore usually means the C code
  also overflowed; check the macro domain before "fixing" it. If the overflow is reachable from
  the public API, wrap exactly those operations with a comment citing the C UB and add a
  regression test vs the oracle (see "Overflow hardening").
* **Literals shared between builds**: `OpusVal32::default()` for a zero; per-build helpers
  (`#[cfg]` pairs, as `pulse16` in `celt/cwrs.rs`) for int→value conversions; `qconst16(x, bits)`
  takes an `f64` — for an `f`-suffixed C literal pass `LIT_f32 as f64` (the power-of-two scaling
  is exact in either precision).
* **`OPUS_FAST_INT64`** (`celt::arch::OPUS_FAST_INT64`): the 32-bit forms of `MULT32_32_Q31`,
  `MULT32_32_P31(_ovflw)`, `MULT32_32_Q32` drop partial products and are *not* bit-identical to the
  64-bit forms (the 16×32 and `MULT32_32_Q16` forms are). The port follows C per target
  (x86_64, 64-bit pointers, MIPS → 64-bit forms; armv7, wasm32, x86 → 32-bit forms), so it is
  bit-exact with libopus built for the same target. Tests on a 64-bit host verify both forms
  against C (`fixed_foundation_int32.c` recompiles `fixed_generic.h` with `OPUS_FAST_INT64 0`).
* SILK already uses the 64-bit macro forms everywhere (PLAN D-012, proven identical).
* ARM/MIPS/x86 assembly and intrinsics overrides (`fixed_arm64.h`, `fixed_armv5e.h`,
  `celt/arm/*`, `silk/fixed/{arm,mips,x86}`) are not ported (as for float, PLAN D-002).

## Conversion plan

Every C file whose code depends on `FIXED_POINT` (an `#ifdef` or a type-dependent macro) and the
unit that converts it. Units in one layer are independent and can run in parallel; each unit
removes the gates of its modules, adds `// oracle-build: any|fixed` shims + `#![cfg]`-adapted
tests, and verifies bit-exactness with `fixed-point`, `fixed-res24` and `qext` combinations.

| Layer | Unit | Rust files | C sources (`#ifdef FIXED_POINT` count) | Notes |
|---|---|---|---|---|
| FX0 | `fixed_foundation` ✅ | `celt/arch*.rs`, `celt/mathops*.rs`, `celt/static_modes*.rs`, `celt/cwrs.rs`, `packet.rs` (soft clip `ABS16`) | celt/arch.h (6), fixed_generic.h, float_cast.h (1), mathops.h (8), mathops.c (1), static_modes_fixed.h, kiss_fft.h/mdct.h/modes.h (types), cwrs.c (typed `MAC16_16`) | this unit |
| FX1 | `fixed_fft` ✅ | `celt/kiss_fft.rs`, `celt/mdct.rs`, `celt/mini_kfft.rs` | kiss_fft.c (6), _kiss_fft_guts.h (3), kiss_fft.h (2), mdct.c (8), mdct.h (1); mini_kfft.c is float-only (give it its own float complex type instead of `KissFftCpx`, then un-gate for `qext_compare`) | `opus_fft_impl(downshift)`, `scale_shift`, `S_MUL`/`C_MUL` Q15/Q31 twiddles. Done: modules un-gated, `celt_fft` shim `any`, `celt_fft.rs` runs in every fixed config (+ `custom-modes`) |
| FX1 | `fixed_modes` ✅ | `celt/modes.rs`, `celt/rate.rs` | modes.c (2: custom-mode window/preemph), rate.c (1: QEXT) | custom modes compute `celt_coef` windows; `clt_compute_allocation` is integer |
| FX1 | `fixed_pitch_lpc` ✅ | `celt/pitch.rs`, `celt/celt_lpc.rs` | pitch.c (26), pitch.h (4), celt_lpc.c (11), celt_lpc.h | `xcorr_kernel`, `pitch_downsample` shifts, `_celt_lpc` Q-scaling, `celt_fir`/`celt_iir`. Done: modules un-gated, shim `any` (+ `celt_pitch_lpc_int32.c`, `fixed`), tests `celt_pitch_lpc_fixed.rs`; fixed `celt_pitch_xcorr` returns `maxcorr`, fixed `find_best_pitch` takes `yshift, maxcorr`, `_celt_lpc_fast_int64::<bool>` exposes both multiply forms |
| FX1 | `fixed_silk_shared` ✅ | `silk/decoder.rs`, `silk/plc.rs`, `silk/cng.rs`, `silk/nsq*.rs`, `silk/vad.rs`, `silk/encoder_common.rs`, `silk/sigproc.rs` | dec_API.c (`opus_res` output), sort.c (1: `silk_insertion_sort_decreasing_int16`), HP_variable_cutoff.c (1: include), LPC_analysis_filter.c (2: `USE_CELT_FIR` = 0, no change) | already compile; make `silk_decoder`/`silk_encoder_common` shims `any` (`opus_res` = `short`/`int`) and run their tests in fixed builds |
| FX1 | `fixed_packet` ✅ | `mapping_matrix.rs` | src/mapping_matrix.c (4) | integer matrix multiply paths |
| FX2 | `fixed_bands` ✅ | `celt/vq.rs`, `celt/quant_bands.rs`, `celt/bands.rs`, `celt/celt.rs` | vq.c (25), vq.h (2), quant_bands.c (12), quant_bands.h (1), bands.c (19), bands.h, celt.c (4), celt.h | Done: modules un-gated, `celt_bands` shim `any`, `celt_bands.rs` runs in every fixed config. `vq.rs` exports `norm_scaleup`/`norm_scaledown`/`celt_inner_prod_norm(_shift)` (no-ops/`celt_inner_prod` in float) and the `v16`/`v32`/`qc16`/`qc32` int/literal helpers; `quant_bands::E_MEANS` is `[i8; 25]` Q4 in fixed builds (used by `rate.rs` too); `op_pvq_search` returns `opus_val16`; `opus_get_version_string` has the `-fixed` suffix; `opus_custom_mode_create_custom` is available in fixed builds |
| FX2 | `fixed_silk_encoder` ✅ | new `silk/fixed.rs` (+ `silk/fixed/*.rs`), `silk/encoder.rs` | silk/fixed/*.c (23 files, `SILK_SOURCES_FIXED`), main_FIX.h, structs_FIX.h, enc_API.c (1), init_encoder.c (1), control_codec.c (4) | replaces `silk/float.rs` in fixed builds; needs `fixed_pitch_lpc` (`celt_pitch_xcorr`) |
| FX2 | `fixed_analysis` ✅ | `analysis.rs` (`mlp.rs` unchanged: float) | src/analysis.c (7) | float analysis fed by fixed PCM (`downmix`, `celt_inner_prod` scaling) |
| FX3 | `fixed_celt_decoder` ✅ | `celt/celt_decoder.rs` | celt_decoder.c (9) | Done: module un-gated, `celt_decoder` shim `any` (the C encoder of the oracle makes the packets from `FLOAT2RES`-converted signals; state dumps widened to `opus_int32`), `celt_decoder.rs` runs in every fixed config (+ `qext`, `custom-modes`). Shared code uses the `arch` macros; `#[cfg]` pairs for the pitch-PLC branches (autocorr noise floor/lag window, LPC bandwidth expansion, decay shift, `SIG_SAT` after `celt_iir`, explosion test), the post-filter gain and `opus_custom_decode`/`decode24`/`decode_float` (direct `opus_res` output for the matching resolution, else `RES2INT16`/`RES2INT24`/`RES2FLOAT`) |
| FX3 | `fixed_celt_encoder` ✅ | `celt/celt_encoder.rs` | celt_encoder.c (30) | preemphasis, transient analysis, tf/dynalloc in fixed; needs FX1+FX2 |
| FX4 | `fixed_opus_decoder` ✅ | `decoder.rs`, `ms_decoder.rs`, `projection_decoder.rs` | opus_decoder.c (9), opus_multistream_decoder.c (1), opus_projection_decoder.c (1) | Done: modules and `Decoder`/`MsDecoder`/`ProjectionDecoder` re-exports un-gated, `opus_decoder` shim `any` (no `softclip_mem` in fixed: dumped as zeros), `opus_decoder.rs` runs in every fixed config (+ `qext`, `custom-modes`) incl. the RFC 8251 vectors (bit-exact, `.bit` final ranges, `opus_compare`). `opus_res` outputs (`RES2INT16/24/FLOAT`; direct `opus_decode` / `opus_decode24` for the matching resolution), `smooth_fade` 16-bit/RES24 forms, fixed decode gain (RES24 keeps C's `SATURATE(x, 32767)` bound), no soft clip (`OPTIONAL_CLIP` 0) |
| FX4 | `fixed_opus_encoder` ✅ | `encoder.rs`, `ms_encoder.rs`, `projection_encoder.rs` | opus_encoder.c (16), opus_multistream_encoder.c (2), opus_projection_encoder.c | Done: modules un-gated, `opus_encoder` shim `any` (renames the non-static fixed `silk_biquad_res`), `opus_encoder.rs` runs in every fixed config (+ `qext`, `custom-modes`). `#[cfg]` pairs for `silk_biquad_res`/`dc_reject`/`stereo_fade`/`gain_fade`/`compute_stereo_width`/`compute_frame_energy`/`logSum`, `hp_cutoff` uses `silk_biquad_alt_stride1/2` with 16-bit `opus_res`, analysis from complexity 10, Q15 HB/stereo gains, Q24 surround masking; `opus_encode` (res16) / `opus_encode24` (res24) pass the input through. Stereo streams with `OPUS_SET_LFE(1)` overflow in `stereo_itheta` (C UB): wrapping since FX5 (the `opus_encoder.rs` suite still keeps LFE on mono encoders in fixed builds) |
| FX5 | `fixed_integration` ✅ (`fx5_api` part) | `lib.rs` docs, `opusorus/tests`, examples, `opusorus-tools`, overflow hardening (`celt/vq.rs`, `celt/mathops/fixed.rs`, `celt/celt_encoder/fixed.rs`), `opusorus-capi`, `opusorus-bench` | — | `fx5_api`: public tests + examples un-gated (all fixed configs, wasmtime), `opus_demo`/`qext_compare` un-gated, C `opus_demo` comparison vs the fixed oracle, RFC 8251 + Opus HD procedure with the fixed decoder, public-API overflow sweep + fixes, docs. C ABI and benchmarks: separate FX5 parts |

Not ported (upstream-specific): `celt/fixed_debug.h` (`FIXED_DEBUG`, a checking variant of the
same macros), `celt/opus_custom_demo.c`, `celt/dump_modes/*`, the `arm`/`mips`/`x86` overrides.

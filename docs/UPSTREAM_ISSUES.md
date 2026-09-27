# Upstream libopus 1.6.1 issues found while porting

Found by differential testing and fuzzing against the C oracle. The Rust port reproduces
behaviour that is well defined (and, where C has undefined behaviour that wraps in practice, wraps
identically — see docs/FIXED_POINT.md "overflow hardening"); the cases below are either
reproduced faithfully (quirks) or avoided by the harness (memory-safety bugs, which cannot occur in
safe Rust: the port returns an error or behaves as documented instead).

## Quirks reproduced bit-exactly

| Area | Description |
|---|---|
| `ENABLE_RES24` decoder gain | `opus_decode_frame` applies `SATURATE(x, 32767)` to Q8 (24-bit) samples after the decode gain, clamping to ±1/256 of full scale when `OPUS_SET_GAIN` is large. |
| fixed-point surround | `logSum` returns `opus_val16`, truncating its Q24 result; `surround_analysis` adds a Q10 `channel_offset` to Q24 masks without scaling. |
| Static CELT FFT tables (Q15) | differ from runtime `celt_cos_norm` twiddles by up to 2 LSB (static tables are what the default modes use). |
| `test_opus_custom.c` | `CUSTOM_MODEES` typo means the test only exercises 48/96 kHz custom modes. |
| `CUSTOM_MODES` `compute_ebands` | the `celt_assert`s on the band layout fail for e.g. 12000 Hz / 52, 104, 208 samples, so `opus_custom_mode_create` aborts (hardened or assertion builds) instead of returning an error; the port checks them only with `assertions` (panic), otherwise it creates the mode. |

## Undefined behaviour in C (the port wraps like C in practice)

| Area | Description |
|---|---|
| fixed-point stereo `OPUS_SET_LFE(1)` | CELT LFE band-energy clamp makes `normalise_bands` `SHL32` wrap; `stereo_itheta` sums overflow `opus_val32`. |
| fixed-point CELT `alloc_trim_analysis` | spectral-tilt sum overflows with up-sampled input + full end band (direct CELT use). |
| `ENABLE_RES24` `OPUS_SET_GAIN` near +128 dB | `MULT32_32_Q16` exceeds 32 bits; the 32-bit multiply form (armv7/wasm32/x86) overflows its adds. |
| fixed-point `FUZZING` surround (LFE stream) | the random allocation decisions put an LFE band far above unit energy; the `opus_val32` energy of `op_pvq_search_c` wraps and `celt_ilog2` gets a negative value (with `ENABLE_ASSERTIONS`: `celt_sig_assert(x>0)` aborts). |
| fixed-point `ENABLE_ASSERTIONS` | the stereo-LFE wrap above fails `celt_sig_assert(x>0)` (`celt_ilog2` via `celt_sqrt32`): libopus aborts on that valid API input. |
| fixed-point 24-bit input outside ±2^23 | `INT24TORES`/`INT24TOSIG`, `downmix_int24`, `silk_resampler_down2_hp` overflow (out-of-contract input; not hardened). |

## Memory-safety bugs (C only; unreachable through the Opus encoder/decoder API)

Found by the `differential_custom` fuzz target (Opus Custom + QEXT):

1. QEXT with 90-sample short-MDCT layouts (48000/720, 96000/1440): `quant_all_bands` points its scratch at the last, narrower band of `qext_eBands_180` and writes past the end of `X`.
2. QEXT with overlap 240 at a rate other than 96 kHz: `comb_filter_qext` reads twice the allocated filter history.
3. QEXT custom frames above 1024 samples: `celt_decode_lost` reads before `decode_mem`.
4. QEXT: a main end band of 2 or 14 is taken for the QEXT pass (`extra_bands`); `cubic_quant_partition` then divides by zero on 1-bin bands.
5. `special_hybrid_folding` with start band 17 in custom band layouts copies out of `_norm`.
6. Signalled CBR encode into a 2-byte buffer writes one byte past the buffer.
7. `opus_fft_alloc_twiddles` failure path (e.g. custom mode 8000/44, unfactorable FFT size) frees uninitialized pointers (also `opus_custom_mode_create`'s `failure:` path).
8. Stereo encoding with fewer than 13 effective bands: `stereo_analysis` reads uninitialized `X` (non-deterministic packets).
9. An end band below 3 makes dynalloc read out of bounds; an end band beyond `effEBands` reads past `X`.
10. Inputs that break `start < end`, start bands other than 0/17, and NaN input hit `celt_assert`s (abort with `ENABLE_HARDENING`).

11. Fixed point, Opus Custom 32000 Hz / 640 samples, mono (`celt_encoder` test `custom_api`):
    the C encoder's first packet differs between otherwise identical runs depending on the
    memory layout of the test binary (e.g. it changes when the Rust side is built with
    `CARGO_INCREMENTAL=0`), while the Rust output stays the same — the signature of a read of
    uninitialized memory in C, like 8 and 9 (not located further: valgrind cannot run the test
    binary on the aarch64 host). The test passes in the default build configuration.

Also: `opus_packet_*` on empty packets reads `data[0]` (the port returns `BadArg`/`InvalidPacket`),
and `clt_compute_extra_allocation` reads an uninitialised `follower` entry with fewer than 5 bands
(the port zero-initialises).

## Build-system issues

| Issue | Workaround |
|---|---|
| CMake `OPUS_FIXED_POINT=ON` on AArch64 defines the ARMv7 asm switches `OPUS_ARM_MAY_HAVE_NEON`/`PRESUME_NEON`, so `celt_pitch_xcorr_neon` is referenced but not built (link error). | `-U` both switches (as autotools/meson do on AArch64). |
| Published Opus HD `qext_vector*` bitstreams stop with a range-coder mismatch in libopus 1.6.1 itself. | Only the RFC vectors at 96 kHz are required to pass `qext_compare`. |

# Port tracker

Per-unit status. A unit owns exactly the listed Rust files (+ its oracle/test files `crates/opusorus-oracle/{src,csrc}/<unit>.*`, `crates/opusorus-conformance/tests/<unit>.rs`).

Status: ⬜ pending · 🟨 in progress · ✅ bit-exact vs oracle

## Foundation (phase 0)

| Rust file | C source | Status | Tests |
|---|---|---|---|
| celt/entcode.rs, entenc.rs, entdec.rs | celt/entcode.c, entenc.c, entdec.c, mfrngcod.h | ✅ | foundation.rs: 3000 random op sequences enc+dec, garbage decode, shrink/patch — bit-exact |
| celt/arch.rs | celt/arch.h (float) | ✅ | used by all |
| celt/mathops.rs | celt/mathops.c/.h, float_cast.h (float) | ✅ | foundation.rs: 200k random inputs per fn — bit-exact |
| silk/macros.rs | silk/macros.h, SigProc_FIX.h, Inlines.h, typedef.h | ✅ | unit tests (64/32-bit form equivalence) |
| silk/define.rs, tuning_parameters.rs, errors.rs | silk/define.h, tuning_parameters.h, errors.h | ✅ | generated |
| math.rs, error.rs, constants.rs | — | ✅ | unit tests |
| celt/arch/fixed.rs, celt/mathops/fixed.rs, celt/static_modes_fixed.rs (+ shared types in static_modes.rs, arch.rs, mathops.rs) | celt/arch.h + fixed_generic.h (fixed), mathops.h/.c (fixed), float_cast.h (fixed), static_modes_fixed.h | ✅ | fixed_foundation.rs (unit `fixed_foundation`, features fixed-point / fixed-res24 × qext): 95 macros/conversions (+ both `OPUS_FAST_INT64` forms of the 8 32-bit multiplies), 27 fixed mathops fns, float API, constants, static modes field-by-field, range coder/CWRS/Laplace — bit-exact |

## Units

| Layer | Unit | Rust files | C sources | Status | Tests / notes |
|---|---|---|---|---|---|
| A | `celt_fft` | `celt/kiss_fft.rs`<br>`celt/mdct.rs`<br>`celt/mini_kfft.rs` | celt/kiss_fft.c, celt/kiss_fft.h, celt/_kiss_fft_guts.h<br>celt/mdct.c, celt/mdct.h<br>celt/mini_kfft.c (qext) | ✅ | celt_fft.rs: FFT/IFFT all static states, MDCT fwd/bwd all shifts & strides, runtime twiddles (custom-modes), mini_kfft (qext) — bit-exact |
| A | `celt_modes` | `celt/modes.rs`<br>`celt/static_modes.rs`<br>`celt/laplace.rs`<br>`celt/cwrs.rs`<br>`celt/rate.rs` | celt/modes.c, celt/modes.h<br>celt/static_modes_float.h<br>celt/laplace.c<br>celt/cwrs.c<br>celt/rate.c, celt/rate.h | ✅ | celt_modes.rs: static mode tables verified field-by-field, laplace, cwrs all (n,k), clt_compute_allocation enc+dec all configs, custom mode creation — bit-exact |
| A | `celt_pitch_lpc` | `celt/pitch.rs`<br>`celt/celt_lpc.rs` | celt/pitch.c, celt/pitch.h<br>celt/celt_lpc.c, celt/celt_lpc.h | ✅ | celt_pitch_lpc.rs: xcorr, inner prods, pitch_downsample/search, remove_doubling, lpc/fir/iir/autocorr — bit-exact |
| A | `silk_common` | `silk/tables.rs`<br>`silk/structs.rs`<br>`silk/sigproc.rs`<br>`silk/nlsf.rs`<br>`silk/coding.rs` | silk/tables*.c, silk/table_LSF_cos.c, silk/pitch_est_tables.c, silk/pitch_est_defines.h<br>silk/structs.h, silk/control.h<br>silk/{lin2log,log2lin,sigm_Q15,sort,bwexpander,bwexpander_32,inner_prod_aligned,sum_sqr_shift,interpolate,biquad_alt,LP_variable_cutoff,ana_filt_bank_1,LPC_inv_pred_gain,LPC_fit,LPC_analysis_filter}.c<br>silk/{NLSF2A,A2NLSF,NLSF_decode,NLSF_unpack,NLSF_stabilize,NLSF_VQ_weights_laroia,NLSF_VQ,NLSF_encode,NLSF_del_dec_quant}.c<br>silk/{code_signs,shell_coder,gain_quant,decode_pulses,encode_pulses,decode_pitch,stereo_decode_pred,stereo_encode_pred}.c | ✅ | silk_common.rs: every table vs C, sigproc, NLSF enc/dec (NB/MB/WB), pulses/shell/signs/gains/pitch/stereo coding — bit-exact |
| A | `silk_resampler` | `silk/resampler.rs` | silk/resampler*.c, silk/resampler_rom.c, silk/resampler_*.h | ✅ | silk_resampler.rs: every Fs pair enc+dec (incl. qext 96k), streamed random chunks — bit-exact |
| A | `opus_packet` | `packet.rs`<br>`extensions.rs`<br>`repacketizer.rs`<br>`mapping_matrix.rs`<br>`mlp.rs`<br>`multistream.rs` | src/opus.c (packet parsing, soft clip, TOC helpers), src/opus_private.h helpers<br>src/extensions.c<br>src/repacketizer.c (incl. opus_packet_pad/unpad, multistream pad/unpad)<br>src/mapping_matrix.c, src/mapping_matrix.h<br>src/mlp.c, src/mlp_data.c, src/mlp.h<br>src/opus_multistream.c (shared multistream layout helpers) | ✅ | opus_packet.rs: parse (+self-delimited) fuzz, soft clip, repacketizer, pad/unpad, extensions, mapping matrix, MLP — bit-exact |
| B | `celt_bands` | `celt/vq.rs`<br>`celt/quant_bands.rs`<br>`celt/bands.rs`<br>`celt/celt.rs` | celt/vq.c, celt/vq.h<br>celt/quant_bands.c, celt/quant_bands.h<br>celt/bands.c, celt/bands.h<br>celt/celt.c, celt/celt.h | ✅ | celt_bands.rs: vq/pvq/cubic, energy quant enc/dec over 3-frame chains, quant_all_bands enc+dec 30k cases (+15k qext) incl. theta RDO — bit-exact |
| B | `silk_decoder` | `silk/decoder.rs`<br>`silk/plc.rs`<br>`silk/cng.rs` | silk/{dec_API,init_decoder,decoder_set_fs,decode_frame,decode_core,decode_indices,decode_parameters,stereo_MS_to_LR}.c<br>silk/PLC.c, silk/PLC.h<br>silk/CNG.c | ✅ | silk_decoder.rs: real SILK packets from C encoder, all API rates, PLC, FEC/LBRR, CNG, stereo, garbage — bit-exact |
| B | `silk_encoder_common` | `silk/nsq.rs`<br>`silk/nsq_del_dec.rs`<br>`silk/vad.rs`<br>`silk/encoder_common.rs` | silk/NSQ.c, silk/NSQ.h<br>silk/NSQ_del_dec.c<br>silk/VAD.c<br>silk/{encode_indices,process_NLSFs,quant_LTP_gains,VQ_WMat_EC,stereo_LR_to_MS,stereo_find_predictor,stereo_quant_pred,HP_variable_cutoff,control_SNR,control_audio_bandwidth,check_control_input}.c | ✅ | silk_encoder_common.rs: NSQ + NSQ_del_dec multi-frame state, VAD, encode_indices bytes, NLSF/LTP quant, stereo enc — bit-exact |
| B | `analysis` | `analysis.rs` | src/analysis.c, src/analysis.h | ✅ | analysis.rs: streamed signals all rates, AnalysisInfo + state — bit-exact |
| C | `celt_decoder` | `celt/celt_decoder.rs` | celt/celt_decoder.c | ✅ | celt_decoder.rs: all LM/ch/rates/budgets, hybrid shared ec, downsample, PLC noise+pitch, corrupt input, custom modes, qext 96k — bit-exact (PCM, rng, full state); release perf 0.92x C time |
| C | `celt_encoder` | `celt/celt_encoder.rs` | celt/celt_encoder.c | ✅ | celt_encoder.rs: all LM/ch/bitrates/VBR modes/complexity/lsb depth/lfe/masks/hybrid, qext — bytes + rng bit-exact |
| C | `silk_encoder_flp` | `silk/float.rs`<br>`silk/encoder.rs` | silk/float/*.c, silk/float/*.h (may add submodules under silk/float/)<br>silk/{enc_API,init_encoder,control_codec}.c | ✅ | silk_encoder_flp.rs: silk_Encode all rates/internal bw/packet sizes/complexity/CBR-VBR/FEC/DTX/stereo — bytes + rng bit-exact |
| D | `opus_decoder` | `decoder.rs`<br>`ms_decoder.rs`<br>`projection_decoder.rs` | src/opus_decoder.c<br>src/opus_multistream_decoder.c<br>src/opus_projection_decoder.c | ✅ | opus_decoder.rs: 180 C-encoded streams all apps/rates/frame sizes/transitions, FEC/PLC/DTX, garbage, extensions, multistream 0/1/255, projection 2/3, RFC 8251 vectors all rates bit-exact + opus_compare pass, Opus HD vectors (qext) — bit-exact; release ≈0.86–0.94× C time |
| D | `opus_encoder` | `encoder.rs`<br>`ms_encoder.rs`<br>`projection_encoder.rs` | src/opus_encoder.c<br>src/opus_multistream_encoder.c<br>src/opus_projection_encoder.c | ✅ | opus_encoder.rs: long streams over all apps/rates/bitrates/VBR modes/complexity/frame durations/FEC/DTX/ctl switches, multistream+surround, projection, qext — bytes+rng bit-exact |

## Fixed-point units (docs/FIXED_POINT.md)

| Layer | Unit | Status |
|---|---|---|
| FX0 | `fixed_foundation` (arch/fixed_generic macros, fixed mathops, static modes, gating, fixed oracle) | ✅ |
| FX1 | `fixed_fft`, `fixed_modes`, `fixed_pitch_lpc`, `fixed_silk_shared`, `fixed_packet` | ⬜ |
| FX2 | `fixed_bands`, `fixed_silk_encoder`, `fixed_analysis` | ⬜ |
| FX3 | `fixed_celt_decoder`, `fixed_celt_encoder` | ⬜ |
| FX4 | `fixed_opus_decoder`, `fixed_opus_encoder` | ⬜ |
| FX5 | `fixed_integration` (API, tools, C ABI, vectors, benches) | ⬜ |

## DNN units

| Unit | Rust files | Status | Tests |
|---|---|---|---|
| `dnn_core` | `dnn/*` (nnet, weights blob parse, vec math, burg, freq, pitchdnn, lpcnet_enc, nndsp, kiss99) | ✅ | dnn_core.rs (features deep-plc/osce/dred): layers + real model weights via blob — bit-exact |
| `dnn_plc` | `dnn/fargan.rs`, `dnn/lpcnet_plc.rs` | ✅ | dnn_plc.rs: FARGAN + neural PLC sequences vs C — bit-exact |
| `dnn_dred` | `dnn/dred_*.rs` | ✅ | dnn_dred.rs: RDOVAE enc/dec, latents, payload bytes, dred_ec_decode — bit-exact |
| `dnn_osce` | `dnn/osce.rs`, `dnn/osce_features.rs` | ✅ | dnn_osce.rs: LACE/NoLACE/BBWENet + features — bit-exact |
| `dnn_integration` | decoder/encoder/celt_decoder/silk decoder hooks, `Decoder::dred_decode*`, DRED parse/process, DNN blob CTLs | ✅ | dnn_integration.rs: 11k decode calls (neural PLC, LACE/NoLACE, BWE), 1259 DRED packets bytes-identical, 2000 DRED parses — bit-exact; all opus_* suites pass with DNN features |

## Infrastructure

| opusorus-tools: opus_compare / qext_compare port (lib + CLIs) | ✅ | tools_compare.rs: quality numbers identical to C on RFC/HD vectors |

| Item | Status | Notes |
|---|---|---|
| Oracle crate (libopus 1.6.1 via cc) | ✅ | float, no intrinsics, -ffp-contract=off, qext/custom-modes features |
| Conformance crate | ✅ | shared Rng, signal generators, bit-exact asserts |
| justfile (check/clippy/test/cross/test-wasm/vectors/bench/size/fuzz) | ✅ | |
| RFC 8251 + Opus HD vectors (`tests/vectors.rs`, `scripts/run_vectors.sh`) | ✅ | bit-exact, same quality numbers as C |
| C ABI crate `opusorus-capi` (libopusorus.so/.a) | ✅ | upstream C tests all pass; `just cross-capi` builds staticlib for Android/iOS |
| Fuzz targets (`fuzz/`) | ✅ | 11 targets, ~7.5M execs, 0 findings; DNN/custom-mode fuzz targets are follow-ups |
| Benchmarks `opusorus-bench` (criterion, Rust vs C scalar vs C NEON) | ✅ | see STATUS |
| Size report (`scripts/size_report.sh`) | ✅ | see STATUS |
| opus_demo port (`opusorus-tools`) | ✅ | byte-identical to C |
| Rust port of libopus tests (`crates/opusorus/tests`, `libopus_unit.rs`) | ✅ | host + wasm |

## Performance log

| Date | Item | Result |
|---|---|---|
| 2026-09-26 | Full benchmark suite (layer E) | Rust ≈ C scalar (0.87–1.05×), SILK decode 1.39×; vs C NEON 1.07–1.37× |
| 2026-09-26 | CELT decoder (layer C) | 0.92× C time |
| 2026-09-26 | Opus decoder (layer D) | 0.86–0.94× C time |

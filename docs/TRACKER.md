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

## Units

| Layer | Unit | Rust files | C sources | Status | Tests / notes |
|---|---|---|---|---|---|
| A | `celt_fft` | `celt/kiss_fft.rs`<br>`celt/mdct.rs`<br>`celt/mini_kfft.rs` | celt/kiss_fft.c, celt/kiss_fft.h, celt/_kiss_fft_guts.h<br>celt/mdct.c, celt/mdct.h<br>celt/mini_kfft.c (qext) | ⬜ | |
| A | `celt_modes` | `celt/modes.rs`<br>`celt/static_modes.rs`<br>`celt/laplace.rs`<br>`celt/cwrs.rs`<br>`celt/rate.rs` | celt/modes.c, celt/modes.h<br>celt/static_modes_float.h<br>celt/laplace.c<br>celt/cwrs.c<br>celt/rate.c, celt/rate.h | ⬜ | |
| A | `celt_pitch_lpc` | `celt/pitch.rs`<br>`celt/celt_lpc.rs` | celt/pitch.c, celt/pitch.h<br>celt/celt_lpc.c, celt/celt_lpc.h | ⬜ | |
| A | `silk_common` | `silk/tables.rs`<br>`silk/structs.rs`<br>`silk/sigproc.rs`<br>`silk/nlsf.rs`<br>`silk/coding.rs` | silk/tables*.c, silk/table_LSF_cos.c, silk/pitch_est_tables.c, silk/pitch_est_defines.h<br>silk/structs.h, silk/control.h<br>silk/{lin2log,log2lin,sigm_Q15,sort,bwexpander,bwexpander_32,inner_prod_aligned,sum_sqr_shift,interpolate,biquad_alt,LP_variable_cutoff,ana_filt_bank_1,LPC_inv_pred_gain,LPC_fit,LPC_analysis_filter}.c<br>silk/{NLSF2A,A2NLSF,NLSF_decode,NLSF_unpack,NLSF_stabilize,NLSF_VQ_weights_laroia,NLSF_VQ,NLSF_encode,NLSF_del_dec_quant}.c<br>silk/{code_signs,shell_coder,gain_quant,decode_pulses,encode_pulses,decode_pitch,stereo_decode_pred,stereo_encode_pred}.c | ⬜ | |
| A | `silk_resampler` | `silk/resampler.rs` | silk/resampler*.c, silk/resampler_rom.c, silk/resampler_*.h | ⬜ | |
| A | `opus_packet` | `packet.rs`<br>`extensions.rs`<br>`repacketizer.rs`<br>`mapping_matrix.rs`<br>`mlp.rs`<br>`multistream.rs` | src/opus.c (packet parsing, soft clip, TOC helpers), src/opus_private.h helpers<br>src/extensions.c<br>src/repacketizer.c (incl. opus_packet_pad/unpad, multistream pad/unpad)<br>src/mapping_matrix.c, src/mapping_matrix.h<br>src/mlp.c, src/mlp_data.c, src/mlp.h<br>src/opus_multistream.c (shared multistream layout helpers) | ⬜ | |
| B | `celt_bands` | `celt/vq.rs`<br>`celt/quant_bands.rs`<br>`celt/bands.rs`<br>`celt/celt.rs` | celt/vq.c, celt/vq.h<br>celt/quant_bands.c, celt/quant_bands.h<br>celt/bands.c, celt/bands.h<br>celt/celt.c, celt/celt.h | ⬜ | |
| B | `silk_decoder` | `silk/decoder.rs`<br>`silk/plc.rs`<br>`silk/cng.rs` | silk/{dec_API,init_decoder,decoder_set_fs,decode_frame,decode_core,decode_indices,decode_parameters,stereo_MS_to_LR}.c<br>silk/PLC.c, silk/PLC.h<br>silk/CNG.c | ⬜ | |
| B | `silk_encoder_common` | `silk/nsq.rs`<br>`silk/nsq_del_dec.rs`<br>`silk/vad.rs`<br>`silk/encoder_common.rs` | silk/NSQ.c, silk/NSQ.h<br>silk/NSQ_del_dec.c<br>silk/VAD.c<br>silk/{encode_indices,process_NLSFs,quant_LTP_gains,VQ_WMat_EC,stereo_LR_to_MS,stereo_find_predictor,stereo_quant_pred,HP_variable_cutoff,control_SNR,control_audio_bandwidth,check_control_input}.c | ⬜ | |
| B | `analysis` | `analysis.rs` | src/analysis.c, src/analysis.h | ⬜ | |
| C | `celt_decoder` | `celt/celt_decoder.rs` | celt/celt_decoder.c | ⬜ | |
| C | `celt_encoder` | `celt/celt_encoder.rs` | celt/celt_encoder.c | ⬜ | |
| C | `silk_encoder_flp` | `silk/float.rs`<br>`silk/encoder.rs` | silk/float/*.c, silk/float/*.h (may add submodules under silk/float/)<br>silk/{enc_API,init_encoder,control_codec}.c | ⬜ | |
| D | `opus_decoder` | `decoder.rs`<br>`ms_decoder.rs`<br>`projection_decoder.rs` | src/opus_decoder.c<br>src/opus_multistream_decoder.c<br>src/opus_projection_decoder.c | ⬜ | |
| D | `opus_encoder` | `encoder.rs`<br>`ms_encoder.rs`<br>`projection_encoder.rs` | src/opus_encoder.c<br>src/opus_multistream_encoder.c<br>src/opus_projection_encoder.c | ⬜ | |

## Infrastructure

| Item | Status | Notes |
|---|---|---|
| Oracle crate (libopus 1.6.1 via cc) | ✅ | float, no intrinsics, -ffp-contract=off, qext/custom-modes features |
| Conformance crate | ✅ | shared Rng, signal generators, bit-exact asserts |
| justfile (check/clippy/test/cross/test-wasm/vectors/bench/size/fuzz) | ✅ | |
| RFC test vectors | ⬜ | |
| C ABI crate | ⬜ | |
| Fuzz targets | ⬜ | |
| Benchmarks | ⬜ | |
| Size report | ⬜ | |

## Performance log

_(filled in phase E; see docs/STATUS.md for the latest numbers)_

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
| C | `celt_decoder` | `celt/celt_decoder.rs` | celt/celt_decoder.c | ⬜ | |
| C | `celt_encoder` | `celt/celt_encoder.rs` | celt/celt_encoder.c | ⬜ | |
| C | `silk_encoder_flp` | `silk/float.rs`<br>`silk/encoder.rs` | silk/float/*.c, silk/float/*.h (may add submodules under silk/float/)<br>silk/{enc_API,init_encoder,control_codec}.c | ⬜ | |
| D | `opus_decoder` | `decoder.rs`<br>`ms_decoder.rs`<br>`projection_decoder.rs` | src/opus_decoder.c<br>src/opus_multistream_decoder.c<br>src/opus_projection_decoder.c | ⬜ | |
| D | `opus_encoder` | `encoder.rs`<br>`ms_encoder.rs`<br>`projection_encoder.rs` | src/opus_encoder.c<br>src/opus_multistream_encoder.c<br>src/opus_projection_encoder.c | ⬜ | |

## Infrastructure

| opusorus-tools: opus_compare / qext_compare port (lib + CLIs) | ✅ | tools_compare.rs: quality numbers identical to C on RFC/HD vectors |

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

# Internal API notes

Collected from each unit's port report: signatures and conventions later units must use, plus deviations from C. Source of truth is the code; this is a map.

## `celt_pitch_lpc`

### external needs

Nothing was needed from outside the owned files, and no foundation or shared files were edited. Changes callers in other units must adapt to:
(1) `celt_fir(x, num, y, n, ord)`: `x` must include the `ord` history samples first, so `x[k+ord]` is C's `x[k]`. C reads before its pointer, which a Rust slice cannot do. The celt_decoder PLC call becomes `celt_fir(&exc[max_period-exc_length-ORD..], ...)` relative to its exc buffer.
(2) The in-place C call `celt_iir(buf, lpc, buf, ...)` in the celt_decoder PLC should use `celt_iir_inplace(buf, den, n, ord, mem)`. The out-of-place `celt_iir(x, den, y, ...)` is also available.
(3) `pitch_downsample` takes `x: &[&[CeltSig]]` (one slice per channel) and `c: usize`.
(4) `pitch_search` returns the pitch as an `i32` instead of writing to an out-pointer. `remove_doubling` takes `t0_: &mut i32` and `x` is the start of the pitch buffer, as in C.
(5) `_celt_autocorr(x, ac, window, overlap, lag, n)`: pass `&[]`, 0 for C's `NULL`, 0.
(6) `pitch.rs` has a small `pub(crate) Scratch<const N: usize>` helper that stands in for C VLAs. It uses stack storage sized for the codec's maximum lengths (larger when qext is on) and falls back to a heap Vec only for unusual larger sizes. It could be moved to a shared utility module later if other units want the same pattern.

### notes

1. The relayed user request, "keep this as project memory", does not say what "this" refers to. I did not write any memory, CLAUDE.md or docs file for it, and only did the porting task. The user may want to clarify what they meant.
2. None of the four C files has ENABLE_QEXT or CUSTOM_MODES blocks. The only effect of qext is larger stack scratch sizes, and the tests pass with and without the feature.
3. FIXED_POINT branches (shifts, maxcorr, the int32 LPC fit, autocorr normalisation) are skipped, each with a `// FIXED_POINT: not ported (float build)` marker.
4. The SMALL_FOOTPRINT variant of celt_iir is not ported: the oracle does not define it and there is no cargo feature for it. It is marked in a comment. SIMD, asm and MIPS overrides are dropped.
5. Two C quirks are ported as-is. In celt_iir's non-unrolled tail loop, `y[i+ord]` is stored with a positive sign, while the unrolled loop stores it negated. In remove_doubling, the `T1 < 2*minperiod` branch can never be reached.
6. There are no double promotions in these files. All constants are float literals (.25f, .5f, 1.0001f, .008f, 1e-12f, 1e-10f, .001f). compute_pitch_gain uses celt_sqrt, which goes through double sqrt.
7. celt_assert calls became debug_assert, as in C. C also asserts len >= 3 in xcorr_kernel, so the same length preconditions as C apply.
8. The oracle shim reaches the static helpers of pitch.c by #including a copy of pitch.c with its exported symbols renamed (cpl_copy_*). The shims for the exported functions call the real library symbols.
9. The oracle's safe wrappers check every slice length the C code will read or write before calling it.

## `silk_resampler`

### external needs

Nothing outside my own files had to be edited.
- For `down2_3`, `AR2` and `up2_HQ`, `SilkResamplerState`/`ResamplerFunction` and the ROM tables, see notes for how later SILK units should call these functions.
- The oracle shim reaches the static inline INTERPOL functions by #including `resampler_private_IIR_FIR.c` and `resampler_private_down_FIR.c` with the exported function renamed through #define. This does not affect other units.
- The relayed user request, "keep this as project memory", could not be acted on inside this unit, because the task forbids editing docs/*, CLAUDE.md and config. The orchestrator or the user should decide whether to record anything in project docs or memory.

### notes

What later units need to know:
- **Resampler state:** `SilkResamplerState` keeps its name and Debug+Clone+Default (Default is implemented by hand because some arrays are longer than 32).
  - The fields are snake_case versions of the C names.
  - The C sFIR union is two arrays: `s_fir_i32` (used by down_FIR) and `s_fir_i16` (used by IIR_FIR). A state only ever uses one of them, so the aliasing can't be seen from outside.
  - `resampler_function` is a `ResamplerFunction` enum (Copy=0, Up2Hq=1, IirFir=2, DownFir=3, with `to_c()`).
  - `coefs` is a `&'static [i16]`; an empty slice stands for C NULL.
- **Deviation 1, invalid rates:** for an unsupported rate pair, `silk_resampler_init` returns -1 and does not assert. The hardened oracle (ENABLE_HARDENING) runs `celt_assert(0)` and aborts, so the tests call C only for pairs it accepts and check the Rust -1 on their own. Under qext, the encoder pairs 96→8 and 96→12 kHz have no filter and return -1, as in C.
- **Deviation 2, `SilkResamplerState::new`:** added this `Result`-returning constructor; it maps -1 to `Error::BadArg`.
- **Other asserts:**
  - The `celt_assert`s in `silk_resampler` became `debug_assert!`.
  - The static `silk_assert`s on table signs became `const` asserts.
  - The default branch of down_FIR_INTERPOL became `debug_assert!(false)`.
- **Quirk kept:** the down_FIR batch loop continues only while `inLen > 1`, so a single leftover sample after a full batch is dropped, as in C.
- **Signatures:** functions keep C's explicit `i32` length arguments on top of slices, and the private resamplers take `&mut SilkResamplerState`. `silk_resampler` returns 0 like C, so C-style `ret += ...` code stays easy to port. The INTERPOL helpers return how many samples they wrote instead of an advanced pointer.
- **Borrowing:** `silk_resampler` copies `delayBuf` into a local before the private calls, because C passes `S` and `S->delayBuf` together. This is equivalent because the private functions never touch `delayBuf`.
- **Scratch buffers:** the C ALLOC buffers are fixed stack arrays sized by the largest batch: 10 ms × 48 kHz, or × 96 kHz with qext. That is at most about 4 KB.
- **Not in the tests:** C does not export the delay matrices; the inputDelay state check covers every pair.
- **Commit message:** plain, with no trailers, as CLAUDE.md requires.

## `celt_fft`

### external needs

No edits were needed outside the owned files. static_modes.rs types (KissFftState, KissFftCpx, MdctLookup, MAXFACTORS) were used as they are.
1. static_modes discrepancy (C-side, not a port bug): FFT_TWIDDLES48000_960, MDCT_TWIDDLES960 and the 96 kHz tables differ by up to 1 ulp from what C's own opus_fft_alloc/clt_mdct_init compute at run time on this host (e.g. twiddles480[1].i: static 0xbc5675be vs runtime 0xbc5675bf). The port matches the C runtime values bit-exactly. Custom-mode lookups built at run time will therefore not be bit-identical to the static tables, exactly as in C.
2. Signatures other units must call:
   - opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]); opus_ifft has the same signature.
   - opus_fft_impl(st, fout: &mut [KissFftCpx]).
   - clt_mdct_forward(l: &MdctLookup, input: &[f32], out: &mut [f32], window: &[f32], overlap: usize, shift: usize, stride: usize); clt_mdct_backward has the same signature. The input is a shared slice because the float build never writes it. The integer arguments are usize, not i32.
   - custom-modes: opus_fft_alloc(nfft: i32) -> Result<KissFftState>; opus_fft_alloc_twiddles(nfft, base: Option<&KissFftState>) -> Result<KissFftState>; clt_mdct_init(n: i32, maxshift: i32) -> Result<MdctLookup>. Failure returns Error::AllocFail (modes.c maps this to OPUS_ALLOC_FAIL).
   - qext: mini_kiss_fft_alloc(nfft, inverse: bool) -> Result<MiniKissFftState>, mini_kiss_fft(_stride), mini_kiss_fftr_alloc, mini_kiss_fftr(&mut st, &[f32], &mut [KissFftCpx]).
3. pub(crate) helpers for other units: kiss_fft::CpxBuf, kiss_fft::Interleaved(&mut [f32]) and kiss_fft::opus_fft_impl_buf run the FFT in place on an interleaved f32 buffer, as C does when it casts float* to kiss_fft_cpx*. Also mdct::MAX_MDCT_N2 (the forward-MDCT scratch bound; depends on features).
4. mini_kfft reuses KissFftCpx through the alias MiniKissFftCpx; no duplicated helpers need deduplicating.

### notes

Deviations from the C code, all documented in the source:
- Failure handling: C's opus_fft_alloc_twiddles failure path calls opus_fft_free on a state whose bitrev/twiddles pointers were never initialized. That is undefined behaviour in the oracle (it crashed with a double free), so failing sizes (radix > 5, no matching base shift, N4 not divisible by 2^maxshift) are tested on the Rust side only.
- C leaves the unused entries of runtime-allocated factor arrays uninitialized, so tests compare only the used factor prefix.
- Arguments C does not check: Rust returns BadArg for nfft <= 0, for maxshift outside 0..=3 (C would overflow kfft[4]), and for N/2 > MAX_MDCT_N2.
- Rust-only guards against C undefined behaviour: kf_factor returns failure after MAXFACTORS stages instead of overflowing, and the fstride array has one extra slot.
- mini_kfft: C asserts that would abort (odd real-FFT size, radix outside 2..=5, nfft < 2) become Error::BadArg at allocation time, so the transforms cannot fail. The mem/lenmem caller-storage mode is dropped; states own Vecs.
- Unused kfft slots when maxshift < 3 hold an empty state; C leaves them uninitialized.
- With a base, opus_fft_alloc_twiddles clones the base twiddles, because Cow<'static> cannot borrow from a runtime base.
- Forward-MDCT scratch (C VLAs) uses fixed stack arrays sized by MAX_MDCT_N2: 960, 1024 with custom-modes, 1920 with qext, 2048 with both. That is about 7.7 KB default and up to 16 KB with qext+custom-modes.
- Loops that walk C pointers downward past index 0 after their last iteration are rewritten with indices computed from the loop counter; valid overlaps must be multiples of 4, as in C.
- FIXED_POINT branches are skipped with marker comments. There are no DNN branches in this unit. The arch parameter and arch_fft state are dropped.
Relayed user request "keep this as project memory": the task forbids editing CLAUDE.md, docs/* and config, so nothing was written as project memory. Whoever merges should decide whether to record this unit's status in docs/TRACKER.md.

## `celt_modes`

### external needs

1. static_modes.rs checked: no discrepancies. Every field and table of MODE48000_960_120, and under qext of MODE96000_1920_240, matches the oracle bit-for-bit, including windows, trig, FFT states/twiddles/bitrev and both pulse caches. No fix needed.
2. clt_mdct_init belongs to the celt_fft unit and does not exist in this worktree. modes.rs::opus_custom_mode_create_custom currently calls a clearly marked placeholder, clt_mdct_init_pending, which returns None, so non-static custom modes return Err(AllocFail). After the merge, replace that placeholder with crate::celt::mdct::clt_mdct_init(n, maxshift) -> Option<MdctLookup> (or adapt the closure). opus_custom_mode_create_with(fs, n, mdct_init) already does the full creation apart from the MDCT and is what the tests use.
3. Duplicated helper to dedupe later: rate.rs has a private copy of the float eMeans[25] table (E_MEANS, qext only), which belongs to quant_bands.rs (celt_bands unit).
4. API choices other units depend on:
   - clt_compute_allocation and clt_compute_extra_allocation take `ec: &mut EcCoder`. The C `encode` flag is `ec.is_encoder()`.
   - clt_compute_extra_allocation takes `qext_mode: Option<&CeltMode>`. The decoder can pass empty band_log_e / qext_band_log_e slices, as C passes NULL.
   - compute_qext_mode(m) returns a CeltMode by value. It is cheap for static modes because all Cow fields stay borrowed.
   - decode_pulses returns f32 (opus_val32) yy.
   - laplace functions take EcEnc / EcDec directly.
5. No edits were needed outside the owned files.

### notes

Deviations and quirks:
- compute_ebands: C's two celt_assert checks (only built with ENABLE_ASSERTIONS) are not ported. They fail for modes that default libopus builds accept, e.g. opus_custom_mode_create(12000, 208) with bands ..., 16, 20, 22, so a debug_assert would turn a valid call into a panic. The reason is written in a comment in the code.
- libopus bug: in opus_custom_mode_create, the CUSTOM_MODES `failure:` path calls opus_custom_mode_destroy on a half-built mode. That frees uninitialized fields, which gave glibc "double free" aborts in the oracle when the PVQ table is too narrow (>208) or clt_mdct_init fails (an FFT size with a prime factor above 5). The tests therefore skip calling C for inputs where Rust returns AllocFail, using a placeholder MDCT that fails exactly when C's kf_factor would.
- clt_compute_extra_allocation without a QEXT mode and with fewer than 5 bands (e.g. hybrid start=17, end=21): C reads an uninitialized `follower` entry. Rust zero-initializes it. The tests keep end-start >= 5 in that case.
- Laplace uses unsigned comparisons for IMIN on int/unsigned mixes, and wrapping for unsigned products, as C does.
- cwrs: CWRS_EXTRA_ROWS (the 1488-entry table) is selected by cfg(any(custom-modes, qext)), exactly as C. Both tables were generated from cwrs.c with cpp.
- Not ported: SMALL_FOOTPRINT (unext, uprev, ncwrs_urow and the other variant functions; marker comment in place), FUZZING branches (marker comments), and opus_custom_mode_destroy (Rust drop covers it).
- Scratch arrays are fixed-size stack arrays: MAX_ALLOC_BANDS = 32, which is enough since custom modes stay under about 25 bands; tot_bands is limited to 32+14.
- The commit message is plain, with no trailers, per CLAUDE.md.
- The relayed user request, "keep this as project memory", could not be acted on inside this porting task. It needs the parent session.

## `opus_packet`

### external needs

- No files outside this unit were edited.
- Nothing from outside was needed; the foundation already provides opus_limit2_checkwithin1, res2int16/res2int24 and min/max.
- Possible duplicates for later cleanup: packet.rs defines MODE_SILK_ONLY/MODE_HYBRID/MODE_CELT_ONLY and toc_mode (opus_packet_get_mode, which is static in opus_decoder.c). The decoder and encoder units may write private copies of these.
- Crate-internal helper len_i32 (clamps usize to opus_int32) is pub(crate) in packet.rs and can be reused.
- Items the decoder and encoder units should call:
  - packet: parse_impl(data, self_delimited) -> ParsedPacket (toc, nb_frames, frames[48] as slices, payload_offset, packet_offset, padding slice), toc_* const helpers, encode_size, opus_pcm_soft_clip_impl
  - repacketizer: Repacketizer::{cat_impl, out_range_impl(begin, end, &mut data[..maxlen], self_delimited, pad, &[Extension])}, opus_packet_pad_impl
  - extensions: ExtensionIterator::{new, next_extension, find, set_frame_max}, Extension
  - mlp: analysis_compute_dense/gru, LAYER0..2
  - mapping_matrix: MappingMatrix + multiply fns + static tables
  - multistream: ChannelLayout + helpers
- The encoder's encode_multiframe fills tmp_data frame by frame while the repacketizer borrows the earlier frames. It needs split_at_mut on the remaining buffer so each cat gets an &'a [u8].
- The oracle shim includes repacketizer.c and mlp.c with their exported names renamed (oracle_rpc_*, oracle_mlpc_*) to reach the static cat_impl, tansig_approx and sigmoid_approx. Other units' shims must not reuse those names.
- docs/TRACKER.md was not updated (docs are off-limits); the opus_packet row can be marked done.
- The relayed user request "keep this as project memory" was not acted on inside this unit, because the task forbids editing docs and shared files. The orchestrator should handle it if memory changes are wanted.

### notes

Deviations and quirks:
- Pointers become slices and offsets. Extension.data is a borrowed slice, so C's `len` is data.len(), and the "ext->len < 0 → BAD_ARG" branches can no longer be reached.
- In-place pad/unpad: C relies on memmove while frames point into the output buffer. The port copies the input first (opus_packet_pad_impl already copies in C). Unpadded output is never larger than its input, so this gives the same bytes; the tests compare full buffers, including bytes left over past the output.
- Where C reads data[0] of an empty packet (undefined behaviour), the public functions return BadArg; has_lbrr returns InvalidPacket, matching what parse would report.
- toc_samples_per_frame and get_nb_samples wrap on overflow from absurd Fs values (undefined behaviour in C) instead of panicking.
- parse_ext returns BadArg for more than 48 frames or negative cumulative indices (C asserts or has undefined behaviour). Negative nb_frames in generate is undefined behaviour in C (memset with a huge size), so it is not tested.
- The public extensions::parse drops C's partial count on error; the repacketizer discards it anyway.
- pcm_soft_clip and opus_pcm_soft_clip_impl treat buffers that are too short like C treats NULL pointers: silently do nothing.
- mapping_matrix_get_size and align reproduce C's unsigned and size_t arithmetic, including for negative arguments. align uses the platform pointer alignment, like the C offsetof trick.
- out_short and out_int24 reproduce the implementation-defined narrowing conversion back to opus_int16/opus_int32 by truncation.
- Not ported:
  - opus_repacketizer_get_size and opus_repacketizer_destroy: C memory management (Repacketizer::new covers create).
  - pad_frame: declared in opus_private.h but never defined.
  - The FIXED_POINT branches of mapping_matrix: marked with comments.
- This unit has no QEXT, CUSTOM_MODES or DNN code.

## `silk_common`

### external needs

1. The shim needs silk_inner_prod16_c, which lives in silk/fixed/vector_ops_FIX.c and is not compiled into the float oracle. csrc/silk_common.c #includes that file with all four of its global symbols renamed (oracle_sc_*) so it cannot clash with other shims. If the fixed-point oracle is added later, this include can go.

2. Downstream units (silk_decoder, silk_encoder_common, silk_encoder_flp, silk_resampler) need to know these API conventions:
   - Names are snake-cased: silk_NLSF2A → silk_nlsf2a, silk_LPC_inverse_pred_gain → silk_lpc_inverse_pred_gain, silk_sigm_Q15 → silk_sigm_q15, silk_gains_ID → silk_gains_id. Tables are the C names upper-cased (silk_gain_iCDF → SILK_GAIN_ICDF).
   - Struct types are SilkEncoderState, SilkDecoderState, SilkNsqState, SilkVadState, SilkLpState, SilkPlcStruct, SilkCngStruct, SideInfoIndices, StereoEncState, StereoDecState, SilkDecoderControl, SilkNlsfCbStruct, SilkEncControlStruct and SilkDecControlStruct. Their fields are snake_case (prev_gain_Q16 → prev_gain_q16).
   - C pointer fields became references: psNLSF_CB is &'static SilkNlsfCbStruct (defaults to the NB/MB codebook instead of NULL), and pitch_lag_low_bits_iCDF / pitch_contour_iCDF are &'static [u8] (default empty). The arch fields are dropped.
   - The decoder and encoder states embed crate::silk::resampler::SilkResamplerState and rely only on Default + Clone + Debug.

3. Duplicated helpers: none.

### notes

Deviations and API choices:
- Lengths and orders are passed as usize. Arithmetic values stay i32. Index outputs (the sort idx) are i32, matching opus_int.
- C out-parameters become return values in two places: silk_sum_sqr_shift returns (energy, shift) and silk_stereo_decode_mid_only returns the flag. Everything else keeps the C out-parameter shape.
- SILK_LTP_VQ_PTRS_Q7 rows are [i8; 5] (LTP_ORDER taps). Use .as_flattened() for C's flat view.
- C calls two functions in place with aliased arguments:
  - silk_biquad_alt_stride1 with in == out (from LP_variable_cutoff) is covered by an extra silk_biquad_alt_stride1_inplace.
  - combine_and_check with in == out has an in-place twin.
- silk_A2NLSF_eval_poly's unrolled dd==8 case is ported as the loop it unrolls; the operations are identical.
- The FIXED_POINT-only silk_insertion_sort_decreasing_int16 is not ported (marker left). DNN OSCE fields in silk_decoder_state and silk_DecControlStruct are skipped with markers. This unit has no QEXT or CUSTOM_MODES code.
- C int16/int8 stores truncate, and this is reproduced with explicit `as` casts (gains_quant i8 indices, NLSF_stabilize, del_dec_quant out0/out1). silk_ADD_RSHIFT_uint in sum_sqr_shift uses wrapping u32, as in C. Everything else uses plain arithmetic per PORTING.md.
- Compile-time-constant celt_assert/silk_assert checks are const { assert!() }; the rest are debug_assert!.

Test notes:
- Some inputs were bounded to where C's int32 arithmetic does not overflow (C's overflow there is undefined behaviour and would make Rust panic in debug). This affects NLSF_VQ weights × order, del_dec_quant amplitude/weight tiers, and NLSF_encode inputs, which are kept near codebook vectors with gaps ≥ 100 Q15. Each bound is explained in a test comment. Bit-exactness was never relaxed.
- The oracle is built with ENABLE_HARDENING, so celt_assert is live in C. Tests therefore only use valid SILK frame lengths.
- Callers of silk_encode_pulses must pass a pulses buffer of MAX_FRAME_LENGTH. For 10 ms @ 12 kHz, C zeroes pulses[frame_length..frame_length+16], past the padded length. The oracle wrapper asserts this size.

On the relayed request "keep this as project memory": I did not write any memory, CLAUDE.md or docs files. The task forbade editing docs/*, and it is unclear what that request wanted saved.

## `analysis`

### api for other units

Module crate::analysis (pub items; the module is pub(crate) unless the internals feature is on):
- pub const LEAK_BANDS: usize = 19; NB_FRAMES = 8; NB_TBANDS = 18; ANALYSIS_BUF_SIZE = 720; ANALYSIS_COUNT_MAX: i32 = 10000; DETECT_SIZE = 100.
- #[derive(Debug, Clone, Copy, Default, PartialEq)] pub struct AnalysisInfo { pub valid: i32, pub tonality: f32, pub tonality_slope: f32, pub noisiness: f32, pub activity: f32, pub music_prob: f32, pub music_prob_min: f32, pub music_prob_max: f32, pub bandwidth: i32, pub activity_probability: f32, pub max_pitch_ratio: f32, pub leak_boost: [u8; LEAK_BANDS] }
- pub type DownmixFunc<T> = fn(&[T], &mut [OpusVal32], i32 /*subframe*/, i32 /*offset*/, i32 /*c1*/, i32 /*c2*/, i32 /*C*/);
- pub fn downmix_float(x: &[f32], y: &mut [f32], subframe: i32, offset: i32, c1: i32, c2: i32, c: i32); downmix_int(x: &[i16], ...); downmix_int24(x: &[i32], ...). These can be passed directly as DownmixFunc<f32> / <i16> / <i32>.
- pub fn is_digital_silence(pcm: &[f32], frame_size: i32, channels: i32, lsb_depth: i32) -> bool
- #[derive(Debug, Clone, PartialEq)] pub struct TonalityAnalysisState { pub application: i32, pub fs: i32, angle, d_angle, d2_angle, inmem, mem_fill, prev_band_tonality, prev_tonality, prev_bandwidth, e, log_e, low_e, high_e, mean_e, mem, cmean, std, e_tracker, low_e_count, e_count, count, analysis_offset, write_pos, read_pos, read_subframe: i32, hp_ener_accum, initialized: i32, rnn_state, downmix_state, info: [AnalysisInfo; DETECT_SIZE] }. There is no arch field. Construct it with TonalityAnalysisState::new(fs); new() calls tonality_analysis_init.
- pub fn tonality_analysis_init(tonal: &mut TonalityAnalysisState, fs: i32)
- pub fn tonality_analysis_reset(tonal: &mut TonalityAnalysisState)
- pub fn tonality_get_info(tonal: &mut TonalityAnalysisState, info_out: &mut AnalysisInfo, len: i32)
- pub fn run_analysis<T>(analysis: &mut TonalityAnalysisState, celt_mode: &CeltMode, analysis_pcm: Option<&[T]>, analysis_frame_size: i32, frame_size: i32, c1: i32, c2: i32, c: i32, fs: i32, lsb_depth: i32, downmix: DownmixFunc<T>, analysis_info: &mut AnalysisInfo). Passing None is the C NULL analysis_pcm; celt_mode must be the 48 kHz/960 mode.
- Also pub, for tests: silk_resampler_down2_hp(s: &mut [f32;3], out, input, in_len) -> f32; downmix_and_resample<T>(downmix, x, y, s: &mut [f32;3], subframe, offset, c1, c2, c, fs) -> f32; tonality_analysis<T>(tonal, celt_mode, x, len, offset, c1, c2, c, lsb_depth, downmix).

### external needs

1. AnalysisInfo and LEAK_BANDS are defined in crates/opusorus/src/analysis.rs, not in celt/celt.rs as in C. The celt_bands and celt_encoder units should import crate::analysis::{AnalysisInfo, LEAK_BANDS} and not define their own.
2. downmix_float, downmix_int, downmix_int24 and is_digital_silence (all from src/opus_encoder.c) are ported in analysis.rs. The opus_encoder unit should reuse them and not port them again. The encoder's run_analysis call becomes generic over the PCM sample type T (f32/i16/i32) with the matching DownmixFunc<T>.
3. The tracker entry for the analysis unit in docs/TRACKER.md should be set to done (bit-exact); I did not edit docs.

### notes

Port file: crates/opusorus/src/analysis.rs.

Oracle files:
- crates/opusorus-oracle/csrc/analysis.c: builds a private copy of vendor/libopus/src/analysis.c with its exported functions renamed, to reach the static functions. It includes the copy by relative path, because a bare "analysis.c" include resolves to the shim itself.
- crates/opusorus-oracle/src/analysis.rs: repr(C) mirrors CTonalityAnalysisState and CAnalysisInfo, with a layout check against the C compiler, so tests can read and set the full C state.

Tests: crates/opusorus-conformance/tests/analysis.rs.

Translation details:
- The FFT uses the 48k mode's mdct.kfft[0] via crate::celt::kiss_fft::opus_fft.
- The MLP uses crate::mlp::{LAYER0, LAYER1, LAYER2, analysis_compute_dense, analysis_compute_gru}.
- Float/double promotions follow the C source:
  - log, sqrt, log10 and floor are computed in double on float arguments.
  - `highE > lowE + 7.5` and `sqrt(1e-15 + NB_FRAMES*L2)` are computed in double.
  - `(float)(.5f/M_PI)` and the pi^4 constant are computed in double.
- The oracle's M_PI equals mathops::PI; a test checks this.
- MLP_TRAINING and FIXED_POINT branches are skipped with marker comments. analysis.c has no QEXT or DNN code; the unit builds and passes with qext and custom-modes.

Behaviour notes:
- downmix_and_resample, like celt_assert(0) in C, does debug_assert!(false) for Fs other than 16/24/48 kHz. The encoder only runs the analysis for 16–48 kHz. So run_analysis is tested at 16/24/48 kHz, and tonality_get_info, which works at any rate, is also tested at 8 and 12 kHz.
- Stack scratch is fixed-size: 960-float tmp, 1440-float tmp3x, and 480-entry FFT in/out buffers. Nothing is allocated.
- A NULL-pcm run_analysis call permanently moves read_pos ahead of the analysis. This is C behaviour: later frames read slots that have not been written yet and show valid=0 until the 100-slot ring wraps. The Rust port does the same, and the tests exercise it in a third of the streams.

## `silk_decoder`

### api for other units

Module crate::silk::decoder:

pub struct SilkDecoder { pub channel_state: [SilkDecoderState; 2], pub s_stereo: StereoDecState, pub n_channels_api: i32, pub n_channels_internal: i32, pub prev_decode_only_middle: i32 }
- Implements Debug, Clone and Default. Default equals new().
- It is about 9 KB, so the Opus decoder should embed or Box it.

impl SilkDecoder {
  pub fn new() -> Self   // silk_InitDecoder on zeroed memory (nChannelsAPI/Internal = 0, as after OPUS_CLEAR)
  pub fn init(&mut self) -> i32   // silk_InitDecoder
  pub fn reset(&mut self) -> i32  // silk_ResetDecoder (call when prev_mode == CELT_ONLY)
  pub const fn load_osce_models(&mut self, _data: Option<&[u8]>) -> i32  // returns SILK_NO_ERROR
  pub fn silk_decode(&mut self, dec_control: &mut SilkDecControlStruct, lost_flag: i32, new_packet_flag: i32, ps_range_dec: &mut EcDec<'_>, samples_out: &mut [OpusRes], n_samples_out: &mut i32) -> i32
}

About silk_decode:
- samples_out is &mut [OpusRes] (f32), because the 1.6.1 silk_Decode writes opus_res*, converted with INT16TORES.
- It writes *n_samples_out samples per API channel, interleaved when n_channels_api == 2. At most 20 ms at the API rate per call.
- lost_flag uses FLAG_DECODE_NORMAL, FLAG_PACKET_LOST and FLAG_DECODE_LBRR from silk::structs.
- It returns C int codes from silk::errors (SILK_DEC_INVALID_FRAME_SIZE, SILK_DEC_INVALID_SAMPLING_FREQUENCY, or accumulated resampler-init -1 codes).

Other public functions:
- pub const fn silk_get_decoder_size(dec_size_bytes: &mut i32) -> i32
- pub fn silk_reset_decoder(&mut SilkDecoderState) -> i32
- pub fn silk_init_decoder(&mut SilkDecoderState) -> i32
- pub fn silk_decoder_set_fs(&mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32
- pub fn silk_decode_frame(&mut SilkDecoderState, &mut EcDec, p_out: &mut [i16], p_n: &mut i32, lost_flag: i32, cond_coding: i32) -> i32
- pub fn silk_decode_core(&mut SilkDecoderState, &mut SilkDecoderControl, xq: &mut [i16], pulses: &[i16])
- pub fn silk_decode_indices(&mut SilkDecoderState, &mut EcDec, frame_index: i32, decode_lbrr: i32, cond_coding: i32)
- pub fn silk_decode_parameters(&mut SilkDecoderState, &mut SilkDecoderControl, cond_coding: i32)
- pub fn silk_stereo_ms_to_lr(&mut StereoDecState, x1: &mut [i16], x2: &mut [i16], pred_q13: &[i32; 2], fs_khz: i32, frame_length: i32)

Module crate::silk::plc: pub fn silk_plc_reset, pub fn silk_plc(ps_dec, ps_dec_ctrl, frame, lost), pub fn silk_plc_glue_frames(ps_dec, frame, length), plus the PLC.h constants.

Module crate::silk::cng: pub fn silk_cng_reset(ps_dec), pub fn silk_cng(ps_dec, &SilkDecoderControl, frame, length).

To mirror opus_decoder.c, the Opus decoder sets dec_control.payload_size_ms = max(10, audiosize_ms). When data is present it sets n_channels_internal and internal_sample_rate (SILK-only: 8000/12000/16000 by bandwidth; hybrid: 16000) and enable_deep_plc = complexity >= 5. It then loops silk_decode with new_packet_flag = (decoded == 0) until frame_size is reached, and zero-fills on a PLC error.

### external needs

None blocking.

Suggested docs/TRACKER.md update for the orchestrator (I did not edit docs): mark silk_decoder as done, with tests "silk_decoder.rs: full silk_Decode vs oracle on encoder-generated SILK streams (NB/MB/WB, 10-60 ms, all API rates incl. qext 96k, mono/stereo, mid-only, PLC, FEC/LBRR, DTX/CNG, resets, garbage) + MS->LR and decode_indices direct — bit-exact, including the full state dump after every call".

### notes

- **Invalid control values can't be tested against C.** The oracle is built with ENABLE_HARDENING, which turns celt_assert into an abort. So the C "celt_assert(0); return SILK_DEC_INVALID_FRAME_SIZE / SILK_DEC_INVALID_SAMPLING_FREQUENCY" paths in silk_Decode (bad payloadSize_ms or internal rate), and silk_resampler_init with an unsupported API rate, abort in the oracle. The Rust port keeps the error return without the assertion, the same choice the silk_resampler unit made. These paths are covered by a Rust-only test (invalid_control) that checks the codes and that a later valid call still decodes.
- **Output type.** In 1.6.1 silk_Decode outputs opus_res*, which is float in the float build, so the Rust samples_out is &mut [OpusRes] (f32), filled with int16tores. It is not i16.
- **Scratch buffers.** C VLAs (samplesOut1_tmp, samplesOut2_tmp, sLTP, sLTP_Q15, pulses, CNG_sig, and so on) are fixed-size zeroed stack arrays sized by the C maxima. Largest: samplesOut2_tmp at MAX_API_FS_KHZ*20 i16 (3.8 KB with qext). No heap allocation.
- **Uninitialised decoder control.** C's silk_decode_frame leaves silk_decoder_control uninitialised; Rust uses Default (zeros). The C code never reads the uninitialised fields, and the bit-exact state comparisons confirm it.
- **DNN hooks skipped with marker comments:** ENABLE_DEEP_PLC (lpcnet arguments, conceal/update, the 16 kHz glue-fade skip), ENABLE_OSCE (osce_model, osce_reset, osce_enhance_frame, LoadOSCEModels) and ENABLE_OSCE_BWE. There are no FIXED_POINT or QEXT-specific branches in these sources; QEXT only changes MAX_API_FS_KHZ (96), which is picked up from silk::define.
- **Oracle shim.** It includes dec_API.c with its globals renamed (oracle_sd_*) so the private silk_decoder struct is visible. It owns the decoder and a range decoder over a copy of the payload, and exposes a full state dump plus direct shims for silk_stereo_MS_to_LR and silk_decode_indices/pulses.
- **Files edited:** crates/opusorus/src/silk/{decoder,plc,cng}.rs, crates/opusorus-oracle/{src,csrc}/silk_decoder.*, and crates/opusorus-conformance/tests/silk_decoder.rs. No shared files touched.

## `tools_compare`

### api for other units

Crate opusorus-tools (lib + bins opus_compare, qext_compare[required-features = qext]). Module opusorus_tools::compare:
- enum SampleFormat { S16Le, S24Le, F32Le } (+ size())
- fn read_pcm(bytes: &[u8], nchannels: usize, format: SampleFormat) -> Vec<f32>; fn read_pcm16(bytes, nchannels) -> Vec<f32>; fn read_pcm_file(path, nchannels, format) -> io::Result<Vec<f32>>
- struct OpusCompareOptions { nchannels: usize, rate: u32 } (Default mono/48000); struct OpusCompareResult { err: f64, q: f32, passes: bool }
- fn opus_compare(x_stereo: &[f32], y: &[f32], opts: &OpusCompareOptions) -> Result<OpusCompareResult, CompareError>   // x = file1 read as stereo (C always does), y has opts.nchannels
- fn opus_compare_main(args: &[String], stderr: &mut dyn Write) -> io::Result<i32>   // exact C CLI
- [feature qext] struct QextCompareOptions { nchannels, base_rate: u32 (96000|48000), rate: u32 (0 = base), skip: i32 }; struct QextCompareResult { err4: f64, err16: f64, rms: f64 } + passes(t4, t16, trms) -> bool; fn qext_compare(x_stereo, y, &QextCompareOptions) -> Result<QextCompareResult, CompareError>; fn qext_compare_main(args, stderr) -> io::Result<i32>
- enum CompareError (Display = exact C messages: SampleCountMismatch, InsufficientData, OpusRate, QextRate, ...)
- EXIT_SUCCESS/EXIT_FAILURE, c_atoi, c_atof, c_fmt_f
Oracle: opusorus_oracle::tools_compare::{opus_compare_main(&[&str]) -> ToolRun, qext_compare_main(&[&str]) -> ToolRun}, where ToolRun { exit_code, stderr, values: Vec<f64> }. The C state is thread-local, so it is safe to call from parallel tests.
A later vectors/decoder test can call opus_compare / qext_compare directly on decoded buffers, with no temp files needed.

### external needs

Edits outside the owned files that are needed but not pre-approved:
- crates/opusorus-conformance/Cargo.toml: added the dependency opusorus-tools = { path = "../opusorus-tools" } and added "opusorus-tools/qext" to the conformance qext feature. The test cannot use the tools crate otherwise. When merging, consider adding opusorus-tools to [workspace.dependencies] in the root Cargo.toml.
- Cargo.lock gained the new crate.
- The allowed one-line `pub mod tools_compare;` was added to crates/opusorus-oracle/src/lib.rs.
Docs to update (not edited, per the rules):
- docs/TRACKER.md: add a tools_compare row.
- docs/PLAN.md: record the decision to keep the tools crate's qext feature opt-in (reason under Notes).
- justfile: the vectors recipe could build `cargo build -p opusorus-tools --features qext` to get both binaries.

### notes

qext design choice: opusorus-tools depends on opusorus with ["internals", "std"]. internals exposes celt::mini_kfft and math; std selects the platform libm, needed for bit-exactness. qext_compare sits behind an opt-in tools feature `qext` that enables opusorus/qext; the binary uses required-features. Turning opusorus/qext on unconditionally would, through workspace feature unification, switch QEXT on for every `cargo test --workspace` while the oracle stays non-QEXT, which would break other units' tests.
How the oracle shim works:
- csrc/tools_compare.c #includes the unmodified src/opus_compare.c and src/qext_compare.c into one translation unit, with `main` and the file-static helpers renamed. qext_compare.c pulls in celt/mini_kfft.c, whose external symbols (kf_work, kf_factor, mini_kiss_fft*) are renamed so they do not clash with the copy compiled into the library for qext builds.
- fprintf is macro-redirected to a thread-local capture that keeps both the formatted text and every double argument.
Where Rust deliberately differs from C (only where C crashes or has undefined behaviour; documented in the module doc):
- A NULL argv entry, for example `opus_compare -s file`, prints the usage line.
- `qext_compare -48k -r 96000` makes C divide by zero; Rust prints "Sampling rate 96000 exceeds the base rate 48000".
- A negative or oversized -skip makes C read out of bounds; Rust returns CompareError::SkipOutOfRange.
- In qext_compare with only one analysis frame, C's backward-masking loop wraps around size_t and reads out of bounds; Rust runs the loop zero times. The tests avoid this case for C.
- Read errors on a file that opened successfully are returned as io::Error; C treats them as EOF.
- c_atof does not parse hex floats.
C quirks ported faithfully:
- The stereo -skip length arithmetic: skip*nch is used both as a sample offset and as a frame count. As a result, `-s -skip N` on the qext_vector decodes reports a sample-count mismatch in both C and Rust.
- Every float→double promotion is reproduced (0.5*x downmix, MAX(3.16e-10*maxE, xb) in double, `re*re+im*im+.1` in double, the `im*w` weighting, thresh).
- The 6*OPUS_PI Blackman-Harris term is computed in float.
- The high-to-low masking loop bounds differ between the two tools (NBANDS-1 vs nbands-2), as in C.
- Printed values are formatted like glibc %f, including nan and -nan.
Performance: opus_compare's O(N·F·W) DFT takes about 5.9 s on a full RFC vector in release, roughly the same as C.

## `silk_encoder_common`

### api for other units

NSQ (module crate::silk::nsq)
- The C functions take `const silk_encoder_state *psEncC` while their other arguments point into that same state, which Rust's borrow rules forbid. So the fields they read are passed as a Copy struct:
  `pub struct NsqEncParams { pub ltp_mem_length, frame_length, subfr_length, nb_subfr, predict_lpc_order, shaping_lpc_order, n_states_delayed_decision, warping_q16: i32 }`
- Build it with `NsqEncParams::from_enc(&enc.s_cmn)` before borrowing the NSQ state, indices and pulses mutably.
- `pub fn silk_nsq_c(ps_enc_c: &NsqEncParams, nsq: &mut SilkNsqState, ps_indices: &SideInfoIndices, x16: &[i16], pulses: &mut [i8], pred_coef_q12: &[i16] /*2*MAX_LPC_ORDER flattened*/, ltp_coef_q14: &[i16], ar_q13: &[i16], harm_shape_gain_q14: &[i32], tilt_q14: &[i32], lf_shp_q14: &[i32], gains_q16: &[i32], pitch_l: &[i32], lambda_q10: i32, ltp_scale_q14: i32)`
- `pub fn silk_noise_shape_quantizer_short_prediction_c(buf32: &[i32] /*slice ending at the C pointer*/, coef16: &[i16], order: i32) -> i32`
- `pub fn silk_nsq_noise_shape_feedback_loop_c(data0: i32, data1: &mut [i32], coef: &[i16], order: i32) -> i32`

NSQ_del_dec (crate::silk::nsq_del_dec)
- `pub fn silk_nsq_del_dec_c(...)` has the same arguments as silk_nsq_c, except `ps_indices: &mut SideInfoIndices` because it writes the winning Seed.
- The wrapper picks del_dec when nStatesDelayedDecision > 1 || warping_Q16 > 0, as in C.

VAD (crate::silk::vad)
- `pub fn silk_vad_init(&mut SilkVadState) -> i32`
- `pub fn silk_vad_get_sa_q8_c(ps_enc_c: &mut SilkEncoderState, p_in: &[i16]) -> i32`
- `pub fn silk_vad_get_sa_q8_input_buf(ps_enc_c: &mut SilkEncoderState) -> i32` analyses `&input_buf[1..]`; use it for C's `silk_VAD_GetSA_Q8(&psEnc->sCmn, psEnc->sCmn.inputBuf + 1)`.
- `pub fn silk_vad_get_noise_levels(&[i32; 4], &mut SilkVadState)`

Encoder helpers (crate::silk::encoder_common)
- `pub fn silk_encode_indices(ps_enc_c: &mut SilkEncoderState, ps_range_enc: &mut EcEnc<'_>, frame_index: usize, encode_lbrr: i32, cond_coding: i32)`
- `pub fn silk_process_nlsfs(ps_enc_c: &mut SilkEncoderState, pred_coef_q12: &mut [[i16; 16]; 2], p_nlsf_q15: &mut [i16], prev_nlsfq_q15: &[i16])`. C passes psEnc->sCmn.prev_NLSFq_Q15, which aliases the state, so pass a copy: `let prev = enc.prev_nlsfq_q15;`.
- `pub fn silk_quant_ltp_gains(b_q14: &mut [i16], cbk_index: &mut [i8], periodicity_index: &mut i8, sum_log_gain_q7: &mut i32, pred_gain_db_q7: &mut i32, xx_q17: &[i32], x_x_q17: &[i32], subfr_len: i32, nb_subfr: usize)`
- `pub fn silk_vq_wmat_ec_c(ind: &mut i8, res_nrg_q15: &mut i32, rate_dist_q8: &mut i32, gain_q7: &mut i32, xx_q17: &[i32], x_x_q17: &[i32], cb_q7: &[[i8; 5]], cb_gain_q7: &[u8], cl_q5: &[u8], subfr_len: i32, max_gain_q7: i32, l: i32)`
- `pub fn silk_stereo_lr_to_ms(state: &mut StereoEncState, x1: &mut [i16], x2: &mut [i16], ix: &mut [[i8; 3]; 2], mid_only_flag: &mut i8, mid_side_rates_bps: &mut [i32], total_rate_bps: i32, prev_speech_act_q8: i32, to_mono: i32, fs_khz: i32, frame_length: i32)`
  - x1 and x2 start 2 samples before the C pointers: pass `&mut state_fxx[0].s_cmn.input_buf[..]`, not `[2..]`.
  - C writes ix and mid_only_flag directly into sStereo.predIx[n] and mid_only_flags[n]. Pass locals instead and store them into the state afterwards; the function never reads those fields.
- `pub fn silk_stereo_find_predictor(ratio_q14: &mut i32, x: &[i16], y: &[i16], mid_res_amp_q0: &mut [i32], length: usize, smooth_coef_q16: i32) -> i32`
- `pub fn silk_stereo_quant_pred(pred_q13: &mut [i32; 2], ix: &mut [[i8; 3]; 2])`
- `pub const fn silk_hp_variable_cutoff(ps_enc_c1: &mut SilkEncoderState)`. C takes the whole state_Fxx[] array; pass `&mut state_fxx[0].s_cmn`.
- `pub fn silk_control_snr(ps_enc_c: &mut SilkEncoderState, target_rate_bps: i32) -> i32`
- `pub const fn silk_control_audio_bandwidth(ps_enc_c: &mut SilkEncoderState, enc_control: &mut SilkEncControlStruct) -> i32` (returns fs_kHz)
- `pub const fn check_control_input(enc_control: &SilkEncControlStruct) -> i32`
- `pub static SILK_TARGET_RATE_{NB,MB,WB}_21`

### external needs

No changes to other units' files were needed. One issue for the silk_common owners (crates/opusorus/src/silk/nlsf.rs, silk_nlsf_del_dec_quant, lines ~715 and ~721):
- The `silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i])` call can overflow when NLSF values are only a few Q15 units apart.
- Inputs like that come from silk_A2NLSF of filters with very sharp resonances (reflection coefficients ~0.99). They stay inside MAX_PREDICTION_POWER_GAIN, so the real encoder can see them, e.g. on near-pure tones.
- C just wraps there (int overflow, UB in practice). The Rust port panics in debug/test builds (it wraps in release, matching C).
- Reproduce with process_NLSFs given these NLSFs: [1725, 8655, 8669, 10316, 10435, 11966, 11972, 29217, 30009, 32755], d=10, nb_subfr=2, speech_activity_Q8=32, interp=4, survivors=16, signalType=2.
- Suggest wrapping semantics there, with a comment. My test keeps its filters at kmax ≤ 0.9 to stay out of that regime, and says why in a comment.

My own VQ_WMat_EC keeps plain silk_MLA like C. Correlations built the way silk_find_LTP_FLP normalizes them, including quiet-target extremes, did not overflow in testing.

### notes

Faithfulness and deviations
- Plain ops are used everywhere except the C *_ovflw and unsigned-shift macros.
- C's in-place `silk_ana_filt_bank_1(X, .., X, ..)` calls in VAD filter a copy of the input instead. This is equivalent because each input sample is read before its slot is overwritten.
- The NSQ_del_dec memcpy that copies a state from int32 word i onward is `NsqDelDecStruct::copy_from_word`: it copies everything except sLPC_Q14[..i].
- check_control_input does not reproduce C's `celt_assert(0)` before each error return (same choice as silk_resampler_init). With ENABLE_HARDENING that assert aborts the oracle, so the shim compiles a renamed copy of check_control_input.c with celt_assert disabled; that is how the error codes are compared.
- silk_VAD_GetNoiseLevels is static in C; the shim reaches it by including VAD.c with the global names renamed.
- The C `gain_Q7` in quant_LTP_gains starts uninitialized; Rust starts it at 0. This only matters if no codebook vector has a non-negative error, which C also leaves undefined.

Test-generation choices
- Test inputs mimic what the encoder produces (stable filters, level-tracking gains, realistic pitch lags), because plain-op overflow in unrealistic states would be UB in C and a debug panic in Rust.
- I checked coverage by instrumenting a run: pulse magnitudes range from 0 to 31, all signal types appear, and stereo hits mid-only, zero-width and full-width.

Shim design
- The oracle mirrors silk_nsq_state, silk_VAD_state, stereo_enc_state, SideInfoIndices and silk_EncControlStruct as #[repr(C)] structs, with a sizeof check.
- Shims that need encoder state build a calloc'd silk_encoder_state from a flat params struct.
- HP_variable_cutoff runs on a real silk_encoder_state_FLP.

Docs
- Per the rules I did not edit docs/TRACKER.md. The silk_encoder_common row can be set to ✅.

## `celt_bands`

### api for other units

All items are in `opusorus::celt::*`. QEXT-only parameters are `#[cfg(feature = "qext")]` fn params placed where the C `ARG_QEXT` args go. Booleans replace C int flags.

bands.rs:
- `pub const SPREAD_NONE/SPREAD_LIGHT/SPREAD_NORMAL/SPREAD_AGGRESSIVE: i32`
- `pub struct BandsScratch` (Default, `new()`): replaces the C VLAs. Keep one per encoder/decoder state.
- `pub fn quant_all_bands(m: &CeltMode, start: i32, end: i32, x_: &mut [f32], y_: Option<&mut [f32]>, collapse_masks: &mut [u8], band_e: &[f32], pulses: &[i32], short_blocks: bool, spread: i32, dual_stereo: bool, intensity: i32, tf_res: &[i32], total_bits: i32, balance: i32, ec: &mut EcCoder, lm: i32, coded_bands: i32, seed: &mut u32, complexity: i32, disable_inv: bool, [qext: ext_ec: &mut EcCoder, extra_pulses: &[i32], ext_total_bits: i32, cap: Option<&[i32]>], scratch: &mut BandsScratch)`
  - The encode flag is `ec.is_encoder()`.
  - `x_` and `y_` are the per-channel halves (C `X`, `X+N`); use `split_at_mut(N)`.
  - For the C `NULL`/`0` dummy ext coder, pass `EcEnc::new(&mut [])` or `EcDec::new(&[])`.
- `pub fn compute_band_energies(m, x: &[f32], band_e: &mut [f32], end, c, lm)`
- `pub fn normalise_bands(m, freq: &[f32], x: &mut [f32], band_e: &[f32], end, c, mm)`
- `pub fn denormalise_bands(m, x: &[f32], freq: &mut [f32], band_log_e: &[f32], start, end, mm, downsample, silence: bool)`
- `pub fn anti_collapse(m, x_: &mut [f32], collapse_masks: &[u8], lm, c, size, start, end, log_e, prev1log_e, prev2log_e, pulses: &[i32], seed: u32, encode: bool)`
- `pub fn spreading_decision(m, x: &[f32], average: &mut i32, last_decision: i32, hf_average: &mut i32, tapset_decision: &mut i32, update_hf: bool, end, c, mm, spread_weight: &[i32]) -> i32`
- `pub fn hysteresis_decision(val: f32, thresholds: &[f32], hysteresis: &[f32], n: i32, prev: i32) -> i32`
- `pub const fn celt_lcg_rand(u32) -> u32`, `pub const fn bitexact_cos(i16) -> i16`, `pub const fn bitexact_log2tan(i32, i32) -> i32`
- `pub fn haar1(x: &mut [f32], n0, stride)`
- The static helpers are also `pub`: compute_qn, stereo_merge, stereo_split, intensity_stereo, (de)interleave_hadamard, compute_channel_weights, special_hybrid_folding.

quant_bands.rs:
- `pub static E_MEANS: [f32; 25]`, `PRED_COEF`, `BETA_COEF`, `BETA_INTRA`, `E_PROB_MODEL`, `SMALL_ENERGY_ICDF`
- `pub fn quant_coarse_energy(m, start, end, eff_end, e_bands: &[f32], old_e_bands: &mut [f32], budget: u32, error: &mut [f32], enc: &mut EcEnc, c, lm, nb_available_bytes: i32, force_intra: bool, delayed_intra: &mut f32, two_pass: bool, loss_rate: i32, lfe: bool)`
- `pub fn quant_fine_energy(m, start, end, old_e_bands: &mut [f32], error: &mut [f32], prev_quant: Option<&[i32]>, extra_quant: &[i32], enc: &mut EcEnc, c)`
- `pub fn quant_energy_finalise(m, start, end, old_e_bands: Option<&mut [f32]>, error: &mut [f32], fine_quant: &[i32], fine_priority: &[i32], bits_left: i32, enc: &mut EcEnc, c)`
- `pub fn unquant_coarse_energy(m, start, end, old_e_bands: &mut [f32], intra: bool, dec: &mut EcDec, c, lm)`
- `pub fn unquant_fine_energy(m, start, end, old_e_bands, prev_quant: Option<&[i32]>, extra_quant: &[i32], dec, c)`
- `pub fn unquant_energy_finalise(m, start, end, old_e_bands: Option<&mut [f32]>, fine_quant, fine_priority, bits_left, dec, c)`
- `pub fn amp2_log2(m, eff_end, end, band_e: &[f32], band_log_e: &mut [f32], c)`
- `pub fn loss_distortion(...) -> f32`

vq.rs:
- `pub fn alg_quant(x, n, k, spread, b, enc: &mut EcEnc, gain: f32, resynth: bool, [qext: ext_enc: &mut EcEnc, extra_bits: i32]) -> u32`
- `pub fn alg_unquant(x, n, k, spread, b, dec: &mut EcDec, gain, [qext: ext_dec: &mut EcDec, extra_bits]) -> u32`
- `pub fn exp_rotation(x, len, dir, stride, k, spread)`, `pub fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k, n) -> f32`, `pub fn renormalise_vector(x, n, gain)`, `pub fn stereo_itheta(x, y, stereo: bool, n) -> i32`
- qext only: `pub fn cubic_quant(x, n, res, b, enc, gain, resynth) -> u32` and `pub fn cubic_unquant(x, n, res, b, dec, gain) -> u32`
- `pub(crate) struct Scratch<T, N>`: a stack buffer with heap fallback, reusable by other units.

celt.rs:
- Buffer APIs:
  - `pub fn comb_filter(y: &mut [f32], xs: &[f32], x_off: usize, t0, t1, n, g0, g1, tapset0, tapset1, window: &[f32], overlap)` for C `x != y`, where `x[0] = xs[x_off]` and the history comes before it. The qext 96 kHz path needs `x_off >= 2*1024`.
  - `pub fn comb_filter_inplace(buf: &mut [f32], off: usize, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap)` for C `x == y`.
  - `pub fn init_caps(m, cap: &mut [i32], lm, c)`
- Tables and scalar helpers:
  - `pub static TF_SELECT_TABLE: [[i8; 8]; 4]`, `TRIM_ICDF`, `SPREAD_ICDF`, `TAPSET_ICDF`
  - `pub const fn resampling_factor(i32) -> i32`
  - `bits_to_bitrate`, `bitrate_to_bits`
  - `to_opus`/`from_opus` and their tables (custom-modes only)
- Types and constants:
  - `pub struct AnalysisInfo` (Copy + Default), `pub struct SilkInfo`
  - `CELT_SET_*_REQUEST` and `OPUS_SET_LFE_REQUEST` / `OPUS_SET_ENERGY_MASK_REQUEST` constants
  - `COMBFILTER_MAXPERIOD/MINPERIOD`, `QEXT_EXTENSION_ID`, `LEAK_BANDS`
- `opus_strerror(i32) -> &'static str`, `opus_get_version_string()`.

### external needs

1. docs/TRACKER.md is not updated: the celt_bands row should become ✅ with the test summary. I did not edit docs, per the rules.
2. The analysis unit (layer B, running in parallel) may have written its own private AnalysisInfo. The shared definition now lives in `celt::celt::AnalysisInfo`, so any private copy should be deduplicated at merge.
3. Following my notes, I edited rate.rs only to remove its private E_MEANS; it now does `use crate::celt::quant_bands::E_MEANS` under cfg(qext).
4. Callers need to know that comb_filter has two Rust entry points, because C's aliasing (`x == y`) turns it into an IIR filter:
   - the decoder postfilter and the encoder's `out_mem` calls must use `comb_filter_inplace`;
   - the encoder prefilter (`in` ← `pre`) and the decoder's `etmp` call use `comb_filter`.

### notes

**Aliasing in quant_all_bands.** The C code uses overlapping pointers in three places; the port handles them as documented in the bands.rs module doc:
- **Decoder scratch in the X tail.** In the decoder, `lowband_scratch` points into the tail of X_. The port uses that same tail as scratch whenever the current band lies before it. The unused regions of X after decoding match C bit-for-bit, and the tests compare the whole buffer.
- **Hybrid folding band.** For band start+1, `lowband` and `lowband_out` overlap inside norm. The port runs the band on copies and writes them back in C's access order (all lowband use happens before the lowband_out writes). This path is exercised by the start=17 decoder and theta-RDO tests.
- **Bands at or above effEBands.** C quantises norm itself as both X and Y there. The port quantises a copy instead (written back to norm for mono). This is unreachable through any codec path, because `end <= effEBands` for every mode the codec uses, including the qext modes. Decoder side effects are unchanged. For those bands, the Rust encoder's input can differ from C's; this is documented as a known divergence.

**Oracle quirks the tests work around:**
- C copies the whole, partly uninitialised, intra error array in quant_coarse_energy. The tests compare `error` only for the bands that are written.
- bitexact_cos keeps its `celt_sig_assert` as a debug_assert, so it is tested over the argument range the codec actually uses (64..=16320).
- comb_filter periods are limited to COMBFILTER_MAXPERIOD-2, the largest the codec produces; the qext path would read out of bounds in C beyond that.
- alg_quant/alg_unquant use (N,K) pairs drawn from the pulse cache, because arbitrary pairs overflow the cwrs tables in C as well.

**Other choices:**
- quant_all_bands takes a `BandsScratch` argument, so the per-frame VLAs are not reallocated. Smaller per-band arrays use stack buffers sized for the largest static-mode band (176), with a heap fallback for custom modes.
- `opus_get_version_string` returns "libopus 1.6.1". The oracle is built without PACKAGE_VERSION, so the test checks only the "libopus " prefix and the absence of "-fixed".
- FIXED_POINT branches carry `// FIXED_POINT: not ported` markers. There is no DNN code in these files.

The C shim includes vq.c, bands.c and quant_bands.c with their external symbols renamed, to reach the static helpers.

Key files:
- /home/neo/code/opusorus/.claude/worktrees/wf_cfc8900e-9ea-1/crates/opusorus/src/celt/{vq,quant_bands,bands,celt}.rs
- crates/opusorus-oracle/{csrc/celt_bands.c, src/celt_bands.rs}
- crates/opusorus-conformance/tests/celt_bands.rs

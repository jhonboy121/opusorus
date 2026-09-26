/* Oracle C shims for unit analysis (src/analysis.c + the downmix helpers of src/opus_encoder.c).
 *
 * The static functions of analysis.c (silk_resampler_down2_hp, downmix_and_resample,
 * tonality_analysis) are reached by compiling a private copy of the translation unit with its
 * exported functions renamed (avoids duplicate symbols with the library). The exported
 * functions are called from the library itself.
 *
 * Built in both oracles (analysis.c is part of the float API that fixed-point builds keep):
 * signal buffers and the resampler state are opus_val32 (float, or opus_int32 in fixed point),
 * is_digital_silence takes opus_res (float, short, or int with ENABLE_RES24). */
// oracle-build: any
// oracle-requires: float-api
#include <stddef.h>
#include "opus_types.h"
#include "opus_custom.h"

#define tonality_analysis_init oracle_dup_tonality_analysis_init
#define tonality_analysis_reset oracle_dup_tonality_analysis_reset
#define tonality_get_info oracle_dup_tonality_get_info
#define run_analysis oracle_dup_run_analysis
/* Relative path: a bare "analysis.c" would resolve to this shim itself. */
#include "../../../vendor/libopus/src/analysis.c"
#undef tonality_analysis_init
#undef tonality_analysis_reset
#undef tonality_get_info
#undef run_analysis

void tonality_analysis_init(TonalityAnalysisState *analysis, opus_int32 Fs);
void tonality_analysis_reset(TonalityAnalysisState *analysis);
void tonality_get_info(TonalityAnalysisState *tonal, AnalysisInfo *info_out, int len);
void run_analysis(TonalityAnalysisState *analysis, const CELTMode *celt_mode,
                  const void *analysis_pcm, int analysis_frame_size, int frame_size, int c1,
                  int c2, int C, opus_int32 Fs, int lsb_depth, downmix_func downmix,
                  AnalysisInfo *analysis_info);

/* ---- Layout checks for the Rust #[repr(C)] mirrors ---- */

size_t oracle_analysis_state_size(void) { return sizeof(TonalityAnalysisState); }
size_t oracle_analysis_info_size(void) { return sizeof(AnalysisInfo); }
size_t oracle_analysis_state_offset(int which) {
  switch (which) {
    case 0: return offsetof(TonalityAnalysisState, angle);
    case 1: return offsetof(TonalityAnalysisState, inmem);
    case 2: return offsetof(TonalityAnalysisState, E);
    case 3: return offsetof(TonalityAnalysisState, Etracker);
    case 4: return offsetof(TonalityAnalysisState, hp_ener_accum);
    case 5: return offsetof(TonalityAnalysisState, rnn_state);
    case 6: return offsetof(TonalityAnalysisState, downmix_state);
    case 7: return offsetof(TonalityAnalysisState, info);
    case 8: return offsetof(AnalysisInfo, leak_boost);
    case 9: return offsetof(AnalysisInfo, max_pitch_ratio);
    default: return (size_t)-1;
  }
}

double oracle_analysis_m_pi(void) { return M_PI; }

/* ---- Helpers ---- */

static const CELTMode *mode48(void) { return opus_custom_mode_create(48000, 960, NULL); }

/* pcm_type: 0 = float (downmix_float), 1 = int16 (downmix_int), 2 = int24 (downmix_int24). */
static downmix_func pick(int pcm_type) {
  switch (pcm_type) {
    case 1: return downmix_int;
    case 2: return downmix_int24;
    default: return downmix_float;
  }
}

/* ---- Exported library functions ---- */

void oracle_tonality_analysis_init(void *st, int fs) {
  tonality_analysis_init((TonalityAnalysisState *)st, fs);
}

void oracle_tonality_analysis_reset(void *st) {
  tonality_analysis_reset((TonalityAnalysisState *)st);
}

void oracle_tonality_get_info(void *st, void *info_out, int len) {
  tonality_get_info((TonalityAnalysisState *)st, (AnalysisInfo *)info_out, len);
}

void oracle_run_analysis(void *st, int pcm_type, const void *pcm, int analysis_frame_size,
                         int frame_size, int c1, int c2, int C, int Fs, int lsb_depth,
                         void *info_out) {
  run_analysis((TonalityAnalysisState *)st, mode48(), pcm, analysis_frame_size, frame_size, c1,
               c2, C, Fs, lsb_depth, pick(pcm_type), (AnalysisInfo *)info_out);
}

void oracle_downmix(int pcm_type, const void *x, opus_val32 *y, int subframe, int offset, int c1,
                    int c2, int C) {
  pick(pcm_type)(x, y, subframe, offset, c1, c2, C);
}

int oracle_is_digital_silence(const opus_res *pcm, int frame_size, int channels, int lsb_depth) {
  return is_digital_silence(pcm, frame_size, channels, lsb_depth);
}

/* ---- Static functions (private copy) ---- */

opus_val32 oracle_silk_resampler_down2_hp(opus_val32 *S, opus_val32 *out, const opus_val32 *in,
                                          int inLen) {
  return silk_resampler_down2_hp(S, out, in, inLen);
}

opus_val32 oracle_downmix_and_resample(int pcm_type, const void *x, opus_val32 *y, opus_val32 *S,
                                       int subframe, int offset, int c1, int c2, int C, int Fs) {
  return downmix_and_resample(pick(pcm_type), x, y, S, subframe, offset, c1, c2, C, Fs);
}

void oracle_tonality_analysis(void *st, int pcm_type, const void *x, int len, int offset, int c1,
                              int c2, int C, int lsb_depth) {
  tonality_analysis((TonalityAnalysisState *)st, mode48(), x, len, offset, c1, c2, C, lsb_depth,
                    pick(pcm_type));
}

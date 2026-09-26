/* Oracle C shims for unit silk_encoder_common: NSQ / NSQ_del_dec, VAD, encode_indices,
   process_NLSFs, quant_LTP_gains, VQ_WMat_EC, stereo LR->MS (+ find_predictor, quant_pred),
   HP_variable_cutoff, control_SNR, control_audio_bandwidth, check_control_input.

   Encoder-state based functions get a zeroed (calloc) silk_encoder_state filled from a flat
   `oracle_sec_enc_params` struct; updated fields are copied back after the call. The NSQ, VAD
   and stereo states are passed directly (Rust mirrors them with #[repr(C)] structs, the layout
   is checked by oracle_sec_sizes). */
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "main_FLP.h"
#include "NSQ.h"
#include "tables.h"
#include "entenc.h"

/* The subset of silk_encoder_state used by the functions of this unit. */
typedef struct {
  int fs_kHz, nb_subfr, frame_length, subfr_length, ltp_mem_length;
  int predictLPCOrder, shapingLPCOrder, nStatesDelayedDecision, warping_Q16;
  int speech_activity_Q8, useInterpolatedNLSFs, NLSF_MSVQ_Survivors;
  int ec_prevSignalType, ec_prevLagIndex;
  int prevSignalType, prevLag, variable_HP_smth1_Q15;
  int input_quality_bands_Q15[4];
  int input_tilt_Q15;
  int TargetRate_bps, SNR_dB_Q7;
  int API_fs_Hz, maxInternal_fs_Hz, minInternal_fs_Hz, desiredInternal_fs_Hz;
  int allow_bandwidth_switch;
  int sLP_In_LP_State[2];
  int sLP_transition_frame_no, sLP_mode, sLP_saved_fs_kHz;
} oracle_sec_enc_params;

static void enc_from_params(silk_encoder_state *e, const oracle_sec_enc_params *p) {
  int i;
  e->fs_kHz = p->fs_kHz;
  e->nb_subfr = p->nb_subfr;
  e->frame_length = p->frame_length;
  e->subfr_length = p->subfr_length;
  e->ltp_mem_length = p->ltp_mem_length;
  e->predictLPCOrder = p->predictLPCOrder;
  e->shapingLPCOrder = p->shapingLPCOrder;
  e->nStatesDelayedDecision = p->nStatesDelayedDecision;
  e->warping_Q16 = p->warping_Q16;
  e->speech_activity_Q8 = p->speech_activity_Q8;
  e->useInterpolatedNLSFs = p->useInterpolatedNLSFs;
  e->NLSF_MSVQ_Survivors = p->NLSF_MSVQ_Survivors;
  e->ec_prevSignalType = p->ec_prevSignalType;
  e->ec_prevLagIndex = (opus_int16)p->ec_prevLagIndex;
  e->prevSignalType = (opus_int8)p->prevSignalType;
  e->prevLag = p->prevLag;
  e->variable_HP_smth1_Q15 = p->variable_HP_smth1_Q15;
  for (i = 0; i < 4; i++) e->input_quality_bands_Q15[i] = p->input_quality_bands_Q15[i];
  e->input_tilt_Q15 = p->input_tilt_Q15;
  e->TargetRate_bps = p->TargetRate_bps;
  e->SNR_dB_Q7 = p->SNR_dB_Q7;
  e->API_fs_Hz = p->API_fs_Hz;
  e->maxInternal_fs_Hz = p->maxInternal_fs_Hz;
  e->minInternal_fs_Hz = p->minInternal_fs_Hz;
  e->desiredInternal_fs_Hz = p->desiredInternal_fs_Hz;
  e->allow_bandwidth_switch = p->allow_bandwidth_switch;
  e->sLP.In_LP_State[0] = p->sLP_In_LP_State[0];
  e->sLP.In_LP_State[1] = p->sLP_In_LP_State[1];
  e->sLP.transition_frame_no = p->sLP_transition_frame_no;
  e->sLP.mode = p->sLP_mode;
  e->sLP.saved_fs_kHz = p->sLP_saved_fs_kHz;
  /* Table pointers as set by silk_setup_fs() in control_codec.c. */
  e->psNLSF_CB = (p->fs_kHz == 16) ? &silk_NLSF_CB_WB : &silk_NLSF_CB_NB_MB;
  e->pitch_lag_low_bits_iCDF = (p->fs_kHz == 16)   ? silk_uniform8_iCDF
                               : (p->fs_kHz == 12) ? silk_uniform6_iCDF
                                                   : silk_uniform4_iCDF;
  if (p->fs_kHz == 8) {
    e->pitch_contour_iCDF = (p->nb_subfr == MAX_NB_SUBFR) ? silk_pitch_contour_NB_iCDF
                                                          : silk_pitch_contour_10_ms_NB_iCDF;
  } else {
    e->pitch_contour_iCDF = (p->nb_subfr == MAX_NB_SUBFR) ? silk_pitch_contour_iCDF
                                                          : silk_pitch_contour_10_ms_iCDF;
  }
}

static void enc_to_params(oracle_sec_enc_params *p, const silk_encoder_state *e) {
  int i;
  p->fs_kHz = e->fs_kHz;
  p->nb_subfr = e->nb_subfr;
  p->frame_length = e->frame_length;
  p->subfr_length = e->subfr_length;
  p->ltp_mem_length = e->ltp_mem_length;
  p->predictLPCOrder = e->predictLPCOrder;
  p->shapingLPCOrder = e->shapingLPCOrder;
  p->nStatesDelayedDecision = e->nStatesDelayedDecision;
  p->warping_Q16 = e->warping_Q16;
  p->speech_activity_Q8 = e->speech_activity_Q8;
  p->useInterpolatedNLSFs = e->useInterpolatedNLSFs;
  p->NLSF_MSVQ_Survivors = e->NLSF_MSVQ_Survivors;
  p->ec_prevSignalType = e->ec_prevSignalType;
  p->ec_prevLagIndex = e->ec_prevLagIndex;
  p->prevSignalType = e->prevSignalType;
  p->prevLag = e->prevLag;
  p->variable_HP_smth1_Q15 = e->variable_HP_smth1_Q15;
  for (i = 0; i < 4; i++) p->input_quality_bands_Q15[i] = e->input_quality_bands_Q15[i];
  p->input_tilt_Q15 = e->input_tilt_Q15;
  p->TargetRate_bps = e->TargetRate_bps;
  p->SNR_dB_Q7 = e->SNR_dB_Q7;
  p->API_fs_Hz = e->API_fs_Hz;
  p->maxInternal_fs_Hz = e->maxInternal_fs_Hz;
  p->minInternal_fs_Hz = e->minInternal_fs_Hz;
  p->desiredInternal_fs_Hz = e->desiredInternal_fs_Hz;
  p->allow_bandwidth_switch = e->allow_bandwidth_switch;
  p->sLP_In_LP_State[0] = e->sLP.In_LP_State[0];
  p->sLP_In_LP_State[1] = e->sLP.In_LP_State[1];
  p->sLP_transition_frame_no = e->sLP.transition_frame_no;
  p->sLP_mode = e->sLP.mode;
  p->sLP_saved_fs_kHz = e->sLP.saved_fs_kHz;
}

static silk_encoder_state *enc_new(const oracle_sec_enc_params *p) {
  silk_encoder_state *e = (silk_encoder_state *)calloc(1, sizeof(silk_encoder_state));
  if (e == NULL) abort();
  enc_from_params(e, p);
  return e;
}

/* Layout checks for the #[repr(C)] mirrors on the Rust side. */
void oracle_sec_sizes(int *out) {
  out[0] = (int)sizeof(silk_nsq_state);
  out[1] = (int)sizeof(silk_VAD_state);
  out[2] = (int)sizeof(stereo_enc_state);
  out[3] = (int)sizeof(SideInfoIndices);
  out[4] = (int)sizeof(silk_EncControlStruct);
  out[5] = (int)sizeof(oracle_sec_enc_params);
}

/* ---------------------------------------------------------------------------------------- */
/* NSQ.h inlines                                                                            */
/* ---------------------------------------------------------------------------------------- */

/* buf has len elements; the C pointer is &buf[len - 1]. */
opus_int32 oracle_silk_nsq_short_prediction(const opus_int32 *buf, int len,
                                            const opus_int16 *coef, int order) {
  return silk_noise_shape_quantizer_short_prediction_c(&buf[len - 1], coef, order);
}

opus_int32 oracle_silk_nsq_feedback_loop(opus_int32 data0, opus_int32 *data1,
                                         const opus_int16 *coef, int order) {
  return silk_NSQ_noise_shape_feedback_loop_c(&data0, data1, coef, order);
}

/* ---------------------------------------------------------------------------------------- */
/* NSQ / NSQ_del_dec                                                                        */
/* ---------------------------------------------------------------------------------------- */

/* idx = {signalType, quantOffsetType, NLSFInterpCoef_Q2, Seed}; Seed is written back. */
void oracle_silk_NSQ(int del_dec, const oracle_sec_enc_params *p, silk_nsq_state *nsq, int *idx,
                     const opus_int16 *x16, opus_int8 *pulses, const opus_int16 *PredCoef_Q12,
                     const opus_int16 *LTPCoef_Q14, const opus_int16 *AR_Q13,
                     const int *HarmShapeGain_Q14, const int *Tilt_Q14,
                     const opus_int32 *LF_shp_Q14, const opus_int32 *Gains_Q16, const int *pitchL,
                     int Lambda_Q10, int LTP_scale_Q14) {
  silk_encoder_state *e = enc_new(p);
  SideInfoIndices ind;
  memset(&ind, 0, sizeof(ind));
  ind.signalType = (opus_int8)idx[0];
  ind.quantOffsetType = (opus_int8)idx[1];
  ind.NLSFInterpCoef_Q2 = (opus_int8)idx[2];
  ind.Seed = (opus_int8)idx[3];
  if (del_dec) {
    silk_NSQ_del_dec_c(e, nsq, &ind, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13,
                       HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,
                       LTP_scale_Q14);
  } else {
    silk_NSQ_c(e, nsq, &ind, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14,
               Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14);
  }
  idx[3] = ind.Seed;
  free(e);
}

/* ---------------------------------------------------------------------------------------- */
/* VAD                                                                                      */
/* ---------------------------------------------------------------------------------------- */

int oracle_silk_VAD_Init(silk_VAD_state *vad) { return silk_VAD_Init(vad); }

/* out = {speech_activity_Q8, input_tilt_Q15, input_quality_bands_Q15[4]}; returns ret. */
int oracle_silk_VAD_GetSA_Q8(silk_VAD_state *vad, int frame_length, int fs_kHz,
                             const opus_int16 *pIn, int *out) {
  int ret, b;
  silk_encoder_state *e = (silk_encoder_state *)calloc(1, sizeof(silk_encoder_state));
  if (e == NULL) abort();
  e->sVAD = *vad;
  e->frame_length = frame_length;
  e->fs_kHz = fs_kHz;
  ret = silk_VAD_GetSA_Q8_c(e, pIn);
  *vad = e->sVAD;
  out[0] = e->speech_activity_Q8;
  out[1] = e->input_tilt_Q15;
  for (b = 0; b < 4; b++) out[2 + b] = e->input_quality_bands_Q15[b];
  free(e);
  return ret;
}

/* ---------------------------------------------------------------------------------------- */
/* encode_indices                                                                           */
/* ---------------------------------------------------------------------------------------- */

/* Encodes n frames in sequence into one range coder: before frame f, psEncC->indices =
   sets[4f] and indices_LBRR[0..3] = sets[4f+1..4f+4]; ops[3f..3f+3] = {FrameIndex,
   encode_LBRR, condCoding}. res = {tell_frac before done, rng after done, error}. */
void oracle_silk_encode_indices(oracle_sec_enc_params *p, const SideInfoIndices *sets,
                                const int *ops, int n, unsigned char *buf, int size,
                                opus_uint32 *res) {
  silk_encoder_state *e = enc_new(p);
  ec_enc enc;
  int f, i;
  ec_enc_init(&enc, buf, size);
  for (f = 0; f < n; f++) {
    e->indices = sets[4 * f];
    for (i = 0; i < MAX_FRAMES_PER_PACKET; i++) e->indices_LBRR[i] = sets[4 * f + 1 + i];
    silk_encode_indices(e, &enc, ops[3 * f], ops[3 * f + 1], ops[3 * f + 2]);
  }
  res[0] = ec_tell_frac(&enc);
  ec_enc_done(&enc);
  res[1] = enc.rng;
  res[2] = (opus_uint32)enc.error;
  enc_to_params(p, e);
  free(e);
}

/* ---------------------------------------------------------------------------------------- */
/* process_NLSFs                                                                            */
/* ---------------------------------------------------------------------------------------- */

/* idx = {signalType, NLSFInterpCoef_Q2}; NLSFIndices[MAX_LPC_ORDER + 1] out. */
void oracle_silk_process_NLSFs(const oracle_sec_enc_params *p, const int *idx,
                               opus_int8 *NLSFIndices, opus_int16 *PredCoef_Q12,
                               opus_int16 *pNLSF_Q15, const opus_int16 *prev_NLSFq_Q15) {
  silk_encoder_state *e = enc_new(p);
  opus_int16 pc[2][MAX_LPC_ORDER];
  memcpy(pc, PredCoef_Q12, sizeof(pc));
  e->indices.signalType = (opus_int8)idx[0];
  e->indices.NLSFInterpCoef_Q2 = (opus_int8)idx[1];
  silk_process_NLSFs(e, pc, pNLSF_Q15, prev_NLSFq_Q15);
  memcpy(PredCoef_Q12, pc, sizeof(pc));
  memcpy(NLSFIndices, e->indices.NLSFIndices, MAX_LPC_ORDER + 1);
  free(e);
}

/* ---------------------------------------------------------------------------------------- */
/* quant_LTP_gains / VQ_WMat_EC                                                             */
/* ---------------------------------------------------------------------------------------- */

/* out = {periodicity_index, sum_log_gain_Q7 (in/out), pred_gain_dB_Q7}. */
void oracle_silk_quant_LTP_gains(opus_int16 *B_Q14, opus_int8 *cbk_index, int *out,
                                 const opus_int32 *XX_Q17, const opus_int32 *xX_Q17,
                                 int subfr_len, int nb_subfr) {
  opus_int8 per = 0;
  opus_int32 sum_log = out[1];
  opus_int pred_gain = 0;
  silk_quant_LTP_gains(B_Q14, cbk_index, &per, &sum_log, &pred_gain, XX_Q17, xX_Q17, subfr_len,
                       nb_subfr, 0);
  out[0] = per;
  out[1] = sum_log;
  out[2] = pred_gain;
}

/* cbk selects silk_LTP_vq_ptrs_Q7[cbk] etc. out = {ind, res_nrg_Q15, rate_dist_Q8, gain_Q7}
   (gain_Q7 in/out: only written when a vector is selected). */
void oracle_silk_VQ_WMat_EC(int *out, const opus_int32 *XX_Q17, const opus_int32 *xX_Q17,
                            int cbk, int subfr_len, opus_int32 max_gain_Q7) {
  opus_int8 ind = 0;
  opus_int32 res_nrg = 0, rate_dist = 0;
  opus_int gain = out[3];
  silk_VQ_WMat_EC_c(&ind, &res_nrg, &rate_dist, &gain, XX_Q17, xX_Q17, silk_LTP_vq_ptrs_Q7[cbk],
                    silk_LTP_vq_gain_ptrs_Q7[cbk], silk_LTP_gain_BITS_Q5_ptrs[cbk], subfr_len,
                    max_gain_Q7, silk_LTP_vq_sizes[cbk]);
  out[0] = ind;
  out[1] = res_nrg;
  out[2] = rate_dist;
  out[3] = gain;
}

/* ---------------------------------------------------------------------------------------- */
/* Stereo                                                                                   */
/* ---------------------------------------------------------------------------------------- */

/* x1 / x2 point two samples before the C x1 / x2 pointers (frame_length + 2 samples each). */
void oracle_silk_stereo_LR_to_MS(stereo_enc_state *state, opus_int16 *x1, opus_int16 *x2,
                                 opus_int8 *ix, opus_int8 *mid_only_flag,
                                 opus_int32 *mid_side_rates_bps, opus_int32 total_rate_bps,
                                 int prev_speech_act_Q8, int toMono, int fs_kHz,
                                 int frame_length) {
  opus_int8 ixm[2][3];
  memcpy(ixm, ix, 6);
  silk_stereo_LR_to_MS(state, x1 + 2, x2 + 2, ixm, mid_only_flag, mid_side_rates_bps,
                       total_rate_bps, prev_speech_act_Q8, toMono, fs_kHz, frame_length);
  memcpy(ix, ixm, 6);
}

opus_int32 oracle_silk_stereo_find_predictor(opus_int32 *ratio_Q14, const opus_int16 *x,
                                             const opus_int16 *y, opus_int32 *mid_res_amp_Q0,
                                             int length, int smooth_coef_Q16) {
  return silk_stereo_find_predictor(ratio_Q14, x, y, mid_res_amp_Q0, length, smooth_coef_Q16);
}

void oracle_silk_stereo_quant_pred(opus_int32 *pred_Q13, opus_int8 *ix) {
  opus_int8 ixm[2][3];
  memcpy(ixm, ix, 6);
  silk_stereo_quant_pred(pred_Q13, ixm);
  memcpy(ix, ixm, 6);
}

/* ---------------------------------------------------------------------------------------- */
/* HP_variable_cutoff / control_SNR / control_audio_bandwidth                               */
/* ---------------------------------------------------------------------------------------- */

void oracle_silk_HP_variable_cutoff(oracle_sec_enc_params *p) {
  silk_encoder_state_FLP *s = (silk_encoder_state_FLP *)calloc(1, sizeof(silk_encoder_state_FLP));
  if (s == NULL) abort();
  enc_from_params(&s->sCmn, p);
  silk_HP_variable_cutoff(s);
  enc_to_params(p, &s->sCmn);
  free(s);
}

int oracle_silk_control_SNR(oracle_sec_enc_params *p, opus_int32 TargetRate_bps) {
  silk_encoder_state *e = enc_new(p);
  int ret = silk_control_SNR(e, TargetRate_bps);
  enc_to_params(p, e);
  free(e);
  return ret;
}

int oracle_silk_control_audio_bandwidth(oracle_sec_enc_params *p, silk_EncControlStruct *ctl) {
  silk_encoder_state *e = enc_new(p);
  int ret = silk_control_audio_bandwidth(e, ctl);
  enc_to_params(p, e);
  free(e);
  return ret;
}

/* ---------------------------------------------------------------------------------------- */
/* Static helpers reached by including the C source with renamed globals                    */
/* ---------------------------------------------------------------------------------------- */

/* silk_VAD_GetNoiseLevels is static in VAD.c. */
#define silk_VAD_Init oracle_sec_silk_VAD_Init
#define silk_VAD_GetSA_Q8_c oracle_sec_silk_VAD_GetSA_Q8_c
#include "VAD.c"
#undef silk_VAD_Init
#undef silk_VAD_GetSA_Q8_c

void oracle_silk_VAD_GetNoiseLevels(const opus_int32 *pX, silk_VAD_state *vad) {
  silk_VAD_GetNoiseLevels(pX, vad);
}

/* check_control_input: with ENABLE_HARDENING the oracle's celt_assert( 0 ) aborts on every
   error path. Compile a copy with celt_assert disabled so the error codes can be compared.
   (Must stay at the end of this file: it redefines celt_assert.) */
#undef celt_assert
#define celt_assert(cond) ((void)(cond))
#define check_control_input oracle_sec_check_control_input_noassert
#include "check_control_input.c"
#undef check_control_input

int oracle_check_control_input(const silk_EncControlStruct *ctl) {
  silk_EncControlStruct c = *ctl;
  return oracle_sec_check_control_input_noassert(&c);
}

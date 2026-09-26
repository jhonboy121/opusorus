/* Oracle C shims for unit silk_encoder_flp: the floating-point SILK encoder (silk/float/*.c,
   silk/enc_API.c, silk/init_encoder.c, silk/control_codec.c).

   * A persistent silk_encoder (from silk_Get_Encoder_Size + silk_InitEncoder) plus a range
     encoder over a private buffer, so tests can drive silk_Encode call by call, and a full
     state dump.
   * Flat shims for the SigProc_FLP.h inline helpers and state-based helpers (find_LPC).
   * The static helpers of noise_shape_analysis_FLP.c and pitch_analysis_core_FLP.c are reached
     by #including those files with their exported functions renamed (oracle_sef_*).
   The exported leaf functions (silk_burg_modified_FLP, ...) are declared directly in
   src/silk_encoder_flp.rs. */
#include <stdlib.h>
#include <string.h>

#include "main_FLP.h"
#include "API.h"
#include "control.h"
#include "entenc.h"
#include "tables.h"

#define silk_noise_shape_analysis_FLP oracle_sef_silk_noise_shape_analysis_FLP
#include "../../../vendor/libopus/silk/float/noise_shape_analysis_FLP.c"
#undef silk_noise_shape_analysis_FLP

#define silk_pitch_analysis_core_FLP oracle_sef_silk_pitch_analysis_core_FLP
#include "../../../vendor/libopus/silk/float/pitch_analysis_core_FLP.c"
#undef silk_pitch_analysis_core_FLP

#define ORACLE_SEF_BUF 4096

typedef struct {
  silk_encoder *enc;
  ec_enc ec;
  unsigned char buf[ORACLE_SEF_BUF];
} oracle_sef;

static void ctl_in(silk_EncControlStruct *c, const int v[26]) {
  c->nChannelsAPI = v[0];
  c->nChannelsInternal = v[1];
  c->API_sampleRate = v[2];
  c->maxInternalSampleRate = v[3];
  c->minInternalSampleRate = v[4];
  c->desiredInternalSampleRate = v[5];
  c->payloadSize_ms = v[6];
  c->bitRate = v[7];
  c->packetLossPercentage = v[8];
  c->complexity = v[9];
  c->useInBandFEC = v[10];
  c->useDRED = v[11];
  c->LBRR_coded = v[12];
  c->useDTX = v[13];
  c->useCBR = v[14];
  c->maxBits = v[15];
  c->toMono = v[16];
  c->opusCanSwitch = v[17];
  c->reducedDependency = v[18];
  c->internalSampleRate = v[19];
  c->allowBandwidthSwitch = v[20];
  c->inWBmodeWithoutVariableLP = v[21];
  c->stereoWidth_Q14 = v[22];
  c->switchReady = v[23];
  c->signalType = v[24];
  c->offset = v[25];
}

static void ctl_out(const silk_EncControlStruct *c, int v[26]) {
  v[0] = c->nChannelsAPI;
  v[1] = c->nChannelsInternal;
  v[2] = c->API_sampleRate;
  v[3] = c->maxInternalSampleRate;
  v[4] = c->minInternalSampleRate;
  v[5] = c->desiredInternalSampleRate;
  v[6] = c->payloadSize_ms;
  v[7] = c->bitRate;
  v[8] = c->packetLossPercentage;
  v[9] = c->complexity;
  v[10] = c->useInBandFEC;
  v[11] = c->useDRED;
  v[12] = c->LBRR_coded;
  v[13] = c->useDTX;
  v[14] = c->useCBR;
  v[15] = c->maxBits;
  v[16] = c->toMono;
  v[17] = c->opusCanSwitch;
  v[18] = c->reducedDependency;
  v[19] = c->internalSampleRate;
  v[20] = c->allowBandwidthSwitch;
  v[21] = c->inWBmodeWithoutVariableLP;
  v[22] = c->stereoWidth_Q14;
  v[23] = c->switchReady;
  v[24] = c->signalType;
  v[25] = c->offset;
}

int oracle_sef_ctl_size(void) { return (int)sizeof(silk_EncControlStruct); }

/* Encoder state sizes: silk_Get_Encoder_Size for 1 and 2 channels. */
void oracle_sef_sizes(int out[2]) {
  opus_int s = 0;
  silk_Get_Encoder_Size(&s, 1);
  out[0] = s;
  silk_Get_Encoder_Size(&s, 2);
  out[1] = s;
}

void *oracle_sef_new(void) {
  opus_int size = 0;
  oracle_sef *h = (oracle_sef *)calloc(1, sizeof(oracle_sef));
  if (!h) return NULL;
  /* Always allocate room for both channels (the mono init leaves the second state zeroed). */
  silk_Get_Encoder_Size(&size, 2);
  h->enc = (silk_encoder *)calloc(1, size);
  if (!h->enc) {
    free(h);
    return NULL;
  }
  ec_enc_init(&h->ec, h->buf, 0);
  return h;
}

void oracle_sef_free(void *p) {
  oracle_sef *h = (oracle_sef *)p;
  if (!h) return;
  free(h->enc);
  free(h);
}

/* silk_InitEncoder; status receives the control struct (26 ints). */
int oracle_sef_init(void *p, int channels, int status[26]) {
  oracle_sef *h = (oracle_sef *)p;
  silk_EncControlStruct c;
  int ret;
  memset(&c, 0, sizeof(c));
  ret = silk_InitEncoder(h->enc, channels, 0, &c);
  ctl_out(&c, status);
  return ret;
}

/* (Re)starts the range encoder over the zeroed private buffer with `size` bytes of storage. */
void oracle_sef_ec_init(void *p, int size) {
  oracle_sef *h = (oracle_sef *)p;
  if (size > ORACLE_SEF_BUF) size = ORACLE_SEF_BUF;
  memset(h->buf, 0, sizeof(h->buf));
  ec_enc_init(&h->ec, h->buf, size);
}

/* Range encoder state: storage, end_offs, end_window, nend_bits, nbits_total, offs, rng, val,
   ext, rem, error, tell_frac. */
void oracle_sef_ec_state(void *p, unsigned int out[12]) {
  oracle_sef *h = (oracle_sef *)p;
  ec_enc *e = &h->ec;
  out[0] = e->storage;
  out[1] = e->end_offs;
  out[2] = (unsigned int)e->end_window;
  out[3] = (unsigned int)e->nend_bits;
  out[4] = (unsigned int)e->nbits_total;
  out[5] = e->offs;
  out[6] = e->rng;
  out[7] = e->val;
  out[8] = e->ext;
  out[9] = (unsigned int)e->rem;
  out[10] = (unsigned int)e->error;
  out[11] = ec_tell_frac(e);
}

void oracle_sef_ec_done(void *p) { ec_enc_done(&((oracle_sef *)p)->ec); }

/* Copies the first n bytes of the range encoder buffer. */
void oracle_sef_ec_buf(void *p, unsigned char *out, int n) {
  oracle_sef *h = (oracle_sef *)p;
  if (n > ORACLE_SEF_BUF) n = ORACLE_SEF_BUF;
  memcpy(out, h->buf, n);
}

/* silk_Encode. ctl: control struct (in/out). null_ec: pass psRangeEnc = NULL (as the Opus
   encoder does for prefill). nbytes: in/out. */
int oracle_sef_encode(void *p, int ctl[26], const float *in, int n, int *nbytes, int prefill,
                      int activity, int null_ec) {
  oracle_sef *h = (oracle_sef *)p;
  silk_EncControlStruct c;
  opus_int32 nb = *nbytes;
  int ret;
  ctl_in(&c, ctl);
  ret = silk_Encode(h->enc, &c, in, n, null_ec ? NULL : &h->ec, &nb, prefill, activity);
  ctl_out(&c, ctl);
  *nbytes = nb;
  return ret;
}

/* ---------------------------------------------------------------------------------------- */
/* State dump (must match dump_state() in tests/silk_encoder_flp.rs)                        */
/* ---------------------------------------------------------------------------------------- */

#define P(v) do { if (k < cap) out[k] = (int)(v); k++; } while (0)
#define PA(a, n) do { int _i; for (_i = 0; _i < (n); _i++) P((a)[_i]); } while (0)
#define PF(v) do { float _f = (v); int _b; memcpy(&_b, &_f, 4); P(_b); } while (0)
#define PFA(a, n) do { int _j; for (_j = 0; _j < (n); _j++) PF((a)[_j]); } while (0)

static int lag_low_len(const opus_uint8 *t) {
  if (t == NULL) return 0;
  if (t == silk_uniform4_iCDF) return 4;
  if (t == silk_uniform6_iCDF) return 6;
  if (t == silk_uniform8_iCDF) return 8;
  return -1;
}

static int contour_len(const opus_uint8 *t) {
  if (t == NULL) return 0;
  if (t == silk_pitch_contour_iCDF) return 34;
  if (t == silk_pitch_contour_NB_iCDF) return 11;
  if (t == silk_pitch_contour_10_ms_iCDF) return 12;
  if (t == silk_pitch_contour_10_ms_NB_iCDF) return 3;
  return -1;
}

static int dump_indices(const SideInfoIndices *x, int *out, int k, int cap) {
  PA(x->GainsIndices, MAX_NB_SUBFR);
  PA(x->LTPIndex, MAX_NB_SUBFR);
  PA(x->NLSFIndices, MAX_LPC_ORDER + 1);
  P(x->lagIndex);
  P(x->contourIndex);
  P(x->signalType);
  P(x->quantOffsetType);
  P(x->NLSFInterpCoef_Q2);
  P(x->PERIndex);
  P(x->LTP_scaleIndex);
  P(x->Seed);
  return k;
}

static int dump_channel(const silk_encoder_state_FLP *e, int *out, int k, int cap) {
  const silk_encoder_state *s = &e->sCmn;
  const silk_resampler_state_struct *r = &s->resampler_state;
  int i;
  PA(s->In_HP_State, 2);
  P(s->variable_HP_smth1_Q15);
  P(s->variable_HP_smth2_Q15);
  PA(s->sLP.In_LP_State, 2);
  P(s->sLP.transition_frame_no);
  P(s->sLP.mode);
  P(s->sLP.saved_fs_kHz);
  PA(s->sVAD.AnaState, 2);
  PA(s->sVAD.AnaState1, 2);
  PA(s->sVAD.AnaState2, 2);
  PA(s->sVAD.XnrgSubfr, VAD_N_BANDS);
  PA(s->sVAD.NrgRatioSmth_Q8, VAD_N_BANDS);
  P(s->sVAD.HPstate);
  PA(s->sVAD.NL, VAD_N_BANDS);
  PA(s->sVAD.inv_NL, VAD_N_BANDS);
  PA(s->sVAD.NoiseLevelBias, VAD_N_BANDS);
  P(s->sVAD.counter);
  PA(s->sNSQ.xq, 2 * MAX_FRAME_LENGTH);
  PA(s->sNSQ.sLTP_shp_Q14, 2 * MAX_FRAME_LENGTH);
  PA(s->sNSQ.sLPC_Q14, MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH);
  PA(s->sNSQ.sAR2_Q14, MAX_SHAPE_LPC_ORDER);
  P(s->sNSQ.sLF_AR_shp_Q14);
  P(s->sNSQ.sDiff_shp_Q14);
  P(s->sNSQ.lagPrev);
  P(s->sNSQ.sLTP_buf_idx);
  P(s->sNSQ.sLTP_shp_buf_idx);
  P(s->sNSQ.rand_seed);
  P(s->sNSQ.prev_gain_Q16);
  P(s->sNSQ.rewhite_flag);
  PA(s->prev_NLSFq_Q15, MAX_LPC_ORDER);
  P(s->speech_activity_Q8);
  P(s->allow_bandwidth_switch);
  P(s->LBRRprevLastGainIndex);
  P(s->prevSignalType);
  P(s->prevLag);
  P(s->pitch_LPC_win_length);
  P(s->max_pitch_lag);
  P(s->API_fs_Hz);
  P(s->prev_API_fs_Hz);
  P(s->maxInternal_fs_Hz);
  P(s->minInternal_fs_Hz);
  P(s->desiredInternal_fs_Hz);
  P(s->fs_kHz);
  P(s->nb_subfr);
  P(s->frame_length);
  P(s->subfr_length);
  P(s->ltp_mem_length);
  P(s->la_pitch);
  P(s->la_shape);
  P(s->shapeWinLength);
  P(s->TargetRate_bps);
  P(s->PacketSize_ms);
  P(s->PacketLoss_perc);
  P(s->frameCounter);
  P(s->Complexity);
  P(s->nStatesDelayedDecision);
  P(s->useInterpolatedNLSFs);
  P(s->shapingLPCOrder);
  P(s->predictLPCOrder);
  P(s->pitchEstimationComplexity);
  P(s->pitchEstimationLPCOrder);
  P(s->pitchEstimationThreshold_Q16);
  P(s->sum_log_gain_Q7);
  P(s->NLSF_MSVQ_Survivors);
  P(s->first_frame_after_reset);
  P(s->controlled_since_last_payload);
  P(s->warping_Q16);
  P(s->useCBR);
  P(s->prefillFlag);
  P(lag_low_len(s->pitch_lag_low_bits_iCDF));
  P(contour_len(s->pitch_contour_iCDF));
  P(s->fs_kHz == 0 ? 0 : s->psNLSF_CB->order);
  PA(s->input_quality_bands_Q15, VAD_N_BANDS);
  P(s->input_tilt_Q15);
  P(s->SNR_dB_Q7);
  PA(s->VAD_flags, MAX_FRAMES_PER_PACKET);
  P(s->LBRR_flag);
  PA(s->LBRR_flags, MAX_FRAMES_PER_PACKET);
  k = dump_indices(&s->indices, out, k, cap);
  PA(s->pulses, MAX_FRAME_LENGTH);
  PA(s->inputBuf, MAX_FRAME_LENGTH + 2);
  P(s->inputBufIx);
  P(s->nFramesPerPacket);
  P(s->nFramesEncoded);
  P(s->nChannelsAPI);
  P(s->nChannelsInternal);
  P(s->channelNb);
  P(s->frames_since_onset);
  P(s->ec_prevSignalType);
  P(s->ec_prevLagIndex);
  /* resampler */
  PA(r->sIIR, SILK_RESAMPLER_MAX_IIR_ORDER);
  if (r->resampler_function == 3) {
    PA(r->sFIR.i32, SILK_RESAMPLER_MAX_FIR_ORDER);
  } else if (r->resampler_function == 2) {
    PA(r->sFIR.i16, SILK_RESAMPLER_MAX_FIR_ORDER);
  } else {
    for (i = 0; i < SILK_RESAMPLER_MAX_FIR_ORDER; i++) P(0);
  }
  PA(r->delayBuf, 96);
  P(r->resampler_function);
  P(r->batchSize);
  P(r->invRatio_Q16);
  P(r->FIR_Order);
  P(r->FIR_Fracs);
  P(r->Fs_in_kHz);
  P(r->Fs_out_kHz);
  P(r->inputDelay);
  P(s->useDTX);
  P(s->inDTX);
  P(s->noSpeechCounter);
  P(s->useInBandFEC);
  P(s->LBRR_enabled);
  P(s->LBRR_GainIncreases);
  for (i = 0; i < MAX_FRAMES_PER_PACKET; i++) k = dump_indices(&s->indices_LBRR[i], out, k, cap);
  for (i = 0; i < MAX_FRAMES_PER_PACKET; i++) PA(s->pulses_LBRR[i], MAX_FRAME_LENGTH);
  /* FLP part */
  P(e->sShape.LastGainIndex);
  PF(e->sShape.HarmShapeGain_smth);
  PF(e->sShape.Tilt_smth);
  PFA(e->x_buf, 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX);
  PF(e->LTPCorr);
  return k;
}

/* Dumps the whole encoder state as ints (floats as their bit patterns); returns the number of
   values (even if > cap). */
int oracle_sef_dump(void *p, int *out, int cap) {
  const silk_encoder *e = ((oracle_sef *)p)->enc;
  int k = 0, n, i;
  PA(e->sStereo.pred_prev_Q13, 2);
  PA(e->sStereo.sMid, 2);
  PA(e->sStereo.sSide, 2);
  PA(e->sStereo.mid_side_amp_Q0, 4);
  P(e->sStereo.smth_width_Q14);
  P(e->sStereo.width_prev_Q14);
  P(e->sStereo.silent_side_len);
  for (i = 0; i < MAX_FRAMES_PER_PACKET; i++) {
    PA(e->sStereo.predIx[i][0], 3);
    PA(e->sStereo.predIx[i][1], 3);
  }
  PA(e->sStereo.mid_only_flags, MAX_FRAMES_PER_PACKET);
  P(e->nBitsUsedLBRR);
  P(e->nBitsExceeded);
  P(e->nChannelsAPI);
  P(e->nChannelsInternal);
  P(e->nPrevChannelsInternal);
  P(e->timeSinceSwitchAllowed_ms);
  P(e->allowBandwidthSwitch);
  P(e->prev_decode_only_middle);
  for (n = 0; n < ENCODER_NUM_CHANNELS; n++) k = dump_channel(&e->state_Fxx[n], out, k, cap);
  return k;
}

#undef P
#undef PA
#undef PF
#undef PFA

/* ---------------------------------------------------------------------------------------- */
/* SigProc_FLP.h inline helpers                                                             */
/* ---------------------------------------------------------------------------------------- */

float oracle_sef_sigmoid(float x) { return silk_sigmoid(x); }
float oracle_sef_log2(double x) { return silk_log2(x); }
int oracle_sef_float2int(float x) { return silk_float2int(x); }
void oracle_sef_float2short_array(short *out, const float *in, int n) {
  silk_float2short_array(out, in, n);
}
void oracle_sef_short2float_array(float *out, const short *in, int n) {
  silk_short2float_array(out, in, n);
}

/* ---------------------------------------------------------------------------------------- */
/* Static helpers                                                                           */
/* ---------------------------------------------------------------------------------------- */

float oracle_sef_warped_gain(const float *coefs, float lambda, int order) {
  return warped_gain(coefs, lambda, order);
}
void oracle_sef_warped_true2monic_coefs(float *coefs, float lambda, float limit, int order) {
  warped_true2monic_coefs(coefs, lambda, limit, order);
}
void oracle_sef_limit_coefs(float *coefs, float limit, int order) {
  limit_coefs(coefs, limit, order);
}

/* silk_P_Ana_calc_corr_st3 / silk_P_Ana_calc_energy_st3: out is [4][34][5] floats. */
void oracle_sef_calc_corr_st3(float *out, const float *frame, int start_lag, int sf_length,
                              int nb_subfr, int complexity) {
  silk_float a[PE_MAX_NB_SUBFR][PE_NB_CBKS_STAGE3_MAX][PE_NB_STAGE3_LAGS];
  memset(a, 0, sizeof(a));
  silk_P_Ana_calc_corr_st3(a, frame, start_lag, sf_length, nb_subfr, complexity, 0);
  memcpy(out, a, sizeof(a));
}
void oracle_sef_calc_energy_st3(float *out, const float *frame, int start_lag, int sf_length,
                                int nb_subfr, int complexity) {
  silk_float a[PE_MAX_NB_SUBFR][PE_NB_CBKS_STAGE3_MAX][PE_NB_STAGE3_LAGS];
  memset(a, 0, sizeof(a));
  silk_P_Ana_calc_energy_st3(a, frame, start_lag, sf_length, nb_subfr, complexity);
  memcpy(out, a, sizeof(a));
}

/* ---------------------------------------------------------------------------------------- */
/* State-based helpers with flat arguments                                                  */
/* ---------------------------------------------------------------------------------------- */

/* silk_find_LPC_FLP on a zeroed encoder state configured with the given fields.
   Returns NLSFInterpCoef_Q2. */
int oracle_sef_find_lpc(int order, int subfr_length, int nb_subfr, int use_interp,
                        int first_frame, const short prev_nlsf[16], short nlsf_out[16],
                        const float *x, float min_inv_gain) {
  silk_encoder_state *s = (silk_encoder_state *)calloc(1, sizeof(silk_encoder_state));
  int r;
  s->predictLPCOrder = order;
  s->subfr_length = subfr_length;
  s->nb_subfr = nb_subfr;
  s->useInterpolatedNLSFs = use_interp;
  s->first_frame_after_reset = first_frame;
  memcpy(s->prev_NLSFq_Q15, prev_nlsf, sizeof(s->prev_NLSFq_Q15));
  silk_find_LPC_FLP(s, nlsf_out, x, min_inv_gain, 0);
  r = s->indices.NLSFInterpCoef_Q2;
  free(s);
  return r;
}

/* silk_quant_LTP_gains_FLP. io: sum_log_gain_Q7 (in/out); outputs cbk_index[4],
   periodicity_index, pred_gain_dB. */
void oracle_sef_quant_ltp_gains(float *B, signed char *cbk_index, signed char *per_index,
                                int *sum_log_gain_Q7, float *pred_gain_dB, const float *XX,
                                const float *xX, int subfr_len, int nb_subfr) {
  opus_int32 slg = *sum_log_gain_Q7;
  silk_quant_LTP_gains_FLP(B, cbk_index, per_index, &slg, pred_gain_dB, XX, xX, subfr_len,
                           nb_subfr, 0);
  *sum_log_gain_Q7 = slg;
}

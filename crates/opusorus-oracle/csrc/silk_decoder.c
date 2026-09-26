/* Oracle C shims for unit silk_decoder: the SILK decoder (silk/dec_API.c and the functions it
   calls). dec_API.c is included with every global renamed so the `silk_decoder` super-struct
   (private to that file) is visible; the renamed copies are compiled from the same source as
   the library ones. */
#include <stdlib.h>
#include <string.h>

#define silk_LoadOSCEModels oracle_sd_silk_LoadOSCEModels
#define silk_Get_Decoder_Size oracle_sd_silk_Get_Decoder_Size
#define silk_ResetDecoder oracle_sd_silk_ResetDecoder
#define silk_InitDecoder oracle_sd_silk_InitDecoder
#define silk_Decode oracle_sd_silk_Decode
#include "dec_API.c"
#undef silk_LoadOSCEModels
#undef silk_Get_Decoder_Size
#undef silk_ResetDecoder
#undef silk_InitDecoder
#undef silk_Decode

#include "entdec.h"
#include "tables.h"

#define ORACLE_SD_BUF 4096

/* Persistent decoder + range decoder (the latter owns a copy of the payload). */
typedef struct {
  silk_decoder *dec;
  ec_dec ec;
  unsigned char buf[ORACLE_SD_BUF];
} oracle_sd;

void *oracle_sd_new(void) {
  opus_int size = 0;
  oracle_sd *h = (oracle_sd *)calloc(1, sizeof(oracle_sd));
  if (!h) return NULL;
  oracle_sd_silk_Get_Decoder_Size(&size);
  /* calloc: nChannelsAPI/nChannelsInternal are left untouched by silk_InitDecoder and are zero
     in the Opus decoder (OPUS_CLEAR). */
  h->dec = (silk_decoder *)calloc(1, size);
  if (!h->dec) {
    free(h);
    return NULL;
  }
  oracle_sd_silk_InitDecoder(h->dec);
  ec_dec_init(&h->ec, h->buf, 0);
  return h;
}

void oracle_sd_free(void *p) {
  oracle_sd *h = (oracle_sd *)p;
  if (!h) return;
  free(h->dec);
  free(h);
}

int oracle_sd_size(void) {
  opus_int size = 0;
  oracle_sd_silk_Get_Decoder_Size(&size);
  return size;
}

int oracle_sd_init(void *p) { return oracle_sd_silk_InitDecoder(((oracle_sd *)p)->dec); }
int oracle_sd_reset(void *p) { return oracle_sd_silk_ResetDecoder(((oracle_sd *)p)->dec); }

/* Starts a range decoder over a copy of data[0..len) (len <= ORACLE_SD_BUF). */
void oracle_sd_ec_init(void *p, const unsigned char *data, int len) {
  oracle_sd *h = (oracle_sd *)p;
  if (len > ORACLE_SD_BUF) len = ORACLE_SD_BUF;
  if (len > 0) memcpy(h->buf, data, len);
  ec_dec_init(&h->ec, h->buf, len);
}

/* Range decoder state: storage, end_offs, end_window, nend_bits, nbits_total, offs, rng, val,
   ext, rem, error, tell_frac. */
void oracle_sd_ec_state(void *p, unsigned int out[12]) {
  oracle_sd *h = (oracle_sd *)p;
  ec_dec *d = &h->ec;
  out[0] = d->storage;
  out[1] = d->end_offs;
  out[2] = (unsigned int)d->end_window;
  out[3] = (unsigned int)d->nend_bits;
  out[4] = (unsigned int)d->nbits_total;
  out[5] = d->offs;
  out[6] = d->rng;
  out[7] = d->val;
  out[8] = d->ext;
  out[9] = (unsigned int)d->rem;
  out[10] = (unsigned int)d->error;
  out[11] = ec_tell_frac(d);
}

/* ctrl: nChannelsAPI, nChannelsInternal, API_sampleRate, internalSampleRate, payloadSize_ms,
   prevPitchLag, enable_deep_plc (in/out). */
int oracle_sd_decode(void *p, int ctrl[7], int lost, int new_packet, float *out, int *n_out) {
  oracle_sd *h = (oracle_sd *)p;
  silk_DecControlStruct c;
  opus_int32 n = 0;
  int ret;
  memset(&c, 0, sizeof(c));
  c.nChannelsAPI = ctrl[0];
  c.nChannelsInternal = ctrl[1];
  c.API_sampleRate = ctrl[2];
  c.internalSampleRate = ctrl[3];
  c.payloadSize_ms = ctrl[4];
  c.prevPitchLag = ctrl[5];
  c.enable_deep_plc = ctrl[6];
#ifdef ENABLE_DEEP_PLC
  /* DNN oracle builds (dnn_core build.rs): silk_Decode takes an extra LPCNetPLCState*. */
  ret = oracle_sd_silk_Decode(h->dec, &c, lost, new_packet, &h->ec, out, &n, NULL, 0);
#else
  ret = oracle_sd_silk_Decode(h->dec, &c, lost, new_packet, &h->ec, out, &n, 0);
#endif
  ctrl[0] = c.nChannelsAPI;
  ctrl[1] = c.nChannelsInternal;
  ctrl[2] = c.API_sampleRate;
  ctrl[3] = c.internalSampleRate;
  ctrl[4] = c.payloadSize_ms;
  ctrl[5] = c.prevPitchLag;
  ctrl[6] = c.enable_deep_plc;
  *n_out = n;
  return ret;
}

/* ---------------------------------------------------------------------------------------- */
/* State dump (must match dump_state() in tests/silk_decoder.rs)                            */
/* ---------------------------------------------------------------------------------------- */

#define P(v) do { if (k < cap) out[k] = (int)(v); k++; } while (0)
#define PA(a, n) do { int _i; for (_i = 0; _i < (n); _i++) P((a)[_i]); } while (0)

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

static int dump_channel(const silk_decoder_state *s, int *out, int k, int cap) {
  const silk_resampler_state_struct *r = &s->resampler_state;
  P(s->prev_gain_Q16);
  PA(s->exc_Q14, MAX_FRAME_LENGTH);
  PA(s->sLPC_Q14_buf, MAX_LPC_ORDER);
  PA(s->outBuf, MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH);
  P(s->lagPrev);
  P(s->LastGainIndex);
  P(s->fs_kHz);
  P(s->fs_API_hz);
  P(s->nb_subfr);
  P(s->frame_length);
  P(s->subfr_length);
  P(s->ltp_mem_length);
  P(s->LPC_order);
  PA(s->prevNLSF_Q15, MAX_LPC_ORDER);
  P(s->first_frame_after_reset);
  P(lag_low_len(s->pitch_lag_low_bits_iCDF));
  P(contour_len(s->pitch_contour_iCDF));
  P(s->nFramesDecoded);
  P(s->nFramesPerPacket);
  P(s->ec_prevSignalType);
  P(s->ec_prevLagIndex);
  PA(s->VAD_flags, MAX_FRAMES_PER_PACKET);
  P(s->LBRR_flag);
  PA(s->LBRR_flags, MAX_FRAMES_PER_PACKET);
  /* resampler */
  PA(r->sIIR, SILK_RESAMPLER_MAX_IIR_ORDER);
  if (r->resampler_function == 3) {
    PA(r->sFIR.i32, SILK_RESAMPLER_MAX_FIR_ORDER);
  } else if (r->resampler_function == 2) {
    PA(r->sFIR.i16, SILK_RESAMPLER_MAX_FIR_ORDER);
  } else {
    int i;
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
  /* NLSF codebook (set together with LPC_order) */
  P(s->LPC_order == 0 ? 0 : (s->psNLSF_CB == NULL ? -1 : s->psNLSF_CB->order));
  /* indices */
  PA(s->indices.GainsIndices, MAX_NB_SUBFR);
  PA(s->indices.LTPIndex, MAX_NB_SUBFR);
  PA(s->indices.NLSFIndices, MAX_LPC_ORDER + 1);
  P(s->indices.lagIndex);
  P(s->indices.contourIndex);
  P(s->indices.signalType);
  P(s->indices.quantOffsetType);
  P(s->indices.NLSFInterpCoef_Q2);
  P(s->indices.PERIndex);
  P(s->indices.LTP_scaleIndex);
  P(s->indices.Seed);
  /* CNG */
  PA(s->sCNG.CNG_exc_buf_Q14, MAX_FRAME_LENGTH);
  PA(s->sCNG.CNG_smth_NLSF_Q15, MAX_LPC_ORDER);
  PA(s->sCNG.CNG_synth_state, MAX_LPC_ORDER);
  P(s->sCNG.CNG_smth_Gain_Q16);
  P(s->sCNG.rand_seed);
  P(s->sCNG.fs_kHz);
  P(s->lossCnt);
  P(s->prevSignalType);
  /* PLC */
  P(s->sPLC.pitchL_Q8);
  PA(s->sPLC.LTPCoef_Q14, LTP_ORDER);
  PA(s->sPLC.prevLPC_Q12, MAX_LPC_ORDER);
  P(s->sPLC.last_frame_lost);
  P(s->sPLC.rand_seed);
  P(s->sPLC.randScale_Q14);
  P(s->sPLC.conc_energy);
  P(s->sPLC.conc_energy_shift);
  P(s->sPLC.prevLTP_scale_Q14);
  PA(s->sPLC.prevGain_Q16, 2);
  P(s->sPLC.fs_kHz);
  P(s->sPLC.nb_subfr);
  P(s->sPLC.subfr_length);
  P(s->sPLC.enable_deep_plc);
  return k;
}

/* Dumps the whole decoder state as ints; returns the number of values (even if > cap). */
int oracle_sd_dump(void *p, int *out, int cap) {
  const silk_decoder *d = ((oracle_sd *)p)->dec;
  int k = 0, n;
  for (n = 0; n < DECODER_NUM_CHANNELS; n++) k = dump_channel(&d->channel_state[n], out, k, cap);
  PA(d->sStereo.pred_prev_Q13, 2);
  PA(d->sStereo.sMid, 2);
  PA(d->sStereo.sSide, 2);
  P(d->nChannelsAPI);
  P(d->nChannelsInternal);
  P(d->prev_decode_only_middle);
  return k;
}

#undef P
#undef PA

/* ---------------------------------------------------------------------------------------- */
/* Direct function shims                                                                    */
/* ---------------------------------------------------------------------------------------- */

/* silk_stereo_MS_to_LR. state: pred_prev_Q13[2], sMid[2], sSide[2] (in/out); x1, x2 hold
   frame_length + 2 samples. */
void oracle_sd_stereo_MS_to_LR(short state[6], short *x1, short *x2, const int pred[2],
                               int fs_kHz, int frame_length) {
  stereo_dec_state st;
  opus_int32 p[2];
  st.pred_prev_Q13[0] = state[0];
  st.pred_prev_Q13[1] = state[1];
  st.sMid[0] = state[2];
  st.sMid[1] = state[3];
  st.sSide[0] = state[4];
  st.sSide[1] = state[5];
  p[0] = pred[0];
  p[1] = pred[1];
  silk_stereo_MS_to_LR(&st, x1, x2, p, fs_kHz, frame_length);
  state[0] = st.pred_prev_Q13[0];
  state[1] = st.pred_prev_Q13[1];
  state[2] = st.sMid[0];
  state[3] = st.sMid[1];
  state[4] = st.sSide[0];
  state[5] = st.sSide[1];
}

/* silk_decode_indices + silk_decode_pulses on a fresh channel state configured for fs_kHz /
   nb_subfr, with VAD flag, previous entropy-coding state and conditional coding given.
   out: signalType, quantOffsetType, GainsIndices[4], LTPIndex[4], NLSFIndices[17], lagIndex,
   contourIndex, NLSFInterpCoef_Q2, PERIndex, LTP_scaleIndex, Seed, ec_prevSignalType,
   ec_prevLagIndex, tell_frac, rng (37 ints); pulses: frame_length (rounded up to 16) */
void oracle_sd_decode_indices(const unsigned char *data, int len, int fs_kHz, int nb_subfr,
                              int vad, int decode_lbrr, int cond, int prev_type, int prev_lag,
                              int *out, short *pulses) {
  silk_decoder_state *s = (silk_decoder_state *)calloc(1, sizeof(silk_decoder_state));
  ec_dec d;
  int k = 0, i;
  silk_init_decoder(s);
  s->nb_subfr = nb_subfr;
  silk_decoder_set_fs(s, fs_kHz, 48000);
  s->VAD_flags[0] = vad;
  s->ec_prevSignalType = prev_type;
  s->ec_prevLagIndex = (opus_int16)prev_lag;
  ec_dec_init(&d, (unsigned char *)data, len);
  silk_decode_indices(s, &d, 0, decode_lbrr, cond);
  silk_decode_pulses(&d, pulses, s->indices.signalType, s->indices.quantOffsetType,
                     s->frame_length);
  out[k++] = s->indices.signalType;
  out[k++] = s->indices.quantOffsetType;
  for (i = 0; i < MAX_NB_SUBFR; i++) out[k++] = s->indices.GainsIndices[i];
  for (i = 0; i < MAX_NB_SUBFR; i++) out[k++] = s->indices.LTPIndex[i];
  for (i = 0; i < MAX_LPC_ORDER + 1; i++) out[k++] = s->indices.NLSFIndices[i];
  out[k++] = s->indices.lagIndex;
  out[k++] = s->indices.contourIndex;
  out[k++] = s->indices.NLSFInterpCoef_Q2;
  out[k++] = s->indices.PERIndex;
  out[k++] = s->indices.LTP_scaleIndex;
  out[k++] = s->indices.Seed;
  out[k++] = s->ec_prevSignalType;
  out[k++] = s->ec_prevLagIndex;
  out[k++] = (int)ec_tell_frac(&d);
  out[k++] = (int)d.rng;
  free(s);
}

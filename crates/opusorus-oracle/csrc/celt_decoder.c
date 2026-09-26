/* Oracle C shims for unit celt_decoder (celt/celt_decoder.c).
 *
 * celt_decoder.c is included with every external symbol renamed (oracle_cdc_*) so the private
 * `struct OpusCustomDecoder` and the static helpers (tf_decode, deemphasis, celt_synthesis,
 * celt_plc_pitch_search) are reachable. The full-decoder handles call the *library* functions
 * (celt_decoder_init, celt_decode_with_ec[_dred], opus_custom_decoder_ctl, ...); the renamed
 * copy is only used for the static helpers and for the struct layout (state dumps).
 *
 * A CELT encoder handle (library celt_encode_with_ec) generates realistic test packets. */
#include <stdlib.h>
#include <string.h>

#define validate_celt_decoder oracle_cdc_validate_celt_decoder
#define celt_decoder_get_size oracle_cdc_celt_decoder_get_size
#define opus_custom_decoder_get_size oracle_cdc_opus_custom_decoder_get_size
#define opus_custom_decoder_create oracle_cdc_opus_custom_decoder_create
#define celt_decoder_init oracle_cdc_celt_decoder_init
#define opus_custom_decoder_init oracle_cdc_opus_custom_decoder_init
#define opus_custom_decoder_destroy oracle_cdc_opus_custom_decoder_destroy
#define deemphasis oracle_cdc_deemphasis
#define celt_synthesis oracle_cdc_celt_synthesis
#define celt_decode_with_ec_dred oracle_cdc_celt_decode_with_ec_dred
#define celt_decode_with_ec oracle_cdc_celt_decode_with_ec
#define opus_custom_decode oracle_cdc_opus_custom_decode
#define opus_custom_decode24 oracle_cdc_opus_custom_decode24
#define opus_custom_decode_float oracle_cdc_opus_custom_decode_float
#define opus_custom_decoder_ctl oracle_cdc_opus_custom_decoder_ctl
#define update_plc_state oracle_cdc_update_plc_state
#include "../../../vendor/libopus/celt/celt_decoder.c"
#undef validate_celt_decoder
#undef celt_decoder_get_size
#undef opus_custom_decoder_get_size
#undef opus_custom_decoder_create
#undef celt_decoder_init
#undef opus_custom_decoder_init
#undef opus_custom_decoder_destroy
#undef deemphasis
#undef celt_synthesis
#undef celt_decode_with_ec_dred
#undef celt_decode_with_ec
#undef opus_custom_decode
#undef opus_custom_decode24
#undef opus_custom_decode_float
#undef opus_custom_decoder_ctl
#undef update_plc_state

#include "entenc.h"

/* Library prototypes (the header declarations above were renamed). */
int celt_decoder_get_size(int channels);
int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels);
int celt_decode_with_ec(CELTDecoder *st, const unsigned char *data, int len, opus_res *pcm,
                        int frame_size, ec_dec *dec, int accum);
#ifdef ENABLE_QEXT
int celt_decode_with_ec_dred(CELTDecoder *st, const unsigned char *data, int len,
                             opus_res *pcm, int frame_size, ec_dec *dec, int accum,
                             const unsigned char *qext_payload, int qext_payload_len);
#endif
int opus_custom_decoder_ctl(CELTDecoder *st, int request, ...);
#ifdef CUSTOM_MODES
CELTDecoder *opus_custom_decoder_create(const CELTMode *mode, int channels, int *error);
void opus_custom_decoder_destroy(CELTDecoder *st);
int opus_custom_decode(CELTDecoder *st, const unsigned char *data, int len, opus_int16 *pcm,
                       int frame_size);
int opus_custom_decode24(CELTDecoder *st, const unsigned char *data, int len, opus_int32 *pcm,
                         int frame_size);
int opus_custom_decode_float(CELTDecoder *st, const unsigned char *data, int len, float *pcm,
                             int frame_size);
#endif

int oracle_cd_has_qext(void) {
#ifdef ENABLE_QEXT
  return 1;
#else
  return 0;
#endif
}

int oracle_cd_has_custom(void) {
#ifdef CUSTOM_MODES
  return 1;
#else
  return 0;
#endif
}

/* ------------------------------------------------------------------------------------------ */
/* Decoder handle                                                                               */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  CELTDecoder *st;
  CELTMode *custom_mode; /* owned custom mode (custom decoders) */
  int custom;
} oracle_cd;

void *oracle_cd_new(int fs, int channels, int *err) {
  oracle_cd *h = (oracle_cd *)calloc(1, sizeof(oracle_cd));
  if (!h) return NULL;
  h->st = (CELTDecoder *)calloc(1, celt_decoder_get_size(2));
  if (!h->st) {
    free(h);
    return NULL;
  }
  *err = celt_decoder_init(h->st, fs, channels);
  return h;
}

#ifdef CUSTOM_MODES
void *oracle_cd_custom_new(int fs, int frame_size, int channels, int *err) {
  oracle_cd *h = (oracle_cd *)calloc(1, sizeof(oracle_cd));
  if (!h) return NULL;
  h->custom_mode = opus_custom_mode_create(fs, frame_size, err);
  if (!h->custom_mode) {
    free(h);
    return NULL;
  }
  h->st = opus_custom_decoder_create(h->custom_mode, channels, err);
  if (!h->st) {
    opus_custom_mode_destroy(h->custom_mode);
    free(h);
    return NULL;
  }
  h->custom = 1;
  return h;
}
#endif

void oracle_cd_free(void *p) {
  oracle_cd *h = (oracle_cd *)p;
  if (!h) return;
#ifdef CUSTOM_MODES
  if (h->custom) {
    opus_custom_decoder_destroy(h->st);
    opus_custom_mode_destroy(h->custom_mode);
    free(h);
    return;
  }
#endif
  free(h->st);
  free(h);
}

int oracle_cd_ctl_set(void *p, int request, int value) {
  oracle_cd *h = (oracle_cd *)p;
  if (request == OPUS_RESET_STATE) return opus_custom_decoder_ctl(h->st, OPUS_RESET_STATE);
  return opus_custom_decoder_ctl(h->st, request, (opus_int32)value);
}

int oracle_cd_ctl_get(void *p, int request, int *value) {
  oracle_cd *h = (oracle_cd *)p;
  opus_int32 v = 0;
  int ret = opus_custom_decoder_ctl(h->st, request, &v);
  *value = v;
  return ret;
}

/* CELT_GET_MODE: returns 1 when the mode pointer is the decoder's mode. */
int oracle_cd_get_mode_ok(void *p) {
  oracle_cd *h = (oracle_cd *)p;
  const CELTMode *m = NULL;
  int ret = opus_custom_decoder_ctl(h->st, CELT_GET_MODE(&m));
  return ret == OPUS_OK && m == h->st->mode;
}

/* celt_decode_with_ec[_dred] with its own range decoder (data may be NULL). */
int oracle_cd_decode(void *p, const unsigned char *data, int len, float *pcm, int frame_size,
                     int accum, const unsigned char *qext, int qext_len) {
  oracle_cd *h = (oracle_cd *)p;
#ifdef ENABLE_QEXT
  return celt_decode_with_ec_dred(h->st, data, len, pcm, frame_size, NULL, accum, qext,
                                  qext_len);
#else
  (void)qext;
  (void)qext_len;
  return celt_decode_with_ec(h->st, data, len, pcm, frame_size, NULL, accum);
#endif
}

/* Hybrid-style decode: a range decoder over data[0..len) first decodes n uniform symbols
   (ec_dec_uint(fts[i]), written to vals[i]) standing in for the SILK layer, then CELT decodes
   from the same range decoder. ec_out: rng, tell, tell_frac, error. */
int oracle_cd_decode_shared(void *p, const unsigned char *data, int len, float *pcm,
                            int frame_size, int accum, const unsigned *fts, unsigned *vals,
                            int n, unsigned ec_out[4]) {
  oracle_cd *h = (oracle_cd *)p;
  ec_dec dec;
  int i, ret;
  ec_dec_init(&dec, (unsigned char *)data, len);
  for (i = 0; i < n; i++) vals[i] = ec_dec_uint(&dec, fts[i]);
  ret = celt_decode_with_ec(h->st, data, len, pcm, frame_size, &dec, accum);
  ec_out[0] = dec.rng;
  ec_out[1] = (unsigned)ec_tell(&dec);
  ec_out[2] = ec_tell_frac(&dec);
  ec_out[3] = (unsigned)dec.error;
  return ret;
}

#ifdef CUSTOM_MODES
int oracle_cd_custom_decode(void *p, const unsigned char *data, int len, short *pcm,
                            int frame_size) {
  return opus_custom_decode(((oracle_cd *)p)->st, data, len, pcm, frame_size);
}
int oracle_cd_custom_decode24(void *p, const unsigned char *data, int len, int *pcm,
                              int frame_size) {
  return opus_custom_decode24(((oracle_cd *)p)->st, data, len, pcm, frame_size);
}
int oracle_cd_custom_decode_float(void *p, const unsigned char *data, int len, float *pcm,
                                  int frame_size) {
  return opus_custom_decode_float(((oracle_cd *)p)->st, data, len, pcm, frame_size);
}
#endif

#define ORACLE_CD_NINTS 22

/* State dump. ints: overlap, channels, stream_channels, downsample, start, end, signalling,
   disable_inv, complexity, qext_scale (1 without QEXT), rng, error, last_pitch_index,
   loss_duration, plc_duration, last_frame_type, skip_plc, postfilter_period,
   postfilter_period_old, postfilter_tapset, postfilter_tapset_old, prefilter_and_fold.
   floats (returns the count; pass NULL to query): postfilter_gain, postfilter_gain_old,
   preemph_memD[2], qext_oldBandE[2*NB_QEXT_BANDS] (QEXT only), _decode_mem
   (channels*(dbs+overlap)), oldEBands, oldLogE, oldLogE2, backgroundLogE (2*nbEBands each),
   lpc (channels*CELT_LPC_ORDER). */
int oracle_cd_state(void *p, int *ints, float *floats) {
  CELTDecoder *st = ((oracle_cd *)p)->st;
  int qext_scale = 1;
  int nb = st->mode->nbEBands;
  int dbs, mem_len, cnt, i;
  float *oldBandE;
#ifdef ENABLE_QEXT
  qext_scale = st->qext_scale;
#endif
  dbs = qext_scale * DECODE_BUFFER_SIZE;
  mem_len = st->channels * (dbs + st->overlap);
  cnt = 4 + mem_len + 8 * nb + st->channels * CELT_LPC_ORDER;
#ifdef ENABLE_QEXT
  cnt += 2 * NB_QEXT_BANDS;
#endif
  if (ints) {
    ints[0] = st->overlap;
    ints[1] = st->channels;
    ints[2] = st->stream_channels;
    ints[3] = st->downsample;
    ints[4] = st->start;
    ints[5] = st->end;
    ints[6] = st->signalling;
    ints[7] = st->disable_inv;
    ints[8] = st->complexity;
    ints[9] = qext_scale;
    ints[10] = (int)st->rng;
    ints[11] = st->error;
    ints[12] = st->last_pitch_index;
    ints[13] = st->loss_duration;
    ints[14] = st->plc_duration;
    ints[15] = st->last_frame_type;
    ints[16] = st->skip_plc;
    ints[17] = st->postfilter_period;
    ints[18] = st->postfilter_period_old;
    ints[19] = st->postfilter_tapset;
    ints[20] = st->postfilter_tapset_old;
    ints[21] = st->prefilter_and_fold;
  }
  if (floats) {
    int k = 0;
    floats[k++] = st->postfilter_gain;
    floats[k++] = st->postfilter_gain_old;
    floats[k++] = st->preemph_memD[0];
    floats[k++] = st->preemph_memD[1];
#ifdef ENABLE_QEXT
    for (i = 0; i < 2 * NB_QEXT_BANDS; i++) floats[k++] = st->qext_oldBandE[i];
#endif
    for (i = 0; i < mem_len; i++) floats[k++] = st->_decode_mem[i];
    oldBandE = (float *)(st->_decode_mem + mem_len);
    for (i = 0; i < 8 * nb + st->channels * CELT_LPC_ORDER; i++) floats[k++] = oldBandE[i];
  }
  return cnt;
}

/* ------------------------------------------------------------------------------------------ */
/* Static helpers                                                                               */
/* ------------------------------------------------------------------------------------------ */

/* tf_decode on a range decoder over buf[0..len) after `pre` ec_dec_bit_logp(1) reads.
   ec_out: rng, tell. */
void oracle_cd_tf_decode(int start, int end, int isTransient, int *tf_res, int LM,
                         const unsigned char *buf, int len, int pre, unsigned ec_out[2]) {
  ec_dec dec;
  int i;
  ec_dec_init(&dec, (unsigned char *)buf, len);
  for (i = 0; i < pre; i++) ec_dec_bit_logp(&dec, 1);
  tf_decode(start, end, isTransient, tf_res, LM, &dec);
  ec_out[0] = dec.rng;
  ec_out[1] = (unsigned)ec_tell(&dec);
}

void oracle_cd_deemphasis(const float *in0, const float *in1, float *pcm, int N, int C,
                          int downsample, const float coef[4], float mem[2], int accum) {
  celt_sig *in[2];
  in[0] = (celt_sig *)in0;
  in[1] = (celt_sig *)in1;
  oracle_cdc_deemphasis(in, pcm, N, C, downsample, coef, mem, accum);
}

/* Mode of the static-helper shims: the standard 48 kHz mode, or 96 kHz (QEXT). */
static const CELTMode *cd_mode(int fs) { return opus_custom_mode_create(fs, fs / 50, NULL); }

/* celt_synthesis. out0/out1: channel buffers; out_syn[c] = outc + off. qext: when use_qext,
   qext_mode = compute_qext_mode(mode). */
void oracle_cd_celt_synthesis(int fs, float *X, float *out0, float *out1, int off,
                              float *oldBandE, int start, int effEnd, int C, int CC,
                              int isTransient, int LM, int downsample, int silence, int use_qext,
                              float *qext_bandLogE, int qext_end) {
  const CELTMode *mode = cd_mode(fs);
  celt_sig *out_syn[2];
  out_syn[0] = out0 + off;
  out_syn[1] = out1 ? out1 + off : NULL;
#ifdef ENABLE_QEXT
  {
    CELTMode qm;
    const CELTMode *qmp = NULL;
    if (use_qext) {
      compute_qext_mode(&qm, mode);
      qmp = &qm;
    }
    oracle_cdc_celt_synthesis(mode, X, out_syn, oldBandE, start, effEnd, C, CC, isTransient,
                              LM, downsample, silence, 0, qmp, qext_bandLogE, qext_end);
  }
#else
  (void)use_qext;
  (void)qext_bandLogE;
  (void)qext_end;
  oracle_cdc_celt_synthesis(mode, X, out_syn, oldBandE, start, effEnd, C, CC, isTransient, LM,
                            downsample, silence, 0);
#endif
}

/* celt_plc_pitch_search on mem0/mem1 (each dbs samples) for a decoder at fs. */
int oracle_cd_plc_pitch_search(int fs, float *mem0, float *mem1, int C) {
  CELTDecoder *st = (CELTDecoder *)calloc(1, celt_decoder_get_size(2));
  celt_sig *decode_mem[2];
  int ret;
  celt_decoder_init(st, fs, 1);
  decode_mem[0] = mem0;
  decode_mem[1] = mem1;
  ret = celt_plc_pitch_search(st, decode_mem, C, 0);
  free(st);
  return ret;
}

/* ------------------------------------------------------------------------------------------ */
/* Encoder handle (packet generation)                                                           */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  CELTEncoder *st;
  CELTMode *custom_mode;
  int custom;
} oracle_cd_enc;

void *oracle_cd_enc_new(int fs, int channels, int *err) {
  oracle_cd_enc *h = (oracle_cd_enc *)calloc(1, sizeof(oracle_cd_enc));
  if (!h) return NULL;
  h->st = (CELTEncoder *)calloc(1, celt_encoder_get_size(2));
  *err = celt_encoder_init(h->st, fs, channels, 0);
  opus_custom_encoder_ctl(h->st, CELT_SET_SIGNALLING(0));
  return h;
}

#ifdef CUSTOM_MODES
void *oracle_cd_enc_custom_new(int fs, int frame_size, int channels, int *err) {
  oracle_cd_enc *h = (oracle_cd_enc *)calloc(1, sizeof(oracle_cd_enc));
  if (!h) return NULL;
  h->custom_mode = opus_custom_mode_create(fs, frame_size, err);
  if (!h->custom_mode) {
    free(h);
    return NULL;
  }
  h->st = opus_custom_encoder_create(h->custom_mode, channels, err);
  if (!h->st) {
    opus_custom_mode_destroy(h->custom_mode);
    free(h);
    return NULL;
  }
  h->custom = 1;
  return h;
}
#endif

void oracle_cd_enc_free(void *p) {
  oracle_cd_enc *h = (oracle_cd_enc *)p;
  if (!h) return;
#ifdef CUSTOM_MODES
  if (h->custom) {
    opus_custom_encoder_destroy(h->st);
    opus_custom_mode_destroy(h->custom_mode);
    free(h);
    return;
  }
#endif
  free(h->st);
  free(h);
}

int oracle_cd_enc_ctl(void *p, int request, int value) {
  return opus_custom_encoder_ctl(((oracle_cd_enc *)p)->st, request, (opus_int32)value);
}

/* Encodes into out[1..1+nbytes) (out[0] is a TOC placeholder the QEXT path may modify). */
int oracle_cd_enc_encode(void *p, const float *pcm, int frame_size, unsigned char *out,
                         int nbytes) {
  out[0] = 0;
  return celt_encode_with_ec(((oracle_cd_enc *)p)->st, pcm, frame_size, out + 1, nbytes, NULL);
}

/* Hybrid-style: n uniform symbols (vals[i] < fts[i]) are range coded first, then CELT encodes
   into the same range coder (as the Opus encoder does after SILK). */
int oracle_cd_enc_encode_shared(void *p, const float *pcm, int frame_size, unsigned char *out,
                                int nbytes, const unsigned *fts, const unsigned *vals, int n) {
  ec_enc enc;
  int i;
  ec_enc_init(&enc, out, nbytes);
  for (i = 0; i < n; i++) ec_enc_uint(&enc, vals[i], fts[i]);
  return celt_encode_with_ec(((oracle_cd_enc *)p)->st, pcm, frame_size, NULL, nbytes, &enc);
}

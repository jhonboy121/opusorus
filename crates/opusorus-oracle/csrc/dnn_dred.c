/* Oracle C shims for unit dnn_dred: dnn/dred_rdovae_enc.c, dred_rdovae_dec.c,
   dred_rdovae_stats_data.c, dred_coding.c, dred_encoder.c and dred_decoder.c.

   Only compiled into oracle builds with ENABLE_DRED. The static helpers of dred_encoder.c are
   reached through a private copy with its exported symbols renamed (dd_copy_*). The C models
   are the compiled-in tables; handles wrap C structs on the heap so the Rust side never sees C
   layouts. */
#ifdef ENABLE_DRED

#include <stdlib.h>
#include <string.h>

#include "dred_encoder.h"
#include "dred_decoder.h"
#include "dred_coding.h"
#include "dred_rdovae_enc.h"
#include "dred_rdovae_dec.h"
#include "dred_rdovae_stats_data.h"
#include "celt/entenc.h"
#include "celt/entdec.h"
#include "celt/laplace.h"

/* ---- private copy of dred_encoder.c (statics) ---- */
#define dred_encoder_load_model dd_copy_dred_encoder_load_model
#define dred_encoder_reset dd_copy_dred_encoder_reset
#define dred_encoder_init dd_copy_dred_encoder_init
#define filter_df2t dd_copy_filter_df2t
#define dred_compute_latents dd_copy_dred_compute_latents
#define dred_encode_silk_frame dd_copy_dred_encode_silk_frame
#include "dred_encoder.c"
#undef dred_encoder_load_model
#undef dred_encoder_reset
#undef dred_encoder_init
#undef filter_df2t
#undef dred_compute_latents
#undef dred_encode_silk_frame

/* Library functions without a header prototype. */
void filter_df2t(const float *in, float *out, int len, float b0, const float *b, const float *a,
                 int order, float *mem);
void dred_decode_latents(ec_dec *dec, float *x, const opus_uint8 *scale, const opus_uint8 *r,
                         const opus_uint8 *p0, int dim);

/* From the dnn_core shim (same library). */
int oracle_dc_lpcnet_enc_state(void *p, float *out);

/* ---- stats tables / dred_coding ---- */
int oracle_dd_stats(int which, unsigned char *out) {
  const opus_uint8 *t;
  int n;
  switch (which) {
  case 0: t = dred_latent_quant_scales_q8; n = sizeof(dred_latent_quant_scales_q8); break;
  case 1: t = dred_latent_dead_zone_q8; n = sizeof(dred_latent_dead_zone_q8); break;
  case 2: t = dred_latent_r_q8; n = sizeof(dred_latent_r_q8); break;
  case 3: t = dred_latent_p0_q8; n = sizeof(dred_latent_p0_q8); break;
  case 4: t = dred_state_quant_scales_q8; n = sizeof(dred_state_quant_scales_q8); break;
  case 5: t = dred_state_dead_zone_q8; n = sizeof(dred_state_dead_zone_q8); break;
  case 6: t = dred_state_r_q8; n = sizeof(dred_state_r_q8); break;
  case 7: t = dred_state_p0_q8; n = sizeof(dred_state_p0_q8); break;
  default: return -1;
  }
  if (out) memcpy(out, t, n);
  return n;
}

int oracle_dd_compute_quantizer(int q0, int dQ, int qmax, int i) {
  return compute_quantizer(q0, dQ, qmax, i);
}

/* Constants the Rust side checks against. */
void oracle_dd_constants(int *out) {
  int k = 0;
  out[k++] = DRED_NUM_FEATURES;
  out[k++] = DRED_LATENT_DIM;
  out[k++] = DRED_STATE_DIM;
  out[k++] = DRED_PADDED_LATENT_DIM;
  out[k++] = DRED_PADDED_STATE_DIM;
  out[k++] = DRED_NUM_QUANTIZATION_LEVELS;
  out[k++] = DRED_MAX_RNN_NEURONS;
  out[k++] = DRED_MAX_CONV_INPUTS;
  out[k++] = DRED_ENC_MAX_RNN_NEURONS;
  out[k++] = DRED_ENC_MAX_CONV_INPUTS;
  out[k++] = DRED_DEC_MAX_RNN_NEURONS;
  out[k++] = DRED_EXTENSION_ID;
  out[k++] = DRED_EXPERIMENTAL_VERSION;
  out[k++] = DRED_EXPERIMENTAL_BYTES;
  out[k++] = DRED_MIN_BYTES;
  out[k++] = DRED_SILK_ENCODER_DELAY;
  out[k++] = DRED_FRAME_SIZE;
  out[k++] = DRED_DFRAME_SIZE;
  out[k++] = DRED_MAX_DATA_SIZE;
  out[k++] = DRED_ENC_Q0;
  out[k++] = DRED_ENC_Q1;
  out[k++] = DRED_MAX_LATENTS;
  out[k++] = DRED_NUM_REDUNDANCY_FRAMES;
  out[k++] = DRED_MAX_FRAMES;
  out[k++] = RESAMPLING_ORDER;
  out[k++] = MAX_DOWNMIX_BUFFER;
  out[k++] = (int)sizeof(((OpusDRED *)0)->fec_features) / (int)sizeof(float);
  out[k++] = (int)sizeof(((OpusDRED *)0)->latents) / (int)sizeof(float);
}

/* ---- RDOVAE encoder ---- */
typedef struct {
  RDOVAEEnc model;
  RDOVAEEncState st;
} DdRdovaeEnc;

void *oracle_dd_rdovae_enc_new(void) {
  DdRdovaeEnc *h = calloc(1, sizeof(*h));
  if (init_rdovaeenc(&h->model, rdovaeenc_arrays) != 0) {
    free(h);
    return NULL;
  }
  return h;
}
void oracle_dd_rdovae_enc_free(void *p) { free(p); }

void oracle_dd_rdovae_encode_dframe(void *p, float *latents, float *initial_state,
                                    const float *input) {
  DdRdovaeEnc *h = p;
  dred_rdovae_encode_dframe(&h->st, &h->model, latents, initial_state, input, 0);
}

#define DD_PUT(field)                                                     \
  do {                                                                    \
    memcpy(out + pos, &st->field, sizeof(st->field));                     \
    pos += (int)(sizeof(st->field) / sizeof(float));                      \
  } while (0)

static int dd_rdovae_enc_state(const RDOVAEEncState *st, float *out) {
  int pos = 0;
  DD_PUT(gru1_state);
  DD_PUT(gru2_state);
  DD_PUT(gru3_state);
  DD_PUT(gru4_state);
  DD_PUT(gru5_state);
  DD_PUT(conv1_state);
  DD_PUT(conv2_state);
  DD_PUT(conv3_state);
  DD_PUT(conv4_state);
  DD_PUT(conv5_state);
  return pos;
}

/* Returns `initialized`; writes the float fields (gru1..5, conv1..5) to out. */
int oracle_dd_rdovae_enc_state(void *p, float *out, int *len) {
  DdRdovaeEnc *h = p;
  *len = dd_rdovae_enc_state(&h->st, out);
  return h->st.initialized;
}

/* ---- RDOVAE decoder ---- */
typedef struct {
  RDOVAEDec model;
  RDOVAEDecState st;
} DdRdovaeDec;

void *oracle_dd_rdovae_dec_new(void) {
  DdRdovaeDec *h = calloc(1, sizeof(*h));
  if (init_rdovaedec(&h->model, rdovaedec_arrays) != 0) {
    free(h);
    return NULL;
  }
  return h;
}
void oracle_dd_rdovae_dec_free(void *p) { free(p); }

void oracle_dd_rdovae_dec_init_states(void *p, const float *initial_state) {
  DdRdovaeDec *h = p;
  dred_rdovae_dec_init_states(&h->st, &h->model, initial_state, 0);
}

void oracle_dd_rdovae_decode_qframe(void *p, float *qframe, const float *z) {
  DdRdovaeDec *h = p;
  dred_rdovae_decode_qframe(&h->st, &h->model, qframe, z, 0);
}

void oracle_dd_rdovae_decode_all(void *p, float *features, const float *state,
                                 const float *latents, int nb_latents) {
  DdRdovaeDec *h = p;
  DRED_rdovae_decode_all(&h->model, features, state, latents, nb_latents, 0);
}

int oracle_dd_rdovae_dec_state(void *p, float *out, int *len) {
  DdRdovaeDec *h = p;
  const RDOVAEDecState *st = &h->st;
  int pos = 0;
  DD_PUT(gru1_state);
  DD_PUT(gru2_state);
  DD_PUT(gru3_state);
  DD_PUT(gru4_state);
  DD_PUT(gru5_state);
  DD_PUT(conv1_state);
  DD_PUT(conv2_state);
  DD_PUT(conv3_state);
  DD_PUT(conv4_state);
  DD_PUT(conv5_state);
  *len = pos;
  return st->initialized;
}

/* ---- DREDEnc ---- */
void *oracle_dd_enc_new(int Fs, int channels) {
  DREDEnc *enc = calloc(1, sizeof(*enc));
  dred_encoder_init(enc, Fs, channels);
  return enc;
}
void oracle_dd_enc_free(void *p) { free(p); }
void oracle_dd_enc_reset(void *p) { dred_encoder_reset((DREDEnc *)p); }
int oracle_dd_enc_loaded(void *p) { return ((DREDEnc *)p)->loaded; }

int oracle_dd_enc_load_model(void *p, const unsigned char *data, int len) {
  return dred_encoder_load_model((DREDEnc *)p, data, len);
}

void oracle_dd_compute_latents(void *p, const float *pcm, int frame_size, int extra_delay) {
  dred_compute_latents((DREDEnc *)p, pcm, frame_size, extra_delay, 0);
}

int oracle_dd_encode_silk_frame(void *p, unsigned char *buf, int max_chunks, int max_bytes,
                                int q0, int dQ, int qmax, unsigned char *activity_mem) {
  return dred_encode_silk_frame((DREDEnc *)p, buf, max_chunks, max_bytes, q0, dQ, qmax,
                                activity_mem, 0);
}

/* Static dred_convert_to_16k (private copy). */
void oracle_dd_convert_to_16k(void *p, const float *in, int in_len, float *out, int out_len) {
  dred_convert_to_16k((DREDEnc *)p, in, in_len, out, out_len);
}

/* Static dred_process_frame on the current input buffer (private copy). */
void oracle_dd_process_frame(void *p) { dred_process_frame((DREDEnc *)p, 0); }

/* ints: input_buffer_fill, dred_offset, latent_offset, last_extra_dred_offset,
   latents_buffer_fill, loaded, Fs, channels, rdovae_enc.initialized. */
#define DD_ENC_INTS 9
void oracle_dd_enc_ints(void *p, int *out) {
  DREDEnc *e = p;
  out[0] = e->input_buffer_fill;
  out[1] = e->dred_offset;
  out[2] = e->latent_offset;
  out[3] = e->last_extra_dred_offset;
  out[4] = e->latents_buffer_fill;
  out[5] = e->loaded;
  out[6] = e->Fs;
  out[7] = e->channels;
  out[8] = e->rdovae_enc.initialized;
}

/* floats: input_buffer, latents_buffer, state_buffer, resample_mem, rdovae_enc state, then the
   LPCNet encoder state (oracle_dc_lpcnet_enc_state order). Returns the count. */
int oracle_dd_enc_floats(void *p, float *out) {
  DREDEnc *st = p;
  int pos = 0;
  DD_PUT(input_buffer);
  DD_PUT(latents_buffer);
  DD_PUT(state_buffer);
  DD_PUT(resample_mem);
  pos += dd_rdovae_enc_state(&st->rdovae_enc, out + pos);
  pos += oracle_dc_lpcnet_enc_state(&st->lpcnet_enc_state, out + pos);
  return pos;
}

/* Overwrites the latent/state buffers and the offsets (to test the payload coder on arbitrary
   latents). */
void oracle_dd_enc_set(void *p, const float *latents, const float *state, int latents_fill,
                       int dred_offset, int latent_offset, int last_extra_dred_offset) {
  DREDEnc *e = p;
  memcpy(e->latents_buffer, latents, sizeof(e->latents_buffer));
  memcpy(e->state_buffer, state, sizeof(e->state_buffer));
  e->latents_buffer_fill = latents_fill;
  e->dred_offset = dred_offset;
  e->latent_offset = latent_offset;
  e->last_extra_dred_offset = last_extra_dred_offset;
}

/* Overwrites the 16 kHz input buffer (to drive dred_process_frame directly). */
void oracle_dd_enc_set_input(void *p, const float *input, int fill) {
  DREDEnc *e = p;
  memcpy(e->input_buffer, input, sizeof(e->input_buffer));
  e->input_buffer_fill = fill;
}

/* ---- helpers ---- */
void oracle_dd_filter_df2t(const float *in, float *out, int len, float b0, const float *b,
                           const float *a, int order, float *mem, int inplace) {
  if (inplace) {
    memcpy(out, in, len * sizeof(float));
    filter_df2t(out, out, len, b0, b, a, order, mem);
  } else {
    filter_df2t(in, out, len, b0, b, a, order, mem);
  }
}

int oracle_dd_voice_active(const unsigned char *activity_mem, int offset) {
  return dred_voice_active(activity_mem, offset);
}

/* Static dred_encode_latents on a fresh encoder of `max_bytes`; finishes the stream. Returns
   ec_tell before ec_enc_done; *rng = final range, *err = error flag. */
int oracle_dd_encode_latents(unsigned char *buf, int max_bytes, const float *x,
                             const unsigned char *scale, const unsigned char *dzone,
                             const unsigned char *r, const unsigned char *p0, int dim,
                             unsigned *rng, int *err) {
  ec_enc enc;
  int tell;
  ec_enc_init(&enc, buf, max_bytes);
  dred_encode_latents(&enc, x, scale, dzone, r, p0, dim, 0);
  tell = ec_tell(&enc);
  ec_enc_done(&enc);
  *rng = enc.rng;
  *err = enc.error;
  return tell;
}

/* dred_decode_latents on a fresh decoder. Returns ec_tell after decoding. */
int oracle_dd_decode_latents(const unsigned char *bytes, int len, float *x,
                             const unsigned char *scale, const unsigned char *r,
                             const unsigned char *p0, int dim, unsigned *rng) {
  ec_dec dec;
  ec_dec_init(&dec, (unsigned char *)bytes, len);
  dred_decode_latents(&dec, x, scale, r, p0, dim);
  *rng = dec.rng;
  return ec_tell(&dec);
}

/* dred_ec_decode on a zeroed OpusDRED (or `init` state when non-NULL). ints: nb_latents,
   process_stage, dred_offset. */
int oracle_dd_ec_decode(const unsigned char *bytes, int num_bytes, int min_feature_frames,
                        int dred_frame_offset, float *state, float *latents, int *ints) {
  OpusDRED *d = calloc(1, sizeof(*d));
  int ret = dred_ec_decode(d, bytes, num_bytes, min_feature_frames, dred_frame_offset);
  memcpy(state, d->state, sizeof(d->state));
  memcpy(latents, d->latents, sizeof(d->latents));
  ints[0] = d->nb_latents;
  ints[1] = d->process_stage;
  ints[2] = d->dred_offset;
  free(d);
  return ret;
}

#endif /* ENABLE_DRED */

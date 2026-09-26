/* Oracle C shims for unit dnn_osce: dnn/osce.c, osce_features.c (LACE, NoLACE, BBWENet BWE).

   Only compiled into oracle builds with the osce feature (build.rs defines ENABLE_OSCE and
   ENABLE_OSCE_BWE).

   Two parts:
   1. Private copies of osce.c and osce_features.c with their exported functions renamed
      (oc_copy_*), which makes the static helpers reachable. They are compiled from the same
      source with the same flags as the library.
   2. A "capturing" SILK decoder: renamed copies of silk/decode_frame.c and silk/dec_API.c
      whose calls to the OSCE API (osce_enhance_frame, osce_reset, osce_bwe, osce_bwe_reset,
      osce_bwe_cross_fade_10ms) and to silk_init_decoder / silk_reset_decoder are routed through
      logging wrappers. The wrappers record every call with its inputs and outputs and then run
      the library function, so a Rust test can replay the exact OSCE call sequence of real SILK
      decoding (with the decoder control, signal and bit count of each frame) and compare.

   `osce_bwe` is also a field name of silk_decoder_state (`.osce_bwe`), so it is always renamed
   with a function-like macro (an object-like one would rename the field too). Part 1 uses
   function-like macros throughout; part 2 uses object-like ones for the rest because the
   silk_* definitions have #ifdef'd parameter lists, which must not appear inside the arguments
   of a function-like macro. */
#ifdef ENABLE_OSCE

#include <stdlib.h>
#include <string.h>

/* ---- 1. private copies (statics) ---- */
#define osce_enhance_frame(...) oc_copy_osce_enhance_frame(__VA_ARGS__)
#define osce_load_models(...) oc_copy_osce_load_models(__VA_ARGS__)
#define osce_reset(...) oc_copy_osce_reset(__VA_ARGS__)
#define osce_bwe(...) oc_copy_osce_bwe(__VA_ARGS__)
#define osce_bwe_reset(...) oc_copy_osce_bwe_reset(__VA_ARGS__)
#define osce_calculate_features(...) oc_copy_osce_calculate_features(__VA_ARGS__)
#define osce_bwe_calculate_features(...) oc_copy_osce_bwe_calculate_features(__VA_ARGS__)
#define osce_cross_fade_10ms(...) oc_copy_osce_cross_fade_10ms(__VA_ARGS__)
#define osce_bwe_cross_fade_10ms(...) oc_copy_osce_bwe_cross_fade_10ms(__VA_ARGS__)
#include "osce_features.c"
#include "osce.c"
#undef osce_enhance_frame
#undef osce_load_models
#undef osce_reset
#undef osce_bwe
#undef osce_bwe_reset
#undef osce_calculate_features
#undef osce_bwe_calculate_features
#undef osce_cross_fade_10ms
#undef osce_bwe_cross_fade_10ms

/* Library prototypes (the header ones were renamed above). */
void osce_enhance_frame(OSCEModel *model, silk_decoder_state *psDec,
                        silk_decoder_control *psDecCtrl, opus_int16 xq[], opus_int32 num_bits,
                        int arch);
int osce_load_models(OSCEModel *hModel, const void *data, int len);
void osce_reset(silk_OSCE_struct *hOSCE, int method);
void osce_bwe(OSCEModel *model, silk_OSCE_BWE_struct *psOSCEBWE, opus_int16 xq48[],
              opus_int16 xq16[], opus_int32 xq16_len, int arch);
void osce_bwe_reset(silk_OSCE_BWE_struct *hOSCEBWE);
void osce_calculate_features(silk_decoder_state *psDec, silk_decoder_control *psDecCtrl,
                             float *features, float *numbits, int *periods,
                             const opus_int16 xq[], opus_int32 num_bits);
void osce_bwe_calculate_features(OSCEBWEFeatureState *psFeatures, float *features,
                                 const opus_int16 xq[], int num_samples);
void osce_cross_fade_10ms(float *x_enhanced, float *x_in, int length);
void osce_bwe_cross_fade_10ms(opus_int16 *x_fadein, opus_int16 *x_fadeout, int length);
opus_int silk_init_decoder(silk_decoder_state *psDec);
opus_int silk_reset_decoder(silk_decoder_state *psDec);

/* ---- state dumps (u32 words: float bit patterns / ints) ---- */
static void oc_put_f(unsigned int *out, int *n, const float *x, int cnt) {
  int i;
  for (i = 0; i < cnt; i++) {
    if (out) memcpy(&out[*n], &x[i], 4);
    (*n)++;
  }
}
static void oc_put_i(unsigned int *out, int *n, int v) {
  if (out) out[*n] = (unsigned int)v;
  (*n)++;
}
static void oc_put_conv(unsigned int *out, int *n, const AdaConvState *s) {
  oc_put_f(out, n, s->history, ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS);
  oc_put_f(out, n, s->last_kernel,
           ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS);
  oc_put_f(out, n, &s->last_gain, 1);
}
static void oc_put_comb(unsigned int *out, int *n, const AdaCombState *s) {
  oc_put_f(out, n, s->history, ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG);
  oc_put_f(out, n, s->last_kernel, ADACOMB_MAX_KERNEL_SIZE);
  oc_put_f(out, n, &s->last_global_gain, 1);
  oc_put_i(out, n, s->last_pitch_lag);
}
static void oc_put_shape(unsigned int *out, int *n, const AdaShapeState *s) {
  oc_put_f(out, n, s->conv_alpha1f_state, ADASHAPE_MAX_INPUT_DIM);
  oc_put_f(out, n, s->conv_alpha1t_state, ADASHAPE_MAX_INPUT_DIM);
  oc_put_f(out, n, s->conv_alpha2_state, ADASHAPE_MAX_FRAME_SIZE);
  oc_put_f(out, n, s->interpolate_state, 1);
}

/* silk_OSCE_struct: method, feature state, then the union member of the current method. */
static int oc_dump_osce(const silk_OSCE_struct *o, unsigned int *out) {
  int n = 0, i;
  const OSCEFeatureState *f = &o->features;
  oc_put_i(out, &n, o->method);
  oc_put_f(out, &n, &f->numbits_smooth, 1);
  oc_put_i(out, &n, f->pitch_hangover_count);
  oc_put_i(out, &n, f->last_lag);
  oc_put_i(out, &n, f->last_type);
  oc_put_i(out, &n, f->reset);
  oc_put_f(out, &n, f->signal_history, OSCE_FEATURES_MAX_HISTORY);
  if (o->method == OSCE_METHOD_LACE) {
    const LACEState *s = &o->state.lace;
    oc_put_f(out, &n, s->feature_net_conv2_state, LACE_FNET_CONV2_STATE_SIZE);
    oc_put_f(out, &n, s->feature_net_gru_state, LACE_COND_DIM);
    oc_put_comb(out, &n, &s->cf1_state);
    oc_put_comb(out, &n, &s->cf2_state);
    oc_put_conv(out, &n, &s->af1_state);
    oc_put_f(out, &n, &s->preemph_mem, 1);
    oc_put_f(out, &n, &s->deemph_mem, 1);
  } else if (o->method == OSCE_METHOD_NOLACE) {
    const NoLACEState *s = &o->state.nolace;
    oc_put_f(out, &n, s->feature_net_conv2_state, NOLACE_FNET_CONV2_STATE_SIZE);
    oc_put_f(out, &n, s->feature_net_gru_state, NOLACE_COND_DIM);
    oc_put_f(out, &n, s->post_cf1_state, NOLACE_COND_DIM);
    oc_put_f(out, &n, s->post_cf2_state, NOLACE_COND_DIM);
    oc_put_f(out, &n, s->post_af1_state, NOLACE_COND_DIM);
    oc_put_f(out, &n, s->post_af2_state, NOLACE_COND_DIM);
    oc_put_f(out, &n, s->post_af3_state, NOLACE_COND_DIM);
    oc_put_comb(out, &n, &s->cf1_state);
    oc_put_comb(out, &n, &s->cf2_state);
    oc_put_conv(out, &n, &s->af1_state);
    oc_put_conv(out, &n, &s->af2_state);
    oc_put_conv(out, &n, &s->af3_state);
    oc_put_conv(out, &n, &s->af4_state);
    oc_put_shape(out, &n, &s->tdshape1_state);
    oc_put_shape(out, &n, &s->tdshape2_state);
    oc_put_shape(out, &n, &s->tdshape3_state);
    oc_put_f(out, &n, &s->preemph_mem, 1);
    oc_put_f(out, &n, &s->deemph_mem, 1);
  }
  (void)i;
  return n;
}

/* silk_OSCE_BWE_struct. */
static int oc_dump_bwe(const silk_OSCE_BWE_struct *b, unsigned int *out) {
  int n = 0, i;
  const BBWENetState *s = &b->state.bbwenet;
  oc_put_f(out, &n, b->features.signal_history, OSCE_BWE_HALF_WINDOW_SIZE);
  oc_put_f(out, &n, b->features.last_spec, 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2);
  oc_put_f(out, &n, s->feature_net_conv1_state, BBWENET_FNET_CONV1_STATE_SIZE);
  oc_put_f(out, &n, s->feature_net_conv2_state, BBWENET_FNET_CONV2_STATE_SIZE);
  oc_put_f(out, &n, s->feature_net_gru_state, BBWENET_FNET_GRU_STATE_SIZE);
  for (i = 0; i < OSCE_BWE_OUTPUT_DELAY; i++) oc_put_i(out, &n, s->outbut_buffer[i]);
  oc_put_conv(out, &n, &s->af1_state);
  oc_put_conv(out, &n, &s->af2_state);
  oc_put_conv(out, &n, &s->af3_state);
  oc_put_shape(out, &n, &s->tdshape1_state);
  oc_put_shape(out, &n, &s->tdshape2_state);
  for (i = 0; i < 3; i++) {
    oc_put_f(out, &n, s->resampler_state[i].upsamp_buffer[0], 3);
    oc_put_f(out, &n, s->resampler_state[i].upsamp_buffer[1], 3);
    oc_put_f(out, &n, s->resampler_state[i].interpol_buffer, 8);
  }
  return n;
}

/* ---- 2. capturing SILK decoder ---- */
#define OC_EV_INIT 1       /* silk_init_decoder(channel) */
#define OC_EV_RESETDEC 2   /* silk_reset_decoder(channel) */
#define OC_EV_RESET 3      /* osce_reset(channel, method) */
#define OC_EV_ENHANCE 4    /* osce_enhance_frame */
#define OC_EV_BWE_RESET 5  /* osce_bwe_reset(channel) */
#define OC_EV_BWE 6        /* osce_bwe */
#define OC_EV_BWE_XFADE 7  /* osce_bwe_cross_fade_10ms */
#define OC_MAX_EVENTS 48

typedef struct {
  int kind;
  int ch;
  int method;
  int fs_kHz;
  int nb_subfr;
  int LPC_order;
  int signal_type;
  int num_bits;
  int len;
  int model_loaded;
  int pitchL[4];
  int Gains_Q16[4];
  short PredCoef_Q12[2 * 16];
  short LTPCoef_Q14[4 * 5];
  short in[960];
  short in2[960];
  short out[960];
} oracle_osce_event;

int oracle_osce_event_size(void) { return (int)sizeof(oracle_osce_event); }

typedef struct {
  silk_decoder_state *ch[2];
  int n;
  int overflow;
  oracle_osce_event ev[OC_MAX_EVENTS];
} oc_log;

static __thread oc_log *oc_cur;

static oracle_osce_event *oc_new_event(int kind) {
  oracle_osce_event *e;
  if (!oc_cur) return NULL;
  if (oc_cur->n >= OC_MAX_EVENTS) {
    oc_cur->overflow = 1;
    return NULL;
  }
  e = &oc_cur->ev[oc_cur->n++];
  memset(e, 0, sizeof(*e));
  e->kind = kind;
  e->ch = -1;
  return e;
}

static int oc_ch_of_state(const silk_decoder_state *p) {
  int n;
  if (!oc_cur) return -1;
  for (n = 0; n < 2; n++)
    if (oc_cur->ch[n] == p) return n;
  return -1;
}
static int oc_ch_of_osce(const silk_OSCE_struct *p) {
  int n;
  if (!oc_cur) return -1;
  for (n = 0; n < 2; n++)
    if (oc_cur->ch[n] && &oc_cur->ch[n]->osce == p) return n;
  return -1;
}
static int oc_ch_of_bwe(const silk_OSCE_BWE_struct *p) {
  int n;
  if (!oc_cur) return -1;
  for (n = 0; n < 2; n++)
    if (oc_cur->ch[n] && &oc_cur->ch[n]->osce_bwe == p) return n;
  return -1;
}

static opus_int cap_silk_init_decoder(silk_decoder_state *psDec) {
  oracle_osce_event *e = oc_new_event(OC_EV_INIT);
  if (e) e->ch = oc_ch_of_state(psDec);
  return silk_init_decoder(psDec);
}

static opus_int cap_silk_reset_decoder(silk_decoder_state *psDec) {
  oracle_osce_event *e = oc_new_event(OC_EV_RESETDEC);
  if (e) e->ch = oc_ch_of_state(psDec);
  return silk_reset_decoder(psDec);
}

static void cap_osce_reset(silk_OSCE_struct *hOSCE, int method) {
  oracle_osce_event *e = oc_new_event(OC_EV_RESET);
  if (e) {
    e->ch = oc_ch_of_osce(hOSCE);
    e->method = method;
  }
  osce_reset(hOSCE, method);
}

static void cap_osce_enhance_frame(OSCEModel *model, silk_decoder_state *psDec,
                                   silk_decoder_control *psDecCtrl, opus_int16 xq[],
                                   opus_int32 num_bits, int arch) {
  oracle_osce_event *e = oc_new_event(OC_EV_ENHANCE);
  int len = psDec->frame_length, k;
  if (e) {
    e->ch = oc_ch_of_state(psDec);
    e->method = psDec->osce.method;
    e->fs_kHz = psDec->fs_kHz;
    e->nb_subfr = psDec->nb_subfr;
    e->LPC_order = psDec->LPC_order;
    e->signal_type = psDec->indices.signalType;
    e->num_bits = num_bits;
    e->len = len;
    e->model_loaded = model->loaded;
    for (k = 0; k < 4; k++) {
      e->pitchL[k] = psDecCtrl->pitchL[k];
      e->Gains_Q16[k] = psDecCtrl->Gains_Q16[k];
    }
    for (k = 0; k < 16; k++) {
      e->PredCoef_Q12[k] = psDecCtrl->PredCoef_Q12[0][k];
      e->PredCoef_Q12[16 + k] = psDecCtrl->PredCoef_Q12[1][k];
    }
    for (k = 0; k < 20; k++) e->LTPCoef_Q14[k] = psDecCtrl->LTPCoef_Q14[k];
    memcpy(e->in, xq, len * sizeof(short));
  }
  osce_enhance_frame(model, psDec, psDecCtrl, xq, num_bits, arch);
  if (e) memcpy(e->out, xq, len * sizeof(short));
}

static void cap_osce_bwe_reset(silk_OSCE_BWE_struct *hOSCEBWE) {
  oracle_osce_event *e = oc_new_event(OC_EV_BWE_RESET);
  if (e) e->ch = oc_ch_of_bwe(hOSCEBWE);
  osce_bwe_reset(hOSCEBWE);
}

static void cap_osce_bwe(OSCEModel *model, silk_OSCE_BWE_struct *psOSCEBWE, opus_int16 xq48[],
                         opus_int16 xq16[], opus_int32 xq16_len, int arch) {
  oracle_osce_event *e = oc_new_event(OC_EV_BWE);
  if (e) {
    e->ch = oc_ch_of_bwe(psOSCEBWE);
    e->len = xq16_len;
    e->model_loaded = model->loaded;
    memcpy(e->in, xq16, xq16_len * sizeof(short));
  }
  osce_bwe(model, psOSCEBWE, xq48, xq16, xq16_len, arch);
  if (e) memcpy(e->out, xq48, 3 * xq16_len * sizeof(short));
}

static void cap_osce_bwe_cross_fade_10ms(opus_int16 *x_fadein, opus_int16 *x_fadeout,
                                         int length) {
  oracle_osce_event *e = oc_new_event(OC_EV_BWE_XFADE);
  if (e) {
    e->len = length;
    memcpy(e->in, x_fadein, 480 * sizeof(short));
    memcpy(e->in2, x_fadeout, 480 * sizeof(short));
  }
  osce_bwe_cross_fade_10ms(x_fadein, x_fadeout, length);
  if (e) memcpy(e->out, x_fadein, 480 * sizeof(short));
}

/* Object-like renames, except osce_bwe (also a field name): the silk_* definitions have
   #ifdef'd parameter lists, which must not appear inside function-like macro arguments. */
#define osce_enhance_frame cap_osce_enhance_frame
#define osce_reset cap_osce_reset
#define osce_bwe(...) cap_osce_bwe(__VA_ARGS__)
#define osce_bwe_reset cap_osce_bwe_reset
#define osce_bwe_cross_fade_10ms cap_osce_bwe_cross_fade_10ms
#define silk_init_decoder cap_silk_init_decoder
#define silk_reset_decoder cap_silk_reset_decoder
#define silk_decode_frame oracle_osce_silk_decode_frame
#define silk_LoadOSCEModels oracle_osce_silk_LoadOSCEModels
#define silk_Get_Decoder_Size oracle_osce_silk_Get_Decoder_Size
#define silk_ResetDecoder oracle_osce_silk_ResetDecoder
#define silk_InitDecoder oracle_osce_silk_InitDecoder
#define silk_Decode oracle_osce_silk_Decode
#include "decode_frame.c"
#include "dec_API.c"
#undef osce_enhance_frame
#undef osce_reset
#undef osce_bwe
#undef osce_bwe_reset
#undef osce_bwe_cross_fade_10ms
#undef silk_init_decoder
#undef silk_reset_decoder
#undef silk_decode_frame
#undef silk_LoadOSCEModels
#undef silk_Get_Decoder_Size
#undef silk_ResetDecoder
#undef silk_InitDecoder
#undef silk_Decode

#include "entdec.h"

#define OC_BUF 4096

typedef struct {
  silk_decoder *dec;
  silk_DecControlStruct ctrl;
  ec_dec ec;
  unsigned char buf[OC_BUF];
  float pcm[2 * 960 * 3];
  oc_log log;
} oracle_osce_cap;

static void oc_begin(oracle_osce_cap *h) {
  h->log.ch[0] = &h->dec->channel_state[0];
  h->log.ch[1] = &h->dec->channel_state[1];
  h->log.n = 0;
  h->log.overflow = 0;
  oc_cur = &h->log;
}

void *oracle_osce_cap_new(void) {
  oracle_osce_cap *h = (oracle_osce_cap *)calloc(1, sizeof(oracle_osce_cap));
  if (!h) return NULL;
  h->dec = (silk_decoder *)calloc(1, sizeof(silk_decoder));
  if (!h->dec) {
    free(h);
    return NULL;
  }
  oc_begin(h);
  oracle_osce_silk_InitDecoder(h->dec);
  oc_cur = NULL;
  ec_dec_init(&h->ec, h->buf, 0);
  return h;
}

void oracle_osce_cap_free(void *p) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  if (!h) return;
  free(h->dec);
  free(h);
}

/* silk_LoadOSCEModels (data NULL: compiled-in tables). Returns its result; logs nothing. */
int oracle_osce_cap_load(void *p, const unsigned char *data, int len) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  return oracle_osce_silk_LoadOSCEModels(h->dec, data, len);
}

int oracle_osce_cap_loaded(void *p) { return ((oracle_osce_cap *)p)->dec->osce_model.loaded; }

void oracle_osce_cap_set_loaded(void *p, int v) {
  ((oracle_osce_cap *)p)->dec->osce_model.loaded = v;
}

int oracle_osce_cap_reset(void *p) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  int ret;
  oc_begin(h);
  ret = oracle_osce_silk_ResetDecoder(h->dec);
  oc_cur = NULL;
  return ret;
}

int oracle_osce_cap_init(void *p) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  int ret;
  oc_begin(h);
  ret = oracle_osce_silk_InitDecoder(h->dec);
  oc_cur = NULL;
  return ret;
}

void oracle_osce_cap_ec_init(void *p, const unsigned char *data, int len) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  if (len > OC_BUF) len = OC_BUF;
  if (len > 0) memcpy(h->buf, data, len);
  ec_dec_init(&h->ec, h->buf, len);
}

/* cfg: nChannelsAPI, nChannelsInternal, API_sampleRate, internalSampleRate, payloadSize_ms,
   enable_deep_plc, osce_method, enable_osce_bwe, osce_extended_mode. prev_osce_extended_mode
   persists in the handle (updated by silk_Decode). Events are available until the next call. */
int oracle_osce_cap_decode(void *p, const int cfg[9], int lost, int new_packet, int *n_out) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  opus_int32 n = 0;
  int ret;
  h->ctrl.nChannelsAPI = cfg[0];
  h->ctrl.nChannelsInternal = cfg[1];
  h->ctrl.API_sampleRate = cfg[2];
  h->ctrl.internalSampleRate = cfg[3];
  h->ctrl.payloadSize_ms = cfg[4];
  h->ctrl.enable_deep_plc = cfg[5];
  h->ctrl.osce_method = cfg[6];
  h->ctrl.enable_osce_bwe = cfg[7];
  h->ctrl.osce_extended_mode = cfg[8];
  oc_begin(h);
  ret = oracle_osce_silk_Decode(h->dec, &h->ctrl, lost, new_packet, &h->ec, h->pcm, &n, NULL, 0);
  oc_cur = NULL;
  *n_out = n;
  return ret;
}

int oracle_osce_cap_prev_ext_mode(void *p) {
  return ((oracle_osce_cap *)p)->ctrl.prev_osce_extended_mode;
}
void oracle_osce_cap_set_prev_ext_mode(void *p, int v) {
  ((oracle_osce_cap *)p)->ctrl.prev_osce_extended_mode = v;
}

/* Number of events of the last call (-1 if the log overflowed). */
int oracle_osce_cap_nevents(void *p) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  return h->log.overflow ? -1 : h->log.n;
}

void oracle_osce_cap_event(void *p, int i, oracle_osce_event *out) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  *out = h->log.ev[i];
}

/* Dumps (out NULL: returns the word count). */
int oracle_osce_cap_dump_osce(void *p, int ch, unsigned int *out) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  return oc_dump_osce(&h->dec->channel_state[ch].osce, out);
}
int oracle_osce_cap_dump_bwe(void *p, int ch, unsigned int *out) {
  oracle_osce_cap *h = (oracle_osce_cap *)p;
  return oc_dump_bwe(&h->dec->channel_state[ch].osce_bwe, out);
}

/* ---- standalone model / state handles ---- */

/* OSCEModel loaded with osce_load_models (data NULL: compiled-in tables); *ret receives the
   result and model->loaded is set like silk_LoadOSCEModels. */
void *oracle_osce_model_new(const unsigned char *data, int len, int *ret) {
  OSCEModel *m = (OSCEModel *)calloc(1, sizeof(OSCEModel));
  if (!m) return NULL;
  *ret = osce_load_models(m, data, len);
  m->loaded = (*ret == 0);
  return m;
}
void oracle_osce_model_free(void *p) { free(p); }
void oracle_osce_model_set_loaded(void *p, int v) { ((OSCEModel *)p)->loaded = v; }

/* Model windows: which 0 lace, 1 nolace, 2..4 bbwenet 16/32/48. */
int oracle_osce_model_window(void *p, int which, float *out) {
  OSCEModel *m = (OSCEModel *)p;
  switch (which) {
    case 0: memcpy(out, m->lace.window, sizeof(m->lace.window)); return LACE_OVERLAP_SIZE;
    case 1: memcpy(out, m->nolace.window, sizeof(m->nolace.window)); return LACE_OVERLAP_SIZE;
    case 2: memcpy(out, m->bbwenet.window16, sizeof(m->bbwenet.window16)); return BBWENET_AF1_OVERLAP_SIZE;
    case 3: memcpy(out, m->bbwenet.window32, sizeof(m->bbwenet.window32)); return BBWENET_AF2_OVERLAP_SIZE;
    default: memcpy(out, m->bbwenet.window48, sizeof(m->bbwenet.window48)); return BBWENET_AF3_OVERLAP_SIZE;
  }
}

/* A silk_decoder_state (only the OSCE part and the fields OSCE reads are used). */
void *oracle_osce_dec_new(void) {
  return calloc(1, sizeof(silk_decoder_state));
}
void oracle_osce_dec_free(void *p) { free(p); }

static void oc_set_dec(silk_decoder_state *d, const int info[4]) {
  d->fs_kHz = info[0];
  d->nb_subfr = info[1];
  d->LPC_order = info[2];
  d->indices.signalType = (opus_int8)info[3];
  d->frame_length = d->nb_subfr * 5 * d->fs_kHz;
}

static void oc_set_ctrl(silk_decoder_control *c, const int pitchL[4], const int gains[4],
                        const short pred[32], const short ltp[20]) {
  int k;
  memset(c, 0, sizeof(*c));
  for (k = 0; k < 4; k++) {
    c->pitchL[k] = pitchL[k];
    c->Gains_Q16[k] = gains[k];
  }
  for (k = 0; k < 16; k++) {
    c->PredCoef_Q12[0][k] = pred[k];
    c->PredCoef_Q12[1][k] = pred[16 + k];
  }
  for (k = 0; k < 20; k++) c->LTPCoef_Q14[k] = ltp[k];
}

void oracle_osce_dec_reset(void *p, int method) {
  osce_reset(&((silk_decoder_state *)p)->osce, method);
}

void oracle_osce_dec_features(void *p, const int info[4], const int pitchL[4],
                              const int gains[4], const short pred[32], const short ltp[20],
                              const short *xq, int num_bits, float *features, float *numbits,
                              int *periods) {
  silk_decoder_state *d = (silk_decoder_state *)p;
  silk_decoder_control c;
  oc_set_dec(d, info);
  oc_set_ctrl(&c, pitchL, gains, pred, ltp);
  osce_calculate_features(d, &c, features, numbits, periods, xq, num_bits);
}

void oracle_osce_dec_enhance(void *p, void *model, const int info[4], const int pitchL[4],
                             const int gains[4], const short pred[32], const short ltp[20],
                             short *xq, int num_bits) {
  silk_decoder_state *d = (silk_decoder_state *)p;
  silk_decoder_control c;
  oc_set_dec(d, info);
  oc_set_ctrl(&c, pitchL, gains, pred, ltp);
  osce_enhance_frame((OSCEModel *)model, d, &c, xq, num_bits, 0);
}

int oracle_osce_dec_dump(void *p, unsigned int *out) {
  return oc_dump_osce(&((silk_decoder_state *)p)->osce, out);
}

int oracle_osce_dec_pitch_postprocessing(void *p, int lag, int type) {
  return pitch_postprocessing(&((silk_decoder_state *)p)->osce.features, lag, type);
}

/* LACE / NoLACE internals on the dec handle's union member. */
void oracle_osce_lace_feature_net(void *model, void *p, float *out, const float *features,
                                  const float numbits[2], const int periods[4]) {
  lace_feature_net(&((OSCEModel *)model)->lace, &((silk_decoder_state *)p)->osce.state.lace, out,
                   features, numbits, periods, 0);
}
void oracle_osce_lace_frame(void *model, void *p, float *x_out, const float *x_in,
                            const float *features, const float numbits[2], const int periods[4]) {
  lace_process_20ms_frame(&((OSCEModel *)model)->lace, &((silk_decoder_state *)p)->osce.state.lace,
                          x_out, x_in, features, numbits, periods, 0);
}
void oracle_osce_nolace_feature_net(void *model, void *p, float *out, const float *features,
                                    const float numbits[2], const int periods[4]) {
  nolace_feature_net(&((OSCEModel *)model)->nolace,
                     &((silk_decoder_state *)p)->osce.state.nolace, out, features, numbits,
                     periods, 0);
}
void oracle_osce_nolace_frame(void *model, void *p, float *x_out, const float *x_in,
                              const float *features, const float numbits[2],
                              const int periods[4]) {
  nolace_process_20ms_frame(&((OSCEModel *)model)->nolace,
                            &((silk_decoder_state *)p)->osce.state.nolace, x_out, x_in, features,
                            numbits, periods, 0);
}

/* BWE handle: silk_OSCE_BWE_struct. */
void *oracle_osce_bwe_new(void) { return calloc(1, sizeof(silk_OSCE_BWE_struct)); }
void oracle_osce_bwe_free(void *p) { free(p); }
void oracle_osce_bwe_reset(void *p) { osce_bwe_reset((silk_OSCE_BWE_struct *)p); }
void oracle_osce_bwe_features(void *p, float *features, const short *xq, int n) {
  osce_bwe_calculate_features(&((silk_OSCE_BWE_struct *)p)->features, features, xq, n);
}
void oracle_osce_bwe_run(void *model, void *p, short *xq48, short *xq16, int len) {
  osce_bwe((OSCEModel *)model, (silk_OSCE_BWE_struct *)p, xq48, xq16, len, 0);
}
void oracle_osce_bwe_process_frames(void *model, void *p, float *x_out, const float *x_in,
                                    const float *features, int num_frames) {
  bbwenet_process_frames(&((OSCEModel *)model)->bbwenet,
                         &((silk_OSCE_BWE_struct *)p)->state.bbwenet, x_out, x_in, features,
                         num_frames, 0);
}
void oracle_osce_bwe_feature_net(void *model, void *p, float *out, const float *features,
                                 int num_frames) {
  bbwe_feature_net(&((OSCEModel *)model)->bbwenet, &((silk_OSCE_BWE_struct *)p)->state.bbwenet,
                   out, features, num_frames, 0);
}
int oracle_osce_bwe_dump(void *p, unsigned int *out) {
  return oc_dump_bwe((silk_OSCE_BWE_struct *)p, out);
}

/* ---- static helpers ---- */
void oracle_osce_numbits_embedding(int nolace, float *emb, float numbits, float min_val,
                                   float max_val, int logscale) {
  if (nolace)
    compute_nolace_numbits_embedding(emb, numbits, 8, min_val, max_val, logscale);
  else
    compute_lace_numbits_embedding(emb, numbits, 8, min_val, max_val, logscale);
}

/* bank: 0 clean, 1 noisy, 2 bwe. */
void oracle_osce_apply_filterbank(int bank, float *x_out, const float *x_in) {
  float tmp[OSCE_SPEC_NUM_FREQS];
  memcpy(tmp, x_in, sizeof(tmp));
  if (bank == 0)
    apply_filterbank(x_out, tmp, center_bins_clean, band_weights_clean, OSCE_CLEAN_SPEC_NUM_BANDS);
  else if (bank == 1)
    apply_filterbank(x_out, tmp, center_bins_noisy, band_weights_noisy, OSCE_NOISY_SPEC_NUM_BANDS);
  else
    apply_filterbank(x_out, tmp, center_bins_bwe, band_weights_bwe, OSCE_BWE_NUM_BANDS);
}

void oracle_osce_mag_spec(float *out, const float *in) {
  float tmp[OSCE_SPEC_WINDOW_SIZE];
  memcpy(tmp, in, sizeof(tmp));
  mag_spec_320_onesided(out, tmp);
}

void oracle_osce_log_spectrum_from_lpc(float *spec, const short *a_q12, int lpc_order) {
  opus_int16 a[16];
  memcpy(a, a_q12, lpc_order * sizeof(short));
  calculate_log_spectrum_from_lpc(spec, a, lpc_order);
}

void oracle_osce_cepstrum(float *cepstrum, const float *signal) {
  float tmp[OSCE_SPEC_WINDOW_SIZE];
  memcpy(tmp, signal, sizeof(tmp));
  calculate_cepstrum(cepstrum, tmp);
}

void oracle_osce_acorr(float *acorr, const float *buf, int pos, int lag) {
  calculate_acorr(acorr, (float *)buf + pos, lag);
}

/* resamp_state as 14 floats: upsamp_buffer[2][3], interpol_buffer[8] (in/out). */
void oracle_osce_upsamp_2x(float st[14], float *x_out, const float *x_in, int n) {
  resamp_state s;
  memcpy(&s, st, sizeof(s));
  upsamp_2x(&s, x_out, x_in, n);
  memcpy(st, &s, sizeof(s));
}
void oracle_osce_interpol_3_2(float st[14], float *x_out, const float *x_in, int n) {
  resamp_state s;
  memcpy(&s, st, sizeof(s));
  interpol_3_2(&s, x_out, x_in, n);
  memcpy(st, &s, sizeof(s));
}
int oracle_osce_resamp_state_size(void) { return (int)sizeof(resamp_state); }

void oracle_osce_valin_activation(float *x, int len) { apply_valin_activation(x, len); }

void oracle_osce_cross_fade_10ms(float *x_enhanced, const float *x_in, int length) {
  osce_cross_fade_10ms(x_enhanced, (float *)x_in, length);
}
void oracle_osce_bwe_cross_fade_10ms(short *x_fadein, const short *x_fadeout, int length) {
  osce_bwe_cross_fade_10ms(x_fadein, (short *)x_fadeout, length);
}

#endif /* ENABLE_OSCE */

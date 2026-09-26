/* Oracle C shims for unit dnn_core: dnn/nnet.c, nnet_arch.h (generic path), vec.h, common.h,
   kiss99.c, parse_lpcnet_weights.c, write_lpcnet_weights.c (write_weights), burg.c, freq.c,
   lpcnet_tables.c, pitchdnn.c, lpcnet_enc.c and (OSCE) nndsp.c.

   Only compiled into DNN oracle builds (build.rs defines ENABLE_DEEP_PLC for the deep-plc, dred
   and osce features). Static helpers are reached by #including private copies of burg.c,
   freq.c, lpcnet_enc.c, nndsp.c and nnet_arch.h with their exported symbols renamed (dc_copy_*),
   so they do not clash with the library symbols. Handles wrap C structs on the heap so the Rust
   side never sees C layouts. */
#ifdef ENABLE_DEEP_PLC

#include <stdlib.h>
#include <string.h>

#include "nnet.h"
#include "vec.h"
#include "common.h"
/* kiss99.c is not in any upstream source list (only the unbuilt lpcnet.c uses it): compile it
   here. */
#include "kiss99.c"
#include "pitchdnn.h"
#include "lpcnet.h"
#include "lpcnet_private.h"

/* ---- private copies (statics) ---- */
#define silk_burg_analysis dc_copy_silk_burg_analysis
#include "burg.c"

#define lpcn_compute_band_energy dc_copy_lpcn_compute_band_energy
#define burg_cepstral_analysis dc_copy_burg_cepstral_analysis
#define apply_window dc_copy_apply_window
#define dct dc_copy_dct
#define forward_transform dc_copy_forward_transform
#define lpc_from_cepstrum dc_copy_lpc_from_cepstrum
#define lpc_weighting dc_copy_lpc_weighting
/* freq.h (already included with the library names) is skipped by its guard: declare the
   renamed functions freq.c calls before defining them. */
void dct(float *out, const float *in);
void forward_transform(kiss_fft_cpx *out, const float *in);
#include "freq.c"

#define lpcnet_encoder_get_size dc_copy_lpcnet_encoder_get_size
#define lpcnet_encoder_init dc_copy_lpcnet_encoder_init
#define lpcnet_encoder_load_model dc_copy_lpcnet_encoder_load_model
#define lpcnet_encoder_create dc_copy_lpcnet_encoder_create
#define lpcnet_encoder_destroy dc_copy_lpcnet_encoder_destroy
#define compute_frame_features dc_copy_compute_frame_features
#define preemphasis dc_copy_preemphasis
#define lpcnet_compute_single_frame_features dc_copy_lpcnet_compute_single_frame_features
#define lpcnet_compute_single_frame_features_float dc_copy_lpcnet_compute_single_frame_features_float
#include "lpcnet_enc.c"
#undef lpcnet_encoder_get_size
#undef lpcnet_encoder_init
#undef lpcnet_encoder_load_model
#undef lpcnet_encoder_create
#undef lpcnet_encoder_destroy
#undef compute_frame_features
#undef preemphasis
#undef lpcnet_compute_single_frame_features
#undef lpcnet_compute_single_frame_features_float
#undef lpcn_compute_band_energy
#undef burg_cepstral_analysis
#undef apply_window
#undef dct
#undef forward_transform
#undef lpc_from_cepstrum
#undef lpc_weighting
#undef silk_burg_analysis

/* Library prototypes (the header ones were renamed above). */
float silk_burg_analysis(float A[], const float x[], const float minInvGain,
                         const int subfr_length, const int nb_subfr, const int D);
void lpcn_compute_band_energy(float *bandE, const kiss_fft_cpx *X);
void burg_cepstral_analysis(float *ceps, const float *x);
void apply_window(float *x);
void dct(float *out, const float *in);
void forward_transform(kiss_fft_cpx *out, const float *in);
float lpc_from_cepstrum(float *lpc, const float *cepstrum);
void lpc_weighting(float *lpc, float gamma);
LPCNetEncState *lpcnet_encoder_create(void);
void lpcnet_encoder_destroy(LPCNetEncState *st);
void compute_frame_features(LPCNetEncState *st, const float *in, int arch);
void preemphasis(float *y, float *mem, const float *x, float coef, int N);
int lpcnet_compute_single_frame_features(LPCNetEncState *st, const opus_int16 *pcm,
                                         float features[NB_TOTAL_FEATURES], int arch);
int lpcnet_compute_single_frame_features_float(LPCNetEncState *st, const float *pcm,
                                               float features[NB_TOTAL_FEATURES], int arch);

/* Private copy of the generic nnet_arch.h (conv2d_float, conv2d_3x3_float, vec_swish, relu). */
#undef RTCD_ARCH
#define RTCD_ARCH _dccopy
#include "nnet_arch.h"

#ifdef ENABLE_OSCE
#include "nndsp.h"
#define init_adaconv_state dc_copy_init_adaconv_state
#define init_adacomb_state dc_copy_init_adacomb_state
#define init_adashape_state dc_copy_init_adashape_state
#define compute_overlap_window dc_copy_compute_overlap_window
#define adaconv_process_frame dc_copy_adaconv_process_frame
#define adacomb_process_frame dc_copy_adacomb_process_frame
#define adashape_process_frame dc_copy_adashape_process_frame
#include "nndsp.c"
#undef init_adaconv_state
#undef init_adacomb_state
#undef init_adashape_state
#undef compute_overlap_window
#undef adaconv_process_frame
#undef adacomb_process_frame
#undef adashape_process_frame
/* nndsp.h prototypes (library) were renamed while nndsp.c was included; redeclare. */
void init_adaconv_state(AdaConvState *hAdaConv);
void init_adacomb_state(AdaCombState *hAdaComb);
void init_adashape_state(AdaShapeState *hAdaShape);
void compute_overlap_window(float *window, int overlap_size);
void adaconv_process_frame(AdaConvState *hAdaConv, float *x_out, const float *x_in,
                           const float *features, const LinearLayer *kernel_layer,
                           const LinearLayer *gain_layer, int feature_dim, int frame_size,
                           int overlap_size, int in_channels, int out_channels, int kernel_size,
                           int left_padding, float filter_gain_a, float filter_gain_b,
                           float shape_gain, float *window, int arch);
void adacomb_process_frame(AdaCombState *hAdaComb, float *x_out, const float *x_in,
                           const float *features, const LinearLayer *kernel_layer,
                           const LinearLayer *gain_layer, const LinearLayer *global_gain_layer,
                           int pitch_lag, int feature_dim, int frame_size, int overlap_size,
                           int kernel_size, int left_padding, float filter_gain_a,
                           float filter_gain_b, float log_gain_limit, float *window, int arch);
void adashape_process_frame(AdaShapeState *hAdaShape, float *x_out, const float *x_in,
                            const float *features, const LinearLayer *alpha1f,
                            const LinearLayer *alpha1t, const LinearLayer *alpha2,
                            int feature_dim, int frame_size, int avg_pool_k, int interpolate_k,
                            int arch);
#endif

/* ---- model array tables ---- */
static const WeightArray *dc_model(int model) {
  switch (model) {
  case 0: return pitchdnn_arrays;
  case 1: return plcmodel_arrays;
  case 2: return fargan_arrays;
#ifdef ENABLE_DRED
  case 3: return rdovaeenc_arrays;
  case 4: return rdovaedec_arrays;
#endif
#ifdef ENABLE_OSCE
  case 5: return lacelayers_arrays;
  case 6: return nolacelayers_arrays;
  case 7: return bbwenetlayers_arrays;
#endif
  default: return NULL;
  }
}

int oracle_dc_model_count(int model) {
  const WeightArray *a = dc_model(model);
  int n = 0;
  if (a == NULL) return -1;
  while (a[n].name != NULL) n++;
  return n;
}

/* name (NUL-terminated, up to 255 bytes), type, size of array i. */
void oracle_dc_model_array_info(int model, int i, char *name, int *type, int *size) {
  const WeightArray *a = &dc_model(model)[i];
  strncpy(name, a->name, 255);
  name[255] = 0;
  *type = a->type;
  *size = a->size;
}

void oracle_dc_model_array_data(int model, int i, void *out) {
  const WeightArray *a = &dc_model(model)[i];
  memcpy(out, a->data, a->size);
}

/* write_lpcnet_weights.c:write_weights into memory (returns the number of bytes; writes only
   if out != NULL). */
int oracle_dc_write_blob(int model, unsigned char *out) {
  const WeightArray *list = dc_model(model);
  int i = 0;
  int pos = 0;
  while (list[i].name != NULL) {
    WeightHead h;
    memcpy(h.head, "DNNw", 4);
    h.version = WEIGHT_BLOB_VERSION;
    h.type = list[i].type;
    h.size = list[i].size;
    h.block_size = (h.size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
    OPUS_CLEAR(h.name, sizeof(h.name));
    strncpy(h.name, list[i].name, sizeof(h.name));
    h.name[sizeof(h.name) - 1] = 0;
    if (out) {
      memcpy(out + pos, &h, WEIGHT_BLOCK_SIZE);
      memcpy(out + pos + WEIGHT_BLOCK_SIZE, list[i].data, h.size);
      memset(out + pos + WEIGHT_BLOCK_SIZE + h.size, 0, h.block_size - h.size);
    }
    pos += WEIGHT_BLOCK_SIZE + h.block_size;
    i++;
  }
  return pos;
}

/* parse_weights on a blob: returns the count (or -1); info gets [type, size, data offset,
   name offset] per array (up to max arrays). */
int oracle_dc_parse_weights(const unsigned char *data, int len, int *info, int max) {
  WeightArray *list;
  int n, i;
  n = parse_weights(&list, data, len);
  if (n < 0) return n;
  for (i = 0; i < n && i < max; i++) {
    info[4 * i] = list[i].type;
    info[4 * i + 1] = list[i].size;
    info[4 * i + 2] = (int)((const unsigned char *)list[i].data - data);
    info[4 * i + 3] = (int)((const unsigned char *)list[i].name - data);
  }
  opus_free(list);
  return n;
}

/* ---- LinearLayer handles ---- */
typedef struct {
  LinearLayer l;
  void *own[7];
} dc_linear;

static void *dc_dup(const void *src, size_t n) {
  void *p;
  if (src == NULL) return NULL;
  p = malloc(n > 0 ? n : 1);
  memcpy(p, src, n);
  return p;
}

void *oracle_dc_linear_new(const float *bias, const float *subias, const signed char *weights,
                           int nweights, const float *float_weights, int nfloat,
                           const int *idx, int nidx, const float *diag, const float *scale,
                           int nb_in, int nb_out) {
  dc_linear *h = calloc(1, sizeof(*h));
  h->own[0] = dc_dup(bias, nb_out * sizeof(float));
  h->own[1] = dc_dup(subias, nb_out * sizeof(float));
  h->own[2] = dc_dup(weights, nweights);
  h->own[3] = dc_dup(float_weights, nfloat * sizeof(float));
  h->own[4] = dc_dup(idx, nidx * sizeof(int));
  h->own[5] = dc_dup(diag, nb_out * sizeof(float));
  h->own[6] = dc_dup(scale, nb_out * sizeof(float));
  h->l.bias = h->own[0];
  h->l.subias = h->own[1];
  h->l.weights = h->own[2];
  h->l.float_weights = h->own[3];
  h->l.weights_idx = h->own[4];
  h->l.diag = h->own[5];
  h->l.scale = h->own[6];
  h->l.nb_inputs = nb_in;
  h->l.nb_outputs = nb_out;
  return h;
}

/* linear_init on the compiled-in arrays of `model`; *ret gets the C return value. */
void *oracle_dc_linear_init(int model, const char *bias, const char *subias, const char *weights,
                            const char *float_weights, const char *weights_idx, const char *diag,
                            const char *scale, int nb_in, int nb_out, int *ret) {
  dc_linear *h = calloc(1, sizeof(*h));
  *ret = linear_init(&h->l, dc_model(model), bias, subias, weights, float_weights, weights_idx,
                     diag, scale, nb_in, nb_out);
  return h;
}

void oracle_dc_linear_free(void *p) {
  dc_linear *h = p;
  int i;
  for (i = 0; i < 7; i++) free(h->own[i]);
  free(h);
}

/* Field `f` (0 bias, 1 subias, 2 weights, 3 float_weights, 4 weights_idx, 5 diag, 6 scale):
   returns 0 if NULL, else copies nbytes and returns 1. */
int oracle_dc_linear_field(void *p, int f, void *out, int nbytes) {
  dc_linear *h = p;
  const void *src = NULL;
  switch (f) {
  case 0: src = h->l.bias; break;
  case 1: src = h->l.subias; break;
  case 2: src = h->l.weights; break;
  case 3: src = h->l.float_weights; break;
  case 4: src = h->l.weights_idx; break;
  case 5: src = h->l.diag; break;
  case 6: src = h->l.scale; break;
  }
  if (src == NULL) return 0;
  memcpy(out, src, nbytes);
  return 1;
}

void oracle_dc_linear_dims(void *p, int *nb_in, int *nb_out) {
  dc_linear *h = p;
  *nb_in = h->l.nb_inputs;
  *nb_out = h->l.nb_outputs;
}

void oracle_dc_compute_linear(void *p, float *out, const float *in) {
  compute_linear_c(&((dc_linear *)p)->l, out, in);
}

void oracle_dc_compute_generic_dense(void *p, float *out, const float *in, int act) {
  compute_generic_dense(&((dc_linear *)p)->l, out, in, act, 0);
}

void oracle_dc_compute_generic_gru(void *pin, void *prec, float *state, const float *in) {
  compute_generic_gru(&((dc_linear *)pin)->l, &((dc_linear *)prec)->l, state, in, 0);
}

/* inplace: output is also the input (in is ignored). */
void oracle_dc_compute_glu(void *p, float *out, const float *in, int inplace) {
  if (inplace) compute_glu(&((dc_linear *)p)->l, out, out, 0);
  else compute_glu(&((dc_linear *)p)->l, out, in, 0);
}

void oracle_dc_compute_generic_conv1d(void *p, float *out, float *mem, const float *in,
                                      int input_size, int act) {
  compute_generic_conv1d(&((dc_linear *)p)->l, out, mem, in, input_size, act, 0);
}

void oracle_dc_compute_generic_conv1d_dilation(void *p, float *out, float *mem, const float *in,
                                               int input_size, int dilation, int act) {
  compute_generic_conv1d_dilation(&((dc_linear *)p)->l, out, mem, in, input_size, dilation, act,
                                  0);
}

/* inplace: output is also the input (in is ignored). */
void oracle_dc_compute_activation(float *out, const float *in, int n, int act, int inplace) {
  if (inplace) compute_activation_c(out, out, n, act);
  else compute_activation_c(out, in, n, act);
}

void oracle_dc_vec_swish(float *y, const float *x, int n) { vec_swish(y, x, n); }

/* ---- Conv2dLayer handles ---- */
typedef struct {
  Conv2dLayer l;
  void *own[2];
} dc_conv2d;

void *oracle_dc_conv2d_new(const float *bias, const float *w, int in_ch, int out_ch, int ktime,
                           int kheight) {
  dc_conv2d *h = calloc(1, sizeof(*h));
  h->own[0] = dc_dup(bias, out_ch * sizeof(float));
  h->own[1] = dc_dup(w, (size_t)in_ch * out_ch * ktime * kheight * sizeof(float));
  h->l.bias = h->own[0];
  h->l.float_weights = h->own[1];
  h->l.in_channels = in_ch;
  h->l.out_channels = out_ch;
  h->l.ktime = ktime;
  h->l.kheight = kheight;
  return h;
}

void *oracle_dc_conv2d_init(int model, const char *bias, const char *float_weights, int in_ch,
                            int out_ch, int ktime, int kheight, int *ret) {
  dc_conv2d *h = calloc(1, sizeof(*h));
  *ret = conv2d_init(&h->l, dc_model(model), bias, float_weights, in_ch, out_ch, ktime, kheight);
  return h;
}

void oracle_dc_conv2d_free(void *p) {
  dc_conv2d *h = p;
  free(h->own[0]);
  free(h->own[1]);
  free(h);
}

int oracle_dc_conv2d_field(void *p, int f, void *out, int nbytes) {
  dc_conv2d *h = p;
  const void *src = f == 0 ? (const void *)h->l.bias : (const void *)h->l.float_weights;
  if (src == NULL) return 0;
  memcpy(out, src, nbytes);
  return 1;
}

void oracle_dc_compute_conv2d(void *p, float *out, float *mem, const float *in, int height,
                              int hstride, int act) {
  compute_conv2d_c(&((dc_conv2d *)p)->l, out, mem, in, height, hstride, act);
}

void oracle_dc_conv2d_float(float *out, const float *w, int in_ch, int out_ch, int ktime,
                            int kheight, const float *in, int height, int hstride) {
  conv2d_float(out, w, in_ch, out_ch, ktime, kheight, in, height, hstride);
}

void oracle_dc_conv2d_3x3_float(float *out, const float *w, int in_ch, int out_ch,
                                const float *in, int height, int hstride) {
  conv2d_3x3_float(out, w, in_ch, out_ch, in, height, hstride);
}

/* ---- vec.h ---- */
void oracle_dc_sgemv(float *out, const float *w, int rows, int cols, int stride, const float *x) {
  sgemv(out, w, rows, cols, stride, x);
}
void oracle_dc_sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows,
                               const float *x) {
  sparse_sgemv8x4(out, w, idx, rows, x);
}
void oracle_dc_cgemv8x4(float *out, const signed char *w, const float *scale, int rows, int cols,
                        const float *x) {
  cgemv8x4(out, (const opus_int8 *)w, scale, rows, cols, x);
}
void oracle_dc_sparse_cgemv8x4(float *out, const signed char *w, const int *idx,
                               const float *scale, int rows, int cols, const float *x) {
  sparse_cgemv8x4(out, (const opus_int8 *)w, idx, scale, rows, cols, x);
}
/* which: 0 tanh_approx, 1 sigmoid_approx, 2 lpcnet_exp, 3 lpcnet_exp2, 4 log2_approx,
   5 log_approx, 6 ulaw2lin, 7 relu. */
void oracle_dc_scalar(int which, const float *x, float *y, int n) {
  int i;
  for (i = 0; i < n; i++) {
    switch (which) {
    case 0: y[i] = tanh_approx(x[i]); break;
    case 1: y[i] = sigmoid_approx(x[i]); break;
    case 2: y[i] = lpcnet_exp(x[i]); break;
    case 3: y[i] = lpcnet_exp2(x[i]); break;
    case 4: y[i] = log2_approx(x[i]); break;
    case 5: y[i] = log_approx(x[i]); break;
    case 6: y[i] = ulaw2lin(x[i]); break;
    case 7: y[i] = relu(x[i]); break;
    }
  }
}
void oracle_dc_lin2ulaw(const float *x, int *y, int n) {
  int i;
  for (i = 0; i < n; i++) y[i] = lin2ulaw(x[i]);
}

/* ---- kiss99 ---- */
void oracle_dc_kiss99(const unsigned char *seed, int nseed, unsigned *state4, unsigned *out,
                      int n) {
  kiss99_ctx c;
  int i;
  kiss99_srand(&c, seed, nseed);
  state4[0] = c.z;
  state4[1] = c.w;
  state4[2] = c.jsr;
  state4[3] = c.jcong;
  for (i = 0; i < n; i++) out[i] = kiss99_rand(&c);
}

/* ---- burg ---- */
float oracle_dc_silk_burg_analysis(float *A, const float *x, float minInvGain, int subfr_length,
                                   int nb_subfr, int D) {
  return silk_burg_analysis(A, x, minInvGain, subfr_length, nb_subfr, D);
}
double oracle_dc_silk_energy_flp(const float *x, int n) { return silk_energy_FLP(x, n); }
double oracle_dc_silk_inner_product_flp(const float *x, const float *y, int n) {
  return silk_inner_product_FLP(x, y, n);
}

/* ---- freq ---- */
void oracle_dc_lpcn_compute_band_energy(float *bandE, const float *X) {
  lpcn_compute_band_energy(bandE, (const kiss_fft_cpx *)X);
}
void oracle_dc_compute_band_energy_inverse(float *bandE, const float *X) {
  compute_band_energy_inverse(bandE, (const kiss_fft_cpx *)X);
}
float oracle_dc_lpcn_lpc(float *lpc, float *rc, const float *ac, int p) {
  return lpcn_lpc(lpc, rc, ac, p);
}
void oracle_dc_compute_burg_cepstrum(const float *pcm, float *ceps, int len, int order) {
  compute_burg_cepstrum(pcm, ceps, len, order);
}
void oracle_dc_burg_cepstral_analysis(float *ceps, const float *x) {
  burg_cepstral_analysis(ceps, x);
}
void oracle_dc_interp_band_gain(float *g, const float *bandE) { interp_band_gain(g, bandE); }
void oracle_dc_dct(float *out, const float *in) { dct(out, in); }
void oracle_dc_idct(float *out, const float *in) { idct(out, in); }
void oracle_dc_forward_transform(float *out, const float *in) {
  forward_transform((kiss_fft_cpx *)out, in);
}
void oracle_dc_inverse_transform(float *out, const float *in) {
  inverse_transform(out, (const kiss_fft_cpx *)in);
}
float oracle_dc_lpc_from_bands(float *lpc, const float *Ex) { return lpc_from_bands(lpc, Ex); }
void oracle_dc_lpc_weighting(float *lpc, float gamma) { lpc_weighting(lpc, gamma); }
float oracle_dc_lpc_from_cepstrum(float *lpc, const float *ceps) {
  return lpc_from_cepstrum(lpc, ceps);
}
void oracle_dc_apply_window(float *x) { apply_window(x); }

/* Tables: kfft (nfft, shift, scale, factors[16], bitrev[320], twiddles[640]), half_window,
   dct_table. */
void oracle_dc_tables(int *nfft, int *shift, float *scale, short *factors, short *bitrev,
                      float *twiddles, float *half_win, float *dct_tab) {
  int i;
  *nfft = kfft.nfft;
  *shift = kfft.shift;
  *scale = kfft.scale;
  for (i = 0; i < 2 * MAXFACTORS; i++) factors[i] = kfft.factors[i];
  for (i = 0; i < kfft.nfft; i++) {
    bitrev[i] = kfft.bitrev[i];
    twiddles[2 * i] = kfft.twiddles[i].r;
    twiddles[2 * i + 1] = kfft.twiddles[i].i;
  }
  memcpy(half_win, half_window, OVERLAP_SIZE * sizeof(float));
  memcpy(dct_tab, dct_table, NB_BANDS * NB_BANDS * sizeof(float));
}

/* ---- pitchdnn ---- */
void *oracle_dc_pitchdnn_new(void) {
  PitchDNNState *st = malloc(sizeof(*st));
  pitchdnn_init(st);
  return st;
}
void oracle_dc_pitchdnn_free(void *p) { free(p); }
float oracle_dc_compute_pitchdnn(void *p, const float *if_features, const float *xcorr) {
  return compute_pitchdnn((PitchDNNState *)p, if_features, xcorr, 0);
}
/* gru_state[64], xcorr_mem1[452], xcorr_mem2[3616] */
void oracle_dc_pitchdnn_state(void *p, float *out) {
  PitchDNNState *st = p;
  memcpy(out, st->gru_state, sizeof(st->gru_state));
  out += GRU_1_STATE_SIZE;
  memcpy(out, st->xcorr_mem1, sizeof(st->xcorr_mem1));
  out += sizeof(st->xcorr_mem1) / sizeof(float);
  memcpy(out, st->xcorr_mem2, sizeof(st->xcorr_mem2));
}

/* ---- lpcnet_enc ---- */
void *oracle_dc_lpcnet_enc_new(void) { return lpcnet_encoder_create(); }
void oracle_dc_lpcnet_enc_free(void *p) { lpcnet_encoder_destroy(p); }
void oracle_dc_lpcnet_features(void *p, const short *pcm, float *features) {
  lpcnet_compute_single_frame_features((LPCNetEncState *)p, pcm, features, 0);
}
void oracle_dc_lpcnet_features_float(void *p, const float *pcm, float *features) {
  lpcnet_compute_single_frame_features_float((LPCNetEncState *)p, pcm, features, 0);
}
void oracle_dc_compute_frame_features(void *p, const float *in) {
  compute_frame_features((LPCNetEncState *)p, in, 0);
}
void oracle_dc_frame_analysis(void *p, float *X, float *Ex, const float *in) {
  frame_analysis((LPCNetEncState *)p, (kiss_fft_cpx *)X, Ex, in);
}
void oracle_dc_biquad(float *y, float *mem, const float *x, const float *b, const float *a,
                      int n, int inplace) {
  if (inplace) biquad(y, mem, y, b, a, n);
  else biquad(y, mem, x, b, a, n);
}
void oracle_dc_preemphasis(float *y, float *mem, const float *x, float coef, int n, int inplace) {
  if (inplace) preemphasis(y, mem, y, coef, n);
  else preemphasis(y, mem, x, coef, n);
}
#define DC_PUT(field)                                   \
  do {                                                  \
    memcpy(out + pos, &(st->field), sizeof(st->field)); \
    pos += sizeof(st->field) / (sizeof(float));           \
  } while (0)
/* Flattened float fields in struct order (see Rust `flatten_enc_state`); returns the count. */
int oracle_dc_lpcnet_enc_state(void *p, float *out) {
  LPCNetEncState *st = p;
  int pos = 0;
  DC_PUT(analysis_mem);
  DC_PUT(mem_preemph);
  DC_PUT(prev_if);
  DC_PUT(if_features);
  DC_PUT(xcorr_features);
  DC_PUT(dnn_pitch);
  DC_PUT(pitch_mem);
  DC_PUT(pitch_filt);
  DC_PUT(exc_buf);
  DC_PUT(lp_buf);
  DC_PUT(lp_mem);
  DC_PUT(lpc);
  DC_PUT(features);
  DC_PUT(sig_mem);
  DC_PUT(burg_cepstrum);
  DC_PUT(pitchdnn.gru_state);
  DC_PUT(pitchdnn.xcorr_mem1);
  DC_PUT(pitchdnn.xcorr_mem2);
  return pos;
}

#ifdef ENABLE_OSCE
/* ---- nndsp ---- */
void *oracle_dc_adaconv_new(void) {
  AdaConvState *s = malloc(sizeof(*s));
  init_adaconv_state(s);
  return s;
}
void *oracle_dc_adacomb_new(void) {
  AdaCombState *s = malloc(sizeof(*s));
  init_adacomb_state(s);
  return s;
}
void *oracle_dc_adashape_new(void) {
  AdaShapeState *s = malloc(sizeof(*s));
  init_adashape_state(s);
  return s;
}
void oracle_dc_nndsp_free(void *p) { free(p); }

/* history[96], last_kernel[288], last_gain */
void oracle_dc_adaconv_state(void *p, float *out) {
  AdaConvState *s = p;
  memcpy(out, s->history, sizeof(s->history));
  memcpy(out + 96, s->last_kernel, sizeof(s->last_kernel));
  out[96 + 288] = s->last_gain;
}
/* history[316], last_kernel[16], last_global_gain; *lag = last_pitch_lag */
void oracle_dc_adacomb_state(void *p, float *out, int *lag) {
  AdaCombState *s = p;
  memcpy(out, s->history, sizeof(s->history));
  memcpy(out + 316, s->last_kernel, sizeof(s->last_kernel));
  out[316 + 16] = s->last_global_gain;
  *lag = s->last_pitch_lag;
}
/* conv_alpha1f_state[512], conv_alpha1t_state[512], conv_alpha2_state[240],
   interpolate_state[1] */
void oracle_dc_adashape_state(void *p, float *out) {
  AdaShapeState *s = p;
  memcpy(out, s->conv_alpha1f_state, sizeof(s->conv_alpha1f_state));
  memcpy(out + 512, s->conv_alpha1t_state, sizeof(s->conv_alpha1t_state));
  memcpy(out + 1024, s->conv_alpha2_state, sizeof(s->conv_alpha2_state));
  out[1264] = s->interpolate_state[0];
}

void oracle_dc_compute_overlap_window(float *window, int n) { compute_overlap_window(window, n); }
void oracle_dc_scale_kernel(float *kernel, int in_ch, int out_ch, int ksize, float *gain) {
  scale_kernel(kernel, in_ch, out_ch, ksize, gain);
}
void oracle_dc_transform_gains(float *gains, int n, float a, float b) {
  transform_gains(gains, n, a, b);
}

/* inplace: x_out is also x_in. */
void oracle_dc_adaconv(void *st, float *x_out, const float *x_in, const float *features,
                       void *kernel, void *gain, int feature_dim, int frame_size,
                       int overlap_size, int in_ch, int out_ch, int ksize, int left_padding,
                       float ga, float gb, float shape_gain, float *window, int inplace) {
  adaconv_process_frame(st, x_out, inplace ? x_out : x_in, features, &((dc_linear *)kernel)->l,
                        &((dc_linear *)gain)->l, feature_dim, frame_size, overlap_size, in_ch,
                        out_ch, ksize, left_padding, ga, gb, shape_gain, window, 0);
}
void oracle_dc_adacomb(void *st, float *x_out, const float *x_in, const float *features,
                       void *kernel, void *gain, void *global_gain, int pitch_lag,
                       int feature_dim, int frame_size, int overlap_size, int ksize,
                       int left_padding, float ga, float gb, float log_gain_limit,
                       float *window, int inplace) {
  adacomb_process_frame(st, x_out, inplace ? x_out : x_in, features, &((dc_linear *)kernel)->l,
                        &((dc_linear *)gain)->l, &((dc_linear *)global_gain)->l, pitch_lag,
                        feature_dim, frame_size, overlap_size, ksize, left_padding, ga, gb,
                        log_gain_limit, window, 0);
}
void oracle_dc_adashape(void *st, float *x_out, const float *x_in, const float *features,
                        void *a1f, void *a1t, void *a2, int feature_dim, int frame_size,
                        int avg_pool_k, int interpolate_k, int inplace) {
  adashape_process_frame(st, x_out, inplace ? x_out : x_in, features, &((dc_linear *)a1f)->l,
                         &((dc_linear *)a1t)->l, &((dc_linear *)a2)->l, feature_dim, frame_size,
                         avg_pool_k, interpolate_k, 0);
}
#endif /* ENABLE_OSCE */

#endif /* ENABLE_DEEP_PLC */

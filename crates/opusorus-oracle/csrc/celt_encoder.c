/* Oracle C shims for unit celt_encoder (celt/celt_encoder.c).
 *
 * Part A compiles a private copy of celt/celt_encoder.c with its external symbols renamed
 * (oracle_ce_dup_*). It provides the private `struct OpusCustomEncoder` definition (so encoder
 * states can be dumped field by field) and reaches the static helpers.
 *
 * Part B compiles a private copy of src/opus_encoder.c (external symbols renamed) in which every
 * call to celt_encode_with_ec() is redirected to an interceptor. The interceptor records the
 * complete CELT encoder state, input, output buffers and range coder state before and after
 * each call (then runs the real library function), so the Rust port can replay every CELT call
 * the real Opus encoder makes (CELT-only, hybrid with a shared range coder, redundancy frames,
 * prefill, QEXT, ...).
 *
 * Part C is a direct API over a persistent library CELT encoder plus shims for the static
 * helpers.
 *
 * Built in the float and in the fixed-point oracle: every signal/state value uses the build's
 * C type (opus_res, celt_sig, celt_norm, celt_glog, opus_val16/32); the AnalysisInfo fields
 * are float in both builds. */
// oracle-build: any
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "opus_types.h"
#include "opus_defines.h"

/* ------------------------------------------------------------------------------------------ */
/* Part A: private copy of celt_encoder.c                                                       */
/* ------------------------------------------------------------------------------------------ */

#define celt_encoder_get_size oracle_ce_dup_celt_encoder_get_size
#define opus_custom_encoder_get_size oracle_ce_dup_opus_custom_encoder_get_size
#define opus_custom_encoder_create oracle_ce_dup_opus_custom_encoder_create
#define opus_custom_encoder_init oracle_ce_dup_opus_custom_encoder_init
#define celt_encoder_init oracle_ce_dup_celt_encoder_init
#define opus_custom_encoder_destroy oracle_ce_dup_opus_custom_encoder_destroy
#define celt_preemphasis oracle_ce_dup_celt_preemphasis
#define celt_encode_with_ec oracle_ce_dup_celt_encode_with_ec
#define opus_custom_encode oracle_ce_dup_opus_custom_encode
#define opus_custom_encode24 oracle_ce_dup_opus_custom_encode24
#define opus_custom_encode_float oracle_ce_dup_opus_custom_encode_float
#define opus_custom_encoder_ctl oracle_ce_dup_opus_custom_encoder_ctl
/* Non-static fixed-point helpers (also defined by the library's celt_encoder.o). */
#define normalize_tone_input oracle_ce_dup_normalize_tone_input
#define acos_approx oracle_ce_dup_acos_approx
/* Relative path: a bare "celt_encoder.c" would resolve to this shim itself. */
#include "../../../vendor/libopus/celt/celt_encoder.c"
#undef celt_encoder_get_size
#undef opus_custom_encoder_get_size
#undef opus_custom_encoder_create
#undef opus_custom_encoder_init
#undef celt_encoder_init
#undef opus_custom_encoder_destroy
#undef celt_preemphasis
#undef celt_encode_with_ec
#undef opus_custom_encode
#undef opus_custom_encode24
#undef opus_custom_encode_float
#undef opus_custom_encoder_ctl
#undef normalize_tone_input
#undef acos_approx

/* The real library functions (their header prototypes were renamed above). */
int celt_encoder_get_size(int channels);
int celt_encoder_init(CELTEncoder *st, opus_int32 sampling_rate, int channels, int arch);
int opus_custom_encoder_ctl(OpusCustomEncoder *st, int request, ...);
int celt_encode_with_ec(OpusCustomEncoder *st, const opus_res *pcm, int frame_size,
                        unsigned char *compressed, int nbCompressedBytes, ec_enc *enc);
void celt_preemphasis(const opus_res *pcmp, celt_sig *inp, int N, int CC, int upsample,
                      const opus_val16 *coef, celt_sig *mem, int clip);
#ifdef CUSTOM_MODES
OpusCustomEncoder *opus_custom_encoder_create(const OpusCustomMode *mode, int channels,
                                              int *error);
void opus_custom_encoder_destroy(OpusCustomEncoder *st);
int opus_custom_encode(OpusCustomEncoder *st, const opus_int16 *pcm, int frame_size,
                       unsigned char *compressed, int maxCompressedBytes);
int opus_custom_encode24(OpusCustomEncoder *st, const opus_int32 *pcm, int frame_size,
                         unsigned char *compressed, int maxCompressedBytes);
int opus_custom_encode_float(OpusCustomEncoder *st, const float *pcm, int frame_size,
                             unsigned char *compressed, int maxCompressedBytes);
#endif

int oracle_ce_has_qext(void) {
#ifdef ENABLE_QEXT
  return 1;
#else
  return 0;
#endif
}

int oracle_ce_has_custom(void) {
#ifdef CUSTOM_MODES
  return 1;
#else
  return 0;
#endif
}

/* ---- Flat state dump (mirrored by a #[repr(C)] Rust struct) ---- */

#define CE_MAX_OVERLAP 1024
#define CE_MAX_PERIOD 2048
#define CE_MAX_BANDS 64
#define CE_QEXT_BANDS 14

typedef struct {
  int channels, stream_channels, force_intra, clip, disable_pf, complexity, upsample, start, end;
  int bitrate, vbr, signalling, constrained_vbr, loss_rate, lsb_depth, lfe, disable_inv;
  int enable_qext, qext_scale;
  int mode_fs, mode_short_mdct_size, mode_nb_short_mdcts, mode_nb_ebands, mode_overlap;
  unsigned int rng;
  int spread_decision;
  opus_val32 delayed_intra;
  int tonal_average, last_coded_bands, hf_average, tapset_decision;
  int prefilter_period;
  opus_val16 prefilter_gain;
  int prefilter_tapset, consec_transient;
  int an_valid;
  float an_tonality, an_tonality_slope, an_noisiness, an_activity, an_music_prob,
      an_music_prob_min, an_music_prob_max;
  int an_bandwidth;
  float an_activity_probability, an_max_pitch_ratio;
  unsigned char an_leak_boost[20];
  int silk_signal_type, silk_offset;
  opus_val32 preemph_mem_e[2], preemph_mem_d[2];
  int vbr_reservoir, vbr_drift, vbr_offset, vbr_count;
  opus_val32 overlap_max;
  opus_val16 stereo_saving;
  int intensity;
  int has_energy_mask;
  celt_glog energy_mask[2 * CE_MAX_BANDS];
  celt_glog spec_avg;
  celt_sig in_mem[2 * CE_MAX_OVERLAP];
  celt_sig prefilter_mem[2 * CE_MAX_PERIOD];
  celt_glog old_band_e[2 * CE_MAX_BANDS];
  celt_glog old_log_e[2 * CE_MAX_BANDS];
  celt_glog old_log_e2[2 * CE_MAX_BANDS];
  celt_glog energy_error[2 * CE_MAX_BANDS];
  celt_glog qext_old_band_e[2 * CE_QEXT_BANDS];
} OracleCeState;

size_t oracle_ce_state_size(void) { return sizeof(OracleCeState); }
size_t oracle_ce_state_offset(int which) {
  switch (which) {
    case 0: return offsetof(OracleCeState, rng);
    case 1: return offsetof(OracleCeState, an_leak_boost);
    case 2: return offsetof(OracleCeState, silk_signal_type);
    case 3: return offsetof(OracleCeState, energy_mask);
    case 4: return offsetof(OracleCeState, in_mem);
    case 5: return offsetof(OracleCeState, qext_old_band_e);
    case 6: return offsetof(OracleCeState, prefilter_gain);
    case 7: return offsetof(OracleCeState, stereo_saving);
    case 8: return offsetof(OracleCeState, spec_avg);
    default: return 0;
  }
}

static int st_qext_scale(const CELTEncoder *st) {
#ifdef ENABLE_QEXT
  return st->qext_scale;
#else
  (void)st;
  return 1;
#endif
}

static void dump_state(const CELTEncoder *st, OracleCeState *d) {
  const CELTMode *m = st->mode;
  int CC = st->channels, nb = m->nbEBands, ov = m->overlap;
  int mp = st_qext_scale(st) * COMBFILTER_MAXPERIOD;
  const celt_sig *prefilter_mem = st->in_mem + CC * ov;
  const celt_glog *oldBandE = (const celt_glog *)(st->in_mem + CC * (ov + mp));
  const celt_glog *oldLogE = oldBandE + CC * nb;
  const celt_glog *oldLogE2 = oldLogE + CC * nb;
  const celt_glog *energyError = oldLogE2 + CC * nb;
  memset(d, 0, sizeof(*d));
  d->channels = st->channels;
  d->stream_channels = st->stream_channels;
  d->force_intra = st->force_intra;
  d->clip = st->clip;
  d->disable_pf = st->disable_pf;
  d->complexity = st->complexity;
  d->upsample = st->upsample;
  d->start = st->start;
  d->end = st->end;
  d->bitrate = st->bitrate;
  d->vbr = st->vbr;
  d->signalling = st->signalling;
  d->constrained_vbr = st->constrained_vbr;
  d->loss_rate = st->loss_rate;
  d->lsb_depth = st->lsb_depth;
  d->lfe = st->lfe;
  d->disable_inv = st->disable_inv;
#ifdef ENABLE_QEXT
  d->enable_qext = st->enable_qext;
  d->qext_scale = st->qext_scale;
#else
  d->qext_scale = 1;
#endif
  d->mode_fs = m->Fs;
  d->mode_short_mdct_size = m->shortMdctSize;
  d->mode_nb_short_mdcts = m->nbShortMdcts;
  d->mode_nb_ebands = nb;
  d->mode_overlap = ov;
  d->rng = st->rng;
  d->spread_decision = st->spread_decision;
  d->delayed_intra = st->delayedIntra;
  d->tonal_average = st->tonal_average;
  d->last_coded_bands = st->lastCodedBands;
  d->hf_average = st->hf_average;
  d->tapset_decision = st->tapset_decision;
  d->prefilter_period = st->prefilter_period;
  d->prefilter_gain = st->prefilter_gain;
  d->prefilter_tapset = st->prefilter_tapset;
  d->consec_transient = st->consec_transient;
  d->an_valid = st->analysis.valid;
  d->an_tonality = st->analysis.tonality;
  d->an_tonality_slope = st->analysis.tonality_slope;
  d->an_noisiness = st->analysis.noisiness;
  d->an_activity = st->analysis.activity;
  d->an_music_prob = st->analysis.music_prob;
  d->an_music_prob_min = st->analysis.music_prob_min;
  d->an_music_prob_max = st->analysis.music_prob_max;
  d->an_bandwidth = st->analysis.bandwidth;
  d->an_activity_probability = st->analysis.activity_probability;
  d->an_max_pitch_ratio = st->analysis.max_pitch_ratio;
  memcpy(d->an_leak_boost, st->analysis.leak_boost, LEAK_BANDS);
  d->silk_signal_type = st->silk_info.signalType;
  d->silk_offset = st->silk_info.offset;
  memcpy(d->preemph_mem_e, st->preemph_memE, sizeof(d->preemph_mem_e));
  memcpy(d->preemph_mem_d, st->preemph_memD, sizeof(d->preemph_mem_d));
  d->vbr_reservoir = st->vbr_reservoir;
  d->vbr_drift = st->vbr_drift;
  d->vbr_offset = st->vbr_offset;
  d->vbr_count = st->vbr_count;
  d->overlap_max = st->overlap_max;
  d->stereo_saving = st->stereo_saving;
  d->intensity = st->intensity;
  d->has_energy_mask = st->energy_mask != NULL;
  if (st->energy_mask) memcpy(d->energy_mask, st->energy_mask, CC * nb * sizeof(celt_glog));
  d->spec_avg = st->spec_avg;
  memcpy(d->in_mem, st->in_mem, CC * ov * sizeof(celt_sig));
  memcpy(d->prefilter_mem, prefilter_mem, CC * mp * sizeof(celt_sig));
  memcpy(d->old_band_e, oldBandE, CC * nb * sizeof(celt_glog));
  memcpy(d->old_log_e, oldLogE, CC * nb * sizeof(celt_glog));
  memcpy(d->old_log_e2, oldLogE2, CC * nb * sizeof(celt_glog));
  memcpy(d->energy_error, energyError, CC * nb * sizeof(celt_glog));
#ifdef ENABLE_QEXT
  memcpy(d->qext_old_band_e, energyError + CC * nb, CC * NB_QEXT_BANDS * sizeof(celt_glog));
#endif
}

/* ---- Range coder snapshots ---- */

typedef struct {
  unsigned int storage, end_offs, end_window;
  int nend_bits, nbits_total;
  unsigned int offs, rng, val, ext;
  int rem, error;
} OracleEcState;

static void ec_snap(const ec_enc *e, OracleEcState *o) {
  o->storage = e->storage;
  o->end_offs = e->end_offs;
  o->end_window = e->end_window;
  o->nend_bits = e->nend_bits;
  o->nbits_total = e->nbits_total;
  o->offs = e->offs;
  o->rng = e->rng;
  o->val = e->val;
  o->ext = e->ext;
  o->rem = e->rem;
  o->error = e->error;
}

/* Applies a sequence of (kind, value, param) symbol writes (simulated SILK data). */
static void apply_ops(ec_enc *enc, const int *ops, int n_ops) {
  int i;
  for (i = 0; i < n_ops; i++) {
    int kind = ops[3 * i], v = ops[3 * i + 1], p = ops[3 * i + 2];
    switch (kind) {
      case 0: ec_enc_bit_logp(enc, v, p); break;
      case 1: ec_enc_uint(enc, (opus_uint32)v, (opus_uint32)p); break;
      case 2: ec_enc_bits(enc, (opus_uint32)v, (unsigned)p); break;
      default: break;
    }
  }
}

/* ------------------------------------------------------------------------------------------ */
/* Part B: private copy of opus_encoder.c with celt_encode_with_ec intercepted                  */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  OracleCeState pre, post;
  int frame_size, nb_compressed_bytes, ret;
  int has_compressed, has_enc;
  opus_res *pcm;
  int pcm_len;
  unsigned char *comp_before, *comp_after;
  int comp_len;
  OracleEcState enc_before, enc_after;
  /* [toc, buf[0..storage)] of the range coder's buffer at entry. */
  unsigned char *buf_before, *buf_after;
  int buf_len;
  int buf_shift;
} OracleCeCall;

static __thread OracleCeCall *ce_calls = NULL;
static __thread int ce_ncalls = 0, ce_cap = 0;
static __thread int ce_recording = 0;

static void rec_free_one(OracleCeCall *c) {
  free(c->pcm);
  free(c->comp_before);
  free(c->comp_after);
  free(c->buf_before);
  free(c->buf_after);
}

void oracle_ce_rec_clear(void) {
  int i;
  for (i = 0; i < ce_ncalls; i++) rec_free_one(&ce_calls[i]);
  ce_ncalls = 0;
}

int oracle_ce_rec_count(void) { return ce_ncalls; }
const OracleCeCall *oracle_ce_rec_get(int i) {
  return (i >= 0 && i < ce_ncalls) ? &ce_calls[i] : NULL;
}
size_t oracle_ce_call_size(void) { return sizeof(OracleCeCall); }

static int oracle_ce_intercept(CELTEncoder *st, const opus_res *pcm, int frame_size,
                               unsigned char *compressed, int nbCompressedBytes, ec_enc *enc) {
  OracleCeCall *c;
  int ret;
  unsigned char *base = NULL;
  if (!ce_recording) return celt_encode_with_ec(st, pcm, frame_size, compressed,
                                                nbCompressedBytes, enc);
  if (ce_ncalls == ce_cap) {
    ce_cap = ce_cap ? 2 * ce_cap : 16;
    ce_calls = (OracleCeCall *)realloc(ce_calls, ce_cap * sizeof(OracleCeCall));
  }
  c = &ce_calls[ce_ncalls++];
  memset(c, 0, sizeof(*c));
  dump_state(st, &c->pre);
  c->frame_size = frame_size;
  c->nb_compressed_bytes = nbCompressedBytes;
  c->pcm_len = st->channels * frame_size;
  c->pcm = (opus_res *)malloc(sizeof(opus_res) * (c->pcm_len > 0 ? c->pcm_len : 1));
  memcpy(c->pcm, pcm, sizeof(opus_res) * c->pcm_len);
  c->has_compressed = compressed != NULL;
  if (compressed) {
    c->comp_len = nbCompressedBytes;
    c->comp_before = (unsigned char *)malloc(nbCompressedBytes);
    c->comp_after = (unsigned char *)malloc(nbCompressedBytes);
    memcpy(c->comp_before, compressed, nbCompressedBytes);
  }
  c->has_enc = enc != NULL;
  if (enc) {
    base = enc->buf - 1;
    ec_snap(enc, &c->enc_before);
    c->buf_len = enc->storage + 1;
    c->buf_before = (unsigned char *)malloc(c->buf_len);
    c->buf_after = (unsigned char *)malloc(c->buf_len);
    memcpy(c->buf_before, base, c->buf_len);
  }
  ret = celt_encode_with_ec(st, pcm, frame_size, compressed, nbCompressedBytes, enc);
  c->ret = ret;
  dump_state(st, &c->post);
  if (compressed) memcpy(c->comp_after, compressed, nbCompressedBytes);
  if (enc) {
    ec_snap(enc, &c->enc_after);
    memcpy(c->buf_after, base, c->buf_len);
    c->buf_shift = (int)(enc->buf - (base + 1));
  }
  return ret;
}

#define opus_encoder_get_size oracle_ce_dup_opus_encoder_get_size
#define opus_encoder_init oracle_ce_dup_opus_encoder_init
#define opus_encoder_create oracle_ce_dup_opus_encoder_create
#define downmix_float oracle_ce_dup_downmix_float
#define downmix_int oracle_ce_dup_downmix_int
#define downmix_int24 oracle_ce_dup_downmix_int24
#define frame_size_select oracle_ce_dup_frame_size_select
#define compute_stereo_width oracle_ce_dup_compute_stereo_width
#define is_digital_silence oracle_ce_dup_is_digital_silence
#define opus_encode_native oracle_ce_dup_opus_encode_native
#define opus_encode oracle_ce_dup_opus_encode
#define opus_encode24 oracle_ce_dup_opus_encode24
#define opus_encode_float oracle_ce_dup_opus_encode_float
#define opus_encoder_ctl oracle_ce_dup_opus_encoder_ctl
#define opus_encoder_destroy oracle_ce_dup_opus_encoder_destroy
#define silk_biquad_res oracle_ce_dup_silk_biquad_res
#define celt_encode_with_ec oracle_ce_intercept
#include "../../../vendor/libopus/src/opus_encoder.c"
#undef celt_encode_with_ec
#undef opus_encoder_get_size
#undef opus_encoder_init
#undef opus_encoder_create
#undef downmix_float
#undef downmix_int
#undef downmix_int24
#undef frame_size_select
#undef compute_stereo_width
#undef is_digital_silence
#undef opus_encode_native
#undef opus_encode
#undef opus_encode24
#undef opus_encode_float
#undef opus_encoder_ctl
#undef opus_encoder_destroy
#undef silk_biquad_res

OpusEncoder *oracle_ce_opus_create(int fs, int channels, int application, int *err) {
  return oracle_ce_dup_opus_encoder_create(fs, channels, application, err);
}
void oracle_ce_opus_destroy(OpusEncoder *st) { oracle_ce_dup_opus_encoder_destroy(st); }
int oracle_ce_opus_ctl(OpusEncoder *st, int request, int value) {
  return oracle_ce_dup_opus_encoder_ctl(st, request, (opus_int32)value);
}
/* Encodes one frame of 16-bit PCM, recording every CELT call (records are cleared first). */
int oracle_ce_opus_encode(OpusEncoder *st, const opus_int16 *pcm, int frame_size,
                          unsigned char *out, int max_bytes) {
  int ret;
  oracle_ce_rec_clear();
  ce_recording = 1;
  ret = oracle_ce_dup_opus_encode(st, pcm, frame_size, out, max_bytes);
  ce_recording = 0;
  return ret;
}
#ifndef DISABLE_FLOAT_API
/* Encodes one frame, recording every CELT call (records are cleared first). */
int oracle_ce_opus_encode_float(OpusEncoder *st, const float *pcm, int frame_size,
                                unsigned char *out, int max_bytes) {
  int ret;
  oracle_ce_rec_clear();
  ce_recording = 1;
  ret = oracle_ce_dup_opus_encode_float(st, pcm, frame_size, out, max_bytes);
  ce_recording = 0;
  return ret;
}
#endif

/* ------------------------------------------------------------------------------------------ */
/* Part C: direct API over a library CELT encoder                                               */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  CELTEncoder *st;
  CELTMode *custom_mode;
  celt_glog mask[2 * CE_MAX_BANDS];
} CeHandle;

CeHandle *oracle_ce_new(int fs, int channels, int *err) {
  CeHandle *h = (CeHandle *)calloc(1, sizeof(CeHandle));
  h->st = (CELTEncoder *)malloc(celt_encoder_get_size(channels));
  *err = celt_encoder_init(h->st, fs, channels, 0);
  if (*err != OPUS_OK) {
    free(h->st);
    free(h);
    return NULL;
  }
  opus_custom_encoder_ctl(h->st, CELT_SET_SIGNALLING(0));
  return h;
}

CeHandle *oracle_ce_custom_new(int fs, int frame_size, int channels, int *err) {
#ifdef CUSTOM_MODES
  CeHandle *h = (CeHandle *)calloc(1, sizeof(CeHandle));
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
  return h;
#else
  (void)fs;
  (void)frame_size;
  (void)channels;
  *err = OPUS_UNIMPLEMENTED;
  return NULL;
#endif
}

void oracle_ce_free(CeHandle *h) {
  if (!h) return;
#ifdef CUSTOM_MODES
  if (h->custom_mode) {
    opus_custom_encoder_destroy(h->st);
    opus_custom_mode_destroy(h->custom_mode);
    free(h);
    return;
  }
#endif
  free(h->st);
  free(h);
}

int oracle_ce_ctl(CeHandle *h, int request, int value) {
  return opus_custom_encoder_ctl(h->st, request, (opus_int32)value);
}

int oracle_ce_ctl_get(CeHandle *h, int request, int *value) {
  opus_int32 v = 0;
  int ret;
  if (request == OPUS_GET_FINAL_RANGE_REQUEST) {
    opus_uint32 r = 0;
    ret = opus_custom_encoder_ctl(h->st, request, &r);
    *value = (int)r;
    return ret;
  }
  ret = opus_custom_encoder_ctl(h->st, request, &v);
  *value = v;
  return ret;
}

void oracle_ce_set_analysis(CeHandle *h, int valid, const float *f /* 9 floats */, int bandwidth,
                            const unsigned char *leak_boost) {
  AnalysisInfo a;
  memset(&a, 0, sizeof(a));
  a.valid = valid;
  a.tonality = f[0];
  a.tonality_slope = f[1];
  a.noisiness = f[2];
  a.activity = f[3];
  a.music_prob = f[4];
  a.music_prob_min = f[5];
  a.music_prob_max = f[6];
  a.bandwidth = bandwidth;
  a.activity_probability = f[7];
  a.max_pitch_ratio = f[8];
  memcpy(a.leak_boost, leak_boost, LEAK_BANDS);
  opus_custom_encoder_ctl(h->st, CELT_SET_ANALYSIS(&a));
}

void oracle_ce_set_silk_info(CeHandle *h, int signal_type, int offset) {
  SILKInfo s;
  s.signalType = signal_type;
  s.offset = offset;
  opus_custom_encoder_ctl(h->st, CELT_SET_SILK_INFO(&s));
}

void oracle_ce_set_energy_mask(CeHandle *h, const celt_glog *mask, int n) {
  if (!mask) {
    opus_custom_encoder_ctl(h->st, OPUS_SET_ENERGY_MASK((celt_glog *)NULL));
    return;
  }
  memset(h->mask, 0, sizeof(h->mask));
  memcpy(h->mask, mask, n * sizeof(celt_glog));
  opus_custom_encoder_ctl(h->st, OPUS_SET_ENERGY_MASK(h->mask));
}

void oracle_ce_get_state(const CeHandle *h, OracleCeState *out) { dump_state(h->st, out); }

int oracle_ce_encode(CeHandle *h, const opus_res *pcm, int frame_size, unsigned char *out,
                     int nb) {
  return celt_encode_with_ec(h->st, pcm, frame_size, out, nb, NULL);
}

/* Hybrid-style call: buf[0] is the TOC byte, the range coder covers buf[1..buf_size). The
 * (kind, value, param) ops simulate SILK data, then the coder is shrunk to `nb` bytes (when
 * smaller) and CELT encodes with the shared coder. */
int oracle_ce_encode_with_ec(CeHandle *h, const opus_res *pcm, int frame_size, unsigned char *buf,
                             int buf_size, int nb, const int *ops, int n_ops,
                             OracleEcState *out_enc, int *buf_shift) {
  ec_enc enc;
  int ret;
  ec_enc_init(&enc, buf + 1, buf_size - 1);
  apply_ops(&enc, ops, n_ops);
  if (nb < buf_size - 1) ec_enc_shrink(&enc, nb);
  ret = celt_encode_with_ec(h->st, pcm, frame_size, NULL, nb, &enc);
  ec_snap(&enc, out_enc);
  *buf_shift = (int)(enc.buf - (buf + 1));
  return ret;
}

#ifdef CUSTOM_MODES
int oracle_ce_custom_encode(CeHandle *h, const opus_int16 *pcm, int frame_size,
                            unsigned char *out, int nb) {
  return opus_custom_encode(h->st, pcm, frame_size, out, nb);
}
int oracle_ce_custom_encode24(CeHandle *h, const opus_int32 *pcm, int frame_size,
                              unsigned char *out, int nb) {
  return opus_custom_encode24(h->st, pcm, frame_size, out, nb);
}
#ifndef DISABLE_FLOAT_API
int oracle_ce_custom_encode_float(CeHandle *h, const float *pcm, int frame_size,
                                  unsigned char *out, int nb) {
  return opus_custom_encode_float(h->st, pcm, frame_size, out, nb);
}
#endif
#else
int oracle_ce_custom_encode(CeHandle *h, const opus_int16 *pcm, int frame_size,
                            unsigned char *out, int nb) {
  (void)h; (void)pcm; (void)frame_size; (void)out; (void)nb;
  return OPUS_UNIMPLEMENTED;
}
int oracle_ce_custom_encode24(CeHandle *h, const opus_int32 *pcm, int frame_size,
                              unsigned char *out, int nb) {
  (void)h; (void)pcm; (void)frame_size; (void)out; (void)nb;
  return OPUS_UNIMPLEMENTED;
}
int oracle_ce_custom_encode_float(CeHandle *h, const float *pcm, int frame_size,
                                  unsigned char *out, int nb) {
  (void)h; (void)pcm; (void)frame_size; (void)out; (void)nb;
  return OPUS_UNIMPLEMENTED;
}
#endif

/* ---- Static helper shims (private copy) ---- */

static const CELTMode *ce_mode(int fs) {
  return opus_custom_mode_create(fs, fs / 50, NULL);
}

int oracle_ce_transient_analysis(const opus_val32 *in, int len, int C, opus_val16 *tf_estimate,
                                 int *tf_chan, int allow_weak, int *weak, opus_val16 tone_freq,
                                 opus_val32 toneishness) {
  return transient_analysis(in, len, C, tf_estimate, tf_chan, allow_weak, weak, tone_freq,
                            toneishness);
}

int oracle_ce_patch_transient_decision(celt_glog *newE, celt_glog *oldE, int nbEBands, int start,
                                       int end, int C) {
  return patch_transient_decision(newE, oldE, nbEBands, start, end, C);
}

void oracle_ce_compute_mdcts(int fs, int shortBlocks, celt_sig *in, celt_sig *out, int C, int CC,
                             int LM, int upsample) {
  compute_mdcts(ce_mode(fs), shortBlocks, in, out, C, CC, LM, upsample, 0);
}

void oracle_ce_preemphasis(const opus_res *pcm, celt_sig *inp, int N, int CC, int upsample,
                           const opus_val16 *coef, celt_sig *mem, int clip) {
  celt_preemphasis(pcm, inp, N, CC, upsample, coef, mem, clip);
}

int oracle_ce_tf_analysis(int fs, int len, int isTransient, int *tf_res, int lambda,
                          celt_norm *X, int N0, int LM, opus_val16 tf_estimate, int tf_chan,
                          int *importance) {
  return tf_analysis(ce_mode(fs), len, isTransient, tf_res, lambda, X, N0, LM, tf_estimate,
                     tf_chan, importance);
}

void oracle_ce_tf_encode(int start, int end, int isTransient, int *tf_res, int LM,
                         int tf_select, unsigned char *buf, int size, const int *ops, int n_ops,
                         OracleEcState *out) {
  ec_enc enc;
  ec_enc_init(&enc, buf, size);
  apply_ops(&enc, ops, n_ops);
  tf_encode(start, end, isTransient, tf_res, LM, tf_select, &enc);
  ec_enc_done(&enc);
  ec_snap(&enc, out);
}

int oracle_ce_alloc_trim_analysis(int fs, const celt_norm *X, const celt_glog *bandLogE, int end,
                                  int LM, int C, int N0, int an_valid, float tonality_slope,
                                  opus_val16 *stereo_saving, opus_val16 tf_estimate,
                                  int intensity, celt_glog surround_trim, int equiv_rate) {
  AnalysisInfo a;
  memset(&a, 0, sizeof(a));
  a.valid = an_valid;
  a.tonality_slope = tonality_slope;
  return alloc_trim_analysis(ce_mode(fs), X, bandLogE, end, LM, C, N0, &a, stereo_saving,
                             tf_estimate, intensity, surround_trim, equiv_rate, 0);
}

int oracle_ce_stereo_analysis(int fs, const celt_norm *X, int LM, int N0) {
  return stereo_analysis(ce_mode(fs), X, LM, N0);
}

celt_glog oracle_ce_median_of_5(const celt_glog *x) { return median_of_5(x); }
celt_glog oracle_ce_median_of_3(const celt_glog *x) { return median_of_3(x); }

celt_glog oracle_ce_dynalloc_analysis(int fs, const celt_glog *bandLogE,
                                      const celt_glog *bandLogE2, const celt_glog *oldBandE,
                                      int start, int end, int C, int *offsets, int lsb_depth,
                                      int isTransient, int vbr, int constrained_vbr, int LM,
                                      int effectiveBytes, int *tot_boost, int lfe,
                                      celt_glog *surround_dynalloc, int an_valid,
                                      const unsigned char *leak_boost, int *importance,
                                      int *spread_weight, opus_val16 tone_freq,
                                      opus_val32 toneishness) {
  const CELTMode *m = ce_mode(fs);
  AnalysisInfo a;
  memset(&a, 0, sizeof(a));
  a.valid = an_valid;
  memcpy(a.leak_boost, leak_boost, LEAK_BANDS);
  return dynalloc_analysis(bandLogE, bandLogE2, oldBandE, m->nbEBands, start, end, C, offsets,
                           lsb_depth, m->logN, isTransient, vbr, constrained_vbr, m->eBands, LM,
                           effectiveBytes, tot_boost, lfe, surround_dynalloc, &a, importance,
                           spread_weight, tone_freq, toneishness
                           ARG_QEXT(fs == 96000 ? 2 : 1));
}

int oracle_ce_tone_lpc(const opus_val16 *x, int len, int delay, opus_val32 *lpc) {
  return tone_lpc(x, len, delay, lpc);
}

opus_val16 oracle_ce_tone_detect(const celt_sig *in, int CC, int N, opus_val32 *toneishness,
                                 int Fs) {
  return tone_detect(in, CC, N, toneishness, Fs);
}

#ifdef FIXED_POINT
void oracle_ce_normalize_tone_input(opus_val16 *x, int len) {
  oracle_ce_dup_normalize_tone_input(x, len);
}
int oracle_ce_acos_approx(opus_val32 x) { return oracle_ce_dup_acos_approx(x); }
#endif

/* run_prefilter on a scratch encoder state: the st-> fields it reads/writes are passed flat. */
int oracle_ce_run_prefilter(int fs, int CC, int N, celt_sig *in, celt_sig *prefilter_mem,
                            celt_sig *in_mem, int *prefilter_period, opus_val16 prefilter_gain,
                            int prefilter_tapset, int loss_rate, int tapset, int *pitch,
                            opus_val16 *gain, int *qgain, int enabled, int complexity,
                            opus_val16 tf_estimate, int nbAvailableBytes, int an_valid,
                            float max_pitch_ratio, opus_val16 tone_freq,
                            opus_val32 toneishness) {
  const CELTMode *m = ce_mode(fs);
  int qext_scale = fs == 96000 ? 2 : 1;
  int ov = m->overlap;
  int ret;
  AnalysisInfo a;
  CELTEncoder *st = (CELTEncoder *)calloc(1, oracle_ce_dup_opus_custom_encoder_get_size(m, CC));
  memset(&a, 0, sizeof(a));
  a.valid = an_valid;
  a.max_pitch_ratio = max_pitch_ratio;
  st->mode = m;
  st->channels = CC;
  st->loss_rate = loss_rate;
  st->prefilter_period = *prefilter_period;
  st->prefilter_gain = prefilter_gain;
  st->prefilter_tapset = prefilter_tapset;
#ifdef ENABLE_QEXT
  st->qext_scale = qext_scale;
#else
  (void)qext_scale;
#endif
  memcpy(st->in_mem, in_mem, CC * ov * sizeof(celt_sig));
  ret = run_prefilter(st, in, prefilter_mem, CC, N, tapset, pitch, gain, qgain, enabled,
                      complexity, tf_estimate, nbAvailableBytes, &a, tone_freq, toneishness
                      ARG_QEXT(qext_scale));
  memcpy(in_mem, st->in_mem, CC * ov * sizeof(celt_sig));
  *prefilter_period = st->prefilter_period;
  free(st);
  return ret;
}

int oracle_ce_compute_vbr(int fs, int an_valid, float activity, float tonality, int base_target,
                          int LM, int bitrate, int lastCodedBands, int C, int intensity,
                          int constrained_vbr, opus_val16 stereo_saving, int tot_boost,
                          opus_val16 tf_estimate, int pitch_change, celt_glog maxDepth, int lfe,
                          int has_surround_mask, celt_glog surround_masking,
                          celt_glog temporal_vbr, int enable_qext) {
  AnalysisInfo a;
  memset(&a, 0, sizeof(a));
  a.valid = an_valid;
  a.activity = activity;
  a.tonality = tonality;
#ifndef ENABLE_QEXT
  (void)enable_qext;
#endif
  return compute_vbr(ce_mode(fs), &a, base_target, LM, bitrate, lastCodedBands, C, intensity,
                     constrained_vbr, stereo_saving, tot_boost, tf_estimate, pitch_change,
                     maxDepth, lfe, has_surround_mask, surround_masking, temporal_vbr
                     ARG_QEXT(enable_qext));
}

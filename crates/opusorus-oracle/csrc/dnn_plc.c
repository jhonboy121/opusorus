/* Oracle C shims for unit dnn_plc: dnn/fargan.c and dnn/lpcnet_plc.c (FARGAN vocoder and the
   neural PLC), float build, generic C path.

   Only compiled into DNN oracle builds (build.rs defines ENABLE_DEEP_PLC for the deep-plc, dred
   and osce features). The public functions are called in the library; the static helpers
   (compute_fargan_cond, fargan_deemphasis, run_fargan_subframe, compute_plc_pred,
   get_fec_or_pred, queue_features) are reached through private copies of fargan.c and
   lpcnet_plc.c whose exported symbols are renamed (pc_copy_*). States live on the C heap and
   are initialised with the compiled-in model tables. */
#ifdef ENABLE_DEEP_PLC

#include <stdlib.h>
#include <string.h>

#include "nnet.h"
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "fargan.h"

/* ---- private copies (statics) ---- */
#define fargan_init pc_copy_fargan_init
#define fargan_load_model pc_copy_fargan_load_model
#define fargan_cont pc_copy_fargan_cont
#define fargan_synthesize pc_copy_fargan_synthesize
#define fargan_synthesize_int pc_copy_fargan_synthesize_int
#include "fargan.c"
#undef fargan_init
#undef fargan_load_model
#undef fargan_cont
#undef fargan_synthesize
#undef fargan_synthesize_int

#define lpcnet_plc_reset pc_copy_lpcnet_plc_reset
#define lpcnet_plc_init pc_copy_lpcnet_plc_init
#define lpcnet_plc_load_model pc_copy_lpcnet_plc_load_model
#define lpcnet_plc_fec_add pc_copy_lpcnet_plc_fec_add
#define lpcnet_plc_fec_clear pc_copy_lpcnet_plc_fec_clear
#define lpcnet_plc_update pc_copy_lpcnet_plc_update
#define lpcnet_plc_conceal pc_copy_lpcnet_plc_conceal
#include "lpcnet_plc.c"
#undef lpcnet_plc_reset
#undef lpcnet_plc_init
#undef lpcnet_plc_load_model
#undef lpcnet_plc_fec_add
#undef lpcnet_plc_fec_clear
#undef lpcnet_plc_update
#undef lpcnet_plc_conceal

/* Library prototypes (fargan.h / lpcnet.h were seen with the library names, but redeclare to
   be explicit). */
void fargan_init(FARGANState *st);
int fargan_load_model(FARGANState *st, const void *data, int len);
void fargan_cont(FARGANState *st, const float *pcm0, const float *features0);
void fargan_synthesize(FARGANState *st, float *pcm, const float *features);
void fargan_synthesize_int(FARGANState *st, opus_int16 *pcm, const float *features);
int lpcnet_plc_init(LPCNetPLCState *st);
void lpcnet_plc_reset(LPCNetPLCState *st);
int lpcnet_plc_update(LPCNetPLCState *st, opus_int16 *pcm);
int lpcnet_plc_conceal(LPCNetPLCState *st, opus_int16 *pcm);
void lpcnet_plc_fec_add(LPCNetPLCState *st, const float *features);
void lpcnet_plc_fec_clear(LPCNetPLCState *st);
int lpcnet_plc_load_model(LPCNetPLCState *st, const void *data, int len);

/* ---- constants for the Rust side ---- */
int oracle_pc_const(int which) {
  switch (which) {
  case 0: return FARGAN_COND_SIZE;
  case 1: return COND_NET_FDENSE2_OUT_SIZE;
  case 2: return SIG_NET_INPUT_SIZE;
  case 3: return PLC_BUF_SIZE;
  case 4: return PLC_MAX_FEC;
  case 5: return (int)sizeof(FARGANState);
  case 6: return (int)sizeof(LPCNetPLCState);
  default: return -1;
  }
}

/* ---- FARGAN ---- */
#define PC_PUT(field)                                                        \
  do {                                                                       \
    memcpy(&out[pos], &(field), sizeof(field));                              \
    pos += (int)(sizeof(field) / sizeof(float));                             \
  } while (0)
#define PC_GET(field)                                                        \
  do {                                                                       \
    memcpy(&(field), &in[pos], sizeof(field));                               \
    pos += (int)(sizeof(field) / sizeof(float));                             \
  } while (0)

static int pc_fargan_floats(const FARGANState *st, float *out) {
  int pos = 0;
  PC_PUT(st->deemph_mem);
  PC_PUT(st->pitch_buf);
  PC_PUT(st->cond_conv1_state);
  PC_PUT(st->fwc0_mem);
  PC_PUT(st->gru1_state);
  PC_PUT(st->gru2_state);
  PC_PUT(st->gru3_state);
  return pos;
}

void *oracle_pc_fargan_new(void) {
  FARGANState *st = malloc(sizeof(*st));
  fargan_init(st);
  return st;
}
void oracle_pc_fargan_free(void *p) { free(p); }
void oracle_pc_fargan_init(void *p) { fargan_init((FARGANState *)p); }
int oracle_pc_fargan_load_model(void *p, const unsigned char *data, int len) {
  return fargan_load_model((FARGANState *)p, data, len);
}
void oracle_pc_fargan_cont(void *p, const float *pcm0, const float *features0) {
  fargan_cont((FARGANState *)p, pcm0, features0);
}
void oracle_pc_fargan_synthesize(void *p, float *pcm, const float *features) {
  fargan_synthesize((FARGANState *)p, pcm, features);
}
void oracle_pc_fargan_synthesize_int(void *p, short *pcm, const float *features) {
  fargan_synthesize_int((FARGANState *)p, pcm, features);
}
int oracle_pc_fargan_state(void *p, float *out, int *ints) {
  FARGANState *st = p;
  ints[0] = st->cont_initialized;
  ints[1] = st->last_period;
  return pc_fargan_floats(st, out);
}
int oracle_pc_fargan_set_state(void *p, const float *in, const int *ints) {
  FARGANState *st = p;
  int pos = 0;
  st->cont_initialized = ints[0];
  st->last_period = ints[1];
  PC_GET(st->deemph_mem);
  PC_GET(st->pitch_buf);
  PC_GET(st->cond_conv1_state);
  PC_GET(st->fwc0_mem);
  PC_GET(st->gru1_state);
  PC_GET(st->gru2_state);
  PC_GET(st->gru3_state);
  return pos;
}
void oracle_pc_compute_fargan_cond(void *p, float *cond, const float *features, int period) {
  compute_fargan_cond((FARGANState *)p, cond, features, period);
}
void oracle_pc_fargan_deemphasis(float *pcm, float *mem) { fargan_deemphasis(pcm, mem); }
void oracle_pc_run_fargan_subframe(void *p, float *pcm, const float *cond, int period) {
  run_fargan_subframe((FARGANState *)p, pcm, cond, period);
}

/* ---- LPCNet PLC ---- */
void *oracle_pc_plc_new(void) {
  LPCNetPLCState *st = malloc(sizeof(*st));
  lpcnet_plc_init(st);
  return st;
}
void oracle_pc_plc_free(void *p) { free(p); }
void oracle_pc_plc_reset(void *p) { lpcnet_plc_reset((LPCNetPLCState *)p); }
int oracle_pc_plc_load_model(void *p, const unsigned char *data, int len) {
  return lpcnet_plc_load_model((LPCNetPLCState *)p, data, len);
}
int oracle_pc_plc_update(void *p, short *pcm) {
  return lpcnet_plc_update((LPCNetPLCState *)p, pcm);
}
int oracle_pc_plc_conceal(void *p, short *pcm) {
  return lpcnet_plc_conceal((LPCNetPLCState *)p, pcm);
}
void oracle_pc_plc_fec_add(void *p, const float *features) {
  lpcnet_plc_fec_add((LPCNetPLCState *)p, features);
}
void oracle_pc_plc_fec_clear(void *p) { lpcnet_plc_fec_clear((LPCNetPLCState *)p); }
void oracle_pc_compute_plc_pred(void *p, float *out, const float *in) {
  compute_plc_pred((LPCNetPLCState *)p, out, in);
}
int oracle_pc_get_fec_or_pred(void *p, float *out) {
  return get_fec_or_pred((LPCNetPLCState *)p, out);
}
void oracle_pc_queue_features(void *p, const float *features) {
  queue_features((LPCNetPLCState *)p, features);
}

/* Floats: fec, pcm, features, cont_features, plc_net, plc_bak[0..2], fargan (as
   oracle_pc_fargan_state), enc (the dnn_core LPCNetEncState order). Ints: loaded,
   analysis_gap, fec_read_pos, fec_fill_pos, fec_skip, analysis_pos, predict_pos, blend,
   loss_count, fargan.cont_initialized, fargan.last_period. */
int oracle_pc_plc_state(void *p, float *out, int *ints) {
  LPCNetPLCState *st = p;
  int pos = 0;
  PC_PUT(st->fec);
  PC_PUT(st->pcm);
  PC_PUT(st->features);
  PC_PUT(st->cont_features);
  PC_PUT(st->plc_net);
  PC_PUT(st->plc_bak);
  pos += pc_fargan_floats(&st->fargan, &out[pos]);
  PC_PUT(st->enc.analysis_mem);
  PC_PUT(st->enc.mem_preemph);
  PC_PUT(st->enc.prev_if);
  PC_PUT(st->enc.if_features);
  PC_PUT(st->enc.xcorr_features);
  PC_PUT(st->enc.dnn_pitch);
  PC_PUT(st->enc.pitch_mem);
  PC_PUT(st->enc.pitch_filt);
  PC_PUT(st->enc.exc_buf);
  PC_PUT(st->enc.lp_buf);
  PC_PUT(st->enc.lp_mem);
  PC_PUT(st->enc.lpc);
  PC_PUT(st->enc.features);
  PC_PUT(st->enc.sig_mem);
  PC_PUT(st->enc.burg_cepstrum);
  PC_PUT(st->enc.pitchdnn.gru_state);
  PC_PUT(st->enc.pitchdnn.xcorr_mem1);
  PC_PUT(st->enc.pitchdnn.xcorr_mem2);
  ints[0] = st->loaded;
  ints[1] = st->analysis_gap;
  ints[2] = st->fec_read_pos;
  ints[3] = st->fec_fill_pos;
  ints[4] = st->fec_skip;
  ints[5] = st->analysis_pos;
  ints[6] = st->predict_pos;
  ints[7] = st->blend;
  ints[8] = st->loss_count;
  ints[9] = st->fargan.cont_initialized;
  ints[10] = st->fargan.last_period;
  return pos;
}

#endif /* ENABLE_DEEP_PLC */

/* Oracle C shims for unit silk_resampler (silk/resampler*.c). */
// oracle-build: any
#include <stdlib.h>
#include <string.h>
#include "opus_types.h"
#include "SigProc_FIX.h"
#include "resampler_private.h"

/* Reach the static inline interpolators by compiling private copies of their translation units
   with the exported function renamed (avoids duplicate symbols with the library). */
#define silk_resampler_private_IIR_FIR oracle_dup_silk_resampler_private_IIR_FIR
#include "resampler_private_IIR_FIR.c"
#undef silk_resampler_private_IIR_FIR
#define silk_resampler_private_down_FIR oracle_dup_silk_resampler_private_down_FIR
#include "resampler_private_down_FIR.c"
#undef silk_resampler_private_down_FIR

/* ---- Resampler state (opaque, heap allocated) ---- */

void *oracle_silk_resampler_new(void) {
  silk_resampler_state_struct *s = (silk_resampler_state_struct *)malloc(sizeof(*s));
  if (s) memset(s, 0, sizeof(*s));
  return s;
}

void oracle_silk_resampler_free(void *s) { free(s); }

int oracle_silk_resampler_init(void *s, int fs_in, int fs_out, int for_enc) {
  return silk_resampler_init((silk_resampler_state_struct *)s, fs_in, fs_out, for_enc);
}

int oracle_silk_resampler(void *s, opus_int16 *out, const opus_int16 *in, int in_len) {
  return silk_resampler((silk_resampler_state_struct *)s, out, in, in_len);
}

void oracle_silk_resampler_private_IIR_FIR(void *s, opus_int16 *out, const opus_int16 *in,
                                           int in_len) {
  silk_resampler_private_IIR_FIR(s, out, in, in_len);
}

void oracle_silk_resampler_private_down_FIR(void *s, opus_int16 *out, const opus_int16 *in,
                                            int in_len) {
  silk_resampler_private_down_FIR(s, out, in, in_len);
}

void oracle_silk_resampler_private_up2_HQ_wrapper(void *s, opus_int16 *out, const opus_int16 *in,
                                                  int len) {
  silk_resampler_private_up2_HQ_wrapper(s, out, in, len);
}

static int coefs_id(const opus_int16 *c) {
  if (c == NULL) return 0;
  if (c == silk_Resampler_3_4_COEFS) return 1;
  if (c == silk_Resampler_2_3_COEFS) return 2;
  if (c == silk_Resampler_1_2_COEFS) return 3;
  if (c == silk_Resampler_1_3_COEFS) return 4;
  if (c == silk_Resampler_1_4_COEFS) return 5;
  if (c == silk_Resampler_1_6_COEFS) return 6;
  return -1;
}

/* Dumps the whole state. ints[9] = { resampler_function, batchSize, invRatio_Q16, FIR_Order,
   FIR_Fracs, Fs_in_kHz, Fs_out_kHz, inputDelay, coefs id (0 = NULL, 1..6 as coefs_id) }. */
void oracle_silk_resampler_get(const void *sp, opus_int32 *s_iir, opus_int32 *s_fir_i32,
                               opus_int16 *s_fir_i16, opus_int16 *delay_buf, int *ints) {
  const silk_resampler_state_struct *s = (const silk_resampler_state_struct *)sp;
  memcpy(s_iir, s->sIIR, sizeof(s->sIIR));
  memcpy(s_fir_i32, s->sFIR.i32, sizeof(s->sFIR.i32));
  memcpy(s_fir_i16, s->sFIR.i16, sizeof(s->sFIR.i16));
  memcpy(delay_buf, s->delayBuf, sizeof(s->delayBuf));
  ints[0] = s->resampler_function;
  ints[1] = s->batchSize;
  ints[2] = s->invRatio_Q16;
  ints[3] = s->FIR_Order;
  ints[4] = s->FIR_Fracs;
  ints[5] = s->Fs_in_kHz;
  ints[6] = s->Fs_out_kHz;
  ints[7] = s->inputDelay;
  ints[8] = coefs_id(s->Coefs);
}

/* Overwrites the filter memories (sIIR, one sFIR view, delayBuf) of an initialized state.
   use_i16 selects which sFIR union view is written. */
void oracle_silk_resampler_set_mem(void *sp, const opus_int32 *s_iir, const opus_int32 *s_fir_i32,
                                   const opus_int16 *s_fir_i16, int use_i16,
                                   const opus_int16 *delay_buf) {
  silk_resampler_state_struct *s = (silk_resampler_state_struct *)sp;
  memcpy(s->sIIR, s_iir, sizeof(s->sIIR));
  if (use_i16) {
    memcpy(s->sFIR.i16, s_fir_i16, sizeof(s->sFIR.i16));
  } else {
    memcpy(s->sFIR.i32, s_fir_i32, sizeof(s->sFIR.i32));
  }
  memcpy(s->delayBuf, delay_buf, sizeof(s->delayBuf));
}

/* ---- Standalone filters ---- */

void oracle_silk_resampler_down2(opus_int32 *s, opus_int16 *out, const opus_int16 *in, int len) {
  silk_resampler_down2(s, out, in, len);
}

void oracle_silk_resampler_down2_3(opus_int32 *s, opus_int16 *out, const opus_int16 *in, int len) {
  silk_resampler_down2_3(s, out, in, len);
}

void oracle_silk_resampler_private_AR2(opus_int32 *s, opus_int32 *out_q8, const opus_int16 *in,
                                       const opus_int16 *a_q14, int len) {
  silk_resampler_private_AR2(s, out_q8, in, a_q14, len);
}

void oracle_silk_resampler_private_up2_HQ(opus_int32 *s, opus_int16 *out, const opus_int16 *in,
                                          int len) {
  silk_resampler_private_up2_HQ(s, out, in, len);
}

int oracle_silk_resampler_private_IIR_FIR_INTERPOL(opus_int16 *out, opus_int16 *buf,
                                                   int max_index_q16, int index_increment_q16) {
  opus_int16 *end =
      silk_resampler_private_IIR_FIR_INTERPOL(out, buf, max_index_q16, index_increment_q16);
  return (int)(end - out);
}

static const opus_int16 *coefs_by_id(int id) {
  switch (id) {
    case 1: return silk_Resampler_3_4_COEFS;
    case 2: return silk_Resampler_2_3_COEFS;
    case 3: return silk_Resampler_1_2_COEFS;
    case 4: return silk_Resampler_1_3_COEFS;
    case 5: return silk_Resampler_1_4_COEFS;
    case 6: return silk_Resampler_1_6_COEFS;
    default: return NULL;
  }
}

/* FIR coefficients are taken from table `coefs_id` (skipping the 2 IIR coefficients, as
   silk_resampler_private_down_FIR does). */
int oracle_silk_resampler_private_down_FIR_INTERPOL(opus_int16 *out, opus_int32 *buf, int coefs_id,
                                                    int fir_order, int fir_fracs,
                                                    int max_index_q16, int index_increment_q16) {
  const opus_int16 *c = coefs_by_id(coefs_id);
  opus_int16 *end = silk_resampler_private_down_FIR_INTERPOL(
      out, buf, &c[2], fir_order, fir_fracs, max_index_q16, index_increment_q16);
  return (int)(end - out);
}

/* ---- ROM tables ---- */

/* Copies every resampler ROM table, concatenated in this order:
   down2_0, down2_1, up2_hq_0[3], up2_hq_1[3], 3_4[29], 2_3[20], 1_2[14], 1_3[20], 1_4[20],
   1_6[20], 2_3_LQ[6], frac_FIR_12[12][4]. Returns the number of values written. */
int oracle_silk_resampler_rom(opus_int16 *out) {
  int n = 0;
  out[n++] = silk_resampler_down2_0;
  out[n++] = silk_resampler_down2_1;
  memcpy(&out[n], silk_resampler_up2_hq_0, sizeof(silk_resampler_up2_hq_0));
  n += 3;
  memcpy(&out[n], silk_resampler_up2_hq_1, sizeof(silk_resampler_up2_hq_1));
  n += 3;
#define CP(t, len)                              \
  memcpy(&out[n], t, (len) * sizeof(opus_int16)); \
  n += (len);
  CP(silk_Resampler_3_4_COEFS, 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2)
  CP(silk_Resampler_2_3_COEFS, 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2)
  CP(silk_Resampler_1_2_COEFS, 2 + RESAMPLER_DOWN_ORDER_FIR1 / 2)
  CP(silk_Resampler_1_3_COEFS, 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2)
  CP(silk_Resampler_1_4_COEFS, 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2)
  CP(silk_Resampler_1_6_COEFS, 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2)
  CP(silk_Resampler_2_3_COEFS_LQ, 2 + 2 * 2)
  CP(silk_resampler_frac_FIR_12, 12 * RESAMPLER_ORDER_FIR_12 / 2)
#undef CP
  return n;
}

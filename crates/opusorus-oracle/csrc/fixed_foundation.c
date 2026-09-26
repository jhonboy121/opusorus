/* Oracle shims for unit fixed_foundation: the fixed-point arithmetic macros of celt/arch.h and
   celt/fixed_generic.h, the fixed-point mathops (celt/mathops.h, mathops.c), the float API
   conversions of celt/float_cast.h, the fixed-point static modes, CWRS and Laplace coding. */
// oracle-build: fixed
#include <string.h>
/* fast_atan2f is only declared for analysis.c in fixed-point builds. */
#define ANALYSIS_C
#include "arch.h"
#include "mathops.h"
#include "float_cast.h"
#include "modes.h"
#include "cwrs.h"
#include "laplace.h"
#include "entenc.h"
#include "entdec.h"
#include "opus_custom.h"

#ifdef FIXED_DEBUG
/* celt/fixed_debug.h has no SHL and no MULT16_16_Q11 (libopus code does not use them): their
   fixed_generic.h expansions over the checking macros. */
#define SHL(a,shift) SHL32(a,shift)
#define MULT16_16_Q11(a,b) (SHR(MULT16_16((a),(b)),11))
#endif

/* Defined in celt/mathops.c but not declared in a header. */
opus_val16 celt_rcp_norm16(opus_val16 x);

/* ---- Macros, by operation id (keep in sync with crates/opusorus-oracle/src/fixed_foundation.rs).
   Arguments are ints, as the macros see them after the usual promotions. ---- */

int ofx_op1(int op, int a) {
  switch (op) {
    case 0: return NEG16(a);
    case 1: return NEG32(a);
    case 2: return NEG32_ovflw(a);
    case 3: return EXTRACT16(a);
    case 4: return EXTEND32(a);
    case 5: return SATURATE16(a);
    case 6: return HALF16(a);
    case 7: return HALF32(a);
    case 8: return ABS16(a);
    case 9: return ABS32(a);
    case 10: return SAT16(a);
    case 11: return SIG2WORD16(a);
    case 12: return celt_isnan(a);
    case 13: return COEF2VAL16((celt_coef)a);
    /* opus_res / celt_sig conversions (the caller passes values of the macro's argument type) */
    case 14: return SIG2RES(a);
    case 15: return RES2INT16((opus_res)a);
    case 16: return RES2INT24((opus_res)a);
    case 17: return INT16TORES((opus_int16)a);
    case 18: return INT24TORES(a);
    case 19: return RES2SIG((opus_res)a);
    case 20: return RES2VAL16((opus_res)a);
    case 21: return INT16TOSIG((opus_int16)a);
    case 22: return INT24TOSIG(a);
    default: return 0x7fffffff;
  }
}

int ofx_op2(int op, int a, int b) {
  int r;
  switch (op) {
    case 0: return MULT16_16SU(a, b);
    case 1: return MULT16_32_Q16(a, b);
    case 2: return MULT16_32_P16(a, b);
    case 3: return MULT16_32_Q15(a, b);
    case 4: return MULT32_32_Q16(a, b);
    case 5: return MULT32_32_Q31(a, b);
    case 6: return MULT32_32_P31(a, b);
    case 7: return MULT32_32_P31_ovflw(a, b);
    case 8: return MULT32_32_Q32(a, b);
    case 9: return SHR16(a, b);
    case 10: return SHL16(a, b);
    case 11: return SHR32(a, b);
    case 12: return SHL32(a, b);
    case 13: return PSHR32(a, b);
    case 14: return VSHR32(a, b);
    case 15: return SHR(a, b);
    case 16: return SHL(a, b);
    case 17: return PSHR(a, b);
    case 18: return SATURATE(a, b);
    case 19: return ROUND16(a, b);
    case 20: r = SROUND16(a, b) /* the macro ends with ';' */
      return r;
    case 21: return ADD16(a, b);
    case 22: return SUB16(a, b);
    case 23: return ADD32(a, b);
    case 24: return SUB32(a, b);
    case 25: return ADD32_ovflw(a, b);
    case 26: return SUB32_ovflw(a, b);
    case 27: return SHL32_ovflw(a, b);
    case 28: return PSHR32_ovflw(a, b);
    case 29: return MULT16_16_16(a, b);
    case 30: return MULT32_32_32(a, b);
    case 31: return MULT16_16(a, b);
    case 32: return MULT16_16_Q11_32(a, b);
    case 33: return MULT16_16_Q11(a, b);
    case 34: return MULT16_16_Q13(a, b);
    case 35: return MULT16_16_Q14(a, b);
    case 36: return MULT16_16_Q15(a, b);
    case 37: return MULT16_16_P13(a, b);
    case 38: return MULT16_16_P14(a, b);
    case 39: return MULT16_16_P15(a, b);
    case 40: return DIV32_16(a, b);
    case 41: return DIV32(a, b);
    case 42: return MIN16(a, b);
    case 43: return MAX16(a, b);
    case 44: return MIN32(a, b);
    case 45: return MAX32(a, b);
    case 46: return IMIN(a, b);
    case 47: return IMAX(a, b);
    case 48: return MULT_COEF_32(a, b);
    case 49: return MULT_COEF(a, b);
    case 50: return MULT_COEF_TAPS(a, b);
    case 51: return IMUL32(a, b);
    case 52: return MING(a, b);
    case 53: return MAXG(a, b);
    /* opus_res: the caller passes opus_res values; the result is stored in an opus_res as in
       every libopus use of these macros. */
    case 54: { opus_res x = ADD_RES((opus_res)a, (opus_res)b); return x; } /* ';' in res16 */
    case 55: { opus_res x = MULT16_RES_Q15(a, (opus_res)b); return x; }
    default: return 0x7fffffff;
  }
}

int ofx_op3(int op, int c, int a, int b) {
  switch (op) {
    case 0: return MAC16_16(c, a, b);
    case 1: return MAC16_32_Q15(c, a, b);
    case 2: return MAC16_32_Q16(c, a, b);
    case 3: return MAC_COEF_32_ARM(c, a, b);
    default: return 0x7fffffff;
  }
}

unsigned ofx_uadd32(unsigned a, unsigned b) { return UADD32(a, b); }
unsigned ofx_usub32(unsigned a, unsigned b) { return USUB32(a, b); }
long long ofx_shr64(long long a, int shift) { return SHR64(a, shift); }
int ofx_qconst16(double x, int bits) { return QCONST16(x, bits); }
int ofx_qconst16f(float x, int bits) { return QCONST16(x, bits); }
int ofx_qconst32(double x, int bits) { return QCONST32(x, bits); }
int ofx_qconst32f(float x, int bits) { return QCONST32(x, bits); }
int ofx_gconst(double x) { return GCONST(x); }
int ofx_gconst2(double x, int bits) { return GCONST2(x, bits); }
int ofx_frac_mul16(int a, int b) { return FRAC_MUL16(a, b); }
float ofx_res2float(int a) { return RES2FLOAT((opus_res)a); }
int ofx_float2int(float x) { return float2int(x); }
#ifndef DISABLE_FLOAT_API
int ofx_float2res(float a) { return FLOAT2RES(a); }
int ofx_float2int16(float x) { return FLOAT2INT16(x); }
int ofx_float2int24(float x) { return FLOAT2INT24(x); }
int ofx_float2sig(float x) { return FLOAT2SIG(x); }
#endif
float ofx_fast_atan2f(float y, float x) { return fast_atan2f(y, x); }
#ifdef ENABLE_QEXT
float ofx_celt_cos_norm2(float x) { return celt_cos_norm2(x); }
#endif
#ifndef DISABLE_FLOAT_API
void ofx_celt_float2int16(const float *in, short *out, int cnt) { celt_float2int16_c(in, out, cnt); }
int ofx_opus_limit2_checkwithin1(float *samples, int cnt) {
  return opus_limit2_checkwithin1_c(samples, cnt);
}
#endif

/* Build constants: see `constants()` in the Rust bindings for the order. */
void ofx_constants(int *out) {
  int i = 0;
  out[i++] = Q15ONE;
  out[i++] = Q31ONE;
  out[i++] = COEF_ONE;
  out[i++] = SIG_SHIFT;
  out[i++] = SIG_SAT;
  out[i++] = NORM_SHIFT;
  out[i++] = NORM_SCALING;
  out[i++] = DB_SHIFT;
  out[i++] = EPSILON;
  out[i++] = VERY_SMALL;
  out[i++] = VERY_LARGE16;
  out[i++] = Q15_ONE;
  out[i++] = RES_SHIFT;
  out[i++] = MAX_ENCODING_DEPTH;
  out[i++] = OPUS_FAST_INT64;
  out[i++] = (int)sizeof(opus_val16);
  out[i++] = (int)sizeof(opus_val32);
  out[i++] = (int)sizeof(opus_val64);
  out[i++] = (int)sizeof(celt_sig);
  out[i++] = (int)sizeof(celt_norm);
  out[i++] = (int)sizeof(celt_ener);
  out[i++] = (int)sizeof(celt_glog);
  out[i++] = (int)sizeof(opus_res);
  out[i++] = (int)sizeof(celt_coef);
  out[i++] = (int)sizeof(kiss_fft_scalar);
  out[i++] = (int)sizeof(kiss_twiddle_scalar);
  out[i++] = GLOBAL_STACK_SIZE;
}

/* ---- Fixed-point mathops, by function id ---- */

int ofx_math1(int op, int x) {
  switch (op) {
    case 0: return celt_rsqrt_norm(x);
    case 1: return celt_rsqrt_norm32(x);
    case 2: return celt_sqrt(x);
    case 3: return celt_sqrt32(x);
    case 4: return celt_cos_norm(x);
    case 5: return celt_cos_norm32(x);
    case 6: return celt_log2(x);
    case 7: return celt_exp2_frac((opus_val16)x);
    case 8: return celt_exp2((opus_val16)x);
    case 9: return celt_log2_db(x);
    case 10: return celt_exp2_db_frac(x);
    case 11: return celt_exp2_db(x);
    case 12: return celt_rcp(x);
    case 13: return celt_rcp_norm16((opus_val16)x);
    case 14: return celt_rcp_norm32(x);
    case 15: return celt_atan_norm(x);
    case 16: return celt_atan01((opus_val16)x);
    case 17: return celt_ilog2(x);
    case 18: return celt_zlog2(x);
    default: return 0x7fffffff;
  }
}

int ofx_math2(int op, int a, int b) {
  switch (op) {
    case 0: return celt_div(a, b);
    case 1: return frac_div32_q29(a, b);
    case 2: return frac_div32(a, b);
    case 3: return celt_atan2p_norm(a, b);
    case 4: return celt_atan2p((opus_val16)a, (opus_val16)b);
    default: return 0x7fffffff;
  }
}

unsigned ofx_isqrt32(unsigned x) { return isqrt32(x); }
int ofx_maxabs16(const opus_val16 *x, int len) { return celt_maxabs16(x, len); }
int ofx_maxabs_res(const opus_res *x, int len) { return celt_maxabs_res(x, len); }
int ofx_maxabs32(const opus_val32 *x, int len) { return celt_maxabs32(x, len); }

/* ---- Static modes ---- */

static const CELTMode *mode_for(int fs, int frame_size) {
  int err;
  return opus_custom_mode_create(fs, frame_size, &err);
}

/* Scalar fields: Fs, overlap, nbEBands, effEBands, preemph[4], maxLM, nbShortMdcts,
   shortMdctSize, nbAllocVectors, mdct.n, mdct.maxshift, cache.size, qext_cache.size (or -1).
   Returns 0 on success, -1 if the mode does not exist. */
int ofx_mode_scalars(int fs, int frame_size, int *out) {
  const CELTMode *m = mode_for(fs, frame_size);
  int i = 0, k;
  if (m == NULL) return -1;
  out[i++] = m->Fs;
  out[i++] = m->overlap;
  out[i++] = m->nbEBands;
  out[i++] = m->effEBands;
  for (k = 0; k < 4; k++) out[i++] = m->preemph[k];
  out[i++] = m->maxLM;
  out[i++] = m->nbShortMdcts;
  out[i++] = m->shortMdctSize;
  out[i++] = m->nbAllocVectors;
  out[i++] = m->mdct.n;
  out[i++] = m->mdct.maxshift;
  out[i++] = m->cache.size;
#ifdef ENABLE_QEXT
  out[i++] = m->qext_cache.size;
#else
  out[i++] = -1;
#endif
  return 0;
}

/* Copies n entries of one of the mode's arrays (as ints): 0 eBands, 1 allocVectors, 2 logN,
   3 window, 4 mdct.trig, 5 cache.index, 6 cache.bits, 7 cache.caps, 8 qext_cache.index,
   9 qext_cache.bits, 10 qext_cache.caps. */
int ofx_mode_array(int fs, int frame_size, int which, int *out, int n) {
  const CELTMode *m = mode_for(fs, frame_size);
  int i;
  if (m == NULL) return -1;
  for (i = 0; i < n; i++) {
    switch (which) {
      case 0: out[i] = m->eBands[i]; break;
      case 1: out[i] = m->allocVectors[i]; break;
      case 2: out[i] = m->logN[i]; break;
      case 3: out[i] = m->window[i]; break;
      case 4: out[i] = m->mdct.trig[i]; break;
      case 5: out[i] = m->cache.index[i]; break;
      case 6: out[i] = m->cache.bits[i]; break;
      case 7: out[i] = m->cache.caps[i]; break;
#ifdef ENABLE_QEXT
      case 8: out[i] = m->qext_cache.index[i]; break;
      case 9: out[i] = m->qext_cache.bits[i]; break;
      case 10: out[i] = m->qext_cache.caps[i]; break;
#endif
      default: return -1;
    }
  }
  return 0;
}

/* FFT state `k` of the mode's MDCT: out = {nfft, scale, scale_shift, shift, factors[16]};
   bitrev (nfft entries) and twiddles (ntw complex entries, interleaved r/i). */
int ofx_mode_fft(int fs, int frame_size, int k, int *scalars, int *bitrev, int *twiddles,
                 int ntw) {
  const CELTMode *m = mode_for(fs, frame_size);
  const kiss_fft_state *st;
  int i;
  if (m == NULL) return -1;
  st = m->mdct.kfft[k];
  scalars[0] = st->nfft;
  scalars[1] = st->scale;
  scalars[2] = st->scale_shift;
  scalars[3] = st->shift;
  for (i = 0; i < 2 * MAXFACTORS; i++) scalars[4 + i] = st->factors[i];
  for (i = 0; i < st->nfft; i++) bitrev[i] = st->bitrev[i];
  for (i = 0; i < ntw; i++) {
    twiddles[2 * i] = st->twiddles[i].r;
    twiddles[2 * i + 1] = st->twiddles[i].i;
  }
  return 0;
}

/* ---- CWRS + Laplace through the range coder ---- */

/* Encodes pulse vector y (n, k) into buf, then decodes it back: y_out, returns the decoded
   squared norm (opus_val32); *rng_out = final encoder rng. */
int ofx_cwrs_roundtrip(const int *y, int n, int k, unsigned char *buf, int size, int *y_out,
                       unsigned *rng_out) {
  ec_enc enc;
  ec_dec dec;
  ec_enc_init(&enc, buf, size);
  encode_pulses(y, n, k, &enc);
  ec_enc_done(&enc);
  *rng_out = enc.rng;
  ec_dec_init(&dec, buf, size);
  return decode_pulses(y_out, n, k, &dec);
}

/* Laplace-encodes values[] with (fs, decay) (values are updated as C clamps them), then decodes
   them into out[]. */
void ofx_laplace_roundtrip(int *values, int count, const unsigned *fs, const int *decay,
                           unsigned char *buf, int size, int *out, unsigned *rng_out) {
  ec_enc enc;
  ec_dec dec;
  int i;
  ec_enc_init(&enc, buf, size);
  for (i = 0; i < count; i++) ec_laplace_encode(&enc, &values[i], fs[i], decay[i]);
  ec_enc_done(&enc);
  *rng_out = enc.rng;
  ec_dec_init(&dec, buf, size);
  for (i = 0; i < count; i++) out[i] = ec_laplace_decode(&dec, fs[i], decay[i]);
}

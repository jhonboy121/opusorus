/* Oracle shims for the foundation unit (range coder, mathops). */
// oracle-build: any
/* The range coder shims are shared by the float and fixed-point oracles; the float mathops shims
   are float-only (fixed-point mathops: fixed_foundation.c). */
#include <string.h>
#include "entenc.h"
#include "entdec.h"
#include "mathops.h"
#include "float_cast.h"

/* A fixed ICDF table shared with the Rust test. */
static const unsigned char ORACLE_ICDF[8] = {250, 200, 150, 100, 60, 30, 10, 0};

/* Runs a sequence of encoder ops. ops[i*4..i*4+4] = {kind, a, b, c}.
   kinds: 0 encode(a,b,c) 1 encode_bin(a,b,c) 2 bit_logp(a,b) 3 icdf(a, ORACLE_ICDF, b)
          4 uint(a,b) 5 bits(a,b) 6 patch_initial_bits(a,b)
   After ops: ec_enc_done. Returns error flag; writes tell/tell_frac after every op into
   tells[2*i], and final rng into *rng_out. Output buffer is `buf` of `size` bytes. */
int oracle_ec_encode_ops(const unsigned *ops, int nops, unsigned char *buf, int size,
                         int *tells, unsigned *rng_out, int shrink_to) {
  ec_enc enc;
  int i;
  ec_enc_init(&enc, buf, size);
  for (i = 0; i < nops; i++) {
    const unsigned *o = ops + 4 * i;
    switch (o[0]) {
      case 0: ec_encode(&enc, o[1], o[2], o[3]); break;
      case 1: ec_encode_bin(&enc, o[1], o[2], o[3]); break;
      case 2: ec_enc_bit_logp(&enc, o[1], o[2]); break;
      case 3: ec_enc_icdf(&enc, o[1], ORACLE_ICDF, o[2]); break;
      case 4: ec_enc_uint(&enc, o[1], o[2]); break;
      case 5: ec_enc_bits(&enc, o[1], o[2]); break;
      case 6: ec_enc_patch_initial_bits(&enc, o[1], o[2]); break;
    }
    tells[2 * i] = ec_tell(&enc);
    tells[2 * i + 1] = (int)ec_tell_frac(&enc);
  }
  if (shrink_to > 0) ec_enc_shrink(&enc, shrink_to);
  ec_enc_done(&enc);
  *rng_out = enc.rng;
  return enc.error;
}

/* Decoder ops: kinds 0 decode(ft)+update(chosen by C as the decoded symbol, width 1)
   1 decode_bin(bits)+update  2 bit_logp(logp)  3 icdf(ORACLE_ICDF, ftb)  4 uint(ft)  5 bits(n).
   ops[i*2..] = {kind, param}. out[i] = decoded value, tells as encoder. */
int oracle_ec_decode_ops(const unsigned *ops, int nops, unsigned char *buf, int size,
                         unsigned *out, int *tells, unsigned *rng_out) {
  ec_dec dec;
  int i;
  ec_dec_init(&dec, buf, size);
  for (i = 0; i < nops; i++) {
    const unsigned *o = ops + 2 * i;
    unsigned v = 0;
    switch (o[0]) {
      case 0: v = ec_decode(&dec, o[1]); ec_dec_update(&dec, v, v + 1, o[1]); break;
      case 1: v = ec_decode_bin(&dec, o[1]); ec_dec_update(&dec, v, v + 1, 1u << o[1]); break;
      case 2: v = ec_dec_bit_logp(&dec, o[1]); break;
      case 3: v = ec_dec_icdf(&dec, ORACLE_ICDF, o[1]); break;
      case 4: v = ec_dec_uint(&dec, o[1]); break;
      case 5: v = ec_dec_bits(&dec, o[1]); break;
    }
    out[i] = v;
    tells[2 * i] = ec_tell(&dec);
    tells[2 * i + 1] = (int)ec_tell_frac(&dec);
  }
  *rng_out = dec.rng;
  return dec.error;
}

#ifndef FIXED_POINT
float oracle_celt_log2(float x) { return celt_log2(x); }
float oracle_celt_exp2(float x) { return celt_exp2(x); }
float oracle_celt_cos_norm(float x) { return celt_cos_norm(x); }
float oracle_celt_cos_norm2(float x) { return celt_cos_norm2(x); }
float oracle_celt_sin(float x) { return celt_sin(x); }
float oracle_fast_atan2f(float y, float x) { return fast_atan2f(y, x); }
float oracle_celt_atan2p_norm(float y, float x) { return celt_atan2p_norm(y, x); }
float oracle_celt_sqrt(float x) { return celt_sqrt(x); }
float oracle_celt_rsqrt(float x) { return celt_rsqrt(x); }
#endif
unsigned oracle_isqrt32(unsigned x) { return isqrt32(x); }
int oracle_float2int(float x) { return float2int(x); }
short oracle_float2int16(float x) { return FLOAT2INT16(x); }

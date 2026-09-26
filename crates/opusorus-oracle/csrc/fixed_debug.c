/* Oracle shims for the `fixed-point-debug` feature (libopus FIXED_DEBUG): capture of the
 * diagnostics printed by celt/fixed_debug.h and silk/MacroDebug.h (fprintf is renamed to
 * oracle_fixed_debug_fprintf by the force-included fixed_debug_capture.h), access to the
 * `celt_mips` operation counter, and entry points evaluating each checking macro.
 *
 * The capture buffer is thread-local (tests run concurrently); `celt_mips` is libopus' global
 * (tests that read it serialize themselves). Without FIXED_DEBUG this file is empty. */
// oracle-build: fixed
#ifdef FIXED_DEBUG
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "arch.h"
#include "SigProc_FIX.h"
#undef fprintf

/* Captured messages: the first ORACLE_FDBG_KEEP messages since the last clear (in a growing
   thread-local buffer), and the number of messages since the last clear. */
#define ORACLE_FDBG_KEEP 65536
#define ORACLE_FDBG_LEN 256
static _Thread_local char (*fdbg_msg)[ORACLE_FDBG_LEN];
static _Thread_local long long fdbg_cap;
static _Thread_local long long fdbg_count;

int oracle_fixed_debug_fprintf(FILE *stream, const char *fmt, ...) {
  va_list ap;
  int n = 0;
  (void)stream;
  if (fdbg_count < ORACLE_FDBG_KEEP) {
    if (fdbg_count >= fdbg_cap) {
      long long cap = fdbg_cap ? 2 * fdbg_cap : 256;
      char (*m)[ORACLE_FDBG_LEN] = realloc(fdbg_msg, (size_t)cap * ORACLE_FDBG_LEN);
      if (!m) abort();
      fdbg_msg = m;
      fdbg_cap = cap;
    }
    va_start(ap, fmt);
    n = vsnprintf(fdbg_msg[fdbg_count], ORACLE_FDBG_LEN, fmt, ap);
    va_end(ap);
  }
  fdbg_count++;
  return n;
}

/* Number of messages since the last clear. */
long long oracle_fdbg_count(void) { return fdbg_count; }

/* Copies message `i` (0 = first) into `buf`; returns its length or -1. */
int oracle_fdbg_message(long long i, char *buf, int cap) {
  const char *src;
  int len;
  if (i < 0 || i >= fdbg_count || i >= ORACLE_FDBG_KEEP || cap <= 0) return -1;
  src = fdbg_msg[i];
  len = (int)strlen(src);
  if (len >= cap) len = cap - 1;
  memcpy(buf, src, len);
  buf[len] = 0;
  return len;
}

void oracle_fdbg_clear(void) { fdbg_count = 0; }

long long oracle_fdbg_celt_mips(void) { return celt_mips; }
void oracle_fdbg_set_celt_mips(long long v) { celt_mips = v; }

/* ---- celt/fixed_debug.h: every macro, by id ----
   Arguments are passed as 64-bit values and converted to each C parameter type by the call
   (as in libopus, where they are whatever expression the caller wrote). */
long long oracle_fdbg_celt(int op, long long a, long long b, long long c) {
  switch (op) {
    case 0: return NEG16((int)a);
    case 1: return NEG32(a);
    case 2: return EXTRACT16((int)a);
    case 3: return EXTEND32((int)a);
    case 4: return SHR16((int)a, (int)b);
    case 5: return SHL16((int)a, (int)b);
    case 6: return SHR32(a, (int)b);
    case 7: return SHL32(a, (int)b);
    case 8: return PSHR32((opus_int32)a, (int)b);
    case 9: return VSHR32((opus_int32)a, (int)b);
    case 10: return SHR64(a, (int)b);
    case 11: return ROUND16((opus_int32)a, (int)b);
    case 12: return SROUND16((opus_int32)a, (int)b);
    case 13: return HALF16((int)a);
    case 14: return HALF32((opus_int32)a);
    case 15: return ADD16((int)a, (int)b);
    case 16: return SUB16((int)a, (int)b);
    case 17: return ADD32(a, b);
    case 18: return SUB32(a, b);
    case 19: return UADD32((opus_uint32)a, (opus_uint32)b);
    case 20: return USUB32((opus_uint32)a, (opus_uint32)b);
    case 21: return MULT16_16_16((int)a, (int)b);
    case 22: return MULT32_32_32(a, b);
    case 23: return MULT32_32_Q16(a, b);
    case 24: return MULT16_16((int)a, (int)b);
    case 25: return MAC16_16((opus_int32)c, (int)a, (int)b);
    case 26: return MULT16_32_Q15((int)a, b);
    case 27: return MULT16_32_P16((int)a, b);
    case 28: return MAC16_32_Q15((opus_int32)c, (int)a, (opus_int32)b);
    case 29: return MAC16_32_Q16((opus_int32)c, (int)a, (opus_int32)b);
    case 30: return SATURATE((int)a, (int)b);
    case 31: return SATURATE16((opus_int32)a);
    case 32: return MULT16_16_Q11_32((int)a, (int)b);
    case 33: return MULT16_16_Q13((int)a, (int)b);
    case 34: return MULT16_16_Q14((int)a, (int)b);
    case 35: return MULT16_16_Q15((int)a, (int)b);
    case 36: return MULT16_16_P13((int)a, (int)b);
    case 37: return MULT16_16_P14((int)a, (int)b);
    case 38: return MULT16_16_P15((int)a, (int)b);
    case 39: return DIV32_16(a, b);
    case 40: return DIV32(a, b);
    case 41: return SIG2WORD16((opus_int32)a);
    case 42: return MULT16_32_Q16((opus_int32)a, (opus_int32)b);
    case 43: return MULT32_32_Q31((opus_int32)a, (opus_int32)b);
    case 44: return MULT32_32_P31((opus_int32)a, (opus_int32)b);
    case 45: return MULT32_32_P31_ovflw((opus_int32)a, (opus_int32)b);
    case 46: return MULT32_32_Q32((opus_int32)a, (opus_int32)b);
    case 47: return ADD32_ovflw((opus_int32)a, (opus_int32)b);
    case 48: return SUB32_ovflw((opus_int32)a, (opus_int32)b);
    case 49: return NEG32_ovflw((opus_int32)a);
    case 50: return SHL32_ovflw((opus_int32)a, (int)b);
    case 51: return PSHR32_ovflw((opus_int32)a, (int)b);
    case 52: return MULT16_16SU((opus_int32)a, (opus_int32)b);
    case 53: return MULT16_16U((opus_uint32)a, (opus_uint32)b);
    case 54: return SHR((opus_int32)a, (int)b);
    case 55: return PSHR((opus_int32)a, (int)b);
    /* arch.h macros built on them */
    case 60: return SIG2RES((opus_int32)a);
    case 61: return RES2INT16((opus_res)a);
    case 62: return RES2INT24((opus_res)a);
    case 63: return INT16TORES((opus_int16)a);
    case 64: return INT24TORES((opus_int32)a);
    case 65: return ADD_RES((opus_res)a, (opus_res)b);
    case 66: return RES2SIG((opus_res)a);
    case 67: return MULT16_RES_Q15((opus_int16)a, (opus_res)b);
    case 68: return INT16TOSIG((opus_int16)a);
    case 69: return INT24TOSIG((opus_int32)a);
    case 70: return MULT_COEF_32((celt_coef)a, (opus_int32)b);
    case 71: return MAC_COEF_32_ARM((opus_int32)c, (celt_coef)a, (opus_int32)b);
    case 72: return MULT_COEF((celt_coef)a, (celt_coef)b);
    case 73: return MULT_COEF_TAPS((opus_int16)a, (opus_int16)b);
    case 74: return COEF2VAL16((celt_coef)a);
  }
  return 0x7eadbeefLL;
}

/* ---- silk/MacroDebug.h: every checked macro, by id ---- */
long long oracle_fdbg_silk(int op, long long a, long long b, long long c) {
  switch (op) {
    case 0: return silk_ADD16((opus_int16)a, (opus_int16)b);
    case 1: return silk_ADD32((opus_int32)a, (opus_int32)b);
    case 2: return silk_ADD64(a, b);
    case 3: return silk_SUB16((opus_int16)a, (opus_int16)b);
    case 4: return silk_SUB32((opus_int32)a, (opus_int32)b);
    case 5: return silk_SUB64(a, b);
    case 6: return silk_ADD_SAT16((opus_int16)a, (opus_int16)b);
    case 7: return silk_ADD_SAT32((opus_int32)a, (opus_int32)b);
    case 8: return silk_ADD_SAT64(a, b);
    case 9: return silk_SUB_SAT16((opus_int16)a, (opus_int16)b);
    case 10: return silk_SUB_SAT32((opus_int32)a, (opus_int32)b);
    case 11: return silk_SUB_SAT64(a, b);
    case 12: return silk_MUL((opus_int32)a, (opus_int32)b);
    case 13: return silk_MUL_uint((opus_uint32)a, (opus_uint32)b);
    case 14: return silk_MLA((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 15: return silk_MLA_uint((opus_uint32)a, (opus_uint32)b, (opus_uint32)c);
    case 16: return silk_SMULWB((opus_int32)a, (opus_int32)b);
    case 17: return silk_SMLAWB((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 18: return silk_SMULWT((opus_int32)a, (opus_int32)b);
    case 19: return silk_SMLAWT((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 20: return silk_SMULL(a, b);
    case 21: return silk_SMLABB((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 22: return silk_SMLABT((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 23: return silk_SMLATT((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 24: return silk_SMULWW((opus_int32)a, (opus_int32)b);
    case 25: return silk_SMLAWW((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 26: return b == 0 ? 0 : silk_DIV32((opus_int32)a, (opus_int32)b);
    case 27: return b == 0 ? 0 : silk_DIV32_16((opus_int32)a, (opus_int32)b);
    case 28: return silk_LSHIFT8((opus_int8)a, (opus_int32)b);
    case 29: return silk_LSHIFT16((opus_int16)a, (opus_int32)b);
    case 30: return silk_LSHIFT32((opus_int32)a, (opus_int32)b);
    case 31: return silk_LSHIFT64(a, (opus_int)b);
    case 32: return silk_LSHIFT_ovflw((opus_int32)a, (opus_int32)b);
    case 33: return silk_LSHIFT_uint((opus_uint32)a, (opus_int32)b);
    case 34: return silk_RSHIFT32((opus_int32)a, (opus_int32)b);
    case 35: return silk_RSHIFT64(a, b);
    case 36: return silk_RSHIFT_uint((opus_uint32)a, (opus_int32)b);
    case 37: return silk_ADD_LSHIFT((int)a, (int)b, (int)c);
    case 38: return silk_ADD_LSHIFT32((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 39: return silk_ADD_LSHIFT_uint((opus_uint32)a, (opus_uint32)b, (opus_int32)c);
    case 40: return silk_ADD_RSHIFT((int)a, (int)b, (int)c);
    case 41: return silk_ADD_RSHIFT32((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 42: return silk_ADD_RSHIFT_uint((opus_uint32)a, (opus_uint32)b, (opus_int32)c);
    case 43: return silk_SUB_LSHIFT32((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 44: return silk_SUB_RSHIFT32((opus_int32)a, (opus_int32)b, (opus_int32)c);
    case 45: return silk_RSHIFT_ROUND((opus_int32)a, (opus_int32)b);
    case 46: return silk_RSHIFT_ROUND64(a, (opus_int32)b);
    case 47: return silk_abs_int64(a);
    case 48: return silk_abs_int32((opus_int32)a);
    case 49: return silk_CHECK_FIT8(a);
    case 50: return silk_CHECK_FIT16(a);
    case 51: return silk_CHECK_FIT32(a);
  }
  return 0x7eadbeefLL;
}
#endif

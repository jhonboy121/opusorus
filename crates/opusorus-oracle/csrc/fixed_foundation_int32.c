/* Oracle shims for unit fixed_foundation: the 32-bit (`OPUS_FAST_INT64 == 0`) forms of the
   fixed_generic.h multiply macros, which a 64-bit host build of libopus never uses. This file
   re-includes the unmodified upstream fixed_generic.h with OPUS_FAST_INT64 forced to 0. */
// oracle-build: fixed
#include "arch.h"

#undef OPUS_FAST_INT64
#define OPUS_FAST_INT64 0
#undef FIXED_GENERIC_H
#undef MULT16_32_Q16
#undef MULT16_32_P16
#undef MULT16_32_Q15
#undef MULT32_32_Q16
#undef MULT32_32_Q31
#undef MULT32_32_P31
#undef MULT32_32_P31_ovflw
#undef MULT32_32_Q32
/* Keep the second copy of the static inline helper from clashing with the first one. */
#define SIG2WORD16_generic SIG2WORD16_generic_int32
#include "fixed_generic.h"

/* Same op ids as ofx_op2 (0..8). */
int ofx32_op2(int op, int a, int b) {
  switch (op) {
    case 1: return MULT16_32_Q16(a, b);
    case 2: return MULT16_32_P16(a, b);
    case 3: return MULT16_32_Q15(a, b);
    case 4: return MULT32_32_Q16(a, b);
    case 5: return MULT32_32_Q31(a, b);
    case 6: return MULT32_32_P31(a, b);
    case 7: return MULT32_32_P31_ovflw(a, b);
    case 8: return MULT32_32_Q32(a, b);
    default: return 0x7fffffff;
  }
}

/* MULT16_16U only exists in the 32-bit configuration. */
unsigned ofx_mult16_16u(unsigned a, unsigned b) { return MULT16_16U(a, b); }

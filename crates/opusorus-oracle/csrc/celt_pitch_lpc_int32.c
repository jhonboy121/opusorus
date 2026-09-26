/* Oracle shims for unit celt_pitch_lpc (fixed-point oracle only): a private copy of
   celt/celt_lpc.c compiled with the 32-bit (`OPUS_FAST_INT64 == 0`) forms of the
   fixed_generic.h multiply macros, which a 64-bit host build of libopus never uses. `_celt_lpc`
   is the only function of the unit whose result depends on them (`MULT32_32_Q31` and its
   `OPUS_FAST_INT64` accumulation); `frac_div32` stays the library (host) version, as it is a
   function of mathops.c. Same technique as fixed_foundation_int32.c. */
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
#define SIG2WORD16_generic SIG2WORD16_generic_cpl_int32
#include "fixed_generic.h"

/* Rename the exported functions of the private copy. */
#define _celt_lpc cpl32_celt_lpc
#define celt_fir_c cpl32_celt_fir_c
#define celt_iir cpl32_celt_iir
#define _celt_autocorr cpl32_celt_autocorr
#include "celt_lpc.c"
#undef _celt_lpc
#undef celt_fir_c
#undef celt_iir
#undef _celt_autocorr

void oracle_celt_lpc_int32(opus_val16 *lpc, const opus_val32 *ac, int p) {
  cpl32_celt_lpc(lpc, ac, p);
}

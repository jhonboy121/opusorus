/* Oracle shims for the upstream build options FLOAT_APPROX, ENABLE_ASSERTIONS, FUZZING and
   DISABLE_UPDATE_DRAFT (cargo features float-approx, assertions, fuzzing, disable-rfc8251). */
// oracle-build: any
#include <stdlib.h>
#include "arch.h"
#include "entenc.h"
#include "mathops.h"
#include "SigProc_FIX.h"

/* Marker symbols: the tests that link programs against the oracle's libopus.a pick the archive
   whose options match their features by these names. */
#ifdef FLOAT_APPROX
const int opusorus_oracle_float_approx = 1;
#endif
#ifdef ENABLE_ASSERTIONS
const int opusorus_oracle_assertions = 1;
#endif
#ifdef FUZZING
const int opusorus_oracle_fuzzing = 1;
#endif
#ifdef DISABLE_UPDATE_DRAFT
const int opusorus_oracle_disable_update_draft = 1;
#endif

/* Bit 0: FLOAT_APPROX, 1: ENABLE_ASSERTIONS, 2: FUZZING, 3: DISABLE_UPDATE_DRAFT. */
int oracle_build_options(void) {
  int v = 0;
#ifdef FLOAT_APPROX
  v |= 1;
#endif
#ifdef ENABLE_ASSERTIONS
  v |= 2;
#endif
#ifdef FUZZING
  v |= 4;
#endif
#ifdef DISABLE_UPDATE_DRAFT
  v |= 8;
#endif
  return v;
}

/* The C library's rand()/srand(), which the FUZZING encoder draws from. */
void oracle_srand(unsigned seed) { srand(seed); }
int oracle_rand(void) { return rand(); }

#ifndef FIXED_POINT
/* celt/arch.h, celt/mathops.h (float build; FLOAT_APPROX selects the approximations). */
int oracle_opt_celt_isnan(float x) { return celt_isnan(x); }
float oracle_opt_celt_log2(float x) { return celt_log2(x); }
float oracle_opt_celt_exp2(float x) { return celt_exp2(x); }
#endif

/* Each fails one libopus check (the process aborts when the check is enabled). */
void oracle_fail_celt_assert(void) {
  unsigned char buf[16];
  ec_enc enc;
  ec_enc_init(&enc, buf, sizeof(buf));
  ec_enc_uint(&enc, 0, 1); /* celt_assert(_ft>1) */
}

int oracle_fail_celt_sig_assert(void) {
#ifdef FIXED_POINT
  return celt_ilog2(0); /* celt_sig_assert(x>0) */
#else
  return (int)celt_atan2p_norm(-1.f, 1.f); /* celt_sig_assert(x>=0 && y>=0) */
#endif
}

void oracle_fail_silk_assert(void) {
  opus_int16 nlsf[16];
  opus_int16 delta[17];
  int i;
  for (i = 0; i < 16; i++) nlsf[i] = (opus_int16)(1000 * (i + 1));
  for (i = 0; i < 17; i++) delta[i] = 100;
  delta[16] = 0;
  silk_NLSF_stabilize(nlsf, delta, 16); /* silk_assert( NDeltaMin_Q15[L] >= 1 ) */
}

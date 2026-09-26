/* Oracle C shims for unit celt_pitch_lpc (celt/pitch.c, celt/celt_lpc.c).

   Compiled in the float and in the fixed-point oracle: the shims use the libopus types
   (opus_val16/opus_val32/celt_sig/celt_coef are float in the float build, integers in the
   fixed-point build), mirrored by the type aliases of opusorus-oracle/src/celt_pitch_lpc.rs.

   The static helpers of pitch.c (find_best_pitch, celt_fir5, compute_pitch_gain) are reached by
   #including a private copy of pitch.c whose exported functions are renamed (so they do not
   clash with the library symbols). All exported functions below call the *library* versions. */
// oracle-build: any
#include <stddef.h>

#define pitch_downsample cpl_copy_pitch_downsample
#define pitch_search cpl_copy_pitch_search
#define remove_doubling cpl_copy_remove_doubling
#define celt_pitch_xcorr_c cpl_copy_celt_pitch_xcorr_c
#include "pitch.c"
#undef pitch_downsample
#undef pitch_search
#undef remove_doubling
#undef celt_pitch_xcorr_c

#include "celt_lpc.h"

/* Library prototypes (the ones from pitch.h were renamed above). */
void pitch_downsample(celt_sig * OPUS_RESTRICT x[], opus_val16 * OPUS_RESTRICT x_lp,
      int len, int C, int factor, int arch);
void pitch_search(const opus_val16 * OPUS_RESTRICT x_lp, opus_val16 * OPUS_RESTRICT y,
                  int len, int max_pitch, int *pitch, int arch);
opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
      int N, int *T0, int prev_period, opus_val16 prev_gain, int arch);
#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_c(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch);

void oracle_xcorr_kernel(const opus_val16 *x, const opus_val16 *y, opus_val32 *sum, int len) {
  xcorr_kernel_c(x, y, sum, len);
}

void oracle_dual_inner_prod(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
                            int n, opus_val32 *xy1, opus_val32 *xy2) {
  dual_inner_prod_c(x, y01, y02, n, xy1, xy2);
}

opus_val32 oracle_celt_inner_prod(const opus_val16 *x, const opus_val16 *y, int n) {
  return celt_inner_prod_c(x, y, n);
}

/* Returns maxcorr in the fixed-point build, 0 in the float build. */
opus_val32 oracle_celt_pitch_xcorr(const opus_val16 *x, const opus_val16 *y, opus_val32 *xcorr,
                                   int len, int max_pitch) {
#ifdef FIXED_POINT
  return celt_pitch_xcorr_c(x, y, xcorr, len, max_pitch, 0);
#else
  celt_pitch_xcorr_c(x, y, xcorr, len, max_pitch, 0);
  return 0;
#endif
}

/* yshift/maxcorr are only used by the fixed-point build. */
void oracle_find_best_pitch(const opus_val32 *xcorr, const opus_val16 *y, int len,
                            int max_pitch, int *best_pitch, int yshift, opus_val32 maxcorr) {
#ifdef FIXED_POINT
  find_best_pitch((opus_val32 *)xcorr, (opus_val16 *)y, len, max_pitch, best_pitch, yshift,
                  maxcorr);
#else
  (void)yshift;
  (void)maxcorr;
  find_best_pitch((opus_val32 *)xcorr, (opus_val16 *)y, len, max_pitch, best_pitch);
#endif
}

void oracle_celt_fir5(opus_val16 *x, const opus_val16 *num, int n) { celt_fir5(x, num, n); }

void oracle_pitch_downsample(const celt_sig *x0, const celt_sig *x1, opus_val16 *x_lp, int len,
                             int C, int factor) {
  celt_sig *x[2];
  x[0] = (celt_sig *)x0;
  x[1] = (celt_sig *)x1;
  pitch_downsample(x, x_lp, len, C, factor, 0);
}

int oracle_pitch_search(const opus_val16 *x_lp, const opus_val16 *y, int len, int max_pitch) {
  int pitch = -12345;
  pitch_search(x_lp, (opus_val16 *)y, len, max_pitch, &pitch, 0);
  return pitch;
}

opus_val16 oracle_compute_pitch_gain(opus_val32 xy, opus_val32 xx, opus_val32 yy) {
  return compute_pitch_gain(xy, xx, yy);
}

opus_val16 oracle_remove_doubling(const opus_val16 *x, int maxperiod, int minperiod, int n,
                                  int *t0, int prev_period, opus_val16 prev_gain) {
  return remove_doubling((opus_val16 *)x, maxperiod, minperiod, n, t0, prev_period, prev_gain, 0);
}

void oracle_celt_lpc(opus_val16 *lpc, const opus_val32 *ac, int p) { _celt_lpc(lpc, ac, p); }

/* x points at `ord` history samples followed by the n input samples. */
void oracle_celt_fir(const opus_val16 *x, const opus_val16 *num, opus_val16 *y, int n, int ord) {
  celt_fir_c(x + ord, num, y, n, ord, 0);
}

/* in_place != 0 filters y in place (x ignored), like the CELT decoder PLC does. */
void oracle_celt_iir(const opus_val32 *x, const opus_val16 *den, opus_val32 *y, int n, int ord,
                     opus_val16 *mem, int in_place) {
  celt_iir(in_place ? y : x, den, y, n, ord, mem, 0);
}

int oracle_celt_autocorr(const opus_val16 *x, opus_val32 *ac, const celt_coef *window,
                         int overlap, int lag, int n) {
  return _celt_autocorr(x, ac, overlap ? window : NULL, overlap, lag, n, 0);
}

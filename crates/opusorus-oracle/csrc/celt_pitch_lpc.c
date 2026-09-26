/* Oracle C shims for unit celt_pitch_lpc (celt/pitch.c, celt/celt_lpc.c).

   The static helpers of pitch.c (find_best_pitch, celt_fir5, compute_pitch_gain) are reached by
   #including a private copy of pitch.c whose exported functions are renamed (so they do not
   clash with the library symbols). All exported functions below call the *library* versions. */
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
void celt_pitch_xcorr_c(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch);

void oracle_xcorr_kernel(const float *x, const float *y, float *sum, int len) {
  xcorr_kernel_c(x, y, sum, len);
}

void oracle_dual_inner_prod(const float *x, const float *y01, const float *y02, int n,
                            float *xy1, float *xy2) {
  dual_inner_prod_c(x, y01, y02, n, xy1, xy2);
}

float oracle_celt_inner_prod(const float *x, const float *y, int n) {
  return celt_inner_prod_c(x, y, n);
}

void oracle_celt_pitch_xcorr(const float *x, const float *y, float *xcorr, int len,
                             int max_pitch) {
  celt_pitch_xcorr_c(x, y, xcorr, len, max_pitch, 0);
}

void oracle_find_best_pitch(const float *xcorr, const float *y, int len, int max_pitch,
                            int *best_pitch) {
  find_best_pitch((opus_val32 *)xcorr, (opus_val16 *)y, len, max_pitch, best_pitch);
}

void oracle_celt_fir5(float *x, const float *num, int n) { celt_fir5(x, num, n); }

void oracle_pitch_downsample(const float *x0, const float *x1, float *x_lp, int len, int C,
                             int factor) {
  celt_sig *x[2];
  x[0] = (celt_sig *)x0;
  x[1] = (celt_sig *)x1;
  pitch_downsample(x, x_lp, len, C, factor, 0);
}

int oracle_pitch_search(const float *x_lp, const float *y, int len, int max_pitch) {
  int pitch = -12345;
  pitch_search(x_lp, (opus_val16 *)y, len, max_pitch, &pitch, 0);
  return pitch;
}

float oracle_compute_pitch_gain(float xy, float xx, float yy) {
  return compute_pitch_gain(xy, xx, yy);
}

float oracle_remove_doubling(const float *x, int maxperiod, int minperiod, int n, int *t0,
                             int prev_period, float prev_gain) {
  return remove_doubling((opus_val16 *)x, maxperiod, minperiod, n, t0, prev_period, prev_gain, 0);
}

void oracle_celt_lpc(float *lpc, const float *ac, int p) { _celt_lpc(lpc, ac, p); }

/* x points at `ord` history samples followed by the n input samples. */
void oracle_celt_fir(const float *x, const float *num, float *y, int n, int ord) {
  celt_fir_c(x + ord, num, y, n, ord, 0);
}

/* in_place != 0 filters y in place (x ignored), like the CELT decoder PLC does. */
void oracle_celt_iir(const float *x, const float *den, float *y, int n, int ord, float *mem,
                     int in_place) {
  celt_iir(in_place ? y : x, den, y, n, ord, mem, 0);
}

int oracle_celt_autocorr(const float *x, float *ac, const float *window, int overlap, int lag,
                         int n) {
  return _celt_autocorr(x, ac, overlap ? window : NULL, overlap, lag, n, 0);
}

/* Oracle C shims for unit celt_fft (kiss_fft.c, mdct.c, mini_kfft.c). */
#include <stdlib.h>
#include <string.h>
#include "opus_types.h"
#include "modes.h"
#include "kiss_fft.h"
#include "mdct.h"

/* Static mode for `fs` (48000, or 96000 with ENABLE_QEXT). */
static const CELTMode *oracle_celt_fft_mode(int fs) {
  if (fs == 96000) return opus_custom_mode_create(96000, 1920, NULL);
  return opus_custom_mode_create(48000, 960, NULL);
}

static void oracle_celt_fft_state_info(const kiss_fft_state *st, int *nfft, float *scale,
                                       int *shift, short *factors, short *bitrev) {
  int i;
  *nfft = st->nfft;
  *scale = st->scale;
  *shift = st->shift;
  for (i = 0; i < 2 * MAXFACTORS; i++) factors[i] = st->factors[i];
  for (i = 0; i < st->nfft; i++) bitrev[i] = st->bitrev[i];
}

/* kind: 0 opus_fft_c, 1 opus_ifft_c, 2 opus_fft_impl (fout := fin, then in place). */
static void oracle_celt_fft_run(const kiss_fft_state *st, int kind, const float *fin,
                                float *fout) {
  switch (kind) {
    case 0: opus_fft_c(st, (const kiss_fft_cpx *)fin, (kiss_fft_cpx *)fout); break;
    case 1: opus_ifft_c(st, (const kiss_fft_cpx *)fin, (kiss_fft_cpx *)fout); break;
    default:
      memcpy(fout, fin, sizeof(kiss_fft_cpx) * st->nfft);
      opus_fft_impl(st, (kiss_fft_cpx *)fout);
      break;
  }
}

/* dir: 0 forward, 1 backward. window NULL -> mode window. */
static void oracle_celt_mdct_run(const mdct_lookup *l, int dir, const float *in, float *out,
                                 const float *window, int overlap, int shift, int stride) {
  if (dir == 0)
    clt_mdct_forward_c(l, (float *)in, out, window, overlap, shift, stride, 0);
  else
    clt_mdct_backward_c(l, (float *)in, out, window, overlap, shift, stride, 0);
}

/* Info on static FFT state `idx` of the static mode at `fs`. bitrev must hold nfft values. */
void oracle_celt_fft_static_info(int fs, int idx, int *nfft, float *scale, int *shift,
                                 short *factors, short *bitrev) {
  const CELTMode *m = oracle_celt_fft_mode(fs);
  oracle_celt_fft_state_info(m->mdct.kfft[idx], nfft, scale, shift, factors, bitrev);
}

void oracle_celt_fft_static(int fs, int idx, int kind, const float *fin, float *fout) {
  const CELTMode *m = oracle_celt_fft_mode(fs);
  oracle_celt_fft_run(m->mdct.kfft[idx], kind, fin, fout);
}

void oracle_celt_mdct_static(int fs, int dir, const float *in, float *out, const float *window,
                             int overlap, int shift, int stride) {
  const CELTMode *m = oracle_celt_fft_mode(fs);
  if (window == NULL) window = m->window;
  oracle_celt_mdct_run(&m->mdct, dir, in, out, window, overlap, shift, stride);
}

#ifdef CUSTOM_MODES
/* opus_fft_alloc_twiddles(nfft, base) with base = opus_fft_alloc(base_nfft) if base_nfft > 0.
   Returns 0 when C returns NULL. twiddles (cap 2*max(nfft, base_nfft) floats) receives the
   state's twiddle table and *ntw its length in complex values. If kind >= 0 also runs
   oracle_celt_fft_run(kind, fin, fout). */
int oracle_celt_fft_alloc(int nfft, int base_nfft, float *scale, int *shift, short *factors,
                          short *bitrev, float *twiddles, int *ntw, int kind, const float *fin,
                          float *fout) {
  kiss_fft_state *base = NULL;
  kiss_fft_state *st;
  int i, n;
  if (base_nfft > 0) {
    base = opus_fft_alloc(base_nfft, NULL, NULL, 0);
    if (base == NULL) return 0;
  }
  st = opus_fft_alloc_twiddles(nfft, NULL, NULL, base, 0);
  if (st == NULL) {
    opus_fft_free(base, 0);
    return 0;
  }
  oracle_celt_fft_state_info(st, &n, scale, shift, factors, bitrev);
  *ntw = base != NULL ? base->nfft : nfft;
  for (i = 0; i < *ntw; i++) {
    twiddles[2 * i] = st->twiddles[i].r;
    twiddles[2 * i + 1] = st->twiddles[i].i;
  }
  if (kind >= 0) oracle_celt_fft_run(st, kind, fin, fout);
  opus_fft_free(st, 0);
  opus_fft_free(base, 0);
  return 1;
}

/* clt_mdct_init(n, maxshift). Returns 0 on failure. trig receives the trig table
   (n - (n/2 >> maxshift) floats), nffts[i] the FFT size of kfft[i], shifts[i] its shift.
   If dir >= 0 also runs oracle_celt_mdct_run on the lookup. */
int oracle_celt_mdct_init(int n, int maxshift, float *trig, int *nffts, int *shifts, int dir,
                          const float *in, float *out, const float *window, int overlap,
                          int shift, int stride) {
  mdct_lookup l;
  int i;
  memset(&l, 0, sizeof(l));
  if (!clt_mdct_init(&l, n, maxshift, 0)) return 0;
  for (i = 0; i < n - ((n >> 1) >> maxshift); i++) trig[i] = l.trig[i];
  for (i = 0; i <= maxshift; i++) {
    nffts[i] = l.kfft[i]->nfft;
    shifts[i] = l.kfft[i]->shift;
  }
  if (dir >= 0) oracle_celt_mdct_run(&l, dir, in, out, window, overlap, shift, stride);
  clt_mdct_clear(&l, 0);
  return 1;
}
#endif /* CUSTOM_MODES */

#ifdef ENABLE_QEXT
/* Layout-compatible re-declarations of the private types of celt/mini_kfft.c. */
typedef struct { float r; float i; } oracle_mini_cpx;
typedef struct {
  int nfft;
  int inverse;
  int factors[2 * 32];
  oracle_mini_cpx twiddles[1];
} oracle_mini_state;
typedef struct {
  oracle_mini_state *substate;
  oracle_mini_cpx *tmpbuf;
  oracle_mini_cpx *super_twiddles;
} oracle_mini_fftr_state;

oracle_mini_state *mini_kiss_fft_alloc(int nfft, int inverse_fft, void *mem, size_t *lenmem);
void mini_kiss_fft_stride(oracle_mini_state *st, const oracle_mini_cpx *fin,
                          oracle_mini_cpx *fout, int in_stride);
oracle_mini_fftr_state *mini_kiss_fftr_alloc(int nfft, int inverse_fft, void *mem,
                                             size_t *lenmem);
void mini_kiss_fftr(oracle_mini_fftr_state *st, const float *timedata, oracle_mini_cpx *freqdata);

/* mini_kiss_fft_alloc + mini_kiss_fft_stride. factors: 64 ints, twiddles: 2*nfft floats. */
void oracle_mini_kfft(int nfft, int inverse, const float *fin, float *fout, int in_stride,
                      int *factors, float *twiddles) {
  oracle_mini_state *st = mini_kiss_fft_alloc(nfft, inverse, NULL, NULL);
  int i;
  for (i = 0; i < 64; i++) factors[i] = st->factors[i];
  for (i = 0; i < nfft; i++) {
    twiddles[2 * i] = st->twiddles[i].r;
    twiddles[2 * i + 1] = st->twiddles[i].i;
  }
  mini_kiss_fft_stride(st, (const oracle_mini_cpx *)fin, (oracle_mini_cpx *)fout, in_stride);
  free(st);
}

/* mini_kiss_fftr_alloc(nfft, 0) + mini_kiss_fftr. freqdata: nfft/2+1 complex values,
   super_twiddles: nfft/4 complex values. */
void oracle_mini_kfftr(int nfft, const float *timedata, float *freqdata,
                       float *super_twiddles) {
  oracle_mini_fftr_state *st = mini_kiss_fftr_alloc(nfft, 0, NULL, NULL);
  int i;
  for (i = 0; i < (nfft >> 1) / 2; i++) {
    super_twiddles[2 * i] = st->super_twiddles[i].r;
    super_twiddles[2 * i + 1] = st->super_twiddles[i].i;
  }
  mini_kiss_fftr(st, timedata, (oracle_mini_cpx *)freqdata);
  free(st);
}
#endif /* ENABLE_QEXT */

/* Micro-benchmark shims over libopus internals (MDCT, FFT, range coder, SILK resampler).
 *
 * This file is compiled twice by build.rs, each time archived together with one libopus build
 * whose global symbols (including these bench_* entry points) then get a prefix:
 *  - with the oracle's configuration (scalar C, -O2 -ffp-contract=off, no intrinsics), next
 *    to a libopus copy built the same way: prefix `scalopus_`;
 *  - with the flags CMake used for the optimized Release build, next to that library:
 *    prefix `optopus_`.
 *
 * The shims only touch struct fields whose layout does not depend on the build configuration
 * (CELTMode up to `mdct`, mdct_lookup, kiss_fft_state, ec_ctx, silk_resampler_state_struct).
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include "opus_types.h"
#include "cpu_support.h"
#include "modes.h"
#include "kiss_fft.h"
#include "mdct.h"
#include "entenc.h"
#include "entdec.h"
#include "laplace.h"
#include "SigProc_FIX.h"

static const CELTMode *bench_mode48(void) { return opus_custom_mode_create(48000, 960, NULL); }

/* The architecture index the library's RTCD dispatch would use (always 0 without RTCD). */
int bench_arch(void) { return opus_select_arch(); }

/* Size of the 48 kHz mode's MDCT (N = 1920) and its overlap (120). */
int bench_mdct_n(void) { return bench_mode48()->mdct.n; }
int bench_mdct_overlap(void) { return bench_mode48()->overlap; }

/* Forward MDCT of the 48 kHz static mode: `in` holds N/2 + overlap samples, `out` N/2
   (N = 1920 >> shift), stride 1, the mode window. */
void bench_mdct_forward(const float *in, float *out, int shift, int arch) {
  const CELTMode *m = bench_mode48();
  clt_mdct_forward(&m->mdct, (float *)in, out, m->window, m->overlap, shift, 1, arch);
  (void)arch;
}

/* Backward MDCT of the 48 kHz static mode: `in` holds N/2 coefficients, `out` N/2 + overlap
   samples (the first overlap/2 are read as the previous overlap). */
void bench_mdct_backward(const float *in, float *out, int shift, int arch) {
  const CELTMode *m = bench_mode48();
  clt_mdct_backward(&m->mdct, (float *)in, out, m->window, m->overlap, shift, 1, arch);
  (void)arch;
}

/* FFT size of the 48 kHz mode's kfft[idx] (idx 0: 480 points). */
int bench_fft_nfft(int idx) { return bench_mode48()->mdct.kfft[idx]->nfft; }

/* Scaled forward FFT (opus_fft) with kfft[idx]; `fin`/`fout` interleaved complex. */
void bench_fft(int idx, const float *fin, float *fout, int arch) {
  const CELTMode *m = bench_mode48();
  opus_fft(m->mdct.kfft[idx], (const kiss_fft_cpx *)fin, (kiss_fft_cpx *)fout, arch);
  (void)arch;
}

/* Range coder ops, 4 words each: {kind, a, b, c}.
     0: symbol a of b equiprobable ones  (ec_encode / ec_decode + ec_dec_update)
     1: symbol a with the ICDF table below, ftb = 8  (ec_enc_icdf / ec_dec_icdf)
     2: bit a with logp b  (ec_enc_bit_logp / ec_dec_bit_logp)
     3: uint a of b  (ec_enc_uint / ec_dec_uint)
     4: a in b raw bits  (ec_enc_bits / ec_dec_bits)
     5: Laplace value (int)a with fs b, decay c  (ec_laplace_encode / ec_laplace_decode)
   Encodes all ops into buf[0..size), then decodes them from the same buffer. Returns a
   checksum over the final ranges and decoded values (identical in every implementation). */
static const unsigned char BENCH_ICDF[8] = {250, 200, 150, 100, 60, 30, 10, 0};

unsigned bench_ec_roundtrip(const unsigned *ops, int nops, unsigned char *buf, int size) {
  ec_enc enc;
  ec_dec dec;
  unsigned sum;
  int i;
  ec_enc_init(&enc, buf, size);
  for (i = 0; i < nops; i++) {
    const unsigned *o = ops + 4 * i;
    switch (o[0]) {
      case 0: ec_encode(&enc, o[1], o[1] + 1, o[2]); break;
      case 1: ec_enc_icdf(&enc, (int)o[1], BENCH_ICDF, 8); break;
      case 2: ec_enc_bit_logp(&enc, (int)o[1], o[2]); break;
      case 3: ec_enc_uint(&enc, o[1], o[2]); break;
      case 4: ec_enc_bits(&enc, o[1], o[2]); break;
      default: {
        int v = (int)o[1];
        ec_laplace_encode(&enc, &v, o[2], (int)o[3]);
        break;
      }
    }
  }
  ec_enc_done(&enc);
  sum = enc.rng ^ (unsigned)enc.error;
  ec_dec_init(&dec, buf, size);
  for (i = 0; i < nops; i++) {
    const unsigned *o = ops + 4 * i;
    unsigned v;
    switch (o[0]) {
      case 0:
        v = ec_decode(&dec, o[2]);
        ec_dec_update(&dec, v, v + 1, o[2]);
        break;
      case 1: v = (unsigned)ec_dec_icdf(&dec, BENCH_ICDF, 8); break;
      case 2: v = (unsigned)ec_dec_bit_logp(&dec, o[2]); break;
      case 3: v = ec_dec_uint(&dec, o[2]); break;
      case 4: v = ec_dec_bits(&dec, o[2]); break;
      default: v = (unsigned)ec_laplace_decode(&dec, o[2], (int)o[3]); break;
    }
    sum = sum * 31u + v;
  }
  return sum ^ dec.rng;
}

/* SILK resampler state on the heap (NULL on allocation or init failure). */
void *bench_resampler_new(int fs_in, int fs_out, int for_enc) {
  silk_resampler_state_struct *s = calloc(1, sizeof(*s));
  if (s == NULL) return NULL;
  if (silk_resampler_init(s, fs_in, fs_out, for_enc) != 0) {
    free(s);
    return NULL;
  }
  return s;
}

void bench_resampler_run(void *st, opus_int16 *out, const opus_int16 *in, int len) {
  silk_resampler((silk_resampler_state_struct *)st, out, in, len);
}

void bench_resampler_free(void *st) { free(st); }

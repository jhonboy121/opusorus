/* Oracle C shims for unit celt_modes: modes.c, laplace.c, cwrs.c, rate.c. */
#include <string.h>
#include "opus_types.h"
#include "opus_defines.h"
#include "modes.h"
#include "laplace.h"
#include "rate.h"
#include "entenc.h"
#include "entdec.h"

/* ------------------------------------------------------------------------------------------ */
/* Modes                                                                                        */
/* ------------------------------------------------------------------------------------------ */

const void *oracle_mode_create(int fs, int frame_size, int *err) {
  return opus_custom_mode_create(fs, frame_size, err);
}

void oracle_mode_destroy(const void *m) {
#ifdef CUSTOM_MODES
  opus_custom_mode_destroy((CELTMode *)m);
#else
  (void)m;
#endif
}

/* out[0..12]: Fs overlap nbEBands effEBands maxLM nbShortMdcts shortMdctSize nbAllocVectors
   mdct.n mdct.maxshift cache.size qext_cache.size (-1 when absent/NULL). */
void oracle_mode_ints(const void *mv, int *out) {
  const CELTMode *m = (const CELTMode *)mv;
  out[0] = m->Fs;
  out[1] = m->overlap;
  out[2] = m->nbEBands;
  out[3] = m->effEBands;
  out[4] = m->maxLM;
  out[5] = m->nbShortMdcts;
  out[6] = m->shortMdctSize;
  out[7] = m->nbAllocVectors;
  out[8] = m->mdct.n;
  out[9] = m->mdct.maxshift;
  out[10] = m->cache.size;
#ifdef ENABLE_QEXT
  out[11] = m->qext_cache.index ? m->qext_cache.size : -1;
#else
  out[11] = -1;
#endif
}

void oracle_mode_preemph(const void *mv, float *out) {
  const CELTMode *m = (const CELTMode *)mv;
  int i;
  for (i = 0; i < 4; i++) out[i] = m->preemph[i];
}

/* which: 0 eBands, 1 logN, 2 cache.index, 3 qext_cache.index */
void oracle_mode_i16(const void *mv, int which, short *out, int n) {
  const CELTMode *m = (const CELTMode *)mv;
  const opus_int16 *src = NULL;
  switch (which) {
    case 0: src = m->eBands; break;
    case 1: src = m->logN; break;
    case 2: src = m->cache.index; break;
#ifdef ENABLE_QEXT
    case 3: src = m->qext_cache.index; break;
#endif
  }
  if (src) memcpy(out, src, n * sizeof(*out));
}

/* which: 0 allocVectors, 1 cache.bits, 2 cache.caps, 3 qext_cache.bits, 4 qext_cache.caps */
void oracle_mode_u8(const void *mv, int which, unsigned char *out, int n) {
  const CELTMode *m = (const CELTMode *)mv;
  const unsigned char *src = NULL;
  switch (which) {
    case 0: src = m->allocVectors; break;
    case 1: src = m->cache.bits; break;
    case 2: src = m->cache.caps; break;
#ifdef ENABLE_QEXT
    case 3: src = m->qext_cache.bits; break;
    case 4: src = m->qext_cache.caps; break;
#endif
  }
  if (src) memcpy(out, src, n);
}

/* which: 0 window, 1 mdct.trig */
void oracle_mode_f32(const void *mv, int which, float *out, int n) {
  const CELTMode *m = (const CELTMode *)mv;
  const float *src = which == 0 ? m->window : m->mdct.trig;
  memcpy(out, src, n * sizeof(*out));
}

/* FFT state idx of the MDCT. ints: nfft shift; factors[16]; bitrev[nfft]; twiddles as
   (r,i) pairs, tw_n entries. */
void oracle_mode_fft(const void *mv, int idx, int *ints, float *scale, short *factors,
                     short *bitrev, float *twiddles, int tw_n) {
  const CELTMode *m = (const CELTMode *)mv;
  const kiss_fft_state *st = m->mdct.kfft[idx];
  int i;
  ints[0] = st->nfft;
  ints[1] = st->shift;
  *scale = st->scale;
  for (i = 0; i < 2 * MAXFACTORS; i++) factors[i] = st->factors[i];
  if (bitrev) memcpy(bitrev, st->bitrev, st->nfft * sizeof(*bitrev));
  if (twiddles) {
    for (i = 0; i < tw_n; i++) {
      twiddles[2 * i] = st->twiddles[i].r;
      twiddles[2 * i + 1] = st->twiddles[i].i;
    }
  }
}

/* ------------------------------------------------------------------------------------------ */
/* Laplace                                                                                      */
/* ------------------------------------------------------------------------------------------ */

/* ops[i*4..]: {kind (0 laplace, 1 p0), value, fs|p0, decay}. vals_out[i] = coded value
   (ec_laplace_encode may clamp). tells[i] = ec_tell_frac after op i. Returns error flag. */
int oracle_laplace_encode(const int *ops, int nops, unsigned char *buf, int size, int *vals_out,
                          unsigned *tells, unsigned *rng_out) {
  ec_enc enc;
  int i;
  ec_enc_init(&enc, buf, size);
  for (i = 0; i < nops; i++) {
    const int *o = ops + 4 * i;
    int v = o[1];
    if (o[0] == 0) {
      ec_laplace_encode(&enc, &v, (unsigned)o[2], o[3]);
    } else {
      ec_laplace_encode_p0(&enc, v, (opus_uint16)o[2], (opus_uint16)o[3]);
    }
    vals_out[i] = v;
    tells[i] = ec_tell_frac(&enc);
  }
  ec_enc_done(&enc);
  *rng_out = enc.rng;
  return enc.error;
}

/* ops[i*3..]: {kind, fs|p0, decay}. */
int oracle_laplace_decode(const int *ops, int nops, unsigned char *buf, int size, int *vals_out,
                          unsigned *tells, unsigned *rng_out) {
  ec_dec dec;
  int i;
  ec_dec_init(&dec, buf, size);
  for (i = 0; i < nops; i++) {
    const int *o = ops + 3 * i;
    if (o[0] == 0) {
      vals_out[i] = ec_laplace_decode(&dec, (unsigned)o[1], o[2]);
    } else {
      vals_out[i] = ec_laplace_decode_p0(&dec, (opus_uint16)o[1], (opus_uint16)o[2]);
    }
    tells[i] = ec_tell_frac(&dec);
  }
  *rng_out = dec.rng;
  return dec.error;
}

/* ------------------------------------------------------------------------------------------ */
/* rate.h inline helpers                                                                        */
/* ------------------------------------------------------------------------------------------ */

static const CELTMode *pick_mode(const void *mv, int use_qext, CELTMode *tmp) {
  const CELTMode *m = (const CELTMode *)mv;
#ifdef ENABLE_QEXT
  if (use_qext) {
    compute_qext_mode(tmp, m);
    return tmp;
  }
#else
  (void)use_qext;
  (void)tmp;
#endif
  return m;
}

int oracle_get_pulses(int i) { return get_pulses(i); }

int oracle_bits2pulses(const void *mv, int use_qext, int band, int LM, int bits) {
  CELTMode tmp;
  return bits2pulses(pick_mode(mv, use_qext, &tmp), band, LM, bits);
}

int oracle_pulses2bits(const void *mv, int use_qext, int band, int LM, int pulses) {
  CELTMode tmp;
  return pulses2bits(pick_mode(mv, use_qext, &tmp), band, LM, pulses);
}

/* compute_qext_mode: ints out {nbEBands effEBands nbAllocVectors cache.size}, eBands (15),
   logN (nbEBands). */
void oracle_compute_qext_mode(const void *mv, int *ints, short *ebands, short *logn) {
#ifdef ENABLE_QEXT
  CELTMode q;
  int i;
  compute_qext_mode(&q, (const CELTMode *)mv);
  ints[0] = q.nbEBands;
  ints[1] = q.effEBands;
  ints[2] = q.nbAllocVectors;
  ints[3] = q.cache.size;
  for (i = 0; i <= q.nbEBands; i++) ebands[i] = q.eBands[i];
  for (i = 0; i < q.nbEBands; i++) logn[i] = q.logN[i];
#else
  (void)mv; (void)ints; (void)ebands; (void)logn;
#endif
}

/* ------------------------------------------------------------------------------------------ */
/* rate.c                                                                                       */
/* ------------------------------------------------------------------------------------------ */

/* io[0] intensity, io[1] dual_stereo (in/out), io[2] balance (out), io[3] codedBands (out),
   io[4] final tell_frac (out), io[5] rng (out), io[6] error (out).
   When encode, `buf` receives the encoded bytes (after ec_enc_done); else it is decoded. */
void oracle_clt_compute_allocation(const void *mv, int use_qext, int start, int end,
                                   const int *offsets, const int *cap, int alloc_trim,
                                   int total, int *pulses, int *ebits, int *fine_priority,
                                   int C, int LM, unsigned char *buf, int size, int encode,
                                   int prev, int signalBandwidth, int *io) {
  CELTMode tmp;
  const CELTMode *m = pick_mode(mv, use_qext, &tmp);
  int intensity = io[0], dual_stereo = io[1];
  opus_int32 balance = 0;
  int coded;
  if (encode) {
    ec_enc enc;
    ec_enc_init(&enc, buf, size);
    coded = clt_compute_allocation(m, start, end, offsets, cap, alloc_trim, &intensity,
                                   &dual_stereo, total, &balance, pulses, ebits, fine_priority,
                                   C, LM, &enc, 1, prev, signalBandwidth);
    io[4] = (int)ec_tell_frac(&enc);
    ec_enc_done(&enc);
    io[5] = (int)enc.rng;
    io[6] = enc.error;
  } else {
    ec_dec dec;
    ec_dec_init(&dec, buf, size);
    coded = clt_compute_allocation(m, start, end, offsets, cap, alloc_trim, &intensity,
                                   &dual_stereo, total, &balance, pulses, ebits, fine_priority,
                                   C, LM, &dec, 0, prev, signalBandwidth);
    io[4] = (int)ec_tell_frac(&dec);
    io[5] = (int)dec.rng;
    io[6] = dec.error;
  }
  io[0] = intensity;
  io[1] = dual_stereo;
  io[2] = balance;
  io[3] = coded;
}

/* io[0] final tell_frac, io[1] rng, io[2] error (all out). */
void oracle_clt_compute_extra_allocation(const void *mv, int with_qext, int start, int end,
                                         int qext_end, const float *bandLogE,
                                         const float *qext_bandLogE, int total,
                                         int *extra_pulses, int *extra_equant, int C, int LM,
                                         unsigned char *buf, int size, int encode,
                                         float tone_freq, float toneishness, int *io) {
#ifdef ENABLE_QEXT
  const CELTMode *m = (const CELTMode *)mv;
  CELTMode q;
  const CELTMode *qm = NULL;
  if (with_qext) {
    compute_qext_mode(&q, m);
    qm = &q;
  }
  if (encode) {
    ec_enc enc;
    ec_enc_init(&enc, buf, size);
    clt_compute_extra_allocation(m, qm, start, end, qext_end, bandLogE, qext_bandLogE, total,
                                 extra_pulses, extra_equant, C, LM, &enc, 1, tone_freq,
                                 toneishness);
    io[0] = (int)ec_tell_frac(&enc);
    ec_enc_done(&enc);
    io[1] = (int)enc.rng;
    io[2] = enc.error;
  } else {
    ec_dec dec;
    ec_dec_init(&dec, buf, size);
    clt_compute_extra_allocation(m, qm, start, end, qext_end, bandLogE, qext_bandLogE, total,
                                 extra_pulses, extra_equant, C, LM, &dec, 0, tone_freq,
                                 toneishness);
    io[0] = (int)ec_tell_frac(&dec);
    io[1] = (int)dec.rng;
    io[2] = dec.error;
  }
#else
  (void)mv; (void)with_qext; (void)start; (void)end; (void)qext_end; (void)bandLogE;
  (void)qext_bandLogE; (void)total; (void)extra_pulses; (void)extra_equant; (void)C; (void)LM;
  (void)buf; (void)size; (void)encode; (void)tone_freq; (void)toneishness; (void)io;
#endif
}

#ifdef CUSTOM_MODES
int oracle_log2_frac(unsigned val, int frac) { return log2_frac(val, frac); }
void oracle_get_required_bits(short *bits, int n, int maxk, int frac) {
  get_required_bits(bits, n, maxk, frac);
}
#else
int oracle_log2_frac(unsigned val, int frac) { (void)val; (void)frac; return 0; }
void oracle_get_required_bits(short *bits, int n, int maxk, int frac) {
  (void)bits; (void)n; (void)maxk; (void)frac;
}
#endif

/* ------------------------------------------------------------------------------------------ */
/* cwrs.c                                                                                       */
/* ------------------------------------------------------------------------------------------ */

/* Encodes `count` pulse vectors; vector v has dimension ns[v] and ks[v] pulses and is stored
   at ys[offs(v)] (consecutive). */
int oracle_cwrs_encode(const int *ys, const int *ns, const int *ks, int count,
                       unsigned char *buf, int size, unsigned *rng_out) {
  ec_enc enc;
  int v, off = 0;
  ec_enc_init(&enc, buf, size);
  for (v = 0; v < count; v++) {
    encode_pulses(ys + off, ns[v], ks[v], &enc);
    off += ns[v];
  }
  ec_enc_done(&enc);
  *rng_out = enc.rng;
  return enc.error;
}

int oracle_cwrs_decode(const int *ns, const int *ks, int count, unsigned char *buf, int size,
                       int *ys_out, float *yy_out, unsigned *rng_out) {
  ec_dec dec;
  int v, off = 0;
  ec_dec_init(&dec, buf, size);
  for (v = 0; v < count; v++) {
    yy_out[v] = decode_pulses(ys_out + off, ns[v], ks[v], &dec);
    off += ns[v];
  }
  *rng_out = dec.rng;
  return dec.error;
}

/* The PVQ U/V tables are static in cwrs.c: include it with its external symbols renamed so the
   table is reachable from this translation unit without clashing with the library. */
#define log2_frac oracle_cwrs_copy_log2_frac
#define get_required_bits oracle_cwrs_copy_get_required_bits
#define encode_pulses oracle_cwrs_copy_encode_pulses
#define decode_pulses oracle_cwrs_copy_decode_pulses
#include "cwrs.c"

unsigned oracle_pvq_u(int n, int k) { return CELT_PVQ_U(n, k); }
unsigned oracle_pvq_v(int n, int k) { return CELT_PVQ_V(n, k); }

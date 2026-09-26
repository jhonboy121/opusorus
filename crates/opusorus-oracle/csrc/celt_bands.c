/* Oracle C shims for unit celt_bands: vq.c, quant_bands.c, bands.c, celt.c.
 *
 * Compiled in the float and in the fixed-point oracle: the shims use the libopus types
 * (celt_norm/celt_sig/celt_ener/celt_glog/opus_val16/opus_val32/celt_coef are float in the float
 * build, integers in the fixed-point build), mirrored by the type aliases of
 * opusorus-oracle/src/celt_bands.rs.
 *
 * Part A calls the library's public functions. Part B includes vq.c, bands.c and
 * quant_bands.c with their external symbols renamed so their static helpers can be reached
 * (the copies are compiled with the same flags as the library). */
// oracle-build: any
#include <string.h>
#include "opus_types.h"
#include "opus_defines.h"
#include "modes.h"
#include "rate.h"
#include "entenc.h"
#include "entdec.h"
#include "vq.h"
#include "bands.h"
#include "quant_bands.h"
#include "celt.h"
#include "pitch.h"

static const CELTMode *cb_mode(int fs, int qext, CELTMode *tmp) {
  const CELTMode *m = opus_custom_mode_create(fs, fs / 50, NULL);
#ifdef ENABLE_QEXT
  if (qext) {
    compute_qext_mode(tmp, m);
    return tmp;
  }
#else
  (void)qext;
  (void)tmp;
#endif
  return m;
}

int oracle_cb_has_qext(void) {
#ifdef ENABLE_QEXT
  return 1;
#else
  return 0;
#endif
}

/* ------------------------------------------------------------------------------------------ */
/* celt.c / celt.h                                                                              */
/* ------------------------------------------------------------------------------------------ */

int oracle_resampling_factor(int rate) { return resampling_factor(rate); }

void oracle_init_caps(int fs, int qext, int *cap, int LM, int C) {
  CELTMode tmp;
  init_caps(cb_mode(fs, qext, &tmp), cap, LM, C);
}

void oracle_celt_tables(signed char *tf, unsigned char *trim, unsigned char *spread,
                        unsigned char *tapset) {
  memcpy(tf, tf_select_table, 32);
  memcpy(trim, trim_icdf, 11);
  memcpy(spread, spread_icdf, 4);
  memcpy(tapset, tapset_icdf, 3);
}

/* In place when `inplace` (y ignored): filters x_base[x_off..x_off+N]. */
void oracle_comb_filter(opus_val32 *y, opus_val32 *x_base, int x_off, int inplace, int T0,
                        int T1, int N, opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
                        int fs, int use_window, int overlap) {
  CELTMode tmp;
  const CELTMode *m = cb_mode(fs, 0, &tmp);
  const celt_coef *w = use_window ? m->window : NULL;
  opus_val32 *x = x_base + x_off;
  comb_filter(inplace ? x : y, x, T0, T1, N, g0, g1, tapset0, tapset1, w, overlap, 0);
}

/* ------------------------------------------------------------------------------------------ */
/* vq.c                                                                                         */
/* ------------------------------------------------------------------------------------------ */

void oracle_exp_rotation(celt_norm *x, int len, int dir, int stride, int K, int spread) {
  exp_rotation(x, len, dir, stride, K, spread);
}

opus_val16 oracle_op_pvq_search(celt_norm *X, int *iy, int K, int N) {
  return op_pvq_search_c(X, iy, K, N, 0);
}

/* out: cm, tell_frac, rng, error, ext tell_frac, ext rng, ext error (after ec_enc_done). */
void oracle_alg_quant(celt_norm *X, int N, int K, int spread, int B, unsigned char *buf,
                      int size, opus_val32 gain, int resynth, unsigned char *ext_buf,
                      int ext_size, int extra_bits, unsigned *out) {
  ec_enc enc, ext;
  ec_enc_init(&enc, buf, size);
  ec_enc_init(&ext, ext_size ? ext_buf : NULL, ext_size);
#ifdef ENABLE_QEXT
  out[0] = alg_quant(X, N, K, spread, B, &enc, gain, resynth, &ext, extra_bits, 0);
#else
  (void)extra_bits;
  out[0] = alg_quant(X, N, K, spread, B, &enc, gain, resynth, 0);
#endif
  out[1] = ec_tell_frac(&enc);
  out[4] = ec_tell_frac(&ext);
  ec_enc_done(&enc);
  ec_enc_done(&ext);
  out[2] = enc.rng;
  out[3] = enc.error;
  out[5] = ext.rng;
  out[6] = ext.error;
}

void oracle_alg_unquant(celt_norm *X, int N, int K, int spread, int B, unsigned char *buf,
                        int size, opus_val32 gain, unsigned char *ext_buf, int ext_size,
                        int extra_bits, unsigned *out) {
  ec_dec dec, ext;
  ec_dec_init(&dec, buf, size);
  ec_dec_init(&ext, ext_size ? ext_buf : NULL, ext_size);
#ifdef ENABLE_QEXT
  out[0] = alg_unquant(X, N, K, spread, B, &dec, gain, &ext, extra_bits);
#else
  (void)extra_bits;
  out[0] = alg_unquant(X, N, K, spread, B, &dec, gain);
#endif
  out[1] = ec_tell_frac(&dec);
  out[2] = dec.rng;
  out[3] = dec.error;
  out[4] = ec_tell_frac(&ext);
  out[5] = ext.rng;
  out[6] = ext.error;
}

void oracle_renormalise_vector(celt_norm *X, int N, opus_val32 gain) {
  renormalise_vector(X, N, gain, 0);
}

int oracle_stereo_itheta(const celt_norm *X, const celt_norm *Y, int stereo, int N) {
  return stereo_itheta(X, Y, stereo, N, 0);
}

/* out: cm, tell_frac, rng, error. */
void oracle_cubic_quant(celt_norm *X, int N, int res, int B, unsigned char *buf, int size,
                        opus_val32 gain, int resynth, unsigned *out) {
#ifdef ENABLE_QEXT
  ec_enc enc;
  ec_enc_init(&enc, buf, size);
  out[0] = cubic_quant(X, N, res, B, &enc, gain, resynth);
  out[1] = ec_tell_frac(&enc);
  ec_enc_done(&enc);
  out[2] = enc.rng;
  out[3] = enc.error;
#else
  (void)X; (void)N; (void)res; (void)B; (void)buf; (void)size; (void)gain; (void)resynth;
  (void)out;
#endif
}

void oracle_cubic_unquant(celt_norm *X, int N, int res, int B, unsigned char *buf, int size,
                          opus_val32 gain, unsigned *out) {
#ifdef ENABLE_QEXT
  ec_dec dec;
  ec_dec_init(&dec, buf, size);
  out[0] = cubic_unquant(X, N, res, B, &dec, gain);
  out[1] = ec_tell_frac(&dec);
  out[2] = dec.rng;
  out[3] = dec.error;
#else
  (void)X; (void)N; (void)res; (void)B; (void)buf; (void)size; (void)gain; (void)out;
#endif
}

/* ------------------------------------------------------------------------------------------ */
/* bands.c                                                                                      */
/* ------------------------------------------------------------------------------------------ */

int oracle_hysteresis_decision(opus_val16 val, const opus_val16 *thresholds,
                               const opus_val16 *hysteresis, int N, int prev) {
  return hysteresis_decision(val, thresholds, hysteresis, N, prev);
}
unsigned oracle_celt_lcg_rand(unsigned seed) { return celt_lcg_rand(seed); }
int oracle_bitexact_cos(int x) { return bitexact_cos((opus_int16)x); }
int oracle_bitexact_log2tan(int isin, int icos) { return bitexact_log2tan(isin, icos); }

void oracle_compute_band_energies(int fs, int qext, const celt_sig *X, celt_ener *bandE,
                                  int end, int C, int LM) {
  CELTMode tmp;
  compute_band_energies(cb_mode(fs, qext, &tmp), X, bandE, end, C, LM, 0);
}

void oracle_normalise_bands(int fs, int qext, const celt_sig *freq, celt_norm *X,
                            const celt_ener *bandE, int end, int C, int M) {
  CELTMode tmp;
  normalise_bands(cb_mode(fs, qext, &tmp), freq, X, bandE, end, C, M);
}

void oracle_denormalise_bands(int fs, int qext, const celt_norm *X, celt_sig *freq,
                              const celt_glog *bandLogE, int start, int end, int M,
                              int downsample, int silence) {
  CELTMode tmp;
  denormalise_bands(cb_mode(fs, qext, &tmp), X, freq, bandLogE, start, end, M, downsample,
                    silence);
}

void oracle_anti_collapse(int fs, int qext, celt_norm *X, unsigned char *cm, int LM, int C,
                          int size, int start, int end, const celt_glog *logE,
                          const celt_glog *prev1, const celt_glog *prev2, const int *pulses,
                          unsigned seed, int encode) {
  CELTMode tmp;
  anti_collapse(cb_mode(fs, qext, &tmp), X, cm, LM, C, size, start, end, logE, prev1, prev2,
                pulses, seed, encode, 0);
}

/* io: average, hf_average, tapset_decision (in/out). */
int oracle_spreading_decision(int fs, const celt_norm *X, int *io, int last_decision,
                              int update_hf, int end, int C, int M, const int *spread_weight) {
  CELTMode tmp;
  return spreading_decision(cb_mode(fs, 0, &tmp), X, &io[0], last_decision, &io[1], &io[2],
                            update_hf, end, C, M, spread_weight);
}

void oracle_haar1(celt_norm *X, int N0, int stride) { haar1(X, N0, stride); }

enum {
  QP_ENCODE, QP_FS, QP_QEXTMODE, QP_START, QP_END, QP_C, QP_LM, QP_SHORT, QP_SPREAD, QP_DUAL,
  QP_INTENSITY, QP_TOTAL_BITS, QP_BALANCE, QP_CODED, QP_COMPLEXITY, QP_DISABLE_INV, QP_SIZE,
  QP_DO_ALLOC, QP_ALLOC_TRIM, QP_PREV, QP_SIGBW, QP_EXT_SIZE, QP_EXT_TOTAL, QP_HAS_CAP,
  QP_PREFIX_FT, QP_PREFIX_VAL, QP_COUNT
};

/* out: [0] codedBands [1] balance [2] intensity [3] dual_stereo [4] tell_frac after
   [5] rng [6] error [7] ext tell_frac [8] ext rng [9] ext error [10] tell_frac before. */
void oracle_quant_all_bands(const int *ip, celt_norm *X, unsigned char *collapse_masks,
                            const celt_ener *bandE, int *pulses, const int *tf_res,
                            const int *offsets, const int *cap, const int *extra_pulses,
                            unsigned char *buf, unsigned char *ext_buf, unsigned *seed,
                            int *out) {
  CELTMode tmp;
  const CELTMode *m = cb_mode(ip[QP_FS], ip[QP_QEXTMODE], &tmp);
  int encode = ip[QP_ENCODE];
  int C = ip[QP_C], LM = ip[QP_LM];
  int N = (1 << LM) * m->shortMdctSize;
  int intensity = ip[QP_INTENSITY], dual = ip[QP_DUAL];
  opus_int32 balance = ip[QP_BALANCE];
  int coded = ip[QP_CODED];
  int size = ip[QP_SIZE], ext_size = ip[QP_EXT_SIZE];
  int fine_quant[32], fine_priority[32];
  ec_ctx ec, ext;
  if (encode) {
    ec_enc_init(&ec, buf, size);
    ec_enc_init(&ext, ext_size ? ext_buf : NULL, ext_size);
    if (ip[QP_PREFIX_FT] > 1) ec_enc_uint(&ec, ip[QP_PREFIX_VAL], ip[QP_PREFIX_FT]);
  } else {
    ec_dec_init(&ec, buf, size);
    ec_dec_init(&ext, ext_size ? ext_buf : NULL, ext_size);
    if (ip[QP_PREFIX_FT] > 1) ec_dec_uint(&ec, ip[QP_PREFIX_FT]);
  }
  if (ip[QP_DO_ALLOC]) {
    opus_int32 bits = (((opus_int32)size * 8) << BITRES) - (opus_int32)ec_tell_frac(&ec) - 1;
    coded = clt_compute_allocation(m, ip[QP_START], ip[QP_END], offsets, cap,
                                   ip[QP_ALLOC_TRIM], &intensity, &dual, bits, &balance, pulses,
                                   fine_quant, fine_priority, C, LM, &ec, encode, ip[QP_PREV],
                                   ip[QP_SIGBW]);
  }
  out[10] = ec_tell_frac(&ec);
  quant_all_bands(encode, m, ip[QP_START], ip[QP_END], X, C == 2 ? X + N : NULL, collapse_masks,
                  bandE, pulses, ip[QP_SHORT], ip[QP_SPREAD], dual, intensity, (int *)tf_res,
                  ip[QP_TOTAL_BITS], balance, &ec, LM, coded, seed, ip[QP_COMPLEXITY], 0,
                  ip[QP_DISABLE_INV]
#ifdef ENABLE_QEXT
                  , &ext, (int *)extra_pulses, ip[QP_EXT_TOTAL], ip[QP_HAS_CAP] ? cap : NULL
#endif
  );
#ifndef ENABLE_QEXT
  (void)extra_pulses;
#endif
  out[0] = coded;
  out[1] = balance;
  out[2] = intensity;
  out[3] = dual;
  out[4] = ec_tell_frac(&ec);
  out[7] = ec_tell_frac(&ext);
  if (encode) {
    ec_enc_done(&ec);
    ec_enc_done(&ext);
  }
  out[5] = ec.rng;
  out[6] = ec.error;
  out[8] = ext.rng;
  out[9] = ext.error;
}

/* ------------------------------------------------------------------------------------------ */
/* quant_bands.c                                                                                */
/* ------------------------------------------------------------------------------------------ */

/* eMeans: `signed char` (Q4) in the fixed-point build, `opus_val16` (float) otherwise. */
#ifdef FIXED_POINT
void oracle_emeans(signed char *out) { memcpy(out, eMeans, sizeof(eMeans)); }
#else
void oracle_emeans(float *out) { memcpy(out, eMeans, sizeof(eMeans)); }
#endif

void oracle_amp2log2(int fs, int qext, int effEnd, int end, celt_ener *bandE,
                     celt_glog *bandLogE, int C) {
  CELTMode tmp;
  amp2Log2(cb_mode(fs, qext, &tmp), effEnd, end, bandE, bandLogE, C);
}

enum {
  QE_FS, QE_QEXT, QE_START, QE_END, QE_EFFEND, QE_C, QE_LM, QE_SIZE, QE_BUDGET, QE_NBAVAIL,
  QE_FORCE_INTRA, QE_TWO_PASS, QE_LOSS_RATE, QE_LFE, QE_PREFIX_FT, QE_PREFIX_VAL, QE_NULL_OLD,
  QE_COUNT
};

/* out: tell_frac after coarse, after fine, after finalise, rng, error. */
void oracle_quant_energy(const int *ip, const celt_glog *eBands, celt_glog *oldEBands,
                         celt_glog *error, unsigned char *buf, opus_val32 *delayedIntra,
                         int *fine_quant, int *prev_quant, int *fine_priority, int *out) {
  CELTMode tmp;
  const CELTMode *m = cb_mode(ip[QE_FS], ip[QE_QEXT], &tmp);
  int C = ip[QE_C];
  ec_enc enc;
  ec_enc_init(&enc, buf, ip[QE_SIZE]);
  if (ip[QE_PREFIX_FT] > 1) ec_enc_uint(&enc, ip[QE_PREFIX_VAL], ip[QE_PREFIX_FT]);
  quant_coarse_energy(m, ip[QE_START], ip[QE_END], ip[QE_EFFEND], eBands, oldEBands,
                      ip[QE_BUDGET], error, &enc, C, ip[QE_LM], ip[QE_NBAVAIL],
                      ip[QE_FORCE_INTRA], delayedIntra, ip[QE_TWO_PASS], ip[QE_LOSS_RATE],
                      ip[QE_LFE]);
  out[0] = ec_tell_frac(&enc);
  quant_fine_energy(m, ip[QE_START], ip[QE_END], oldEBands, error, prev_quant, fine_quant, &enc,
                    C);
  out[1] = ec_tell_frac(&enc);
  quant_energy_finalise(m, ip[QE_START], ip[QE_END], ip[QE_NULL_OLD] ? NULL : oldEBands, error,
                        fine_quant, fine_priority, ip[QE_SIZE] * 8 - ec_tell(&enc), &enc, C);
  out[2] = ec_tell_frac(&enc);
  ec_enc_done(&enc);
  out[3] = enc.rng;
  out[4] = enc.error;
}

/* ip: fs qext start end C LM size intra(-1: read flag) prefix_ft null_old.
   out: intra used, tell after coarse, after fine, after finalise, rng, error. */
void oracle_unquant_energy(const int *ip, celt_glog *oldEBands, unsigned char *buf,
                           int *fine_quant, int *prev_quant, int *fine_priority, int *out) {
  CELTMode tmp;
  const CELTMode *m = cb_mode(ip[0], ip[1], &tmp);
  int start = ip[2], end = ip[3], C = ip[4], LM = ip[5], size = ip[6];
  int intra = ip[7];
  ec_dec dec;
  ec_dec_init(&dec, buf, size);
  if (ip[8] > 1) ec_dec_uint(&dec, ip[8]);
  if (intra < 0) intra = ec_tell(&dec) + 3 <= size * 8 ? ec_dec_bit_logp(&dec, 3) : 0;
  out[0] = intra;
  unquant_coarse_energy(m, start, end, oldEBands, intra, &dec, C, LM);
  out[1] = ec_tell_frac(&dec);
  unquant_fine_energy(m, start, end, oldEBands, prev_quant, fine_quant, &dec, C);
  out[2] = ec_tell_frac(&dec);
  unquant_energy_finalise(m, start, end, ip[9] ? NULL : oldEBands, fine_quant, fine_priority,
                          size * 8 - ec_tell(&dec), &dec, C);
  out[3] = ec_tell_frac(&dec);
  out[4] = dec.rng;
  out[5] = dec.error;
}

/* The library's celt_inner_prod_norm / celt_inner_prod_norm_shift (celt_inner_prod in the float
   build). */
opus_val32 oracle_celt_inner_prod_norm(const celt_norm *x, const celt_norm *y, int len) {
  return celt_inner_prod_norm(x, y, len, 0);
}
opus_val32 oracle_celt_inner_prod_norm_shift(const celt_norm *x, const celt_norm *y, int len) {
  return celt_inner_prod_norm_shift(x, y, len, 0);
}

/* ------------------------------------------------------------------------------------------ */
/* Part B: static helpers (renamed copies of vq.c, bands.c, quant_bands.c)                     */
/* ------------------------------------------------------------------------------------------ */

#ifdef FIXED_POINT
/* Functions in the fixed-point build (macros in the float build). */
#define norm_scaleup oracle_copy_norm_scaleup
#define norm_scaledown oracle_copy_norm_scaledown
#define celt_inner_prod_norm oracle_copy_celt_inner_prod_norm
#define celt_inner_prod_norm_shift oracle_copy_celt_inner_prod_norm_shift
#endif
#define exp_rotation oracle_copy_exp_rotation
#define op_pvq_search_c oracle_copy_op_pvq_search_c
#define alg_quant oracle_copy_alg_quant
#define alg_unquant oracle_copy_alg_unquant
#define renormalise_vector oracle_copy_renormalise_vector
#define stereo_itheta oracle_copy_stereo_itheta
#define cubic_quant oracle_copy_cubic_quant
#define cubic_unquant oracle_copy_cubic_unquant
#include "vq.c"

#define hysteresis_decision oracle_copy_hysteresis_decision
#define celt_lcg_rand oracle_copy_celt_lcg_rand
#define bitexact_cos oracle_copy_bitexact_cos
#define bitexact_log2tan oracle_copy_bitexact_log2tan
#define compute_band_energies oracle_copy_compute_band_energies
#define normalise_bands oracle_copy_normalise_bands
#define denormalise_bands oracle_copy_denormalise_bands
#define anti_collapse oracle_copy_anti_collapse
#define spreading_decision oracle_copy_spreading_decision
#define haar1 oracle_copy_haar1
#define quant_all_bands oracle_copy_quant_all_bands
#include "bands.c"

#define eMeans oracle_copy_eMeans
#define amp2Log2 oracle_copy_amp2Log2
#define quant_coarse_energy oracle_copy_quant_coarse_energy
#define quant_fine_energy oracle_copy_quant_fine_energy
#define quant_energy_finalise oracle_copy_quant_energy_finalise
#define unquant_coarse_energy oracle_copy_unquant_coarse_energy
#define unquant_fine_energy oracle_copy_unquant_fine_energy
#define unquant_energy_finalise oracle_copy_unquant_energy_finalise
#include "quant_bands.c"

void oracle_exp_rotation1(celt_norm *X, int len, int stride, opus_val16 c, opus_val16 s) {
  exp_rotation1(X, len, stride, c, s);
}
void oracle_normalise_residual(int *iy, celt_norm *X, int N, opus_val32 Ryy, opus_val32 gain,
                               int shift) {
  normalise_residual(iy, X, N, Ryy, gain, shift);
}
unsigned oracle_extract_collapse_mask(int *iy, int N, int B) {
  return extract_collapse_mask(iy, N, B);
}

opus_val32 oracle_op_pvq_search_n2(const celt_norm *X, int *iy, int *up_iy, int K, int up,
                                   int *refine, int shift) {
#ifdef ENABLE_QEXT
  return op_pvq_search_N2(X, iy, up_iy, K, up, refine, shift);
#else
  (void)X; (void)iy; (void)up_iy; (void)K; (void)up; (void)refine; (void)shift;
  return 0;
#endif
}

opus_val32 oracle_op_pvq_search_extra(const celt_norm *X, int *iy, int *up_iy, int K, int up,
                                      int *refine, int N, int shift) {
#ifdef ENABLE_QEXT
  return op_pvq_search_extra(X, iy, up_iy, K, up, refine, N, shift);
#else
  (void)X; (void)iy; (void)up_iy; (void)K; (void)up; (void)refine; (void)N; (void)shift;
  return 0;
#endif
}

int oracle_compute_qn(int N, int b, int offset, int pulse_cap, int stereo) {
  return compute_qn(N, b, offset, pulse_cap, stereo);
}

void oracle_compute_channel_weights(celt_ener Ex, celt_ener Ey, opus_val16 *w) {
  compute_channel_weights(Ex, Ey, w);
}

void oracle_intensity_stereo(int fs, celt_norm *X, const celt_norm *Y, const celt_ener *bandE,
                             int bandID, int N) {
  CELTMode tmp;
  intensity_stereo(cb_mode(fs, 0, &tmp), X, Y, bandE, bandID, N);
}

void oracle_stereo_split(celt_norm *X, celt_norm *Y, int N) { stereo_split(X, Y, N); }

void oracle_stereo_merge(celt_norm *X, celt_norm *Y, opus_val32 mid, int N) {
  stereo_merge(X, Y, mid, N, 0);
}

void oracle_deinterleave_hadamard(celt_norm *X, int N0, int stride, int hadamard) {
  deinterleave_hadamard(X, N0, stride, hadamard);
}

void oracle_interleave_hadamard(celt_norm *X, int N0, int stride, int hadamard) {
  interleave_hadamard(X, N0, stride, hadamard);
}

opus_val32 oracle_loss_distortion(const celt_glog *eBands, celt_glog *oldEBands, int start,
                                  int end, int len, int C) {
  return loss_distortion(eBands, oldEBands, start, end, len, C);
}

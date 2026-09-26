/* Oracle shims for unit silk_common: SILK tables, structs, signal-processing helpers, NLSF code
   and entropy-coding helpers. Every shim takes flat arrays + ints. */
#include <string.h>
#include "main.h"
#include "tables.h"
#include "pitch_est_defines.h"
#include "entenc.h"
#include "entdec.h"

/* silk_inner_prod16_c lives in silk/fixed/vector_ops_FIX.c, which the float oracle does not
   compile. Include it here with every global renamed so it cannot clash with other shims. */
#define silk_scale_copy_vector16 oracle_sc_silk_scale_copy_vector16
#define silk_scale_vector32_Q26_lshift_18 oracle_sc_silk_scale_vector32_Q26_lshift_18
#define silk_inner_prod_aligned oracle_sc_silk_inner_prod_aligned
#define silk_inner_prod16_c oracle_sc_silk_inner_prod16_c
#include "fixed/vector_ops_FIX.c"
#undef silk_scale_copy_vector16
#undef silk_scale_vector32_Q26_lshift_18
#undef silk_inner_prod_aligned
#undef silk_inner_prod16_c

/* ---------------------------------------------------------------------------------------- */
/* Tables                                                                                   */
/* ---------------------------------------------------------------------------------------- */

#define T(id, arr, n) case id: src = (const void *)(arr); cnt = (n); break;

/* Copies u8 table `id` into out (at most cap entries); returns its length or -1. */
int oracle_silk_table_u8(int id, unsigned char *out, int cap) {
  const void *src = NULL;
  int cnt = -1;
  switch (id) {
    T(0, silk_gain_iCDF, 24)
    T(1, silk_delta_gain_iCDF, 41)
    T(2, silk_pitch_lag_iCDF, 32)
    T(3, silk_pitch_delta_iCDF, 21)
    T(4, silk_pitch_contour_iCDF, 34)
    T(5, silk_pitch_contour_NB_iCDF, 11)
    T(6, silk_pitch_contour_10_ms_iCDF, 12)
    T(7, silk_pitch_contour_10_ms_NB_iCDF, 3)
    T(8, silk_pulses_per_block_iCDF, 180)
    T(9, silk_pulses_per_block_BITS_Q5, 162)
    T(10, silk_rate_levels_iCDF, 18)
    T(11, silk_rate_levels_BITS_Q5, 18)
    T(12, silk_max_pulses_table, 4)
    T(13, silk_shell_code_table0, 152)
    T(14, silk_shell_code_table1, 152)
    T(15, silk_shell_code_table2, 152)
    T(16, silk_shell_code_table3, 152)
    T(17, silk_shell_code_table_offsets, 17)
    T(18, silk_lsb_iCDF, 2)
    T(19, silk_sign_iCDF, 42)
    T(20, silk_uniform3_iCDF, 3)
    T(21, silk_uniform4_iCDF, 4)
    T(22, silk_uniform5_iCDF, 5)
    T(23, silk_uniform6_iCDF, 6)
    T(24, silk_uniform8_iCDF, 8)
    T(25, silk_NLSF_EXT_iCDF, 7)
    T(26, silk_LTP_per_index_iCDF, 3)
    T(27, silk_LTP_gain_iCDF_ptrs[0], 8)
    T(28, silk_LTP_gain_iCDF_ptrs[1], 16)
    T(29, silk_LTP_gain_iCDF_ptrs[2], 32)
    T(30, silk_LTP_gain_BITS_Q5_ptrs[0], 8)
    T(31, silk_LTP_gain_BITS_Q5_ptrs[1], 16)
    T(32, silk_LTP_gain_BITS_Q5_ptrs[2], 32)
    T(33, silk_LTP_vq_gain_ptrs_Q7[0], 8)
    T(34, silk_LTP_vq_gain_ptrs_Q7[1], 16)
    T(35, silk_LTP_vq_gain_ptrs_Q7[2], 32)
    T(36, silk_LTPscale_iCDF, 3)
    T(37, silk_type_offset_VAD_iCDF, 4)
    T(38, silk_type_offset_no_VAD_iCDF, 2)
    T(39, silk_stereo_pred_joint_iCDF, 25)
    T(40, silk_stereo_only_code_mid_iCDF, 2)
    T(41, silk_LBRR_flags_iCDF_ptr[0], 3)
    T(42, silk_LBRR_flags_iCDF_ptr[1], 7)
    T(43, silk_NLSF_interpolation_factor_iCDF, 5)
    T(44, silk_NLSF_CB_NB_MB.CB1_NLSF_Q8, 320)
    T(45, silk_NLSF_CB_NB_MB.CB1_iCDF, 64)
    T(46, silk_NLSF_CB_NB_MB.pred_Q8, 18)
    T(47, silk_NLSF_CB_NB_MB.ec_sel, 160)
    T(48, silk_NLSF_CB_NB_MB.ec_iCDF, 72)
    T(49, silk_NLSF_CB_NB_MB.ec_Rates_Q5, 72)
    T(50, silk_NLSF_CB_WB.CB1_NLSF_Q8, 512)
    T(51, silk_NLSF_CB_WB.CB1_iCDF, 64)
    T(52, silk_NLSF_CB_WB.pred_Q8, 30)
    T(53, silk_NLSF_CB_WB.ec_sel, 256)
    T(54, silk_NLSF_CB_WB.ec_iCDF, 72)
    T(55, silk_NLSF_CB_WB.ec_Rates_Q5, 72)
    default: return -1;
  }
  if (cnt > cap) return -1;
  memcpy(out, src, cnt);
  return cnt;
}

int oracle_silk_table_i8(int id, signed char *out, int cap) {
  const void *src = NULL;
  int cnt = -1;
  switch (id) {
    T(0, silk_LTP_vq_ptrs_Q7[0], 8 * 5)
    T(1, silk_LTP_vq_ptrs_Q7[1], 16 * 5)
    T(2, silk_LTP_vq_ptrs_Q7[2], 32 * 5)
    T(3, silk_LTP_vq_sizes, 3)
    T(4, silk_CB_lags_stage2, PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE2_EXT)
    T(5, silk_CB_lags_stage3, PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX)
    T(6, silk_Lag_range_stage3, (SILK_PE_MAX_COMPLEX + 1) * PE_MAX_NB_SUBFR * 2)
    T(7, silk_nb_cbk_searchs_stage3, SILK_PE_MAX_COMPLEX + 1)
    T(8, silk_CB_lags_stage2_10_ms, (PE_MAX_NB_SUBFR >> 1) * PE_NB_CBKS_STAGE2_10MS)
    T(9, silk_CB_lags_stage3_10_ms, (PE_MAX_NB_SUBFR >> 1) * PE_NB_CBKS_STAGE3_10MS)
    T(10, silk_Lag_range_stage3_10_ms, (PE_MAX_NB_SUBFR >> 1) * 2)
    default: return -1;
  }
  if (cnt > cap) return -1;
  memcpy(out, src, cnt);
  return cnt;
}

int oracle_silk_table_i16(int id, opus_int16 *out, int cap) {
  const void *src = NULL;
  int cnt = -1;
  opus_int16 scal[4];
  switch (id) {
    T(0, silk_LTPScales_table_Q14, 3)
    T(1, silk_stereo_pred_quant_Q13, STEREO_QUANT_TAB_SIZE)
    T(2, silk_Quantization_Offsets_Q10, 4)
    T(3, silk_LSFCosTab_FIX_Q12, LSF_COS_TAB_SZ_FIX + 1)
    T(4, silk_NLSF_CB_NB_MB.CB1_Wght_Q9, 320)
    T(5, silk_NLSF_CB_NB_MB.deltaMin_Q15, 11)
    T(6, silk_NLSF_CB_WB.CB1_Wght_Q9, 512)
    T(7, silk_NLSF_CB_WB.deltaMin_Q15, 17)
    case 8:
    case 9: {
      const silk_NLSF_CB_struct *cb = id == 8 ? &silk_NLSF_CB_NB_MB : &silk_NLSF_CB_WB;
      scal[0] = cb->nVectors;
      scal[1] = cb->order;
      scal[2] = cb->quantStepSize_Q16;
      scal[3] = cb->invQuantStepSize_Q6;
      src = scal;
      cnt = 4;
      break;
    }
    default: return -1;
  }
  if (cnt > cap) return -1;
  memcpy(out, src, cnt * sizeof(opus_int16));
  return cnt;
}

int oracle_silk_table_i32(int id, opus_int32 *out, int cap) {
  const void *src = NULL;
  int cnt = -1;
  switch (id) {
    T(0, silk_Transition_LP_B_Q28, TRANSITION_INT_NUM * TRANSITION_NB)
    T(1, silk_Transition_LP_A_Q28, TRANSITION_INT_NUM * TRANSITION_NA)
    default: return -1;
  }
  if (cnt > cap) return -1;
  memcpy(out, src, cnt * sizeof(opus_int32));
  return cnt;
}
#undef T

/* Array lengths (in elements) of every array field of the SILK structs, in a fixed order that
   the Rust test mirrors. Returns the count. */
#define D(s, f) out[n++] = (int)(sizeof(((s *)0)->f) / sizeof(((s *)0)->f[0]));
int oracle_silk_struct_dims(int *out) {
  int n = 0;
  D(silk_nsq_state, xq)
  D(silk_nsq_state, sLTP_shp_Q14)
  D(silk_nsq_state, sLPC_Q14)
  D(silk_nsq_state, sAR2_Q14)
  D(silk_VAD_state, AnaState)
  D(silk_VAD_state, AnaState1)
  D(silk_VAD_state, AnaState2)
  D(silk_VAD_state, XnrgSubfr)
  D(silk_VAD_state, NrgRatioSmth_Q8)
  D(silk_VAD_state, NL)
  D(silk_VAD_state, inv_NL)
  D(silk_VAD_state, NoiseLevelBias)
  D(silk_LP_state, In_LP_State)
  D(stereo_enc_state, pred_prev_Q13)
  D(stereo_enc_state, sMid)
  D(stereo_enc_state, sSide)
  D(stereo_enc_state, mid_side_amp_Q0)
  D(stereo_enc_state, predIx)
  D(stereo_enc_state, mid_only_flags)
  D(stereo_dec_state, pred_prev_Q13)
  D(stereo_dec_state, sMid)
  D(stereo_dec_state, sSide)
  D(SideInfoIndices, GainsIndices)
  D(SideInfoIndices, LTPIndex)
  D(SideInfoIndices, NLSFIndices)
  D(silk_encoder_state, In_HP_State)
  D(silk_encoder_state, prev_NLSFq_Q15)
  D(silk_encoder_state, input_quality_bands_Q15)
  D(silk_encoder_state, VAD_flags)
  D(silk_encoder_state, LBRR_flags)
  D(silk_encoder_state, pulses)
  D(silk_encoder_state, inputBuf)
  D(silk_encoder_state, indices_LBRR)
  D(silk_encoder_state, pulses_LBRR)
  D(silk_encoder_state, pulses_LBRR[0])
  D(silk_PLC_struct, LTPCoef_Q14)
  D(silk_PLC_struct, prevLPC_Q12)
  D(silk_PLC_struct, prevGain_Q16)
  D(silk_CNG_struct, CNG_exc_buf_Q14)
  D(silk_CNG_struct, CNG_smth_NLSF_Q15)
  D(silk_CNG_struct, CNG_synth_state)
  D(silk_decoder_state, exc_Q14)
  D(silk_decoder_state, sLPC_Q14_buf)
  D(silk_decoder_state, outBuf)
  D(silk_decoder_state, prevNLSF_Q15)
  D(silk_decoder_state, VAD_flags)
  D(silk_decoder_state, LBRR_flags)
  D(silk_decoder_control, pitchL)
  D(silk_decoder_control, Gains_Q16)
  D(silk_decoder_control, PredCoef_Q12)
  D(silk_decoder_control, PredCoef_Q12[0])
  D(silk_decoder_control, LTPCoef_Q14)
  return n;
}
#undef D

/* ---------------------------------------------------------------------------------------- */
/* Signal processing                                                                        */
/* ---------------------------------------------------------------------------------------- */

opus_int32 oracle_silk_lin2log(opus_int32 x) { return silk_lin2log(x); }
opus_int32 oracle_silk_log2lin(opus_int32 x) { return silk_log2lin(x); }
int oracle_silk_sigm_Q15(int x) { return silk_sigm_Q15(x); }

void oracle_silk_insertion_sort_increasing(opus_int32 *a, int *idx, int L, int K) {
  silk_insertion_sort_increasing(a, idx, L, K);
}
void oracle_silk_insertion_sort_increasing_all_values_int16(opus_int16 *a, int L) {
  silk_insertion_sort_increasing_all_values_int16(a, L);
}
void oracle_silk_bwexpander(opus_int16 *ar, int d, opus_int32 chirp) { silk_bwexpander(ar, d, chirp); }
void oracle_silk_bwexpander_32(opus_int32 *ar, int d, opus_int32 chirp) {
  silk_bwexpander_32(ar, d, chirp);
}
opus_int32 oracle_silk_inner_prod_aligned_scale(const opus_int16 *a, const opus_int16 *b, int scale,
                                                int len) {
  return silk_inner_prod_aligned_scale(a, b, scale, len);
}
long long oracle_silk_inner_prod16(const opus_int16 *a, const opus_int16 *b, int len) {
  return oracle_sc_silk_inner_prod16_c(a, b, len);
}
void oracle_silk_sum_sqr_shift(opus_int32 *energy, int *shift, const opus_int16 *x, int len) {
  silk_sum_sqr_shift(energy, shift, x, len);
}
void oracle_silk_interpolate(opus_int16 *xi, const opus_int16 *x0, const opus_int16 *x1, int ifact,
                             int d) {
  silk_interpolate(xi, x0, x1, ifact, d);
}
void oracle_silk_biquad_alt_stride1(const opus_int16 *in, const opus_int32 *B, const opus_int32 *A,
                                    opus_int32 *S, opus_int16 *out, int len) {
  silk_biquad_alt_stride1(in, B, A, S, out, len);
}
void oracle_silk_biquad_alt_stride2(const opus_int16 *in, const opus_int32 *B, const opus_int32 *A,
                                    opus_int32 *S, opus_int16 *out, int len) {
  silk_biquad_alt_stride2_c(in, B, A, S, out, len);
}
/* st = {In_LP_State[0], In_LP_State[1], transition_frame_no, mode, saved_fs_kHz} (in/out). */
void oracle_silk_LP_variable_cutoff(opus_int32 *st, opus_int16 *frame, int len) {
  silk_LP_state lp;
  lp.In_LP_State[0] = st[0];
  lp.In_LP_State[1] = st[1];
  lp.transition_frame_no = st[2];
  lp.mode = st[3];
  lp.saved_fs_kHz = st[4];
  silk_LP_variable_cutoff(&lp, frame, len);
  st[0] = lp.In_LP_State[0];
  st[1] = lp.In_LP_State[1];
  st[2] = lp.transition_frame_no;
  st[3] = lp.mode;
  st[4] = lp.saved_fs_kHz;
}
void oracle_silk_ana_filt_bank_1(const opus_int16 *in, opus_int32 *S, opus_int16 *outL,
                                 opus_int16 *outH, int N) {
  silk_ana_filt_bank_1(in, S, outL, outH, N);
}
opus_int32 oracle_silk_LPC_inverse_pred_gain(const opus_int16 *A, int order) {
  return silk_LPC_inverse_pred_gain_c(A, order);
}
void oracle_silk_LPC_fit(opus_int16 *a_QOUT, opus_int32 *a_QIN, int QOUT, int QIN, int d) {
  silk_LPC_fit(a_QOUT, a_QIN, QOUT, QIN, d);
}
void oracle_silk_LPC_analysis_filter(opus_int16 *out, const opus_int16 *in, const opus_int16 *B,
                                     int len, int d) {
  silk_LPC_analysis_filter(out, in, B, len, d, 0);
}

/* ---------------------------------------------------------------------------------------- */
/* NLSF                                                                                     */
/* ---------------------------------------------------------------------------------------- */

static const silk_NLSF_CB_struct *cb_of(int wb) { return wb ? &silk_NLSF_CB_WB : &silk_NLSF_CB_NB_MB; }

void oracle_silk_NLSF2A(opus_int16 *a_Q12, const opus_int16 *NLSF, int d) {
  silk_NLSF2A(a_Q12, NLSF, d, 0);
}
void oracle_silk_A2NLSF(opus_int16 *NLSF, opus_int32 *a_Q16, int d) { silk_A2NLSF(NLSF, a_Q16, d); }
void oracle_silk_NLSF_decode(opus_int16 *pNLSF_Q15, opus_int8 *NLSFIndices, int wb) {
  silk_NLSF_decode(pNLSF_Q15, NLSFIndices, cb_of(wb));
}
void oracle_silk_NLSF_unpack(opus_int16 *ec_ix, opus_uint8 *pred_Q8, int wb, int CB1_index) {
  silk_NLSF_unpack(ec_ix, pred_Q8, cb_of(wb), CB1_index);
}
void oracle_silk_NLSF_stabilize(opus_int16 *NLSF_Q15, const opus_int16 *NDeltaMin_Q15, int L) {
  silk_NLSF_stabilize(NLSF_Q15, NDeltaMin_Q15, L);
}
void oracle_silk_NLSF_VQ_weights_laroia(opus_int16 *out, const opus_int16 *NLSF, int D) {
  silk_NLSF_VQ_weights_laroia(out, NLSF, D);
}
void oracle_silk_NLSF_VQ(opus_int32 *err_Q24, const opus_int16 *in_Q15, const opus_uint8 *pCB_Q8,
                         const opus_int16 *pWght_Q9, int K, int LPC_order) {
  silk_NLSF_VQ(err_Q24, in_Q15, pCB_Q8, pWght_Q9, K, LPC_order);
}
opus_int32 oracle_silk_NLSF_encode(opus_int8 *NLSFIndices, opus_int16 *pNLSF_Q15, int wb,
                                   const opus_int16 *pW_Q2, int NLSF_mu_Q20, int nSurvivors,
                                   int signalType) {
  return silk_NLSF_encode(NLSFIndices, pNLSF_Q15, cb_of(wb), pW_Q2, NLSF_mu_Q20, nSurvivors,
                          signalType);
}
opus_int32 oracle_silk_NLSF_del_dec_quant(opus_int8 *indices, const opus_int16 *x_Q10,
                                          const opus_int16 *w_Q5, const opus_uint8 *pred_coef_Q8,
                                          const opus_int16 *ec_ix, const opus_uint8 *ec_rates_Q5,
                                          int quant_step_size_Q16, int inv_quant_step_size_Q6,
                                          int mu_Q20, int order) {
  return silk_NLSF_del_dec_quant(indices, x_Q10, w_Q5, pred_coef_Q8, ec_ix, ec_rates_Q5,
                                 quant_step_size_Q16, (opus_int16)inv_quant_step_size_Q6, mu_Q20,
                                 (opus_int16)order);
}

/* ---------------------------------------------------------------------------------------- */
/* Entropy coding helpers                                                                   */
/* ---------------------------------------------------------------------------------------- */

/* Encoder harness: `pre_n` raw bits `pre` are coded first (to vary the coder state); after the
   operation res = {tell_frac before done, rng after done, error}. */
static void enc_begin(ec_enc *enc, unsigned char *buf, int size, unsigned pre, int pre_n) {
  ec_enc_init(enc, buf, size);
  if (pre_n > 0) ec_enc_bits(enc, pre, pre_n);
}
static void enc_end(ec_enc *enc, opus_uint32 *res) {
  res[0] = ec_tell_frac(enc);
  ec_enc_done(enc);
  res[1] = enc->rng;
  res[2] = (opus_uint32)enc->error;
}
static void dec_begin(ec_dec *dec, unsigned char *buf, int size, int pre_n) {
  ec_dec_init(dec, buf, size);
  if (pre_n > 0) (void)ec_dec_bits(dec, pre_n);
}
static void dec_end(ec_dec *dec, opus_uint32 *res) {
  res[0] = ec_tell_frac(dec);
  res[1] = dec->rng;
  res[2] = (opus_uint32)dec->error;
}

void oracle_silk_encode_signs(unsigned char *buf, int size, unsigned pre, int pre_n,
                              const opus_int8 *pulses, int length, int signalType,
                              int quantOffsetType, const int *sum_pulses, opus_uint32 *res) {
  ec_enc enc;
  enc_begin(&enc, buf, size, pre, pre_n);
  silk_encode_signs(&enc, pulses, length, signalType, quantOffsetType, sum_pulses);
  enc_end(&enc, res);
}
void oracle_silk_decode_signs(unsigned char *buf, int size, int pre_n, opus_int16 *pulses,
                              int length, int signalType, int quantOffsetType,
                              const int *sum_pulses, opus_uint32 *res) {
  ec_dec dec;
  dec_begin(&dec, buf, size, pre_n);
  silk_decode_signs(&dec, pulses, length, signalType, quantOffsetType, sum_pulses);
  dec_end(&dec, res);
}
void oracle_silk_shell_encoder(unsigned char *buf, int size, unsigned pre, int pre_n,
                               const int *pulses0, opus_uint32 *res) {
  ec_enc enc;
  enc_begin(&enc, buf, size, pre, pre_n);
  silk_shell_encoder(&enc, pulses0);
  enc_end(&enc, res);
}
void oracle_silk_shell_decoder(unsigned char *buf, int size, int pre_n, opus_int16 *pulses0,
                               int pulses4, opus_uint32 *res) {
  ec_dec dec;
  dec_begin(&dec, buf, size, pre_n);
  silk_shell_decoder(pulses0, &dec, pulses4);
  dec_end(&dec, res);
}
void oracle_silk_encode_pulses(unsigned char *buf, int size, unsigned pre, int pre_n,
                               int signalType, int quantOffsetType, opus_int8 *pulses,
                               int frame_length, opus_uint32 *res) {
  ec_enc enc;
  enc_begin(&enc, buf, size, pre, pre_n);
  silk_encode_pulses(&enc, signalType, quantOffsetType, pulses, frame_length);
  enc_end(&enc, res);
}
void oracle_silk_decode_pulses(unsigned char *buf, int size, int pre_n, opus_int16 *pulses,
                               int signalType, int quantOffsetType, int frame_length,
                               opus_uint32 *res) {
  ec_dec dec;
  dec_begin(&dec, buf, size, pre_n);
  silk_decode_pulses(&dec, pulses, signalType, quantOffsetType, frame_length);
  dec_end(&dec, res);
}
/* ix = {ix[0][0..3], ix[1][0..3]}; mid_only < 0 skips the mid-only flag. */
void oracle_silk_stereo_encode(unsigned char *buf, int size, unsigned pre, int pre_n,
                               const opus_int8 *ix, int mid_only, opus_uint32 *res) {
  ec_enc enc;
  opus_int8 ixm[2][3];
  memcpy(ixm, ix, 6);
  enc_begin(&enc, buf, size, pre, pre_n);
  silk_stereo_encode_pred(&enc, ixm);
  if (mid_only >= 0) silk_stereo_encode_mid_only(&enc, (opus_int8)mid_only);
  enc_end(&enc, res);
}
/* pred_Q13[2] out; *mid_only out (only if want_mid). */
void oracle_silk_stereo_decode(unsigned char *buf, int size, int pre_n, opus_int32 *pred_Q13,
                               int want_mid, int *mid_only, opus_uint32 *res) {
  ec_dec dec;
  dec_begin(&dec, buf, size, pre_n);
  silk_stereo_decode_pred(&dec, pred_Q13);
  if (want_mid) silk_stereo_decode_mid_only(&dec, mid_only);
  dec_end(&dec, res);
}

void oracle_silk_gains_quant(opus_int8 *ind, opus_int32 *gain_Q16, opus_int8 *prev_ind,
                             int conditional, int nb_subfr) {
  silk_gains_quant(ind, gain_Q16, prev_ind, conditional, nb_subfr);
}
void oracle_silk_gains_dequant(opus_int32 *gain_Q16, const opus_int8 *ind, opus_int8 *prev_ind,
                               int conditional, int nb_subfr) {
  silk_gains_dequant(gain_Q16, ind, prev_ind, conditional, nb_subfr);
}
opus_int32 oracle_silk_gains_ID(const opus_int8 *ind, int nb_subfr) {
  return silk_gains_ID(ind, nb_subfr);
}
void oracle_silk_decode_pitch(int lagIndex, int contourIndex, int *pitch_lags, int Fs_kHz,
                              int nb_subfr) {
  silk_decode_pitch((opus_int16)lagIndex, (opus_int8)contourIndex, pitch_lags, Fs_kHz, nb_subfr);
}

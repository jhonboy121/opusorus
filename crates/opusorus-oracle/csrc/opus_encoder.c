/* Oracle C shims for unit opus_encoder (src/opus_encoder.c, src/opus_multistream_encoder.c,
 * src/opus_projection_encoder.c).
 *
 * Private copies of opus_encoder.c and opus_multistream_encoder.c are compiled here with their
 * exported symbols renamed (oracle_oe_dup_*), which gives access to their static helpers and to
 * the OpusEncoder struct (state dumps). The public API tests use the library itself.
 *
 * Built in the float and in the fixed-point oracle: every signal/state value uses the build's
 * C type (opus_res, opus_val16/32, celt_glog). */
// oracle-build: any
#include <stddef.h>
#include <string.h>
#include "opus_types.h"
#include "opus_defines.h"

#define opus_encoder_get_size oracle_oe_dup_opus_encoder_get_size
#define opus_encoder_init oracle_oe_dup_opus_encoder_init
#define opus_encoder_create oracle_oe_dup_opus_encoder_create
#define downmix_float oracle_oe_dup_downmix_float
#define downmix_int oracle_oe_dup_downmix_int
#define downmix_int24 oracle_oe_dup_downmix_int24
#define frame_size_select oracle_oe_dup_frame_size_select
#define compute_stereo_width oracle_oe_dup_compute_stereo_width
#define is_digital_silence oracle_oe_dup_is_digital_silence
#define opus_encode_native oracle_oe_dup_opus_encode_native
#define opus_encode oracle_oe_dup_opus_encode
#define opus_encode24 oracle_oe_dup_opus_encode24
#define opus_encode_float oracle_oe_dup_opus_encode_float
#define opus_encoder_ctl oracle_oe_dup_opus_encoder_ctl
#define opus_encoder_destroy oracle_oe_dup_opus_encoder_destroy
/* Non-static in the fixed-point build (also defined by the library's opus_encoder.o). */
#define silk_biquad_res oracle_oe_dup_silk_biquad_res
#include "../../../vendor/libopus/src/opus_encoder.c"

#define surround_analysis oracle_oe_dup_surround_analysis
#define opus_multistream_encoder_get_size oracle_oe_dup_opus_multistream_encoder_get_size
#define opus_multistream_surround_encoder_get_size \
  oracle_oe_dup_opus_multistream_surround_encoder_get_size
#define opus_multistream_encoder_init oracle_oe_dup_opus_multistream_encoder_init
#define opus_multistream_surround_encoder_init oracle_oe_dup_opus_multistream_surround_encoder_init
#define opus_multistream_encoder_create oracle_oe_dup_opus_multistream_encoder_create
#define opus_multistream_surround_encoder_create \
  oracle_oe_dup_opus_multistream_surround_encoder_create
#define opus_multistream_encode_native oracle_oe_dup_opus_multistream_encode_native
#define opus_multistream_encode oracle_oe_dup_opus_multistream_encode
#define opus_multistream_encode24 oracle_oe_dup_opus_multistream_encode24
#define opus_multistream_encode_float oracle_oe_dup_opus_multistream_encode_float
#define opus_multistream_encoder_ctl_va_list oracle_oe_dup_opus_multistream_encoder_ctl_va_list
#define opus_multistream_encoder_ctl oracle_oe_dup_opus_multistream_encoder_ctl
#define opus_multistream_encoder_destroy oracle_oe_dup_opus_multistream_encoder_destroy
#include "../../../vendor/libopus/src/opus_multistream_encoder.c"

/* ------------------------------------------------------------------------------------------ */
/* Static helpers of opus_encoder.c                                                             */
/* ------------------------------------------------------------------------------------------ */

int oracle_oe_gen_toc(int mode, int framerate, int bandwidth, int channels) {
  return gen_toc(mode, framerate, bandwidth, channels);
}

void oracle_oe_hp_cutoff(const opus_res *in, int cutoff_hz, opus_res *out, opus_val32 *hp_mem,
                         int len, int channels, int fs) {
  hp_cutoff(in, cutoff_hz, out, hp_mem, len, channels, fs, 0);
}

void oracle_oe_dc_reject(const opus_res *in, int cutoff_hz, opus_res *out, opus_val32 *hp_mem,
                         int len, int channels, int fs) {
  dc_reject(in, cutoff_hz, out, hp_mem, len, channels, fs);
}

/* In place, as the encoder calls them. The window is that of the 48 kHz (or 96 kHz) mode. */
void oracle_oe_stereo_fade(opus_res *buf, opus_val16 g1, opus_val16 g2, int fs96, int frame_size,
                           int channels, int fs) {
  const CELTMode *m = opus_custom_mode_create(fs96 ? 96000 : 48000, fs96 ? 1920 : 960, NULL);
  stereo_fade(buf, buf, g1, g2, m->overlap, frame_size, channels, m->window, fs);
}
void oracle_oe_gain_fade(opus_res *buf, opus_val16 g1, opus_val16 g2, int fs96, int frame_size,
                         int channels, int fs) {
  const CELTMode *m = opus_custom_mode_create(fs96 ? 96000 : 48000, fs96 ? 1920 : 960, NULL);
  gain_fade(buf, buf, g1, g2, m->overlap, frame_size, channels, m->window, fs);
}

int oracle_oe_frame_size_select(int application, int frame_size, int variable_duration, int fs) {
  return oracle_oe_dup_frame_size_select(application, frame_size, variable_duration, fs);
}

/* mem: XX, XY, YY, smoothed_width, max_follower */
opus_val16 oracle_oe_compute_stereo_width(const opus_res *pcm, int frame_size, int fs,
                                          opus_val32 *mem) {
  StereoWidthState s;
  opus_val16 r;
  s.XX = mem[0];
  s.XY = mem[1];
  s.YY = mem[2];
  s.smoothed_width = mem[3];
  s.max_follower = mem[4];
  r = oracle_oe_dup_compute_stereo_width(pcm, frame_size, fs, &s);
  mem[0] = s.XX;
  mem[1] = s.XY;
  mem[2] = s.YY;
  mem[3] = s.smoothed_width;
  mem[4] = s.max_follower;
  return r;
}

int oracle_oe_decide_fec(int use_fec, int loss, int last_fec, int mode, int *bandwidth, int rate) {
  return decide_fec(use_fec, loss, last_fec, mode, bandwidth, rate);
}

int oracle_oe_compute_silk_rate_for_hybrid(int rate, int bandwidth, int frame20ms, int vbr,
                                           int fec, int channels) {
  return compute_silk_rate_for_hybrid(rate, bandwidth, frame20ms, vbr, fec, channels);
}

int oracle_oe_compute_equiv_rate(int bitrate, int channels, int frame_rate, int vbr, int mode,
                                 int complexity, int loss) {
  return compute_equiv_rate(bitrate, channels, frame_rate, vbr, mode, complexity, loss);
}

opus_val32 oracle_oe_compute_frame_energy(const opus_res *pcm, int frame_size, int channels) {
  return compute_frame_energy(pcm, frame_size, channels, 0);
}

int oracle_oe_decide_dtx_mode(int activity, int *nb_no_activity_ms_q1, int frame_size_ms_q1) {
  return decide_dtx_mode(activity, nb_no_activity_ms_q1, frame_size_ms_q1);
}

int oracle_oe_compute_redundancy_bytes(int max_data_bytes, int bitrate_bps, int frame_rate,
                                       int channels) {
  return compute_redundancy_bytes(max_data_bytes, bitrate_bps, frame_rate, channels);
}

/* ------------------------------------------------------------------------------------------ */
/* Static helpers of opus_multistream_encoder.c                                                 */
/* ------------------------------------------------------------------------------------------ */

opus_val16 oracle_oe_log_sum(celt_glog a, celt_glog b) { return logSum(a, b); }

void oracle_oe_channel_pos(int channels, int *pos) { channel_pos(channels, pos); }

#ifndef DISABLE_FLOAT_API
static void oe_copy_in_float(opus_res *dst, int dst_stride, const void *src, int src_stride,
                             int src_channel, int frame_size, void *user_data) {
  const float *s = (const float *)src;
  int i;
  (void)user_data;
  for (i = 0; i < frame_size; i++) dst[i * dst_stride] = FLOAT2RES(s[i * src_stride + src_channel]);
}

/* surround_analysis on float input with the mode for `rate` (48 kHz mode, or 96 kHz with QEXT).
 * mem: channels*overlap, preemph_mem: channels, band_log_e: 21*channels. */
void oracle_oe_surround_analysis(const float *pcm, celt_glog *band_log_e, opus_val32 *mem,
                                 opus_val32 *preemph_mem, int len, int channels, int rate) {
  const CELTMode *m = opus_custom_mode_create(rate == 96000 ? 96000 : 48000,
                                              rate == 96000 ? 1920 : 960, NULL);
  oracle_oe_dup_surround_analysis(m, pcm, band_log_e, mem, preemph_mem, len, m->overlap,
                                  channels, rate, oe_copy_in_float, 0);
}
#endif

/* ------------------------------------------------------------------------------------------ */
/* Private Opus encoder with state dumps                                                        */
/* ------------------------------------------------------------------------------------------ */

OpusEncoder *oracle_oe_create(int fs, int channels, int application, int *err) {
  return oracle_oe_dup_opus_encoder_create(fs, channels, application, err);
}
void oracle_oe_destroy(OpusEncoder *st) { oracle_oe_dup_opus_encoder_destroy(st); }
int oracle_oe_ctl_set(OpusEncoder *st, int request, int value) {
  return oracle_oe_dup_opus_encoder_ctl(st, request, (opus_int32)value);
}
#ifndef DISABLE_FLOAT_API
int oracle_oe_encode_float(OpusEncoder *st, const float *pcm, int frame_size, unsigned char *out,
                           int max_bytes) {
  return oracle_oe_dup_opus_encode_float(st, pcm, frame_size, out, max_bytes);
}
#endif
int oracle_oe_encode(OpusEncoder *st, const opus_int16 *pcm, int frame_size, unsigned char *out,
                     int max_bytes) {
  return oracle_oe_dup_opus_encode(st, pcm, frame_size, out, max_bytes);
}
int oracle_oe_encode24(OpusEncoder *st, const opus_int32 *pcm, int frame_size, unsigned char *out,
                       int max_bytes) {
  return oracle_oe_dup_opus_encode24(st, pcm, frame_size, out, max_bytes);
}
int oracle_oe_ctl_get(OpusEncoder *st, int request, opus_int32 *value) {
  return oracle_oe_dup_opus_encoder_ctl(st, request, value);
}

#define OE_STATE_INTS 64
/* Flat dump of the OpusEncoder fields (floats as bit patterns, fixed-point values sign-extended
 * to 32 bits). Returns the number of values written; delay_buffer (encoder_buffer*channels
 * opus_res samples) goes to `delay`. */
int oracle_oe_dump(const OpusEncoder *st, opus_uint32 *v, opus_res *delay) {
  int n = 0;
#define I(x) v[n++] = (opus_uint32)(x)
#ifdef FIXED_POINT
#define F(x) I((opus_int32)(x))
#else
#define F(x) do { float f_ = (x); memcpy(&v[n++], &f_, 4); } while (0)
#endif
  I(st->application);
  I(st->channels);
  I(st->delay_compensation);
  I(st->force_channels);
  I(st->signal_type);
  I(st->user_bandwidth);
  I(st->max_bandwidth);
  I(st->user_forced_mode);
  I(st->voice_ratio);
  I(st->Fs);
  I(st->use_vbr);
  I(st->vbr_constraint);
  I(st->variable_duration);
  I(st->bitrate_bps);
  I(st->user_bitrate_bps);
  I(st->lsb_depth);
  I(st->encoder_buffer);
  I(st->lfe);
  I(st->use_dtx);
  I(st->fec_config);
  I(st->stream_channels);
  I(st->hybrid_stereo_width_Q14);
  I(st->variable_HP_smth2_Q15);
  F(st->prev_HB_gain);
  F(st->hp_mem[0]);
  F(st->hp_mem[1]);
  F(st->hp_mem[2]);
  F(st->hp_mem[3]);
  I(st->mode);
  I(st->prev_mode);
  I(st->prev_channels);
  I(st->prev_framesize);
  I(st->bandwidth);
  I(st->auto_bandwidth);
  I(st->silk_bw_switch);
  I(st->first);
  F(st->width_mem.XX);
  F(st->width_mem.XY);
  F(st->width_mem.YY);
  F(st->width_mem.smoothed_width);
  F(st->width_mem.max_follower);
#ifndef DISABLE_FLOAT_API
  I(st->detected_bandwidth);
#else
  I(0); /* no detected_bandwidth without the float API */
#endif
  I(st->nb_no_activity_ms_Q1);
  F(st->peak_signal_energy);
  I(st->nonfinal_frame);
  I(st->rangeFinal);
  I(st->silk_mode.bitRate);
  I(st->silk_mode.maxBits);
  I(st->silk_mode.toMono);
  I(st->silk_mode.LBRR_coded);
  I(st->silk_mode.useDTX);
  I(st->silk_mode.useCBR);
  I(st->silk_mode.stereoWidth_Q14);
  I(st->silk_mode.internalSampleRate);
  I(st->silk_mode.opusCanSwitch);
  I(st->silk_mode.desiredInternalSampleRate);
  I(st->silk_mode.maxInternalSampleRate);
  I(st->silk_mode.minInternalSampleRate);
  I(st->silk_mode.payloadSize_ms);
  I(st->silk_mode.nChannelsInternal);
  I(st->silk_mode.allowBandwidthSwitch);
  I(st->silk_mode.inWBmodeWithoutVariableLP);
  I(st->silk_mode.switchReady);
#undef I
#undef F
  if (delay) memcpy(delay, st->delay_buffer, sizeof(opus_res) * st->encoder_buffer * st->channels);
  return n;
}

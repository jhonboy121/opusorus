/* Oracle C shims for unit opus_packet: packet parsing, extensions, repacketizer, mapping
   matrix, analysis MLP and multistream layout helpers. Pointers are converted to offsets so
   the Rust side never sees C structs. The mapping-matrix shims take `opus_res` buffers
   (float, or short/int in the fixed-point builds); the rest is build-independent. */
// oracle-build: any
#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_private.h"
#include "mapping_matrix.h"
#include "mlp.h"

/* ---------------------------------------------------------------- opus.c */

int oracle_pk_align(int i) { return align(i); }

int oracle_pk_encode_size(int size, unsigned char *data) { return encode_size(size, data); }

/* opus_packet_parse_impl. On success writes toc, frame (offset,len) pairs, payload offset,
   packet offset and padding (offset,len). */
int oracle_pk_parse_impl(const unsigned char *data, int len, int self_delimited,
                         unsigned char *toc, int *frame_off, int *frame_len,
                         int *payload_offset, int *packet_offset, int *padding_off,
                         int *padding_len) {
  const unsigned char *frames[48];
  opus_int16 size[48];
  const unsigned char *padding = NULL;
  opus_int32 plen = 0;
  opus_int32 poff = 0;
  int i, ret;
  ret = opus_packet_parse_impl(data, len, self_delimited, toc, frames, size, payload_offset,
                               &poff, &padding, &plen);
  if (ret > 0) {
    for (i = 0; i < ret; i++) {
      frame_off[i] = (int)(frames[i] - data);
      frame_len[i] = size[i];
    }
    *packet_offset = poff;
    *padding_off = (int)(padding - data);
    *padding_len = plen;
  }
  return ret;
}

void oracle_pk_soft_clip_impl(float *x, int n, int c, float *mem) {
  opus_pcm_soft_clip_impl(x, n, c, mem, 0);
}

/* ---------------------------------------------------------------- extensions.c */

static void fill_ext(opus_extension_data *e, int n, const int *ids, const int *frames,
                     const int *offs, const int *lens, const unsigned char *buf) {
  int i;
  for (i = 0; i < n; i++) {
    e[i].id = ids[i];
    e[i].frame = frames[i];
    e[i].data = buf + offs[i];
    e[i].len = lens[i];
  }
}

static void dump_ext(const opus_extension_data *e, int n, const unsigned char *base, int *ids,
                     int *frames, int *offs, int *lens) {
  int i;
  for (i = 0; i < n; i++) {
    ids[i] = e[i].id;
    frames[i] = e[i].frame;
    offs[i] = (int)(e[i].data - base);
    lens[i] = e[i].len;
  }
}

/* opus_packet_extensions_generate; out==NULL (use_null) measures only. */
int oracle_ext_generate(unsigned char *out, int use_null, int len, const int *ids,
                        const int *frames, const int *offs, const int *lens, int n,
                        const unsigned char *buf, int nb_frames, int pad) {
  opus_extension_data e[512];
  if (n > 512) return -100;
  fill_ext(e, n, ids, frames, offs, lens, buf);
  return opus_packet_extensions_generate(use_null ? NULL : out, len, e, n, nb_frames, pad);
}

int oracle_ext_count(const unsigned char *data, int len, int nb_frames) {
  return opus_packet_extensions_count(data, len, nb_frames);
}

int oracle_ext_count_ext(const unsigned char *data, int len, int *nb_frame_exts, int nb_frames) {
  return opus_packet_extensions_count_ext(data, len, nb_frame_exts, nb_frames);
}

/* opus_packet_extensions_parse; *nb_ext is the capacity on input, count on output. */
int oracle_ext_parse(const unsigned char *data, int len, int nb_frames, int *nb_ext, int *ids,
                     int *frames, int *offs, int *lens) {
  opus_extension_data e[4096];
  opus_int32 n = *nb_ext;
  int ret;
  if (n > 4096) return -100;
  ret = opus_packet_extensions_parse(data, len, e, &n, nb_frames);
  *nb_ext = n;
  if (ret >= 0) dump_ext(e, n, data, ids, frames, offs, lens);
  return ret;
}

int oracle_ext_parse_ext(const unsigned char *data, int len, const int *nb_frame_exts,
                         int nb_frames, int *nb_ext, int *ids, int *frames, int *offs,
                         int *lens) {
  opus_extension_data e[4096];
  opus_int32 n = *nb_ext;
  int ret;
  if (n > 4096) return -100;
  ret = opus_packet_extensions_parse_ext(data, len, e, &n, nb_frame_exts, nb_frames);
  *nb_ext = n;
  if (ret >= 0) dump_ext(e, n, data, ids, frames, offs, lens);
  return ret;
}

/* Runs a sequence of iterator operations. ops[2*i] = kind (0 next, 1 find(arg), 2 reset,
   3 set_frame_max(arg)), ops[2*i+1] = arg. res[5*i..] = {ret, id, frame, off, len}. */
void oracle_ext_iter_ops(const unsigned char *data, int len, int nb_frames, const int *ops,
                         int nops, int *res) {
  OpusExtensionIterator iter;
  int i;
  opus_extension_iterator_init(&iter, data, len, nb_frames);
  for (i = 0; i < nops; i++) {
    opus_extension_data ext;
    int ret = 0;
    int *r = res + 5 * i;
    memset(&ext, 0, sizeof(ext));
    switch (ops[2 * i]) {
      case 0: ret = opus_extension_iterator_next(&iter, &ext); break;
      case 1: ret = opus_extension_iterator_find(&iter, &ext, ops[2 * i + 1]); break;
      case 2: opus_extension_iterator_reset(&iter); break;
      case 3: opus_extension_iterator_set_frame_max(&iter, ops[2 * i + 1]); break;
    }
    r[0] = ret;
    if (ret > 0) {
      r[1] = ext.id;
      r[2] = ext.frame;
      r[3] = (int)(ext.data - data);
      r[4] = ext.len;
    } else {
      r[1] = r[2] = r[3] = r[4] = 0;
    }
  }
}

/* ---------------------------------------------------------------- repacketizer.c */

/* Feeds npk packets (concatenated in buf, lengths in lens) to a repacketizer with the given
   self-delimited flag, recording each cat return value, then runs out_range_impl. */
int oracle_rp_run(const unsigned char *buf, const int *lens, int npk, int cat_sd, int *cat_rets,
                  int *nb_frames, int begin, int end, unsigned char *out, int maxlen, int sd,
                  int pad, const int *ids, const int *frames, const int *offs, const int *elens,
                  int next, const unsigned char *ext_buf) {
  OpusRepacketizer rp;
  opus_extension_data e[512];
  int i, off = 0;
  if (next > 512) return -100;
  opus_repacketizer_init(&rp);
  for (i = 0; i < npk; i++) {
    /* opus_repacketizer_cat_impl is static: reach it through the public entry points
       (non-self-delimited) or through opus_multistream_packet_unpad for self-delimited. */
    if (cat_sd) {
      extern int oracle_rp_cat_sd(OpusRepacketizer *rp, const unsigned char *data, int len);
      cat_rets[i] = oracle_rp_cat_sd(&rp, buf + off, lens[i]);
    } else {
      cat_rets[i] = opus_repacketizer_cat(&rp, buf + off, lens[i]);
    }
    off += lens[i];
  }
  *nb_frames = opus_repacketizer_get_nb_frames(&rp);
  fill_ext(e, next, ids, frames, offs, elens, ext_buf);
  return opus_repacketizer_out_range_impl(&rp, begin, end, out, maxlen, sd, pad, e, next);
}

int oracle_pad_impl(unsigned char *data, int len, int new_len, int pad, const int *ids,
                    const int *frames, const int *offs, const int *elens, int next,
                    const unsigned char *ext_buf) {
  opus_extension_data e[512];
  if (next > 512) return -100;
  fill_ext(e, next, ids, frames, offs, elens, ext_buf);
  return opus_packet_pad_impl(data, len, new_len, pad, e, next);
}

/* ---------------------------------------------------------------- mapping_matrix.c */

int oracle_mm_get_size(int rows, int cols) { return mapping_matrix_get_size(rows, cols); }

static MappingMatrix *mm_make(int rows, int cols, int gain, const short *data) {
  MappingMatrix *m = (MappingMatrix *)malloc(mapping_matrix_get_size(rows, cols));
  mapping_matrix_init(m, rows, cols, gain, data, rows * cols * (int)sizeof(opus_int16));
  return m;
}

/* kind: 0 in_float 1 out_float 2 in_short 3 out_short 4 in_int24 5 out_int24.
   in_* : input is `in` (float/short/int per kind), output opus_res, row = output_row,
          stride = output_rows.
   out_*: input opus_res, row = input_row, stride = input_rows. */
void oracle_mm_multiply(int kind, int rows, int cols, const short *mdata, const void *in,
                        int in_rows, void *out, int row, int out_rows, int frame_size) {
  MappingMatrix *m = mm_make(rows, cols, 0, mdata);
  switch (kind) {
    case 0:
      mapping_matrix_multiply_channel_in_float(m, (const float *)in, in_rows, (opus_res *)out,
                                               row, out_rows, frame_size);
      break;
    case 1:
      mapping_matrix_multiply_channel_out_float(m, (const opus_res *)in, row, in_rows,
                                                (float *)out, out_rows, frame_size);
      break;
    case 2:
      mapping_matrix_multiply_channel_in_short(m, (const opus_int16 *)in, in_rows,
                                               (opus_res *)out, row, out_rows, frame_size);
      break;
    case 3:
      mapping_matrix_multiply_channel_out_short(m, (const opus_res *)in, row, in_rows,
                                                (opus_int16 *)out, out_rows, frame_size);
      break;
    case 4:
      mapping_matrix_multiply_channel_in_int24(m, (const opus_int32 *)in, in_rows,
                                               (opus_res *)out, row, out_rows, frame_size);
      break;
    case 5:
      mapping_matrix_multiply_channel_out_int24(m, (const opus_res *)in, row, in_rows,
                                                (opus_int32 *)out, out_rows, frame_size);
      break;
  }
  free(m);
}

/* Static matrices: idx 0..4 mixing foa..fifthoa, 5..9 demixing foa..fifthoa. Returns the
   number of cells copied into data (cap = capacity). */
int oracle_mm_static(int idx, int *rows, int *cols, int *gain, short *data, int cap) {
  static const MappingMatrix *const hdr[10] = {
      &mapping_matrix_foa_mixing,        &mapping_matrix_soa_mixing,
      &mapping_matrix_toa_mixing,        &mapping_matrix_fourthoa_mixing,
      &mapping_matrix_fifthoa_mixing,    &mapping_matrix_foa_demixing,
      &mapping_matrix_soa_demixing,      &mapping_matrix_toa_demixing,
      &mapping_matrix_fourthoa_demixing, &mapping_matrix_fifthoa_demixing};
  static const opus_int16 *const dat[10] = {
      mapping_matrix_foa_mixing_data,        mapping_matrix_soa_mixing_data,
      mapping_matrix_toa_mixing_data,        mapping_matrix_fourthoa_mixing_data,
      mapping_matrix_fifthoa_mixing_data,    mapping_matrix_foa_demixing_data,
      mapping_matrix_soa_demixing_data,      mapping_matrix_toa_demixing_data,
      mapping_matrix_fourthoa_demixing_data, mapping_matrix_fifthoa_demixing_data};
  int n;
  if (idx < 0 || idx > 9) return -1;
  *rows = hdr[idx]->rows;
  *cols = hdr[idx]->cols;
  *gain = hdr[idx]->gain;
  n = *rows * *cols;
  if (n > cap) return -1;
  memcpy(data, dat[idx], n * sizeof(opus_int16));
  return n;
}

/* ---------------------------------------------------------------- mlp.c */

void oracle_mlp_dense(const signed char *bias, const signed char *weights, int nb_inputs,
                      int nb_neurons, int sigmoid, float *output, const float *input) {
  AnalysisDenseLayer l;
  l.bias = (const opus_int8 *)bias;
  l.input_weights = (const opus_int8 *)weights;
  l.nb_inputs = nb_inputs;
  l.nb_neurons = nb_neurons;
  l.sigmoid = sigmoid;
  analysis_compute_dense(&l, output, input);
}

void oracle_mlp_gru(const signed char *bias, const signed char *weights,
                    const signed char *recur, int nb_inputs, int nb_neurons, float *state,
                    const float *input) {
  AnalysisGRULayer l;
  l.bias = (const opus_int8 *)bias;
  l.input_weights = (const opus_int8 *)weights;
  l.recurrent_weights = (const opus_int8 *)recur;
  l.nb_inputs = nb_inputs;
  l.nb_neurons = nb_neurons;
  analysis_compute_gru(&l, state, input);
}

/* The built-in analysis layers: which = 0 (layer0 dense), 1 (layer1 GRU), 2 (layer2 dense). */
void oracle_mlp_builtin(int which, float *out_or_state, const float *input) {
  if (which == 0) analysis_compute_dense(&layer0, out_or_state, input);
  else if (which == 1) analysis_compute_gru(&layer1, out_or_state, input);
  else analysis_compute_dense(&layer2, out_or_state, input);
}

/* ---------------------------------------------------------------- opus_multistream.c */

static void mk_layout(ChannelLayout *l, int nb_channels, int nb_streams, int nb_coupled,
                      const unsigned char *mapping) {
  l->nb_channels = nb_channels;
  l->nb_streams = nb_streams;
  l->nb_coupled_streams = nb_coupled;
  memcpy(l->mapping, mapping, 256);
}

int oracle_ms_validate_layout(int nb_channels, int nb_streams, int nb_coupled,
                              const unsigned char *mapping) {
  ChannelLayout l;
  mk_layout(&l, nb_channels, nb_streams, nb_coupled, mapping);
  return validate_layout(&l);
}

/* which: 0 left, 1 right, 2 mono. */
int oracle_ms_get_channel(int which, int nb_channels, int nb_streams, int nb_coupled,
                          const unsigned char *mapping, int stream_id, int prev) {
  ChannelLayout l;
  mk_layout(&l, nb_channels, nb_streams, nb_coupled, mapping);
  if (which == 0) return get_left_channel(&l, stream_id, prev);
  if (which == 1) return get_right_channel(&l, stream_id, prev);
  return get_mono_channel(&l, stream_id, prev);
}

/* ---------------------------------------------------------------- static functions */

/* Copies of static helpers, reached by including the C files with the exported symbols
   renamed (so the copies do not clash with the library's definitions). */
#define opus_repacketizer_get_size oracle_rpc_get_size
#define opus_repacketizer_init oracle_rpc_init
#define opus_repacketizer_create oracle_rpc_create
#define opus_repacketizer_destroy oracle_rpc_destroy
#define opus_repacketizer_cat oracle_rpc_cat
#define opus_repacketizer_get_nb_frames oracle_rpc_get_nb_frames
#define opus_repacketizer_out_range_impl oracle_rpc_out_range_impl
#define opus_repacketizer_out_range oracle_rpc_out_range
#define opus_repacketizer_out oracle_rpc_out
#define opus_packet_pad_impl oracle_rpc_pad_impl
#define opus_packet_pad oracle_rpc_pad
#define opus_packet_unpad oracle_rpc_unpad
#define opus_multistream_packet_pad oracle_rpc_ms_pad
#define opus_multistream_packet_unpad oracle_rpc_ms_unpad
#include "repacketizer.c"
#undef opus_repacketizer_get_size
#undef opus_repacketizer_init
#undef opus_repacketizer_create
#undef opus_repacketizer_destroy
#undef opus_repacketizer_cat
#undef opus_repacketizer_get_nb_frames
#undef opus_repacketizer_out_range_impl
#undef opus_repacketizer_out_range
#undef opus_repacketizer_out
#undef opus_packet_pad_impl
#undef opus_packet_pad
#undef opus_packet_unpad
#undef opus_multistream_packet_pad
#undef opus_multistream_packet_unpad

int oracle_rp_cat_sd(OpusRepacketizer *rp, const unsigned char *data, int len) {
  return opus_repacketizer_cat_impl(rp, data, len, 1);
}

#define analysis_compute_dense oracle_mlpc_dense
#define analysis_compute_gru oracle_mlpc_gru
#include "mlp.c"
#undef analysis_compute_dense
#undef analysis_compute_gru

float oracle_mlp_tansig(float x) { return tansig_approx(x); }
float oracle_mlp_sigmoid(float x) { return sigmoid_approx(x); }

/* Oracle C shims for the packet loss generator (dnn/lossgen.c + dnn/lossgen_data.c, libopus
   --enable-lossgen), which upstream links into opus_demo and lossgen_demo only.

   Only compiled when build.rs defines ENABLE_LOSSGEN (oracle feature `lossgen`). lossgen.c
   #includes parse_lpcnet_weights.c and nnet_arch.h itself, so every external symbol of this
   translation unit is renamed (oracle_lg_*): they cannot clash with the DNN objects of the
   library (DNN builds) or with the lossgen.c / lossgen_data.c a C opus_demo is linked with. */
// oracle-build: any
#ifdef ENABLE_LOSSGEN

#define parse_record oracle_lg_parse_record
#define parse_weights oracle_lg_parse_weights
#define linear_init oracle_lg_linear_init
#define conv2d_init oracle_lg_conv2d_init
#define compute_linear_c oracle_lg_compute_linear_c
#define compute_activation_c oracle_lg_compute_activation_c
#define compute_conv2d_c oracle_lg_compute_conv2d_c
#define compute_generic_gru_lossgen oracle_lg_compute_generic_gru_lossgen
#define compute_generic_dense_lossgen oracle_lg_compute_generic_dense_lossgen
#define lossgen_init oracle_lg_lossgen_init
#define lossgen_load_model oracle_lg_lossgen_load_model
#define sample_loss oracle_lg_sample_loss
#define init_lossgen oracle_lg_init_lossgen
#define lossgen_arrays oracle_lg_lossgen_arrays

#include <stdlib.h>
#include <string.h>

#include "lossgen_data.c"
#include "lossgen.c"

/* ---- the compiled-in table ---- */

int oracle_lg_array_count(void) {
  int n = 0;
  while (lossgen_arrays[n].name != NULL) n++;
  return n;
}

/* name (NUL-terminated, at most 64 bytes with the NUL), type and size of array i. */
void oracle_lg_array_info(int i, char *name, int *type, int *size) {
  const WeightArray *a = &lossgen_arrays[i];
  strncpy(name, a->name, 63);
  name[63] = 0;
  *type = a->type;
  *size = a->size;
}

void oracle_lg_array_data(int i, void *out) {
  memcpy(out, lossgen_arrays[i].data, lossgen_arrays[i].size);
}

/* ---- generator state ---- */

LossGenState *oracle_lg_new(void) {
  LossGenState *st = (LossGenState *)malloc(sizeof(*st));
  if (st != NULL) lossgen_init(st);
  return st;
}

/* lossgen_load_model on a fresh (zeroed) state, as a caller would do after lossgen_init.
   Returns NULL if the state cannot be allocated; *ret gets lossgen_load_model's result. */
LossGenState *oracle_lg_new_from_blob(const void *data, int len, int *ret) {
  LossGenState *st = (LossGenState *)malloc(sizeof(*st));
  if (st == NULL) return NULL;
  OPUS_CLEAR(st, 1);
  *ret = lossgen_load_model(st, data, len);
  return st;
}

void oracle_lg_free(LossGenState *st) { free(st); }

int oracle_lg_sample(LossGenState *st, float percent_loss) {
  return sample_loss(st, percent_loss);
}

/* gru1 (LOSSGEN_GRU1_STATE_SIZE floats), gru2 (LOSSGEN_GRU2_STATE_SIZE), last_loss, used. */
void oracle_lg_state(const LossGenState *st, float *gru1, float *gru2, int *last_loss, int *used) {
  memcpy(gru1, st->gru1_state, sizeof(st->gru1_state));
  memcpy(gru2, st->gru2_state, sizeof(st->gru2_state));
  *last_loss = st->last_loss;
  *used = st->used;
}

int oracle_lg_gru1_size(void) { return LOSSGEN_GRU1_STATE_SIZE; }
int oracle_lg_gru2_size(void) { return LOSSGEN_GRU2_STATE_SIZE; }

/* ---- C rand() (the generator's randomness) ---- */

void oracle_lg_srand(unsigned seed) { srand(seed); }
int oracle_lg_rand(void) { return rand(); }

#endif /* ENABLE_LOSSGEN */

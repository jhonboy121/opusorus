/* Variadic libopus `*_ctl` entry points for opusorus.
 *
 * Stable Rust cannot define C variadic functions, so this file implements every
 * `opus_*_ctl(st, request, ...)` of opus.h, opus_multistream.h, opus_projection.h and
 * opus_custom.h: it decodes the argument list according to the request number (argument types
 * as in opus_defines.h and the libopus sources) and forwards to non-variadic Rust functions.
 *
 * Unknown requests return OPUS_UNIMPLEMENTED without reading any argument, as libopus does.
 * Requests of known shape that a given state type does not support are forwarded anyway and
 * rejected by the Rust side with OPUS_UNIMPLEMENTED (reading the argument the caller passed is
 * harmless).
 */

#include <stdarg.h>
#include "opus.h"
#include "opus_multistream.h"
#include "opus_projection.h"
#include "opus_custom.h"

/* Private requests (src/opus_private.h, celt/celt.h) accepted by the libopus encoder ctl. */
#define OPUSORUS_SET_FORCE_MODE_REQUEST 11002
#define OPUSORUS_SET_VOICE_RATIO_REQUEST 11018
#define OPUSORUS_GET_VOICE_RATIO_REQUEST 11019
#define OPUSORUS_SET_LFE_REQUEST 10024
#define OPUSORUS_SET_ENERGY_MASK_REQUEST 10026
#define OPUSORUS_CELT_SET_PREDICTION_REQUEST 10002
#define OPUSORUS_CELT_SET_INPUT_CLIPPING_REQUEST 10004
#define OPUSORUS_CELT_GET_AND_CLEAR_ERROR_REQUEST 10007
#define OPUSORUS_CELT_SET_CHANNELS_REQUEST 10008
#define OPUSORUS_CELT_SET_START_BAND_REQUEST 10010
#define OPUSORUS_CELT_SET_END_BAND_REQUEST 10012
#define OPUSORUS_CELT_SET_SIGNALLING_REQUEST 10016

/* State kinds; must match `handle::Kind` in src/handle.rs. */
enum {
   KIND_ENCODER = 1,
   KIND_DECODER = 2,
   KIND_MS_ENCODER = 3,
   KIND_MS_DECODER = 4,
   KIND_PROJECTION_ENCODER = 5,
   KIND_PROJECTION_DECODER = 6,
   KIND_CUSTOM_ENCODER = 8,
   KIND_CUSTOM_DECODER = 9,
   KIND_DRED_DECODER = 10
};

/* Non-variadic Rust implementations (src/ctl.rs). */
int opusorus_ctl_set(void *st, int kind, int request, opus_int32 value);
int opusorus_ctl_get(void *st, int kind, int request, opus_int32 *value);
int opusorus_ctl_get_u32(void *st, int kind, int request, opus_uint32 *value);
int opusorus_ctl_ptr(void *st, int kind, int request, void *ptr, opus_int32 len);
int opusorus_ctl_state(void *st, int kind, int request, opus_int32 stream_id, void **value);

static int opusorus_ctl_dispatch(void *st, int kind, int request, va_list ap)
{
   switch (request)
   {
   /* Requests taking one opus_int32 value. */
   case OPUS_SET_APPLICATION_REQUEST:
   case OPUS_SET_BITRATE_REQUEST:
   case OPUS_SET_MAX_BANDWIDTH_REQUEST:
   case OPUS_SET_VBR_REQUEST:
   case OPUS_SET_BANDWIDTH_REQUEST:
   case OPUS_SET_COMPLEXITY_REQUEST:
   case OPUS_SET_INBAND_FEC_REQUEST:
   case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
   case OPUS_SET_DTX_REQUEST:
   case OPUS_SET_VBR_CONSTRAINT_REQUEST:
   case OPUS_SET_FORCE_CHANNELS_REQUEST:
   case OPUS_SET_SIGNAL_REQUEST:
   case OPUS_SET_GAIN_REQUEST:
   case OPUS_SET_LSB_DEPTH_REQUEST:
   case OPUS_SET_EXPERT_FRAME_DURATION_REQUEST:
   case OPUS_SET_PREDICTION_DISABLED_REQUEST:
   case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
   case OPUS_SET_DRED_DURATION_REQUEST:
   case OPUS_SET_OSCE_BWE_REQUEST:
   case OPUS_SET_QEXT_REQUEST:
   case OPUS_SET_IGNORE_EXTENSIONS_REQUEST:
   case OPUSORUS_SET_FORCE_MODE_REQUEST:
   case OPUSORUS_SET_VOICE_RATIO_REQUEST:
   case OPUSORUS_SET_LFE_REQUEST:
   case OPUSORUS_CELT_SET_PREDICTION_REQUEST:
   case OPUSORUS_CELT_SET_INPUT_CLIPPING_REQUEST:
   case OPUSORUS_CELT_SET_CHANNELS_REQUEST:
   case OPUSORUS_CELT_SET_START_BAND_REQUEST:
   case OPUSORUS_CELT_SET_END_BAND_REQUEST:
   case OPUSORUS_CELT_SET_SIGNALLING_REQUEST:
   {
      opus_int32 value = va_arg(ap, opus_int32);
      return opusorus_ctl_set(st, kind, request, value);
   }
   /* No argument. */
   case OPUS_RESET_STATE:
      return opusorus_ctl_set(st, kind, request, 0);
   /* Requests writing one opus_int32. */
   case OPUS_GET_APPLICATION_REQUEST:
   case OPUS_GET_BITRATE_REQUEST:
   case OPUS_GET_MAX_BANDWIDTH_REQUEST:
   case OPUS_GET_VBR_REQUEST:
   case OPUS_GET_BANDWIDTH_REQUEST:
   case OPUS_GET_COMPLEXITY_REQUEST:
   case OPUS_GET_INBAND_FEC_REQUEST:
   case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
   case OPUS_GET_DTX_REQUEST:
   case OPUS_GET_VBR_CONSTRAINT_REQUEST:
   case OPUS_GET_FORCE_CHANNELS_REQUEST:
   case OPUS_GET_SIGNAL_REQUEST:
   case OPUS_GET_LOOKAHEAD_REQUEST:
   case OPUS_GET_SAMPLE_RATE_REQUEST:
   case OPUS_GET_PITCH_REQUEST:
   case OPUS_GET_GAIN_REQUEST:
   case OPUS_GET_LSB_DEPTH_REQUEST:
   case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
   case OPUS_GET_EXPERT_FRAME_DURATION_REQUEST:
   case OPUS_GET_PREDICTION_DISABLED_REQUEST:
   case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
   case OPUS_GET_IN_DTX_REQUEST:
   case OPUS_GET_DRED_DURATION_REQUEST:
   case OPUS_GET_OSCE_BWE_REQUEST:
   case OPUS_GET_QEXT_REQUEST:
   case OPUS_GET_IGNORE_EXTENSIONS_REQUEST:
   case OPUSORUS_GET_VOICE_RATIO_REQUEST:
   case OPUSORUS_CELT_GET_AND_CLEAR_ERROR_REQUEST:
   case OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST:
   case OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32 *);
      return opusorus_ctl_get(st, kind, request, value);
   }
   /* Requests writing one opus_uint32. */
   case OPUS_GET_FINAL_RANGE_REQUEST:
   {
      opus_uint32 *value = va_arg(ap, opus_uint32 *);
      return opusorus_ctl_get_u32(st, kind, request, value);
   }
   /* (const unsigned char *data, opus_int32 len) */
   case OPUS_SET_DNN_BLOB_REQUEST:
   {
      const unsigned char *data = va_arg(ap, const unsigned char *);
      opus_int32 len = va_arg(ap, opus_int32);
      return opusorus_ctl_ptr(st, kind, request, (void *)data, len);
   }
   /* (unsigned char *matrix, opus_int32 size) */
   case OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST:
   {
      unsigned char *matrix = va_arg(ap, unsigned char *);
      opus_int32 size = va_arg(ap, opus_int32);
      return opusorus_ctl_ptr(st, kind, request, matrix, size);
   }
   /* (opus_val16 *mask): float in the float build. */
   case OPUSORUS_SET_ENERGY_MASK_REQUEST:
   {
      float *mask = va_arg(ap, float *);
      return opusorus_ctl_ptr(st, kind, request, mask, -1);
   }
   /* (opus_int32 stream_id, OpusEncoder ** / OpusDecoder **) */
   case OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST:
   case OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST:
   {
      opus_int32 stream_id = va_arg(ap, opus_int32);
      void **value = va_arg(ap, void **);
      return opusorus_ctl_state(st, kind, request, stream_id, value);
   }
   default:
      return OPUS_UNIMPLEMENTED;
   }
}

/* With OPUSORUS_CTL_TRAMPOLINE (set by build.rs on architectures with a trampoline in
 * src/ctl.rs), the C functions get an `opusorus_va_` prefix and Rust exports the libopus names
 * as naked tail-jumps to them: a Rust cdylib only exports symbols defined in Rust (its linker
 * version script hides everything else), so C-defined functions would be missing from
 * libopusorus.so. The jump leaves registers and stack untouched, so the variadic arguments reach
 * these functions exactly as the caller passed them. */
#ifdef OPUSORUS_CTL_TRAMPOLINE
# define OPUSORUS_CTL_NAME(name) opusorus_va_##name
#else
# define OPUSORUS_CTL_NAME(name) name
#endif

#define OPUSORUS_CTL(name, type, kind) \
   int OPUSORUS_CTL_NAME(name)(type *st, int request, ...) \
   { \
      int ret; \
      va_list ap; \
      va_start(ap, request); \
      ret = opusorus_ctl_dispatch((void *)st, kind, request, ap); \
      va_end(ap); \
      return ret; \
   }

OPUSORUS_CTL(opus_encoder_ctl, OpusEncoder, KIND_ENCODER)
OPUSORUS_CTL(opus_decoder_ctl, OpusDecoder, KIND_DECODER)
OPUSORUS_CTL(opus_multistream_encoder_ctl, OpusMSEncoder, KIND_MS_ENCODER)
OPUSORUS_CTL(opus_multistream_decoder_ctl, OpusMSDecoder, KIND_MS_DECODER)
OPUSORUS_CTL(opus_projection_encoder_ctl, OpusProjectionEncoder, KIND_PROJECTION_ENCODER)
OPUSORUS_CTL(opus_projection_decoder_ctl, OpusProjectionDecoder, KIND_PROJECTION_DECODER)
OPUSORUS_CTL(opus_dred_decoder_ctl, OpusDREDDecoder, KIND_DRED_DECODER)
#ifdef OPUSORUS_CUSTOM_MODES
OPUSORUS_CTL(opus_custom_encoder_ctl, OpusCustomEncoder, KIND_CUSTOM_ENCODER)
OPUSORUS_CTL(opus_custom_decoder_ctl, OpusCustomDecoder, KIND_CUSTOM_DECODER)
#endif

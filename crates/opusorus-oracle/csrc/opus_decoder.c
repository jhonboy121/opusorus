/* Oracle C shims for unit opus_decoder (src/opus_decoder.c, src/opus_multistream_decoder.c,
   src/opus_projection_decoder.c).

   The public API is used directly from Rust (opusorus_oracle::sys). These shims only expose what
   the public API cannot: a dump of the private OpusDecoder struct (its definition is copied
   verbatim from src/opus_decoder.c, including the same #ifdefs, so the layout matches the
   library build), and the per-stream decoder of a multistream/projection decoder.

   Built in both oracles: the fixed-point OpusDecoder has no softclip_mem (dumped as zeros). */
// oracle-build: any

#include <stdarg.h>
#include "celt.h"
#include "opus.h"
#include "opus_multistream.h"
#include "opus_projection.h"
#include "entdec.h"
#include "modes.h"
#include "API.h"
#include "stack_alloc.h"
#include "float_cast.h"
#include "opus_private.h"
#include "os_support.h"
#include "structs.h"
#include "define.h"
#include "mathops.h"
#include "cpu_support.h"

#ifdef ENABLE_DEEP_PLC
#include "dred_rdovae_dec_data.h"
#include "dred_rdovae_dec.h"
#endif

#ifdef ENABLE_OSCE
#include "osce.h"
#endif

/* Verbatim copy of struct OpusDecoder (src/opus_decoder.c). */
struct oracle_odec_OpusDecoder {
   int          celt_dec_offset;
   int          silk_dec_offset;
   int          channels;
   opus_int32   Fs;          /** Sampling rate (at the API level) */
   silk_DecControlStruct DecControl;
   int          decode_gain;
   int          complexity;
   int          ignore_extensions;
   int          arch;
#ifdef ENABLE_DEEP_PLC
    LPCNetPLCState lpcnet;
#endif

   /* Everything beyond this point gets cleared on a reset */
   int          stream_channels;

   int          bandwidth;
   int          mode;
   int          prev_mode;
   int          frame_size;
   int          prev_redundancy;
   int          last_packet_duration;
#ifndef FIXED_POINT
   opus_val16   softclip_mem[2];
#endif

   opus_uint32  rangeFinal;
};

int oracle_odec_has_qext(void)
{
#ifdef ENABLE_QEXT
   return 1;
#else
   return 0;
#endif
}

/* Dumps the Opus-level decoder state. `iout` receives 21 ints, `fout` 2 floats:
   channels, Fs, DecControl.{nChannelsAPI, nChannelsInternal, API_sampleRate,
   internalSampleRate, payloadSize_ms, prevPitchLag, enable_deep_plc}, decode_gain, complexity,
   ignore_extensions, stream_channels, bandwidth, mode, prev_mode, frame_size, prev_redundancy,
   last_packet_duration, rangeFinal (bits), 0; softclip_mem[0..2] (zeros in the fixed-point
   build, which has no soft clipper). */
void oracle_odec_dump(const OpusDecoder *dec, opus_int32 *iout, float *fout)
{
   const struct oracle_odec_OpusDecoder *st = (const struct oracle_odec_OpusDecoder *)dec;
   int k = 0;
   iout[k++] = st->channels;
   iout[k++] = st->Fs;
   iout[k++] = st->DecControl.nChannelsAPI;
   iout[k++] = st->DecControl.nChannelsInternal;
   iout[k++] = st->DecControl.API_sampleRate;
   iout[k++] = st->DecControl.internalSampleRate;
   iout[k++] = st->DecControl.payloadSize_ms;
   iout[k++] = st->DecControl.prevPitchLag;
   iout[k++] = st->DecControl.enable_deep_plc;
   iout[k++] = st->decode_gain;
   iout[k++] = st->complexity;
   iout[k++] = st->ignore_extensions;
   iout[k++] = st->stream_channels;
   iout[k++] = st->bandwidth;
   iout[k++] = st->mode;
   iout[k++] = st->prev_mode;
   iout[k++] = st->frame_size;
   iout[k++] = st->prev_redundancy;
   iout[k++] = st->last_packet_duration;
   iout[k++] = (opus_int32)st->rangeFinal;
   iout[k++] = 0;
#ifndef FIXED_POINT
   fout[0] = st->softclip_mem[0];
   fout[1] = st->softclip_mem[1];
#else
   fout[0] = 0;
   fout[1] = 0;
#endif
}

/* OPUS_MULTISTREAM_GET_DECODER_STATE through the multistream CTL (NULL on error). */
OpusDecoder *oracle_odec_ms_stream(OpusMSDecoder *st, int stream_id)
{
   OpusDecoder *dec = NULL;
   if (opus_multistream_decoder_ctl(st, OPUS_MULTISTREAM_GET_DECODER_STATE(stream_id, &dec)) != OPUS_OK)
      return NULL;
   return dec;
}

/* OPUS_MULTISTREAM_GET_DECODER_STATE through the projection CTL (NULL on error). */
OpusDecoder *oracle_odec_proj_stream(OpusProjectionDecoder *st, int stream_id)
{
   OpusDecoder *dec = NULL;
   if (opus_projection_decoder_ctl(st, OPUS_MULTISTREAM_GET_DECODER_STATE(stream_id, &dec)) != OPUS_OK)
      return NULL;
   return dec;
}

/* Raw multistream/projection GET_DECODER_STATE return codes (for invalid stream ids). */
int oracle_odec_ms_stream_ret(OpusMSDecoder *st, int stream_id)
{
   OpusDecoder *dec = NULL;
   return opus_multistream_decoder_ctl(st, OPUS_MULTISTREAM_GET_DECODER_STATE(stream_id, &dec));
}

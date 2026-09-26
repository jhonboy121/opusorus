/* Oracle C shims for unit dnn_integration (deep PLC / DRED / OSCE wired into the Opus codec).

   The public API (opus_decoder_*, opus_encoder_*, opus_dred_*) is used directly from Rust. These
   shims only expose what it cannot: a dump of the DNN-related private state of an OpusDecoder
   (struct copied verbatim from src/opus_decoder.c with the same #ifdefs, like the opus_decoder
   shim) and of an OpusDRED (struct from dnn/dred_decoder.h).

   Compiled only with ENABLE_DEEP_PLC (every DNN feature defines it). */
#ifdef ENABLE_DEEP_PLC

#include <string.h>
#include "celt.h"
#include "opus.h"
#include "API.h"
#include "opus_private.h"
#include "structs.h"
#include "lpcnet.h"
#include "lpcnet_private.h"

#ifdef ENABLE_OSCE
#include "osce.h"
#endif
#ifdef ENABLE_DRED
#include "dred_decoder.h"
#endif

/* Verbatim copy of struct OpusDecoder (src/opus_decoder.c). */
struct oracle_di_OpusDecoder {
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

#define ORACLE_DI_DEC_INTS 13

/* DNN state of an OpusDecoder. `iout` receives ORACLE_DI_DEC_INTS ints:
   lpcnet.{loaded, analysis_gap, fec_read_pos, fec_fill_pos, fec_skip, analysis_pos, predict_pos,
   blend, loss_count}, DecControl.{osce_method, enable_osce_bwe, osce_extended_mode,
   prev_osce_extended_mode} (0 without ENABLE_OSCE); `fout` receives lpcnet.pcm[PLC_BUF_SIZE]
   then lpcnet.features[NB_TOTAL_FEATURES]. Returns the number of floats. */
int oracle_di_dec_dump(const OpusDecoder *dec, int *iout, float *fout)
{
   const struct oracle_di_OpusDecoder *st = (const struct oracle_di_OpusDecoder *)dec;
   const LPCNetPLCState *l = &st->lpcnet;
   iout[0] = l->loaded;
   iout[1] = l->analysis_gap;
   iout[2] = l->fec_read_pos;
   iout[3] = l->fec_fill_pos;
   iout[4] = l->fec_skip;
   iout[5] = l->analysis_pos;
   iout[6] = l->predict_pos;
   iout[7] = l->blend;
   iout[8] = l->loss_count;
#ifdef ENABLE_OSCE
   iout[9] = st->DecControl.osce_method;
   iout[10] = st->DecControl.enable_osce_bwe;
   iout[11] = st->DecControl.osce_extended_mode;
   iout[12] = st->DecControl.prev_osce_extended_mode;
#else
   iout[9] = iout[10] = iout[11] = iout[12] = 0;
#endif
   if (fout) {
      memcpy(fout, l->pcm, sizeof(l->pcm));
      memcpy(fout + PLC_BUF_SIZE, l->features, sizeof(l->features));
   }
   return PLC_BUF_SIZE + NB_TOTAL_FEATURES;
}

int oracle_di_has_dred(void)
{
#ifdef ENABLE_DRED
   return 1;
#else
   return 0;
#endif
}

int oracle_di_has_osce(void)
{
#ifdef ENABLE_OSCE
   return 1;
#else
   return 0;
#endif
}

#ifdef ENABLE_DRED
/* Layout sizes of OpusDRED: fec_features, state, latents (floats). */
void oracle_di_dred_sizes(int *out)
{
   out[0] = 2*DRED_NUM_REDUNDANCY_FRAMES*DRED_NUM_FEATURES;
   out[1] = DRED_STATE_DIM;
   out[2] = (DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1);
}

/* Dumps an OpusDRED: the three float arrays (sizes from oracle_di_dred_sizes) and
   {nb_latents, process_stage, dred_offset}. */
void oracle_di_dred_dump(const OpusDRED *d, float *fec, float *state, float *latents, int *ints)
{
   memcpy(fec, d->fec_features, sizeof(d->fec_features));
   memcpy(state, d->state, sizeof(d->state));
   memcpy(latents, d->latents, sizeof(d->latents));
   ints[0] = d->nb_latents;
   ints[1] = d->process_stage;
   ints[2] = d->dred_offset;
}

/* Zeroes an OpusDRED (opus_dred_alloc leaves it uninitialized; the Rust port zeroes it). */
void oracle_di_dred_clear(OpusDRED *d)
{
   memset(d, 0, sizeof(*d));
}
#endif

#endif /* ENABLE_DEEP_PLC */

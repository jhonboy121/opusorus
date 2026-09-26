/* FLOAT2INT16 / FLOAT2INT24 for the shims' own float test-signal conversions (FLOAT2RES):
 * celt/float_cast.h compiles them out with DISABLE_FLOAT_API (feature `disable-float-api`).
 * Identical copies of the float_cast.h definitions; include after the libopus headers. */
#ifndef ORACLE_FLOAT_CAST_H
#define ORACLE_FLOAT_CAST_H
#ifdef DISABLE_FLOAT_API
static OPUS_INLINE opus_int16 FLOAT2INT16(float x)
{
   x = x*CELT_SIG_SCALE;
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return (opus_int16)float2int(x);
}

static OPUS_INLINE opus_int32 FLOAT2INT24(float x)
{
   x = x*(CELT_SIG_SCALE*256.f);
   x = MAX32(x, -16777216);
   x = MIN32(x, 16777216);
   return float2int(x);
}
#endif
#endif

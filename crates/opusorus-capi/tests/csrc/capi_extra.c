/* Extra C-level checks of the opusorus C ABI (run by tests/c_suite.rs).
 *
 * Covers what upstream's tests only touch indirectly: byte copies of state blocks (clone and
 * move), multistream stream handles, NULL / wrong-type handles, the repacketizer writing over
 * its own input, projection matrices, DRED stubs, and (with CUSTOM_MODES) a non-48 kHz custom
 * mode. */

/* Public-API view of the headers (declares opus_custom_*_get_size/init). */
#undef OPUS_BUILD

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_multistream.h"
#include "opus_projection.h"
#ifdef CUSTOM_MODES
#include "opus_custom.h"
#endif

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
   fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

#define FS 48000
#define FRAME 960
#define MAXP 1500

static void sine(opus_int16 *pcm, int n, int ch, int k)
{
   int i, c;
   for (i = 0; i < n; i++)
      for (c = 0; c < ch; c++)
         pcm[i*ch + c] = (opus_int16)(8000*sin(0.01*(i + k*n)*(c + 1)) + 300*((i*7 + k) % 13));
}

/* Encodes `count` 20 ms stereo frames into packets[], returns their lengths. */
static void make_packets(unsigned char packets[][MAXP], int *lens, int count)
{
   int err, k;
   opus_int16 pcm[FRAME*2];
   OpusEncoder *enc = opus_encoder_create(FS, 2, OPUS_APPLICATION_AUDIO, &err);
   CHECK(err == OPUS_OK && enc != NULL);
   CHECK(opus_encoder_ctl(enc, OPUS_SET_BITRATE(64000)) == OPUS_OK);
   for (k = 0; k < count; k++) {
      sine(pcm, FRAME, 2, k);
      lens[k] = opus_encode(enc, pcm, FRAME, packets[k], MAXP);
      CHECK(lens[k] > 0);
   }
   opus_encoder_destroy(enc);
}

static int decode(OpusDecoder *dec, const unsigned char *p, int len, opus_int16 *out)
{
   return opus_decode(dec, p, len, out, FRAME, 0);
}

static void test_state_copies(void)
{
   unsigned char packets[6][MAXP];
   int lens[6], err, size;
   opus_int16 a[FRAME*2], b[FRAME*2], r[FRAME*2];
   OpusDecoder *dec, *copy, *ref, *moved;
   opus_uint32 ra, rb;

   make_packets(packets, lens, 6);
   dec = opus_decoder_create(FS, 2, &err);
   ref = opus_decoder_create(FS, 2, &err);
   size = opus_decoder_get_size(2);
   CHECK(decode(dec, packets[0], lens[0], a) == FRAME);
   CHECK(decode(ref, packets[0], lens[0], r) == FRAME);

   /* Clone: the copy continues from the same state, independently. (A copy becomes a state
      of its own when first used; see the handle module docs.) */
   copy = (OpusDecoder *)malloc(size);
   memcpy(copy, dec, size);
   CHECK(decode(copy, packets[1], lens[1], b) == FRAME);
   CHECK(decode(dec, packets[1], lens[1], a) == FRAME);
   CHECK(memcmp(a, b, sizeof(a)) == 0);
   CHECK(opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&ra)) == OPUS_OK);
   CHECK(opus_decoder_ctl(copy, OPUS_GET_FINAL_RANGE(&rb)) == OPUS_OK);
   CHECK(ra == rb);
   /* Diverge the copy; the original must not notice. */
   CHECK(decode(copy, NULL, 0, b) == FRAME);
   CHECK(decode(copy, packets[3], lens[3], b) == FRAME);
   CHECK(decode(ref, packets[1], lens[1], r) == FRAME);
   CHECK(decode(dec, packets[2], lens[2], a) == FRAME);
   CHECK(decode(ref, packets[2], lens[2], r) == FRAME);
   CHECK(memcmp(a, r, sizeof(a)) == 0);
   opus_decoder_destroy(copy); /* malloc'd by us: destroy frees it, as in libopus */

   /* Move: copy to new memory, scramble and destroy the old block. */
   moved = (OpusDecoder *)malloc(size);
   memcpy(moved, dec, size);
   memset(dec, 255, size);
   opus_decoder_destroy(dec);
   CHECK(decode(moved, packets[3], lens[3], a) == FRAME);
   CHECK(decode(ref, packets[3], lens[3], r) == FRAME);
   CHECK(memcmp(a, r, sizeof(a)) == 0);

   /* Backup / trial decode / continue, as test_opus_decode does. */
   copy = (OpusDecoder *)malloc(size);
   memcpy(copy, moved, size);
   CHECK(decode(copy, packets[5], lens[5], b) == FRAME);
   memcpy(copy, moved, size);
   CHECK(decode(copy, packets[4], lens[4], b) == FRAME);
   CHECK(decode(moved, packets[4], lens[4], a) == FRAME);
   CHECK(memcmp(a, b, sizeof(a)) == 0);
   free(copy); /* never destroyed: only its last clone leaks, like a C state freed with free() */

   /* init in caller memory, then re-init in place. */
   copy = (OpusDecoder *)malloc(size);
   CHECK(opus_decoder_init(copy, 16000, 1) == OPUS_OK);
   CHECK(opus_decoder_init(copy, FS, 2) == OPUS_OK);
   CHECK(decode(copy, packets[0], lens[0], b) == FRAME);
   opus_decoder_destroy(copy);

   opus_decoder_destroy(moved);
   opus_decoder_destroy(ref);
}

static void test_bad_handles(void)
{
   int err;
   opus_int32 v;
   opus_int16 pcm[FRAME*2];
   unsigned char p[MAXP];
   OpusEncoder *enc = opus_encoder_create(FS, 1, OPUS_APPLICATION_VOIP, &err);
   CHECK(opus_decode(NULL, NULL, 0, pcm, FRAME, 0) == OPUS_BAD_ARG);
   CHECK(opus_encode(NULL, pcm, FRAME, p, MAXP) == OPUS_BAD_ARG);
   CHECK(opus_encoder_ctl(NULL, OPUS_GET_BITRATE(&v)) == OPUS_BAD_ARG);
   /* An encoder is not a decoder. */
   CHECK(opus_decoder_ctl((OpusDecoder *)enc, OPUS_GET_SAMPLE_RATE(&v)) == OPUS_INVALID_STATE);
   CHECK(opus_encoder_ctl(enc, OPUS_GET_SAMPLE_RATE(&v)) == OPUS_OK && v == FS);
   /* Unknown and wrong-object requests. */
   CHECK(opus_encoder_ctl(enc, 12345) == OPUS_UNIMPLEMENTED);
   CHECK(opus_encoder_ctl(enc, OPUS_GET_GAIN(&v)) == OPUS_UNIMPLEMENTED);
   CHECK(opus_encode(enc, NULL, FRAME, p, MAXP) == OPUS_BAD_ARG);
   CHECK(opus_encode(enc, pcm, FRAME, NULL, MAXP) == OPUS_BAD_ARG);
   opus_encoder_destroy(enc);
   opus_encoder_destroy(NULL);
   CHECK(strncmp(opus_get_version_string(), "libopus ", 8) == 0);
   CHECK(strcmp(opus_strerror(OPUS_INVALID_STATE), "invalid state") == 0);
   CHECK(strcmp(opus_strerror(1), "unknown error") == 0);
}

static void test_stream_handles(void)
{
   int err, streams, coupled;
   unsigned char mapping[6];
   opus_int32 v;
   OpusEncoder *s0 = NULL, *s3 = NULL;
   OpusDecoder *d0 = NULL;
   OpusMSEncoder *ms = opus_multistream_surround_encoder_create(FS, 6, 1, &streams, &coupled,
      mapping, OPUS_APPLICATION_AUDIO, &err);
   OpusMSDecoder *msd;
   CHECK(err == OPUS_OK && ms != NULL);
   CHECK(streams == 4 && coupled == 2);
   CHECK(opus_multistream_encoder_ctl(ms, OPUS_MULTISTREAM_GET_ENCODER_STATE(0, &s0)) == OPUS_OK);
   CHECK(opus_multistream_encoder_ctl(ms, OPUS_MULTISTREAM_GET_ENCODER_STATE(3, &s3)) == OPUS_OK);
   CHECK(opus_multistream_encoder_ctl(ms, OPUS_MULTISTREAM_GET_ENCODER_STATE(4, &s3)) == OPUS_BAD_ARG);
   CHECK(opus_encoder_ctl(s0, OPUS_SET_COMPLEXITY(3)) == OPUS_OK);
   CHECK(opus_multistream_encoder_ctl(ms, OPUS_GET_COMPLEXITY(&v)) == OPUS_OK && v == 3);
   CHECK(opus_encoder_ctl(s3, OPUS_GET_SAMPLE_RATE(&v)) == OPUS_OK && v == FS);
   opus_encoder_destroy(s0); /* ignored: interior handle */
   CHECK(opus_encoder_ctl(s0, OPUS_GET_COMPLEXITY(&v)) == OPUS_OK && v == 3);

   msd = opus_multistream_decoder_create(FS, 6, streams, coupled, mapping, &err);
   CHECK(err == OPUS_OK && msd != NULL);
   CHECK(opus_multistream_decoder_ctl(msd, OPUS_MULTISTREAM_GET_DECODER_STATE(1, &d0)) == OPUS_OK);
   CHECK(opus_decoder_ctl(d0, OPUS_SET_GAIN(-256)) == OPUS_OK);
   CHECK(opus_decoder_ctl(d0, OPUS_GET_GAIN(&v)) == OPUS_OK && v == -256);
   CHECK(opus_multistream_decoder_ctl(msd, OPUS_MULTISTREAM_GET_DECODER_STATE(0, (OpusDecoder **)NULL)) == OPUS_BAD_ARG);
   opus_multistream_decoder_destroy(msd);
   opus_multistream_encoder_destroy(ms);
}

static void test_repacketizer_in_place(void)
{
   unsigned char packets[2][MAXP];
   unsigned char buf[2*MAXP];
   int lens[2], n1, n2, len, err;
   opus_int16 a[FRAME*2*2], b[FRAME*2*2];
   OpusRepacketizer *rp = opus_repacketizer_create();
   OpusDecoder *d1 = opus_decoder_create(FS, 2, &err), *d2 = opus_decoder_create(FS, 2, &err);
   make_packets(packets, lens, 2);
   memcpy(buf, packets[0], lens[0]);
   memcpy(buf + lens[0], packets[1], lens[1]);
   CHECK(opus_repacketizer_cat(rp, buf, lens[0]) == OPUS_OK);
   CHECK(opus_repacketizer_cat(rp, buf + lens[0], lens[1]) == OPUS_OK);
   CHECK(opus_repacketizer_get_nb_frames(rp) == 2);
   /* Output over the input packets (libopus moves data with memmove). */
   len = opus_repacketizer_out(rp, buf, sizeof(buf));
   CHECK(len > 0);
   n1 = opus_decode(d1, buf, len, a, FRAME*2, 0);
   CHECK(n1 == 2*FRAME);
   n2 = opus_decode(d2, packets[0], lens[0], b, FRAME, 0);
   n2 += opus_decode(d2, packets[1], lens[1], b + FRAME*2, FRAME, 0);
   CHECK(n2 == 2*FRAME);
   CHECK(memcmp(a, b, sizeof(a)) == 0);
   opus_repacketizer_destroy(rp);
   opus_decoder_destroy(d1);
   opus_decoder_destroy(d2);
}

static void test_projection(void)
{
   int err, streams, coupled, size, gain, len, ret;
   unsigned char *matrix, packet[MAXP*4];
   opus_int16 pcm[FRAME*9], out[FRAME*9];
   OpusProjectionEncoder *enc = opus_projection_ambisonics_encoder_create(FS, 9, 3, &streams,
      &coupled, OPUS_APPLICATION_AUDIO, &err);
   OpusProjectionDecoder *dec;
   CHECK(err == OPUS_OK && enc != NULL && streams == 5 && coupled == 4);
   CHECK(opus_projection_encoder_ctl(enc, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE(&size)) == OPUS_OK);
   CHECK(size == 9*(streams + coupled)*2);
   CHECK(opus_projection_encoder_ctl(enc, OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN(&gain)) == OPUS_OK);
   matrix = (unsigned char *)malloc(size);
   CHECK(opus_projection_encoder_ctl(enc, OPUS_PROJECTION_GET_DEMIXING_MATRIX(matrix, size - 1)) == OPUS_BAD_ARG);
   CHECK(opus_projection_encoder_ctl(enc, OPUS_PROJECTION_GET_DEMIXING_MATRIX(matrix, size)) == OPUS_OK);
   dec = opus_projection_decoder_create(FS, 9, streams, coupled, matrix, size, &err);
   CHECK(err == OPUS_OK && dec != NULL);
   sine(pcm, FRAME, 9, 0);
   len = opus_projection_encode(enc, pcm, FRAME, packet, sizeof(packet));
   CHECK(len > 0);
   ret = opus_projection_decode(dec, packet, len, out, FRAME, 0);
   CHECK(ret == FRAME);
   CHECK(opus_projection_decoder_create(FS, 9, streams, coupled, matrix, size - 2, &err) == NULL
      && err == OPUS_BAD_ARG);
   free(matrix);
   opus_projection_decoder_destroy(dec);
   opus_projection_encoder_destroy(enc);
}

static void test_dred_stubs(void)
{
   int err = 1;
   OpusDREDDecoder *dd = opus_dred_decoder_create(&err);
   CHECK(err == OPUS_OK && dd != NULL);
   CHECK(opus_dred_decoder_ctl(dd, OPUS_SET_DNN_BLOB("x", 1)) == OPUS_UNIMPLEMENTED);
   CHECK(opus_dred_get_size() == 0);
   CHECK(opus_dred_alloc(&err) == NULL && err == OPUS_UNIMPLEMENTED);
   opus_dred_decoder_destroy(dd);
}

#ifdef CUSTOM_MODES
static void test_custom_16k(void)
{
   int err, k, len, n;
   opus_int16 pcm[320*2], out[320*2];
   unsigned char packet[MAXP];
   opus_uint32 r1, r2;
   OpusCustomMode *mode = opus_custom_mode_create(16000, 320, &err);
   OpusCustomEncoder *enc;
   OpusCustomDecoder *dec, *copy;
   CHECK(err == OPUS_OK && mode != NULL);
   CHECK(opus_custom_mode_create(16000, 7, &err) == NULL && err == OPUS_BAD_ARG);
   enc = opus_custom_encoder_create(mode, 2, &err);
   CHECK(err == OPUS_OK && enc != NULL);
   dec = opus_custom_decoder_create(mode, 2, &err);
   CHECK(err == OPUS_OK && dec != NULL);
   CHECK(opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(48000)) == OPUS_OK);
   CHECK(opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(11)) == OPUS_BAD_ARG);
   copy = (OpusCustomDecoder *)malloc(opus_custom_decoder_get_size(mode, 2));
   CHECK(opus_custom_decoder_init(copy, mode, 2) == OPUS_OK);
   for (k = 0; k < 20; k++) {
      sine(pcm, 320, 2, k);
      len = opus_custom_encode(enc, pcm, 320, packet, 200);
      CHECK(len > 0);
      n = opus_custom_decode(dec, packet, len, out, 320);
      CHECK(n == 320);
      n = opus_custom_decode(copy, packet, len, out, 320);
      CHECK(n == 320);
   }
   CHECK(opus_custom_encode(enc, pcm, 321, packet, 200) == OPUS_BAD_ARG);
   CHECK(opus_custom_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&r1)) == OPUS_OK);
   CHECK(opus_custom_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&r2)) == OPUS_OK);
   CHECK(r1 == r2);
   CHECK(opus_custom_decode(dec, NULL, 0, out, 320) == 320); /* PLC */
   opus_custom_decoder_destroy(copy);
   opus_custom_decoder_destroy(dec);
   opus_custom_encoder_destroy(enc);
   opus_custom_mode_destroy(mode);
}
#endif

int main(void)
{
   test_state_copies();
   test_bad_handles();
   test_stream_handles();
   test_repacketizer_in_place();
   test_projection();
   test_dred_stubs();
#ifdef CUSTOM_MODES
   test_custom_16k();
#endif
   if (failures) {
      fprintf(stderr, "%d check(s) failed\n", failures);
      return 1;
   }
   printf("All opusorus C ABI extra checks passed.\n");
   return 0;
}

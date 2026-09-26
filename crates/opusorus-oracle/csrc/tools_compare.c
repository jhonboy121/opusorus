/* Oracle shim for the tools_compare unit: runs the unmodified src/opus_compare.c and
   src/qext_compare.c `main` functions (renamed) with a given argv and captures what they print
   on stderr, plus the full-precision value of every floating-point printf argument.

   Both tools are #included into this one translation unit, so their file-static helpers,
   tables and macros are renamed/undefined between the two inclusions. qext_compare.c includes
   celt/mini_kfft.c, whose external symbols are renamed so they cannot clash with the copy
   compiled into the library for ENABLE_QEXT builds.

   Captured state is thread-local so Rust tests can run in parallel. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>

#define TC_TEXT_CAP 8192
#define TC_MAX_VALS 16

static _Thread_local char tc_text[TC_TEXT_CAP];
static _Thread_local size_t tc_len;
static _Thread_local double tc_vals[TC_MAX_VALS];
static _Thread_local int tc_nvals;

static void tc_reset(void) {
  tc_len = 0;
  tc_text[0] = 0;
  tc_nvals = 0;
}

/* Records the printf arguments of `fmt`, keeping every double (%f/%e/%g). */
static void tc_record_args(const char *fmt, va_list ap) {
  const char *p = fmt;
  while (*p) {
    int longs = 0;
    if (*p++ != '%') continue;
    if (*p == '%') { p++; continue; }
    while (*p && strchr("-+ #0", *p)) p++;
    while (*p && ((*p >= '0' && *p <= '9') || *p == '.')) p++;
    while (*p == 'l' || *p == 'h' || *p == 'z') { if (*p == 'l') longs++; p++; }
    switch (*p) {
      case 'f': case 'e': case 'g': {
        double v = va_arg(ap, double);
        if (tc_nvals < TC_MAX_VALS) tc_vals[tc_nvals++] = v;
        break;
      }
      case 'd': case 'i': case 'u': case 'x':
        if (longs) (void)va_arg(ap, long); else (void)va_arg(ap, int);
        break;
      case 's': (void)va_arg(ap, const char *); break;
      case 'c': (void)va_arg(ap, int); break;
      default: break;
    }
    if (*p) p++;
  }
}

static int tc_fprintf(FILE *f, const char *fmt, ...) {
  va_list ap;
  int n;
  (void)f;
  va_start(ap, fmt);
  n = vsnprintf(tc_text + tc_len, TC_TEXT_CAP - tc_len, fmt, ap);
  va_end(ap);
  if (n > 0) {
    tc_len += (size_t)n;
    if (tc_len >= TC_TEXT_CAP) tc_len = TC_TEXT_CAP - 1;
  }
  va_start(ap, fmt);
  tc_record_args(fmt, ap);
  va_end(ap);
  return n;
}

#define fprintf tc_fprintf

/* ---- src/opus_compare.c ---- */
#define main tc_opus_compare_main
#define check_alloc tc_oc_check_alloc
#define opus_malloc tc_oc_opus_malloc
#define opus_realloc tc_oc_opus_realloc
#define read_pcm16 tc_oc_read_pcm16
#define band_energy tc_oc_band_energy
#define BANDS tc_oc_BANDS
#include "opus_compare.c"
#undef main
#undef check_alloc
#undef opus_malloc
#undef opus_realloc
#undef read_pcm16
#undef band_energy
#undef BANDS
#undef OPUS_PI
#undef OPUS_COSF
#undef OPUS_SINF
#undef NBANDS
#undef NFREQS
#undef TEST_WIN_SIZE
#undef TEST_WIN_STEP

/* ---- src/qext_compare.c (+ celt/mini_kfft.c) ---- */
#define main tc_qext_compare_main
#define check_alloc tc_qc_check_alloc
#define opus_malloc tc_qc_opus_malloc
#define opus_realloc tc_qc_opus_realloc
#define read_pcm tc_qc_read_pcm
#define band_energy tc_qc_band_energy
#define BANDS tc_qc_BANDS
#define format_size tc_qc_format_size
#define usage tc_qc_usage
#define kf_work tc_mini_kf_work
#define kf_factor tc_mini_kf_factor
#define mini_kiss_fft_alloc tc_mini_kiss_fft_alloc
#define mini_kiss_fft_stride tc_mini_kiss_fft_stride
#define mini_kiss_fft tc_mini_kiss_fft
#define mini_kiss_fftr_alloc tc_mini_kiss_fftr_alloc
#define mini_kiss_fftr tc_mini_kiss_fftr
#include "qext_compare.c"
#undef main

/* Runs a tool with `argc`/`argv`; returns its exit code and the captured stderr text (NUL
   terminated, truncated to `text_cap`) and printed double arguments. */
static int tc_run(int which, int argc, const char **argv, char *text, size_t text_cap,
                  double *vals, int max_vals, int *nvals) {
  int ret;
  int i;
  tc_reset();
  ret = which == 0 ? tc_opus_compare_main(argc, argv) : tc_qext_compare_main(argc, argv);
  if (text_cap > 0) {
    size_t n = tc_len < text_cap - 1 ? tc_len : text_cap - 1;
    memcpy(text, tc_text, n);
    text[n] = 0;
  }
  for (i = 0; i < tc_nvals && i < max_vals; i++) vals[i] = tc_vals[i];
  *nvals = tc_nvals < max_vals ? tc_nvals : max_vals;
  return ret;
}

int oracle_tools_opus_compare(int argc, const char **argv, char *text, size_t text_cap,
                              double *vals, int max_vals, int *nvals) {
  return tc_run(0, argc, argv, text, text_cap, vals, max_vals, nvals);
}

int oracle_tools_qext_compare(int argc, const char **argv, char *text, size_t text_cap,
                              double *vals, int max_vals, int *nvals) {
  return tc_run(1, argc, argv, text, text_cap, vals, max_vals, nvals);
}

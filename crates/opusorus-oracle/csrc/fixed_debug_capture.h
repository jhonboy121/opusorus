/* Force-included (`-include`) into every oracle source of a FIXED_DEBUG build (feature
 * `fixed-point-debug`, see build.rs): after <stdio.h>, `fprintf` is renamed so the diagnostics of
 * celt/fixed_debug.h and silk/MacroDebug.h (`fprintf(stderr, ...)`, the only fprintf calls
 * compiled into libopus) are captured by csrc/fixed_debug.c instead of printed. */
#ifndef ORACLE_FIXED_DEBUG_CAPTURE_H
#define ORACLE_FIXED_DEBUG_CAPTURE_H
#include <stdio.h>
int oracle_fixed_debug_fprintf(FILE *stream, const char *fmt, ...);
#undef fprintf
#define fprintf oracle_fixed_debug_fprintf
#endif

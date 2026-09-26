/* Oracle marker for libopus --enable-dnn-debug-float (oracle feature `dnn-debug-float`): a DNN
   build without DISABLE_DEBUG_FLOAT, whose model tables keep float copies of the int8 layers.

   The layer names alone cannot tell the builds apart (the init_* functions name the float
   arrays either way), so tests that look for the matching oracle libopus.a (e.g. to link a C
   opus_demo against it) search for this string. */
// oracle-build: float
#if defined(ENABLE_DEEP_PLC) && !defined(DISABLE_DEBUG_FLOAT)
const char oracle_dnn_debug_float_marker[] = "opusorus-oracle: dnn-debug-float build";
#endif

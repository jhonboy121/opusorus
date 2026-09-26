#!/usr/bin/env bash
# Generates the libopus DNN weight blob (the `OPUS_SET_DNN_BLOB` format read by
# `parse_weights`) from the fetched model data, for opusorus' `dnn-weights-embedded` feature
# (`OPUSORUS_DNN_BLOB=<blob> cargo build --features opusorus/dnn-weights-embedded,...`) or for
# runtime loading (`Decoder::set_dnn_blob`, `opus_demo`'s `weights_blob.bin`).
#
# Usage: scripts/gen_dnn_blob.sh [--debug-float] [output]
#   default output: target/dnn/weights_blob.bin (--debug-float: weights_blob_debug_float.bin)
#
# --debug-float: the blob of a libopus `--enable-dnn-debug-float` build (DISABLE_DEBUG_FLOAT
# undefined): every int8-quantized layer also carries its float weights, which the DNN then
# computes with (opusorus feature `dnn-debug-float`).
#
# The blob is written by upstream's own serializer (vendor/libopus/dnn/write_lpcnet_weights.c,
# compiled together with the generated *_data.c model files) and holds every model of a full
# DNN build: pitchdnn, fargan, plcmodel (deep PLC), rdovaeenc, rdovaedec (DRED), lace, nolace
# and bbwenet (OSCE). Upstream's `main` omits BBWENet (needed by OSCE BWE); this wrapper adds
# it. The model data must have been extracted first (scripts/fetch_dnn_models.sh).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
debug_float=0
if [ "${1:-}" = "--debug-float" ]; then
    debug_float=1
    shift
fi
if [ "$debug_float" = 1 ]; then
    out="${1:-$root/target/dnn/weights_blob_debug_float.bin}"
    debug_float_def=()
else
    out="${1:-$root/target/dnn/weights_blob.bin}"
    debug_float_def=(-DDISABLE_DEBUG_FLOAT)
fi
dnn="$root/vendor/libopus/dnn"
for f in pitchdnn fargan plc dred_rdovae_enc dred_rdovae_dec lace nolace bbwenet; do
    if [ ! -f "$dnn/${f}_data.c" ]; then
        echo "error: $dnn/${f}_data.c not found; run scripts/fetch_dnn_models.sh first" >&2
        exit 1
    fi
done
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat > "$work/gen_dnn_blob.c" <<'EOF'
/* Upstream write_lpcnet_weights.c with its main() renamed, plus BBWENet. */
#define main upstream_write_lpcnet_weights_main
#include "write_lpcnet_weights.c"
#undef main
#include "bbwenet_data.c"

int main(int argc, char **argv)
{
  FILE *fout;
  if (argc != 2) {
    fprintf(stderr, "usage: %s <output>\n", argv[0]);
    return 1;
  }
  fout = fopen(argv[1], "wb");
  if (fout == NULL) {
    perror(argv[1]);
    return 1;
  }
  /* Same order as upstream's main (ENABLE_OSCE build), then BBWENet. */
  write_weights(pitchdnn_arrays, fout);
  write_weights(fargan_arrays, fout);
  write_weights(plcmodel_arrays, fout);
  write_weights(rdovaeenc_arrays, fout);
  write_weights(rdovaedec_arrays, fout);
  write_weights(lacelayers_arrays, fout);
  write_weights(nolacelayers_arrays, fout);
  write_weights(bbwenetlayers_arrays, fout);
  if (fclose(fout) != 0) {
    perror(argv[1]);
    return 1;
  }
  return 0;
}
EOF
cc="${CC:-cc}"
# DUMP_BINARY_WEIGHTS (as upstream's dump_weights_blob target): only the arrays, no init_*().
# DISABLE_DEBUG_FLOAT (libopus' default, without --enable-dnn-debug-float; not with
# --debug-float): no float copies of the int8 weights, which would change the arithmetic.
# -O0: the program only copies the arrays; optimizing 90 MB of initializers is slow.
"$cc" -O0 -w -DDUMP_BINARY_WEIGHTS ${debug_float_def[@]+"${debug_float_def[@]}"} -DENABLE_OSCE -DENABLE_OSCE_BWE \
    -I"$dnn" -I"$root/vendor/libopus/celt" -I"$root/vendor/libopus/include" \
    -I"$root/vendor/libopus/src" -I"$root/vendor/libopus" \
    "$work/gen_dnn_blob.c" -o "$work/gen_dnn_blob" -lm
"$work/gen_dnn_blob" "$out"
echo "DNN weight blob written to $out ($(wc -c < "$out") bytes)"

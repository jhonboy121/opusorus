#!/usr/bin/env bash
# Official Opus decoder conformance procedure with the Rust tools: runs libopus' own
# tests/run_vectors.sh (RFC 6716 / RFC 8251 vectors) for each rate and, with --hd,
# tests/run_opushd_vectors.sh (Opus HD / QEXT vectors), with OPUS_DEMO, OPUS_COMPARE and
# QEXT_COMPARE pointing at the opusorus-tools binaries (release build).
#
# Usage: scripts/run_vectors.sh [--hd] [--fixed|--fixed-res24] [rate ...]
#        (default rates: 48000 24000 16000 12000 8000)
# --fixed / --fixed-res24 build the tools on the fixed-point decoder (16-bit / 24-bit
# resolution). With --hd --fixed the RFC vectors at 96 kHz fail qext_compare's rms threshold:
# a 16-bit output cannot get below the 1/sqrt(12) LSB quantisation floor (use --fixed-res24).
#
# Vectors: $OPUSORUS_VECTORS, else the first testdata/vectors found in this checkout or a parent
# directory (fetch them with scripts/fetch_vectors.sh). The upstream scripts print
# "No test vectors found" and succeed if the directory is missing.
#
# Note: the published Opus HD qext_vector*.bit files do not match the libopus 1.6.1 QEXT
# bitstream; libopus 1.6.1's own opus_demo stops on them with a range coder mismatch, and so does
# this port (byte-identically), so --hd fails at "Testing Opus HD testvectors" after the RFC
# vectors at 96 kHz have passed.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

hd=0
fixed=""
rates=()
for arg in "$@"; do
  case "$arg" in
    --hd) hd=1 ;;
    --fixed) fixed=fixed-point ;;
    --fixed-res24) fixed=fixed-res24 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) rates+=("$arg") ;;
  esac
done
if [ "${#rates[@]}" -eq 0 ]; then
  rates=(48000 24000 16000 12000 8000)
fi

if [ -n "${OPUSORUS_VECTORS:-}" ]; then
  vectors="$OPUSORUS_VECTORS"
else
  vectors="$root/testdata/vectors"
  dir="$root"
  while [ ! -d "$dir/testdata/vectors" ] && [ "$dir" != "/" ]; do
    dir="$(dirname "$dir")"
  done
  if [ -d "$dir/testdata/vectors" ]; then
    vectors="$dir/testdata/vectors"
  fi
fi

feat=()
if [ "$hd" -eq 1 ]; then
  feat+=(qext)
fi
if [ -n "$fixed" ]; then
  feat+=("$fixed")
fi
features=()
if [ "${#feat[@]}" -gt 0 ]; then
  features=(--features "$(IFS=,; echo "${feat[*]}")")
fi
cargo build --release --manifest-path "$root/Cargo.toml" -p opusorus-tools --bins "${features[@]}"
bin="${CARGO_TARGET_DIR:-$root/target}/release"
case "$bin" in /*) ;; *) bin="$root/$bin" ;; esac

export OPUS_DEMO="$bin/opus_demo"
export OPUS_COMPARE="$bin/opus_compare"
export QEXT_COMPARE="$bin/qext_compare"

# The upstream scripts write tmp.out and logs_*.txt to the current directory.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

for rate in "${rates[@]}"; do
  echo "=============================="
  echo "RFC 8251 vectors at $rate Hz"
  echo "=============================="
  sh "$root/vendor/libopus/tests/run_vectors.sh" "$bin" "$vectors/rfc8251" "$rate"
done

if [ "$hd" -eq 1 ]; then
  echo "=============================="
  echo "Opus HD vectors (96 kHz)"
  echo "=============================="
  sh "$root/vendor/libopus/tests/run_opushd_vectors.sh" "$bin" "$vectors/opushd"
fi

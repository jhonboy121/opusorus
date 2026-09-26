#!/usr/bin/env bash
# Compares stripped shared-library sizes: C libopus (upstream CMake, Release & MinSizeRel)
# vs the Rust C-ABI build (`opusorus-capi` cdylib, release & size-optimised profiles).
# Prints a markdown table (paste into docs/STATUS.md).
#
# Usage: scripts/size_report.sh [--fixed | --fixed-res24]
#   --fixed        fixed-point builds: C with OPUS_FIXED_POINT=ON, Rust with `fixed-point`
#   --fixed-res24  the same with 24-bit opus_res (C: + -DENABLE_RES24, Rust: `fixed-res24`)
# The `opusorus` rlib is listed for information (an rlib holds metadata and LLVM bitcode besides
# machine code, so it is not a linked-size measure). The C ABI is a
# verification harness (docs/PLAN.md D-025); its size is informational only.
set -euo pipefail
cd "$(dirname "$0")/.."

mode=float
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixed) mode=fixed-point ;;
    --fixed-res24) mode=fixed-res24 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

out=target/size-report
mkdir -p "$out"
# "<file bytes> <text+data bytes>" of the stripped library (file sizes are quantised by the
# 64 KiB segment alignment on aarch64; the section total is not).
strip_size() {
  cp "$1" "$out/tmp.so"; strip --strip-unneeded "$out/tmp.so"
  echo "$(stat -c %s "$out/tmp.so") $(size "$out/tmp.so" | awk 'NR == 2 { print $1 + $2 }')"
}

build_c() { # $1 = build type, $2.. = extra cmake args
  local bt=$1; shift
  local dir="$out/c-$bt$*"
  dir=${dir// /_}
  if [ ! -f "$dir/libopus.so" ]; then
    cmake -S vendor/libopus -B "$dir" -DCMAKE_BUILD_TYPE="$bt" -DOPUS_BUILD_SHARED_LIBRARY=ON \
      -DBUILD_TESTING=OFF "$@" >/dev/null
    cmake --build "$dir" -j"$(nproc)" >/dev/null 2>"$dir.log"
  fi
  strip_size "$(readlink -f "$dir/libopus.so")"
}

c_args=()
features=()
label=""
if [[ $mode != float ]]; then
  cflags=""
  features=(--features "$mode")
  label=", $mode"
  [[ $mode == fixed-res24 ]] && cflags="-DENABLE_RES24"
  # Upstream CMake defines the ARMv7 assembly switches on AArch64 too, which leaves
  # `celt_pitch_xcorr_neon` undefined in fixed-point builds; undo them like autotools/meson
  # (see crates/opusorus-bench/opt-sys/build.rs).
  if [[ $(uname -m) == aarch64 ]]; then
    cflags="$cflags -UOPUS_ARM_MAY_HAVE_NEON -UOPUS_ARM_PRESUME_NEON"
  fi
  c_args=(-DOPUS_FIXED_POINT=ON "-DCMAKE_C_FLAGS=${cflags# }")
fi

echo "| Library | Config | Stripped size (bytes) | KiB | text+data (bytes) |"
echo "|---|---|---:|---:|---:|"
row() { # $1 library, $2 config, $3 "<bytes> <text+data>"
  local b=${3% *} s=${3#* }
  printf "| %s | %s | %d | %.1f | %d |\n" "$1" "$2" "$b" "$(echo "$b/1024" | bc -l)" "$s"
}
row "libopus 1.6.1 (C)" "Release (-O3)$label" "$(build_c Release "${c_args[@]}")"
row "libopus 1.6.1 (C)" "MinSizeRel (-Os)$label" "$(build_c MinSizeRel "${c_args[@]}")"
row "libopus 1.6.1 (C)" "Release, no intrinsics$label" \
  "$(build_c Release -DOPUS_DISABLE_INTRINSICS=ON "${c_args[@]}")"
if [ -d crates/opusorus-capi ]; then
  cargo build -q -p opusorus-capi --release "${features[@]}"
  row "opusorus (Rust, C ABI)" "release (opt-level 3, fat LTO)$label" \
    "$(strip_size target/release/libopusorus.so)"
  cargo build -q -p opusorus-capi --profile release-small "${features[@]}"
  row "opusorus (Rust, C ABI)" "release-small (opt-level s, fat LTO)$label" \
    "$(strip_size target/release-small/libopusorus.so)"
else
  echo "| opusorus (Rust, C ABI) | not built yet | - | - | - |"
fi
cargo build -q -p opusorus --release "${features[@]}"
rlib=target/release/libopusorus.rlib
printf "| %s | %s | %d | %.1f | – |\n" "opusorus (Rust rlib, unstripped)" "release$label" \
  "$(stat -c %s "$rlib")" "$(echo "$(stat -c %s "$rlib")/1024" | bc -l)"

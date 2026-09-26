#!/usr/bin/env bash
# Compares stripped shared-library sizes: C libopus (upstream CMake, Release & MinSizeRel)
# vs the Rust C-ABI build (`opusorus-capi` cdylib, release & size-optimised profiles).
# Prints a markdown table (paste into docs/STATUS.md).
set -euo pipefail
cd "$(dirname "$0")/.."
out=target/size-report
mkdir -p "$out"
strip_size() { cp "$1" "$out/tmp.so"; strip --strip-unneeded "$out/tmp.so"; stat -c %s "$out/tmp.so"; }

build_c() { # $1 = build type, $2.. = extra cmake args
  local bt=$1; shift
  local dir="$out/c-$bt$*"
  dir=${dir// /_}
  if [ ! -f "$dir/libopus.so" ]; then
    cmake -S vendor/libopus -B "$dir" -DCMAKE_BUILD_TYPE="$bt" -DOPUS_BUILD_SHARED_LIBRARY=ON \
      -DBUILD_TESTING=OFF "$@" >/dev/null
    cmake --build "$dir" -j"$(nproc)" >/dev/null
  fi
  strip_size "$(readlink -f "$dir/libopus.so")"
}

echo "| Library | Config | Stripped size (bytes) | KiB |"
echo "|---|---|---:|---:|"
row() { printf "| %s | %s | %d | %.1f |\n" "$1" "$2" "$3" "$(echo "$3/1024" | bc -l)"; }
row "libopus 1.6.1 (C)" "Release (-O3), default features" "$(build_c Release)"
row "libopus 1.6.1 (C)" "MinSizeRel (-Os), default features" "$(build_c MinSizeRel)"
row "libopus 1.6.1 (C)" "Release, no intrinsics" "$(build_c Release -DOPUS_DISABLE_INTRINSICS=ON)"
if [ -d crates/opusorus-capi ]; then
  cargo build -q -p opusorus-capi --release
  row "opusorus (Rust, C ABI)" "release (opt-level 3, fat LTO)" "$(strip_size target/release/libopusorus.so)"
  cargo build -q -p opusorus-capi --profile release-small
  row "opusorus (Rust, C ABI)" "release-small (opt-level s, fat LTO)" "$(strip_size target/release-small/libopusorus.so)"
else
  echo "| opusorus (Rust, C ABI) | not built yet | - | - |"
fi

#!/usr/bin/env bash
# Builds libopusorus and generates pkg-config files for it.
#
#   crates/opusorus-capi/scripts/pkgconfig.sh [--profile P] [--features F] [--prefix DIR]
#
# Without --prefix, lays out the build output for uninstalled use:
#   target/<P>/include/opus/*.h          (the shipped libopus headers)
#   target/<P>/pkgconfig/opusorus.pc     (Name: opusorus, -lopusorus)
#   target/<P>/pkgconfig/opus.pc         (drop-in: Name: Opus, same flags)
# so `PKG_CONFIG_PATH=target/<P>/pkgconfig pkg-config --cflags --libs opus` builds unmodified
# libopus users against opusorus.
#
# With --prefix DIR, installs libopusorus.{so,a} to DIR/lib, the headers to DIR/include/opus
# and both .pc files to DIR/lib/pkgconfig (like libopus' `make install`).
set -euo pipefail

crate_dir="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$crate_dir/../.." && pwd)"
profile=release
features=
prefix=
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) profile=$2; shift 2;;
    --features) features=$2; shift 2;;
    --prefix) prefix=$2; shift 2;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

case "$profile" in
  dev|test) outdir=debug;;
  bench) outdir=release;;
  *) outdir=$profile;;
esac
target_dir="${CARGO_TARGET_DIR:-$root/target}"
build_dir="$target_dir/$outdir"

feature_args=()
if [ -n "$features" ]; then feature_args=(--features "$features"); fi

cd "$root"
cargo build -q -p opusorus-capi --profile "$profile" "${feature_args[@]}"
# Libraries the static archive needs (Rust std); printed by rustc for staticlib outputs.
static_libs="$(cargo rustc -q -p opusorus-capi --lib --profile "$profile" "${feature_args[@]}" \
  -- --print native-static-libs 2>&1 | sed -n 's/.*native-static-libs: //p' | tail -1)"
if [ -z "$static_libs" ]; then static_libs="-lm -lpthread -ldl"; fi

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)"

write_pc() { # $1 = file, $2 = Name, $3 = prefix, $4 = libdir, $5 = includedir
  cat > "$1" <<EOF
# opusorus: libopus 1.6.1 C ABI implemented in safe Rust (opusorus $version)

prefix=$3
exec_prefix=\${prefix}
libdir=$4
includedir=$5

Name: $2
Description: Opus IETF audio codec (opusorus, libopus 1.6.1 compatible float build)
URL: https://opus-codec.org/
Version: 1.6.1
Requires:
Conflicts:
Libs: -L\${libdir} -lopusorus
Libs.private: $static_libs
Cflags: -I\${includedir}/opus
EOF
}

if [ -z "$prefix" ]; then
  inc="$build_dir/include"
  pcdir="$build_dir/pkgconfig"
  mkdir -p "$inc/opus" "$pcdir"
  cp "$crate_dir"/include/*.h "$inc/opus/"
  write_pc "$pcdir/opusorus.pc" opusorus "$build_dir" "$build_dir" "$inc"
  write_pc "$pcdir/opus.pc" Opus "$build_dir" "$build_dir" "$inc"
  echo "$pcdir"
else
  mkdir -p "$prefix/lib/pkgconfig" "$prefix/include/opus"
  for f in libopusorus.so libopusorus.dylib libopusorus.a; do
    if [ -f "$build_dir/$f" ]; then cp "$build_dir/$f" "$prefix/lib/"; fi
  done
  cp "$crate_dir"/include/*.h "$prefix/include/opus/"
  write_pc "$prefix/lib/pkgconfig/opusorus.pc" opusorus "$prefix" "\${exec_prefix}/lib" "\${prefix}/include"
  write_pc "$prefix/lib/pkgconfig/opus.pc" Opus "$prefix" "\${exec_prefix}/lib" "\${prefix}/include"
  echo "$prefix/lib/pkgconfig"
fi

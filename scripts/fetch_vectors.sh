#!/usr/bin/env bash
# Downloads the official Opus decoder test vectors (RFC 6716 as updated by RFC 8251, and the
# original RFC 6716 set for `disable-rfc8251` builds) and, for
# QEXT, the Opus HD vectors. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=testdata/vectors
mkdir -p "$dest"
if [ ! -f "$dest/rfc8251/testvector01.bit" ]; then
  curl -fsSL https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz -o "$dest/rfc8251.tar.gz"
  mkdir -p "$dest/rfc8251"
  tar -xzf "$dest/rfc8251.tar.gz" -C "$dest/rfc8251" --strip-components=1
  rm "$dest/rfc8251.tar.gz"
fi
# The original RFC 6716 vectors (the conformance set of a `disable-rfc8251` build).
if [ ! -f "$dest/rfc6716/testvector01.bit" ]; then
  curl -fsSL https://opus-codec.org/static/testvectors/opus_testvectors.tar.gz -o "$dest/rfc6716.tar.gz"
  mkdir -p "$dest/rfc6716"
  tar -xzf "$dest/rfc6716.tar.gz" -C "$dest/rfc6716" --strip-components=1
  rm "$dest/rfc6716.tar.gz"
fi
if [ ! -f "$dest/opushd/qext_vector01.bit" ]; then
  mkdir -p "$dest/opushd"
  curl -fsSL https://media.xiph.org/opus/ietf/opushd_testvectors.tar.gz -o "$dest/opushd.tar.gz"
  tar -xzf "$dest/opushd.tar.gz" -C "$dest/opushd" --strip-components=1
  rm "$dest/opushd.tar.gz"
fi
echo "vectors ready in $dest"

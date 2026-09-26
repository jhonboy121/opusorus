#!/usr/bin/env bash
# Downloads the official Opus decoder test vectors (RFC 6716 as updated by RFC 8251) and, for
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
if [ ! -f "$dest/opushd/qext_vector01.bit" ]; then
  mkdir -p "$dest/opushd"
  curl -fsSL https://media.xiph.org/opus/ietf/opushd_testvectors.tar.gz -o "$dest/opushd.tar.gz"
  tar -xzf "$dest/opushd.tar.gz" -C "$dest/opushd" --strip-components=1
  rm "$dest/opushd.tar.gz"
fi
echo "vectors ready in $dest"

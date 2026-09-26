#!/usr/bin/env bash
# Fetches the libopus 1.6.1 DNN model data (same tarball & checksum as upstream autogen.sh) and
# extracts the generated *_data.c/h files into vendor/libopus/dnn (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
sum=a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e
tgz=testdata/opus_data-$sum.tar.gz
mkdir -p testdata
[ -f "$tgz" ] || curl -fsSL -o "$tgz" "https://media.xiph.org/opus/models/opus_data-$sum.tar.gz"
echo "$sum  $tgz" | sha256sum -c -
tar -xzf "$tgz" -C vendor/libopus --exclude='dnn/models/*'
echo "DNN model data extracted into vendor/libopus/dnn"

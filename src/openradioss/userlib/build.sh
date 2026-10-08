#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENRADIOSS_ROOT:?Set OPENRADIOSS_ROOT to your OpenRadioss source/build tree}"
SDK_ROOT="$ROOT/tools/userlib/userlib_sdk"

export RAD_USERLIB_SDK_PATH="$SDK_ROOT/userlib_sdk"
export RAD_USERLIB_ARCH="linux64_gfortran"
cd "$(dirname "${BASH_SOURCE[0]}")"

"$RAD_USERLIB_SDK_PATH/$RAD_USERLIB_ARCH/build_userlib.bash" \
  starter="lecmuser01.f90" \
  engine="luser01.f90" \
  outfile="libraduser_law291.so" \
  -free

LIB="libraduser_law291.so"
if [[ ! -s "$LIB" ]]; then
  echo "User library build failed: $LIB was not created" >&2
  exit 1
fi
echo "Built: $LIB"

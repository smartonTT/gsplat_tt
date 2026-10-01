#!/usr/bin/env bash
# Build and run the named standalone C++ unit tests (tests/unit/test_*.cpp).
# On macOS the MacOSX27.0 SDK's libSystem.B.tbd fails to link (tapi "unknown
# architecture"), so unless SDKROOT is set, use the newest SDK that links.
set -u
cd "$(dirname "$0")/../.."
if [[ "$(uname)" == Darwin && -z "${SDKROOT:-}" ]]; then
  echo 'int main(){return 0;}' > /tmp/sdkprobe_$$.cpp
  for s in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX[0-9]*.sdk | sort -rV); do
    if c++ -isysroot "$s" /tmp/sdkprobe_$$.cpp -o /tmp/sdkprobe_$$ 2>/dev/null; then
      export SDKROOT="$s"; break
    fi
  done
  rm -f /tmp/sdkprobe_$$ /tmp/sdkprobe_$$.cpp
  echo "SDKROOT=${SDKROOT:-default}"
fi
fail=0
[[ $# -gt 0 ]] || { echo "usage: $0 tests/unit/test_X.cpp... (extra flags via CXXFLAGS)"; exit 2; }
for f in "$@"; do
  b=/tmp/$(basename "$f" .cpp)_$$
  if c++ -O2 -std=c++17 ${CXXFLAGS:-} -I. -Itests/unit "$f" -o "$b" && "$b" >/dev/null; then echo "PASS $f"
  else echo "FAIL $f"; fail=1; fi
  rm -f "$b"
done
exit $fail

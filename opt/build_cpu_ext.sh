#!/usr/bin/env bash
# Build backends/cpu_cpp/_gsplat_cpu<EXT_SUFFIX> for one Python, without CMake.
#   opt/build_cpu_ext.sh [python=python3]
# For tests in a fresh worktree (tests/spec/conftest.py calls this when the
# extension is missing). Scalar/NEON build, no tt-metal; production builds
# still use src/CMakeLists.txt. Takes ~40 s on the Mac.
# macOS: the CLT linker can't read the newest SDK's .tbd files (task #79), so
# the first SDK that links a test program is used, newest first.
set -euo pipefail
PY=${1:-python3}
cd "$(dirname "$0")/.."

read -r EXT PYINC PBINC < <("$PY" - <<'EOF'
import os, sys, sysconfig
import pybind11
v = "python%d.%d" % sys.version_info[:2]
cands = [sysconfig.get_config_var("INCLUDEPY") or "",
         os.path.join(sys.base_prefix, "include", v),
         os.path.join(sys.base_prefix, "Headers")]
inc = next((c for c in cands if c and os.path.exists(os.path.join(c, "Python.h"))), "")
print(sysconfig.get_config_var("EXT_SUFFIX"), inc or "-", pybind11.get_include())
EOF
)
[ "$PYINC" != "-" ] || { echo "build_cpu_ext: Python.h not found for $PY" >&2; exit 2; }

CXX=${CXX:-c++}
SYSROOT=()
LDFLAGS=(-shared)
if [ "$(uname)" = Darwin ]; then
  LDFLAGS=(-bundle -undefined dynamic_lookup -framework Accelerate)
  probe=$(mktemp -d)
  echo 'int main(){return 0;}' > "$probe/p.cpp"
  if ! "$CXX" "$probe/p.cpp" -o "$probe/p" 2>/dev/null; then
    for sdk in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX[0-9]*.sdk 2>/dev/null | sort -rV); do
      if "$CXX" -isysroot "$sdk" "$probe/p.cpp" -o "$probe/p" 2>/dev/null; then
        SYSROOT=(-isysroot "$sdk"); break
      fi
    done
    [ ${#SYSROOT[@]} -gt 0 ] || { echo "build_cpu_ext: no SDK links with $CXX" >&2; rm -rf "$probe"; exit 2; }
    echo "build_cpu_ext: using ${SYSROOT[1]}"
  fi
  rm -rf "$probe"
fi

OBJ=$(mktemp -d)
trap 'rm -rf "$OBJ"' EXIT
FLAGS=(-std=c++20 -O3 -DNDEBUG -fPIC -w ${SYSROOT[@]+"${SYSROOT[@]}"} -Isrc)
pids=()
# Same sources as the gsplat_cpu target in src/CMakeLists.txt.
for n in thread_pool project tile_assign sort blend microblock_cull blend_microblock cull_and_blend; do
  f=src/gsplat_cpu/$n.cpp
  "$CXX" "${FLAGS[@]}" -ffp-contract=off -c "$f" -o "$OBJ/$n.o" & pids+=($!)
done
"$CXX" "${FLAGS[@]}" -I"$PBINC" -I"$PYINC" -c backends/cpu_cpp/pybind_module.cpp \
  -o "$OBJ/pybind_module.o" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
OUT=backends/cpu_cpp/_gsplat_cpu$EXT
"$CXX" ${SYSROOT[@]+"${SYSROOT[@]}"} "${LDFLAGS[@]}" "$OBJ"/*.o -o "$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"
echo "build_cpu_ext: built $OUT"

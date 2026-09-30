#!/usr/bin/env bash
# Sync a committed tree to a remote build dir, build, and flag a stale host .so.
#   opt/sync_remote.sh <host> <remote_dir> [rev=HEAD]
# Extracts with `tar -m` so files get fresh mtimes: git archive keeps commit
# mtimes, which can be older than build outputs, so ninja skips changed host
# sources and a stale render_clean .so hangs the device (task #47).
# Wrap device use in `ttp lock p100 -- opt/sync_remote.sh ...`.
set -euo pipefail
HOST=${1:?host}; DIR=${2:?remote_dir}; REV=${3:-HEAD}
SHA=$(git rev-parse "$REV")
git archive --format=tar "$SHA" | ssh -o BatchMode=yes "$HOST" \
  "mkdir -p '$DIR' && tar -m -x -C '$DIR' && echo $SHA > '$DIR/SHA'"
ssh -o BatchMode=yes "$HOST" "DIR='$DIR' bash -s" <<'REMOTE'
set -eu
cd "$DIR"
# non-interactive ssh has no TT env; cmake needs TT_METAL_HOME (GSPLAT_WITH_TT)
export TT_METAL_HOME=${TT_METAL_HOME:-/localdev/smarton/tt-metal}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME} TT_METAL_ARCH_NAME=${TT_METAL_ARCH_NAME:-blackhole}
[ -f /localdev/smarton/gstt2/.venv/bin/activate ] && . /localdev/smarton/gstt2/.venv/bin/activate
[ -L .venv ] && [ ! -e .venv ] && rm -f .venv   # tracked .venv symlink breaks remote CMake
so_md5() { md5sum render/render_clean*.so 2>/dev/null | awk '{print $1}' | sort | tr '\n' ' '; }
src_md5() { find render -path render/build-tt -prune -o -path '*/kernels' -prune -o \
  -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name CMakeLists.txt \) \
  -print | LC_ALL=C sort | xargs md5sum | md5sum | awk '{print $1}'; }
SRC_NOW=$(src_md5); SRC_OLD=$(cat .host_src.md5 2>/dev/null || true); SO_OLD=$(so_md5)
mkdir -p tmp
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 &&
 cmake --build render/build-tt -j 16 > tmp/build.log 2>&1) || { tail -n 30 tmp/cfg.log tmp/build.log 2>/dev/null; exit 1; }
SO_NEW=$(so_md5); echo "$SRC_NOW" > .host_src.md5
echo "[sync_remote] sha=$(cat SHA) so_md5=$SO_NEW"
if [ -n "$SRC_OLD" ] && [ "$SRC_OLD" != "$SRC_NOW" ] && [ "$SO_OLD" = "$SO_NEW" ]; then
  echo "[sync_remote] WARNING: host sources changed but render_clean .so md5 did not -- stale build? rm render/render_clean*.so and rebuild" >&2
  exit 3
fi
REMOTE

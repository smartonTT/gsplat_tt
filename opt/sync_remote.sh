#!/usr/bin/env bash
# Sync a committed tree to a remote build dir, build, and flag a stale host .so.
#   opt/sync_remote.sh <host> <remote_dir> [rev=HEAD]
# Extracts with `tar -m` so files get fresh mtimes: git archive keeps commit
# mtimes, which can be older than build outputs, so ninja skips changed host
# sources and a stale render_clean .so hangs the device (task #47).
# Wrap device use in `ttp lock p100 -- opt/sync_remote.sh ...`.
# Exit codes: 0 ok, 1 configure/build failed (last log lines printed), 3 stale
# .so. Callers MUST check the exit code: don't pipe the output (e.g. `| tail`)
# without `set -o pipefail`, or a failed build looks like success.
# Remote env defaults: TT_METAL_HOME=/localdev/smarton/tt-metal,
# TT_METAL_ARCH_NAME=blackhole; set them on the remote side to override.
# A fresh remote dir is made self-sufficient: missing .venv and scenes are
# symlinked to $BASE (/localdev/smarton/gstt2), and a missing _gsplat_cpu .so is
# built (build-avx2-tt) or, if that fails, copied from $BASE.
set -euo pipefail
HOST=${1:?host}; DIR=${2:?remote_dir}; REV=${3:-HEAD}
SHA=$(git rev-parse "$REV")
# Skip what a device run never reads: the LFS hero fixtures (*.npz, ~330 MB once
# git-lfs smudges them; test inputs only), committed profiler captures
# (ttw-*/, *.tracy, profile_log_device*.csv, ~600 MB) and screenshots (~150 MB).
# Streaming those made a sync take 15-30+ min (tasks #86, #90); the remote keeps
# any copies it already has. SYNC_ALL=1 sends the whole tree.
EXCL=(":(exclude)tests/fixtures/hero/*.npz" ":(exclude)opt/profiler/ttw-*"
      ":(exclude,glob)**/*.tracy" ":(exclude,glob)**/profile_log_device*.csv"
      ":(exclude)opt/metal-screenshots")
[ "${SYNC_ALL:-0}" = 1 ] && EXCL=()
git archive --format=tar "$SHA" -- . "${EXCL[@]}" | ssh -o BatchMode=yes "$HOST" \
  "mkdir -p '$DIR' && tar -m -x -C '$DIR' && echo $SHA > '$DIR/SHA'"
# REMOTE_TT_METAL_HOME (Mac side) overrides the remote TT_METAL_HOME default below.
ssh -o BatchMode=yes "$HOST" "DIR='$DIR' ${REMOTE_TT_METAL_HOME:+TT_METAL_HOME='$REMOTE_TT_METAL_HOME'} bash -s" <<'REMOTE'
set -eu
cd "$DIR"
# non-interactive ssh has no TT env; cmake needs TT_METAL_HOME (GSPLAT_WITH_TT)
export TT_METAL_HOME=${TT_METAL_HOME:-/localdev/smarton/tt-metal}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME} TT_METAL_ARCH_NAME=${TT_METAL_ARCH_NAME:-blackhole}
[ -f /localdev/smarton/gstt2/.venv/bin/activate ] && . /localdev/smarton/gstt2/.venv/bin/activate
[ -L .venv ] && [ ! -e .venv ] && rm -f .venv   # tracked .venv symlink breaks remote CMake
BASE=${GSTT2_BASE:-/localdev/smarton/gstt2}
if [ "$(cd "$BASE" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
  for d in .venv scenes; do
    [ -e "$d" ] || { ln -sfn "$BASE/$d" "$d"; echo "[sync_remote] linked $d -> $BASE/$d"; }
  done
fi
so_md5() { md5sum render/render_clean*.so 2>/dev/null | awk '{print $1}' | sort | tr '\n' ' '; }
src_md5() { find render -path render/build-tt -prune -o -path '*/kernels' -prune -o \
  -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name CMakeLists.txt \) \
  -print | LC_ALL=C sort | xargs md5sum | md5sum | awk '{print $1}'; }
SRC_NOW=$(src_md5); SRC_OLD=$(cat .host_src.md5 2>/dev/null || true); SO_OLD=$(so_md5)
mkdir -p tmp
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 &&
 cmake --build render/build-tt -j 16 > tmp/build.log 2>&1) || { tail -n 30 tmp/cfg.log tmp/build.log 2>/dev/null; exit 1; }
if ! ls backends/cpu_cpp/_gsplat_cpu*.so >/dev/null 2>&1; then
  if (cmake -G Ninja -S src -B build-avx2-tt -DCMAKE_BUILD_TYPE=Release > tmp/cpu_cfg.log 2>&1 &&
      cmake --build build-avx2-tt --target _gsplat_cpu -j 16 > tmp/cpu_build.log 2>&1) &&
     ls backends/cpu_cpp/_gsplat_cpu*.so >/dev/null 2>&1; then
    echo "[sync_remote] built _gsplat_cpu"
  elif cp "$BASE"/backends/cpu_cpp/_gsplat_cpu*.so backends/cpu_cpp/ 2>/dev/null; then
    echo "[sync_remote] _gsplat_cpu build failed (tmp/cpu_*.log); copied from $BASE"
  else
    echo "[sync_remote] no _gsplat_cpu .so (build failed, none in $BASE)" >&2; exit 1
  fi
fi
SO_NEW=$(so_md5); echo "$SRC_NOW" > .host_src.md5
echo "[sync_remote] sha=$(cat SHA) so_md5=$SO_NEW"
if [ -n "$SRC_OLD" ] && [ "$SRC_OLD" != "$SRC_NOW" ] && [ "$SO_OLD" = "$SO_NEW" ]; then
  echo "[sync_remote] WARNING: host sources changed but render_clean .so md5 did not -- stale build? rm render/render_clean*.so and rebuild" >&2
  exit 3
fi
REMOTE

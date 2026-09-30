#!/bin/bash
# Build one remote tree and run the 30-view bicycle render once with the
# first-frame cull dump (GSPLAT_TT_DUMP_CULL) and --dump-views, printing
# SUMMARY/STAGES. Usage: remote_build_dump.sh <variant>   (tree = /localdev/smarton/gstt2-<variant>)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
source /localdev/smarton/gstt2/.venv/bin/activate
V=$1
cd /localdev/smarton/gstt2-$V || exit 1
mkdir -p tmp
echo "== $V build $(date +%T)"
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 && cmake --build render/build-tt -j 12 > tmp/build.log 2>&1) || { tail -30 tmp/cfg.log tmp/build.log; exit 1; }
md5sum render/render_clean*.so
rm -rf tmp/cull-dump tmp/t59-dump; mkdir -p tmp/cull-dump
echo "== $V run $(date +%T)"
TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=${LOGCC:-0} GSPLAT_TT_DUMP_CULL=$PWD/tmp/cull-dump \
  TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-$V \
  timeout 400 python3 render/run.py --no-ref --iter-dir t59-$V --dump-views t59-dump > tmp/run1.log 2>&1
echo "run rc=$?"
grep -E "^(SUMMARY|STAGES)|rror|FATAL|Timeout|hang" tmp/run1.log | head -20
ls tmp/cull-dump | head
echo "== $V end $(date +%T)"

#!/bin/bash
# Build one remote tree, run the host model test, then render the 30 bicycle
# views once with --dump-views and md5 them.  Runs under devrun.sh.
# Usage: remote_build_dump.sh <variant>   (tree = /localdev/smarton/gstt2-<variant>)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
source /localdev/smarton/gstt2/.venv/bin/activate
V=$1
cd /localdev/smarton/gstt2-$V || exit 1
mkdir -p tmp
echo "== $V build $(date +%T)"
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 && cmake --build render/build-tt -j 12 > tmp/build.log 2>&1) || { tail -30 tmp/cfg.log tmp/build.log; exit 1; }
md5sum render/render_clean*.so
[ -f tests/spec/test_img_pack_u8.py ] && PYTHONPATH=/localdev/smarton/t61_scripts/pylib python3 -m pytest -q -p no:cacheprovider tests/spec/test_img_pack_u8.py 2>&1 | tail -3
rm -rf tmp/t61-dump tmp/t61-md5.txt
echo "== $V run $(date +%T)"
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-$V \
  timeout 300 python3 render/run.py --no-ref --iter-dir t61-$V --dump-views t61-dump > tmp/run1.log 2>&1
echo "run rc=$?"
grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL|Error|REFUSING" tmp/run1.log | head -20
(cd tmp/t61-dump && md5sum * | sort -k2 > ../t61-md5.txt; echo "views: $(wc -l < ../t61-md5.txt)")
echo "== $V end $(date +%T)"

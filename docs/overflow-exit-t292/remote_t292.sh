#!/bin/bash
# t292 device checks (yyzo-bh-07, under `ttp lock p100`). Core dumps off (ulimit -c 0)
# so a teardown crash shows as rc 139 + "Segmentation" in the log, not a slow dump.
#   1. new, 30 bench views: sweep md5 must be the iter-205 golden 906e0435.
#   2. cap 1 (GSPLAT_TT_TEST_TILE_CAP=1), hero only: base vs new. RuntimeError must
#      surface; new must exit rc 1 with no SIGSEGV in static teardown.
# Arg 'new' skips the base runs.
#   3. cull disabled + cap 20000, hero only: base retries uselessly; new fails at once.
set -u
ulimit -c 0
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
NEW=/localdev/smarton/gstt2-t292; BASE=${T292_BASE:-/localdev/smarton/gstt2-t290}
S=$NEW/tmp/t292; mkdir -p $S
source $NEW/.venv/bin/activate
python3 - $S/cam_hero.json $NEW <<'PY'
import json, sys
c = json.load(open(sys.argv[2] + '/benchmarks/cameras_v2.json')); b = c['bicycle']
b['views'] = {'hero': b['views'][b['order'][0]]}; b['order'] = ['hero']
json.dump(c, open(sys.argv[1], 'w'), indent=1)
PY
cat > $S/run_cd.py <<'PY'
# render/run.py main() with the cull disabled (Pipeline overrides the backend flag).
import sys
sys.path.insert(0, sys.argv[1] + '/render')
import run
class CullOff(run.Pipeline):
    def __init__(self, *a, **k):
        k['cull_disabled'] = True
        super().__init__(*a, **k)
run.Pipeline = CullOff
sys.argv = ['run.py'] + sys.argv[2:]
run.main()
PY
run() {  # tag dir py|cd camjson|- dump(0/1) [ENV=V ...]
  local tag=$1 dir=$2 mode=$3 cj=$4 dump=$5; shift 5
  local args=(--no-ref --iter-dir t292-$tag)
  [ "$cj" != - ] && args+=(--cameras $cj)
  [ "$dump" = 1 ] && args+=(--dump-views t292-dump-$tag)
  local cmd=(python3 render/run.py); [ $mode = cd ] && cmd=(python3 $S/run_cd.py $dir)
  echo "=== $tag $(cut -c1-7 $dir/SHA) mode=$mode cams=$cj $* $(date +%T)"
  local t0=$(date +%s)
  (cd $dir && rm -rf tmp/t292-dump-$tag && env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $dir) \
    timeout ${RUN_TO:-300} "${cmd[@]}" "${args[@]}" > $S/run-$tag.log 2>&1)
  local rc=$?
  echo "run rc=$rc wall=$(( $(date +%s) - t0 ))s"
  grep -E "^SUMMARY|TTW_TIMING ms_view|Traceback|TT_THROW|TT_FATAL|RuntimeError|sort failed|tile overflow|bucket capacity|Segmentation|Signal|core dumped" $S/run-$tag.log | sed 's/^\[[^]]*:[0-9]*\] //' | sort | uniq -c | head -14
  if [ -d $dir/tmp/t292-dump-$tag ]; then
    (cd $dir/tmp/t292-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt)"
    rm -rf $dir/tmp/t292-dump-$tag
  fi
  return 0
}
HERO=$S/cam_hero.json
run bench-new $NEW py - 1
[ "${1:-}" = new ] || RUN_TO=180 run cap1-base $BASE py $HERO 0 GSPLAT_TT_TEST_TILE_CAP=1
RUN_TO=180 run cap1-new $NEW py $HERO 0 GSPLAT_TT_TEST_TILE_CAP=1
[ "${1:-}" = new ] || RUN_TO=180 run cd-base $BASE cd $HERO 0 GSPLAT_TT_TEST_TILE_CAP=20000
RUN_TO=180 run cd-new $NEW cd $HERO 0 GSPLAT_TT_TEST_TILE_CAP=20000
echo "=== done $(date +%T)"

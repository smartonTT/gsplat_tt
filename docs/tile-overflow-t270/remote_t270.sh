#!/bin/bash
# t270 device driver (yyzo-bh-07, under `ttp lock p100`).
#   remote_t270.sh   runs: paired A/B (base tip vs t270, 30 views, md5), forced
#   overflow (GSPLAT_TT_TEST_TILE_CAP), and the real overflow pose
#   (hero pulled back 2 units: tile 554 = 34059 records at 1/255 in tilecount.py).
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
NEW=/localdev/smarton/gstt2-t270; BASE=${T270_BASE:-/localdev/smarton/gstt2-t269a}
S=$NEW/tmp/t270; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
source $NEW/.venv/bin/activate
cam() {  # name dolly inv -> camera json with view "hero" = dollied hero, floor 1/inv
  local p=$S/cam_$1.json
  python3 - "$p" "$2" "$3" <<'PY'
import json, sys
p, d, inv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = json.load(open('/localdev/smarton/gstt2-t270/benchmarks/cameras_v2.json'))
b = c['bicycle']; m = [r[:] for r in b['views']['hero']['c2w']]
for i in range(3): m[i][3] += m[i][2] * d
b['views'] = {'hero': {'c2w': m, 'manual': False}}; b['order'] = ['hero']
b['contrib_floor'] = 1.0 / inv
json.dump(c, open(p, 'w'), indent=1)
PY
  echo $p
}
run() {  # tag dir camjson|- [ENV=V ...]
  local tag=$1 dir=$2 cj=$3; shift 3
  local args=(--no-ref --iter-dir t270-$tag)
  [ "$cj" != - ] && args+=(--cameras $cj) || args+=(--dump-views t270-dump-$tag)
  echo "=== $tag $(cut -c1-7 $dir/SHA) cams=$cj $* $(date +%T)"
  (cd $dir && rm -rf tmp/t270-dump-$tag && env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $dir) \
    timeout ${RUN_TO:-300} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1)
  local rc=$?
  echo "run rc=$rc"
  grep -E "^SUMMARY|TTW_TIMING ms_view|Traceback|TT_THROW|TT_FATAL|RuntimeError|sort failed|tile overflow|bucket capacity" $S/run-$tag.log | sort | uniq -c | head -12
  if [ -d $dir/tmp/t270-dump-$tag ]; then
    (cd $dir/tmp/t270-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt) $(diff -q $REF $S/md5-$tag.txt >/dev/null && echo ALL_VIEWS_IDENTICAL_TO_GOLDEN || echo differs_from_golden)"
    rm -rf $dir/tmp/t270-dump-$tag
  fi
  [ -f $dir/tmp/t270-$tag/hero_clean.png ] && cp $dir/tmp/t270-$tag/hero_clean.png $S/hero-$tag.png
  return 0
}
# Paired A/B, ABBA order.
for r in 1 2; do
  run r$r-base $BASE -; run r$r-new $NEW -; run r$r-new2 $NEW -; run r$r-base2 $BASE -
done
# Forced overflow on the bench views: host check cap 20000 (< bench max ~22-25k).
run cap20k $NEW - GSPLAT_TT_TEST_TILE_CAP=20000 GSPLAT_PER_VIEW_STAGES=1
# Real overflow pose at the bench floor and at the t257 viewer floor.
run far2-255-new $NEW $(cam far2_255 -2 255)
run far2-255-base $BASE $S/cam_far2_255.json
run far2-16k-new $NEW $(cam far2_16k -2 16384)
# Same pose at a floor that fits, for comparison with the retried image.
run far2-64-new $NEW $(cam far2_64 -2 64)
echo "=== done $(date +%T)"

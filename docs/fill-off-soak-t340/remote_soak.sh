#!/bin/bash
# t340 soak: the bicycle far pose (hero dollied -2, floor 1/255; the 'far' arm of
# docs/tile-split-t284/remote_t284.sh) as N timed views in one process.
#   remote_soak.sh <tag> <nviews> [ENV=V ...]      (run under devrun / ttp lock p100)
# Prints RUN_RC (124 = hung: per-run timeout RUN_TO, default 150 s), the views
# reached, and the md5 set of the dumped views.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
D=${SOAK_DIR:-/localdev/smarton/gstt2-t340}; S=$D/tmp/t340; mkdir -p $S
source $D/.venv/bin/activate
tag=$1 n=$2; shift 2
cj=$S/cam_$n.json
python3 - "$cj" "$n" "$D" <<'PY'
import json, sys
p, n, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = json.load(open(d + '/benchmarks/cameras_v2.json'))
b = c['bicycle']; m = [r[:] for r in b['views']['hero']['c2w']]
for i in range(3): m[i][3] += m[i][2] * -2.0
names = ['hero'] + ['rep%02d' % i for i in range(1, n)]
b['views'] = {v: {'c2w': m, 'manual': False} for v in names}
b['order'] = names
b['contrib_floor'] = 1.0 / 255
json.dump(c, open(p, 'w'), indent=1)
PY
cd $D && rm -rf tmp/t340-dump-$tag tmp/t340-$tag
echo "=== $tag $(cut -c1-8 SHA) n=$n $* $(date +%T)"
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t340 \
  timeout -k 20 ${RUN_TO:-150} python3 render/run.py --no-ref --cameras $cj \
  --iter-dir t340-$tag --dump-views t340-dump-$tag > $S/run-$tag.log 2>&1
rc=$?; echo "RUN_RC=$rc $(date +%T)"
echo "VIEWS_DONE=$(grep -c '^\[run\]   view=' $S/run-$tag.log)"
grep -E "^SUMMARY|^TTW_TIMING ms_view|Traceback|TT_THROW|TT_FATAL|RuntimeError|tile overflow|hard fail" $S/run-$tag.log | sort | uniq -c | head -8
dd=$(find $D/tmp -maxdepth 2 -type d -name t340-dump-$tag | head -1)
[ -n "$dd" ] && { (cd $dd && md5sum *.png | awk '{print $1}' | sort | uniq -c) | tee $S/md5-$tag.txt; rm -rf $dd; }
exit 0

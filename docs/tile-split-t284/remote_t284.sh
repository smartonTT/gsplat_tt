#!/bin/bash
# t284 device driver (yyzo-bh-07 via devrun.sh, under ttp lock p100).
#   remote_t284.sh <round> <arm>...   arm = name:tree[:ENV=V,...] | far:<tag>:<dolly>:<invfloor>
# name:tree  = bicycle 30 views at the default floor, md5 sweep.
# far:...    = the hero dollied <dolly> units (t270 far pose), 3 copies of that view
#              (hero, rep1, rep2: the first pays the bucket grow + retry), floor 1/<inv>.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
NEW=/localdev/smarton/gstt2-t284; BASE=/localdev/smarton/gstt2-t284b
S=$NEW/tmp/t284; mkdir -p $S
source $NEW/.venv/bin/activate
r=${1:-1}; shift
cam() {  # tag dolly inv -> camera json
  local p=$S/cam_$1.json
  python3 - "$p" "$2" "$3" <<'PY'
import json, sys
p, d, inv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = json.load(open('/localdev/smarton/gstt2-t284/benchmarks/cameras_v2.json'))
b = c['bicycle']; m = [r[:] for r in b['views']['hero']['c2w']]
for i in range(3): m[i][3] += m[i][2] * d
b['views'] = {v: {'c2w': m, 'manual': False} for v in ('hero', 'rep1', 'rep2')}
b['order'] = ['hero', 'rep1', 'rep2']
b['contrib_floor'] = 1.0 / inv
json.dump(c, open(p, 'w'), indent=1)
PY
  echo $p
}
run() {  # tag dir camjson|- [ENV=V ...]
  local tag=$1 dir=$2 cj=$3; shift 3
  local args=(--no-ref --iter-dir t284-$tag --dump-views t284-dump-$tag)
  [ "$cj" != - ] && args+=(--cameras $cj)
  echo "=== $tag $(cut -c1-7 $dir/SHA) cams=$cj $* $(date +%T)"
  (cd $dir && rm -rf tmp/t284-dump-$tag tmp/t284-$tag && env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $dir) \
    timeout ${RUN_TO:-300} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1)
  echo "run rc=$?"
  grep -E "^SUMMARY|Traceback|TT_THROW|TT_FATAL|RuntimeError|sort failed|tile overflow|bucket capacity|hard fail" $S/run-$tag.log | sort | uniq -c | head -12
  local d; d=$(find $dir -maxdepth 3 -type d -name t284-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt)"
    [ "$cj" != - ] && for f in "$d"/*; do cp "$f" $S/view-$tag-$(basename "$f"); done
    rm -rf "$d"
  fi
  [ -f $dir/tmp/t284-$tag/hero_clean.png ] && cp $dir/tmp/t284-$tag/hero_clean.png $S/hero-$tag.png
  return 0
}
for a in "$@"; do
  case $a in
    far:*) IFS=: read -r _ tag d inv envs <<< "$a"; run r$r-$tag $NEW $(cam $tag $d $inv) ${envs//,/ } ;;
    *) IFS=: read -r name tree envs <<< "$a"
       dir=$NEW; [ "$tree" = base ] && dir=$BASE
       run r$r-$name $dir - ${envs//,/ } ;;
  esac
done
echo "=== done round $r $(date +%T)"

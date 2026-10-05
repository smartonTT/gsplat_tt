#!/bin/bash
# t260: untraced bicycle 30-view timing of contrib_floor arms (one build, runtime knobs).
#   remote_time.sh <round> [arm ...]   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
# arm = name:INVFLOOR[:ENV=V,ENV2=V]; INVFLOOR 255 uses benchmarks/cameras_v2.json as is,
# others a copy with only contrib_floor = 1/INVFLOOR. Hero (device) -> tmp/t260-r<r>-<name>/hero_clean.png.
# Special arm "dump" = default config, hero only, GSPLAT_TT_DUMP_PROJ -> tmp/t260/proj.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t260; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t260; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
cam() {  # inv -> camera json path
  [ "$1" = 255 ] && { echo benchmarks/cameras_v2.json; return; }
  local p=$S/cameras_f$1.json
  python3 -c "import json,sys; c=json.load(open('benchmarks/cameras_v2.json')); c['bicycle']['contrib_floor']=1.0/$1; json.dump(c,open('$p','w'),indent=1)"
  echo $p
}
run() {  # name inv [ENV=V ...]
  local tag=r$r-$1 inv=$2; shift 2
  local cj; cj=$(cam $inv)
  echo "=== $tag $(cut -c1-7 SHA) floor=1/$inv cams=$cj $* $(date +%T)"
  rm -rf tmp/t260-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t260 timeout ${RUN_TO:-200} \
    python3 render/run.py --no-ref --cameras $cj --iter-dir t260-$tag --dump-views t260-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^SUMMARY|Traceback|TT_THROW|TT_FATAL|error:|hard fail|too large" $S/run-$tag.log | sort | uniq -c | head -8
  if [ -d tmp/t260-dump-$tag ]; then
    (cd tmp/t260-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt) $(diff -q $REF $S/md5-$tag.txt >/dev/null && echo ALL_VIEWS_IDENTICAL_TO_GOLDEN || echo differs_from_golden)"
    rm -rf tmp/t260-dump-$tag
  fi
  return $rc
}
for s in ${*:-A:255}; do
  if [ "$s" = dump ]; then
    rm -rf $S/proj; mkdir -p $S/proj
    GSPLAT_TT_DUMP_PROJ=$S/proj TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t260 timeout 200 \
      python3 render/run.py --no-ref --view-range 0:1 --iter-dir t260-dump > $S/run-dump.log 2>&1
    echo "dump rc=$? $(ls -la $S/proj | tail -n +2 | awk '{print $5, $9}' | tr '\n' ' ')"
    continue
  fi
  IFS=: read -r name inv envs <<< "$s"
  run $name $inv ${envs//,/ }
done
echo "=== done round $r $(date +%T)"

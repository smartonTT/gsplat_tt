#!/bin/bash
# t266 (from t260 remote_time.sh): untraced bicycle 30-view timing of RECIP_NEWTON arms.
#   remote_time.sh <round> [arm ...]   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
# arm = name:INVFLOOR[:ENV=V,ENV2=V]; INVFLOOR 255 uses benchmarks/cameras_v2.json as is,
# others a copy with only contrib_floor = 1/INVFLOOR. Device hero -> tmp/t266/hero-r<r>-<name>.png.
# arm "dump-<name>[:ENV=V,...]" = default floor, hero only, GSPLAT_TT_DUMP_PROJ -> tmp/t266/proj-<name>,
# compacted by docs/floor-ab-t260/extract_dump.py to tmp/t266/proj_dev-<name>.npz.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t266; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t266; mkdir -p $S
C=/localdev/smarton/.cache/ttmc-gstt2-t266
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
  rm -rf tmp/t266-dump-$tag tmp/t266-$tag
  env "$@" TT_METAL_CACHE_RENDER=$C timeout ${RUN_TO:-240} \
    python3 render/run.py --no-ref --cameras $cj --iter-dir t266-$tag --dump-views t266-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^SUMMARY|Traceback|TT_THROW|TT_FATAL|error:|hard fail|too large" $S/run-$tag.log | sort | uniq -c | head -8
  [ -f tmp/t266-$tag/hero_clean.png ] && cp tmp/t266-$tag/hero_clean.png $S/hero-$tag.png
  local d; d=$(find . -maxdepth 3 -type d -name t266-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt)"
    rm -rf "$d"
  fi
  return $rc
}
for s in ${*:-off:255}; do
  case $s in
    dump-*)
      IFS=: read -r name envs <<< "${s#dump-}"
      rm -rf $S/proj-$name; mkdir -p $S/proj-$name
      env ${envs//,/ } GSPLAT_TT_DUMP_PROJ=$S/proj-$name TT_METAL_CACHE_RENDER=$C timeout 240 \
        python3 render/run.py --no-ref --view-range 0:1 --iter-dir t266-dump-$name > $S/run-dump-$name.log 2>&1
      echo "dump $name rc=$?"
      python3 docs/floor-ab-t260/extract_dump.py $S/proj-$name $S/proj_dev-$name.npz && rm -rf $S/proj-$name
      continue ;;
  esac
  IFS=: read -r name inv envs <<< "$s"
  run $name $inv ${envs//,/ }
done
echo "=== done round $r $(date +%T)"

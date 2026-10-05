#!/bin/bash
# t115: untraced bicycle 30-view timing of the tip on the remote tree, md5 vs md5-r82new.txt.
#   remote_time.sh <round> [arm ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
# arms: base = default env; dc = GSPLAT_TT_DUMP_CULL (hero slab dump -> dead_pairs.py); hp = GSPLAT_TT_HOST_PROFILE=1 + GSPLAT_PER_VIEW_STAGES=1
#       (host attribution, HPGAP/VIEW_STAGES lines); any "name:ENV=V[,ENV2=V2]" variant.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T115_TREE:-/localdev/smarton/gstt2-t115}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t115; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
run() {  # tag [ENV=V ...]
  local tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t115-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t115 timeout 330 \
    python3 render/run.py --no-ref --iter-dir t115-$tag --dump-views t115-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" $S/run-$tag.log | head -12
  local d; d=$(find . -maxdepth 3 -type d -name t115-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($(wc -l < $S/md5-$tag.txt) views)" \
      || echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"
    rm -rf "$d"
  fi
  return $rc
}
for s in ${*:-base}; do
  case $s in
    base) run base ;;
    hp) run hp GSPLAT_TT_HOST_PROFILE=1 GSPLAT_PER_VIEW_STAGES=1
        grep -E "^(HPGAP|HPPY|VIEW_STAGES)" $S/run-r$r-hp.log | head -200 > $S/hp-r$r.txt
        echo "hp lines: $(wc -l < $S/hp-r$r.txt)" ;;
    dc) mkdir -p $S/dumpcull; run dc GSPLAT_TT_DUMP_CULL=$S/dumpcull
        python3 docs/reprofile-t115/dead_pairs.py $S/dumpcull 2>&1 | tail -8 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } ;;
  esac
done
echo "=== done round $r $(date +%T)"

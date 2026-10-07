#!/bin/bash
# t212: re-baseline copy of docs/matblend-ready-t273/t289/remote_time.sh with the tree
# as T (default /localdev/smarton/gstt2-t212). md5 vs the golden list 906e0435.
#   remote_time.sh <round> <arm> ...   arm: base | name:ENV=V,ENV=V
#   (run on the measurement box through devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2-t212}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t212; mkdir -p $S
REF=${REF:-$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt}
r=${1:-1}; shift
run() {  # tag [ENV=V ...]
  local tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t212-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t212 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir t212-$tag --dump-views t212-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|Error|error:" $S/run-$tag.log | head -20
  local d; d=$(find . -maxdepth 3 -type d -name t212-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($(wc -l < $S/md5-$tag.txt) views)" \
      || echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"
    rm -rf "$d"
  fi
  return $rc
}
for s in "$@"; do
  case $s in
    base) run base ;;
    fuse) run fuse GSPLAT_TT_MATBLEND_FUSE=1 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } ;;
  esac
done
echo "=== done round $r $(date +%T)"

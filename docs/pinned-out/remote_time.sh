#!/bin/bash
# t367: untraced bicycle 30-view timing; md5 of each arm vs the base arm of the
# same round (base = default, GSPLAT_TT_OUT_PINNED off; golden 906e0435).
#   remote_time.sh <round> <arm> ...   arm: base | pin | name:ENV=V,ENV=V
#   (run on the measurement box through devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2-t367}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t367; mkdir -p $S
REF=${REF:-$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt}
r=${1:-1}; shift
run() {  # tag [ENV=V ...]
  local tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t367-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t367 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir t367-$tag --dump-views t367-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|Error|error:" $S/run-$tag.log | head -20
  local d; d=$(find . -maxdepth 3 -type d -name t367-dump-$tag | head -1)
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
    pin) run pin GSPLAT_TT_OUT_PINNED=1 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } ;;
  esac
  rc=$?
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $s: tt-smi -r"; tt-smi -r > $S/reset-r$r.log 2>&1; echo "reset rc=$?"; exit 124; fi
done
if [ -f $S/md5-r$r-base.txt ]; then
  for f in $S/md5-r$r-*.txt; do
    diff -q $S/md5-r$r-base.txt $f > /dev/null && echo "$(basename $f): IDENTICAL to base" || echo "$(basename $f): DIFFERS from base ($(diff $S/md5-r$r-base.txt $f | grep -c '^>'))"
  done
fi
echo "=== done round $r $(date +%T)"

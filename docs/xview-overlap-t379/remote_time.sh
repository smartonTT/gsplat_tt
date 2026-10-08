#!/bin/bash
# t379/t386: untraced bicycle 30-view timing of the cross-view overlap; md5 of each
# arm vs the golden 906e0435 and vs the base arm of the same round, and for the xv*
# arms xview_hits == views-1 (every view but the first had its pfwc prefetched).
#   remote_time.sh <round> <arm> ...   arm: base | xv | xvpin | xvzc | name:ENV=V,ENV=V
#   base  = defaults
#   xv    = GSPLAT_TT_XVIEW_OVERLAP=1
#   xvpin = GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_PINNED=1
#   xvzc  = GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_ZEROCOPY=1
#   (run on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2-t379}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t379; mkdir -p $S
REF=${REF:-$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt}
r=${1:-1}; shift
run() {  # tag [ENV=V ...]
  local arm=$1 tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t379-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t379 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir t379-$tag --dump-views t379-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|Error|error:" $S/run-$tag.log | head -20
  local d n=0; d=$(find . -maxdepth 3 -type d -name t379-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    n=$(wc -l < $S/md5-$tag.txt)
    diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($n views)" \
      || { echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $n)"; [ $rc = 0 ] && rc=6; }
    rm -rf "$d"
  fi
  case $arm in xv*)
    local hits; hits=$(grep -m1 "^STAGES " $S/run-$tag.log | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p')
    if [ -n "$hits" ] && [ "$n" -gt 0 ] && [ "$hits" = $((n - 1)) ]; then echo "XVIEW_HITS_OK hits=$hits views=$n"
    else echo "XVIEW_HITS_BAD hits=${hits:-none} views=$n"; [ $rc = 0 ] && rc=7; fi ;;
  esac
  return $rc
}
for s in "$@"; do
  case $s in
    base) run base ;;
    xv) run xv GSPLAT_TT_XVIEW_OVERLAP=1 ;;
    xvpin) run xvpin GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_PINNED=1 ;;
    xvzc) run xvzc GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_ZEROCOPY=1 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } ;;
  esac
  rc=$?
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $s: tt-smi -r"; tt-smi -r > $S/reset-r$r.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
done
if [ -f $S/md5-r$r-base.txt ]; then
  for f in $S/md5-r$r-*.txt; do
    diff -q $S/md5-r$r-base.txt $f > /dev/null && echo "$(basename $f): IDENTICAL to base" || echo "$(basename $f): DIFFERS from base ($(diff $S/md5-r$r-base.txt $f | grep -c '^>'))"
  done
fi
echo "=== done round $r $(date +%T)"
exit ${fail:-0}

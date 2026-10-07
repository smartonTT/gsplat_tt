#!/bin/bash
# t358: remote half of the mover-table A/B. Untraced bicycle 30-view rounds (same command as
# #346/#350, GSPLAT_PER_VIEW_STAGES=1 for the sort bin_emit split), arms interleaved per round:
#   A = GSPLAT_TT_MOVER_TABLE=p100a (the pre-#358 table on every board), B = unset (detected board).
# Per run: md5 of the 30 dumped views vs the 906e0435 golden list, device hero copy.
#   T=<tree> TTMH=<tt-metal> CACHE=<jit cache> O=<out> [MESH=P100] bench_ab.sh <rounds>
set -u
export TT_METAL_HOME=$TTMH TT_METAL_RUNTIME_ROOT=$TTMH TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1
export PYTHONDONTWRITEBYTECODE=1 GSPLAT_PER_VIEW_STAGES=1
[ -n "${MESH:-}" ] && export MESH_DEVICE=$MESH
cd "$T" || exit 1; source .venv/bin/activate; mkdir -p "$O"
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ)"
run_one() {  # round arm
  local t=t358-r$1-$2 rr
  rm -rf tmp/$t tmp/$t-dump
  echo "=== r$1 $2 start $(date -u +%T)"
  local e=(); [ "$2" = A ] && e=(GSPLAT_TT_MOVER_TABLE=p100a)
  env "${e[@]}" TT_METAL_CACHE_RENDER=$CACHE timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/r$1-$2.log 2>&1
  rr=$?; echo "run r$1 $2 rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|mover table|Traceback|TT_THROW|TT_FATAL" $O/r$1-$2.log | head -8
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-r$1-$2.log 2>&1; return 124; fi
  [ $rr = 0 ] || return $rr
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-r$1-$2.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-r$1-$2.png
  diff -q $REF $O/md5-r$1-$2.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$1 $2 ($(wc -l < $O/md5-r$1-$2.txt))" \
    || echo "VIEWS_DIFFER r$1 $2 ($(diff $REF $O/md5-r$1-$2.txt | grep -c '^>'))"
}
rc=0
for r in $(seq 1 ${1:-3}); do
  arms="A B"; [ $((r % 2)) = 0 ] && arms="B A"
  for a in $arms; do run_one $r $a || { rc=$?; break 2; }; done
done
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-A.log 2>/dev/null
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

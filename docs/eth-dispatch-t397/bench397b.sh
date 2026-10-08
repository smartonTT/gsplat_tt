#!/bin/bash
# t397b (from bench397.sh): remote half on bh-30, viewer stopped by drive397b.sh. Isolates why eth
# dispatch output differs from the 906e0435 golden: GSPLAT_TT_GRID_X caps every stage's compute grid.
#   E11 = eth at 11x10 (the worker shape): 906e0435 here means the difference is the core count.
#   W11 = worker 11x10 and E12 = eth 12x10 recheck this build (expect 906e0435 / the A/B eth list).
#   W10 = worker at 10x10: does a core-count change alone move bits under worker dispatch?
# Untraced, GSPLAT_PER_VIEW_STAGES=1, 30 views each. Always restarts the viewer at exit (vstart.sh).
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out397b
trap 'echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart397.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
REFE=$T/docs/eth-dispatch-t397/out/md5-r1-E.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
bash opt/eth/make_overlay.sh $V/tt-metal $P/ttm-eth12v2 12 > $O/overlay.log 2>&1 || { echo 2 > $O/bench.rc; exit 0; }
source opt/eth/env.sh $P/ttm-eth12v2 $P/cache392 || { echo 2 > $O/bench.rc; exit 0; }
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <dispatch> <grid_x, 0 = full>
  local t=t397b-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 dispatch=$2 grid_x=$3 start $(date -u +%T)"
  GSPLAT_TT_DISPATCH=$2 GSPLAT_TT_GRID_X=$3 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-400 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$1.log 2>&1; rc=124; return 1; fi
  [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  echo "MD5 $1 list=$(md5sum < $O/md5-$1.txt | cut -c1-8) vs906e0435_differ=$(diff $REF $O/md5-$1.txt | grep -c '^>') vsEth12_differ=$(diff $REFE $O/md5-$1.txt | grep -c '^>')"
}
run E11 eth 11 && run W11 worker 0 && run E12 eth 0 && run W10 worker 10
ls -la $O | cut -c25-
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

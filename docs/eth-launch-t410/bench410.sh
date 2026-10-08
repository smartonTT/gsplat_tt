#!/bin/bash
# t410: remote half on bh-30, viewer stopped by drive410.sh. Tree = p150bench/tree392 synced to the t410 head.
# Equal-grid launch-overhead check: E arm = eth dispatch (GSPLAT_TT_DISPATCH unset -> render/eth_default.py
#   overlay tmp/ttm-eth12) capped to 11x10 with GSPLAT_TT_GRID_X=11; W arm = GSPLAT_TT_DISPATCH=worker (11x10).
#   ARMS="E W" by default; ARMS can add X = a candidate env (EXTRA_X) on the eth default 12x10.
# Step 0: one 1-view smoke per arm. Then $1 alternating untraced 30-view rounds, GSPLAT_PER_VIEW_STAGES=1,
# md5 checked by opt/md5_golden.py against the golden of its own grid (11x10 906e0435). Restarts the viewer at exit.
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out410; ROUNDS=${1:-3}; ARMS=${ARMS:-E W}
trap 'echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart410.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache410w/render
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) arms=$ARMS card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
armenv() {  # armenv <E|W|X> -> env assignments
  case $1 in E) echo GSPLAT_TT_GRID_X=11 ;; W) echo GSPLAT_TT_DISPATCH=worker ;; X) echo ${EXTRA_X:-GSPLAT_X=0} ;; esac
}
armwant() { case $1 in E) echo 'dispatch eth .*compute grid 11x10' ;; W) echo 'dispatch worker .*compute grid 11x10' ;; X) echo 'dispatch eth .*compute grid 12x10' ;; esac; }
for a in $ARMS; do
  echo "=== s0-$a start $(date -u +%T) env: $(armenv $a)"
  env $(armenv $a) timeout 300 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t410-s0-$a > $O/s0-$a.log 2>&1; s0=$?
  echo "s0-$a rc=$s0 $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|$BAD|Traceback" $O/s0-$a.log | cut -c1-300 | head -8
  if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0-$a.log || ! grep -qE "^\[DEV\] $(armwant $a)" $O/s0-$a.log; then
    hang $s0 s0-$a; echo "S0_FAIL $a"; echo 10 > $O/bench.rc; exit 0
  fi
  echo "S0_PASS $a"
done
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <arm>
  local t=t410-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 start $(date -u +%T)"
  env $(armenv $2) timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|^(STAGES|SORT_STAGES|PROJECT_STAGES|MATBLEND_PROGRAM)|XVIEW_HITS|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-500 | head -9
  hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  python3 opt/md5_golden.py $O/md5-$1.txt $O/$1.log || rc=20
}
set -- $ARMS
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 0 ]; then order=$(echo $ARMS | awk '{for(i=NF;i>0;i--) printf "%s ", $i}'); else order=$ARMS; fi
  for a in $order; do run r$r-$a $a || break 2; done
done
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-*.log 2>/dev/null | head -2
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

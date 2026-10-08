#!/bin/bash
# t435: p150 A/B of iter 216 on bh-30 (viewer stopped by drive.sh). Arms, both ETH 12x10 default
# (GSPLAT_TT_DISPATCH unset): N = GSPLAT_TT_PERM_NOC default (7), Z = GSPLAT_TT_PERM_NOC=0 (iter-215 path).
# 1-view smoke per arm warms its JIT, then alternating untraced 30-view rounds; md5 checked by opt/md5_golden.py.
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out435; ROUNDS=${1:-3}
trap 'echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart435.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache435/render
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
for a in N Z; do
  [ $a = Z ] && pn=0 || pn=
  echo "=== s0-$a start $(date -u +%T)"
  env ${pn:+GSPLAT_TT_PERM_NOC=$pn} timeout 300 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t435-s0-$a > $O/s0-$a.log 2>&1; s0=$?
  echo "s0-$a rc=$s0 $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|$BAD|Traceback" $O/s0-$a.log | cut -c1-300 | head -8
  want='dispatch eth .*compute grid 12x10'
  if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0-$a.log || ! grep -qE "^\[DEV\] $want" $O/s0-$a.log; then
    hang $s0 s0-$a; echo "S0_FAIL $a"; echo 10 > $O/bench.rc; exit 0
  fi
  echo "S0_PASS $a"
done
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <N|Z>
  local t=t435-$1 rr pn=; [ $2 = Z ] && pn=0; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 start $(date -u +%T)"
  env ${pn:+GSPLAT_TT_PERM_NOC=$pn} timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|^(STAGES|SORT_STAGES|MATBLEND_PROGRAM)|XVIEW_HITS|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-400 | head -8
  hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  python3 opt/md5_golden.py $O/md5-$1.txt $O/$1.log || rc=20
}
for r in $(seq 1 $ROUNDS); do
  case $((r % 2)) in
    0) run r$r-Z Z && run r$r-N N ;;
    *) run r$r-N N && run r$r-Z Z ;;
  esac || break
done
cat tmp/ttm-eth12/.gsplat-eth-overlay > $O/overlay-marker.txt 2>&1
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-N.log 2>/dev/null
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

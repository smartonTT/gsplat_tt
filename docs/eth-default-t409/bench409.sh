#!/bin/bash
# t409: remote half on bh-30, viewer stopped by drive409.sh. Tree = p150bench/tree392 synced to the t409 head.
# D arm = the new default: GSPLAT_TT_DISPATCH unset, render/eth_default.py makes the overlay at
#   tmp/ttm-eth12 (from the viewer's tt-metal, read-only) with its own JIT cache and picks eth 12x10.
# W arm = GSPLAT_TT_DISPATCH=worker (11x10), own JIT cache. Both: xvpin + zero-copy defaults of the tip.
# Step 0: one 1-view smoke per arm (warms each JIT cache; D must log 'dispatch eth ... compute grid 12x10').
# 3 alternating untraced 30-view rounds (r1 W,D  r2 D,W  r3 W,D), GSPLAT_PER_VIEW_STAGES=1, every run's
# md5 list checked by opt/md5_golden.py against the golden of its own grid. Restarts the viewer at exit.
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out409; ROUNDS=${1:-3}
trap 'echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart409.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache409w/render
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
for a in D W; do
  [ $a = W ] && dsp=worker || dsp=
  echo "=== s0-$a start $(date -u +%T)"
  env ${dsp:+GSPLAT_TT_DISPATCH=$dsp} timeout 300 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t409-s0-$a > $O/s0-$a.log 2>&1; s0=$?
  echo "s0-$a rc=$s0 $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|$BAD|Traceback" $O/s0-$a.log | cut -c1-300 | head -8
  want='dispatch worker .*compute grid 11x10'; [ $a = D ] && want='dispatch eth .*compute grid 12x10'
  if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0-$a.log || ! grep -qE "^\[DEV\] $want" $O/s0-$a.log; then
    hang $s0 s0-$a; echo "S0_FAIL $a"; echo 10 > $O/bench.rc; exit 0
  fi
  echo "S0_PASS $a"
done
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <D|W>
  local t=t409-$1 rr dsp=; [ $2 = W ] && dsp=worker; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 start $(date -u +%T)"
  env ${dsp:+GSPLAT_TT_DISPATCH=$dsp} timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|^(STAGES|SORT_STAGES|MATBLEND_PROGRAM)|XVIEW_HITS|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-400 | head -8
  hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  python3 opt/md5_golden.py $O/md5-$1.txt $O/$1.log || rc=20
}
for r in $(seq 1 $ROUNDS); do
  case $r in
    2) run r2-D D && run r2-W W ;;
    *) run r$r-W W && run r$r-D D ;;
  esac || break
done
cat tmp/ttm-eth12/.gsplat-eth-overlay > $O/overlay-marker.txt 2>&1
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-W.log 2>/dev/null
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

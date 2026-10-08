#!/bin/bash
# t397 (from t387 bench387.sh): remote half on bh-30, viewer stopped by drive_bh30.sh.
# tree392 = t392 e2d6e822 (opt tip 58b47824 + t383 GSPLAT_TT_DISPATCH + opt/eth overlay v2).
# Step 0: overlay (12 ETH cores, idle-ERISC .ld 32 KB) + 1-view eth smoke (120 s).
# Steps 1-3: 3 alternating untraced 30-view rounds (r1 W,E  r2 E,W  r3 W,E), GSPLAT_PER_VIEW_STAGES=1,
# md5 of the 30 dumped views vs the 906e0435 golden list. Both arms use the same overlay and cache.
# Then one Tracy capture per arm (eth first) for per-core busy (120 vs 110 cores).
# Always restarts the viewer at exit (vstart.sh), in case the Mac-side driver is gone.
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out392; ROUNDS=${1:-3}
trap 'echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart397.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
bash opt/eth/make_overlay.sh $V/tt-metal $P/ttm-eth12v2 12 > $O/overlay.log 2>&1; orc=$?
echo "overlay rc=$orc"; tail -3 $O/overlay.log
[ $orc = 0 ] || { echo 2 > $O/bench.rc; exit 0; }
source opt/eth/env.sh $P/ttm-eth12v2 $P/cache392 || { echo 2 > $O/bench.rc; exit 0; }
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
echo "=== s0 start $(date -u +%T)"
GSPLAT_TT_DISPATCH=eth timeout 120 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t392-s0 > $O/s0.log 2>&1; s0=$?
echo "s0 rc=$s0 $(date -u +%T)"
grep -E "^\[DEV\] dispatch|$BAD|Traceback" $O/s0.log | cut -c1-300 | head -8
if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0.log || ! grep -qE '^\[DEV\] dispatch eth .*compute grid 12x10' $O/s0.log; then
  if [ $s0 = 124 ] || [ $s0 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-s0.log 2>&1; echo "reset rc=$?"; fi
  echo "S0_FAIL"; echo 10 > $O/bench.rc; exit 0
fi
echo "S0_PASS"
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <dispatch>
  local t=t397-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 dispatch=$2 start $(date -u +%T)"
  GSPLAT_TT_DISPATCH=$2 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-400 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$1.log 2>&1; rc=124; return 1; fi
  [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  diff -q $REF $O/md5-$1.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL $1 ($(wc -l < $O/md5-$1.txt))" \
    || echo "VIEWS_DIFFER $1 ($(diff $REF $O/md5-$1.txt | grep -c '^>'))"
}
for r in $(seq 1 $ROUNDS); do
  case $r in
    2) run r2-E eth && run r2-W worker ;;
    *) run r$r-W worker && run r$r-E eth ;;
  esac || break
done
unset GSPLAT_PER_VIEW_STAGES
if [ $rc = 0 ]; then
  source opt/eth/env.sh $P/ttm-eth12v2 $P/cache392-prof
  for arm in eth worker; do
    PR=$P/prof397-$arm; rm -rf $PR; mkdir -p $PR
    printf '#!/bin/bash\ncd %s; source .venv/bin/activate\nGSPLAT_TT_DISPATCH=%s python3 render/run.py --no-ref --iter-dir t397-T-%s\n' $T $arm $arm > $PR/inner.sh
    chmod +x $PR/inner.sh
    echo "=== tracy ($arm) start $(date -u +%T)"
    TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
      timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T-$arm.log 2>&1
    tr=$?; echo "tracy $arm rc=$tr $(date -u +%T)"
    grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES)|Traceback|TT_FATAL" $O/T-$arm.log | cut -c1-300 | head -4
    if [ $tr = 124 ] || [ $tr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-T-$arm.log 2>&1; fi
    trf=$(find $PR -name '*.tracy' | head -1)
    CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
    [ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-$arm.csv.gz && echo "tracy csv $arm $(stat -c %s $O/tracy-u-$arm.csv.gz) B"
    [ $tr = 0 ] || break
  done
fi
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-W.log 2>/dev/null
ls -la $O | cut -c25-
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc

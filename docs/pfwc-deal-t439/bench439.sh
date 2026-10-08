#!/bin/bash
# t439: p150 A/B of the weighted pfwc tile deal on bh-30 (viewer stopped by drive439.sh; restarted
# at exit via vstart439.sh, then out439/vstarted). Both arms ETH 12x10 default (GSPLAT_TT_DISPATCH unset):
#   S = strided deal (defaults), L = GSPLAT_TT_PFWC_DEAL=lpt with docs/pfwc-deal-t439/bicycle-hero-w.txt.
# 1-view smoke per arm, alternating untraced 30-view rounds (md5 per run; S checked by opt/md5_golden.py),
# then one Tracy capture per arm (views 0:10 + warmup, profiler overlay ttm-eth12p36 as #427).
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out439; ROUNDS=${1:-3}
rc=99
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc; echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart439.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache439/render
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED GSPLAT_TT_PFWC_DEAL
W=$T/docs/pfwc-deal-t439/bicycle-hero-w.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
armenv() { [ $1 = L ] && echo "GSPLAT_TT_PFWC_DEAL=lpt GSPLAT_TT_PFWC_DEAL_W=$W" || echo "GSPLAT_TT_PFWC_DEAL=0"; }
for a in S L; do
  echo "=== s0-$a start $(date -u +%T)"
  env $(armenv $a) timeout 300 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t439-s0-$a > $O/s0-$a.log 2>&1; s0=$?
  echo "s0-$a rc=$s0 $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|pfwc deal|$BAD|Traceback" $O/s0-$a.log | cut -c1-300 | head -8
  if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0-$a.log || ! grep -qE "^\[DEV\] dispatch eth .*compute grid 12x10" $O/s0-$a.log; then
    hang $s0 s0-$a; echo "S0_FAIL $a"; rc=10; exit
  fi
  if [ $a = L ] && ! grep -q 'pfwc deal lpt' $O/s0-$a.log; then echo "S0_FAIL L: no deal line"; rc=11; exit; fi
  echo "S0_PASS $a"
done
export GSPLAT_PER_VIEW_STAGES=1
rc=0
run() {  # run <tag> <S|L>
  local t=t439-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 start $(date -u +%T)"
  env $(armenv $2) timeout ${RUN_TIMEOUT:-330} python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES)|$BAD|Traceback" $O/$1.log | cut -c1-400 | head -6
  hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  echo "LIST_MD5 $1 $(md5sum < $O/md5-$1.txt | cut -c1-8)"
  if [ $2 = S ]; then python3 opt/md5_golden.py $O/md5-$1.txt $O/$1.log || rc=20; fi
}
for r in $(seq 1 $ROUNDS); do
  case $((r % 2)) in
    0) run r$r-L L && run r$r-S S ;;
    *) run r$r-S S && run r$r-L L ;;
  esac || break
done
[ $rc = 0 ] || exit
# Tracy, one 10-view chunk per arm (as #427 E12 c0)
python3 opt/profiler/zone_hash_check.py --repo $T > $O/zone_hash_check.txt 2>&1 || { echo "ZONE_HASH_FAIL"; tail -3 $O/zone_hash_check.txt; rc=30; exit; }
source opt/eth/env.sh $P/ttm-eth12p36 $P/cache439-p36 || { rc=13; exit; }
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
for a in S L; do
  PR=$P/prof439/$a; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
$(armenv $a) GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t439-T$a --view-range 0:10
IN
  chmod +x $PR/inner.sh
  echo "=== tracy $a start $(date -u +%T)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T$a.log 2>&1
  tr=$?; echo "tracy $a rc=$tr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES)|pfwc deal|$BAD|Traceback|DRAM buffers were full" $O/T$a.log | cut -c1-400 | head -6
  hang $tr T$a; [ $tr = 0 ] || { rc=40; continue; }
  dl=$(find $PR -name profile_log_device.csv | head -1)
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev-$a.csv.gz || { echo "no device csv $a"; rc=41; }
done

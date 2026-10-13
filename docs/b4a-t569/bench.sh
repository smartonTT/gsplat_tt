#!/bin/bash
# t569 step 3 (remote, bh-30 = p150, ETH dispatch, viewer stopped by drive.sh): with the fix, fold verify,
# the 30-view md5 sweep + device hero (fold on), then back-to-back A/B fold off/on (2 rounds, ABBA).
#   bench.sh <out dir> <tree>       on = GSPLAT_TT_B4A_FOLD=1 (forced: ETH default is off), off = GSPLAT_TT_B4A_FOLD=0
set -u
O=$1; T=$2; rc=0
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/probe.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_TT_B4A_FOLD \
  GSPLAT_TT_B4A_VERIFY GSPLAT_TT_B4A_SHADOW GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y GSPLAT_TT_OUT_PINNED TT_METAL_WATCHER
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$V/tt-metal
echo "=== host $(hostname) $(date -u +%FT%TZ) sha=$(cat $T/SHA) load=$(cat /proc/loadavg)" | tee $O/host.txt
cd $T || exit 9; source .venv/bin/activate
source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13
run() {  # tag timeout env... -- run.py args...
  local tag=$1 to=$2 r; shift 2; local e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
  echo "=== $tag [${e[*]}] $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t569-$tag tmp/t569-dump-$tag
  env "${e[@]}" timeout -k 15 $to python3 render/run.py --no-ref --iter-dir t569-$tag "$@" > $O/run-$tag.log 2>&1; r=$?
  echo "$tag rc=$r"
  grep -E "^(B2B |SUMMARY|\[DEV\] dispatch|\[eth\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL" $O/run-$tag.log | cut -c1-300 | head -8
  grep -E "^\[B4A\] (fold verify view|fold verify .*MISMATCH|fold off|fold:)" $O/run-$tag.log | head -6
  if [ $r = 124 ] || [ $r = 137 ]; then echo "HANG $tag"; rc=124; tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $r = 0 ] || rc=$r
  return $r
}
chk_fold() { grep -q "^\[B4A\] fold: view K2 from the previous blend" $O/run-$1.log || { echo "FOLD_NOT_RUN $1"; rc=9; }; }
run vlat 200 GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3 -- ; chk_fold vlat
grep -q "^\[B4A\] fold verify view .* OK" $O/run-vlat.log && ! grep -q MISMATCH $O/run-vlat.log || { echo "B4A_VERIFY_FAIL vlat"; rc=8; exit 8; }
run vb2b 400 GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3 -- --back-to-back; chk_fold vb2b
grep -q "^\[B4A\] fold verify view .* OK" $O/run-vb2b.log && ! grep -q MISMATCH $O/run-vb2b.log || { echo "B4A_VERIFY_FAIL vb2b"; rc=8; exit 8; }
run dump 300 GSPLAT_TT_B4A_FOLD=1 -- --dump-views t569-dump-dump; chk_fold dump
d=$(find tmp -maxdepth 2 -type d -name t569-dump-dump | head -1)
if [ -n "$d" ]; then (cd "$d" && md5sum * | sort -k2) > $O/md5-dump.txt; rm -rf "$d"
  echo "VIEWS=$(wc -l < $O/md5-dump.txt) SWEEP_MD5=$(md5sum < $O/md5-dump.txt | cut -c1-8)"
else echo NO_DUMP_DIR; rc=7; fi
cp tmp/t569-dump/hero_clean.png $O/hero.png || { echo NO_HERO; rc=5; }
for a in off on on off; do
  n=$(( ${n:-0} + 1 ))
  if [ $a = off ]; then run b2b$n-$a 400 GSPLAT_TT_B4A_FOLD=0 -- --back-to-back
  else run b2b$n-$a 400 GSPLAT_TT_B4A_FOLD=1 -- --back-to-back; chk_fold b2b$n-$a; fi
done
rm -rf tmp/t569-*

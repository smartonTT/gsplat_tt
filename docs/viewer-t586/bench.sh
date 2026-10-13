#!/bin/bash
# t586 (remote, bh-30 = p150, ETH dispatch, viewer stopped by drive.sh; from docs/b4a-t577/bench.sh):
# best-iter-227 (GSPLAT_TT_BLEND_EXP1 default on) at the ETH 12x10 grid: the 30-view md5 sweep +
# device hero with the tree defaults, the same sweep with the fold off (the viewer's setting, must
# match), then 2 alternating b2b rounds fold off/on (device ms next to the viewer's page FPS).
#   bench.sh <out dir> <tree>
set -u
O=$1; T=$2; rc=0
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/probe.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_TT_B4A_FOLD \
  GSPLAT_TT_B4A_VERIFY GSPLAT_TT_B4A_SHADOW GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y GSPLAT_TT_OUT_PINNED TT_METAL_WATCHER \
  GSPLAT_TT_BLEND_EXP1
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$V/tt-metal
echo "=== host $(hostname) $(date -u +%FT%TZ) sha=$(cat $T/SHA) load=$(cat /proc/loadavg)" | tee $O/host.txt
cd $T || exit 9; source .venv/bin/activate
source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache586-eth || exit 13
run() {  # tag timeout env... -- run.py args...
  local tag=$1 to=$2 r; shift 2; local e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
  echo "=== $tag [${e[*]}] $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t586-$tag tmp/t586-dump-$tag
  env "${e[@]}" timeout -k 15 $to python3 render/run.py --no-ref --iter-dir t586-$tag "$@" > $O/run-$tag.log 2>&1; r=$?
  echo "$tag rc=$r"
  grep -E "^(B2B |SUMMARY|\[DEV\] dispatch|\[eth\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|BRISC_STACK_FAIL|\[run\] B2B pass .* differ" $O/run-$tag.log | cut -c1-300 | head -8
  grep -E "^\[B4A\] (fold off|fold:)|EXP1" $O/run-$tag.log | head -3
  if [ $r = 124 ] || [ $r = 137 ]; then echo "HANG $tag"; rc=124; tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $r = 0 ] || rc=$r
  return $r
}
sweep() {  # tag env... : 30-view dump, md5 list into $O/md5-<tag>.txt
  local tag=$1; shift
  run $tag 300 "$@" -- --dump-views t586-dump-$tag
  local d; d=$(find tmp -maxdepth 2 -type d -name t586-dump-$tag | head -1)
  if [ -n "$d" ]; then (cd "$d" && md5sum * | sort -k2) > $O/md5-$tag.txt; rm -rf "$d"
    echo "$tag VIEWS=$(wc -l < $O/md5-$tag.txt) SWEEP_MD5=$(md5sum < $O/md5-$tag.txt | cut -c1-8)"
  else echo "NO_DUMP_DIR $tag"; rc=7; fi
}
sweep dump
cp tmp/t586-dump/hero_clean.png $O/hero.png || { echo NO_HERO; rc=5; }
sweep dump0 GSPLAT_TT_B4A_FOLD=0
cmp -s $O/md5-dump.txt $O/md5-dump0.txt && echo "FOLD_OFF_MD5_SAME" || { echo "FOLD_OFF_MD5_DIFFERS"; rc=6; }
n=0
for a in off on on off; do  # 2 rounds, ABBA
  n=$((n + 1))
  if [ $a = off ]; then run b2b$n-$a 400 GSPLAT_TT_B4A_FOLD=0 -- --back-to-back
  else run b2b$n-$a 400 GSPLAT_TT_B4A_FOLD=1 -- --back-to-back; fi
done
for f in $O/run-b2b*-o*.log; do echo "$(basename $f .log) $(grep -o 'b2b_ms_frame=[0-9.]*' $f | tail -1)"; done | tee $O/ab.txt
rm -rf tmp/t586-*

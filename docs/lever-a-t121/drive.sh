#!/bin/bash
# t121 driver (Mac side): device A/B of lever A (one-launch sort v2) in ONE
# `ttp lock p100 -- ` hold, using the t124 remote scripts on a t121 tree.
# Same arms as docs/sort-onelaunch-v2-t124/drive.sh, but every phase stops the
# chain on failure (a hang must not keep the device for 13 x 330 s), and the
# Tracy text outputs are copied back to $OUT.
#   drive.sh <rev> [phase ...]   phases: chk ab tracy (default: all three)
set -o pipefail
REV=${1:?rev}; shift; PHASES=${*:-chk ab tracy}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t121
OUT=${OUT:-tmp/t121-out}; mkdir -p "$OUT"
J="T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_job.sh"
CHK=chk:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_SORT_ONELAUNCH_CHECK=1
CHKW=chkw:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_SORT_ONELAUNCH_CHECK=1,GSPLAT_TT_SFPU_VIS=0,TT_METAL_WATCHER=2
KILL=kill:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_PB=1,GSPLAT_TT_OL_RING=0,GSPLAT_TT_OL_MAT_SELECT=0
NOSEL=nosel:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_MAT_SELECT=0
V1SEL=v1sel:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_PB=1,GSPLAT_TT_OL_RING=0
r() { ssh -o BatchMode=yes $H "$@"; }
step() {  # name cmd...: run one remote step, stop the chain on failure
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
for PH in $PHASES; do case $PH in
chk)
  out=$(r "VIEWS=0:5 TMO=600 $J c $CHK"); rc=$?; echo "$out"; echo "chk rc=$rc"
  [ $rc = 0 ] || { echo CHK_FAIL; exit 2; }
  echo "$out" | grep -q 'bad_tiles=[1-9]' && { echo CHK_BAD_TILES; exit 3; }
  echo "$out" | grep -q 'VIEWS DIFFER' && { echo CHK_VIEWS_DIFFER; exit 4; }
  echo "$out" | grep -q 'ALL_VIEWS_IDENTICAL' || { echo CHK_NO_DUMP; exit 5; }
  echo "$out" | grep -q 'ONELAUNCH v2 OL_PB=8 OL_RING=8 OL_WIN_PAGES=1024 OL_MAT_SELECT=1' || { echo CHK_NO_V2_LINE; exit 7; }
  out=$(r "VIEWS=0:3 TMO=600 $J w $CHKW"); rc=$?; echo "$out"; echo "chkw rc=$rc"
  echo "$out" | grep -q 'bad_tiles=[1-9]' && { echo CHKW_BAD_TILES; exit 6; }
  ;;
ab)
  step hang "$J h on on on"
  step r1 "$J 1 base on"
  step r2 "$J 2 on base"
  step r3 "$J 3 base on"
  step r4 "$J 4 $KILL $NOSEL"
  step r5 "$J 5 $V1SEL"
  step ms "$J m ms:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_MAT_STATS=1"
  ;;
tracy)
  for t in t121-ol t121-mprof; do
    mp=0; [ $t = t121-mprof ] && mp=1
    step tracy-$t "MPROF=$mp T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh $t"
    for f in gaps zones deep; do
      scp -q -o BatchMode=yes $H:$D/opt/profiler/$t/$f.txt "$OUT/$t-$f.txt" || echo "no $t/$f.txt"
    done
  done
  ;;
*) echo "unknown phase $PH"; exit 9 ;;
esac; done
echo ALLDONE

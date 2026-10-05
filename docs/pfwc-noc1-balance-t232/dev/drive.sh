#!/bin/bash
# t232: device runs of the BRISC-reader column mask (GSPLAT_TT_PFWC_RD_BRISC) with P2
# (GSPLAT_TT_PFWC_COV2D_SFPU=1) against today's default. Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps]   steps: any of sync smoke p<name> 1 2 3 4
#   MASK (default 0xF000 = physical NoC0 x 12..15) is the fix arm's column mask.
#   smoke: 30 views fix at the default open, md5-gated; on a "too large" TT_FATAL retry at
#          KX=32 and run every later untraced arm at KX=32.
#   p<name>: Tracy STEPCYC=1 STEPRISC=9 views 0:4, KX 32 (+8 on a too-large TT_FATAL), arms
#          pf = P2 + MASK, p0 = P2 alone, pb = default, pa = P2 + all columns,
#          pm<hex> = P2 + that mask (e.g. pmE000).
#   1-4:   swapped untraced 30-view rounds base (default) / fix (P2 + MASK) [/ rdb (MASK alone)
#          with ARM3=1]; hero_clean.png of each fix arm is fetched (device screenshot).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t232
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/pfwc-noc1-balance-t232/dev
O=$P/out; mkdir -p $O
MASK=${MASK:-0xF000}
CV=GSPLAT_TT_PFWC_COV2D_SFPU=1
FIX=$CV,GSPLAT_TT_PFWC_RD_BRISC=$MASK
KX=${KX:-0}   # 0 = default open (auto +24 KB)
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() {
  scp -q -o BatchMode=yes "$H:$T/tmp/t232/run-r$1-$2.log" "$H:$T/tmp/t232/md5-r$1-$2.txt" $O/ 2>/dev/null
  scp -q -o BatchMode=yes "$H:$T/tmp/t232-r$1-$2/hero_clean.png" $O/hero-r$1-$2.png 2>/dev/null
}
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t232/md5-r$1-$2.txt" >/dev/null; }
gate() {  # round arm...: fetch logs, fail the chain on any md5 mismatch
  local r=$1; shift; local bad=0
  for a in "$@"; do
    fetch $r $a
    if md5ok $r $a; then echo "MD5_OK r$r-$a"; else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
big() { grep -qE "too large|Program size" $O/run-r$1-$2.log 2>/dev/null; }
tm() {  # round timeout arms...
  local r=$1 to=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t232-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
STEPS=" ${2:-sync smoke pf p0 1 2 3 4} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  ../../harness/bin/ssh-preflight $H || { rc=$?; echo "PREFLIGHT_RC=$rc"; echo CHAIN_DONE; exit $rc; }
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  if [ $KX = 0 ]; then
    RT=300 tm s 460 fix:$FIX; fetch s fix
    if md5ok s fix; then echo "MD5_OK s-fix (default open, auto +24 KB)"
    elif big s fix; then echo "FIX_TOO_LARGE at default open: retry KX=32"; KX=32
    else echo "MD5_GATE_FAIL s-fix"; echo CHAIN_DONE; exit 4; fi
  fi
  if [ $KX != 0 ]; then RT=300 tm s 460 fix$KX:$FIX,GSPLAT_TT_KCFG_EXTRA_KB=$KX; gate s fix$KX; fi
fi
echo "KX=$KX"
prof() {  # name kx env...
  local n=$1 kx=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t232-$n -- "KX=$kx bash $T/$P/remote_prof.sh t232-$n 1 $*"
  echo "PROF_${n}_RC=$?"
  local D=$T/opt/profiler/t232-$n
  for x in capture.log pc_split.txt dev.csv.gz; do scp -q -o BatchMode=yes "$H:$D/$x" $O/prof-$n-$x 2>/dev/null; done
  if [ $kx -lt 40 ] && grep -qE "too large|Program size" $O/prof-$n-capture.log 2>/dev/null; then
    echo "PROF_${n}_TOO_LARGE at KX=$kx: retry at $((kx + 8))"; prof $n $((kx + 8)) "$@"; return
  fi
  python3 $P/percol.py $O/prof-$n-dev.csv.gz | tee $O/percol-$n.txt
}
for s in $STEPS; do
  case $s in
    pf) prof pf 32 $CV GSPLAT_TT_PFWC_RD_BRISC=$MASK ;;
    p0) prof p0 32 $CV ;;
    pb) prof pb 32 GSPLAT_TT_NOOP=0 ;;
    pa) prof pa 32 $CV GSPLAT_TT_PFWC_RD_BRISC=0xFFFE ;;
    pm*) prof $s 32 $CV GSPLAT_TT_PFWC_RD_BRISC=0x${s#pm} ;;
  esac
done
if [ $KX = 0 ]; then X=GSPLAT_TT_NOOP=0; else X=GSPLAT_TT_KCFG_EXTRA_KB=$KX; fi
B=base:$X; F=fix:$FIX,$X; M=rdb:GSPLAT_TT_PFWC_RD_BRISC=$MASK,$X
if [ "${ARM3:-0}" = 1 ]; then
  ord=("$B $F $M" "$F $M $B" "$M $B $F" "$B $M $F"); arms="base fix rdb"
else
  ord=("$B $F" "$F $B" "$B $F" "$F $B"); arms="base fix"
fi
for r in 1 2 3 4; do
  has $r || continue
  tm $r 460 ${ord[$((r-1))]}; gate $r $arms
done
python3 $P/summ.py $O 2>&1 | tee $O/summary.txt
echo CHAIN_DONE

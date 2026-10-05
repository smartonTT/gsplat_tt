#!/bin/bash
# t228: device A/B of P2 (GSPLAT_TT_PFWC_COV2D_SFPU) on top of today's default (writer split +
# SFPU cov_cam, auto +24 KB). Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps]   steps: any of sync smoke size pk1 pk0 1 2 3 4
#   smoke: 30 views knob=1 at the default open, md5-gated; on a "too large" TT_FATAL retry at
#          KX=32 and run every later untraced arm (both knobs) at KX=32.
#   pk1 pk0: Tracy STEPCYC=1 STEPRISC=9 views 0:4, KX 32 (+8 on a too-large TT_FATAL).
#   1-4:   swapped untraced 30-view rounds base (knob 0) / cov2d (knob 1).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t228
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/pfwc-cov2d-sfpu-t228/dev
O=$P/out; mkdir -p $O
CV=GSPLAT_TT_PFWC_COV2D_SFPU=1
KX=${KX:-0}   # 0 = default open (auto +24 KB)
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t228/run-r$1-$2.log" "$H:$T/tmp/t228/md5-r$1-$2.txt" $O/ 2>/dev/null; }
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t228/md5-r$1-$2.txt" >/dev/null; }
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
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t228-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
STEPS=" ${2:-sync smoke size pk1 pk0 1 2 3 4} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  if [ $KX = 0 ]; then
    RT=300 tm s 460 cov2d:$CV; fetch s cov2d
    if md5ok s cov2d; then echo "MD5_OK s-cov2d (default open, auto +24 KB)"
    elif big s cov2d; then echo "COV2D_TOO_LARGE at default open: retry KX=32"; KX=32
    else echo "MD5_GATE_FAIL s-cov2d"; echo CHAIN_DONE; exit 4; fi
  fi
  if [ $KX != 0 ]; then RT=300 tm s 460 cov2d$KX:$CV,GSPLAT_TT_KCFG_EXTRA_KB=$KX; gate s cov2d$KX; fi
fi
echo "KX=$KX"
if has size; then  # pfwc ELF sizes in the untraced kernel cache (all variants compiled so far)
  lk ssh -o BatchMode=yes $H "find /localdev/smarton/.cache/ttmc-gstt2-t228 -name '*.elf' | grep -iE 'pfwc' | xargs -r ls -l | awk '{print \$5, \$9}' | sort -n | tail -24" > $O/elf-sizes.txt
  cat $O/elf-sizes.txt
fi
prof() {  # name kx env...
  local n=$1 kx=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t228-$n -- "KX=$kx bash $T/$P/remote_prof.sh t228-$n 1 $*"
  echo "PROF_${n}_RC=$?"
  local D=$T/opt/profiler/t228-$n
  for x in capture.log pc_split.txt dev.csv.gz; do scp -q -o BatchMode=yes "$H:$D/$x" $O/prof-$n-$x 2>/dev/null; done
  cat $O/prof-$n-pc_split.txt 2>/dev/null
  if [ $kx -lt 40 ] && grep -qE "too large|Program size" $O/prof-$n-capture.log 2>/dev/null; then
    echo "PROF_${n}_TOO_LARGE at KX=$kx: retry at $((kx + 8))"; prof $n $((kx + 8)) "$@"
  fi
}
has pk1 && prof k1 32 $CV
has pk0 && prof k0 32 GSPLAT_TT_PFWC_COV2D_SFPU=0
if [ $KX = 0 ]; then X=GSPLAT_TT_NOOP=0; else X=GSPLAT_TT_KCFG_EXTRA_KB=$KX; fi
B=base:$X; C=cov2d:$CV,$X
ord=("$B $C" "$C $B" "$B $C" "$C $B")
for r in 1 2 3 4; do
  has $r || continue
  tm $r 460 ${ord[$((r-1))]}; gate $r base cov2d
done
python3 $P/summ.py $O 2>&1 | tee $O/summary.txt
echo CHAIN_DONE

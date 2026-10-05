#!/bin/bash
# t229: device gate of the t219 stall-free blend bodies (GSPLAT_TT_BLEND_SCHED 0 / 1 = F / 2 = A2).
# Mac side, one ttp lock p100 per step, one sync + build for every level (the knob is a JIT define).
#   drive.sh [rev] [steps]   steps: any of sync smoke 1 2 3 p0 p1 p2
#   smoke: 30 views at levels 1 and 2 with the default device open, md5-gated. On a kernel config
#          "too large" TT_FATAL, retry at KX=32 and use GSPLAT_TT_KCFG_EXTRA_KB=$KX for all arms.
#   1-3:   rotated untraced 30-view rounds of levels 0/1/2, md5-gated.
#   p0-p2: 30-view Tracy capture per level at GSPLAT_TT_KCFG_EXTRA_KB=32 (+8 on a too-large TT_FATAL).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t229
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/blend-sched-t219/dev-t229
O=$P/out; mkdir -p $O
KX=${KX:-0}
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t229/run-r$1-$2.log" "$H:$T/tmp/t229/md5-r$1-$2.txt" $O/ 2>/dev/null; }
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t229/md5-r$1-$2.txt" >/dev/null; }
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
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t229-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
arm() {  # level -> remote_time arm spec
  local e="GSPLAT_TT_BLEND_SCHED=$1"; [ $KX != 0 ] && e="$e,GSPLAT_TT_KCFG_EXTRA_KB=$KX"
  echo "L$1:$e"
}
STEPS=" ${2:-sync smoke 1 2 3 p0 p1 p2} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  RT=240 tm s 600 $(arm 1) $(arm 2); fetch s L1; fetch s L2
  if [ $KX = 0 ] && { big s L1 || big s L2; }; then
    echo "SMOKE_TOO_LARGE at default open: retry KX=32 for all arms"; KX=32
    RT=240 tm s 600 $(arm 1) $(arm 2)
  fi
  gate s L1 L2
fi
echo "KX=$KX"
ord=("0 1 2" "1 2 0" "2 0 1")
for r in 1 2 3; do
  has $r || continue
  a=""; for l in ${ord[$((r-1))]}; do a="$a $(arm $l)"; done
  tm $r 600 $a; gate $r L0 L1 L2
done
prof() {  # level kx
  local l=$1 kx=$2 n=t229-p$1
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n GSPLAT_TT_BLEND_SCHED=$l GSPLAT_TT_KCFG_EXTRA_KB=$kx"
  echo "PROF_p${l}_RC=$?"
  for f in zones.txt gaps.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/tracy-p$l-$f 2>/dev/null; done
  cat $O/tracy-p$l-gaps.txt 2>/dev/null
  if [ $kx -lt 48 ] && grep -qE "too large|Program size" $O/tracy-p$l-capture.log 2>/dev/null; then
    echo "PROF_p${l}_TOO_LARGE at KX=$kx: retry at $((kx + 8))"; prof $l $((kx + 8))
  fi
}
for l in 0 1 2; do has p$l && prof $l 32; done
echo CHAIN_DONE

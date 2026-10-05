#!/bin/bash
# t221: device A/B of the t207 pfwc writer split (GSPLAT_TT_PFWC_WRITER_SPLIT), alone and on
# top of t206 COVCAM_SFPU (cherry-picked for this run only). Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps]   steps: any of sync smoke size 1 2 3 4 c1 c2 c3 pk0 pws pboth
#   smoke: 30 views split=1 with the default device open (auto +8 KB), md5-gated; on a
#          "too large" TT_FATAL retry at KX=16. Then 30 views cov+split at KXC (16, else 24).
#   1-4:   swapped untraced 30-view rounds base/split, both arms KX (8) KB extra.
#   c1-c3: rotated rounds base/cov/both (cov = COVCAM_SFPU=1, both = cov + split), KXC.
#   pk0 pws pboth: Tracy STEPCYC=1 STEPRISC=9 views 0:4, KX 16 (pboth KXC), pfwc_p* / pfwc_ws.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t221
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/pfwc-writer-split-t207/dev-t221
O=$P/out; mkdir -p $O
WS=GSPLAT_TT_PFWC_WRITER_SPLIT=1
CV=GSPLAT_TT_PFWC_COVCAM_SFPU=1
KX=${KX:-8}; KXC=${KXC:-16}
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t221/run-r$1-$2.log" "$H:$T/tmp/t221/md5-r$1-$2.txt" $O/ 2>/dev/null; }
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t221/md5-r$1-$2.txt" >/dev/null; }
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
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t221-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
STEPS=" ${2:-sync smoke size 1 2 3 4 c1 c2 c3 pk0 pws pboth} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
COMBO=1
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  RT=240 tm s 400 split:$WS
  fetch s split
  if md5ok s split; then echo "MD5_OK s-split (default open, auto +8 KB)"
  elif big s split; then
    echo "SPLIT_TOO_LARGE at default open: retry KX=16"; KX=16
    tm s 400 split16:$WS,GSPLAT_TT_KCFG_EXTRA_KB=16; gate s split16
  else echo "MD5_GATE_FAIL s-split"; echo CHAIN_DONE; exit 4; fi
  tm s 400 both$KXC:$CV,$WS,GSPLAT_TT_KCFG_EXTRA_KB=$KXC
  fetch s both$KXC
  if md5ok s both$KXC; then echo "MD5_OK s-both$KXC"
  elif big s both$KXC; then
    KXC=24; echo "BOTH_TOO_LARGE: retry KXC=$KXC"
    tm s 400 both$KXC:$CV,$WS,GSPLAT_TT_KCFG_EXTRA_KB=$KXC; fetch s both$KXC
    md5ok s both$KXC && echo "MD5_OK s-both$KXC" || { echo "BOTH_FAIL: combo steps skipped"; COMBO=0; }
  else echo "BOTH_MD5_FAIL: combo steps skipped"; COMBO=0; fi
fi
echo "KX=$KX KXC=$KXC COMBO=$COMBO"
if has size; then  # pfwc ELF sizes in the untraced kernel cache (all variants compiled so far)
  lk ssh -o BatchMode=yes $H "find /localdev/smarton/.cache/ttmc-gstt2-t221 -name '*.elf' | grep -iE 'pfwc' | xargs -r ls -l | awk '{print \$5, \$9}' | sort -n | tail -24" > $O/elf-sizes.txt
  cat $O/elf-sizes.txt
fi
X=GSPLAT_TT_KCFG_EXTRA_KB=$KX
B=base:$X; S=split:$WS,$X
ord=("$B $S" "$S $B" "$B $S" "$S $B")
for r in 1 2 3 4; do
  has $r || continue
  tm $r 400 ${ord[$((r-1))]}; gate $r base split
done
XC=GSPLAT_TT_KCFG_EXTRA_KB=$KXC
B=cbase:$XC; C=cov:$CV,$XC; W=both:$CV,$WS,$XC
ordc=("$B $C $W" "$C $W $B" "$W $B $C")
for r in 1 2 3; do
  [ $COMBO = 1 ] && has c$r || continue
  tm c$r 500 ${ordc[$((r-1))]}; gate c$r cbase cov both
done
prof() {  # name kx env...
  local n=$1 kx=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t221-$n -- "KX=$kx bash $T/$P/remote_prof.sh t221-$n 1 $*"
  echo "PROF_${n}_RC=$?"
  local D=$T/opt/profiler/t221-$n
  for x in capture.log pc_split.txt dev.csv.gz; do scp -q -o BatchMode=yes "$H:$D/$x" $O/prof-$n-$x 2>/dev/null; done
  cat $O/prof-$n-pc_split.txt 2>/dev/null
  if [ $kx -lt 24 ] && grep -qE "too large|Program size" $O/prof-$n-capture.log 2>/dev/null; then
    echo "PROF_${n}_TOO_LARGE at KX=$kx: retry at $((kx + 8))"; prof $n $((kx + 8)) "$@"
  fi
}
has pk0 && prof k0 16 GSPLAT_TT_PFWC_WRITER_SPLIT=0
has pws && prof ws 16 $WS
[ $COMBO = 1 ] && has pboth && prof both $(( KXC > 16 ? KXC : 16 )) $CV $WS
echo CHAIN_DONE

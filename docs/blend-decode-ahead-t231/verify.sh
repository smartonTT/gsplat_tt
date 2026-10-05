#!/bin/bash
# t231: default verify after the BLEND_DECODE_AHEAD default flip to 2. One sync + build, two swapped
# untraced 30-view rounds of def (default env) vs D0 (GSPLAT_TT_BLEND_DECODE_AHEAD=0), md5-gated, the
# device hero of each def arm fetched to tmp/t231/hero-rv<N>-def.png, then a 30-view Tracy capture of
# the default at GSPLAT_TT_KCFG_EXTRA_KB=32 (+8 on a too-large TT_FATAL).   verify.sh [rev] [steps]
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t231
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/blend-decode-ahead-t231
O=$P/out; mkdir -p $O tmp/t231
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync v1 v2 prof} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
bad=0
for r in v1 v2; do
  has $r || continue
  if [ $r = v1 ]; then a="def:GSPLAT_TT_DUMMY=0 D0:GSPLAT_TT_BLEND_DECODE_AHEAD=0"; else a="D0:GSPLAT_TT_BLEND_DECODE_AHEAD=0 def:GSPLAT_TT_DUMMY=0"; fi
  lk $DEVRUN --host $H --no-verify --timeout 360 --tag t231-$r -- "RUN_TO=150 bash $T/$P/remote_time.sh $r $a"
  echo "TIME_${r}_RC=$?"
  for s in def D0; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t231/run-r$r-$s.log" "$H:$T/tmp/t231/md5-r$r-$s.txt" $O/
    scp -q -o BatchMode=yes "$H:$T/tmp/t231-r$r-$s/hero_clean.png" tmp/t231/hero-r$r-$s.png 2>/dev/null
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t231/md5-r$r-$s.txt" >/dev/null; then echo "MD5_OK r$r-$s"; else echo "MD5_GATE_FAIL r$r-$s"; bad=1; fi
  done
done
[ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
prof() {  # kx
  local kx=$1 n=t231-def
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n GSPLAT_TT_KCFG_EXTRA_KB=$kx"
  echo "PROF_RC=$?"
  for f in zones.txt gaps.txt roofline.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/tracy-$f 2>/dev/null; done
  scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/dev30.csv "${TTP_RUN_DIR:-tmp/t231}/t231-dev30.csv" 2>/dev/null
  cat $O/tracy-gaps.txt 2>/dev/null
  if [ $kx -lt 48 ] && grep -qE "too large|Program size" $O/tracy-capture.log 2>/dev/null; then
    echo "PROF_TOO_LARGE at KX=$kx: retry at $((kx + 8))"; prof $((kx + 8))
  fi
}
has prof && prof 32
echo CHAIN_DONE

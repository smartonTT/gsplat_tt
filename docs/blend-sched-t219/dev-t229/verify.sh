#!/bin/bash
# t229: default verify after the BLEND_SCHED default flip to 2. One sync + build, then two rotated
# untraced 30-view rounds of def (default env) vs L0 (GSPLAT_TT_BLEND_SCHED=0), md5-gated.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t229
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/blend-sched-t219/dev-t229
O=$P/out; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T "${1:-HEAD}" || { echo SYNC_FAIL; exit 1; }
bad=0
for r in v1 v2; do
  if [ $r = v1 ]; then a="def:GSPLAT_TT_DUMMY=0 L0:GSPLAT_TT_BLEND_SCHED=0"; else a="L0:GSPLAT_TT_BLEND_SCHED=0 def:GSPLAT_TT_DUMMY=0"; fi
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t229-$r -- "RUN_TO=150 bash $T/$P/remote_time.sh $r $a"
  for s in def L0; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t229/run-r$r-$s.log" "$H:$T/tmp/t229/md5-r$r-$s.txt" $O/
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t229/md5-r$r-$s.txt" >/dev/null; then echo "MD5_OK r$r-$s"; else echo "MD5_GATE_FAIL r$r-$s"; bad=1; fi
  done
done
echo VERIFY_DONE; exit $bad

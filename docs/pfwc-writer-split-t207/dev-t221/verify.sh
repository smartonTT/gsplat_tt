#!/bin/bash
# t221 default flip check: sync HEAD, then untraced 30-view rounds of the new default
# (split + SFPU cov_cam, auto +24 KB), both off (=0, no extra KB) and cov alone (auto +8 KB).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t221; REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/pfwc-writer-split-t207/dev-t221; O=$P/out; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
D=def:GSPLAT_TT_NOOP=0
F=off:GSPLAT_TT_PFWC_WRITER_SPLIT=0,GSPLAT_TT_PFWC_COVCAM_SFPU=0
C=cov:GSPLAT_TT_PFWC_WRITER_SPLIT=0
lk opt/sync_remote.sh $H $T HEAD; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
for r in "v1 $D $F $C" "v2 $F $C $D" "v3 $C $D $F"; do
  set -- $r; n=$1; shift
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t221-$n -- "RUN_TO=200 bash $T/$P/remote_time.sh $n $*"
  echo "TIME_${n}_RC=$?"
  for a in def off cov; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t221/run-r$n-$a.log" "$H:$T/tmp/t221/md5-r$n-$a.txt" $O/ 2>/dev/null
    ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t221/md5-r$n-$a.txt" >/dev/null && echo "MD5_OK r$n-$a" || echo "MD5_FAIL r$n-$a"
  done
done
echo CHAIN_DONE

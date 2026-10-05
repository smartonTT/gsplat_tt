#!/bin/bash
# t187 device chain (Mac side, run detached via `ttp detach`; one `ttp lock p100` per step).
#   drive.sh <repo dir> <out dir>
# 1 sync trees a (f4d91df, stack cur_lm) and f (t177 cd707df + fix = fba971e); b (e8754ac+) synced before.
# 2 three paired rounds a/b, order swapped, md5 vs md5-r82new.txt (stop if b differs in round 1).
# 3 fold hang check on tree f: GSPLAT_TT_OL_EMIT_FOLD=1 untraced, then a Tracy capture with EMIT_PROF=1.
# 4 control: the same untraced fold run on tree c (cd707df, no fix), the build that hung in #177.
set -u
cd "$1"; O=$2; mkdir -p $O
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; L=/localdev/smarton
A=$L/gstt2-t187a; B=$L/gstt2-t187b; F=$L/gstt2-t187f; C=$L/gstt2-t187c
RR="bash $B/docs/emit-cur-l1-t187/remote_run.sh"
SR=${SR:-opt/sync_remote.sh}
lk() { ttp lock --timeout 0 p100 -- "$@"; }
fetch() { scp -q -o BatchMode=yes "$H:$L/t187-out/*" $O/ 2>/dev/null; }
lk bash -c "$SR $H $A f4d91df && $SR $H $F fba971ea5d76d1cfa41c00be3415b2579ad48980 && $SR $H $C cd707df"
rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
for r in 1 2 3; do
  if [ $r = 2 ]; then o1=a; o2=b; else o1=b; o2=a; fi
  t1=$([ $o1 = a ] && echo $A || echo $B); t2=$([ $o2 = a ] && echo $A || echo $B)
  lk $DEVRUN --host $H --no-verify --timeout 900 --tag t187-r$r -- "$RR r$r-$o1 $t1; $RR r$r-$o2 $t2"
  echo "ROUND${r}_RC=$?"; fetch
  if [ $r = 1 ] && ! ssh -o BatchMode=yes $H "diff -q $L/t82_scripts/md5-r82new.txt $L/t187-out/md5-r1-b.txt" >/dev/null; then
    echo "MD5_GATE_FAIL (r1 b)"; echo CHAIN_DONE; exit 4
  fi
done
lk $DEVRUN --host $H --no-verify --timeout 600 --tag t187-fold -- "RUN_TIMEOUT=300 $RR f-fold $F GSPLAT_TT_OL_EMIT_FOLD=1"
echo "FOLD_RC=$?"; fetch
lk $DEVRUN --host $H --no-verify --timeout 600 --tag t187-foldprof -- \
  "T187_TREE=$F GSPLAT_TT_OL_EMIT_FOLD=1 bash $B/docs/emit-cur-l1-t187/remote_tracy.sh t187-foldprof"
echo "FOLDPROF_RC=$?"
for f in zones.txt emit_parts.txt capture.log; do scp -q -o BatchMode=yes $H:$F/opt/profiler/t187-foldprof/$f $O/foldprof-$f; done
lk $DEVRUN --host $H --no-verify --timeout 600 --tag t187-ctl -- "RUN_TIMEOUT=240 $RR c-fold $C GSPLAT_TT_OL_EMIT_FOLD=1"
echo "CTL_RC=$?"; fetch
echo CHAIN_DONE

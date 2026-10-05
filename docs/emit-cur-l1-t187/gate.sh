#!/bin/bash
# t187 fold + v2 gate (Mac side, run detached via `ttp detach`; one `ttp lock p100` per step).
#   gate.sh <repo dir> <out dir> <tip rev> <cand rev>
# tip = smarton/tt-project-opt with the fix (fold off); cand = tip + t177 v2 table + t164 fold
# (GSPLAT_TT_OL_EMIT_FOLD=1). Three paired untraced rounds, order swapped, md5 vs md5-r82new.txt.
set -u
cd "$1"; O=$2; TIP=$3; CAND=$4; mkdir -p $O
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; L=/localdev/smarton
A=$L/gstt2-t187a; F=$L/gstt2-t187f
RR="bash $A/docs/emit-cur-l1-t187/remote_run.sh"
lk() { ttp lock --timeout 0 p100 -- "$@"; }
fetch() { scp -q -o BatchMode=yes "$H:$L/t187-out/{run,md5}-g*" $O/ 2>/dev/null; }
lk bash -c "opt/sync_remote.sh $H $A $TIP && opt/sync_remote.sh $H $F $CAND"
rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo GATE_DONE; exit $rc; }
for r in 1 2 3; do
  if [ $r = 2 ]; then o1=t; o2=c; else o1=c; o2=t; fi
  for o in $o1 $o2; do
    if [ $o = t ]; then a="$A"; else a="$F GSPLAT_TT_OL_EMIT_FOLD=1"; fi
    lk $DEVRUN --host $H --no-verify --timeout 450 --tag t187-g$r$o -- "RUN_TIMEOUT=360 $RR g$r-$o $a"
    echo "G${r}${o}_RC=$?"
  done
  fetch
  if [ $r = 1 ] && ! ssh -o BatchMode=yes $H "diff -q $L/t82_scripts/md5-r82new.txt $L/t187-out/md5-g1-c.txt" >/dev/null; then
    echo "MD5_GATE_FAIL (g1 c)"; echo GATE_DONE; exit 4
  fi
done
echo GATE_DONE

#!/bin/bash
# t140 lever C (GSPLAT_TT_PRECULL) device A/B (Mac side). Run from the worktree root, detached:
#   ttp detach t140-ab -- bash docs/precull-t140/drive.sh [SHA]
# Each device step is its own `ttp lock p100` + devrun job (t115 / t125 pattern).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t140
R="T115_TREE=$T bash $T/docs/reprofile-t115"
tt-project/harness/bin/ssh-preflight $H || exit $?
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || exit $rc
# 3 interleaved untraced rounds, kill switch (base) vs pre-cull, md5 vs md5-r82new.txt.
for r in 1 2 3; do
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t140-time-$r -- \
    "$R/remote_time.sh $r base precull:GSPLAT_TT_PRECULL=1"
  echo "TIME_${r}_RC=$?"
done
# Dead-record share without / with the pre-cull (hero slab dump -> dead_pairs.py).
# Host model predicts 19.6% -> ~14.3% dead and 6.1% fewer records on the hero view.
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t140-dc -- \
  "rm -rf $T/tmp/t115/dumpcull; $R/remote_time.sh 4 dc; rm -rf $T/tmp/t115/dumpcull; GSPLAT_TT_PRECULL=1 $R/remote_time.sh 5 dc"
echo "DC_RC=$?"
# Host attribution of the pre-cull arm (HPGAP / VIEW_STAGES): pfwc should grow, TA/sort/blend shrink.
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t140-hp -- \
  "$R/remote_time.sh 6 hp prehp:GSPLAT_TT_PRECULL=1,GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1"
echo "HP_RC=$?"
# Kernel binary sizes (PFWC_VIS was ~66 KB of the 70.6 KB kernel config buffer).
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 120 --tag t140-size -- \
  "find /localdev/smarton/.cache/ttmc-gstt2-t115 -name '*.elf' -path '*pfwc*' | xargs ls -l | sort -k5 -n | tail -20"
echo "SIZE_RC=$?"
echo CHAIN_DONE

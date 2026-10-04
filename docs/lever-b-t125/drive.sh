#!/bin/bash
# t125 / #122 lever B device A/B (Mac side). Run from the worktree root, detached:
#   ttp detach t122-ab -- bash docs/lever-b-t125/drive.sh [SHA]
# Each device step is its own `ttp lock p100` + devrun job (t115 pattern).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t122
R="T115_TREE=$T bash $T/docs/reprofile-t115"
# The harness lives at the project root, not in task worktrees.
"${TTP_PROJECT:-tt-project}"/harness/bin/ssh-preflight $H || exit $?
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || exit $rc
# 3 interleaved untraced rounds, kill switch (base) vs fused, md5 vs md5-r82new.txt.
for r in 1 2 3; do
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t122-time-$r -- \
    "$R/remote_time.sh $r base fuse:GSPLAT_TT_PFWC_FUSE=1"
  echo "TIME_${r}_RC=$?"
done
# Host attribution of the fused arm (HPGAP / VIEW_STAGES).
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t122-hp -- \
  "$R/remote_time.sh 4 fusehp:GSPLAT_TT_PFWC_FUSE=1,GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1"
echo "HP_RC=$?"
# 30-view Tracy of the fused arm (the kill-switch arm is t115's tip capture).
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t122-tracy -- \
  "GSPLAT_TT_PFWC_FUSE=1 $R/remote_tracy.sh t122-fuse"
echo "TRACY_RC=$?"
# Kernel binary sizes (PFWC_VIS was ~66 KB of the 70.6 KB kernel config buffer).
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 120 --tag t122-size -- \
  "find /localdev/smarton/.cache/ttmc-gstt2-t115 -name '*.elf' -path '*pfwc*' | xargs ls -l | sort -k5 -n | tail -20"
echo "SIZE_RC=$?"
echo CHAIN_DONE

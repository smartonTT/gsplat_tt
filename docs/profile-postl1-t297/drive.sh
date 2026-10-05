#!/bin/bash
# t297: post-L1 profile on the #293 head (fused default on), one ttp lock p100 around
# sync+build, 3 untraced rounds (def = defaults, hp = + GSPLAT_TT_HOST_PROFILE=1,
# GSPLAT_PER_VIEW_STAGES=1, the t267 host timers; order alternates), one Tracy capture
# (GSPLAT_TT_PFWC_WRITER_SPLIT=0: the only config whose fused CBs fit under the profiler,
# #293) and the device screenshot. Each devrun stays under the 600 s ceiling.
#   drive.sh            (Mac, repo root, via ttp detach)
#   drive.sh --locked   (inner part, already under the lock)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t297; P=docs/profile-postl1-t297; O=$P/out
DEF="def:GSPLAT_TT_DUMMY=0"; HP="hp:GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1"
if [ "${1:-}" = --locked ]; then
  set -o pipefail
  opt/sync_remote.sh $H $T HEAD || { echo SYNC_FAIL; exit 3; }
  for r in 1 2 3; do
    if [ $((r % 2)) = 1 ]; then a="$DEF $HP"; else a="$HP $DEF"; fi
    $DEVRUN --host $H --no-verify --timeout 600 --tag t297-r$r -- "RUN_TIMEOUT=280 bash $T/$P/remote_time.sh $r $a"
    echo "ROUND_${r}_RC=$?"
  done
  $DEVRUN --host $H --no-verify --timeout 560 --tag t297-ns -- "bash $T/$P/remote_tracy.sh t297-ns GSPLAT_TT_PFWC_WRITER_SPLIT=0"
  echo "TRACY_RC=$?"
  NAME=t297-postl1 T=/localdev/smarton/gstt2-shot opt/ttw/screenshot.sh 206 HEAD --locked
  echo "SHOT_RC=$?"
  exit 0
fi
mkdir -p $O tmp/t297
echo "=== t297 drive $(git rev-parse --short HEAD) $(date +%T)"
"$TTP_PROJECT/harness/bin/ssh-preflight" $H; rc=$?; echo "PREFLIGHT_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- bash $P/drive.sh --locked; rc=$?; echo "LOCK_RC=$rc"
scp -q -o BatchMode=yes "$H:$T/tmp/t297/run-r*.log" "$H:$T/tmp/t297/md5-r*.txt" "$H:$T/tmp/t297/tracy-t297-ns-*" $O/ 2>/dev/null
scp -q -o BatchMode=yes $H:$T/opt/profiler/t297-ns/dev30.csv tmp/t297/t297-ns-dev30.csv; echo "CSV_RC=$?"
echo "=== t297 drive done $(date +%T)"; echo CHAIN_DONE
exit $rc

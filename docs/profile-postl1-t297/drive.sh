#!/bin/bash
# t297: untraced host bridge on the #293 head (fused default on), one ttp lock p100:
# sync+build, then 3 alternating rounds of def (defaults) and hp (GSPLAT_TT_HOST_PROFILE=1,
# GSPLAT_PER_VIEW_STAGES=1, the t267 host timers). Also fetches #293's post-L1 Tracy CSV
# (t293-ns, GSPLAT_TT_PFWC_WRITER_SPLIT=0) from the t289 tree. Mac, repo root, via ttp detach.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t297; P=docs/profile-postl1-t297; O=$P/out
mkdir -p $O tmp/t297
echo "=== t297 drive $(git rev-parse --short HEAD) $(date +%T)"
"$TTP_PROJECT/harness/bin/ssh-preflight" $H; rc=$?; echo "PREFLIGHT_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
scp -q -o BatchMode=yes $H:/localdev/smarton/gstt2-t289/opt/profiler/t293-ns/dev30.csv tmp/t297/t293-ns-dev30.csv; echo "CSV_RC=$?"
ttp lock p100 -- bash -c "opt/sync_remote.sh $H $T HEAD && $DEVRUN --host $H --no-verify --timeout 1500 --tag t297 -- 'for r in 1 2 3; do if [ \$((r%2)) = 1 ]; then a=\"def:GSPLAT_TT_DUMMY=0 hp:GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1\"; else a=\"hp:GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1 def:GSPLAT_TT_DUMMY=0\"; fi; bash $T/$P/remote_time.sh \$r \$a; done'"
rc=$?; echo "RUN_RC=$rc"
scp -q -o BatchMode=yes "$H:$T/tmp/t297/run-r*.log" "$H:$T/tmp/t297/md5-r*.txt" $O/ 2>/dev/null
echo "=== t297 drive done $(date +%T)"; echo CHAIN_DONE
exit $rc

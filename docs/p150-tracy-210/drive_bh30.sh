#!/bin/bash
# t382: one viewer-downtime window on bh-30. The task holds resource 'viewer' exclusive and runs
# under the existing viewer reservation (no IRD reserve/extend/release). Stops the viewer, runs
# bench382.sh detached on the box, ALWAYS restarts the viewer right after, logs UTC stop/start,
# checks localhost:8091 and fetches the outputs. Uses the newest opt/viewer/viewer.sh
# (supervisor-aware stop/start) from origin/smarton/tt-project-opt, which only touches remote paths.
#   ttp lock p100 -- docs/p150-tracy-210/drive_bh30.sh [rounds]
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/p150-tracy-210; VDIR=/localdev/smarton/viewer; O=$P/out382
VSH=${TMPDIR:-/tmp}/viewer-t382.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench382.sh bh-30:$P/bench382.sh || exit 3
"${SSH[@]}" "rm -rf $O"
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  "${SSH[@]}" "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
trap restart EXIT
"${SSH[@]}" "setsid nohup bash $P/bench382.sh ${1:-3} > $P/bench382.log 2>&1 < /dev/null &"
for _ in $(seq 300); do "${SSH[@]}" "test -e $O/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench382.log"
restart; trap - EXIT
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" $D/out/ 2>/dev/null
echo "=== drive end $(date -u +%FT%TZ)"

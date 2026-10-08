#!/bin/bash
# t394: one viewer-downtime window on bh-30 for the t379 cross-view overlap p150 check. The task
# holds resource 'viewer' exclusive and uses the existing viewer reservation (no IRD
# reserve/extend/release, no p100 lock, no measurement box). Stops the viewer with the newest
# opt/viewer/viewer.sh, runs bench394.sh detached ON the box (it restarts the viewer itself on
# exit), then makes sure the viewer is up (viewer.sh start is a no-op if it is), checks
# localhost:8091 (page 200, websocket 101) and fetches the outputs to p150/out.
#   ttp detach t394-bench -- docs/xview-overlap-t379/p150/drive_bh30.sh [rounds]
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/xview-overlap-t379/p150; VDIR=/localdev/smarton/viewer; O=$P/out394
VSH=${TMPDIR:-/tmp}/viewer-t394.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench394.sh bh-30:$P/bench394.sh || exit 3
"${SSH[@]}" "rm -rf $O"
restart() {
  echo "=== viewer check $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
  sleep 5
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 page=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)" \
    "ws=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
trap restart EXIT
"${SSH[@]}" "setsid nohup bash $P/bench394.sh ${1:-3} > $P/bench394.log 2>&1 < /dev/null &"
for _ in $(seq 400); do "${SSH[@]}" "test -e $O/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench394.log"
restart; trap - EXIT
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" "bh-30:$P/bench394.log" $D/out/ 2>/dev/null
echo "=== drive end rc=$("${SSH[@]}" "cat $O/bench.rc") $(date -u +%FT%TZ)"

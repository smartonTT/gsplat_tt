#!/bin/bash
# t407: one viewer-downtime window on bh-30 under the existing viewer reservation (no IRD
# reserve/extend/release; the task holds resource 'viewer' exclusive). Stops the viewer, runs
# bench407.sh detached on the box (it restarts the viewer itself on exit), waits for it, makes
# sure the viewer is back, checks localhost:8091 and fetches the outputs.
#   ttp lock p100 -- docs/xvpin-tracy/drive_bh30.sh [rounds]
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/xvpin-tracy; VDIR=/localdev/smarton/viewer; O=$P/out407
VSH=${TMPDIR:-/tmp}/viewer-t407.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench407.sh bh-30:$P/bench407.sh || exit 3
"${SSH[@]}" "rm -rf $O"
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
"${SSH[@]}" "setsid nohup bash $P/bench407.sh ${1:-3} > $P/bench407.log 2>&1 < /dev/null &"
for _ in $(seq 360); do "${SSH[@]}" "test -e $O/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench407.log"
for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
"${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" $D/out/ 2>/dev/null
echo "=== drive end $(date -u +%FT%TZ)"

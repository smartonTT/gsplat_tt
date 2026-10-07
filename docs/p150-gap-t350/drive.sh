#!/bin/bash
# t350: one viewer-downtime window on bh-30: stop the viewer, run bench_gap.sh <args> on the box,
# ALWAYS restart the viewer (opt/viewer/viewer.sh start: same tree, port 8080, env), wait for
# READY, check http://localhost:8091 from the Mac, fetch the outputs.
# Runs under the task's exclusive 'viewer' resource; no IRD reserve/extend/release.
#   docs/p150-gap-t350/drive.sh bench r1 U P
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/p150-gap-t350; VDIR=/localdev/smarton/viewer
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench_gap.sh bh-30:$P/bench_gap.sh || exit 3
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 opt/viewer/viewer.sh start && break; sleep 10; done
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 opt/viewer/viewer.sh stop
trap restart EXIT
"${SSH[@]}" "bash $P/bench_gap.sh $*"
echo "bench rc=$?"
restart; trap - EXIT
mkdir -p $D/out
rsync -q -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes" "bh-30:$P/out350/" $D/out/
echo "=== drive end $(date -u +%FT%TZ)"

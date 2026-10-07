#!/bin/bash
# t366: one viewer-downtime window on bh-30 (task holds resource 'viewer' exclusive; existing
# viewer reservation, no IRD reserve/extend/release): stop viewer, run bh30_tracy.sh, ALWAYS
# restart the viewer, check localhost:8091, fetch outputs. Pattern: docs/p150-gap-t350/drive.sh.
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench-t355; D=docs/p150-blend-gap; VDIR=/localdev/smarton/viewer
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bh30_tracy.sh bh-30:$P/t366_tracy.sh || exit 3
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 opt/viewer/viewer.sh start && break; sleep 10; done
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/SHA)"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 opt/viewer/viewer.sh stop
trap restart EXIT
"${SSH[@]}" "bash $P/t366_tracy.sh ${1:-t366}"; echo "bench rc=$?"
restart; trap - EXIT
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$P/out366/*" $D/out/
echo "=== drive end $(date -u +%FT%TZ)"

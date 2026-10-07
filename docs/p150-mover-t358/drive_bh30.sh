#!/bin/bash
# t358: p150 A/B on bh-30 under the existing viewer reservation (no IRD reserve/extend/release).
# Stop the viewer, run bench_ab.sh detached on the box, ALWAYS restart the viewer right away
# (opt/viewer/viewer.sh start), log UTC stop/start, check localhost:8091, fetch the outputs.
# The tree (/localdev/smarton/p150bench/tree358) is built beforehand with nice/ionice, -j nproc/2.
#   ttp lock viewer -- docs/p150-mover-t358/drive_bh30.sh <rounds>
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/p150-mover-t358; VDIR=/localdev/smarton/viewer; O=$P/out358
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench_ab.sh bh-30:$P/bench_ab.sh || exit 3
"${SSH[@]}" "rm -rf $O"
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
"${SSH[@]}" "T=$P/tree358 TTMH=$VDIR/tt-metal CACHE=$P/cache O=$O setsid nohup bash $P/bench_ab.sh ${1:-3} > $P/bench358.log 2>&1 < /dev/null &"
for _ in $(seq 240); do "${SSH[@]}" "test -e $O/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench358.log"
restart; trap - EXIT
mkdir -p $D/out-p150
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" $D/out-p150/
echo "=== drive end $(date -u +%FT%TZ)"

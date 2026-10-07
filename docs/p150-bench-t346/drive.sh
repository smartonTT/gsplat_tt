#!/bin/bash
# t346: stop the bh-30 viewer, run the p150 bench (bench_remote.sh, detached on the box so an ssh
# drop does not kill it), then ALWAYS restart the viewer as before (opt/viewer/viewer.sh start:
# same tree 746e9e9d, port 8080, same env) and fetch the outputs.
#   ttp detach t346-drive -- ttp lock viewer -- docs/p150-bench-t346/drive.sh <rounds>
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/p150-bench-t346
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench_remote.sh bh-30:$P/bench_remote.sh || exit 3
"${SSH[@]}" "rm -rf $P/out"
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 opt/viewer/viewer.sh start && break; sleep 10; done
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 opt/viewer/viewer.sh stop
trap restart EXIT
"${SSH[@]}" "setsid nohup bash $P/bench_remote.sh ${1:-1} > $P/bench.log 2>&1 < /dev/null &"
for _ in $(seq 180); do "${SSH[@]}" "test -e $P/out/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench.log"
restart; trap - EXIT
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$P/out/*" $D/out/
echo "=== drive end $(date -u +%FT%TZ)"

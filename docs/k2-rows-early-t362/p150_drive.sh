#!/bin/bash
# t362: p150 A/B on bh-30 under the existing viewer reservation (no reserve/extend/release),
# holding resource 'viewer' exclusively. Sync + build into a separate tree (nice -n 19 ionice -c3,
# half the cores, viewer's own tt-metal/venv/scenes) while the viewer still runs; then stop the
# viewer for the bench only (p150_remote.sh, detached on the box), and ALWAYS restart it.
#   ttp detach t362-p150 -- ttp lock viewer -- docs/k2-rows-early-t362/p150_drive.sh <rev> [rounds]
set -u
cd "$(git rev-parse --show-toplevel)"
rev=${1:?rev}; P=/localdev/smarton/p150bench-t362; V=/localdev/smarton/viewer; D=docs/k2-rows-early-t362
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
SYNC_REMOTE_VIEWER=1 GSTT2_BASE=$V GSTT2_VENV=$V/venv GSTT2_SCENES=$V/scenes REMOTE_TT_METAL_HOME=$V/tt-metal \
  opt/sync_remote.sh bh-30 $P/tree "$rev" > $D/out/p150-sync.log 2>&1 || { tail -20 $D/out/p150-sync.log; echo "sync failed: viewer untouched"; exit 3; }
tail -2 $D/out/p150-sync.log
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/p150_remote.sh bh-30:$P/p150_remote.sh || exit 3
"${SSH[@]}" "rm -rf $P/out"
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 opt/viewer/viewer.sh start && break; sleep 10; done
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 opt/viewer/viewer.sh stop
trap restart EXIT
"${SSH[@]}" "setsid nohup bash $P/p150_remote.sh ${2:-3} > $P/bench.log 2>&1 < /dev/null &"
for _ in $(seq 240); do "${SSH[@]}" "test -e $P/out/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench.log"
restart; trap - EXIT
mkdir -p $D/out/p150
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$P/out/*" $D/out/p150/
echo "=== drive end $(date -u +%FT%TZ)"

#!/bin/bash
# t397 (from t387 drive_bh30.sh): sync+build tree392 with the viewer running, then ONE viewer-downtime
# window on bh-30 under the existing viewer reservation (no IRD reserve/extend/release; the task holds
# resource 'viewer' exclusive). bench397.sh restarts the viewer itself at exit (vstart397.sh) and
# touches out392/vstarted; this side only falls back to opt/viewer/viewer.sh start if that never
# shows up. Logs UTC stop/start, checks localhost:8091 and fetches the outputs.
#   ttp lock p100 -- docs/eth-dispatch-t397/drive_bh30.sh
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/eth-dispatch-t397; VDIR=/localdev/smarton/viewer; O=$P/out392
VSH=${TMPDIR:-/tmp}/viewer-t397.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"$TTP_PROJECT/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; exit 3; }
bash $D/sync_bh30.sh || { echo "SYNC_FAIL: viewer untouched"; echo 3 > $D/drive.rc; exit 3; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench397.sh bh-30:$P/bench397.sh || exit 3
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/vstart.sh bh-30:$P/vstart397.sh || exit 3
"${SSH[@]}" "rm -rf $O"
restart_fallback() {
  echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"
  "${SSH[@]}" "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
}
check_viewer() {
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
trap 'restart_fallback; check_viewer' EXIT
"${SSH[@]}" "setsid nohup bash $P/bench397.sh ${1:-3} > $P/bench397.log 2>&1 < /dev/null &"
for _ in $(seq 900); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench397.log"
trap - EXIT
"${SSH[@]}" "test -e $O/vstarted" || restart_fallback
check_viewer
mkdir -p $D/out
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" $D/out/ 2>/dev/null
echo "=== drive end $(date -u +%FT%TZ)"
echo 0 > $D/drive.rc

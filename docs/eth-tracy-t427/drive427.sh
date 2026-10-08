#!/bin/bash
# t427 (from docs/eth-default-t409/drive409.sh): sync+build tree392 to the t427 head with the viewer
# running, then ONE viewer-downtime window on bh-30 under the existing viewer reservation (IRD job
# 135630; no reserve/extend/release; the task holds resource 'viewer' exclusive). bench427.sh
# restarts the viewer itself at exit (vstart427.sh) and touches out427/vstarted; this side only
# falls back to opt/viewer/viewer.sh start if that never shows up. Logs UTC stop/start, checks
# localhost:8091 (page 200, websocket 101) and fetches the outputs.
#   ttp detach t427 -- ttp lock p100 -- docs/eth-tracy-t427/drive427.sh
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; D=docs/eth-tracy-t427; VDIR=/localdev/smarton/viewer; O=$P/out427
VSH=${TMPDIR:-/tmp}/viewer-t427.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
SSH=(ssh "${SSHO[@]}" bh-30)
"$TTP_PROJECT/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; echo 3 > $D/drive.rc; exit 3; }
bash $D/sync_bh30.sh || { echo "SYNC_FAIL: viewer untouched"; echo 3 > $D/drive.rc; exit 3; }
scp -q "${SSHO[@]}" $D/bench427.sh bh-30:$P/bench427.sh || exit 3
scp -q "${SSHO[@]}" $D/vstart.sh bh-30:$P/vstart427.sh || exit 3
"${SSH[@]}" "rm -rf $O"
restart_fallback() {
  echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"
  "${SSH[@]}" "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
}
check_viewer() {
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
  local pg ws
  for _ in 1 2 3 4 5 6; do
    pg=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)
    ws=$(curl -s -o /dev/null -w '%{http_code}' -m 5 --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
      -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' http://localhost:8091/)
    [ "$pg" = 200 ] && [ "$ws" = 101 ] && break; sleep 10
  done
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 page $pg websocket $ws"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
trap 'restart_fallback; check_viewer' EXIT
"${SSH[@]}" "setsid nohup bash $P/bench427.sh > $P/bench427.log 2>&1 < /dev/null &"
for _ in $(seq 720); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench427.log"
trap - EXIT
"${SSH[@]}" "test -e $O/vstarted" || restart_fallback
check_viewer
mkdir -p $D/out
scp -q "${SSHO[@]}" "bh-30:$O/*" $D/out/ 2>/dev/null
echo "=== drive end $(date -u +%FT%TZ)"
echo 0 > $D/drive.rc

#!/bin/bash
# t478 (copy of docs/pfwc-rec32-t467/ab-t471/drive471.sh): sync+build tree478 with the viewer running, then ONE
# viewer-downtime window on bh-30 under the existing viewer reservation (no IRD reserve/extend/release;
# the run holds resource 'viewer' exclusive and the p100 lock). bench478.sh restarts the viewer itself at
# exit (vstart478.sh) and touches out478/vstarted; this side falls back to opt/viewer/viewer.sh start.
#   ttp detach t478 -- ttp lock viewer -- ttp lock p100 -- docs/tracy-221-t478/drive478.sh <outdir>
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"; D=$(cd "$(dirname "$0")" && pwd); OUT=${1:?outdir}
P=/localdev/smarton/p150bench; VDIR=/localdev/smarton/viewer; O=$P/out478
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20); SSH=(ssh "${SSHO[@]}" bh-30)
VSH=${TMPDIR:-/tmp}/viewer-t478.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH || exit 3
mkdir -p $OUT
"$TTP_PROJECT/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; exit 3; }
bash $D/sync478.sh || { echo "SYNC_FAIL: viewer untouched"; exit 3; }
for f in bench478.sh vstart478.sh; do scp -q "${SSHO[@]}" $D/$f bh-30:$P/.$f.tmp && "${SSH[@]}" "mv $P/.$f.tmp $P/$f" || exit 3; done
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
"${SSH[@]}" "setsid nohup bash $P/bench478.sh > $P/bench478.log 2>&1 < /dev/null &"
for _ in $(seq 360); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench478.log"
trap - EXIT
"${SSH[@]}" "test -e $O/vstarted" || restart_fallback
check_viewer
scp -q "${SSHO[@]}" "bh-30:$O/*" $OUT/ 2>/dev/null
echo "=== drive end $(date -u +%FT%TZ)"
"${SSH[@]}" "cat $O/bench.rc" 2>/dev/null | head -1 > $OUT/drive.rc

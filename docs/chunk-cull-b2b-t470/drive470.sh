#!/bin/bash
# t470 driver (Mac): b2b A/B of the #433 chunk-cull arms on bh-30 (the only p150 we hold; the measurement
# reservation is the p100 yyzo-bh-04) under the existing viewer reservation and the 2026-10-07 viewer
# exception: no IRD reserve/extend/release; the viewer is stopped only for bench470.sh, which restarts it.
# No build: tree470 is a copy of tree465b (13ce253e), whose device code equals this branch's (checked here).
#   ttp detach t470 -- ttp lock viewer -- ttp lock p100 -- docs/chunk-cull-b2b-t470/drive470.sh
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/chunk-cull-b2b-t470; H=bh-30; P=/localdev/smarton/p150bench; VDIR=/localdev/smarton/viewer
T=$P/tree470; SRC=$P/tree465b; O=$P/out470
MARK=${TTP_RUN_DIR:-tmp}/drive470.done; rm -f "$MARK"
trap 'rc=$?; echo "drive470 exit $rc"; echo $rc > "$MARK"' EXIT
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20); SSH=(ssh "${SSHO[@]}" $H)
put() { scp -q "${SSHO[@]}" "$1" "$H:$2.tmp470" && "${SSH[@]}" "mv $2.tmp470 $2"; }
VSH=${TMPDIR:-/tmp}/viewer-t470.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== prep $(date -u +%FT%TZ)"
"${SSH[@]}" "df -h /localdev | tail -1; ls -d $T $SRC 2>&1"
base=$("${SSH[@]}" "cat $T/SHA 2>/dev/null || cat $SRC/SHA")
dev=$(git diff --name-only "$base" HEAD -- render | grep -v '^render/run.py$' || true)
[ -z "$dev" ] || { echo "device code differs from tree base ${base:0:8}: $dev"; exit 6; }
echo "tree base ${base:0:8}: device code equal to HEAD $(git rev-parse --short HEAD)"
"${SSH[@]}" "test -d $T || { mkdir -p $T && cd $SRC && tar --exclude=./tmp -cf - . | tar -xf - -C $T && mkdir -p $T/tmp && echo copied $SRC; }"
put $D/run_t470.py $T/render/run_t470.py
put $D/bench470.sh $P/bench470.sh; put $D/vstart470.sh $P/vstart470.sh
"${SSH[@]}" "rm -rf $O"
restart_fallback() {
  echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"; "${SSH[@]}" "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=$H bash $VSH start && break; sleep 10; done
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=$H bash $VSH stop
trap 'rc=$?; restart_fallback; echo "drive470 exit $rc"; echo $rc > "$MARK"' EXIT
"${SSH[@]}" "VSTART=$P/vstart470.sh setsid nohup bash $P/bench470.sh $O $T 3 > $P/bench470.log 2>&1 < /dev/null &"
for _ in $(seq 400); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "test -e $O/vstarted" || restart_fallback
trap 'rc=$?; echo "drive470 exit $rc"; echo $rc > "$MARK"' EXIT
for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
"${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
mkdir -p $D/out; scp -q "${SSHO[@]}" "$H:$O/*" $D/out/; scp -q "${SSHO[@]}" "$H:$P/bench470.log" $D/out/
cat $D/out/bench.rc
echo "=== drive end $(date -u +%FT%TZ)"

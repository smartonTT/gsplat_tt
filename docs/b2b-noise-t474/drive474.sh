#!/bin/bash
# t474 driver (Mac): bh-30 b2b noise diagnosis under the existing viewer reservation (no IRD
# reserve/extend/release; the task holds 'viewer' exclusive and the p100 lock). The viewer is stopped only
# for diag474.sh, which restarts it at exit (vstart465.sh); the Mac restarts it as a fallback.
#   ttp detach t474 -- ttp lock viewer -- ttp lock p100 -- docs/b2b-noise-t474/drive474.sh <tag> "<variants>"
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/b2b-noise-t474; TAG=$1; VARS=$2
H=bh-30; P=/localdev/smarton/p150bench; VDIR=/localdev/smarton/viewer; O=$P/out474-$TAG; T=$P/tree465b
MARK=${TTP_RUN_DIR:-tmp}/drive474-$TAG.done; rm -f "$MARK"
trap 'rc=$?; echo "drive474 exit $rc"; echo $rc > "$MARK"' EXIT
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
put() { scp -q "${SSHO[@]}" "$2" "$1:$3.tmp474" && ssh "${SSHO[@]}" "$1" "mv $3.tmp474 $3"; }
VSH=${TMPDIR:-/tmp}/viewer-t474.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
put $H render/run.py $T/render/run_t474.py; put $H $D/diag474.sh $P/diag474.sh
restart_fallback() {
  echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"; ssh "${SSHO[@]}" $H "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 bash $VSH stop
trap 'rc=$?; restart_fallback; echo $rc > "$MARK"' EXIT
ssh "${SSHO[@]}" $H "VSTART=$P/vstart465.sh PASSES=${PASSES:-10} setsid nohup bash $P/diag474.sh $O $T '$VARS' > $P/diag474-$TAG.log 2>&1 < /dev/null &"
for _ in $(seq 160); do ssh "${SSHO[@]}" $H "test -e $O/vstarted" && break; sleep 5; done
ssh "${SSHO[@]}" $H "test -e $O/vstarted" || restart_fallback
trap 'rc=$?; echo $rc > "$MARK"' EXIT
for _ in $(seq 60); do ssh "${SSHO[@]}" $H "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
ssh "${SSHO[@]}" $H "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
mkdir -p $D/out-$TAG; scp -q "${SSHO[@]}" "$H:$O/*" $D/out-$TAG/; scp -q "${SSHO[@]}" "$H:$P/diag474-$TAG.log" $D/out-$TAG/
cat $D/out-$TAG/diag.rc

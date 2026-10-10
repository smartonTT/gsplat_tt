#!/bin/bash
# t465 driver (Mac): p100 leg on the measurement box (yyzo-bh-04), then p150 leg on bh-30 under the existing
# viewer reservation (no IRD reserve/extend/release there; the task holds resource 'viewer' exclusive).
# Builds run with the viewer up; the viewer is stopped only for bench465.sh, which restarts it at exit.
#   ttp detach t465 -- ttp lock viewer -- ttp lock p100 -- docs/b2b-rebaseline-t465/drive465.sh [p100|p150|both]
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/b2b-rebaseline-t465; LEGS=${1:-both}; A=best-iter-216; B=$(git rev-parse HEAD)
MARK=${TTP_RUN_DIR:-tmp}/drive465.done; rm -f "$MARK"
trap 'echo "drive465 exit $?" ; echo $? > "$MARK"' EXIT
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
put() { scp -q "${SSHO[@]}" "$2" "$1:$3.tmp465" && ssh "${SSHO[@]}" "$1" "mv $3.tmp465 $3"; }
if [ $LEGS != p150 ]; then
  H=yyzo-bh-04; R=/localdev/smarton; TA=$R/t465-216; TB=$R/t465-tip; O=$R/t465-out-p100
  "$TTP_PROJECT/harness/bin/ssh-preflight" $H
  echo "=== p100 sync $(date -u +%FT%TZ)"
  opt/sync_remote.sh $H $TA $A > tmp/t465-sync-A.log 2>&1 || { tail -20 tmp/t465-sync-A.log; exit 3; }
  opt/sync_remote.sh $H $TB $B > tmp/t465-sync-B.log 2>&1 || { tail -20 tmp/t465-sync-B.log; exit 3; }
  for T in $TA $TB; do put $H render/run.py $T/render/run_t465.py; done
  put $H $D/bench465.sh $R/t465-bench465.sh
  echo "=== p100 bench $(date -u +%FT%TZ)"
  $HOME/dev/tt-workflows/scripts/devrun.sh --host $H --no-verify --timeout 2400 --tag t465-p100 -- \
    "bash $R/t465-bench465.sh p100 $O $TA $TB 3" > tmp/t465-p100.out 2>&1 || echo "p100 devrun rc=$?"
  mkdir -p $D/out-p100; scp -q "${SSHO[@]}" "$H:$O/*" $D/out-p100/
  cat $D/out-p100/bench.rc
fi
if [ $LEGS != p100 ]; then
  H=bh-30; P=/localdev/smarton/p150bench; VDIR=/localdev/smarton/viewer; O=$P/out465
  VSH=${TMPDIR:-/tmp}/viewer-t465.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH
  "$TTP_PROJECT/harness/bin/ssh-preflight" $H
  echo "=== p150 sync $(date -u +%FT%TZ)"
  bash $D/sync_bh30_t465.sh tree465a $A
  bash $D/sync_bh30_t465.sh tree465b $B
  for t in tree465a tree465b; do put $H render/run.py $P/$t/render/run_t465.py; done
  put $H $D/bench465.sh $P/bench465.sh; put $H $D/vstart465.sh $P/vstart465.sh
  ssh "${SSHO[@]}" $H "rm -rf $O"
  restart_fallback() {
    echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"; ssh "${SSHO[@]}" $H "rm -f $VDIR/viewer.stop"
    for i in 1 2 3; do VIEWER_HOST=bh-30 bash $VSH start && break; sleep 10; done
  }
  echo "=== viewer stop $(date -u +%FT%TZ)"
  VIEWER_HOST=bh-30 bash $VSH stop
  trap 'rc=$?; restart_fallback; echo $rc > "$MARK"' EXIT
  ssh "${SSHO[@]}" $H "VSTART=$P/vstart465.sh setsid nohup bash $P/bench465.sh p150 $O $P/tree465a $P/tree465b 3 > $P/bench465.log 2>&1 < /dev/null &"
  for _ in $(seq 400); do ssh "${SSHO[@]}" $H "test -e $O/vstarted" && break; sleep 5; done
  ssh "${SSHO[@]}" $H "test -e $O/vstarted" || restart_fallback
  trap 'echo $? > "$MARK"' EXIT
  for _ in $(seq 60); do ssh "${SSHO[@]}" $H "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  ssh "${SSHO[@]}" $H "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
  mkdir -p $D/out-p150; scp -q "${SSHO[@]}" "$H:$O/*" $D/out-p150/; cp tmp/t465-p100.out $D/out-p100/devrun.out 2>/dev/null || true
  cat $D/out-p150/bench.rc
fi
echo "=== drive end $(date -u +%FT%TZ)"

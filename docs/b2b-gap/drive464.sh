#!/bin/bash
# t464 driver (Mac): copy scripts, stop viewer, run probe464.sh on bh-30 (it restarts the viewer at exit), fetch outputs.
set -u
WT=/Users/smarton/dev/gsplat_tt/tt-project/state/runs/1251/wt; D=$(cd "$(dirname "$0")" && pwd)
P=/localdev/smarton/p150bench; O=$P/out464; VDIR=/localdev/smarton/viewer
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20); SSH=(ssh "${SSHO[@]}" bh-30)
"$TTP_PROJECT/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; exit 3; }
for f in probe464.sh vstart464.sh run_t464.py; do scp -q "${SSHO[@]}" $D/$f bh-30:$P/.$f.tmp && "${SSH[@]}" "mv $P/.$f.tmp $P/$f" || exit 3; done
"${SSH[@]}" "rm -rf $O"
cd $WT
echo "=== viewer stop $(date -u +%FT%TZ)"; VIEWER_HOST=bh-30 bash opt/viewer/viewer.sh stop
"${SSH[@]}" "setsid nohup bash $P/probe464.sh > $P/probe464.log 2>&1 < /dev/null &"
for _ in $(seq ${WAIT_N:-66}); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "test -e $O/vstarted" || { echo "PROBE_NOT_DONE (viewer restarts at probe exit; check $O/vstarted)"; exit 4; }
"${SSH[@]}" "cat $P/probe464.log"
for _ in $(seq 40); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
"${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2"
echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
mkdir -p $D/out; scp -q "${SSHO[@]}" "bh-30:$O/*" $D/out/
echo "=== drive end $(date -u +%FT%TZ)"

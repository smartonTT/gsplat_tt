#!/bin/bash
# t511 driver (Mac): sync + build <rev> into /localdev/smarton/t511 on bh-33 (existing measurement
# reservation only; no IRD reserve/extend/release), then run bench511.sh there and copy results back.
# Every ssh keeps StrictHostKeyChecking=yes, off the ~/.ssh/config mux.
#   ttp detach t511 -- ttp lock p100 -- docs/blend-qf-t511/drive511.sh <rev>
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/blend-qf-t511; REV=$(git rev-parse "${1:?rev}")
H=bh-33; T=/localdev/smarton/t511; O=/localdev/smarton/t511-out
mkdir -p tmp
MARK=${TTP_RUN_DIR:-tmp}/drive511.done; rm -f "$MARK"
trap 'rc=$?; echo "drive511 exit $rc $(date -u +%FT%TZ)"; echo $rc > "$MARK"' EXIT
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
rsh() { ssh "${SSHO[@]}" $H "$@"; }
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== sync+build ${REV:0:8} $(date -u +%FT%TZ)"
opt/sync_remote.sh $H $T "$REV"
echo "=== build ok $(date -u +%FT%TZ)"
scp -q "${SSHO[@]}" $D/bench511.sh "$H:$T/bench511.sh.tmp" && rsh "mv $T/bench511.sh.tmp $T/bench511.sh"
rsh "rm -f $O/bench.rc; setsid nohup bash $T/bench511.sh $O $T ${ROUNDS:-3} > $T/bench511.log 2>&1 < /dev/null &"
for _ in $(seq 1080); do rsh "test -e $O/bench.rc" && break; sleep 5; done
mkdir -p $D/out; scp -q "${SSHO[@]}" "$H:$O/*" $D/out/; scp -q "${SSHO[@]}" "$H:$T/bench511.log" $D/out/
cat $D/out/bench511.log | grep -E "^===|rc=|^B2B" | tail -40
exit "$(cat $D/out/bench.rc)"

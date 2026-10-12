#!/bin/bash
# t514 driver (Mac): sync + build <rev> into /localdev/smarton/t514 on bh-33 (existing measurement
# reservation only; no IRD reserve/extend/release), then run bench514.sh there and copy results back.
# Every ssh keeps StrictHostKeyChecking=yes, off the ~/.ssh/config mux.
#   ttp detach t514 -- ttp lock p100 -- docs/gap-cut-t514/drive514.sh <rev>
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/gap-cut-t514; REV=$(git rev-parse "${1:?rev}")
H=bh-33; T=/localdev/smarton/t514; O=/localdev/smarton/t514-out
mkdir -p tmp
MARK=${TTP_RUN_DIR:-tmp}/drive514.done; rm -f "$MARK"
trap 'rc=$?; echo "drive514 exit $rc $(date -u +%FT%TZ)"; echo $rc > "$MARK"' EXIT
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
rsh() { ssh "${SSHO[@]}" $H "$@"; }
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== sync+build ${REV:0:8} $(date -u +%FT%TZ)"
opt/sync_remote.sh $H $T "$REV"
echo "=== build ok $(date -u +%FT%TZ)"
scp -q "${SSHO[@]}" $D/bench514.sh "$H:$T/bench514.sh.tmp" && rsh "mv $T/bench514.sh.tmp $T/bench514.sh"
rsh "rm -f $O/bench.rc; setsid nohup bash $T/bench514.sh $O $T ${ROUNDS:-5} > $T/bench514.log 2>&1 < /dev/null &"
for _ in $(seq 720); do rsh "test -e $O/bench.rc" && break; sleep 5; done
mkdir -p $D/out; scp -q "${SSHO[@]}" "$H:$O/*" $D/out/; scp -q "${SSHO[@]}" "$H:$T/bench514.log" $D/out/
cat $D/out/bench514.log | grep -E "^===|rc=|^B2B" | tail -40
exit "$(cat $D/out/bench.rc)"

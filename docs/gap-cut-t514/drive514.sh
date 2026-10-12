#!/bin/bash
# t514 driver (Mac): ship <rev> into /localdev/smarton/t514 on bh-33 (existing measurement
# reservation only; no IRD reserve/extend/release), then build + bench there as one remote
# `ttp detach --remote` job (survives ssh drops, #522) and copy results back.
# Every ssh keeps StrictHostKeyChecking=yes, off the ~/.ssh/config mux.
#   ttp detach t514 -- ttp lock --timeout 0 p100 -- docs/gap-cut-t514/drive514.sh <rev>
# The lock stays held while this script polls the remote job.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/gap-cut-t514; REV=$(git rev-parse "${1:?rev}")
H=bh-33; T=/localdev/smarton/t514; O=/localdev/smarton/t514-out; J=/localdev/smarton/t514-job
mkdir -p tmp
MARK=${TTP_RUN_DIR:-tmp}/drive514.done; rm -f "$MARK"
trap 'rc=$?; echo "drive514 exit $rc $(date -u +%FT%TZ)"; echo $rc > "$MARK"' EXIT
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
rsh() { ssh "${SSHO[@]}" $H "$@"; }
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== ship ${REV:0:8} $(date -u +%FT%TZ)"
git archive --format=tar "$REV" -- . ":(exclude)tests/fixtures/hero/*.npz" ":(exclude)opt/profiler/ttw-*" \
  ":(exclude,glob)**/*.tracy" ":(exclude,glob)**/profile_log_device*.csv" ":(exclude)opt/metal-screenshots" |
  rsh "mkdir -p $T && tar -m -x -C $T && echo $REV > $T/SHA"
# The build half of opt/sync_remote.sh (its REMOTE heredoc), run inside the remote job.
sed -n "/<<'REMOTE'\$/,/^REMOTE\$/p" opt/sync_remote.sh | sed '1d;$d' > tmp/build514.sh
for f in tmp/build514.sh $D/bench514.sh; do
  b=$(basename $f); scp -q "${SSHO[@]}" $f "$H:$T/$b.tmp"; rsh "mv $T/$b.tmp $T/$b"
done
ttp detach --remote $H --dir $J t514b -- bash -c \
  "cd $T && DIR=$T VIEWER_BUILD=0 JOBS=16 bash build514.sh && echo '=== build ok' && bash bench514.sh $O $T ${ROUNDS:-5}"
echo "=== remote job started $(date -u +%FT%TZ)"
for _ in $(seq 1440); do   # up to 2 h; 255 (ssh down) keeps polling
  sleep 5; set +e; ttp detach --check --host $H $J/t514b >/dev/null 2>&1; c=$?; set -e
  [ $c = 0 ] && break
done
mkdir -p $D/out
scp -q "${SSHO[@]}" "$H:$J/t514b.log" $D/out/remote.log || true
scp -q "${SSHO[@]}" "$H:$O/*" $D/out/ || true
grep -E "^===|rc=|^B2B|sync_remote|FAILED|error" $D/out/remote.log | tail -50 || true
exit "$(cat $D/out/bench.rc 2>/dev/null || echo 98)"

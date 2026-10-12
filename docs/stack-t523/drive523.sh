#!/bin/bash
# t523 driver (Mac): combined-stack A/B. Ship <rev> into /localdev/smarton/t523 on bh-33 (existing measurement
# reservation only; no IRD reserve/extend/release), then build + bench there as one remote
# `ttp detach --remote` job (survives ssh drops, #522) and copy results back.
# Every ssh keeps StrictHostKeyChecking=yes, off the ~/.ssh/config mux.
#   ttp detach t523 -- ttp lock --timeout 0 p100 -- docs/stack-t523/drive523.sh <rev>
# The lock stays held while this script polls the remote job.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/stack-t523; REV=$(git rev-parse "${1:?rev}")
H=bh-33; T=/localdev/smarton/t523; O=/localdev/smarton/t523-out; J=/localdev/smarton/t523-job
mkdir -p tmp
MARK=${TTP_RUN_DIR:-tmp}/drive523.done; rm -f "$MARK"
trap 'rc=$?; echo "drive523 exit $rc $(date -u +%FT%TZ)"; echo $rc > "$MARK"' EXIT
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
rsh() { ssh "${SSHO[@]}" $H "$@"; }
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== ship ${REV:0:8} $(date -u +%FT%TZ)"
git archive --format=tar "$REV" -- . ":(exclude)tests/fixtures/hero/*.npz" ":(exclude)opt/profiler/ttw-*" \
  ":(exclude,glob)**/*.tracy" ":(exclude,glob)**/profile_log_device*.csv" ":(exclude)opt/metal-screenshots" |
  rsh "mkdir -p $T && tar -m -x -C $T && echo $REV > $T/SHA"
# The build half of opt/sync_remote.sh (its REMOTE heredoc), run inside the remote job.
sed -n "/<<'REMOTE'\$/,/^REMOTE\$/p" opt/sync_remote.sh | sed '1d;$d' > tmp/build523.sh
for f in tmp/build523.sh $D/bench523.sh; do
  b=$(basename $f); scp -q "${SSHO[@]}" $f "$H:$T/$b.tmp"; rsh "mv $T/$b.tmp $T/$b"
done
ttp detach --remote $H --dir $J t523b -- bash -c \
  "cd $T && DIR=$T VIEWER_BUILD=0 JOBS=16 bash build523.sh && echo '=== build ok' && bash bench523.sh $O $T ${ROUNDS:-6}"
echo "=== remote job started $(date -u +%FT%TZ)"
for _ in $(seq 1440); do   # up to 2 h; 255 (ssh down) keeps polling
  sleep 5; set +e; ttp detach --check --host $H $J/t523b >/dev/null 2>&1; c=$?; set -e
  [ $c = 0 ] && break
done
mkdir -p $D/out
scp -q "${SSHO[@]}" "$H:$J/t523b.log" $D/out/remote.log || true
scp -q "${SSHO[@]}" "$H:$O/*" $D/out/ || true
grep -E "^===|rc=|^B2B|sync_remote|FAILED|error" $D/out/remote.log | tail -50 || true
exit "$(cat $D/out/bench.rc 2>/dev/null || echo 98)"

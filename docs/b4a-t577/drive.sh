#!/bin/bash
# t577 driver (copy of docs/b4a-t569/drive.sh) (Mac): build <rev> on bh-30 (viewer box; nice/ionice, half the cores) in p150bench/tree577,
# stop the viewer, run <probe script> there (it restarts the viewer at exit), fetch outputs, check localhost:8091.
#   ttp detach t577-<n> -- docs/b4a-t577/drive.sh <rev> docs/b4a-t577/bench.sh <out name>
#   SKIP_BUILD=1 reuses tree577 when it is already built at <rev>.
# Holds resource 'viewer' (exclusive, user exception 2026-10-07): existing viewer reservation only, no
# reserve/extend/release, never the p100 lock or the measurement box. Strict host-key checking.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; PROBE=${2:?probe}; N=${3:?out name}
D=docs/b4a-t577; P=/localdev/smarton/p150bench; T=$P/tree577; O=$P/out577-$N; V=/localdev/smarton/viewer
SSHO=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20); SSH=(ssh "${SSHO[@]}" bh-30)
L=$D/out/$N; mkdir -p $L
"${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; exit 3; }
echo "=== build $(git rev-parse --short "$rev") $(TZ=America/Los_Angeles date +%T%Z)"
if [ -n "${SKIP_BUILD:-}" ]; then  # reuse tree577 as built (its SHA must be <rev>)
  [ "$("${SSH[@]}" cat $T/SHA)" = "$(git rev-parse "$rev")" ] || { echo "tree577 is not $rev: viewer untouched"; exit 3; }
else
SYNC_REMOTE_VIEWER=1 GSTT2_BASE=$V GSTT2_VENV=$V/venv GSTT2_SCENES=$V/scenes REMOTE_TT_METAL_HOME=$V/tt-metal \
  opt/sync_remote.sh bh-30 $T "$rev" > $L/sync.log 2>&1 || { echo "build failed: viewer untouched"; tail -20 $L/sync.log; exit 3; }
tail -2 $L/sync.log
fi
for f in $PROBE $D/vstart577.sh; do b=$(basename $f)
  scp -q "${SSHO[@]}" $f bh-30:$P/.$b.tmp && "${SSH[@]}" "mv $P/.$b.tmp $P/$b" || exit 3; done
echo "=== viewer stop $(TZ=America/Los_Angeles date +%T%Z)"; VIEWER_HOST=bh-30 bash opt/viewer/viewer.sh stop
"${SSH[@]}" "VSTART=$P/vstart577.sh setsid nohup bash $P/$(basename $PROBE) $O $T > $P/probe577-$N.log 2>&1 < /dev/null &"
for _ in $(seq ${WAIT_N:-960}); do "${SSH[@]}" "test -e $O/vstarted" && break; sleep 5; done
"${SSH[@]}" "test -e $O/vstarted" || { echo "PROBE_NOT_DONE (viewer restarts at probe exit; check $O/vstarted)"; exit 4; }
"${SSH[@]}" "cat $P/probe577-$N.log"
for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $V/viewer.log" && break; sleep 3; done
"${SSH[@]}" "grep -E 'SELFTEST|READY|eth\]' $V/viewer.log | tail -3; tr '\0' '\n' < /proc/\$(cat $V/viewer.pid)/environ | grep B4A_FOLD"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)
echo "=== viewer ready $(TZ=America/Los_Angeles date +%T%Z) localhost:8091 -> $code"
scp -q -r "${SSHO[@]}" "bh-30:$O/*" $L/
echo "=== drive end $(TZ=America/Los_Angeles date +%T%Z)"

#!/bin/bash
# t528 (Mac, worktree root): per-stage b2b profile of best-iter-224 on the measurement box.
# One `ttp lock p100` around: ssh preflight, sync + build of <rev>, then short devrun steps:
#   warm b2b (discarded), b2b r1..r3 (headline + host stage timers), counts (N/M/P per view),
#   untraced chunk U0 (raw_md5 reference for the Tracy chunk shape), Tracy plain P0/P10/P20,
#   Tracy GSPLAT_TT_MATCULL_PROF=1 M0, Tracy GSPLAT_TT_K2_PROF=1 GSPLAT_TT_OL_EMIT_PROF=1 X0 (views 0:5).
#   ttp detach t528 -- docs/profile-t224/drive.sh <rev>
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/profile-t224; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t528-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t528-bin/$c
  chmod +x tmp/t528-bin/$c
done
export PATH=$PWD/tmp/t528-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t528 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t528 $sha b2b profile ($H)"
$SSH "$H" "rm -rf $T/tmp/t528; mkdir -p $T/tmp/t528"
pat="^(===|run rc|tracy |device csv|no device|B2B |SUMMARY|VIEW_COUNTS lines|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name bench-args...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t528-$n -- \
    "T=$T bash $T/$D/bench.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
  return 0
}
if [ "${STEPS:-full}" = prof2 ]; then
  # second pass (run 1453): device-profiler chunks of 5 views (11 renders, one profiler read), as t506
  step u0 chunk U0 0 5
  for v in 0 5 10 15 20 25; do step p$v tracy P$v $v $((v + 5)); done
  for v in 0 10 20; do step m$v tracy M$v $v $((v + 5)) GSPLAT_TT_MATCULL_PROF=1; done
  step x0 tracy X0 0 5 GSPLAT_TT_K2_PROF=1 GSPLAT_TT_OL_EMIT_PROF=1
  step r4 b2b 4
else
step warm b2b 0
step r1 b2b 1
step counts counts
step u0 chunk U0 0 10
step p0 tracy P0 0 10
step p10 tracy P10 10 20
step p20 tracy P20 20 30
step r2 b2b 2
step m0 tracy M0 0 10 GSPLAT_TT_MATCULL_PROF=1
step x0 tracy X0 0 5 GSPLAT_TT_K2_PROF=1 GSPLAT_TT_OL_EMIT_PROF=1
step r3 b2b 3
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t528/run-*.log" "$H:$T/tmp/t528/T-*.log" "$H:$T/tmp/t528/dev-*.csv.gz" "$O/" || rc=4
echo "=== drive t528 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

#!/bin/bash
# t595 run 2 (Mac, worktree root): the device-profiler chunks only. Run 1 (drive.sh) failed every
# profiler step: the traced sort program (B4a K2 fold) is 106528 B, over the +32 KB profiler kcfg
# (103424 B). So: probe P0 at GSPLAT_TT_KCFG_EXTRA_KB=36 (107520 B); if that does not give a device
# csv, fall back to GSPLAT_TT_B4A_FOLD=0 (t224 program shape, K2 separate) at the default kcfg.
# (Below: run-1 header, steps changed.)
# One `ttp lock p100` around: ssh preflight, sync + build of <rev>, then short devrun steps:
#   warm b2b (discarded), b2b r1..r3 (headline + host stage timers), counts (N/M/P per view),
#   dump (30-view md5 list, golden 077caaba), untraced chunk U0, device-profiler chunks P0..P25 (5 views),
#   GSPLAT_TT_MATCULL_PROF=1 M0 (and M10, M20 if M0 builds under the profiler).
#   ttp detach t595 -- docs/profile-t227/drive.sh <rev>
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/profile-t227; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t595-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive2.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t595-bin/$c
  chmod +x tmp/t595-bin/$c
done
export PATH=$PWD/tmp/t595-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t595 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t595 $sha b2b profile ($H)"
$SSH "$H" "mkdir -p $T/tmp/t595"
pat="^(===|VIEWS=|run rc|tracy |device csv|no device|B2B |SUMMARY|VIEW_COUNTS lines|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name bench-args...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t595-$n -- \
    "T=$T bash $T/$D/bench.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
  return 0
}
# Run 2 result: KCFG_EXTRA_KB=36 fits the kcfg but static CBs then overflow L1 by 1920 B
# (1574784 > 1572864), and 35 KB is below the 106528 B program. So run 3 goes straight to the
# fallback: GSPLAT_TT_B4A_FOLD=0 (K2 as its own program, t224 shape) at the default profiler kcfg.
# (Run 2 also had a bug: grep "device csv P0" matched "no device csv P0", so the fallback never ran.)
E=GSPLAT_TT_B4A_FOLD=0
step p0 tracy P0 0 5 $E
echo "=== profiler env: $E"
if grep -q "device csv P0 rows" "$O/p0.out"; then
  for v in 5 10 15 20 25; do step p$v tracy P$v $v $((v + 5)) $E; done
  step m0 tracy M0 0 5 $E GSPLAT_TT_MATCULL_PROF=1
  if grep -q "device csv M0 rows" "$O/m0.out"; then
    for v in 10 20; do step m$v tracy M$v $v $((v + 5)) $E GSPLAT_TT_MATCULL_PROF=1; done
  fi
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t595/T-*.log" "$H:$T/tmp/t595/dev-*.csv.gz" "$O/" || rc=4
echo "=== drive2 t595 done rc=$rc env=$E $(TZ=America/Los_Angeles date)"
exit $rc

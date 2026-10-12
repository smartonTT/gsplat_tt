#!/bin/bash
# t551 (B4a step 1 ceiling probe, GSPLAT_TT_K2_EXTRA=<n>, profiling only, default off): one `ttp lock p100`
# around: ssh preflight, sync + build of <rev> on the measurement box, a discarded warm-up round, then
# 3 rotating rounds of b2b for x0 (unset) / x1 / x2 extra segment K2 runs per view, and one latency round.
#   ttp detach t551 -- docs/b4a-step1/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session (~1 min), so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4a-step1; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t551-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t551-bin/$c
  chmod +x tmp/t551-bin/$c
done
export PATH=$PWD/tmp/t551-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t551 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t551 $sha K2 extra-run probe ($H)"
$SSH "$H" "rm -rf $T/tmp/t551; mkdir -p $T/tmp/t551"
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 480 --tag t551-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step warm 0 b2b x0 x1
for r in 1 2 3; do
  case $r in 1) arms="x0 x1 x2" ;; 2) arms="x2 x0 x1" ;; *) arms="x1 x2 x0" ;; esac
  step r$r-b2b $r b2b $arms
  [ $r = 1 ] && step r$r-lat $r lat x0 x1
done
scp -q -o BatchMode=yes "$H:$T/tmp/t551/run-*.log" "$O/" || rc=4
echo "=== drive t551 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

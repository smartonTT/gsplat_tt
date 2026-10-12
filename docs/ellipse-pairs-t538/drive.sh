#!/bin/bash
# t538 (lever B2 cost probe): one `ttp lock p100` around: ssh preflight, sync + build of <rev>
# on the measurement box, an md5 dump sweep of the probe arm, a discarded warm-up round, then
# 3 alternating rounds of b2b + latency for off (GSPLAT_TT_ELLIPSE_PROBE unset, = best-iter-225)
# / probe (GSPLAT_TT_ELLIPSE_PROBE=1: B2's per-pair test runs in K2, pairs unchanged).
#   ttp detach t538 -- docs/ellipse-pairs-t538/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/ellipse-pairs-t538; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t538-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t538-bin/$c
  chmod +x tmp/t538-bin/$c
done
export PATH=$PWD/tmp/t538-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t538 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t538 $sha ellipse probe ($H)"
$SSH "$H" "rm -rf $T/tmp/t538; mkdir -p $T/tmp/t538"
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 300 --tag t538-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step dump 0 dump probe
step warm 0 b2b off probe
for r in 1 2 3; do
  case $r in 2) arms="probe off" ;; *) arms="off probe" ;; esac
  step r$r-b2b $r b2b $arms
  step r$r-lat $r lat $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t538/run-*.log" "$O/" || rc=4
echo "=== drive t538 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

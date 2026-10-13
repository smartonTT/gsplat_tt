#!/bin/bash
# t562 (B4a fold default off under ETH dispatch): check on the p100 measurement box (worker dispatch,
# 11x10) that the gate leaves the iter-226 path unchanged: fold verify (default on), the device
# hero of the default config, then one b2b round off/on. bh-33 is a p100, so the ETH hang itself is not
# reproduced here. Reuses docs/b4a-t554/remote_ab.sh (on = flag unset, off = GSPLAT_TT_B4A_FOLD=0).
#   ttp detach t562 -- docs/b4a-t562/drive.sh <rev>
# Restrictions: only the existing measurement reservation, never the viewer box; ssh/scp shims force
# StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4a-t562; R=docs/b4a-t554; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t562-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t562-bin/$c
  chmod +x tmp/t562-bin/$c
done
export PATH=$PWD/tmp/t562-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t562 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t562 $sha B4a fold ETH gate ($H)"
$SSH "$H" "rm -rf $T/tmp/t554; mkdir -p $T/tmp/t554"
pat="^(===|run rc|\[B4A\]|B4A_VERIFY_FAIL|FOLD_NOT_RUN|FOLD_NOT_OFF|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 300 --tag t562-$n -- \
    "T=$T bash $T/$R/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-lat 0 vlat on
step verify-b2b 0 vb2b on
[ $rc = 0 ] || { echo "=== fold verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t554/run-*.log" "$O/"; exit 8; }
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t562 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 562 "$rev" --locked \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || { echo "=== on-arm shot failed, stop"; exit 5; }
step r1-b2b 1 b2b off on
scp -q -o BatchMode=yes "$H:$T/tmp/t554/run-*.log" "$O/" || rc=4
echo "=== drive t562 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

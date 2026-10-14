#!/bin/bash
# t731: device-profiler zones for train-mode b2b on the p150 measurement box (bh-49).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# docs/t731/remote.sh prof (render/run.py with TT_METAL_DEVICE_PROFILER=1: eth_default picks the
# 36 KB profiler overlay), the device CSV pulled back to docs/t731/out.
#   ttp detach t731-prof -- docs/t731/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t731; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t731-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t731-bin/$c
  chmod +x tmp/t731-bin/$c
done
export PATH=$PWD/tmp/t731-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t731 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t731 $sha profiler zones ($H)"
$SSH "$H" "rm -rf $T/tmp/t731 /localdev/smarton/.cache/ttmc-t731-* $T/tmp/ttm-eth12-cache-t731*; mkdir -p $T/tmp/t731"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t731-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t731-remote.sh.tmp $T/tmp/t731-remote.sh" || exit 3
$DEVRUN --host "$H" --no-verify --timeout 560 --tag t731-prof -- \
  "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=480 bash $T/tmp/t731-remote.sh prof" > "$O/prof.out" 2>&1
rc=$?
grep -E "^(===|run rc|B2B |SUMMARY|HANG|NO_DEVICE_CSV|device csv|\[eth\]|\[DEV\] dispatch|TT_FATAL|TT_THROW|.*overflows)" "$O/prof.out" | cut -c1-240
echo "=== devrun rc=$rc"
scp -q -o BatchMode=yes "$H:$T/tmp/t731/*" "$O/" || rc=${rc/#0/4}
$SSH "$H" "rm -f $T/tmp/t731/prof-device.csv.gz; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t731 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

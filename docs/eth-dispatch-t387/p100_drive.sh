#!/bin/bash
# t387 (from t355 drive.sh): one `ttp lock p100` around everything on the measurement box (p100a):
# sync + build, tt-build stamp, 3 alternating untraced rounds (base = default dispatch vs
# GSPLAT_TT_DISPATCH=auto, which must resolve to worker on the p100a). No screenshot here.
#   ttp detach t387p-all -- docs/eth-dispatch-t387/p100_drive.sh <iter> HEAD     (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t387p}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/eth-dispatch-t387/out-p100; mkdir -p "$O" tmp/ttw-state
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/eth-dispatch-t387/p100_drive.sh" "$it" "$rev" --locked
fi
# Build-ID banner for this tree: worktree-local ttw config (host/root of this run) and state,
# seeded from the last id so the number stays monotonic.
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t387p $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t387p iter-$it $sha ETH dispatch auto vs worker ($H)"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes docs/eth-dispatch-t387/p100_remote_time.sh "$H:$T/tmp/t387p_remote_time.sh" || exit 3
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20' > "$O/board.txt" 2>&1
rc=0
for ra in 1:base,auto 2:auto,base 3:base,auto; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t387p-r$r-$a -- \
      "T=$T bash $T/tmp/t387p_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-$a.out" | grep -E "^(===|run rc|SUMMARY|ALL_|VIEWS|HANG|\[DEV\] dispatch)"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t387p/run-r*.log" "$H:$T/tmp/t387p/md5-r*.txt" "$O/" || rc=4
echo "=== rounds done rc=$rc $(date)"
[ $rc -eq 0 ]

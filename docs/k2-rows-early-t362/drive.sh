#!/bin/bash
# t362: one `ttp lock p100` around everything on the measurement box (yyzo-bh-04):
# sync + build, tt-build stamp, 3 alternating untraced A/B rounds (base = K2 rows read early, behind proj_M
# vs GSPLAT_TT_K2_ROWS_EARLY=0), then the device hero shot (opt/ttw/screenshot.sh, same tree, defaults).
#   ttp detach t362-all -- docs/k2-rows-early-t362/drive.sh 207 HEAD      (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t362}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/k2-rows-early-t362/out; mkdir -p "$O" tmp/ttw-state
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/k2-rows-early-t362/drive.sh" "$it" "$rev" --locked
fi
# Build-ID banner for this tree: worktree-local ttw config (host/root of this run) and state,
# seeded from the last id so the number stays monotonic.
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t362 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t362 iter-$it $sha K2 rows early read ($H)"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes docs/k2-rows-early-t362/remote_time.sh "$H:$T/tmp/t362_remote_time.sh" || exit 3
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20' > "$O/board.txt" 2>&1
rc=0
for ra in 1:base,off,chk:GSPLAT_TT_SORT_ONELAUNCH_CHECK=1 2:off,base 3:base,off; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t362-r$r-$a -- \
      "T=$T bash $T/tmp/t362_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-$a.out" | grep -E "^(===|run rc|SUMMARY|ALL_|VIEWS|HANG)"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t362/run-r*.log" "$H:$T/tmp/t362/md5-r*.txt" "$O/" || rc=4
echo "=== rounds done rc=$rc $(date)"
NO_SYNC=1 T=$T NAME=t362-k2early opt/ttw/screenshot.sh "$it" "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date)"
[ $rc -eq 0 ] && [ $s -eq 0 ]

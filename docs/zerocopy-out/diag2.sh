#!/bin/bash
# t374 diagnostic 2: does the zc blend penalty track the number of pinned buffers?
# around sync + build + 2 rounds of: pin (ring of 1), pinhold (copy mode but
# rotating through 3 pinned buffers, GSPLAT_TT_OUT_PIN_HOLD=2), zc, zcpv (zc with
# per-view stage deltas). Every new pinned slot logs "OUT_RING new pinned slot".
#   ttp detach t374-diag -- docs/zerocopy-out/diag.sh HEAD      (Mac, worktree root)
set -u
cd "$(git rev-parse --show-toplevel)"
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t374}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/zerocopy-out/diag; mkdir -p "$O" tmp/ttw-state
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/zerocopy-out/diag2.sh" "$rev" --locked
fi
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== diag t374 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t374 diag $sha ($H)"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes docs/zerocopy-out/remote_time.sh "$H:$T/tmp/t374_remote_time.sh" || exit 3
A1="zc zc3:GSPLAT_TT_OUT_ZEROCOPY=1,GSPLAT_TT_OUT_ZEROCOPY_SLOTS=3 pinhold3:GSPLAT_TT_OUT_PINNED=1,GSPLAT_TT_OUT_PIN_HOLD=3 pin"
A2="pin pinhold3:GSPLAT_TT_OUT_PINNED=1,GSPLAT_TT_OUT_PIN_HOLD=3 zc3:GSPLAT_TT_OUT_ZEROCOPY=1,GSPLAT_TT_OUT_ZEROCOPY_SLOTS=3 zc"
rc=0
for r in 13 14; do
  [ $r = 13 ] && arms=$A1 || arms=$A2
  $DEVRUN --host "$H" --no-verify --timeout 600 --tag t374-diag-r$r -- \
    "T=$T bash $T/tmp/t374_remote_time.sh $r $arms" 2>&1 | tee "$O/round$r.out" | grep -E "^(===|run rc|SUMMARY|ALL_|VIEWS|HANG)"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t374/run-r1[34]-*.log" "$O/" || rc=4
echo "=== diag done rc=$rc $(date)"
exit $rc

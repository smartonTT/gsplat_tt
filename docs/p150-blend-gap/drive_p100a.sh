#!/bin/bash
# t366: p100a (yyzo-bh-04) Tracy capture of iter-209 code in the built #355 tree (8588f1ac),
# opt/profiler/capture_tracy.sh under devrun, then csvexport -u. One `ttp lock p100` around it.
#   ttp detach t366-p100a -- docs/p150-blend-gap/drive_p100a.sh
set -u
cd "$(git rev-parse --show-toplevel)"
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t355}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
D=docs/p150-blend-gap; O=$D/out; mkdir -p "$O" tmp/ttw-state
if [ "${1:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive_p100a.sh" --locked; fi
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
echo "=== p100a capture $(date -u +%FT%TZ) tree $("${SSH[@]}" "cd $T && git rev-parse --short HEAD")"
$DEVRUN --host "$H" --no-verify --timeout 540 --tag t366-p100a -- \
  "cd $T && GSTT2_REPO=$T GSPLAT_PER_VIEW_STAGES=1 bash opt/profiler/capture_tracy.sh ttw-t366" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
"${SSH[@]}" "cd $T/opt/profiler/ttw-t366 && X=\$(ls /localdev/smarton/tt-metal/build*/tools/profiler/bin/csvexport-release | head -1) && \$X -u render.tracy | gzip > p100a-tracy-u.csv.gz && gzip -c profile_log_device.csv > p100a-dev.csv.gz; ls -la"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/opt/profiler/ttw-t366/p100a-*.csv.gz" "$O/" || rc=4
echo "=== p100a done rc=$rc $(date -u +%FT%TZ)"; exit $rc

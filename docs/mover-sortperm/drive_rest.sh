#!/bin/bash
# t418: the rest of drive.sh after its rounds (the first drive.sh run was cut short),
# reusing the tree it built: the xvpair Tracy control (views 0:10), the fetch of all Tracy
# CSVs/logs, and the device hero screenshot. Was drive.sh: ssh preflight, sync + build of <rev> on the measurement box,
# 3 alternating untraced 30-view rounds of xvpk (default: packed sort_record_ids) vs xvpair
# (GSPLAT_TT_SORT_PACKED=0, the old pair radix), md5 vs golden 906e0435 and xview hits per arm,
# Tracy recapture with GSPLAT_TT_MATCULL_PROF=1 (xvpk views 0:30, xvpair views 0:10 as control),
# the fetch of logs/CSVs, and the device hero screenshot (default config).
#   ttp detach t418 -- docs/mover-sortperm/drive_rest.sh <rev>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; MODE=${MODE:-all}  # all | rounds (skip Tracy and the hero)
export MODE
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t418}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/mover-sortperm; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t418-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive_rest.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t418-bin/$c
  chmod +x tmp/t418-bin/$c
done
export PATH=$PWD/tmp/t418-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t418 $sha on $H:$T $(date)"
$SSH "$H" "cd $T && cut -c1-8 SHA"  # the tree drive.sh built (no re-sync)
pat="^(===|run rc|tracy |STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL)"
rc=0
$DEVRUN --host "$H" --no-verify --timeout 540 --tag t418-Tpair0 -- \
  "T=$T bash $T/$D/remote_tracy.sh 0 pair GSPLAT_TT_SORT_PACKED=0" 2>&1 | tee "$O/tracy-pair-c0.out" | grep -E "$pat"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
scp -q -o BatchMode=yes "$H:$T/tmp/t418/T-*.log" "$H:$T/tmp/t418/*.csv.gz" "$O/" || rc=4
NO_SYNC=1 T=$T NAME=t418 opt/ttw/screenshot.sh 418 "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/t418/hero.png and hero_diff10.png)"
[ $s -eq 0 ] || rc=9
echo "=== drive_rest t418 done rc=$rc $(date)"
exit $rc

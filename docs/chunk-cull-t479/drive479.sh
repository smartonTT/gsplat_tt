#!/bin/bash
# t479 profile driver (Mac, repo root): one `ttp lock p100` around ssh preflight, sync + build of <rev>
# on the measurement box (p100 yyzo-bh-04, the existing reservation) and one Tracy device capture per
# arm (views 0:10): off (defaults) and ro (GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0, Morton gids,
# every tile kept). Extra arms: ARMS="off ro x:ENV=V,ENV=V". Device CSVs land in out/.
#   ttp detach t479p -- docs/chunk-cull-t479/drive479.sh <rev>
# Restrictions: no ird reserve/extend/release, never the viewer box; every ssh/scp forces
# StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t479}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight
D=docs/chunk-cull-t479; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t479-bin
MARK=${TTP_RUN_DIR:-tmp}/drive479.done; rm -f "$MARK"
trap 'rc=$?; echo "drive479 exit $rc"; echo $rc > "$MARK"' EXIT
if [ "${2:-}" != --locked ]; then
  ttp lock p100 -- "$PWD/$D/drive479.sh" "$rev" --locked; exit $?
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t479-bin/$c
  chmod +x tmp/t479-bin/$c
done
export PATH=$PWD/tmp/t479-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t479 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t479 $sha chunk cull profile ($H)"
ssh -o BatchMode=yes "$H" "mkdir -p $T/tmp/t479"
pat="^(===|tracy |STAGES|SUMMARY|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL)"
rc=0
for arm in ${ARMS:-off ro}; do
  case $arm in
    off) name=off; env= ;;
    ro) name=ro; env="GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0" ;;
    *:*) name=${arm%%:*}; env=$(echo "${arm#*:}" | tr , ' ') ;;
  esac
  $DEVRUN --host "$H" --no-verify --timeout 540 --tag t479-T$name -- \
    "T=$T bash $T/$D/remote_tracy.sh 0 $name '$env'" 2>&1 | tee "$O/tracy-$name.out" | grep -E "$pat"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
done
scp -q -o BatchMode=yes "$H:$T/tmp/t479/T-*.log" "$H:$T/tmp/t479/dev-*.csv.gz" "$O/" || rc=4
echo "=== drive t479 done rc=$rc $(date)"
exit $rc

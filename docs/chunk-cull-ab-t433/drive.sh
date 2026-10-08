#!/bin/bash
# t433: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# 3 alternating untraced 30-view rounds of off (defaults) / ro (Morton reorder, every tile kept:
# GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0) / cull (GSPLAT_TT_CHUNK_CULL=1), md5 per arm,
# one Tracy capture of the cull arm (views 0:10) and the device hero screenshot of the cull arm.
# Stage A (ro - off) and stage B (cull - off) are judged from the rounds afterwards (README.md).
#   ttp detach t433 -- docs/chunk-cull-ab-t433/drive.sh <rev>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; MODE=${MODE:-all}  # all | rounds (skip Tracy and the hero)
export MODE
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t433}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/chunk-cull-ab-t433; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t433-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t433-bin/$c
  chmod +x tmp/t433-bin/$c
done
export PATH=$PWD/tmp/t433-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t433 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t433 $sha chunk cull A/B ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
[ "$MODE" = rounds ] && keep="" || keep=" $T/tmp/t433"
$SSH "$H" "rm -rf $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt$keep; mkdir -p $T/tmp/t433"
pat="^(===|run rc|tracy |STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL|CHUNK)"
OFF=off: RO=ro:GSPLAT_TT_CHUNK_CULL=1,GSPLAT_TT_CHUNK_SKIP=0 CULL=cull:GSPLAT_TT_CHUNK_CULL=1,GSPLAT_TT_CHUNK_LOG=1
rc=0
for r in 1 2 3; do
  case $r in 1) arms="$OFF $RO $CULL" ;; 2) arms="$CULL $RO $OFF" ;; 3) arms="$RO $OFF $CULL" ;; esac
  for arm in $arms; do  # one devrun per arm: devrun caps a reservation at 600 s
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t433-r$r-${arm%%:*} -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $arm" 2>&1 | tee "$O/round$r-${arm%%:*}.out" | grep -E "$pat"
  done
done
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/" || rc=4
if [ "$MODE" = rounds ]; then echo "=== drive t433 rounds done rc=$rc $(date)"; exit $rc; fi
$DEVRUN --host "$H" --no-verify --timeout 540 --tag t433-Tcull0 -- \
  "T=$T bash $T/$D/remote_tracy.sh 0 cull GSPLAT_TT_CHUNK_CULL=1" 2>&1 | tee "$O/tracy-cull-c0.out" | grep -E "$pat"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
scp -q -o BatchMode=yes "$H:$T/tmp/t433/T-*.log" "$H:$T/tmp/t433/*.csv.gz" "$O/" || rc=4
NO_SYNC=1 T=$T NAME=t433 opt/ttw/screenshot.sh 433 "$rev" --locked GSPLAT_TT_CHUNK_CULL=1; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/t433/hero.png and hero_diff10.png)"
[ $s -eq 0 ] || rc=9
echo "=== drive t433 done rc=$rc $(date)"
exit $rc

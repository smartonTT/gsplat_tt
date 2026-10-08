#!/bin/bash
# t413: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# one untraced 30-view round at defaults (md5 906e0435 30/30, xview_hits 29: the zones are
# compiled out by default), the 3x10-view Tracy capture with per-job mat zones
# (remote_tracy.sh), the fetch of its CSVs/logs, and the device hero screenshot.
#   ttp detach t413 -- docs/mat-job-zones-t413/drive.sh <rev>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t413}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/mat-job-zones-t413; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t413-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t413-bin/$c
  chmod +x tmp/t413-bin/$c
done
export PATH=$PWD/tmp/t413-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t413 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t413 $sha per-job mat zones ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" "rm -rf $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt $T/tmp/t413; mkdir -p $T/tmp/t413"
pat="^(===|run rc|tracy c|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL)"
rc=0
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t413-def -- \
  "T=$T bash $T/tmp/t379_remote_time.sh 1 xvdef:" 2>&1 | tee "$O/round1-xvdef.out" | grep -E "$pat"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
for a in 0 10 20; do
  $DEVRUN --host "$H" --no-verify --timeout 540 --tag t413-T$a -- \
    "T=$T bash $T/$D/remote_tracy.sh $a" 2>&1 | tee "$O/tracy-c$a.out" | grep -E "$pat"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
done
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r1-xvdef.log" "$H:$T/tmp/t379/md5-r1-xvdef.txt" \
  "$H:$T/tmp/t413/T*.log" "$H:$T/tmp/t413/*.csv.gz" "$O/" || rc=4
NO_SYNC=1 T=$T NAME=t413 opt/ttw/screenshot.sh 413 "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/t413/hero.png and hero_diff10.png)"
[ $s -eq 0 ] || rc=9
echo "=== drive t413 done rc=$rc $(date)"
exit $rc

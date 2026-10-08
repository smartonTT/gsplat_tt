#!/bin/bash
# t417 step 2: one `ttp lock p100` around: ssh preflight, sync + build of <rev>, one untraced
# 30-view round at defaults (md5 906e0435 30/30), the 3x10-view Tracy capture with per-job mat
# zones (remote_tracy.sh, EXTRA_ENV passed through), the fetch of its CSVs/logs, and the
# device hero screenshot of iteration <iter> (skipped when <iter> is -).
#   ttp detach t417-tr -- docs/mat-ramp-t417/drive_tr.sh <rev> <iter|->   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; it=${2:?iter or -}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t413} EXTRA_ENV=${EXTRA_ENV:-}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/mat-ramp-t417; O=${O:-$D/out-tr}; mkdir -p "$O" tmp/ttw-state tmp/t417-bin
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive_tr.sh" "$rev" "$it" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t417-bin/$c
  chmod +x tmp/t417-bin/$c
done
export PATH=$PWD/tmp/t417-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive_tr t417 $sha on $H:$T EXTRA_ENV='$EXTRA_ENV' $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t417 $sha mat ramp Tracy ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" "rm -rf $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt $T/tmp/t417; mkdir -p $T/tmp/t417"
pat="^(===|run rc|tracy c|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL|MAT_RAMP)"
rc=0
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t417-def -- \
  "T=$T bash $T/tmp/t379_remote_time.sh 1 xvdef:" 2>&1 | tee "$O/round1-xvdef.out" | grep -E "$pat"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
for a in 0 10 20; do
  $DEVRUN --host "$H" --no-verify --timeout 540 --tag t417-T$a -- \
    "T=$T EXTRA_ENV='$EXTRA_ENV' bash $T/$D/remote_tracy.sh $a" 2>&1 | tee "$O/tracy-c$a.out" | grep -E "$pat"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
done
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r1-xvdef.log" "$H:$T/tmp/t379/md5-r1-xvdef.txt" \
  "$H:$T/tmp/t417/T*.log" "$H:$T/tmp/t417/*.csv.gz" "$O/" || rc=4
if [ "$it" != - ]; then
  NO_SYNC=1 T=$T NAME=ttw-$it opt/ttw/screenshot.sh "$it" "$rev" --locked; s=$?
  echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/ttw-$it/hero.png and hero_diff10.png)"
  [ $s -eq 0 ] || rc=9
fi
echo "=== drive_tr t417 done rc=$rc $(date)"
exit $rc

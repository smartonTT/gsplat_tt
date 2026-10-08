#!/bin/bash
# t425 (copy of docs/mover-sortperm/drive.sh): one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# the sort unit test (both SORT_RECS_PACKED settings), 3 alternating untraced 30-view rounds of
# xvn7 (default: NoC perm + NoC big gather + packed big-tile sort), xvn3 (GSPLAT_TT_PERM_NOC=3,
# no packed big sort) and xvcpu (=0, RISC copy, pair big sort), md5 vs golden 906e0435 per arm
# (ARM_A/B/C override; run 1 at 8353878c was n3 / n1 / cpu), Tracy capture of the default
# arm views 0:10 with GSPLAT_TT_MATCULL_PROF=1,
# the fetch of logs/CSVs, and the device hero screenshot (default config).
#   ttp detach t425 -- docs/perm-noc-t425/drive.sh <rev>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; MODE=${MODE:-all}  # all | rounds (skip Tracy and the hero)
export MODE
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t425}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/perm-noc-t425; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t425-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t425-bin/$c
  chmod +x tmp/t425-bin/$c
done
export PATH=$PWD/tmp/t425-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t425 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t425 $sha NoC perm ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" "cd $T && for p in 1 0; do c++ -O2 -std=c++17 -DSORT_RECS_PACKED=\$p -Irender/kernels/dataflow tests/unit/test_sort_radix_tile.cpp -o tmp/tsrt\$p && ./tmp/tsrt\$p | tail -1; done" 2>&1 | tee "$O/unit.out" || exit 5
grep -q "fails=0" "$O/unit.out" && [ "$(grep -c 'fails=0' "$O/unit.out")" = 2 ] || { echo "=== unit test failed"; exit 5; }
[ "$MODE" = rounds ] && keep="" || keep=" $T/tmp/t425"
$SSH "$H" "rm -rf $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt$keep; mkdir -p $T/tmp/t425"
pat="^(===|run rc|tracy |STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|device csv|no device|DRAM|Trace|TT_FATAL)"
ARM_A=${ARM_A:-xvn7:} ARM_B=${ARM_B:-xvn3:GSPLAT_TT_PERM_NOC=3} ARM_C=${ARM_C:-xvcpu:GSPLAT_TT_PERM_NOC=0}
N3=$ARM_A N1=$ARM_B CPU=$ARM_C
rc=0
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then arms="$N3 $N1 $CPU"; else arms="$CPU $N1 $N3"; fi
  for arm in $arms; do  # one devrun per arm: devrun caps a reservation at 600 s
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t425-r$r-${arm%%:*} -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $arm" 2>&1 | tee "$O/round$r-${arm%%:*}.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
if [ "$MODE" = rounds ]; then
  scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/" || rc=4
  echo "=== drive t425 rounds done rc=$rc $(date)"; exit $rc
fi
$DEVRUN --host "$H" --no-verify --timeout 540 --tag t425-T0 -- \
  "T=$T bash $T/$D/remote_tracy.sh 0 ${ARM_A%%:*}" 2>&1 | tee "$O/tracy-${ARM_A%%:*}-c0.out" | grep -E "$pat"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=2
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" \
  "$H:$T/tmp/t425/T-*.log" "$H:$T/tmp/t425/*.csv.gz" "$O/" || rc=4
NO_SYNC=1 T=$T NAME=t425 opt/ttw/screenshot.sh 418 "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/t425/hero.png and hero_diff10.png)"
[ $s -eq 0 ] || rc=9
echo "=== drive t425 done rc=$rc $(date)"
exit $rc

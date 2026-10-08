#!/bin/bash
# t417: mat ramp A/B. One `ttp lock p100` around: ssh preflight, sync + build of <rev>, the
# worklist unit test (tests/unit/test_sort_tail_dual_mover.cpp) on the box, then 3 alternating
# rounds x 30 untraced views of GSPLAT_TT_MAT_RAMP=0 (off), =1 and =3 on the same binary,
# md5 906e0435 30/30 and xview_hits 29 per run, and a per-arm stage summary.
#   ttp detach t417-ab -- docs/mat-ramp-t417/drive_ab.sh <rev>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t413}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/mat-ramp-t417; O=${O:-$D/out-ab}; mkdir -p "$O" tmp/ttw-state tmp/t417-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive_ab.sh" "$rev" --locked
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
echo "=== drive_ab t417 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t417 $sha mat ramp A/B ($H)"
rc=0
$SSH "$H" "cd $T && c++ -O2 -std=c++17 -Irender/host tests/unit/test_sort_tail_dual_mover.cpp -o tmp/t417_tail && tmp/t417_tail" \
  2>&1 | tee "$O/unit.txt" || rc=5
grep -q '^PASS' "$O/unit.txt" || rc=5
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" "rm -f $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt"
pat="^(===|run rc|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG|MAT_RAMP)"
A0=xvr0:GSPLAT_TT_MAT_RAMP=0 A1=xvr1:GSPLAT_TT_MAT_RAMP=1 A3=xvr3:GSPLAT_TT_MAT_RAMP=3
for ra in "1:$A0,$A1,$A3" "2:$A3,$A1,$A0" "3:$A1,$A0,$A3"; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t417-r$r-${a%%:*} -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-${a%%:*}.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/" || rc=4
for a in xvr0 xvr1 xvr3; do
  echo "RAMP_LOG $a $(grep -h '^MAT_RAMP' "$O"/run-r?-$a.log | sort | uniq -c | tr '\n' ' ')"
  grep -h "^STAGES " "$O"/run-r[123]-$a.log 2>/dev/null | awk -v a=$a '
    { for (i=1;i<=NF;i++) { split($i,kv,"="); if (kv[1] in want) s[kv[1]]+=kv[2] } m=$0; sub(/.*avg_frame_ms=/,"",m); sub(/ .*/,"",m); v=v" "m; n++ }
    BEGIN { want["avg_frame_ms"]; want["blend"]; want["d2h"]; want["project"]; want["sort"]; want["xview"] }
    END { if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s | blend=%.3f d2h=%.3f project=%.3f sort=%.3f xview=%.3f\n",
          a, s["avg_frame_ms"]/n, n, v, s["blend"]/n, s["d2h"]/n, s["project"]/n, s["sort"]/n, s["xview"]/n }'
done | tee "$O/summary.txt"
echo "=== drive_ab done rc=$rc $(date)"
exit $rc

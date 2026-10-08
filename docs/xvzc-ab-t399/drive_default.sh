#!/bin/bash
# t399 step 2: zero-copy is the default now. One `ttp lock p100` around: ssh preflight,
# sync + build of <rev>, 2 alternating rounds x 30 views of the new default (no env) and
# the opt-out GSPLAT_TT_OUT_ZEROCOPY=0 (= the old xvpin default) on the same binary,
# MATBLEND_PROGRAM fz=1 from frame 0 with zerocopy=1 (default) / zerocopy=0 (opt-out),
# md5 906e0435 30/30 and xview_hits 29, then the iteration's device hero screenshot.
#   ttp detach t399-def -- docs/xvzc-ab-t399/drive_default.sh <rev> <iter>   (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; it=${2:?iter}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t399}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
O=${O:-docs/xvzc-ab-t399/out-default}; mkdir -p "$O" tmp/ttw-state tmp/t399-bin
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/xvzc-ab-t399/drive_default.sh" "$rev" "$it" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t399-bin/$c
  chmod +x tmp/t399-bin/$c
done
export PATH=$PWD/tmp/t399-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive_default t399 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t399 $sha zero-copy default ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" "rm -f $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt"
pat="^(===|run rc|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG)"
rc=0
for ra in 1:xvdef:,xvzc0:GSPLAT_TT_OUT_ZEROCOPY=0 2:xvzc0:GSPLAT_TT_OUT_ZEROCOPY=0,xvdef:; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t399-d$r-${a%%:*} -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-${a%%:*}.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/" || rc=4
for f in "$O"/run-r*.log; do
  case $f in *xvdef*) want=1 ;; *) want=0 ;; esac
  if grep -q "^MATBLEND_PROGRAM fz=1 zerocopy=$want .*frame=0$" "$f" && ! grep -q "^MATBLEND_PROGRAM fz=0" "$f"; then
    echo "FZ_OK $(basename $f) $(grep -m1 '^MATBLEND_PROGRAM' $f)"
  else echo "FZ_BAD $(basename $f) want zerocopy=$want: $(grep '^MATBLEND_PROGRAM' $f | head -3 | tr '\n' ' ')"; fi
done | tee "$O/fz.txt"
grep -q FZ_BAD "$O/fz.txt" && rc=8
for a in xvdef xvzc0; do
  grep -h "^STAGES " "$O"/run-r[12]-$a.log 2>/dev/null | awk -v a=$a '
    { for (i=1;i<=NF;i++) { split($i,kv,"="); if (kv[1] in want) s[kv[1]]+=kv[2] } m=$0; sub(/.*avg_frame_ms=/,"",m); sub(/ .*/,"",m); v=v" "m; n++ }
    BEGIN { want["avg_frame_ms"]; want["blend"]; want["d2h"]; want["project"]; want["sort"]; want["xview"] }
    END { if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s | blend=%.3f d2h=%.3f project=%.3f sort=%.3f xview=%.3f\n",
          a, s["avg_frame_ms"]/n, n, v, s["blend"]/n, s["d2h"]/n, s["project"]/n, s["sort"]/n, s["xview"]/n }'
done | tee "$O/summary.txt"
NO_SYNC=1 T=$T NAME=ttw-$it opt/ttw/screenshot.sh "$it" "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/ttw-$it/hero.png and hero_diff10.png)"
[ $s -eq 0 ] || rc=9
echo "=== drive_default done rc=$rc $(date)"
exit $rc

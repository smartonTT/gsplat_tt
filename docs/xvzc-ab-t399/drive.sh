#!/bin/bash
# t399: zero-copy vs pinned-copy output on top of the cross-view overlap default (#393).
# One `ttp lock p100` around everything on the measurement box: ssh preflight, sync +
# build of <rev>, tt-build stamp, a smoke run of each arm, then 3 alternating untraced
# rounds x 30 views of two arms on the same binary:
#   xvpin = GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_PINNED=1 (the #393 default)
#   xvzc  = GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_ZEROCOPY=1 (#395 fused-gate fix)
# Every run must log MATBLEND_PROGRAM fz=1 from frame 0 (never fz=0), md5 906e0435 30/30
# and xview_hits == 29. Optional screenshot: SHOT=<name> SHOT_ENV="K=V ..." after the rounds.
#   ttp detach t399-ab -- docs/xvzc-ab-t399/drive.sh <rev>      (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box, never disable host-key checking: every ssh/scp here, including
# devrun.sh's and screenshot.sh's, goes through tmp/t399-bin shims that force
# StrictHostKeyChecking=yes (~/.ssh/config sets it to no for bh-*).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t393}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
O=${O:-docs/xvzc-ab-t399/out}; mkdir -p "$O" tmp/ttw-state tmp/t399-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/xvzc-ab-t399/drive.sh" "$rev" --locked
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
echo "=== drive t399 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t399 $sha xvpin vs xvzc ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20; cat /sys/module/tenstorrent/version 2>/dev/null' > "$O/board.txt" 2>&1
$SSH "$H" "rm -f $T/tmp/t379/run-r*.log $T/tmp/t379/md5-r*.txt"
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/"; }
pat="^(===|run rc|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG)"
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t399-r0 -- \
  "T=$T bash $T/tmp/t379_remote_time.sh 0 xvpin xvzc" 2>&1 | tee "$O/round0.out" | grep -E "$pat"
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ "$(grep -c ALL_VIEWS_IDENTICAL "$O/round0.out")" != 2 ] || [ "$(grep -c XVIEW_HITS_OK "$O/round0.out")" != 2 ]; then
  fetch; echo "=== smoke failed $(date)"; exit 5
fi
rc=0
for ra in 1:xvpin,xvzc 2:xvzc,xvpin 3:xvpin,xvzc; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t399-r$r-$a -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-$a.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
fetch || rc=4
echo "=== rounds done rc=$rc $(date)"
# fz=1 from frame 0 in every run, never fz=0
for f in "$O"/run-r*.log; do
  if grep -q "^MATBLEND_PROGRAM fz=1 .*frame=0$" "$f" && ! grep -q "^MATBLEND_PROGRAM fz=0" "$f"; then
    echo "FZ_OK $(basename $f) $(grep -m1 '^MATBLEND_PROGRAM' $f)"
  else echo "FZ_BAD $(basename $f) $(grep '^MATBLEND_PROGRAM' $f | head -3 | tr '\n' ' ')"; rc=8; fi
done | tee "$O/fz.txt"
grep -q FZ_BAD "$O/fz.txt" && rc=8
for a in xvpin xvzc; do
  grep -h "^STAGES " "$O"/run-r[123]-$a.log 2>/dev/null | awk -v a=$a '
    { for (i=1;i<=NF;i++) { split($i,kv,"="); if (kv[1] in want) s[kv[1]]+=kv[2] } m=$0; sub(/.*avg_frame_ms=/,"",m); sub(/ .*/,"",m); v=v" "m; n++ }
    BEGIN { want["avg_frame_ms"]; want["blend"]; want["d2h"]; want["project"]; want["sort"]; want["xview"] }
    END { if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s | blend=%.3f d2h=%.3f project=%.3f sort=%.3f xview=%.3f\n",
          a, s["avg_frame_ms"]/n, n, v, s["blend"]/n, s["d2h"]/n, s["project"]/n, s["sort"]/n, s["xview"]/n }'
done | tee "$O/summary.txt"
if [ -n "${SHOT:-}" ]; then
  NO_SYNC=1 T=$T NAME=$SHOT opt/ttw/screenshot.sh "${SHOT_IT:-0}" "$rev" --locked ${SHOT_ENV:-}; s=$?
  echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/$SHOT/hero.png and hero_diff10.png)"
  [ $s -eq 0 ] || rc=9
fi
exit $rc

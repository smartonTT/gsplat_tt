#!/bin/bash
# t379/t386: cross-view overlap A/B. One `ttp lock p100` around everything on the
# measurement box: ssh preflight, sync + build, tt-build stamp, an xv smoke run (stops on
# failure, md5 mismatch or xview_hits != views-1), 3 rotating untraced rounds of four arms
# (base | xv | xvpin | xvzc, see remote_time.sh), then the device hero shot with the
# overlap on (opt/ttw/screenshot.sh, same tree): hero.png + 10x diff + PSNR vs
# benchmarks/reference_v2/hero.png. LOOK at hero.png and the diff for tile artifacts.
#   ttp detach t379-ab -- docs/xview-overlap-t379/drive.sh <iter> <rev>      (Mac, repo root)
# Env: H (measurement box, yyzo-bh-04), T (remote tree, /localdev/smarton/gstt2-t379),
#      SHOT_ENV (the screenshot arm's env, GSPLAT_TT_XVIEW_OVERLAP=1).
# Restrictions: only the existing measurement reservation (no ird reserve/release), never
# the viewer box, never disable host-key checking.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t379}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
O=docs/xview-overlap-t379/out; mkdir -p "$O" tmp/ttw-state
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/xview-overlap-t379/drive.sh" "$it" "$rev" --locked
fi
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t379 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t379 iter-$it $sha cross-view overlap ($H)"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20; cat /sys/module/tenstorrent/version 2>/dev/null' > "$O/board.txt" 2>&1
fetch() { scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/"; }
pat="^(===|run rc|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG)"
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t379-r0-xv -- \
  "T=$T bash $T/tmp/t379_remote_time.sh 0 xv" 2>&1 | tee "$O/round0-xv.out" | grep -E "$pat"
if [ "${PIPESTATUS[0]}" -ne 0 ] || ! grep -q ALL_VIEWS_IDENTICAL "$O/round0-xv.out" || ! grep -q XVIEW_HITS_OK "$O/round0-xv.out"; then
  fetch; echo "=== smoke (xv) failed $(date)"; exit 5
fi
rc=0
for ra in 1:base,xv,xvpin,xvzc 2:xvzc,base,xv,xvpin 3:xvpin,xvzc,base,xv; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t379-r$r-$a -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-$a.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
fetch || rc=4
echo "=== rounds done rc=$rc $(date)"
# Mean ms/view per arm over the 3 rounds (avg_frame_ms of each STAGES line).
for a in base xv xvpin xvzc; do
  grep -h "^STAGES " "$O"/run-r[123]-$a.log 2>/dev/null | sed -n 's/.*avg_frame_ms=\([0-9.]*\).*/\1/p' |
    awk -v a=$a '{s+=$1; n++; v=v" "$1} END {if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s\n", a, s/n, n, v}'
done | tee "$O/summary.txt"
NO_SYNC=1 T=$T NAME=t379-xview-overlap opt/ttw/screenshot.sh "$it" "$rev" --locked ${SHOT_ENV:-GSPLAT_TT_XVIEW_OVERLAP=1}; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/t379-xview-overlap/hero.png and hero_diff10.png)"
[ $rc -eq 0 ] && [ $s -eq 0 ]

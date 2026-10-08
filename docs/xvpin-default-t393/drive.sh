#!/bin/bash
# t393: xvpin-as-default A/B. One `ttp lock p100` around everything on the measurement
# box: ssh preflight, sync + build of <rev>, tt-build stamp, a smoke run of the new
# default (stops on failure, md5 mismatch or xview_hits != views-1), 3 rotating untraced
# rounds of two arms on the same binary:
#   old   = GSPLAT_TT_XVIEW_OVERLAP=0 GSPLAT_TT_OUT_PINNED=0 (the pre-#393 default)
#   xvdef = no env (the #393 default: cross-view overlap + pinned output)
# then the device hero shot with the default config (opt/ttw/screenshot.sh, same tree):
# hero.png + 10x diff + PSNR vs benchmarks/reference_v2/hero.png. LOOK at both images.
#   ttp detach t393-ab -- docs/xvpin-default-t393/drive.sh <iter> <rev>      (Mac, repo root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box, never disable host-key checking: every ssh/scp here, including
# devrun.sh's and screenshot.sh's, goes through tmp/t393-bin shims that force
# StrictHostKeyChecking=yes (~/.ssh/config sets it to no for bh-*).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t393}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
O=docs/xvpin-default-t393/out; mkdir -p "$O" tmp/ttw-state tmp/t393-bin
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/xvpin-default-t393/drive.sh" "$it" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t393-bin/$c
  chmod +x tmp/t393-bin/$c
done
export PATH=$PWD/tmp/t393-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t393 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t393 iter-$it $sha xvpin default ($H)"
scp -q -o BatchMode=yes docs/xview-overlap-t379/remote_time.sh "$H:$T/tmp/t379_remote_time.sh" || exit 3
$SSH "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20; cat /sys/module/tenstorrent/version 2>/dev/null' > "$O/board.txt" 2>&1
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t379/run-r*.log" "$H:$T/tmp/t379/md5-r*.txt" "$O/"; }
pat="^(===|run rc|STAGES|SUMMARY|ALL_|VIEWS|XVIEW_|HANG)"
OLD=old:GSPLAT_TT_XVIEW_OVERLAP=0,GSPLAT_TT_OUT_PINNED=0 NEW=xvdef:
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t393-r0-xvdef -- \
  "T=$T bash $T/tmp/t379_remote_time.sh 0 $NEW" 2>&1 | tee "$O/round0-xvdef.out" | grep -E "$pat"
if [ "${PIPESTATUS[0]}" -ne 0 ] || ! grep -q ALL_VIEWS_IDENTICAL "$O/round0-xvdef.out" || ! grep -q XVIEW_HITS_OK "$O/round0-xvdef.out"; then
  fetch; echo "=== smoke (xvdef) failed $(date)"; exit 5
fi
rc=0
for ra in 1:old,xvdef 2:xvdef,old 3:old,xvdef; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    [ $a = old ] && spec=$OLD || spec=$NEW
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t393-r$r-$a -- \
      "T=$T bash $T/tmp/t379_remote_time.sh $r $spec" 2>&1 | tee "$O/round$r-$a.out" | grep -E "$pat"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
fetch || rc=4
echo "=== rounds done rc=$rc $(date)"
for a in old xvdef; do
  grep -h "^STAGES " "$O"/run-r[123]-$a.log 2>/dev/null | sed -n 's/.*avg_frame_ms=\([0-9.]*\).*/\1/p' |
    awk -v a=$a '{s+=$1; n++; v=v" "$1} END {if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s\n", a, s/n, n, v}'
done | tee "$O/summary.txt"
NO_SYNC=1 T=$T NAME=ttw-$it opt/ttw/screenshot.sh "$it" "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date) (look at opt/metal-screenshots/ttw-$it/hero.png and hero_diff10.png)"
[ $rc -eq 0 ] && [ $s -eq 0 ]

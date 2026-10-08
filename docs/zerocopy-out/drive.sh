#!/bin/bash
# t374: one `ttp lock p100` around everything on the measurement box (yyzo-bh-04):
# sync + build, tt-build stamp, a zero-copy smoke run (stops on failure or md5 mismatch),
# 3 rotating untraced rounds of three arms (base = default; pin = GSPLAT_TT_OUT_PINNED=1;
# zc = GSPLAT_TT_OUT_ZEROCOPY=1), then the device hero shot with zero-copy on (opt/ttw/screenshot.sh, same tree).
#   ttp detach t374-all -- docs/zerocopy-out/drive.sh 210 HEAD      (Mac, repo root)
# t395 rerun (fused-gate fix): O=docs/zerocopy-out/out-t395 SHOT_NAME=t395-zerocopy-out
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t374}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=${O:-docs/zerocopy-out/out}; mkdir -p "$O" tmp/ttw-state
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/zerocopy-out/drive.sh" "$it" "$rev" --locked
fi
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t374 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t374 iter-$it $sha zero-copy pinned output ($H)"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes docs/zerocopy-out/remote_time.sh "$H:$T/tmp/t374_remote_time.sh" || exit 3
$SSH "$H" 'grep -m1 "model name" /proc/cpuinfo; tt-smi -ls 2>/dev/null | head -20; cat /sys/module/tenstorrent/version 2>/dev/null' > "$O/board.txt" 2>&1
fetch() { scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t374/run-r[0-3]-*.log" "$H:$T/tmp/t374/md5-r[0-3]-*.txt" "$O/"; }
$DEVRUN --host "$H" --no-verify --timeout 420 --tag t374-r0-zc -- \
  "T=$T bash $T/tmp/t374_remote_time.sh 0 zc" 2>&1 | tee "$O/round0-zc.out" | grep -E "^(===|run rc|SUMMARY|MATBLEND_PROGRAM|ALL_|VIEWS|HANG)"
if [ "${PIPESTATUS[0]}" -ne 0 ] || ! grep -q ALL_VIEWS_IDENTICAL "$O/round0-zc.out"; then
  fetch; echo "=== smoke (zc) failed $(date)"; exit 5
fi
rc=0
for ra in 1:base,pin,zc 2:zc,base,pin 3:pin,zc,base; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do
    $DEVRUN --host "$H" --no-verify --timeout 420 --tag t374-r$r-$a -- \
      "T=$T bash $T/tmp/t374_remote_time.sh $r $a" 2>&1 | tee "$O/round$r-$a.out" | grep -E "^(===|run rc|SUMMARY|MATBLEND_PROGRAM|ALL_|VIEWS|HANG)"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
done
fetch || rc=4
echo "=== rounds done rc=$rc $(date)"
NO_SYNC=1 T=$T NAME=${SHOT_NAME:-t374-zerocopy-out} opt/ttw/screenshot.sh "$it" "$rev" --locked GSPLAT_TT_OUT_ZEROCOPY=1; s=$?
echo "=== screenshot rc=$s $(date)"
[ $rc -eq 0 ] && [ $s -eq 0 ]

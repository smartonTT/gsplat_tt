#!/bin/bash
# t319: one `ttp lock p100` around everything on the measurement box (yyzo-bh-04): ssh preflight,
# sync + build, one untraced 30-view run (md5 vs golden 906e0435), the device hero shot
# (opt/ttw/screenshot.sh, same tree), then one Tracy capture at defaults with the fz_* counters
# (retry with GSPLAT_TT_PFWC_WRITER_SPLIT=0 if the profiler build fails) and fill_zones.py.
#   ttp detach t319-all -- docs/fill-zones-t319/drive.sh HEAD      (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t319}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/fill-zones-t319/out; mkdir -p "$O" tmp/ttw-state
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/fill-zones-t319/drive.sh" "$rev" --locked
fi
"${TTP_PROJECT:-/Users/smarton/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight" "$H" > "$O/preflight.txt" 2>&1 || { r=$?; cat "$O/preflight.txt"; exit $((100 + r)); }
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t319 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t319 fill-zones $sha ($H)"
scp -q -o BatchMode=yes docs/fill-zones-t319/remote_time.sh "$H:$T/tmp/t319_remote_time.sh" || exit 3
scp -q -o BatchMode=yes docs/fill-zones-t319/remote_tracy.sh "$H:$T/tmp/t319_remote_tracy.sh" || exit 3
rc=0
for try in 1 2; do
  $DEVRUN --host "$H" --no-verify --timeout 420 --tag t319-r1 -- \
    "T=$T bash $T/tmp/t319_remote_time.sh 1 base" 2>&1 | tee "$O/round1-base.out" | grep -E "^(===|run rc|SUMMARY|ALL_|VIEWS|HANG)"
  r=${PIPESTATUS[0]}
  [ "$r" -eq 0 ] && break
  rc=1; grep -q HANG "$O/round1-base.out" || break
  echo "=== hang, retrying once"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t319/run-r1-base.log" "$H:$T/tmp/t319/md5-r1-base.txt" "$O/" || rc=4
echo "=== timing done rc=$rc $(date)"
NO_SYNC=1 NAME=t319-fill-zones opt/ttw/screenshot.sh 207 "$rev" --locked; s=$?
echo "=== screenshot rc=$s $(date)"
P=opt/profiler/t319-fz; mkdir -p "$P"; ok=
for try in "t319-fz" "t319-fz GSPLAT_TT_PFWC_WRITER_SPLIT=0"; do
  $DEVRUN --host "$H" --no-verify --timeout 560 --tag t319-tracy -- "T=$T bash $T/tmp/t319_remote_tracy.sh $try" \
    2>&1 | tee -a "$O/tracy.out" | grep -E "^===|capture_tracy\]|zones rc|fill rc|TT_FATAL|HANG"
  if ssh -o BatchMode=yes "$H" test -s "$T/$P/fill.txt"; then ok="$try"; break; fi
done
if [ -n "$ok" ]; then
  scp -q -o BatchMode=yes "$H:$T/$P/fill.txt" "$H:$T/$P/percore.csv" "$H:$T/$P/zones.txt" "$H:$T/$P/dev30.csv.gz" "$P/" || rc=4
  echo "env=$ok" > "$P/capture_env.txt"
  echo "=== tracy ok ($ok)"
else
  echo "=== no trace"; rc=2
fi
echo "=== drive done rc=$rc shot=$s $(date)"
[ $rc -eq 0 ] && [ $s -eq 0 ]

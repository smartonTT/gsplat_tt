#!/bin/bash
# t519: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# then (each step logged, a failed step does not stop the later ones except a hang):
#   1) shotB: opt/ttw/screenshot.sh with GSPLAT_TT_MAT_EARLY_CULL=1 (device compile check, 30-view
#      md5 sweep, hero + diff + PSNR vs benchmarks/reference_v2/hero.png)
#   2) MATCULL Tracy b2b capture of A (views 0:10), then of B
#   3) 3 alternating b2b rounds A B / B A / A B (run.py --back-to-back, 30 views)
#   ttp detach t519 -- docs/mat-early-cull-t513/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/mat-early-cull-t513; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t519-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t519-bin/$c
  chmod +x tmp/t519-bin/$c
done
export PATH=$PWD/tmp/t519-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t519 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t519 $sha mat early cull A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t519; mkdir -p $T/tmp/t519"
pat="^(===|run rc|tracy |device csv|no device|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name bench-args...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 560 --tag t519-$n -- \
    "T=$T bash $T/$D/bench.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
  return $s
}
echo "=== shotB $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=t519-B SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 519 "$rev" --locked GSPLAT_TT_MAT_EARLY_CULL=1 \
  > "$O/shotB.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)|^ +[0-9]+ " "$O/shotB.out"; echo "shotB rc=$s"
[ $s = 0 ] || rc=5
step trA tracy A 0 10
step trB tracy B 0 10
step r1 b2b 1 A B
step r2 b2b 2 B A
step r3 b2b 3 A B
scp -q -o BatchMode=yes "$H:$T/tmp/t519/run-*.log" "$H:$T/tmp/t519/T-*.log" "$H:$T/tmp/t519/dev-*.csv.gz" "$O/" || rc=4
echo "=== drive t519 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

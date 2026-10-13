#!/bin/bash
# t606 (copy of docs/pfwc-rec-loop-t599/drive.sh, arm env only): device A/B of the RECFAST +
# VIS_FAST bundle (structural plan 4, spec 1). Arms: base = defaults (best-iter-227),
# new = GSPLAT_TT_PFWC_RECFAST=1 GSPLAT_TT_PFWC_VIS_FAST=1, vis = VIS_FAST=1 only (attribution).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T, a 30-view md5 dump
# for base (must match the golden 077caaba) and new (an off-golden new is recorded, not fatal:
# its dump is kept for the per-view diff), the device hero with both switches on
# (opt/ttw/screenshot.sh: hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a
# discarded warm-up round, 4 b2b rounds in ABBA order (base new, new base, base new, new base),
# then one attribution round r5 (vis new base).
#   ttp detach t606 -- docs/pfwc-bundle-t606/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session, so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/pfwc-bundle-t606; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t606-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t606-bin/$c
  chmod +x tmp/t606-bin/$c
done
export PATH=$PWD/tmp/t606-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t606 $sha base vs RECFAST+VIS_FAST (+vis) on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t606 $sha PFWC_RECFAST+VIS_FAST A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t606 /localdev/smarton/.cache/ttmc-t606base /localdev/smarton/.cache/ttmc-t606new /localdev/smarton/.cache/ttmc-t606vis; mkdir -p $T/tmp/t606"
pat="^(===|run rc|B2B |SUMMARY|MD5|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round mode arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t606-$n -- \
    "T=$T RUN_TIMEOUT=$((to - 30)) bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-base 480 0 dump base
[ $rc = 0 ] || { echo "=== base verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t606/*.*" "$O/"; exit 8; }
step verify-new 480 0 dump new
[ $rc = 0 ] || { echo "=== new arm off golden or failed (rc=$rc): measuring anyway, do not land"; rc=9; }
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t606-bundle SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 599 "$rev" --locked \
  GSPLAT_TT_PFWC_RECFAST=1 GSPLAT_TT_PFWC_VIS_FAST=1 > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || [ $rc != 0 ] || rc=5
step warm 300 0 b2b base new
for r in 1 2 3 4; do
  case $r in 2|4) arms="new base" ;; *) arms="base new" ;; esac
  step r$r-b2b 300 $r b2b $arms
done
step r5-b2b 420 5 b2b vis new base
scp -q -o BatchMode=yes "$H:$T/tmp/t606/*.*" "$O/" || rc=4
echo "=== drive t606 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

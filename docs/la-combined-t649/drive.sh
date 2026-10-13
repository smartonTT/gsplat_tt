#!/bin/bash
# t649 (from docs/la-prefetch-t628/drive.sh): L-A (#639) + L-A' (#643) combined, both default on, bh-33.
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T, the device hero +
# 30-view md5 sweep in the default config (both on; opt/ttw/screenshot.sh, iter 228), warm-up, then
# interleaved b2b rounds of base (both off) / la / sif / both (defaults) arms, and latency (remote.sh).
#   ttp detach t649 -- docs/la-combined-t649/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/la-combined-t649; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t649-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t649-bin/$c
  chmod +x tmp/t649-bin/$c
done
export PATH=$PWD/tmp/t649-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t649 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t649 $sha L-A + L-A' combined ($H)"
$SSH "$H" "rm -rf $T/tmp/t649 /localdev/smarton/.cache/ttmc-t649; mkdir -p $T/tmp/t649"
pat="^(===|run rc|B2B |SUMMARY|LEVER |INERT|BASE_NOT_OFF|\[LA\] |HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t649-$n -- \
    "T=$T RUN_TIMEOUT=200 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-228 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 228 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under the 600 s reservation ceiling (4 arms ~ 80-120 s).
if [ -n "${STEPS:-}" ]; then  # e.g. STEPS="w:0:base htl;r4:4:base g3 g7"
  IFS=';' read -ra ST <<< "$STEPS"
  for x in "${ST[@]}"; do IFS=: read -r n r a <<< "$x"; step "$n" 400 "$r" $a; done
else
step warm 400 0 base both
step r1 400 1 base la sif both
step r2 400 2 both sif la base
step r3 400 3 la both base sif
step lat 400 4 lat-base lat-both
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t649/*.*" "$O/" || rc=4
echo "=== drive t649 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

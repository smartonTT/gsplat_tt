#!/bin/bash
# t629 (from docs/b2b-floor-t614/drive.sh): sizing probes for the t614 unsized ideas, bh-33, FOLD=1.
# Default: no screenshot (sizing only); arms: htl (early-sort/projM HTL split), tsN (pfwc TRISC knee).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T, the device hero +
# 30-view md5 sweep (opt/ttw/screenshot.sh, defaults, uninstrumented), warm-up, then rounds of
# base / htl (host timeline) / host-spin / pfwc-device-spin arms (remote.sh).
#   ttp detach t629 -- docs/size-unsized-t629/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/size-unsized-t629; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t629-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t629-bin/$c
  chmod +x tmp/t629-bin/$c
done
export PATH=$PWD/tmp/t629-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t629 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t629 $sha b2b floor probes ($H)"
$SSH "$H" "rm -rf $T/tmp/t629 /localdev/smarton/.cache/ttmc-t629; mkdir -p $T/tmp/t629"
pat="^(===|run rc|B2B |SUMMARY|HTL |HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t629-$n -- \
    "T=$T RUN_TIMEOUT=200 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-1}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t629 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 614 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under the 600 s reservation ceiling (3 arms ~ 60-90 s measured).
if [ -n "${STEPS:-}" ]; then  # e.g. STEPS="w:0:base htl;r4:4:base g3 g7"
  IFS=';' read -ra ST <<< "$STEPS"
  for x in "${ST[@]}"; do IFS=: read -r n r a <<< "$x"; step "$n" 400 "$r" $a; done
else
step warm 400 0 base htl
step r1a 400 1 htl ts50 ts100
step r1b 400 1 ts200 ts400 base
step r2a 400 2 ts400 ts200 htl
step r2b 400 2 ts100 ts50 base
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t629/*.*" "$O/" || rc=4
echo "=== drive t629 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

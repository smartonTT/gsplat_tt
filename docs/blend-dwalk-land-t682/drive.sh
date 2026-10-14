#!/bin/bash
# t682 (from docs/blend-dwalk-t664/drive.sh): land BLEND_DWALK default on (iter 230), bh-33.
# One `ttp lock p100` around: ssh preflight, sync (opt/sync_remote.sh, one-way, repo-scoped, from this
# worktree) + build of <rev> into $T, the device hero + 30-view md5 sweep at defaults (DWALK on;
# opt/ttw/screenshot.sh, iter 230), warm-up, then 4 rotated b2b rounds of base (GSPLAT_TT_BLEND_DWALK=0 =
# the iter 229 blend body) / new (defaults, DWALK 1) / aa (base again, same cache), latency, JIT-hash check.
#   ttp detach t682 -- docs/blend-dwalk-land-t682/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/blend-dwalk-land-t682; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t682-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t682-bin/$c
  chmod +x tmp/t682-bin/$c
done
export PATH=$PWD/tmp/t682-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t682 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
rsha=$($SSH "$H" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$(git rev-parse "$rev^{commit}")"
[ "$rsha" = "$(git rev-parse "$rev^{commit}")" ] || { echo "=== stamp mismatch"; exit 3; }
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t682 $sha BLEND_DWALK default on, iter 230 ($H)"
$SSH "$H" "rm -rf $T/tmp/t682 /localdev/smarton/.cache/ttmc-t682base /localdev/smarton/.cache/ttmc-t682new; mkdir -p $T/tmp/t682"
pat="^(===|run rc|B2B |SUMMARY|LEVER |INERT|BASE_NOT_OFF|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t682-$n -- \
    "T=$T RUN_TIMEOUT=200 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t682 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 230 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under the 600 s reservation ceiling (4 arms ~ 80-120 s).
if [ -n "${STEPS:-}" ]; then  # e.g. STEPS="w:0:base htl;r4:4:base g3 g7"
  IFS=';' read -ra ST <<< "$STEPS"
  for x in "${ST[@]}"; do IFS=: read -r n r a <<< "$x"; step "$n" 400 "$r" $a; done
else
step warm 400 0 base new
step r1 400 1 base new aa
step r2 400 2 aa new base
step r3 400 3 new aa base
step r4 400 4 base aa new
step lat 400 5 lat-base lat-new
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t682/*.*" "$O/" || rc=4
for a in base new; do
  $SSH "$H" "find /localdev/smarton/.cache/ttmc-t682$a -maxdepth 6 -path '*/kernels/*' -type d" > "$O/jit-$a.txt" || rc=4
done
python3 opt/ab_jit_check.py --arm base="$O/jit-base.txt" --arm new="$O/jit-new.txt" --kernel matblend_compute > "$O/jit-check.out" 2>&1
s=$?; cat "$O/jit-check.out"; [ $s = 0 ] || rc=6
$SSH "$H" "rm -rf /localdev/smarton/.cache/ttmc-t682base /localdev/smarton/.cache/ttmc-t682new" || true
echo "=== drive t682 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

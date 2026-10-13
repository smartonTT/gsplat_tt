#!/bin/bash
# t608 (Mac, worktree root): plan-4 spec 1 step 3, one pfwc program window per arm.
# Device-profiler chunk of views 0:5 (the t595 P0 shape: 11 renders, the last 5 are the measured
# b2b pass) for arms base (A), new (B = RECFAST+VIS_FAST), vis (C), rec (RECFAST only), all at
# GSPLAT_TT_B4A_FOLD=0; then base and new again with GSPLAT_TT_PFWC_STEPCYC=1 (mover wait split).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev>, then one short devrun ssh
# session per capture (each <= 380 s), so an ssh drop loses one step only.
#   ttp detach t608 -- docs/pfwc-bundle-t606/profile/drive_prof.sh <rev>
#   CHUNKS="5 10 15 20 25" ... drive_prof.sh <rev>: the other 25 views, plain captures, all four arms
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never
# the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/pfwc-bundle-t606/profile; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t608-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive_prof.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t608-bin/$c
  chmod +x tmp/t608-bin/$c
done
export PATH=$PWD/tmp/t608-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t608 $sha pfwc window profile on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t608 $sha pfwc window profile ($H)"
C=/localdev/smarton/.cache
$SSH "$H" "rm -rf $T/tmp/t608 $C/ttmc-t608base-prof* $C/ttmc-t608new-prof* $C/ttmc-t608vis-prof* $C/ttmc-t608rec-prof*; mkdir -p $T/tmp/t608"
pat="^(===|tracy |device csv|no device|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name remote_prof args...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t608-$n -- \
    "T=$T TRACY_TIMEOUT=330 bash $T/$D/remote_prof.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && { scp -q -o BatchMode=yes "$H:$T/tmp/t608/*.*" "$O/"; exit 124; }
  return 0
}
if [ -n "${CHUNKS:-}" ]; then  # coverage run: CHUNKS="5 10 15 20 25" (views a:a+5), arm order flips per chunk
  i=0
  for a in $CHUNKS; do
    arms=${ARMS:-base new vis rec}; [ $((i % 2)) = 1 ] && arms=$(echo $arms | tr ' ' '\n' | tail -r | tr '\n' ' ')
    for arm in $arms; do step $arm-$a $arm $a $((a + 5)); done
    i=$((i + 1))
  done
else
step base base 0 5
step new new 0 5
step vis vis 0 5
step rec rec 0 5
step base-sc base 0 5 stepcyc
step new-sc new 0 5 stepcyc
fi
scp -q -o BatchMode=yes "$H:$T/tmp/t608/*.*" "$O/" || rc=4
echo "=== drive t608 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

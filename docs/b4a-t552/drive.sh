#!/bin/bash
# t552 (B4a fold, GSPLAT_TT_B4A_FOLD=1, default off: pfwc(N+1) before mat+blend(N), K2(N+1) on its BRISC,
# view N+1 skips its K2): one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the
# measurement box, a verify step per mode (vlat, vb2b: GSPLAT_TT_B4A_VERIFY=3 must print fold verify OK,
# the drive stops otherwise), the device hero of the on arm (opt/ttw/screenshot.sh: 30-view md5 sweep,
# hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a discarded warm-up round, then 3
# alternating rounds of b2b (+ one latency round) for off (flag unset) / on (GSPLAT_TT_B4A_FOLD=1).
#   ttp detach t552 -- docs/b4a-t552/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session (~1 min), so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4a-t552; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t552-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t552-bin/$c
  chmod +x tmp/t552-bin/$c
done
export PATH=$PWD/tmp/t552-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t552 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t552 $sha B4a fold K2(N+1) in blend(N) ($H)"
$SSH "$H" "rm -rf $T/tmp/t552; mkdir -p $T/tmp/t552"
pat="^(===|run rc|\[B4A\]|B4A_VERIFY_FAIL|FOLD_NOT_RUN|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 300 --tag t552-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-lat 0 vlat on
step verify-b2b 0 vb2b on
[ $rc = 0 ] || { echo "=== fold verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t552/run-*.log" "$O/"; exit 8; }
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t552 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 226 "$rev" --locked GSPLAT_TT_B4A_FOLD=1 \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || { echo "=== on-arm shot failed, stop"; exit 5; }
step warm 0 b2b off on
for r in 1 2 3; do
  case $r in 2) arms="on off" ;; *) arms="off on" ;; esac
  step r$r-b2b $r b2b $arms
  [ $r = 1 ] && step r$r-lat $r lat $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t552/run-*.log" "$O/" || rc=4
echo "=== drive t552 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

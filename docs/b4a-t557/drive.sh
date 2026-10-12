#!/bin/bash
# t557: verify the folded K2 under GSPLAT_TT_K2_DIET=0 (one vlat step, B4A_VERIFY=3). Derived from t554 of #552 (B4a fold, now default on; GSPLAT_TT_B4A_FOLD=0 = off arm): same drive as docs/b4a-t552/drive.sh,
# the on arm is the default config (flag unset) and the hero is shot with the default config. One `ttp lock p100` around: ssh preflight, sync + build of <rev> on the
# measurement box, a verify step per mode (vlat, vb2b: GSPLAT_TT_B4A_VERIFY=3 must print fold verify OK,
# the drive stops otherwise), the device hero of the on arm (opt/ttw/screenshot.sh: 30-view md5 sweep,
# hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a discarded warm-up round, then 3
# alternating rounds of b2b (+ one latency round) for off (flag unset) / on (GSPLAT_TT_B4A_FOLD=1).
#   ttp detach t554 -- docs/b4a-t554/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session (~1 min), so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4a-t554; O=${O:-docs/b4a-t557/out}; mkdir -p "$O" tmp/ttw-state tmp/t557-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/b4a-t557/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t557-bin/$c
  chmod +x tmp/t557-bin/$c
done
export PATH=$PWD/tmp/t557-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t554 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t554 $sha B4a fold K2(N+1) in blend(N) ($H)"
$SSH "$H" "rm -rf $T/tmp/t554; mkdir -p $T/tmp/t554"
pat="^(===|run rc|\[B4A\]|B4A_VERIFY_FAIL|FOLD_NOT_RUN|FOLD_NOT_OFF|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 300 --tag t557-$n -- \
    "GSPLAT_TT_K2_DIET=0 T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-lat 0 vlat on
scp -q -o BatchMode=yes "$H:$T/tmp/t554/run-*.log" "$O/"
echo "=== drive t557 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

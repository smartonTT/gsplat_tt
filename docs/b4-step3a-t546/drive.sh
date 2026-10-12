#!/bin/bash
# t546 (B4' step 3a): one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the
# measurement box, the device hero of the nt1 arm (opt/ttw/screenshot.sh with GSPLAT_TT_OL_TOWN_NT=1:
# 30-view md5 sweep, hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a discarded
# warm-up round, then 3 alternating b2b rounds of nt3 (default) / nt1 (TRISC 2 only town).
#   ttp detach t546 -- docs/b4-step3a-t546/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session (<= 600 s), so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4-step3a-t546; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t546-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t546-bin/$c
  chmod +x tmp/t546-bin/$c
done
export PATH=$PWD/tmp/t546-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t546 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t546 $sha town NT=1 ($H)"
$SSH "$H" "rm -rf $T/tmp/t546; mkdir -p $T/tmp/t546"
pat="^(===|run rc|B2B |B2B_STAGES|\[SORT\]|SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name script args...
  local n=$1 s=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout 590 --tag t546-$n -- \
    "T=$T bash $T/$D/$s $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || { echo "=== step $n rc=$r"; rc=$r; }
  [ "$r" = 124 ] && exit 124
}
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t546 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 226 "$rev" --locked GSPLAT_TT_OL_TOWN_NT=1 \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
step warm remote_ab.sh 0 b2b nt3 nt1
for r in 1 2 3; do
  case $r in 2) arms="nt1 nt3" ;; *) arms="nt3 nt1" ;; esac
  step r$r-b2b remote_ab.sh $r b2b $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t546/*.log" "$O/" || rc=4
echo "=== drive t546 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

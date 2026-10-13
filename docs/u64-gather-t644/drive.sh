#!/bin/bash
# t644 (from docs/exp1-t584/drive.sh): device probe of the pfwc u64 unit gather (arms in remote_ab.sh)
# on the measurement box, one `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T,
# a 30-view md5 dump per arm (base must match the golden), a discarded b2b warm-up per arm, then
# 3 rounds (orders base u64 u64ns c1k / reversed / forward) of b2b + serialised latency per arm.
# No kept iteration, so no hero shot.
#   ttp detach t644 -- docs/u64-gather-t644/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session, so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/u64-gather-t644; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t644-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t644-bin/$c
  chmod +x tmp/t644-bin/$c
done
export PATH=$PWD/tmp/t644-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t644 $sha unit-gather probe on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t644 $sha u64 gather probe ($H)"
$SSH "$H" "rm -rf $T/tmp/t644 /localdev/smarton/.cache/ttmc-t644base /localdev/smarton/.cache/ttmc-t644c1k /localdev/smarton/.cache/ttmc-t644u64 /localdev/smarton/.cache/ttmc-t644u64ns; mkdir -p $T/tmp/t644"
pat="^(===|run rc|B2B |SUMMARY|MD5|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round mode arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t644-$n -- \
    "T=$T RUN_TIMEOUT=$((to - 30)) bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
for a in base u64 u64ns c1k; do step verify-$a 480 0 dump $a; done
grep -q "MD5 golden rc=0 (base)" "$O/verify-base.out" || { echo "=== base verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t644/*" "$O/"; exit 8; }
grep -q "^run rc=0" "$O/verify-u64.out" || { echo "=== u64 arm failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t644/*" "$O/"; exit 9; }
for a in base u64 u64ns c1k; do step warm-$a 240 0 b2b $a; done
for r in 1 2 3; do
  case $r in 2) arms="c1k u64ns u64 base" ;; *) arms="base u64 u64ns c1k" ;; esac
  for a in $arms; do step r$r-b2b-$a 240 $r b2b $a; step r$r-lat-$a 240 $r lat $a; done
done
scp -q -o BatchMode=yes "$H:$T/tmp/t644/*" "$O/" || rc=4
echo "=== drive t644 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

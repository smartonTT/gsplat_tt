#!/bin/bash
# t485: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# a discarded b2b warm-up (JIT), 6 b2b runs A B B A A B (#474) with GSPLAT_B2B_ALL_STAGES=1,
# A = GSPLAT_TT_OUT_ZEROCOPY_SLOTS=4 (old 4-slot ring: per-frame re-pin in keep mode),
# B = unset (run.py sizes the ring to n_views+2), then a B b2b --dump-views md5 + hero run.
#   ttp detach t485 -- docs/out-ring-t485/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t485}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/out-ring-t485; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t485-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t485-bin/$c
  chmod +x tmp/t485-bin/$c
done
export PATH=$PWD/tmp/t485-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t485 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t485 $sha b2b out ring ($H)"
$SSH "$H" "rm -rf $T/tmp/t485; mkdir -p $T/tmp/t485"
pat="^(===|run rc|B2B |SUMMARY|MD5_GOLDEN|HANG|NO_DUMP|Trace|TT_FATAL|TT_THROW)"
rc=0
step() {  # name round mode arm
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t485-$n -- \
    "T=$T bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step warm 0 b2b B
i=0; for a in A B B A A B; do i=$((i+1)); step r$i-$a $i b2b $a; done
step dump 9 dump B
scp -q -o BatchMode=yes "$H:$T/tmp/t485/*" "$O/" || rc=4
echo "=== drive t485 done rc=$rc $(date)"
exit $rc

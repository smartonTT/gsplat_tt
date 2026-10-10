#!/bin/bash
# t483: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# a discarded b2b warm-up (JIT), 3 b2b runs with GSPLAT_B2B_ALL_STAGES=1 (every stage timer per
# pass, incl. the new sort_pre/log/cont_* leaves), then one --dump-views md5 pass.
#   ttp detach t483 -- docs/sort-host-t483/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t483}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/sort-host-t483; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t483-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t483-bin/$c
  chmod +x tmp/t483-bin/$c
done
export PATH=$PWD/tmp/t483-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t483 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t483 $sha sort host timers ($H)"
$SSH "$H" "rm -rf $T/tmp/t483; mkdir -p $T/tmp/t483"
pat="^(===|run rc|B2B |SUMMARY|MD5_GOLDEN|HANG|NO_DUMP|Trace|TT_FATAL|TT_THROW)"
rc=0
step() {  # name round mode
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t483-$n -- \
    "T=$T bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step warm 0 b2b
for r in 1 2 3; do step r$r-b2b $r b2b; done
step dump 9 dump
scp -q -o BatchMode=yes "$H:$T/tmp/t483/*" "$O/" || rc=4
echo "=== drive t483 done rc=$rc $(date)"
exit $rc

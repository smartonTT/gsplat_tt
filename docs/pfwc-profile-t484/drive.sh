#!/bin/bash
# t484: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# then the arms of remote_prof.sh (u z s sr), and copy logs + device CSVs back.
#   ttp detach t484 -- docs/pfwc-profile-t484/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t469}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/pfwc-profile-t484; O=${O:-$D/out}; ARMS=${ARMS:-u z s sr}; mkdir -p "$O" tmp/ttw-state tmp/t484-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t484-bin/$c
  chmod +x tmp/t484-bin/$c
done
export PATH=$PWD/tmp/t484-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t484 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t484 $sha pfwc b2b profile ($H)"
$SSH "$H" "rm -rf $T/tmp/t484; mkdir -p $T/tmp/t484"
pat="^(===|run rc|B2B_STAGES|\[run\] B2B|csv rows|NO_DEVICE_CSV|ZONE_HASH|HANG|Traceback|TT_FATAL|TT_THROW)"
rc=0
step() {  # arm [env...]
  local a=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 560 --tag t484-$a -- \
    "$* T=$T bash $T/$D/remote_prof.sh $a" 2>&1 | tee "$O/$a.out" | grep -E "$pat" | cut -c1-400
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $a rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
  return $s
}
for a in $ARMS; do
  if ! step $a && [ "$a" = s ] && grep -q "TT_FATAL\|TT_THROW" "$O/s.out"; then
    echo "=== retry s with KX=56"; rc=0; step s KX=56 || true
  fi
done
scp -q -o BatchMode=yes "$H:$T/tmp/t484/run-*.log" "$H:$T/tmp/t484/dev-*.csv.gz" "$O/" || rc=4
echo "=== drive t484 done rc=$rc $(date)"
exit $rc

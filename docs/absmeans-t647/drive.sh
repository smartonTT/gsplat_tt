#!/bin/bash
# t647 (from docs/absmeans-t634/drive.sh): cull-side timing gate of the absolute-means lever on
# the measurement box, one `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T,
# a 30-view md5 dump of both arms (records the outputs; both are wrong by design), a discarded
# warm-up round (JIT), then 4 b2b rounds over 2 arms in rotating order: abl (ABS_MEANS_ABL=1)
# and raw (ABS_MEANS_ABL=1 + CULL_ABS_RAW=1: microblock_cull takes rec[4]/rec[5] as written,
# no add_sub_roundtrip). abl - raw is the cull-side saving.
#   ttp detach t647 -- docs/absmeans-t647/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session, so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/absmeans-t647; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t647-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t647-bin/$c
  chmod +x tmp/t647-bin/$c
done
export PATH=$PWD/tmp/t647-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t647 $sha abl/raw on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t647 $sha CULL_ABS_RAW gate ($H)"
$SSH "$H" "rm -rf $T/tmp/t647 /localdev/smarton/.cache/ttmc-t647abl /localdev/smarton/.cache/ttmc-t647raw; mkdir -p $T/tmp/t647"
pat="^(===|run rc|B2B |SUMMARY|MD5|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round mode arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t647-$n -- \
    "T=$T RUN_TIMEOUT=$((to - 30)) bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# devrun refuses a --timeout over the 600 s reservation ceiling: one arm per dump/warm step.
step verify-abl 450 0 dump abl
step verify-raw 450 0 dump raw
[ $rc = 0 ] || { echo "=== dump failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t647/*" "$O/"; exit 8; }
step warm-abl 450 0 b2b abl
step warm-raw 450 0 b2b raw
for r in 1 2 3 4; do
  case $r in 1|3) arms="abl raw" ;; *) arms="raw abl" ;; esac
  step r$r-b2b 300 $r b2b $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t647/*" "$O/" || rc=4
echo "=== drive t647 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

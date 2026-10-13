#!/bin/bash
# t634 (from docs/exp1-t584/drive.sh): timing gate of the absolute-means emit (#632) on the
# measurement box, one `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T,
# a 30-view md5 dump of base (must match the golden), a discarded warm-up round (JIT), then
# 4 b2b rounds over 3 arms in rotating order: base (ABS_MEANS_ABL=0), abl (=1, emit skips both
# sub_int32), ctl (=2, subtracts run but the same absolute means as abl are written). abl and ctl
# give the same wrong records, so abl vs ctl is the emit saving with downstream work held equal.
#   ttp detach t634 -- docs/absmeans-t634/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session, so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/absmeans-t634; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t634-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t634-bin/$c
  chmod +x tmp/t634-bin/$c
done
export PATH=$PWD/tmp/t634-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t634 $sha ABS_MEANS_ABL 0/1/2 on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t634 $sha ABS_MEANS gate ($H)"
$SSH "$H" "rm -rf $T/tmp/t634 /localdev/smarton/.cache/ttmc-t634base /localdev/smarton/.cache/ttmc-t634abl /localdev/smarton/.cache/ttmc-t634ctl; mkdir -p $T/tmp/t634"
pat="^(===|run rc|B2B |SUMMARY|MD5|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round mode arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t634-$n -- \
    "T=$T RUN_TIMEOUT=$((to - 30)) bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-base 480 0 dump base
[ $rc = 0 ] || { echo "=== verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t634/*" "$O/"; exit 8; }
step warm 900 0 b2b base abl ctl
for r in 1 2 3 4; do
  case $r in 1) arms="base abl ctl" ;; 2) arms="ctl base abl" ;; 3) arms="abl ctl base" ;; *) arms="ctl abl base" ;; esac
  step r$r-b2b 360 $r b2b $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t634/*" "$O/" || rc=4
echo "=== drive t634 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

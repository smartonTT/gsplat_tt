#!/bin/bash
# t574 (land the #573 BRISC stack fix without SIB): one `ttp lock p100` around: ssh preflight,
# sync + build of best-iter-226 (90268457) into $TB and of <rev> into $T on the measurement box,
# a 30-view md5 check of both arms on fresh JIT caches (with the static BRISC stack margins of
# every compiled kernel, before = base, after = new), the device hero of <rev>
# (opt/ttw/screenshot.sh: hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a
# discarded warm-up round, then 3 alternating b2b rounds base/new.
#   ttp detach t574 -- docs/land-t574/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session, so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; base=${BASE_REV:-best-iter-226}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} TB=${TB:-/localdev/smarton/gstt2-t574b} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/land-t574; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t574-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t574-bin/$c
  chmod +x tmp/t574-bin/$c
done
export PATH=$PWD/tmp/t574-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev"); bsha=$(git rev-parse --short "$base^{commit}")
echo "=== drive t574 $sha vs $bsha on $H:$T,$TB $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$TB" "$bsha" > "$O/sync-base.log" 2>&1 || { tail -20 "$O/sync-base.log"; exit 3; }
tail -2 "$O/sync-base.log"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t574 $sha BRISC stack fix vs $bsha ($H)"
$SSH "$H" "rm -rf $T/tmp/t574 /localdev/smarton/.cache/ttmc-t574base /localdev/smarton/.cache/ttmc-t574new; mkdir -p $T/tmp/t574"
pat="^(===|run rc|B2B |SUMMARY|STACK |BRISC_STACK_FAIL|brisc_stack_lines|stack scan|MD5|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round mode arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t574-$n -- \
    "T=$T TB=$TB RUN_TIMEOUT=$((to - 30)) bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step verify-new 480 0 dump new
step verify-base 480 0 dump base
[ $rc = 0 ] || { echo "=== verify failed, stop"; scp -q -o BatchMode=yes "$H:$T/tmp/t574/*" "$O/"; exit 8; }
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t574 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 574 "$rev" --locked \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
step warm 300 0 b2b base new
for r in 1 2 3; do
  case $r in 2) arms="new base" ;; *) arms="base new" ;; esac
  step r$r-b2b 300 $r b2b $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t574/*" "$O/" || rc=4
echo "=== drive t574 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

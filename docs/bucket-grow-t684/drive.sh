#!/bin/bash
# t684 grown-bucket A/B (model: docs/training-bench/drive.sh @cf89a0da). One `ttp lock p100` around:
# ssh preflight, sync + build of <rev> into $T, device hero screenshot of the grown arm
# (opt/ttw/screenshot.sh, GSPLAT_TT_TEST_PREGROW_TILE_CAP=1), warm-up, 4 rotated b2b rounds of
# base (32768) vs grow (65472) on benchmarks/cameras_v2.json, JIT cache listings per arm.
#   ttp detach t684 -- docs/bucket-grow-t684/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/bucket-grow-t684; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t684-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t684-bin/$c
  chmod +x tmp/t684-bin/$c
done
export PATH=$PWD/tmp/t684-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t684 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t684 $sha training-view bench ($H)"
$SSH "$H" "rm -rf $T/tmp/t684 $T/tmp/t684-dump /localdev/smarton/.cache/ttmc-t684-*; mkdir -p $T/tmp/t684"
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t684-$n -- \
    "T=$T RUN_TIMEOUT=240 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=t684-grow SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 684 "$rev" --locked GSPLAT_TT_TEST_PREGROW_TILE_CAP=1 > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under the 600 s reservation ceiling (2-3 arms each).
if [ -n "${STEPS:-}" ]; then
  IFS=';' read -ra ST <<< "$STEPS"
  for x in "${ST[@]}"; do IFS=: read -r n r a <<< "$x"; step "$n" 500 "$r" $a; done
else
step w0 400 0 base grow
step r1 400 1 base grow
step r2 400 2 grow base
step r3 400 3 base grow
step r4 400 4 grow base
fi
$SSH "$H" "cd /localdev/smarton/.cache && for a in ttmc-t684-*; do find \$a -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t684/jit-\${a#ttmc-t684-}.txt; done"
scp -q -o BatchMode=yes "$H:$T/tmp/t684/*.*" "$O/" || rc=4
echo "=== drive t684 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

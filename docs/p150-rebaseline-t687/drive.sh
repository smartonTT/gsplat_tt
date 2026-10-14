#!/bin/bash
# t687: re-baseline best-iter-229 on the p150 measurement box (model: docs/training-bench/drive.sh, #667).
# One `ttp lock p100` (the project's measurement lock, also on the p150 box) around: ssh preflight,
# sync + build of <rev> into $T (opt/sync_remote.sh), device hero screenshot + 30-view md5 sweep
# (opt/ttw/screenshot.sh), warm-up, 4 b2b rounds of orbit (cameras_v2, 30 views) and train NN
# (169 views, 1237x822) with the order alternated, 3 plain latency runs of the orbit, JIT listing.
#   ttp detach t687-bench -- docs/p150-rebaseline-t687/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/p150-rebaseline-t687; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t687-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t687-bin/$c
  chmod +x tmp/t687-bin/$c
done
export PATH=$PWD/tmp/t687-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t687 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t687 $sha p150 re-baseline ($H)"
$SSH "$H" "rm -rf $T/tmp/t687 /localdev/smarton/.cache/ttmc-t687-*; mkdir -p $T/tmp/t687"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t687-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t687-remote.sh.tmp $T/tmp/t687-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t687-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t687-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=t687-229-p150 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 229 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under 500 s (the 600 s device ceiling).
step w0 500 0 orbit nn
step r1 500 1 orbit nn
step r2 500 2 nn orbit
step r3 500 3 orbit nn
step r4 500 4 nn orbit
step l1 300 5 lat-orbit
step l2 300 6 lat-orbit
step l3 300 7 lat-orbit
$SSH "$H" "cd /localdev/smarton/.cache && for a in ttmc-t687-*; do find \$a -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t687/jit-\${a#ttmc-t687-}.txt; done"
scp -q -o BatchMode=yes "$H:$T/tmp/t687/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t687 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

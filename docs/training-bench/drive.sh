#!/bin/bash
# t667 training-view bench (model: docs/swapfill-land-t655/drive.sh). One `ttp lock p100` around:
# ssh preflight, sync + build of <rev> into $T, device hero screenshot + 30-view md5 sweep
# (opt/ttw/screenshot.sh), warm-up, 4 b2b rounds of 5 camera arms (order rotated each round),
# per-view stage runs (latency mode, GSPLAT_PER_VIEW_STAGES=1) and 3 dumped random train views.
#   ttp detach t667 -- docs/training-bench/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=${MESH_DEVICE:-P100}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/training-bench; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t667-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t667-bin/$c
  chmod +x tmp/t667-bin/$c
done
export PATH=$PWD/tmp/t667-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t667 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t667 $sha training-view bench ($H)"
$SSH "$H" "rm -rf $T/tmp/t667 $T/tmp/t667-dump /localdev/smarton/.cache/ttmc-t667-*; mkdir -p $T/tmp/t667"
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t667-$n -- \
    "T=$T RUN_TIMEOUT=240 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=t667-229 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 229 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under the 600 s reservation ceiling (2-3 arms each).
if [ -n "${STEPS:-}" ]; then
  IFS=';' read -ra ST <<< "$STEPS"
  for x in "${ST[@]}"; do IFS=: read -r n r a <<< "$x"; step "$n" 500 "$r" $a; done
else
step w0a 500 0 orbit orb1237 test
step w0b 500 0 shuf nn
step r1a 500 1 orbit shuf orb1237
step r1b 500 1 nn test
step r2a 500 2 nn test orbit
step r2b 500 2 shuf orb1237
step r3a 500 3 orb1237 nn shuf
step r3b 500 3 test orbit
step r4a 500 4 test orb1237 nn
step r4b 500 4 orbit shuf
step pv  500 5 pv-orbit pv-shuf pv-nn pv-test
step dump 300 6 dump
fi
$SSH "$H" "cd /localdev/smarton/.cache && for a in ttmc-t667-*; do find \$a -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t667/jit-\${a#ttmc-t667-}.txt; done; rm -rf $T/tmp/t667-dump"
scp -q -o BatchMode=yes "$H:$T/tmp/t667/*.*" "$O/" || rc=4
mkdir -p "$O/dump"; scp -q -o BatchMode=yes "$H:$T/tmp/t667/dump/*.png" "$O/dump/" || rc=4
echo "=== drive t667 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

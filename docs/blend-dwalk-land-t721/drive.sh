#!/bin/bash
# t721 (continues #682): land BLEND_DWALK default on, re-measured on the p150 box (bh-49 board 0, p150a).
# One `ttp lock p100` (the measurement lock, now the p150 box) around: ssh preflight, sync + build of
# best-iter-229 (= best-iter-230 code) into $T2 and of <rev> into $T (opt/sync_remote.sh, one-way,
# repo-scoped, from this worktree), device hero + 30-view md5 sweep of <rev> (opt/ttw/screenshot.sh,
# iter $N), warm-up, 4 rotated rounds of orbit b2b b230/new/base/aa + train-NN b2b nb230/nnew,
# orbit latency, JIT listings + opt/ab_jit_check.py, then the $T2 tree and arm caches are removed.
#   ttp detach t721 -- docs/blend-dwalk-land-t721/drive.sh <rev> <N>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev} N=${2:?iter}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} T2=${T2:-/localdev/smarton/gstt2-b230} MESH_DEVICE=${MESH_DEVICE:-P150}
B230=${B230:-best-iter-229}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/blend-dwalk-land-t721; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t721-bin
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" "$N" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t721-bin/$c
  chmod +x tmp/t721-bin/$c
done
export PATH=$PWD/tmp/t721-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t721 $sha (b230=$B230) on $H:$T,$T2 $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
for x in "$T2:$B230:sync-b230" "$T:$rev:sync"; do
  IFS=: read -r t v lg <<< "$x"
  opt/sync_remote.sh "$H" "$t" "$v" > "$O/$lg.log" 2>&1 || { echo "=== $lg failed"; tail -20 "$O/$lg.log"; exit 3; }
  tail -2 "$O/$lg.log"
  rsha=$($SSH "$H" "sed -n 's/^sha=//p' $t/SYNC_STAMP")
  echo "STAMP $t remote=$rsha local=$(git rev-parse "$v^{commit}")"
  [ "$rsha" = "$(git rev-parse "$v^{commit}")" ] || { echo "=== stamp mismatch $t"; exit 3; }
done
# best-iter-229 predates the #667 camera files: ship them next to it (as #687 did).
scp -q -o BatchMode=yes benchmarks/cameras_train_nn.json "$H:$T2/benchmarks/" || exit 3
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t721 $sha BLEND_DWALK default on, iter $N ($H)"
$SSH "$H" "rm -rf $T/tmp/t721 /localdev/smarton/.cache/ttmc-t721-*; mkdir -p $T/tmp/t721"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t721-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t721-remote.sh.tmp $T/tmp/t721-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|LEVER |INERT|BASE_NOT_OFF|B230_HAS|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t721-$n -- \
    "T=$T T2=$T2 MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t721-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-220
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped (SKIP_SHOT=1)"; else
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-$N SHOT_TIMEOUT=500 opt/ttw/screenshot.sh "$N" "$rev" --locked > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
fi
# Each devrun step stays under 500 s (the 600 s device ceiling).
step w0 500 0 b230 new base
step w1 500 0 nb230 nnew
step r1o 500 1 b230 new base aa
step r1n 500 1 nb230 nnew
step r2o 500 2 aa base new b230
step r2n 500 2 nnew nb230
step r3o 500 3 new aa b230 base
step r3n 500 3 nb230 nnew
step r4o 500 4 base b230 aa new
step r4n 500 4 nnew nb230
step lat 500 5 lb230 lnew lb230 lnew
for a in b230 new base; do
  $SSH "$H" "find /localdev/smarton/.cache/ttmc-t721-$a -maxdepth 8 -path '*/kernels/*' -type d" > "$O/jit-$a.txt" || rc=4
done
scp -q -o BatchMode=yes "$H:$T/tmp/t721/*.*" "$O/" || rc=4
for p in "b230 new" "base new"; do
  set -- $p
  echo "--- ab_jit_check $1 vs $2" >> "$O/jit-check.out"
  python3 opt/ab_jit_check.py --arm $1="$O/jit-$1.txt" --arm $2="$O/jit-$2.txt" --kernel matblend_compute >> "$O/jit-check.out" 2>&1
  s=$?; [ $s = 0 ] || rc=6
done
cat "$O/jit-check.out"
$SSH "$H" "rm -rf $T2 /localdev/smarton/.cache/ttmc-t721-*; df -h /localdev/smarton | tail -1" || true
echo "=== drive t721 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

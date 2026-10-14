#!/bin/bash
# t691 (D1): N x resolution sweep of best-iter-229's forward on the 169 train views, p150 box.
# One `ttp lock p100` (the measurement lock, also used for the p150 box, #687) around: ssh preflight,
# sync + build of best-iter-229 (opt/sync_remote.sh), the importance/pair-count prep on the box's CPU
# (d1prep.py), the proxies (makeproxy.py), the device hero screenshot, then b2b (2 rounds) and
# latency (1 round, per-stage split) of 5 N x 3 resolutions.
#   ttp detach t691-d1 -- docs/training-plan/d1-scaling/drive.sh best-iter-229   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes; only files
# from this worktree are copied to the box.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/training-plan/d1-scaling; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t691-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=30 "$@"\n' $c > tmp/t691-bin/$c
  chmod +x tmp/t691-bin/$c
done
export PATH=$PWD/tmp/t691-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev^{commit}")
echo "=== drive t691 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
# best-iter-229 predates the #667 camera files; the D1 scripts come from this worktree.
scp -q -o BatchMode=yes benchmarks/cameras_train_nn.json benchmarks/cameras_train_shuffled.json \
  benchmarks/cameras_test.json benchmarks/cameras_orbit_1237.json "$H:$T/benchmarks/" || exit 3
$SSH "$H" "mkdir -p $T/$D $T/tmp/d1 $T/tmp/t691" || exit 3
for f in d1prep.py blendscore.c makeproxy.py remote.sh; do
  scp -q -o BatchMode=yes "$D/$f" "$H:$T/$D/$f.tmp" && $SSH "$H" "mv $T/$D/$f.tmp $T/$D/$f" || exit 3
done
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t691 $sha D1 sweep ($H)"
if $SSH "$H" "test -s $T/tmp/d1/rank.npy"; then echo "=== prep: reusing $T/tmp/d1/rank.npy"; else
  echo "=== prep $(TZ=America/Los_Angeles date +%T)"
  $SSH "$H" "cd $T && timeout 4000 nice -n 5 .venv/bin/python $D/d1prep.py --ply scenes/bicycle.ply \
    --cams benchmarks/cameras_train_nn.json --out tmp/d1 --workers ${PREP_WORKERS:-5} --threads 2 > tmp/t691/prep.log 2>&1"
  s=$?; $SSH "$H" "tail -3 $T/tmp/t691/prep.log"; [ $s = 0 ] || { echo "=== prep rc=$s"; exit 6; }
fi
scp -q -o BatchMode=yes "$H:$T/tmp/d1/counts.npz" "$H:$T/tmp/d1/prep.json" "$H:$T/tmp/d1/rank.npy" "$O/" || exit 6
if ! $SSH "$H" "test -s $T/tmp/d1/proxies.json"; then
  $SSH "$H" "cd $T && .venv/bin/python $D/makeproxy.py --rank tmp/d1/rank.npy --ply scenes/bicycle.ply \
    --cams benchmarks/cameras_train_nn.json --out tmp/d1 --ns 3000000,1500000,1000000,500000" || { echo "=== makeproxy failed"; exit 7; }
fi
pat="^(===|run rc|B2B |SUMMARY|STAGES|HANG|Trace|TT_FATAL|TT_THROW|error:|missing)"
rc=0
step() {  # name timeout round mode cfgs...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t691-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/$D/remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-220
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shot skipped"; else
  echo "=== shot $(TZ=America/Los_Angeles date +%T)"
  H=$H T=$T NO_SYNC=1 NAME=t691-229-p150 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 229 "$rev" --locked > "$O/shot.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"; [ $s = 0 ] || rc=5
fi
NT="full 3M 1p5M 1M 0p5M"
for nt in $NT; do step r1-$nt 560 1 b2b ${nt}_s1 ${nt}_s2 ${nt}_s4; done
for nt in $NT; do step l1-$nt 560 1 lat ${nt}_s1 ${nt}_s2 ${nt}_s4; done
for nt in $(echo $NT | tr ' ' '\n' | tail -r 2>/dev/null || echo $NT | tr ' ' '\n' | tac); do
  step r2-$nt 560 2 b2b ${nt}_s4 ${nt}_s2 ${nt}_s1; done
$SSH "$H" "find /localdev/smarton/.cache/ttmc-t691 -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t691/jit-t691.txt; \
  find $T/tmp/ttm-eth12-cache -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t691/jit-eth12.txt 2>/dev/null; wc -l $T/tmp/t691/jit-*.txt"
mkdir -p "$O/logs"; scp -q -o BatchMode=yes "$H:$T/tmp/t691/*.*" "$O/logs/" || rc=4
# keep the box lean: the proxies and the scene cache are regenerable (rank.npy stays, 25 MB)
$SSH "$H" "rm -f $T/tmp/d1/bicycle_*.ply $T/tmp/d1/scene.npz; du -sxh $T/tmp/d1; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t691 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

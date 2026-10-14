#!/bin/bash
# t722 (B1'): md5 + b2b A/B of GSPLAT_TT_B1 = 0 / 1 / 2 on the p150 measurement box (bh-49).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> (opt/sync_remote.sh), the D1
# proxies (rebuilt only if missing), then
#   md5:  BLEND_T_PERIOD=0, each arm: 1M_s4 3M_s2 full_s1 (raw_md5 must match arm 0)
#   b2b:  default period, each arm: s4 {0p5M,1M,1p5M,3M}, s2 {1M,3M,full}, full_s1 (arms interleaved)
#   hero: default period, full scene, cameras_v2 hero view (hero_clean.png) per arm
# ARMS (default "0 1 2"; r = B1 full route mode, #754), ROUNDS (b2b grid repeats, default 1),
# UNIT=1 builds and runs tests/unit/test_subtile_{split,unit,route}.cpp on the box first.
#   ttp detach t722-ab -- docs/training-plan/t719-b1-subtile/t722/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes; only files
# from this worktree are copied to the box.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/training-plan/t719-b1-subtile/t722; P=docs/training-plan/d1-scaling
O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t722-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=30 "$@"\n' $c > tmp/t722-bin/$c
  chmod +x tmp/t722-bin/$c
done
export PATH=$PWD/tmp/t722-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev^{commit}")
echo "=== drive t722 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
scp -q -o BatchMode=yes benchmarks/cameras_train_nn.json "$H:$T/benchmarks/" || exit 3
$SSH "$H" "mkdir -p $T/$D $T/$P $T/tmp/d1 $T/tmp/t722" || exit 3
for f in $P/makeproxy.py $D/remote.sh; do
  scp -q -o BatchMode=yes "$f" "$H:$T/$f.tmp" && $SSH "$H" "mv $T/$f.tmp $T/$f" || exit 3
done
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t722 $sha B1 A/B ($H)"
if ! $SSH "$H" "test -s $T/tmp/d1/bicycle_3M.ply && test -s $T/tmp/d1/cams_3M_s4.json"; then
  echo "=== proxies $(TZ=America/Los_Angeles date +%T)"
  $SSH "$H" "test -s $T/tmp/d1/rank.npy && cd $T && .venv/bin/python $P/makeproxy.py --rank tmp/d1/rank.npy \
    --ply scenes/bicycle.ply --cams benchmarks/cameras_train_nn.json --out tmp/d1 --ns 3000000,1500000,1000000,500000" \
    || { echo "=== makeproxy failed"; exit 7; }
fi
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:|missing)|raw_md5|overflow|re-plan"
rc=0
step() {  # name timeout B1 PERIOD cfgs...
  local n=$1 to=$2 b=$3 p=$4; shift 4
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t722-$n -- \
    "T=$T B1=$b PERIOD=$p ROUND=${ROUND:-} MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/$D/remote.sh $n $*" 2>&1 \
    | tee "$O/$n-b$b-p$p${ROUND:+-r$ROUND}-$1.out" | grep -E "$pat" | cut -c1-260
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n b$b rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# devrun refuses timeouts over 600 s: one config per step, 400 s each (a run is RUN_TIMEOUT=240).
$SSH "$H" "rm -f $T/tmp/t722/*.log $T/tmp/t722/*.png $T/tmp/t722/*.txt"  # only this run's outputs are pulled
# MD5_CFGS / GRID_CFGS / HERO=0 select a subset (a rerun of only the failed gates).
A=${ARMS:-0 1 2}
if [ "${UNIT:-0}" = 1 ]; then
  for t in split unit route; do
    $SSH "$H" "cd $T && CXXFLAGS='-Irender/host -Irender/kernels/dataflow' bash tests/unit/run_cpp.sh tests/unit/test_subtile_$t.cpp" \
      > "$O/unit-$t.out" 2>&1; s=$?; tail -1 "$O/unit-$t.out"; [ $s = 0 ] || { echo "=== unit $t rc=$s"; rc=$s; }
  done
fi
for c in ${MD5_CFGS-1M_s4 3M_s2 full_s1}; do for b in $A; do step md5 400 $b 0 $c; done; done
for ROUND in $(seq 1 "${ROUNDS:-1}"); do
  [ "${ROUNDS:-1}" = 1 ] && ROUND=
  for c in ${GRID_CFGS-1M_s4 0p5M_s4 1p5M_s4 3M_s4 1M_s2 3M_s2 full_s2 full_s1}; do
    for b in $A; do step grid 400 $b default $c; done
  done
done
ROUND=
[ "${HERO:-1}" = 0 ] || for b in $A; do step hero 400 $b default hero; done
$SSH "$H" "cd /localdev/smarton/.cache && for d in ttmc-t722-b*; do find \$d -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t722/jit-\$d.txt; done"
mkdir -p "$O/logs"; scp -q -o BatchMode=yes "$H:$T/tmp/t722/*.*" "$O/logs/" || rc=4
$SSH "$H" "rm -f $T/tmp/d1/bicycle_*.ply $T/tmp/d1/scene.npz; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t722 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc

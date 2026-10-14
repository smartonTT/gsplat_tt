#!/bin/bash
# t690: device training-loss bench (render/bench/loss_bench) in ONE `ttp lock p100`:
#   ttp detach t690 -- docs/t690/drive.sh <rev>          (Mac, worktree root; H=<host>)
# ssh preflight, opt/sync_remote.sh <rev> (one-way, repo-scoped), build render/bench, prep inputs
# on the box (numpy, no dataset files copied), run loss_bench on each case, pull tmp/t690 back,
# check against gsplat/train_loss.py on the Mac. Existing measurement reservation only (no ird
# reserve/extend/release), never the viewer box bh-30, StrictHostKeyChecking=yes everywhere.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-sync} MESH=${MESH:-P100} ITERS=${ITERS:-50} MODE=${MODE:-l1}
D=docs/t690; O=$D/out-$H
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t690 $rev=$sha to $H:$T mesh=$MESH $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
timeout 900 "${SSH[@]}" "cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=$MESH TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t690 PYTHONDONTWRITEBYTECODE=1 && \
  CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) && \
  { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/t690cfg.log 2>&1 && \
    nice -n 19 cmake --build render/bench/build --target loss_bench -j 8 > tmp/t690build.log 2>&1 || { tail -n 40 tmp/t690cfg.log tmp/t690build.log; exit 5; }; } && \
  rm -rf tmp/t690 && source .venv/bin/activate && python3 opt/t690/loss_check.py prep tmp/t690 && \
  for c in dump64 syn; do mkdir -p tmp/t690/\$c/out; timeout 300 render/bench/build/loss_bench tmp/t690/\$c tmp/t690/\$c/out $ITERS > tmp/t690/\$c/run.log 2>&1; \
    echo \"run \$c rc=\$?\"; grep -E '^\[LOSS\]|TT_THROW|TT_FATAL|error' tmp/t690/\$c/run.log | head -5; tail -3 tmp/t690/\$c/run.log; done && \
  rm -f tmp/t690/*/image.f32 tmp/t690/*/target.f32"
echo "remote rc=$?"
opt/sync_remote.sh --pull tmp/t690 "$O/remote" "$H" "$T"; echo "pull rc=$?"
python3 opt/t690/loss_check.py check "$O/remote" --mode "$MODE"; echo "check rc=$?"
rm -f "$O"/remote/*/out/grad.f32
echo "=== done $(TZ=America/Los_Angeles date)"

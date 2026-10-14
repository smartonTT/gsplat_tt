#!/bin/bash
# #750 step 2: the dump-free device training step chain (docs/train-step/loop_box.sh) on the
# oracle crop dumps, all in ONE `ttp lock p100`.
#   ttp detach t750loop -- docs/train-step/drive_loop.sh <rev>      (Mac, worktree root; <rev> committed)
# Steps: ssh preflight (@p100), opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build train_step + blend_fwd_bench (render/train) and loss_bench (render/bench), loop_box.sh per
# dump, pull tmp/t750 to docs/train-step/out/loop.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
[ -n "$H" ] || { echo "=== no p100 host: set H= or device.hosts.p100 in $TP/harness/project.json"; exit 2; }
T=${T:-/localdev/smarton/gstt2-sync}
D=docs/train-step; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive_loop.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t750 loop $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync-loop.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync-loop.log"
[ $rc = 0 ] || { tail -30 "$O/sync-loop.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P150 PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t750 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
echo "=== card"; timeout 60 "${SSH[@]}" "$ENV && tt-smi -ls 2>/dev/null | grep -i -m1 'p150\|p100\|blackhole'"
echo "=== build $(date +%T)"
timeout 1200 "${SSH[@]}" "$ENV && mkdir -p tmp/t750 && CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) && \
  (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8 --target train_step blend_fwd_bench \
  && cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX \
  && nice -n 19 cmake --build render/bench/build -j 8 --target loss_bench) > tmp/t750/build.log 2>&1; rc=\$?; tail -15 tmp/t750/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
if [ $brc = 0 ]; then
  for n in ${RUNS:-sh0 sh3}; do
    echo "=== loop $n $(date +%T)"
    timeout 1800 "${SSH[@]}" "$ENV && bash docs/train-step/loop_box.sh $n benchmarks/train_oracle/t0_${n}_64x64_n50.npz ${REPS:-10}"
    echo "loop $n rc=$?"
  done
fi
opt/sync_remote.sh --pull tmp/t750 "$O/loop" "$H" "$T"; echo "pull rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

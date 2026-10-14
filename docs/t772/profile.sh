#!/bin/bash
# t772: pair-setup sub-step profile of the T1 blend backward on p150 bh-49, all in ONE `ttp lock p100`.
#   ttp detach t772prof -- docs/t772/profile.sh <rev>      (Mac, worktree root; <rev> committed)
# Hero dump (#728's, tmp/t772in/hero.npz) at BWD_PRESTAGE=2 with BWD_STATS=1 (issue-side stage and
# sub-step counters) and BWD_STATS=2 (Tensix idle wait before every lap: execution per stage), own
# TT_METAL_CACHE per mode; stats.json per mode is pulled to $O/remote.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never bh-30,
# StrictHostKeyChecking=yes on every ssh, sync only through the repo-scoped one-way opt/sync_remote.sh
# inside this lock.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} REPS=${REPS:-4} O=${O:-docs/t772/out-prof}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" "$PWD/docs/t772/profile.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t772 profile $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_PRESTAGE=2 \
  && source .venv/bin/activate"
C=/localdev/smarton/.cache/ttmc-t772
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t772 $C-st1 $C-st2 && mkdir -p tmp/t772 tmp/t772in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t772/build.log 2>&1; rc=\$?; tail -5 tmp/t772/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
echo "=== push hero $(date +%T)"
timeout 900 "${SSH[@]}" "cat > $T/tmp/t772in/hero.npz.tmp && mv $T/tmp/t772in/hero.npz.tmp $T/tmp/t772in/hero.npz" < tmp/t772in/hero.npz
echo "push rc=$?"
timeout 900 "${SSH[@]}" "$ENV && md5sum tmp/t772in/hero.npz && python3 -m gsplat.blend_backward_dev prep tmp/t772in/hero.npz tmp/t772in/hero > /dev/null"; echo "prep rc=$?"
for st in 1 2; do
  echo "=== hero BWD_STATS=$st $(date +%T)"
  timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-st$st BWD_STATS=$st && d=tmp/t772/st$st && mkdir -p \$d \
    && timeout 240 render/train/build/blend_bwd_bench tmp/t772in/hero 0 $REPS \$d > \$d/run.log 2>&1; rc=\$?; \
    tail -2 \$d/run.log; python3 -c \"import numpy as np; print('slots9 md5', __import__('hashlib').md5(np.fromfile('\$d/slots.f32', np.uint32).reshape(-1, 16)[:, :9].tobytes()).hexdigest())\"; rm -f \$d/slots.f32; \
    if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > \$d/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  echo "run st$st rc=$?"
done
timeout 60 "${SSH[@]}" "rm -rf $T/tmp/t772in"; echo "rm in rc=$?"
opt/sync_remote.sh --pull tmp/t772 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t772"; echo "rm t772 rc=$?"
python3 docs/t772/stats.py "$O"/remote/st1/stats.json "$O"/remote/st2/stats.json
echo "=== done $(TZ=America/Los_Angeles date)"

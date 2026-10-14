#!/bin/bash
# t713 (T2 segment matmul, from t697): device param pass (render/train/kernels/t2_*.cpp) vs t2_step, all in ONE `ttp lock p100`.
#   ttp detach t713 -- docs/t713/drive.sh <rev>      (Mac, worktree root; <rev> must be committed)
# Steps: ssh preflight, opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree), build
# render/train (t2_bench), check run on the T0 dumps (prep-dump, grads out), a synthetic timing run
# (prep-synth --no-ref, N=$SYN_N), pull tmp/t713 back, then `t2_dev check` here.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh. H defaults to bh-49 (p150a, #687).
# Env: OUT (local out dir), TTMC (JIT cache dir name, one per A/B arm), RUNS (sh0 sh3 syn syn<K>), T2_CORES, REPS.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync}
D=docs/t713; O=${OUT:-$D/out}; TTMC=${TTMC:-ttmc-t713}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t713 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/$TTMC \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
echo "=== card"; timeout 60 "${SSH[@]}" "$ENV && tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t713 /localdev/smarton/.cache/$TTMC && mkdir -p tmp/t713 && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8 --target t2_bench) > tmp/t713/build.log 2>&1; rc=\$?; tail -25 tmp/t713/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
RUNS=${RUNS:-"sh0 sh3 syn"}
if [ $brc = 0 ]; then
  for n in $RUNS; do
    echo "=== run $n $(date +%T)"
    case "$n" in syn) k=${SYN_K:-8} ;; syn[0-9]*) k=${n#syn} ;; *) k= ;; esac
    if [ -n "$k" ]; then mk="python3 -m gsplat.t2_dev prep-synth tmp/t713/in-$n --n ${SYN_N:-1048576} --mean-count $k --no-ref"; ex="${T2_CORES:-130} ${REPS:-10}"
    else mk="python3 -m gsplat.t2_dev prep-dump benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t713/in-$n"; ex="${T2_CORES:-130} 3 grads"; fi
    timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t713/in-$n tmp/t713/out-$n && mkdir -p tmp/t713/out-$n && $mk \
      && timeout ${RUN_TIMEOUT:-300} render/train/build/t2_bench tmp/t713/in-$n tmp/t713/out-$n $ex > tmp/t713/out-$n/run.log 2>&1; rc=\$?; tail -20 tmp/t713/out-$n/run.log; \
      if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > tmp/t713/out-$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; \
      [ -n '$k' ] && rm -f tmp/t713/in-$n/*.f32 tmp/t713/in-$n/*.u16 tmp/t713/in-$n/*.u32 tmp/t713/out-$n/*.f32 tmp/t713/out-$n/*.u16; exit \$rc"
    echo "run $n rc=$?"
  done
fi
timeout 60 "${SSH[@]}" "find /localdev/smarton/.cache/$TTMC -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t713/jit-list.txt"; echo "jit list rc=$?"
opt/sync_remote.sh --pull tmp/t713 "$O/remote" "$H" "$T"; echo "pull rc=$?"
for n in $RUNS; do
  case "$n" in syn*) continue ;; esac
  if [ -s "$O/remote/out-$n/params_out.f32" ]; then python3 -m gsplat.t2_dev check "$O/remote/in-$n" "$O/remote/out-$n" | tee "$O/check-$n.json"
  else echo "CHECK $n: no outputs"; fi
done
echo "=== done $(TZ=America/Los_Angeles date)"

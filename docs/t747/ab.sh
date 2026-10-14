#!/bin/bash
# t747: blend backward A/B on p150 bh-49, all in ONE `ttp lock p100`.
#   ttp detach t747ab -- docs/t747/ab.sh <rev>      (Mac, worktree root; <rev> committed)
# base = #728 kernel (32d726ec, tmp/t747in/blend_bwd_compute.base.cpp), cand = <rev>'s kernel.
# One-way repo-scoped sync of this worktree (opt/sync_remote.sh), one build (the bench is the
# same for both arms; only the JIT-compiled kernel source differs, swapped in place), each arm
# with its own TT_METAL_CACHE. Inputs tmp/t747in/{hero,s1,s2}.npz are #728's exact dumps.
# Small dumps (sh0 sh3 s1 s2) once per arm; hero ROUNDS rounds, arm order alternating (ABBA).
# Each ssh step is short (< 5 min). Restrictions: existing measurement reservation only (no
# ird reserve/extend/release), never bh-30, StrictHostKeyChecking=yes on every ssh.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} ROUNDS=${ROUNDS:-5} REPS=${REPS:-8}
O=docs/t747/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/docs/t747/ab.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t747 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_STATS=1 \
  && source .venv/bin/activate"
K=render/train/kernels/blend_bwd_compute.cpp
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t747 && mkdir -p tmp/t747 tmp/t747in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t747/build.log 2>&1; rc=\$?; tail -5 tmp/t747/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
echo "=== push inputs $(date +%T)"
timeout 120 "${SSH[@]}" "cp $T/$K $T/tmp/t747in/cand.cpp"
for f in hero.npz s1.npz s2.npz blend_bwd_compute.base.cpp; do  # plain ssh: the p150 hosts have no rsync
  timeout 900 "${SSH[@]}" "cat > $T/tmp/t747in/$f.tmp && mv $T/tmp/t747in/$f.tmp $T/tmp/t747in/$f" < tmp/t747in/$f
  echo "push $f rc=$?"
done
timeout 60 "${SSH[@]}" "cd $T/tmp/t747in && md5sum *.npz *.cpp"
for n in sh0 sh3 s1 s2 hero; do
  case $n in sh*) src=benchmarks/train_oracle/t0_${n}_64x64_n50.npz ;; *) src=tmp/t747in/$n.npz ;; esac
  timeout 900 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev prep $src tmp/t747in/$n > /dev/null"; echo "prep $n rc=$?"
done
# run <arm> <dump> <outdir> <reps>: swap the arm's kernel in, run the bench, md5 the slots.
run() {
  local src=tmp/t747in/cand.cpp; [ "$1" = base ] && src=tmp/t747in/blend_bwd_compute.base.cpp
  timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t747-$1 && cp $src $K \
    && mkdir -p $3 && timeout 240 render/train/build/blend_bwd_bench tmp/t747in/$2 0 $4 $3 > $3/run.log 2>&1; rc=\$?; \
    tail -3 $3/run.log; md5sum $3/slots.f32 | tee $3/slots.md5; \
    if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > $3/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  echo "run $1 $2 -> $3 rc=$?"
}
for a in base cand; do for n in sh0 sh3 s1 s2; do run $a $n tmp/t747/$a/$n 3; done; done
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then arms="base cand"; else arms="cand base"; fi
  for a in $arms; do
    echo "=== round $r $a $(date +%T)"
    run $a hero tmp/t747/$a/hero-r$r "$REPS"
    # keep only the last cand round's slots (286 MB) for the hero oracle check; md5 kept for all
    [ "$a" = cand ] && [ "$r" = "$ROUNDS" ] || timeout 60 "${SSH[@]}" "rm -f $T/tmp/t747/$a/hero-r$r/slots.f32"
  done
done
timeout 60 "${SSH[@]}" "cp $T/tmp/t747in/cand.cpp $T/$K; for a in base cand; do find /localdev/smarton/.cache/ttmc-t747-\$a -maxdepth 6 -path '*/kernels/*' -type d > $T/tmp/t747/jit-\$a.txt; done; rm -rf $T/tmp/t747in"
echo "restore+jit+rm rc=$?"
opt/sync_remote.sh --pull tmp/t747 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t747"; echo "rm t747 rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

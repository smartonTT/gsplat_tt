#!/bin/bash
# t693 (T1): device blend-backward kernel vs the T0 gradient oracle, all in ONE `ttp lock p100`.
#   ttp detach t693 -- docs/t685/drive.sh <rev>      (Mac, worktree root; <rev> must be committed)
# Steps: ssh preflight, opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build render/train, for each oracle npz: `blend_backward_dev prep` on the box, run
# blend_bwd_bench, pull slots.f32 + timing.json back, then `blend_backward_dev check` here.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never
# the viewer box bh-30, StrictHostKeyChecking=yes on every ssh. H defaults to bh-33 (p100):
# timings from it are 'p100 (provisional)'.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-sync}
D=docs/t685; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t693 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t693 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
# WATCH=1: tt-metal watcher (waypoints per RISC) with its log pulled back under out/remote/logs.
[ "${WATCH:-0}" = 1 ] && ENV="$ENV && export TT_METAL_WATCHER=2 TT_METAL_LOGS_PATH=$T/tmp/t693/logs"
if [ "${RESET_FIRST:-0}" = 1 ]; then echo "=== tt-smi -r first"; timeout 300 "${SSH[@]}" "$ENV && tt-smi -r > /dev/null 2>&1; echo reset rc=\$?"; fi
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && mkdir -p tmp/t693 && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t693/build.log 2>&1; rc=\$?; tail -25 tmp/t693/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
if [ $brc = 0 ]; then
  for n in ${NPZS:-sh0 sh3}; do
    echo "=== run $n $(date +%T)"
    timeout 600 "${SSH[@]}" "$ENV && rm -rf tmp/t693/$n && python3 -m gsplat.blend_backward_dev prep benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t693/$n \
      && timeout ${RUN_TIMEOUT:-240} render/train/build/blend_bwd_bench tmp/t693/$n \${BWD_CORES:-} > tmp/t693/$n/run.log 2>&1; rc=\$?; tail -20 tmp/t693/$n/run.log; \
      if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > tmp/t693/$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
    echo "run $n rc=$?"
  done
fi
opt/sync_remote.sh --pull tmp/t693 "$O/remote" "$H" "$T"; echo "pull rc=$?"
for n in sh0 sh3; do
  f=$O/remote/$n/slots.f32
  if [ -s "$f" ]; then python3 -m gsplat.blend_backward_dev check benchmarks/train_oracle/t0_${n}_64x64_n50.npz "$f" --bar 1e-2 | tee "$O/check-$n.json"
  else echo "CHECK $n: no slots.f32"; fi
done
echo "=== done $(TZ=America/Los_Angeles date)"

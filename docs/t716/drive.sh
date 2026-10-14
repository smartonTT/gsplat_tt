#!/bin/bash
# t716 (T0 forward): device training forward (BLEND_TRAIN=1 production blend compute kernel,
# blend-only harness render/train/blend_fwd_bench) vs the T0 oracle image / final_T, then T1
# (blend_bwd_bench) fed with that device image, checked against the oracle gradients at 1e-2.
# All in ONE `ttp lock p100` (from docs/t685/drive.sh):
#   ttp detach t716 -- docs/t716/drive.sh <rev>      (Mac, worktree root; <rev> must be committed)
# Steps: ssh preflight, opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build render/train, per oracle npz on the box: blend_forward_dev prep -> blend_fwd_bench ->
# blend_forward_dev check --image-out -> blend_backward_dev prep --image -> blend_bwd_bench;
# pull tmp/t716 back, then both checks here.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never
# the viewer box bh-30, StrictHostKeyChecking=yes on every ssh. H = the p100 lock's box (bh-49 p150a).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync}
D=docs/t716; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t716 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
"${SSH[@]}" "tt-smi -ls 2>&1 | grep -E 'Blackhole|p150|p100' | head -2"
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t716 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t716 && mkdir -p tmp/t716 && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t716/build.log 2>&1; rc=\$?; tail -25 tmp/t716/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
if [ $brc = 0 ]; then
  for n in ${NPZS:-sh0 sh3}; do
    echo "=== run $n $(date +%T)"
    src=benchmarks/train_oracle/t0_${n}_64x64_n50.npz; o=tmp/t716/$n
    timeout 600 "${SSH[@]}" "$ENV && mkdir -p $o && python3 -m gsplat.blend_forward_dev prep $src $o/fwd \
      && timeout ${RUN_TIMEOUT:-240} render/train/build/blend_fwd_bench $o/fwd > $o/fwd/run.log 2>&1; rc=\$?; tail -5 $o/fwd/run.log; \
      if [ \$rc != 0 ]; then echo \"fwd rc=\$rc: tt-smi -r\"; tt-smi -r > $o/reset.log 2>&1; exit \$rc; fi; \
      python3 -m gsplat.blend_forward_dev check $src $o/fwd --image-out $o/image.f32; echo \"fwd check rc=\$?\"; \
      python3 -m gsplat.blend_backward_dev prep $src $o/bwd --image $o/image.f32 \
      && timeout ${RUN_TIMEOUT:-240} render/train/build/blend_bwd_bench $o/bwd > $o/bwd/run.log 2>&1; rc=\$?; tail -5 $o/bwd/run.log; \
      if [ \$rc != 0 ]; then echo \"bwd rc=\$rc: tt-smi -r\"; tt-smi -r > $o/reset.log 2>&1; fi; exit \$rc"
    echo "run $n rc=$?"
  done
fi
opt/sync_remote.sh --pull tmp/t716 "$O/remote" "$H" "$T"; echo "pull rc=$?"
fail=0
for n in ${NPZS:-sh0 sh3}; do
  src=benchmarks/train_oracle/t0_${n}_64x64_n50.npz; r=$O/remote/$n
  if [ -s "$r/fwd/train.f32" ]; then
    python3 -m gsplat.blend_forward_dev check "$src" "$r/fwd" --image-out "$r/image-local.f32" | tee "$O/check-fwd-$n.json"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    cmp -s "$r/image.f32" "$r/image-local.f32" && echo "image.f32 $n: box == local" || { echo "image.f32 $n: DIFFERS"; fail=1; }
  else echo "CHECK fwd $n: no train.f32"; fail=1; fi
  if [ -s "$r/bwd/slots.f32" ]; then
    python3 -m gsplat.blend_backward_dev check "$src" "$r/bwd/slots.f32" --bar 1e-2 | tee "$O/check-bwd-$n.json"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
  else echo "CHECK bwd $n: no slots.f32"; fail=1; fi
done
echo "=== done fail=$fail $(TZ=America/Los_Angeles date)"
exit $fail

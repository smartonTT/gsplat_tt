#!/bin/bash
# t720: SFPU fp32 Gaussian filter (LOSS_MODE=sfilt) vs the FPU split3 filter (LOSS_MODE=filt), ONE `ttp lock p100`:
#   ttp detach t720 -- docs/t720/drive.sh <rev>       (Mac, worktree root)
# ssh preflight @p100, opt/sync_remote.sh <rev> (one-way, repo-scoped), build render/bench, prep inputs on
# the box (numpy, no dataset files), correctness runs, ROUNDS interleaved timing rounds of both arms (own
# JIT caches), JIT listings for opt/ab_jit_check.py, pull tmp/t720, check on the Mac. Existing measurement
# reservation only (no ird reserve/extend/release), never bh-30, StrictHostKeyChecking=yes everywhere.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH=${MESH:-P150} ITERS=${ITERS:-50} ROUNDS=${ROUNDS:-4} RUN_TO=${RUN_TO:-120}
D=docs/t720; O=$D/out-$H
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t720 $rev=$sha to $H:$T mesh=$MESH $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
if [ $rc = 3 ]; then echo "sync rc=3 (stale-build heuristic), resync"; opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?; fi
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
C=/localdev/smarton/.cache/ttmc-t720
timeout 1200 "${SSH[@]}" "cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=$MESH PYTHONDONTWRITEBYTECODE=1 && \
  CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) && \
  { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/t720cfg.log 2>&1 && \
    nice -n 19 cmake --build render/bench/build --target loss_bench -j 8 > tmp/t720build.log 2>&1 || { tail -n 40 tmp/t720cfg.log tmp/t720build.log; exit 5; }; } && \
  rm -rf tmp/t720 $C-fpu $C-sfpu && source .venv/bin/activate && python3 opt/t720/sfilt_check.py prep tmp/t720 && \
  python3 opt/t690/loss_check.py prep tmp/t720/fpu > /dev/null && \
  B=render/bench/build/loss_bench && \
  CM=1 && o=tmp/t720/probe && mkdir -p \$o && \
  { TT_METAL_CACHE=$C-probe LOSS_MODE=sfilt LOSS_SF_CMODE=1 LOSS_NOOUT=1 timeout $RUN_TO \$B tmp/t720/dump64 \$o 2 > \$o/run.log 2>&1 || \
    { grep -m4 -E 'error' \$o/run.log; CM=0; }; } && echo \"sfilt SF_CMODE=\$CM (1 = hoisted taps compiled)\" && rm -rf $C-probe && \
  for c in dump64 syncrop syn1; do o=tmp/t720/\$c/out; mkdir -p \$o; \
    TT_METAL_CACHE=$C-sfpu LOSS_SF_CMODE=\$CM LOSS_MODE=sfilt timeout $RUN_TO \$B tmp/t720/\$c \$o $ITERS > \$o/run.log 2>&1; \
    echo \"check-run \$c rc=\$?\"; grep -E '^\[LOSS\]|TT_THROW|TT_FATAL|error' \$o/run.log | head -8; done && \
  for r in \$(seq 1 $ROUNDS); do \
    o=tmp/t720/fpu/syn/r\$r; mkdir -p \$o; TT_METAL_CACHE=$C-fpu LOSS_MODE=filt LOSS_HIBITS=9 timeout $RUN_TO \$B tmp/t720/fpu/syn \$o $ITERS > \$o/run.log 2>&1; \
    echo \"round \$r fpu-syn rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL' \$o/run.log | head -3; rm -f \$o/filt.f32; \
    for c in syn1 syn5 syn6; do o=tmp/t720/\$c/r\$r; mkdir -p \$o; \
      TT_METAL_CACHE=$C-sfpu LOSS_SF_CMODE=\$CM LOSS_MODE=sfilt LOSS_NOOUT=1 timeout $RUN_TO \$B tmp/t720/\$c \$o $ITERS > \$o/run.log 2>&1; \
      echo \"round \$r sfpu-\$c rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL' \$o/run.log | head -3; done; done; \
  for a in fpu sfpu; do find $C-\$a -maxdepth 8 -type d -path '*/kernels/*' > tmp/t720/jit-\$a.txt; done; \
  rm -f tmp/t720/*/planes.f32 tmp/t720/fpu/*/image.f32 tmp/t720/fpu/*/target.f32"
echo "remote rc=$?"
rm -rf "$O/remote"
opt/sync_remote.sh --pull tmp/t720 "$O/remote" "$H" "$T"; echo "pull rc=$?"
python3 opt/ab_jit_check.py --arm fpu="$O/remote/jit-fpu.txt" --arm sfpu="$O/remote/jit-sfpu.txt" \
  --kernel filt_compute --kernel sfilt_compute; echo "ab_jit_check rc=$? (1 expected: each compute kernel exists in one arm only)"
grep -h "^\[LOSS\] mode" "$O"/remote/fpu/syn/r*/run.log "$O"/remote/syn*/r*/run.log
python3 opt/t720/sfilt_check.py check "$O/remote"; echo "check rc=$?"
rm -f "$O"/remote/*/out/planes_out.f32
echo "=== done $(TZ=America/Los_Angeles date)"

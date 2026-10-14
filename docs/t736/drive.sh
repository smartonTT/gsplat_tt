#!/bin/bash
# t736: fused L1 + D-SSIM loss + dL/dimage (LOSS_MODE=ssim) vs the unfused sfilt filter path (LOSS_MODE=sfilt
# syn6: the 18 filtered planes of one training step, filter only), ONE `ttp lock p100`:
#   ttp detach t736 -- docs/t736/drive.sh <rev>       (Mac, worktree root)
# ssh preflight @p100, opt/sync_remote.sh <rev> (one-way, repo-scoped), build render/bench, prep dump64/syncrop/syn
# on the box; the JPG cases (bike_top, bike_mid, hero) are prepped on the Mac into tmp/t736/mac (derived f32,
# never committed) and copied one-way with ssh+tar (bh-49 has no rsync). Correctness runs, ROUNDS interleaved timing rounds of both arms
# (own JIT caches), JIT listings for opt/ab_jit_check.py, pull tmp/t736, T0 oracle check on the Mac.
# Existing measurement reservation only (no ird reserve/extend/release), never bh-30, StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH=${MESH:-P150} ITERS=${ITERS:-50} ROUNDS=${ROUNDS:-4} RUN_TO=${RUN_TO:-120}
D=docs/t736; O=$D/out-$H
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
SO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
SSH=(ssh "${SO[@]}" "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t736 $rev=$sha to $H:$T mesh=$MESH $(TZ=America/Los_Angeles date)"
rm -rf tmp/t736/mac && python3 opt/t736/ssim_check.py prep tmp/t736/mac bike_top bike_mid hero || exit 6
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
if [ $rc = 3 ]; then echo "sync rc=3 (stale-build heuristic), resync"; opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?; fi
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
C=/localdev/smarton/.cache/ttmc-t736
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=$MESH PYTHONDONTWRITEBYTECODE=1 && B=render/bench/build/loss_bench"
timeout 600 "${SSH[@]}" "$ENV && CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) && \
  { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/t736cfg.log 2>&1 && \
    nice -n 19 cmake --build render/bench/build --target loss_bench -j 8 > tmp/t736build.log 2>&1 || { tail -n 40 tmp/t736cfg.log tmp/t736build.log; exit 5; }; } && \
  rm -rf tmp/t736 $C-ssim $C-sfilt && source .venv/bin/activate && python3 opt/t736/ssim_check.py prep tmp/t736 dump64 syncrop syn && \
  python3 opt/t720/sfilt_check.py prep tmp/t736/sf syn6"; rc=$?
echo "build+prep rc=$rc"; [ $rc = 0 ] || exit 5
# bh-49 has no rsync: one-way copies use ssh+tar (bsdtar on the Mac, no ._ files)
TARC=(env COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs)
"${TARC[@]}" -C tmp/t736/mac -cf - . | timeout 300 "${SSH[@]}" "mkdir -p $T/tmp/t736 && tar -C $T/tmp/t736 -xf - 2>/dev/null"; echo "upload rc=$?"
timeout 600 "${SSH[@]}" "$ENV && for c in dump64 syncrop bike_top bike_mid hero syn; do o=tmp/t736/\$c/out; mkdir -p \$o; \
    TT_METAL_CACHE=$C-ssim LOSS_MODE=ssim timeout $RUN_TO \$B tmp/t736/\$c \$o 5 > \$o/run.log 2>&1; \
    echo \"check-run \$c rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL|error' \$o/run.log | head -20; done"
echo "check runs rc=$?"
timeout 600 "${SSH[@]}" "$ENV && for r in \$(seq 1 $ROUNDS); do \
    o=tmp/t736/syn/r\$r; mkdir -p \$o; TT_METAL_CACHE=$C-ssim LOSS_MODE=ssim timeout $RUN_TO \$B tmp/t736/syn \$o $ITERS > \$o/run.log 2>&1; \
    echo \"round \$r ssim rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL' \$o/run.log | head -3; rm -f \$o/grad.f32; \
    o=tmp/t736/sf/syn6/r\$r; mkdir -p \$o; TT_METAL_CACHE=$C-sfilt LOSS_SF_CMODE=1 LOSS_MODE=sfilt LOSS_NOOUT=1 timeout $RUN_TO \$B tmp/t736/sf/syn6 \$o $ITERS > \$o/run.log 2>&1; \
    echo \"round \$r sfilt-syn6 rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL' \$o/run.log | head -3; done; \
  for a in ssim sfilt; do find $C-\$a -maxdepth 8 -type d -path '*/kernels/*' > tmp/t736/jit-\$a.txt; done"
echo "timing rc=$?"
rm -rf "$O/remote"
mkdir -p "$O/remote"
timeout 300 "${SSH[@]}" "cd $T/tmp/t736 && tar --exclude='*.f32' -cf - ." | tar -C "$O/remote" -xf -; echo "pull rc=$?"
for c in dump64 syncrop bike_top bike_mid hero syn; do
  [ -d "$O/remote/$c/out" ] || continue
  timeout 120 "${SSH[@]}" "cat $T/tmp/t736/$c/out/grad.f32" > "$O/remote/$c/out/grad.f32" || rm -f "$O/remote/$c/out/grad.f32"
  for f in hw.txt image.f32 target.f32; do
    if [ -f "tmp/t736/mac/$c/$f" ]; then cp "tmp/t736/mac/$c/$f" "$O/remote/$c/"
    else timeout 120 "${SSH[@]}" "cat $T/tmp/t736/$c/$f" > "$O/remote/$c/$f" || rm -f "$O/remote/$c/$f"; fi
  done
done
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t736"; echo "remote cleanup rc=$?"
python3 opt/ab_jit_check.py --arm ssim="$O/remote/jit-ssim.txt" --arm sfilt="$O/remote/jit-sfilt.txt" \
  --kernel ssim_fwd_compute --kernel ssim_bwd_compute --kernel sfilt_compute; echo "ab_jit_check rc=$? (1 expected: each compute kernel exists in one arm only)"
grep -h "^\[LOSS\] mode" "$O"/remote/syn/r*/run.log "$O"/remote/sf/syn6/r*/run.log
python3 opt/t736/ssim_check.py check "$O/remote"; echo "check rc=$?"
rm -f "$O"/remote/*/out/grad.f32 "$O"/remote/*/*.f32
echo "=== done $(TZ=America/Los_Angeles date)"

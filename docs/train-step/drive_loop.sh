#!/bin/bash
# #750 step 2: the dump-free device training step chain (docs/train-step/loop_box.sh) on the
# oracle crop dumps, all in ONE `ttp lock p100`.
#   ttp detach t750loop -- docs/train-step/drive_loop.sh <rev>      (Mac, worktree root; <rev> committed)
# Steps: ssh preflight (@p100), opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build train_step + blend_fwd_bench (render/train) and loss_bench (render/bench), loop_box.sh per
# dump, pull tmp/t750 to docs/train-step/out/loop.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh.
# RUNS=hero (#781): the full 1237x822 bicycle view. HERO_NPZ (default tmp/t750/hero.npz, made by
# tmp/t750/hero_prep.sh's docs/t685/hero.py step) gets the target image (the dataset JPG, never
# committed) and is pushed zstd-compressed to $T/tmp/t781; loop_box.sh HERO=1 then runs detached
# on the box (setsid nohup, survives an ssh drop) while this driver holds the lock and polls its
# .rc (HERO_WAIT_S, default 3 h). The pushed npz and the big outputs are deleted before the pull.
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
# rc 3 = sync_remote's stale-build heuristic (render/train changed, render_clean did not): one resync (#792).
if [ $rc = 3 ]; then echo "sync rc=3, resync"; opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync-loop.log" 2>&1; rc=$?; fi
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync-loop.log"
[ $rc = 0 ] || { tail -30 "$O/sync-loop.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P150 PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t750 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
hero_loop() {
  local src=${HERO_NPZ:-tmp/t750/hero.npz} L=tmp/t781 R=$T/tmp/t781 t0 rc i
  [ -s "$src" ] || { echo "no $src: run tmp/t750/hero_prep.sh (docs/t685/hero.py) first"; return 5; }
  mkdir -p "$L"
  # The project venv has no PIL (#779): the system python decodes the JPG to raw RGB8.
  local jpg=${HERO_JPG:-$(dirname "$TP")/scenes/mipnerf360/bicycle/images_4/_DSC8679.JPG}
  "${PY_IMG:-/usr/bin/python3}" -c "
import sys; from PIL import Image
open(sys.argv[2], 'wb').write(Image.open(sys.argv[1]).convert('RGB').tobytes())" "$jpg" "$L/target.rgb8" || return 5
  python3 -c "
import hashlib, sys, numpy as np
d = np.load(sys.argv[1]); k = {n: d[n] for n in d.files}
assert hashlib.md5(open(sys.argv[3], 'rb').read()).hexdigest() == str(k['target_md5']), 'target md5'
H, W = int(k['H']), int(k['W'])
k['target'] = np.fromfile(sys.argv[4], np.uint8).reshape(H, W, 3).astype(np.float32) / np.float32(255)
m = np.load(sys.argv[5])   # docs/train-step/hero_model.py: emulate_fast(prep_binned(hero)), the forward gate
k['fwd_model_image'], k['fwd_model_final_T'] = m['image'], m['final_T']
np.savez(sys.argv[2], **k)" "$src" "$L/hero_chain.npz" "$jpg" "$L/target.rgb8" "${HERO_MODEL:-tmp/t781/hero_model.npz}" || return 5
  t0=$(date +%s)
  COPYFILE_DISABLE=1 tar -C "$L" --no-xattrs -cf - hero_chain.npz | zstd -3 -T4 -q \
    | timeout 2400 "${SSH[@]}" "rm -rf $R && mkdir -p $R && zstd -dc | tar -C $R -xf - && df -h $R | tail -1"
  rc=$?; echo "push hero rc=$rc $(( $(date +%s) - t0 ))s"; rm -f "$L/hero_chain.npz"
  [ $rc = 0 ] || { timeout 60 "${SSH[@]}" "rm -rf $R"; return 6; }
  timeout 60 "${SSH[@]}" "$ENV && setsid nohup bash -c 'HERO=1 LOOP_DIR=tmp/t781 bash docs/train-step/loop_box.sh hero \
    tmp/t781/hero_chain.npz ${REPS:-5} > tmp/t781/loop-hero.log 2>&1; echo \$? > tmp/t781/loop-hero.rc' < /dev/null > /dev/null 2>&1 &"
  echo "start hero rc=$?"
  rc=
  for ((i = 0; i < ${HERO_WAIT_S:-10800} / 30; i++)); do
    sleep 30
    rc=$(timeout 30 "${SSH[@]}" "cat $R/loop-hero.rc 2>/dev/null") && [ -n "$rc" ] && break
  done
  echo "hero rc=${rc:-timeout} $(( $(date +%s) - t0 ))s"
  timeout 120 "${SSH[@]}" "cd $T && rm -f tmp/t781/hero_chain.npz && tail -40 tmp/t781/loop-hero.log && du -sh tmp/t781 && df -h tmp | tail -1"
  [ "$rc" = 0 ]
}
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
    if [ "$n" = hero ]; then hero_loop; echo "loop hero rc=$?"; continue; fi
    timeout 1800 "${SSH[@]}" "$ENV && bash docs/train-step/loop_box.sh $n benchmarks/train_oracle/t0_${n}_64x64_n50.npz ${REPS:-10}"
    echo "loop $n rc=$?"
  done
fi
case " ${RUNS:-sh0 sh3} " in *" hero "*)
  opt/sync_remote.sh --pull tmp/t781 "$O/loop-hero" "$H" "$T"; echo "pull hero rc=$?"
  # #798: the device image and tile table leave docs/ (never committed) for the per-tile gate.
  mkdir -p tmp/hero-fwd && mv -f "$O/loop-hero/loop-hero/image.f32" "$O/loop-hero/loop-hero/fwd/tiles.u32" tmp/hero-fwd/
  echo "hero image rc=$?"
  python3 docs/train-step/hero_fwd_tiles.py tmp/hero-fwd/image.f32 tmp/hero-fwd/tiles.u32 \
    "${HERO_MODEL:-tmp/t781/hero_model.npz}" > "$O/loop-hero/fwd_tiles.json"; echo "fwd tiles rc=$?" ;; esac
case " ${RUNS:-sh0 sh3} " in *" sh"*) opt/sync_remote.sh --pull tmp/t750 "$O/loop" "$H" "$T"; echo "pull rc=$?" ;; esac
echo "=== done $(TZ=America/Los_Angeles date)"

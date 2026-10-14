#!/bin/bash
# t748: on-device training step chain (T1 blend backward -> shared 4 KB slot pages -> T2 param
# pass + Adam), render/train/train_step, all in ONE `ttp lock p100`.
#   ttp detach t748 -- docs/train-step/drive.sh <rev>      (Mac, worktree root; <rev> committed)
# Steps: ssh preflight (@p100), opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build render/train (train_step), per dump: train_step_dev prep on the box, train_step run, pull
# tmp/t748 back, then `train_step_dev check` here (T1 slots, T2 vs t2_step, T0 oracle).
# RUNS=hero (#750): the full-view hero chain; see the hero branch below (check runs on the box).
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
T=${T:-/localdev/smarton/gstt2-sync}
[ -n "$H" ] || { echo "=== no p100 host: set H= or device.hosts.p100 in $TP/harness/project.json"; exit 2; }
D=docs/train-step; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t748 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t748 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
echo "=== card"; timeout 60 "${SSH[@]}" "$ENV && tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && mkdir -p tmp/t748 && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8 --target train_step) > tmp/t748/build.log 2>&1; rc=\$?; tail -25 tmp/t748/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
RUNS=${RUNS:-"sh0 sh3"}
if [ $brc = 0 ]; then
  for n in $RUNS; do
    echo "=== run $n $(date +%T)"
    if [ "$n" = hero ]; then
      # #750 hero: dump + prep built on the Mac (tmp/t750, docs/t685/hero.py + train_step_dev prep),
      # pushed over ssh (tar), run, checked on the box, big files deleted there (keep disk lean).
      [ -s tmp/t750/hero.npz ] && [ -s tmp/t750/in-hero/chain.json ] || { echo "no tmp/t750 hero prep"; continue; }
      # The link through the jump host runs ~0.8 MB/s: push zstd-compressed (1.6 GB -> ~0.37 GB) and
      # only a stub of hero.npz (the check reads cfg_json and the key list from it, no arrays).
      python3 -c "import numpy as np;d=np.load('tmp/t750/hero.npz');np.savez('tmp/t750/hero_check.npz',**{k:d[k] for k in ('H','W','N','sh_degree','cfg_json')})" || continue
      t0=$(date +%s)
      COPYFILE_DISABLE=1 tar -C tmp/t750 --no-xattrs -cf - hero_check.npz in-hero | zstd -3 -T4 -q | timeout 1800 "${SSH[@]}" "rm -rf $T/tmp/t748/hero && mkdir -p $T/tmp/t748/hero && zstd -dc | tar -C $T/tmp/t748/hero -xf -"
      prc=$?; echo "push hero rc=$prc $(( $(date +%s) - t0 ))s"; [ $prc = 0 ] || { timeout 60 "${SSH[@]}" "rm -rf $T/tmp/t748/hero"; continue; }
      timeout 2400 "${SSH[@]}" "$ENV && cd tmp/t748/hero && mkdir -p out && (timeout ${RUN_TIMEOUT:-600} ../../../render/train/build/train_step in-hero out ${CORES:-110} ${REPS:-5} grads > out/run.log 2>&1; rc=\$?; tail -20 out/run.log; \
        if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > out/reset.log 2>&1; echo \"reset rc=\$?\"; exit \$rc; fi; \
        cd ../../.. && nice -n 19 python3 -m gsplat.train_step_dev check tmp/t748/hero/hero_check.npz tmp/t748/hero/in-hero tmp/t748/hero/out > tmp/t748/hero/out/check.json 2> tmp/t748/hero/check.log; echo \"CHECK hero rc=\$?\"; cat tmp/t748/hero/out/check.json; tail -20 tmp/t748/hero/check.log); \
        rc=\$?; mkdir -p tmp/t748/out-hero && cp tmp/t748/hero/out/*.json tmp/t748/hero/out/t2_rows.npz tmp/t748/hero/out/run.log tmp/t748/hero/check.log tmp/t748/out-hero/ 2>/dev/null; rm -rf tmp/t748/hero; exit \$rc"
      echo "run hero rc=$?"; continue
    fi
    timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t748/in-$n tmp/t748/out-$n && mkdir -p tmp/t748/out-$n \
      && python3 -m gsplat.train_step_dev prep benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t748/in-$n \
      && timeout ${RUN_TIMEOUT:-300} render/train/build/train_step tmp/t748/in-$n tmp/t748/out-$n ${CORES:-110} ${REPS:-10} grads > tmp/t748/out-$n/run.log 2>&1; rc=\$?; tail -20 tmp/t748/out-$n/run.log; \
      if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > tmp/t748/out-$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
    echo "run $n rc=$?"
  done
fi
opt/sync_remote.sh --pull tmp/t748 "$O/remote" "$H" "$T"; echo "pull rc=$?"
for n in $RUNS; do
  if [ "$n" = hero ]; then cp "$O/remote/out-hero/check.json" "$O/check-hero.json" 2>/dev/null; cat "$O/remote/out-hero/timing.json" 2>/dev/null; continue; fi
  if [ -s "$O/remote/out-$n/params_out.f32" ]; then python3 -m gsplat.train_step_dev check benchmarks/train_oracle/t0_${n}_64x64_n50.npz "$O/remote/in-$n" "$O/remote/out-$n" > "$O/check-$n.json"; echo "CHECK $n rc=$?"; cat "$O/remote/out-$n/timing.json" 2>/dev/null
  else echo "CHECK $n: no outputs"; fi
done
echo "=== done $(TZ=America/Los_Angeles date)"

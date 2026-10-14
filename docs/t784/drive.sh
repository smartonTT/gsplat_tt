#!/bin/bash
# t784: T2 heavy-row split on device, hero A/B, all in ONE `ttp lock p100`.
#   ttp detach t784 -- docs/t784/drive.sh <rev>      (Mac, worktree root; <rev> committed)
# Steps: ssh preflight (@p100), opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build train_step, push the hero T2 prep (tmp/t750 of the t770 worktree, HERO_SRC=), then on the box:
#   g2d   T2_SPLIT=1 T2_G2D=1 (kcap 16, and 8): g2d vs segment_reduce bit for bit (check_g2d.py g2d)
#   same  base vs split (grads): every output bit-equal on the active groups (check_g2d.py same)
#   ab    ROUNDS (4) ABBA rounds of base / split, REPS (10) steps each; timing.json per run
#   kcap8 one split run at T2_KCAP=8
#   t0    sh0/sh3 crops with T2_SPLIT=1 at kcap 1 and 16 + gsplat.train_step_dev check (T0 oracle)
#   jit   opt/ab_jit_check.py on the two arms' JIT caches (ttmc-t784-base / ttmc-t784-split)
# then pull tmp/t784/res and delete the hero data on the box (keep disk lean).
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh, nothing synced from outside the repo
# except the hero T2 prep (derived arrays, no dataset images).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
T=${T:-/localdev/smarton/gstt2-sync}
HERO_SRC=${HERO_SRC:-$TP/worktrees/t770/tmp/t750}
[ -n "$H" ] || { echo "=== no p100 host"; exit 2; }
D=docs/t784; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t784 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
ENV0="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
CB=/localdev/smarton/.cache/ttmc-t784-base; CS=/localdev/smarton/.cache/ttmc-t784-split
echo "=== card / disk"; timeout 60 "${SSH[@]}" "$ENV0 && tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'; df -h /localdev | tail -1"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV0 && mkdir -p tmp/t784/res && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8 --target train_step) > tmp/t784/res/build.log 2>&1; rc=\$?; tail -25 tmp/t784/res/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
[ $brc = 0 ] || { opt/sync_remote.sh --pull tmp/t784/res "$O/remote" "$H" "$T"; exit 4; }

# run_hero <name> <cache> <reps> <env...>: one train_step process on the hero; keeps the outputs
# in tmp/t784/hero/out-<name> (deleted by the caller), copies timing/log to tmp/t784/res/<name>.
run_hero() {
  local n=$1 c=$2 r=$3; shift 3
  echo "=== hero $n ($*) $(date +%T)"
  timeout 1200 "${SSH[@]}" "$ENV0 && export TT_METAL_CACHE=$c $* && cd tmp/t784/hero && rm -rf out-$n && mkdir -p out-$n ../res/$n \
    && timeout ${RUN_TIMEOUT:-600} ../../../render/train/build/train_step in-hero out-$n ${CORES:-110} $r grads > out-$n/run.log 2>&1; rc=\$?; \
    grep -E 'SPLIT|error|Error' out-$n/run.log | head -5; cp out-$n/run.log out-$n/timing.json ../res/$n/ 2>/dev/null; cat out-$n/timing.json 2>/dev/null; echo; \
    if [ \$rc != 0 ]; then tail -20 out-$n/run.log; echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > ../res/$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  local rc=$?; echo "hero $n rc=$rc"; return $rc
}
rmout() { timeout 120 "${SSH[@]}" "cd $T/tmp/t784/hero && rm -rf $(printf 'out-%s ' "$@")"; }
box() { timeout ${2:-1800} "${SSH[@]}" "$ENV0 && $1"; }

[ -s "$HERO_SRC/in-hero/t2/meta.u32" ] || { echo "no hero prep at $HERO_SRC"; exit 5; }
t0=$(date +%s)
COPYFILE_DISABLE=1 tar -C "$HERO_SRC" --no-xattrs -cf - hero_check.npz in-hero | zstd -3 -T4 -q \
  | timeout 2400 "${SSH[@]}" "rm -rf $T/tmp/t784/hero && mkdir -p $T/tmp/t784/hero && zstd -dc | tar -C $T/tmp/t784/hero -xf -"
prc=$?; echo "push hero rc=$prc $(( $(date +%s) - t0 ))s"
if [ $prc = 0 ]; then
  K=${KCAPS:-"16 8"}
  for k in $K; do
    run_hero g2d-k$k $CS 1 T2_SPLIT=1 T2_G2D=1 T2_KCAP=$k \
      && box "nice -n 19 python3 docs/t784/check_g2d.py g2d tmp/t784/hero/in-hero tmp/t784/hero/out-g2d-k$k $k 16 2 ${CORES:-110} | tee tmp/t784/res/g2d-k$k.json; echo \"G2D k$k rc=\${PIPESTATUS[0]}\""
    rmout g2d-k$k
  done
  run_hero base0 $CB 3 T2_SPLIT=0; run_hero split0 $CS 3 T2_SPLIT=1
  box "python3 docs/t784/check_g2d.py same tmp/t784/hero/in-hero tmp/t784/hero/out-base0 tmp/t784/hero/out-split0 | tee tmp/t784/res/same.json; echo \"SAME rc=\${PIPESTATUS[0]}\""
  box "nice -n 19 python3 -m gsplat.train_step_dev check tmp/t784/hero/hero_check.npz tmp/t784/hero/in-hero tmp/t784/hero/out-split0 > tmp/t784/res/check-split0.json 2> tmp/t784/res/check-split0.log; echo \"CHECK split0 rc=\$?\"" 2400 \
    || true
  rmout base0 split0
  for r in $(seq 1 ${ROUNDS:-4}); do
    if [ $((r % 2)) = 1 ]; then o="base split"; else o="split base"; fi
    for a in $o; do
      if [ $a = base ]; then run_hero ab$r-base $CB ${REPS:-10} T2_SPLIT=0; else run_hero ab$r-split $CS ${REPS:-10} T2_SPLIT=1; fi
      rmout ab$r-$a
    done
  done
  for k in ${KCAP_AB:-8}; do run_hero ab-k$k $CS ${REPS:-10} T2_SPLIT=1 T2_KCAP=$k; rmout ab-k$k; done
fi
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t784/hero"; echo "hero dir removed rc=$?"
for n in sh0 sh3; do
  for k in 1 16; do
    echo "=== t0 $n kcap $k $(date +%T)"
    box "export TT_METAL_CACHE=$CS T2_SPLIT=1 T2_KCAP=$k && rm -rf tmp/t784/in-$n tmp/t784/out-$n-k$k && mkdir -p tmp/t784/out-$n-k$k \
      && python3 -m gsplat.train_step_dev prep benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t784/in-$n \
      && timeout 300 render/train/build/train_step tmp/t784/in-$n tmp/t784/out-$n-k$k ${CORES:-110} 3 grads > tmp/t784/out-$n-k$k/run.log 2>&1 \
      && grep SPLIT tmp/t784/out-$n-k$k/run.log; \
      python3 -m gsplat.train_step_dev check benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t784/in-$n tmp/t784/out-$n-k$k > tmp/t784/res/check-$n-k$k.json; \
      echo \"CHECK $n k$k rc=\$?\"; python3 -c \"import json;d=json.load(open('tmp/t784/res/check-$n-k$k.json'));print({k:v.get('pass') if isinstance(v,dict) else v for k,v in d.items() if k in ('t1','t2','t0','pass')})\"" 900
  done
done
echo "=== jit"
box "python3 opt/ab_jit_check.py --arm base=$CB --arm cand=$CS --kernel t2g_reader --kernel t2g_compute | tee tmp/t784/res/jit.txt; echo \"JIT rc=\${PIPESTATUS[0]}\"" 120
opt/sync_remote.sh --pull tmp/t784/res "$O/remote" "$H" "$T"; echo "pull rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

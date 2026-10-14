#!/bin/bash
# t793: T2 split host plan off the critical path, hero A/B, all in ONE `ttp lock p100`.
#   ttp detach t793 -- docs/t793/drive.sh <cand rev>      (Mac, worktree root; revs committed)
# Arms (kernels identical; only the host plan code differs):
#   base  BASE_REV (#784 head 9a30c819) train_step, T2_SPLIT=1 (plan once, serial, 14.5 MB metab copy)
#   cand  <cand rev> train_step, T2_SPLIT=1 T2_PERVIEW=1 (fast plan; per-view loop serial vs overlap)
# Steps: ssh preflight (@p100), opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree)
# of BASE_REV, build, keep the binary as tmp/t793/train_step.base; sync + build <cand rev>; push the
# hero T2 prep (tmp/t750 of the t770 worktree, HERO_SRC=); then on the box:
#   same  base vs cand step-0 outputs (grads) bit-equal (docs/t784/check_g2d.py same)
#   ab    ROUNDS (4) alternating rounds of base / cand, REPS (10) steps each; timing.json per run
#   t0    sh0/sh3 crops, cand binary, T2_SPLIT=1 T2_PERVIEW=1 at kcap 1 and 16 + T0 oracle check
#   jit   opt/ab_jit_check.py --runtime-arg-only on the two arms' JIT caches + run logs
# then pull tmp/t793/res and delete the hero data on the box (keep disk lean).
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
D=docs/t793; O=$D/out; BASE_REV=${BASE_REV:-9a30c819}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
ENV0="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
CB=/localdev/smarton/.cache/ttmc-t793-base; CS=/localdev/smarton/.cache/ttmc-t793-cand
echo "=== card / disk"; timeout 60 "${SSH[@]}" "$ENV0 && tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'; df -h /localdev | tail -1"
# sync_build <rev> <tag>: sync <rev>, check the stamp, build train_step; base keeps a copy.
sync_build() {
  local sha; sha=$(git rev-parse "$1^{commit}")
  echo "=== $2 $1=$sha to $H:$T $(TZ=America/Los_Angeles date)"
  opt/sync_remote.sh --no-build "$H" "$T" "$1" > "$O/sync-$2.log" 2>&1; local rc=$?
  echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync-$2.log"
  [ $rc = 0 ] || { tail -30 "$O/sync-$2.log"; exit 3; }
  local rsha; rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
  echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
  [ "$rsha" = "$sha" ] || exit 3
  echo "=== build $2 $(date +%T)"
  timeout 900 "${SSH[@]}" "$ENV0 && mkdir -p tmp/t793/res && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
    && nice -n 19 cmake --build render/train/build -j 8 --target train_step) > tmp/t793/res/build-$2.log 2>&1; rc=\$?; tail -5 tmp/t793/res/build-$2.log; \
    [ \$rc = 0 ] && cp render/train/build/train_step tmp/t793/train_step.$2; exit \$rc"
  local brc=$?; echo "build $2 rc=$brc"
  [ $brc = 0 ] || { opt/sync_remote.sh --pull tmp/t793/res "$O/remote" "$H" "$T"; exit 4; }
}
sync_build "$BASE_REV" base
sync_build "$rev" cand

# run_hero <name> <cache> <reps> <env...> (BIN=base|cand, default cand): one train_step process on the hero; keeps the outputs
# in tmp/t793/hero/out-<name> (deleted by the caller), copies timing/log to tmp/t793/res/<name>.
run_hero() {
  local n=$1 c=$2 r=$3; shift 3
  echo "=== hero $n ($*) $(date +%T)"
  timeout 1200 "${SSH[@]}" "$ENV0 && export TT_METAL_CACHE=$c $* && cd tmp/t793/hero && rm -rf out-$n && mkdir -p out-$n ../res/$n \
    && timeout ${RUN_TIMEOUT:-600} ../train_step.${BIN:-cand} in-hero out-$n ${CORES:-110} $r grads > out-$n/run.log 2>&1; rc=\$?; \
    grep -E 'SPLIT|PERVIEW|error|Error' out-$n/run.log | head -5; cp out-$n/run.log out-$n/timing.json ../res/$n/ 2>/dev/null; cat out-$n/timing.json 2>/dev/null; echo; \
    if [ \$rc != 0 ]; then tail -20 out-$n/run.log; echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > ../res/$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  local rc=$?; echo "hero $n rc=$rc"; return $rc
}
rmout() { timeout 120 "${SSH[@]}" "cd $T/tmp/t793/hero && rm -rf $(printf 'out-%s ' "$@")"; }
box() { timeout ${2:-1800} "${SSH[@]}" "$ENV0 && $1"; }

[ -s "$HERO_SRC/in-hero/t2/meta.u32" ] || { echo "no hero prep at $HERO_SRC"; exit 5; }
t0=$(date +%s)
COPYFILE_DISABLE=1 tar -C "$HERO_SRC" --no-xattrs -cf - hero_check.npz in-hero | zstd -3 -T4 -q \
  | timeout 2400 "${SSH[@]}" "rm -rf $T/tmp/t793/hero && mkdir -p $T/tmp/t793/hero && zstd -dc | tar -C $T/tmp/t793/hero -xf -"
prc=$?; echo "push hero rc=$prc $(( $(date +%s) - t0 ))s"
if [ $prc = 0 ]; then
  BIN=base run_hero base0 $CB 3 T2_SPLIT=1; BIN=cand run_hero cand0 $CS 3 T2_SPLIT=1 T2_PERVIEW=1
  box "python3 docs/t784/check_g2d.py same tmp/t793/hero/in-hero tmp/t793/hero/out-base0 tmp/t793/hero/out-cand0 | tee tmp/t793/res/same.json; echo \"SAME rc=\${PIPESTATUS[0]}\""
  rmout base0 cand0
  for r in $(seq 1 ${ROUNDS:-4}); do
    if [ $((r % 2)) = 1 ]; then o="base cand"; else o="cand base"; fi
    for a in $o; do
      if [ $a = base ]; then BIN=base run_hero ab$r-base $CB ${REPS:-10} T2_SPLIT=1
      else BIN=cand run_hero ab$r-cand $CS ${REPS:-10} T2_SPLIT=1 T2_PERVIEW=1; fi
      rmout ab$r-$a
    done
  done
fi
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t793/hero"; echo "hero dir removed rc=$?"
for n in sh0 sh3; do
  for k in 1 16; do
    echo "=== t0 $n kcap $k $(date +%T)"
    box "export TT_METAL_CACHE=$CS T2_SPLIT=1 T2_PERVIEW=1 T2_KCAP=$k && rm -rf tmp/t793/in-$n tmp/t793/out-$n-k$k && mkdir -p tmp/t793/out-$n-k$k \
      && python3 -m gsplat.train_step_dev prep benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t793/in-$n \
      && timeout 300 tmp/t793/train_step.cand tmp/t793/in-$n tmp/t793/out-$n-k$k ${CORES:-110} 3 grads > tmp/t793/out-$n-k$k/run.log 2>&1 \
      && grep -E 'SPLIT|PERVIEW' tmp/t793/out-$n-k$k/run.log; \
      python3 -m gsplat.train_step_dev check benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t793/in-$n tmp/t793/out-$n-k$k > tmp/t793/res/check-$n-k$k.json; \
      echo \"CHECK $n k$k rc=\$?\"; python3 -c \"import json;d=json.load(open('tmp/t793/res/check-$n-k$k.json'));print({k:v.get('pass') if isinstance(v,dict) else v for k,v in d.items() if k in ('t1','t2','t0','pass')})\"" 900
  done
done
echo "=== jit"
L=""; for r in $(seq 1 ${ROUNDS:-4}); do L="$L --log base=tmp/t793/res/ab$r-base/run.log --log cand=tmp/t793/res/ab$r-cand/run.log"; done
box "python3 opt/ab_jit_check.py --runtime-arg-only --arm base=$CB --arm cand=$CS --kernel t2g_reader --kernel t2g_compute --kernel t2h_writer \
  --marker 'base=[SPLIT] kcap 16' --marker 'cand=[PERVIEW] plan_stable 1' $L | tee tmp/t793/res/jit.txt; echo \"JIT rc=\${PIPESTATUS[0]}\"" 120
opt/sync_remote.sh --pull tmp/t793/res "$O/remote" "$H" "$T"; echo "pull rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

#!/bin/bash
# t767: blend backward A/B on p150 bh-49, all in ONE `ttp lock p100`.
#   ttp detach t767ab -- docs/t767/ab.sh <rev>      (Mac, worktree root; <rev> committed)
# One source tree and one bench build; the arms differ only in the bench's env BWD_PRESTAGE
# (base 0 = #747 pair setup, cand 1 = #767 reader-staged SFPLOADI words), which changes the
# reader and compute kernels' defines, each arm with its own TT_METAL_CACHE. Per-arm ELF md5s of
# every JIT kernel binary are recorded (ab_jit_check gives false A/A on in-place swaps).
# Inputs tmp/t767in/{hero,s1,s2}.npz are #728's dumps (same md5s as #747's t747in).
# Small dumps (sh0 sh3 s1 s2) once per arm + device T0 oracle check; hero ROUNDS rounds, arm order
# alternating (ABBA); first 9 floats per slot compared base vs cand on the box.
# Each ssh step is short (< 5 min). Restrictions: existing measurement reservation only (no
# ird reserve/extend/release), never bh-30, StrictHostKeyChecking=yes on every ssh, sync only
# through the repo-scoped one-way opt/sync_remote.sh inside this lock.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} ROUNDS=${ROUNDS:-5} REPS=${REPS:-8}
O=${O:-docs/t767/out} CAND_PS=${CAND_PS:-1}  # cand arm's BWD_PRESTAGE mode (1 or 2)
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" CAND_PS="$CAND_PS" "$PWD/docs/t767/ab.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t767 cand BWD_PRESTAGE=$CAND_PS $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_STATS=1 \
  && source .venv/bin/activate"
C=/localdev/smarton/.cache/ttmc-t767
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t767 $C-base $C-cand && mkdir -p tmp/t767 tmp/t767in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t767/build.log 2>&1; rc=\$?; tail -5 tmp/t767/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
echo "=== push inputs $(date +%T)"
for f in hero.npz s1.npz s2.npz; do  # plain ssh: the p150 hosts have no rsync
  timeout 900 "${SSH[@]}" "cat > $T/tmp/t767in/$f.tmp && mv $T/tmp/t767in/$f.tmp $T/tmp/t767in/$f" < tmp/t767in/$f
  echo "push $f rc=$?"
done
timeout 60 "${SSH[@]}" "cd $T/tmp/t767in && md5sum *.npz"
src_of() { case $1 in sh*) echo benchmarks/train_oracle/t0_${1}_64x64_n50.npz ;; *) echo tmp/t767in/$1.npz ;; esac; }
for n in sh0 sh3 s1 s2 hero; do
  timeout 900 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev prep $(src_of $n) tmp/t767in/$n > /dev/null"; echo "prep $n rc=$?"
done
# run <arm> <dump> <outdir> <reps>
run() {
  local ps=$CAND_PS; [ "$1" = base ] && ps=0
  timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-$1 BWD_PRESTAGE=$ps \
    && mkdir -p $3 && timeout 240 render/train/build/blend_bwd_bench tmp/t767in/$2 0 $4 $3 > $3/run.log 2>&1; rc=\$?; \
    tail -3 $3/run.log; md5sum $3/slots.f32 | tee $3/slots.md5; \
    if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > $3/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  echo "run $1 $2 -> $3 rc=$?"
}
# same9 <dir a> <dir b>: the 9 output floats per slot, bit for bit
same9() {
  timeout 240 "${SSH[@]}" "$ENV && python3 -c \"import numpy as np; a=np.fromfile('$1/slots.f32', np.uint32).reshape(-1, 16)[:, :9]; \
b=np.fromfile('$2/slots.f32', np.uint32).reshape(-1, 16)[:, :9]; print('same9 $1 $2 slots', len(a), 'identical' if (a == b).all() else 'DIFFER %d' % int((a != b).any(1).sum()))\""
}
for a in base cand; do for n in sh0 sh3 s1 s2; do run $a $n tmp/t767/$a/$n 3; done; done
for n in sh0 sh3 s1 s2; do
  same9 tmp/t767/base/$n tmp/t767/cand/$n
  for a in base cand; do
    timeout 240 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev check $(src_of $n) tmp/t767/$a/$n/slots.f32 > tmp/t767/check-$n-$a.json 2>&1; echo \"check $n $a rc=\$?\"; tail -c 300 tmp/t767/check-$n-$a.json"
  done
done
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then arms="base cand"; else arms="cand base"; fi
  for a in $arms; do
    echo "=== round $r $a $(date +%T)"
    run $a hero tmp/t767/$a/hero-r$r "$REPS"
  done
  same9 tmp/t767/base/hero-r$r tmp/t767/cand/hero-r$r
  # keep only the last cand round's slots (286 MB) for the hero oracle check on the Mac
  for a in base cand; do
    [ "$a" = cand ] && [ "$r" = "$ROUNDS" ] || timeout 60 "${SSH[@]}" "rm -f $T/tmp/t767/$a/hero-r$r/slots.f32"
  done
done
timeout 60 "${SSH[@]}" "for a in base cand; do find $C-\$a -path '*/kernels/*' -name '*.elf' | sort | xargs md5sum | sed \"s|^|\$a |\"; done > $T/tmp/t767/jit-elf-md5.txt; cat $T/tmp/t767/jit-elf-md5.txt; rm -rf $T/tmp/t767in"
echo "jit+rm rc=$?"
opt/sync_remote.sh --pull tmp/t767 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t767"; echo "rm t767 rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

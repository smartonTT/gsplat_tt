#!/bin/bash
# t785: T1 blend backward load balance A/B on the p150 measurement box, all in ONE `ttp lock p100`.
#   ttp detach t785ab -- docs/t785/ab.sh <rev>      (Mac, worktree root; <rev> committed)
# One source tree, one bench build, same kernels and defines in every arm; the arms differ only in
# the prep dir the bench runs (host-side row order + per-core row boundaries, cores.u32):
#   base  = blend_backward_dev prep (contiguous split by pairs, the #767 bench)
#   lpt   = t1_balance_dev plan: LPT by modelled cycles, whole tiles (bit-identical outputs)
#   split = t1_balance_dev plan --split --kmax 2 (partial slots merged on the box before checks)
# Each arm keeps its own TT_METAL_CACHE; JIT dirs and ELF md5s are recorded per arm.
# Inputs tmp/t785in/{hero,s1,s2}.npz = #767's tmp/t767in dumps (same md5s).
# Small dumps (sh0 sh3 s1 s2) once per arm + device T0 oracle; hero ROUNDS rounds, arm order
# rotated each round; first 9 floats per slot compared to base on the box. Each ssh step < 5 min.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never
# bh-30, StrictHostKeyChecking=yes on every ssh, sync only through the repo-scoped one-way
# opt/sync_remote.sh inside this lock; remote scratch removed at the end.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} ROUNDS=${ROUNDS:-4} REPS=${REPS:-8}
O=${O:-docs/t785/out}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" ROUNDS="$ROUNDS" REPS="$REPS" "$PWD/docs/t785/ab.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t785 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_STATS=1 \
  && source .venv/bin/activate"
C=/localdev/smarton/.cache/ttmc-t785
ARMS="base lpt split"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t785 $C-base $C-lpt $C-split && mkdir -p tmp/t785 tmp/t785in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t785/build.log 2>&1; rc=\$?; tail -5 tmp/t785/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
echo "=== push inputs $(date +%T)"
for f in hero.npz s1.npz s2.npz; do  # plain ssh: the p150 hosts have no rsync
  timeout 900 "${SSH[@]}" "cat > $T/tmp/t785in/$f.tmp && mv $T/tmp/t785in/$f.tmp $T/tmp/t785in/$f" < tmp/t785in/$f
  echo "push $f rc=$?"
done
timeout 60 "${SSH[@]}" "cd $T/tmp/t785in && md5sum *.npz"
src_of() { case $1 in sh*) echo benchmarks/train_oracle/t0_${1}_64x64_n50.npz ;; *) echo tmp/t785in/$1.npz ;; esac; }
for n in sh0 sh3 s1 s2 hero; do
  timeout 900 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev prep $(src_of $n) tmp/t785in/base-$n > /dev/null \
    && python3 -m gsplat.t1_balance_dev plan tmp/t785in/base-$n tmp/t785in/lpt-$n | tee tmp/t785/plan-lpt-$n.json \
    && python3 -m gsplat.t1_balance_dev plan tmp/t785in/base-$n tmp/t785in/split-$n --split --kmax 2 | tee tmp/t785/plan-split-$n.json"
  echo "prep+plan $n rc=$?"
done
# run <arm> <dump> <outdir> <reps>; the split arm's slots are merged to the base slot count in place
run() {
  local mg=":"; [ "$1" = split ] && mg="python3 -m gsplat.t1_balance_dev merge tmp/t785in/split-$2 $3/slots.f32 $3/slots.m.f32 && mv $3/slots.f32 $3/slots.raw.f32 && mv $3/slots.m.f32 $3/slots.f32 && rm -f $3/slots.raw.f32"
  timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-$1 \
    && mkdir -p $3 && timeout 240 render/train/build/blend_bwd_bench tmp/t785in/$1-$2 0 $4 $3 > $3/run.log 2>&1; rc=\$?; \
    tail -2 $3/run.log; if [ \$rc = 0 ]; then $mg; fi; md5sum $3/slots.f32 | tee $3/slots.md5; \
    if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > $3/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  echo "run $1 $2 -> $3 rc=$?"
}
# same9 <dir a> <dir b>: the 9 output floats per slot, bit for bit (else count and max abs diff)
same9() {
  timeout 240 "${SSH[@]}" "$ENV && python3 -c \"import numpy as np; a=np.fromfile('$1/slots.f32', np.float32).reshape(-1, 16)[:, :9]; \
b=np.fromfile('$2/slots.f32', np.float32).reshape(-1, 16)[:, :9]; d=(a.view(np.uint32) != b.view(np.uint32)).any(1); \
print('same9 $1 $2 slots', len(a), 'identical' if not d.any() else 'DIFFER %d maxabs %.3g' % (int(d.sum()), float(np.abs(a - b).max())))\""
}
for a in $ARMS; do for n in sh0 sh3 s1 s2; do run $a $n tmp/t785/$a/$n 3; done; done
for n in sh0 sh3 s1 s2; do
  for a in lpt split; do same9 tmp/t785/base/$n tmp/t785/$a/$n; done
  for a in $ARMS; do
    timeout 240 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev check $(src_of $n) tmp/t785/$a/$n/slots.f32 > tmp/t785/check-$n-$a.json 2>&1; echo \"check $n $a rc=\$?\"; tail -c 300 tmp/t785/check-$n-$a.json"
  done
done
for r in $(seq 1 "$ROUNDS"); do
  case $((r % 3)) in 1) arms="base lpt split" ;; 2) arms="split base lpt" ;; 0) arms="lpt split base" ;; esac
  [ $((r % 2)) = 0 ] && arms=$(echo $arms | awk '{for (i = NF; i > 0; i--) printf "%s ", $i}')
  for a in $arms; do
    echo "=== round $r $a $(date +%T)"
    run $a hero tmp/t785/$a/hero-r$r "$REPS"
  done
  for a in lpt split; do same9 tmp/t785/base/hero-r$r tmp/t785/$a/hero-r$r; done
  # keep only the last split round's merged slots (286 MB) for the hero oracle check on the Mac
  for a in $ARMS; do
    [ "$a" = split ] && [ "$r" = "$ROUNDS" ] || timeout 60 "${SSH[@]}" "rm -f $T/tmp/t785/$a/hero-r$r/slots.f32"
  done
done
timeout 60 "${SSH[@]}" "for a in $ARMS; do find $C-\$a -path '*/kernels/*' -maxdepth 6 -type d > $T/tmp/t785/jit-dirs-\$a.txt; \
  find $C-\$a -path '*/kernels/*' -name '*.elf' | sort | xargs md5sum | sed \"s|^|\$a |;s|$C-\$a||\"; done > $T/tmp/t785/jit-elf-md5.txt; \
  wc -l $T/tmp/t785/jit-elf-md5.txt; md5sum $T/tmp/t785in/*/cores.u32 | tee $T/tmp/t785/cores-md5.txt; rm -rf $T/tmp/t785in $C-base $C-lpt $C-split"
echo "jit+rm rc=$?"
opt/sync_remote.sh --pull tmp/t785 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t785"; echo "rm t785 rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

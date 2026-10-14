#!/bin/bash
# t782/#787: BWD_SKIPDEAD A/B on p150 bh-49, all in ONE `ttp lock p100`.
#   ttp detach t782ab -- docs/t782/ab.sh <rev>      (Mac, worktree root; <rev> committed)
# From docs/t767/ab.sh. One source tree and one build; the arms differ only in env BWD_SKIPDEAD
# (base 0, cand 1; both BWD_PRESTAGE=2), which changes the T1 compute kernel's defines and CBs,
# each arm with its own TT_METAL_CACHE. Per-arm ELF md5s and kernel dir listings are recorded for
# opt/ab_jit_check.py.
# Part 1 (blend_bwd_bench, BWD_STATS=1): sh0 sh3 s1 s2 once per arm + device T0 oracle check;
# hero ROUNDS rounds ABBA, same9 base vs cand per round; hero oracle (--fast) on the last cand
# round, run in the background on the box while part 2 runs.
# Part 2 (render/train/build/train_step, T1 -> T2 chain): hero in-hero (t1 + t2 without the
# ref_in/ref_slots files) pushed once (tar|zstd), CHAIN_ROUNDS rounds ABBA; md5 of grads.f32 and
# the first 9 floats per slot; timing.json kept.
# Each ssh step is short (< 5 min) except the input push. Restrictions: existing measurement
# reservation only (no ird reserve/extend/release), never bh-30, StrictHostKeyChecking=yes on every
# ssh, sync only through the repo-scoped one-way opt/sync_remote.sh inside this lock, remote tmp
# deleted at the end.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
T=${T:-/localdev/smarton/gstt2-sync} ROUNDS=${ROUNDS:-4} REPS=${REPS:-8} CHAIN_ROUNDS=${CHAIN_ROUNDS:-2}
O=${O:-docs/t782/out} INHERO=${INHERO:-$TP/worktrees/t770/tmp/t750/in-hero}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" "$PWD/docs/t782/ab.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t782 BWD_SKIPDEAD A/B $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_STATS=1 BWD_PRESTAGE=2 \
  && source .venv/bin/activate"
C=/localdev/smarton/.cache/ttmc-t782
cleanup() { timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t782in $T/tmp/t782/chain-in $C-base $C-cand"; echo "remote cleanup rc=$?"; }
echo "=== card + disk"; timeout 60 "${SSH[@]}" "tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'; df -h /localdev | tail -1"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t782 $C-base $C-cand && mkdir -p tmp/t782 tmp/t782in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t782/build.log 2>&1; rc=\$?; tail -5 tmp/t782/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
echo "=== push inputs $(date +%T)"
for f in hero.npz s1.npz s2.npz; do  # plain ssh: the p150 hosts have no rsync
  timeout 900 "${SSH[@]}" "cat > $T/tmp/t782in/$f.tmp && mv $T/tmp/t782in/$f.tmp $T/tmp/t782in/$f" < tmp/t782in/$f
  echo "push $f rc=$?"
done
timeout 60 "${SSH[@]}" "cd $T/tmp/t782in && md5sum *.npz"
src_of() { case $1 in sh*) echo benchmarks/train_oracle/t0_${1}_64x64_n50.npz ;; *) echo tmp/t782in/$1.npz ;; esac; }
for n in sh0 sh3 s1 s2 hero; do
  timeout 900 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev prep $(src_of $n) tmp/t782in/$n > /dev/null"; echo "prep $n rc=$?"
done
sd_of() { [ "$1" = base ] && echo 0 || echo 1; }
# run <arm> <dump> <outdir> <reps>
run() {
  timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-$1 BWD_SKIPDEAD=$(sd_of $1) \
    && mkdir -p $3 && timeout 240 render/train/build/blend_bwd_bench tmp/t782in/$2 0 $4 $3 > $3/run.log 2>&1; rc=\$?; \
    tail -3 $3/run.log; md5sum $3/slots.f32 | tee $3/slots.md5; \
    if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > $3/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  echo "run $1 $2 -> $3 rc=$?"
}
# same9 <dir a> <dir b>: the 9 output floats per slot, bit for bit
same9() {
  timeout 240 "${SSH[@]}" "$ENV && python3 -c \"import numpy as np; a=np.fromfile('$1/slots.f32', np.uint32).reshape(-1, 16)[:, :9]; \
b=np.fromfile('$2/slots.f32', np.uint32).reshape(-1, 16)[:, :9]; print('same9 $1 $2 slots', len(a), 'identical' if (a == b).all() else 'DIFFER %d' % int((a != b).any(1).sum()))\""
}
for a in base cand; do for n in sh0 sh3 s1 s2; do run $a $n tmp/t782/$a/$n 3; done; done
for n in sh0 sh3 s1 s2; do
  same9 tmp/t782/base/$n tmp/t782/cand/$n
  for a in base cand; do
    timeout 240 "${SSH[@]}" "$ENV && python3 -m gsplat.blend_backward_dev check $(src_of $n) tmp/t782/$a/$n/slots.f32 > tmp/t782/check-$n-$a.json 2>&1; echo \"check $n $a rc=\$?\"; tail -c 300 tmp/t782/check-$n-$a.json"
  done
done
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then arms="base cand"; else arms="cand base"; fi
  for a in $arms; do
    echo "=== round $r $a $(date +%T)"
    run $a hero tmp/t782/$a/hero-r$r "$REPS"
  done
  same9 tmp/t782/base/hero-r$r tmp/t782/cand/hero-r$r
  for a in base cand; do
    [ "$a" = cand ] && [ "$r" = "$ROUNDS" ] || timeout 60 "${SSH[@]}" "rm -f $T/tmp/t782/$a/hero-r$r/slots.f32"
  done
done
# hero T0 oracle on the last cand round, CPU only, in the background while the chain runs
timeout 60 "${SSH[@]}" "$ENV && (setsid nohup sh -c 'nice -n 19 timeout 2400 python3 -m gsplat.blend_backward_dev check --fast tmp/t782in/hero.npz tmp/t782/cand/hero-r$ROUNDS/slots.f32 > tmp/t782/check-hero-cand.json 2> tmp/t782/check-hero-cand.log; echo \$? > tmp/t782/check-hero-cand.rc' > /dev/null 2>&1 < /dev/null &)"
echo "hero check started rc=$? $(date +%T)"

echo "=== chain push $(date +%T)"
t0=$(date +%s)
if [ -s "$INHERO/chain.json" ]; then
  COPYFILE_DISABLE=1 tar -C "$INHERO" --no-xattrs --exclude ref_in.npz --exclude ref_slots.npz -cf - chain.json t1 t2 | zstd -3 -T4 -q \
    | timeout 1800 "${SSH[@]}" "rm -rf $T/tmp/t782/chain-in && mkdir -p $T/tmp/t782/chain-in && zstd -dc | tar -C $T/tmp/t782/chain-in -xf - && du -sh $T/tmp/t782/chain-in"
  prc=$?
else
  echo "no $INHERO/chain.json"; prc=9
fi
echo "push chain rc=$prc $(( $(date +%s) - t0 ))s"
if [ $prc = 0 ]; then
  for r in $(seq 1 "$CHAIN_ROUNDS"); do
    if [ $((r % 2)) = 1 ]; then arms="base cand"; else arms="cand base"; fi
    for a in $arms; do
      d=tmp/t782/chain-$a-r$r
      echo "=== chain round $r $a $(date +%T)"
      timeout 900 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-$a BWD_SKIPDEAD=$(sd_of $a) && unset BWD_STATS && mkdir -p $d \
        && timeout 600 render/train/build/train_step tmp/t782/chain-in $d 110 5 grads > $d/run.log 2>&1; rc=\$?; grep '^\[STEP\]' $d/run.log; \
        if [ \$rc != 0 ]; then tail -20 $d/run.log; echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > $d/reset.log 2>&1; echo \"reset rc=\$?\"; exit \$rc; fi; \
        md5sum $d/grads.f32 $d/params_out.f32 $d/m_out.u16 $d/v_out.u16 | tee $d/md5.txt; \
        python3 -c \"import numpy as np, hashlib; s=np.fromfile('$d/slots.f32', np.uint32).reshape(-1, 16)[:, :9]; print('slots9 md5', hashlib.md5(np.ascontiguousarray(s).tobytes()).hexdigest())\" | tee -a $d/md5.txt; \
        rm -f $d/*.f32 $d/*.u16; exit 0"
      echo "chain $a r$r rc=$?"
    done
  done
fi
echo "=== hero check wait $(date +%T)"
for i in $(seq 1 120); do
  st=$(timeout 60 "${SSH[@]}" "cat $T/tmp/t782/check-hero-cand.rc 2>/dev/null")
  [ -n "$st" ] && break
  sleep 20
done
echo "hero check rc=${st:-none} $(date +%T)"
timeout 60 "${SSH[@]}" "tail -c 600 $T/tmp/t782/check-hero-cand.json; tail -5 $T/tmp/t782/check-hero-cand.log; rm -f $T/tmp/t782/cand/hero-r$ROUNDS/slots.f32"
timeout 60 "${SSH[@]}" "for a in base cand; do find $C-\$a -path '*/kernels/*' -name '*.elf' | sort | xargs md5sum | sed \"s|^|\$a |\"; \
  find $C-\$a -path '*/kernels/*' -type d > $T/tmp/t782/jit-dirs-\$a.txt; done > $T/tmp/t782/jit-elf-md5.txt; grep -c . $T/tmp/t782/jit-elf-md5.txt"
echo "jit rc=$?"
cleanup
opt/sync_remote.sh --pull tmp/t782 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t782; df -h /localdev | tail -1"; echo "rm t782 rc=$?"
echo "=== done $(TZ=America/Los_Angeles date)"

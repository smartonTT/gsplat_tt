#!/bin/bash
# t780: reader-side feasibility probe on p150 bh-49, all in ONE `ttp lock p100`.
#   ttp detach t780probe -- docs/t780/probe.sh <rev>      (Mac, worktree root; <rev> committed)
# Hero dump (#728's, tmp/t780in/hero.npz) at BWD_PRESTAGE=2, BWD_STATS=1; arm off = BWD_SPROBE=0,
# arm on = BWD_SPROBE=1 (the reader also writes the 12 pair floats replicated to 32 words each,
# 1.5 KB per pair, as an unpack-to-Dst staging needs). Own TT_METAL_CACHE per arm, ABBA rounds.
# Also saves the Blackhole LLK unpack-to-dest sources (read only) for the design.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never bh-30,
# StrictHostKeyChecking=yes on every ssh, sync only through the repo-scoped one-way opt/sync_remote.sh
# inside this lock.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} REPS=${REPS:-4} ROUNDS=${ROUNDS:-2} O=${O:-docs/t780/out-probe}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" "$PWD/docs/t780/probe.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t780 probe $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=60 BWD_PRESTAGE=2 BWD_STATS=1 \
  && source .venv/bin/activate"
C=/localdev/smarton/.cache/ttmc-t780
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t780 $C-off $C-on && mkdir -p tmp/t780 tmp/t780in && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t780/build.log 2>&1; rc=\$?; tail -5 tmp/t780/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"; [ $brc = 0 ] || exit 4
L=/localdev/smarton/tt-metal/tt_metal/tt-llk/tt_llk_blackhole
timeout 120 "${SSH[@]}" "cd $L 2>/dev/null || cd /localdev/smarton/tt-metal/tt_metal/third_party/tt_llk/tt_llk_blackhole; pwd; \
  for f in llk_lib/llk_unpack_A.h llk_lib/llk_math_eltwise_unary_datacopy.h common/inc/cunpack_common.h common/inc/cmath_common.h common/inc/ckernel_globals.h; do echo \"##### \$f\"; cat \$f; done" > "$O/llk.txt" 2>&1
echo "llk rc=$? $(wc -l < "$O/llk.txt") lines"
echo "=== push hero $(date +%T)"
timeout 900 "${SSH[@]}" "cat > $T/tmp/t780in/hero.npz.tmp && mv $T/tmp/t780in/hero.npz.tmp $T/tmp/t780in/hero.npz" < tmp/t780in/hero.npz
echo "push rc=$?"
timeout 900 "${SSH[@]}" "$ENV && md5sum tmp/t780in/hero.npz && python3 -m gsplat.blend_backward_dev prep tmp/t780in/hero.npz tmp/t780in/hero > /dev/null"; echo "prep rc=$?"
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then arms="off on"; else arms="on off"; fi
  for a in $arms; do
    sp=0; [ $a = on ] && sp=1
    echo "=== round $r $a $(date +%T)"
    timeout 600 "${SSH[@]}" "$ENV && export TT_METAL_CACHE=$C-$a BWD_SPROBE=$sp && d=tmp/t780/$a-r$r && mkdir -p \$d \
      && timeout 240 render/train/build/blend_bwd_bench tmp/t780in/hero 0 $REPS \$d > \$d/run.log 2>&1; rc=\$?; \
      tail -2 \$d/run.log; python3 -c \"import numpy as np; print('slots9 md5', __import__('hashlib').md5(np.fromfile('\$d/slots.f32', np.uint32).reshape(-1, 16)[:, :9].tobytes()).hexdigest())\"; rm -f \$d/slots.f32; \
      if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > \$d/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
    echo "run $a r$r rc=$?"
  done
done
timeout 60 "${SSH[@]}" "rm -rf $T/tmp/t780in"; echo "rm in rc=$?"
opt/sync_remote.sh --pull tmp/t780 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t780"; echo "rm t780 rc=$?"
for f in "$O"/remote/*/stats.json; do python3 docs/t780/stats.py "$f"; done
grep -h median "$O"/remote/*/timing.json 2>/dev/null | head; for f in "$O"/remote/*/timing.json; do echo "$f $(cat "$f")"; done
echo "=== done $(TZ=America/Los_Angeles date)"

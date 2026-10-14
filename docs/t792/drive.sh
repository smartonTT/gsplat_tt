#!/bin/bash
# t792: T2 split default on, kcap 8 vs 16 over 4 rounds, and the kept-iteration evidence, all in
# ONE `ttp lock p100`:
#   ttp detach t792 -- docs/t792/drive.sh <rev> <iter>     (Mac, worktree root; <rev> committed)
# <rev> has T2_SPLIT=1 and T2_KCAP=8 as the train_step defaults. On the box (bh-49 p150a):
#   sync  opt/sync_remote.sh (one-way, repo-scoped, this worktree; builds the renderer), then train_step
#   hero  push the hero T2 prep (tmp/t750 of the t770 worktree, HERO_SRC=; derived arrays only)
#   same  base (T2_SPLIT=0) vs defaults: every trainer output bit-equal (docs/t784/check_g2d.py same)
#         + gsplat.train_step_dev check (hero gradients) on the defaults arm
#   ab    ROUNDS (4) ABBA rounds of k8 (defaults) / k16 (T2_KCAP=16), REPS (10) steps each
#   t0    sh0/sh3 T0 oracle crops at the defaults and at T2_KCAP=1 (every row heavy)
#   jit   opt/ab_jit_check.py: k8 vs k16 --runtime-arg-only (same kernels, timing.json kcap markers),
#         base vs defaults (t2g_reader must differ: the T2_SPLIT define)
#   shot  opt/ttw/screenshot_remote.sh: device hero + 30-view md5 sweep -> opt/metal-screenshots/ttw-<iter>
#   b2b   renderer b2b on the orbit (cameras_v2, 20 passes) x3 + one latency run (no --back-to-back)
# then pull the results and delete the hero data on the box (keep disk lean).
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never the
# viewer box bh-30, StrictHostKeyChecking=yes on every ssh/scp, nothing synced from outside the repo
# except the hero T2 prep (derived arrays, no dataset images).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; N=${2:?iter}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
T=${T:-/localdev/smarton/gstt2-sync}
HERO_SRC=${HERO_SRC:-$TP/worktrees/t770/tmp/t750}
[ -n "$H" ] || { echo "=== no p100 host"; exit 2; }
D=docs/t792; O=$D/out
if [ "${3:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" "$N" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
SSH=(ssh "${SO[@]}" "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t792 $rev=$sha iter $N to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log" | tail -4
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
[ "$rsha" = "$sha" ] || exit 3
ENV0="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} && source .venv/bin/activate"
C=/localdev/smarton/.cache
CB=$C/ttmc-t792-base; C8=$C/ttmc-t792-k8; C16=$C/ttmc-t792-k16
echo "=== card / disk"; timeout 60 "${SSH[@]}" "$ENV0 && tt-smi -ls 2>/dev/null | grep -i -m3 'p150\|p100\|blackhole'; df -h /localdev | tail -1; rm -rf $C/ttmc-t792-* $T/tmp/t792"
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV0 && mkdir -p tmp/t792/res && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8 --target train_step) > tmp/t792/res/build.log 2>&1; rc=\$?; tail -5 tmp/t792/res/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
[ $brc = 0 ] || { opt/sync_remote.sh --pull tmp/t792/res "$O/remote" "$H" "$T"; exit 4; }
fail=0

# run_hero <name> <cache> <reps> [env...]: one train_step process on the hero; keeps the outputs in
# tmp/t792/hero/out-<name> (deleted by the caller), copies timing/log to tmp/t792/res/<name>.
run_hero() {
  local n=$1 c=$2 r=$3; shift 3
  echo "=== hero $n (${*:-defaults}) $(date +%T)"
  timeout 1200 "${SSH[@]}" "$ENV0 && unset T2_SPLIT T2_KCAP T2_HL T2_OVH T2_G2D && export TT_METAL_CACHE=$c $* && cd tmp/t792/hero && rm -rf out-$n && mkdir -p out-$n ../res/$n \
    && timeout ${RUN_TIMEOUT:-600} ../../../render/train/build/train_step in-hero out-$n ${CORES:-110} $r grads > out-$n/run.log 2>&1; rc=\$?; \
    grep -E 'error|Error' out-$n/run.log | head -5; cp out-$n/run.log out-$n/timing.json ../res/$n/ 2>/dev/null; cat out-$n/timing.json 2>/dev/null; echo; \
    if [ \$rc != 0 ]; then tail -20 out-$n/run.log; echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > ../res/$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
  local rc=$?; echo "hero $n rc=$rc"; [ $rc = 0 ] || fail=1; return $rc
}
rmout() { timeout 120 "${SSH[@]}" "cd $T/tmp/t792/hero && rm -rf $(printf 'out-%s ' "$@")"; }
box() { timeout ${2:-1800} "${SSH[@]}" "$ENV0 && unset T2_SPLIT T2_KCAP T2_HL T2_OVH T2_G2D && $1"; }

[ -s "$HERO_SRC/in-hero/t2/meta.u32" ] || { echo "no hero prep at $HERO_SRC"; exit 5; }
t0=$(date +%s)
COPYFILE_DISABLE=1 tar -C "$HERO_SRC" --no-xattrs -cf - hero_check.npz in-hero | zstd -3 -T4 -q \
  | timeout 2400 "${SSH[@]}" "rm -rf $T/tmp/t792/hero && mkdir -p $T/tmp/t792/hero && zstd -dc | tar -C $T/tmp/t792/hero -xf -"
prc=$?; echo "push hero rc=$prc $(( $(date +%s) - t0 ))s"
if [ $prc = 0 ]; then
  run_hero base0 $CB 3 T2_SPLIT=0; run_hero def0 $C8 3
  box "python3 docs/t784/check_g2d.py same tmp/t792/hero/in-hero tmp/t792/hero/out-base0 tmp/t792/hero/out-def0 | tee tmp/t792/res/same.json; echo \"SAME rc=\${PIPESTATUS[0]}\"" || fail=1
  box "nice -n 19 python3 -m gsplat.train_step_dev check tmp/t792/hero/hero_check.npz tmp/t792/hero/in-hero tmp/t792/hero/out-def0 > tmp/t792/res/check-def0.json 2> tmp/t792/res/check-def0.log; echo \"CHECK def0 rc=\$?\"" 2400 \
    || fail=1
  rmout base0 def0
  for r in $(seq 1 ${ROUNDS:-4}); do
    if [ $((r % 2)) = 1 ]; then o="k8 k16"; else o="k16 k8"; fi
    for a in $o; do
      if [ $a = k8 ]; then run_hero ab$r-k8 $C8 ${REPS:-10}; else run_hero ab$r-k16 $C16 ${REPS:-10} T2_KCAP=16; fi
      rmout ab$r-$a
    done
  done
else fail=1; fi
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t792/hero"; echo "hero dir removed rc=$?"
for n in sh0 sh3; do
  for k in def 1; do
    if [ $k = def ]; then e=""; else e="export T2_KCAP=$k &&"; fi
    echo "=== t0 $n kcap $k $(date +%T)"
    box "export TT_METAL_CACHE=$C8 && $e rm -rf tmp/t792/in-$n tmp/t792/out-$n-k$k && mkdir -p tmp/t792/out-$n-k$k \
      && python3 -m gsplat.train_step_dev prep benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t792/in-$n \
      && timeout 300 render/train/build/train_step tmp/t792/in-$n tmp/t792/out-$n-k$k ${CORES:-110} 3 grads > tmp/t792/out-$n-k$k/run.log 2>&1 \
      && cp tmp/t792/out-$n-k$k/timing.json tmp/t792/res/t0-$n-k$k.timing.json; \
      python3 -m gsplat.train_step_dev check benchmarks/train_oracle/t0_${n}_64x64_n50.npz tmp/t792/in-$n tmp/t792/out-$n-k$k > tmp/t792/res/check-$n-k$k.json; \
      echo \"CHECK $n k$k rc=\$?\"; python3 -c \"import json;d=json.load(open('tmp/t792/res/check-$n-k$k.json'));print({k:v.get('pass') if isinstance(v,dict) else v for k,v in d.items() if k in ('t1','t2','t0','pass')})\"" 900
  done
done
timeout 60 "${SSH[@]}" "rm -rf $T/tmp/t792/in-sh0 $T/tmp/t792/in-sh3 $T/tmp/t792/out-sh*"
echo "=== jit"
box "cd tmp/t792/res && L8=\$(for r in \$(seq 1 ${ROUNDS:-4}); do printf -- '--log k8=ab%s-k8/timing.json ' \$r; done) \
  && L16=\$(for r in \$(seq 1 ${ROUNDS:-4}); do printf -- '--log k16=ab%s-k16/timing.json ' \$r; done) \
  && python3 ../../../opt/ab_jit_check.py --runtime-arg-only --arm k8=$C8 --arm k16=$C16 --kernel t2g_reader --kernel t2g_compute \
     --marker k8='\"kcap\": 8,' --marker k16='\"kcap\": 16,' \$L8 \$L16 | tee jit-kcap.txt; echo \"JIT kcap rc=\${PIPESTATUS[0]}\"; \
  python3 ../../../opt/ab_jit_check.py --arm base=$CB --arm def=$C8 --kernel t2g_reader | tee jit-split.txt; echo \"JIT split rc=\${PIPESTATUS[0]}\"" 120 || fail=1

echo "=== shot $(TZ=America/Los_Angeles date +%T)"
SH=opt/metal-screenshots/ttw-$N; mkdir -p "$SH" "$O/shot"; tag=ttw-$N-hero
scp -q "${SO[@]}" opt/ttw/screenshot_remote.sh "$H:$T/tmp/screenshot_remote.sh.tmp" && timeout 60 "${SSH[@]}" "mv $T/tmp/screenshot_remote.sh.tmp $T/tmp/screenshot_remote.sh" || exit 6
timeout 560 "${SSH[@]}" "bash $T/tmp/screenshot_remote.sh $T $tag" > "$O/shot/run.out" 2>&1; s=$?
cat "$O/shot/run.out"; echo "shot rc=$s"
if [ $s = 0 ] && scp -q "${SO[@]}" "$H:$T/tmp/shot/hero-$tag.png" "$SH/hero.png" \
   && scp -q "${SO[@]}" "$H:$T/tmp/shot/md5-$tag.txt" "$H:$T/tmp/shot/run-$tag.log" "$O/shot/"; then
python3 - "$SH/hero.png" "$SH/hero_diff10.png" "$O/shot/md5-$tag.txt" "$(git rev-parse --short "$rev")" <<'PY'
import hashlib, sys
import numpy as np
from PIL import Image
hero, diff, md5f, sha = sys.argv[1:5]
def rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return "inf" if m == 0 else round(float(10 * np.log10(255.0 ** 2 / m)), 2)
h = rgb(hero)
gold = rgb("tests/fixtures/hero/hero_golden_8bit.png")
ref = rgb("benchmarks/reference_v2/hero.png")
Image.fromarray(np.clip(np.abs(h - ref) * 10, 0, 255).astype(np.uint8)).save(diff)
sweep = hashlib.md5(open(md5f, "rb").read()).hexdigest()[:8]
print(f"SHOT hero={hero} diff={diff} commit={sha} env='defaults' size={h.shape[1]}x{h.shape[0]} "
      f"sweep_md5={sweep} ref=benchmarks/reference_v2/hero.png psnr_vs_ref={psnr(h, ref)} "
      f"golden_match={str(bool(np.array_equal(h, gold))).lower()} max_lsb_vs_golden={int(np.abs(h - gold).max())}")
PY
else echo "SHOT_FAIL"; fail=1; fi

echo "=== b2b $(TZ=America/Los_Angeles date +%T)"
for r in 1 2 3 lat; do
  if [ $r = lat ]; then a=""; else a="--back-to-back"; fi
  timeout 300 "${SSH[@]}" "$ENV0 && export MESH_DEVICE=P150 TTW_DEVRUN=1 GSPLAT_TT_ETH_CACHE=$C/ttmc-t792-render \
    && timeout 240 python3 render/run.py --no-ref --cameras benchmarks/cameras_v2.json --iter-dir t792-b2b-$r $a > tmp/t792/res/b2b-$r.log 2>&1; rc=\$?; \
    grep -E '^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:' tmp/t792/res/b2b-$r.log | cut -c1-300 | head -8; \
    if [ \$rc = 124 ]; then tt-smi -r > tmp/t792/res/reset-b2b.log 2>&1; fi; exit \$rc"
  s=$?; echo "b2b $r rc=$s"; [ $s = 0 ] || fail=1
done
opt/sync_remote.sh --pull tmp/t792/res "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 "${SSH[@]}" "rm -rf $C/ttmc-t792-* $T/tmp/t792-b2b-* $T/tmp/shot-$tag; df -h /localdev | tail -1" || true
echo "=== done fail=$fail $(TZ=America/Los_Angeles date)"
exit $fail

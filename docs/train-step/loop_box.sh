#!/bin/bash
# #750 step 2, runs ON the box from the sync dir (venv + TT env set by drive_loop.sh): one
# dump-free training step chain on dump <npz>. Device stages: blend_fwd_bench (train-mode fp32
# forward) -> loss_bench LOSS_MODE=ssim (fused L1 + D-SSIM + dL/dimage) -> train_step (T1 -> T2 +
# Adam). Host glue between them (gsplat.train_loop_dev, train_step_dev prep) is timed per step:
#   STEP <name> <step> rc= wall_ms=      (wall_ms includes python start-up; device ms in timing.json)
#   docs/train-step/loop_box.sh <name> <npz> [reps]
# HERO=1 (#781, full 1237x822 view): forward prep binned to 3-sigma tile rectangles (the hero
# dump's forward_fast rule, prep_binned), forward check against the dump image without the numpy
# model (minutes at hero size), outputs under LOOP_DIR (default tmp/t750) and the big files
# (records, chain npz, T1/T2 inputs and outputs) deleted at the end, json and logs kept.
set -u
n=$1 P=$2 REPS=${3:-5} D=${LOOP_DIR:-tmp/t750}/loop-$1
FB=; [ "${HERO:-0}" = 1 ] && FB="--binned"
FC=; [ "${HERO:-0}" = 1 ] && FC="--binned --no-model"
rm -rf "$D"; mkdir -p "$D/fwd" "$D/loss/out" "$D/in" "$D/out"
step() {  # step <fatal 0|1> <dev 0|1> <name> <cmd>: a failed device stage resets the card
  local fatal=$1 dev=$2 s=$3; shift 3; local t; t=$(date +%s%N)
  bash -c "$*"; local rc=$?
  echo "STEP $n $s rc=$rc wall_ms=$(( ($(date +%s%N) - t) / 1000000 ))"
  if [ $rc != 0 ] && [ "$dev" = 1 ]; then tail -20 "$D"/*/run.log 2>/dev/null; tt-smi -r > "$D/reset.log" 2>&1; echo "reset rc=$?"; fi
  if [ $rc != 0 ] && [ "$fatal" = 1 ]; then exit $rc; fi
}
step 1 0 fwd_prep "python3 -m gsplat.blend_forward_dev prep $FB $P $D/fwd > $D/fwd/prep.log 2>&1"
step 1 1 fwd_dev "timeout 300 render/train/build/blend_fwd_bench $D/fwd $REPS > $D/fwd/run.log 2>&1"
step 0 0 fwd_check "python3 -m gsplat.blend_forward_dev check $FC $P $D/fwd --image-out $D/image.f32 > $D/fwd/check.log 2>&1"
step 1 0 loss_prep "python3 -m gsplat.train_loop_dev loss-prep $P $D/image.f32 $D/loss > $D/loss/prep.json"
step 1 1 loss_dev "LOSS_MODE=ssim timeout 300 render/bench/build/loss_bench $D/loss $D/loss/out $REPS > $D/loss/run.log 2>&1"
step 1 0 chain_dump "python3 -m gsplat.train_loop_dev chain-dump $P $D/loss $D/chain.npz > $D/chain.json"
step 1 0 t1t2_prep "python3 -m gsplat.train_step_dev prep $D/chain.npz $D/in > $D/in/prep.json"
step 1 1 step_dev "timeout 300 render/train/build/train_step $D/in $D/out ${CORES:-110} $REPS grads > $D/out/run.log 2>&1"
step 0 0 check "python3 -m gsplat.train_step_dev check $D/chain.npz $D/in $D/out > $D/out/check.log 2>&1"
echo "--- $n fwd"; tail -3 "$D/fwd/check.log"; cat "$D/fwd/timing.json" 2>/dev/null; echo
echo "--- $n loss"; grep '^\[LOSS\]' "$D/loss/run.log"; cat "$D/loss/prep.json" "$D/chain.json"
echo "--- $n step"; cat "$D/out/timing.json" 2>/dev/null; echo
python3 -c "import json;r=json.load(open('$D/out/check.json'));print('CHAIN $n pass=%s t1=%s t2=%s t0=%s' % (r['pass'], r['t1']['pass'], r['t2']['pass'], r.get('t0', {}).get('pass')))"
rm -f "$D/chain.npz"
if [ "${HERO:-0}" = 1 ]; then
  rm -f "$D"/fwd/*.u32 "$D"/fwd/*.f32 "$D"/loss/*.f32 "$D"/loss/out/*.f32 "$D"/image.f32
  find "$D/in" "$D/out" -type f ! -name '*.json' ! -name '*.log' -delete
  du -sh "$D"
fi

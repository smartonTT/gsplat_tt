#!/bin/bash
# Task #580: run model.py on the three t260 input sets in parallel (CPU only, ~20 min).
# Inputs: docs/precision-model/out/model_in_{cpu,dev,cpu_devset}.npz (copied from the t260
# worktree, gitignored). Writes out/model_<set>/model_psnr.json + model_images.npz.
set -eo pipefail
cd "$(dirname "$0")/../.."
pids=()
for k in cpu dev cpu_devset; do
  PM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/precision-model/model.py \
    docs/precision-model/out/model_$k docs/precision-model/out/model_in_$k.npz \
    > docs/precision-model/out/log_$k.txt 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit $rc

#!/bin/bash
# t510 stage 0: CPU model, hero with all variants, then 30 views in 3 parallel chunks.
# Usage: run_cpu.sh <out dir>   (from the worktree root; writes <out>/DONE with the exit code)
set -eo pipefail
O=${1:?out dir}; mkdir -p "$O"
trap 'echo $? > "$O/DONE"' EXIT
M=docs/blend-qf-t510/qfmodel.py
python3 $M --views hero --out "$O" --threads 8 > "$O/hero.log" 2>&1
echo "hero rc 0"
V=base,tf32_center,tf32_center_hilo,tf32_mb,bf16_mb_hilo
pids=()
for r in 1:11 11:21 21:30; do
  python3 $M --views $r --out "$O" --variants $V --threads 4 > "$O/views-${r/:/_}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
echo "views rc $rc"; exit $rc

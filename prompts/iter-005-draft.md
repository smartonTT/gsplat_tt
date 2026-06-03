# iter-005 worker prompt (draft)

You are the iter-005 worker. iter-004 just landed the C++ alpha_blend — the
full forward pipeline now runs in C++ end-to-end. **iter-005 is a benchmark
iter: no algorithm change, no code change. Lock in the stable C++ baseline
numbers that all future algorithmic iters will be measured against.**

## Read first

1. `/Users/smarton/dev/gstt2/opt/plan.md` — iter-005 is the Phase 2 capstone
2. `/Users/smarton/dev/gstt2/prompts/worker.md` — the rules
3. `/Users/smarton/dev/gstt2/opt/iters.jsonl` — previous iters' numbers (the prev-best is 3253 ms from iter-004)
4. `/Users/smarton/dev/gstt2/scripts/render_30frame.py` — the harness

## Iter spec

ITER: 005-cpp-perf-baseline
GOAL: Run the 30-view benchmark **five times** end-to-end on the current C++ pipeline (no edits to src/, no edits to backends/) and produce a stable baseline tuple `(sum_total_ms_median, blend_median_ms, project_median_ms, tile_assign_median_ms, sort_median_ms)`. This number becomes the gate for every future algorithmic iter ("if your change can't beat baseline-cpp on sum_total_ms by ≥ 2%, it's no_commit_valid_but_not_faster").

## What you do

1. **No code changes.** This iter only produces measurement artifacts. Do not edit any file outside `opt/screenshots/iter-005-perf-baseline/`.

2. **Run the benchmark 5 times** with `warmup=1`. The Mac's thermal/clock state stabilizes after the first run; you want runs 2-5 to capture steady-state. Save EACH run's timing.jsonl as `run_N.timing.jsonl` for N in 1..5 in the iter dir:

```bash
source scripts/_env.sh
ITER_DIR=opt/screenshots/iter-005-perf-baseline
mkdir -p "$ITER_DIR"
for N in 1 2 3 4 5; do
  RUN_DIR=/tmp/iter005_run_$N
  rm -rf "$RUN_DIR"
  "$LOCAL_PY" scripts/render_30frame.py --backend cpu_cpp \
    --cameras benchmarks/cameras_v2.json --out-dir "$RUN_DIR" --warmup 1 \
    2>&1 | tail -3
  cp "$RUN_DIR"/timing.jsonl "$ITER_DIR/run_${N}.timing.jsonl"
done
# Use run 3's PNGs as the iter-005 artifacts (any of 2-5 would be fine; pick middle one).
cp /tmp/iter005_run_3/*.png "$ITER_DIR/"
cp /tmp/iter005_run_3/timing.jsonl "$ITER_DIR/timing.jsonl"
```

3. **Produce a baseline summary.** Compute these statistics across runs 2-5 (drop run 1 as warmup):
   - `sum_total_ms`: median across the 4 runs of `sum(row.total_ms for row in run)`
   - `blend_median_ms`: median per-view blend time, across all 30 views × 4 runs (120 samples)
   - `project_median_ms`, `tile_assign_median_ms`, `sort_median_ms`: same
   - `blend_p95_ms`, `total_p95_ms`: 95th percentile of per-view timings
   - `per_view_median_total_ms`: for each view name, median total_ms across runs

   Write to `$ITER_DIR/baseline.json`:

```json
{
  "runs": 5,
  "warmup_runs_dropped": 1,
  "sum_total_ms_median": <float>,
  "sum_total_ms_min": <float>,
  "sum_total_ms_max": <float>,
  "stage_median_ms": {
    "project": <float>, "tile_assign": <float>, "sort": <float>, "blend": <float>
  },
  "stage_p95_ms": { ... },
  "per_view_median_total_ms": { "hero": <float>, "orbit_010": <float>, ... },
  "git_sha": "<HEAD sha>"
}
```

4. **Run compute_metrics + emit a validator verdict.** The 30 PNGs from run 3 are bit-identical to iter-004's (because no code changed), so PSNR vs reference_v2 stays at min 95.05 dB. Write `validator.json` with verdict KEEP, reasoning "baseline lock-in, identical algorithm, PSNR unchanged".

5. **Do NOT call decide_and_log** — the supervisor will. Hand back the iter dir path.

## SUMMARY

End with this line:

```
SUMMARY: iter-005 status=PASS baseline_sum_total_ms_median=X.X blend_median_ms=Y.Y project_median_ms=Z.Z tile_assign_median_ms=W.W sort_median_ms=V.V min_psnr_dB=X.X
```

No code is committed. Begin.

# A1: cross-view coherence on the bicycle NN training tour (task #688)

Mac CPU only, no device. 169 training views of `benchmarks/cameras_train_nn.json` (NN tour),
1237x822, 32x32 tiles, 168 consecutive transitions. Visibility and tile pairs come from the
reference CPU pipeline (`gsplat/rasterization.py`: opacity-aware radius, per-pair contribution
cull), checked against an independent numpy version on view 0 (1,420,543 visible, 2,242,965
pairs, 0 differing pairs). Script: `coherence.py` + `segstats.c`; raw output: `results.json`
(per-transition rows + summaries). Runtime 1727 s, 3 workers.

Models: bicycle.ply (N = 5.56M after the opacity floor) and a 1.2M proxy (the 1.2M Gaussians
with the largest opacity x summed pair count over the train views; a shape proxy, not D1's
alpha*T pruning). All numbers are measured on CPU (M) unless marked as a model (E).

## L1: exact sort warm start (previous view's order as the start of an exact sort)

For each tile of view i+1, the pairs whose Gaussian was visible in view i are put in view i's
depth order; we measure how far that is from view i+1's exact (depth, id) order.

| metric (mean over 168 transitions, pair-weighted) | 5.56M | 1.2M proxy |
|---|---:|---:|
| pairs per view | 1.95M | 0.81M |
| pairs whose Gaussian was visible in view i (carried) | 69.9% | 73.2% |
| pairs that stay in the same tile | 1.9% | 2.6% |
| sorted run length of carried keys | 2.24 | 2.27 |
| inversions per carried key (p50 / p90) | 44 (38 / 79) | 27 (24 / 53) |
| keys an insertion fix-up must move (n - LIS) | 70% | 67% |
| tiles already fully sorted | 12.7% | 13.4% |
| merge passes on warm input, log2(runs), vs cold log2(n) | 10.5 vs 11.6 | 9.5 vs 10.7 |
| global depth order: run length / keys moved / median rank shift | 2.0 / 97% / 9.7k | 2.0 / 97% / 3.2k |

The previous view's order is close to random for the next view: consecutive training cameras
are far enough apart that depth order changes almost everywhere. Modelled saving (E):
- Insertion fix-up costs ~44 compare-moves per key, far more than our per-tile LSD radix
  (`render/kernels/dataflow/sort_radix_tile.cpp`), whose cost does not depend on input order.
- A run-aware merge saves at most 1.2 of 11.6 passes (~10%) against a cold merge sort, which is
  itself slower than the radix; 30% of the pairs are new and need sorting and merging anyway.
- Radix that skips sorted segments: runs are 2 keys long, nothing to skip.
- Upper bound: ~10% of the ~0.4 ms parity-N sort = **<= 0.04 ms/view, likely negative**.

**L1: NO-GO** (bar 0.1 ms/view at parity N).

## L2: exact batch-union pre-cull over K NN-tour views

|union of visible sets over K consecutive views| / |one view| (non-overlapping batches):

| K | 5.56M ratio (p90) | 5.56M union / N | 1.2M ratio (p90) | 1.2M union / N |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.221 | 1.00 | 0.365 |
| 2 | 1.31 (1.62) | 0.288 | 1.30 (1.59) | 0.464 |
| 4 | 1.66 (1.99) | 0.366 | 1.62 (2.03) | 0.575 |
| 8 | 1.93 (2.22) | 0.423 | 1.72 (2.05) | 0.620 |

The plan assumed ~1.2x; the measured union grows to 1.3-1.9x.

Cost model (E): pfwc walk time scales with rows walked; the compaction that builds the union is
at least one full-N walk, paid once per K views. Per view = pfwc(N) x (union/N + 1/K).
pfwc(N) = 1.38 ms at 5.56M (M, p100) and ~0.3 ms at 1.2M (plan §4, E). This ignores the cost of
gathering compacted rows (the u64 gather lost before, #631-#644), so it is an upper bound.

| K | saving at 5.56M | saving at 1.2M (parity N) |
|---:|---:|---:|
| 2 | 0.29 ms | 0.01 ms |
| 4 | 0.53 ms | 0.05 ms |
| 8 | 0.62 ms | 0.08 ms |

**L2: NO-GO at parity N** (best 0.08 ms < 0.1, before gather overhead). It clears the bar only
while the trainer runs on the full 5.56M model (0.5-0.6 ms at K = 4-8). Park it; reopen only if
M1 shows training at ~5.56M with pfwc exposed, and then it also needs the sparse-Adam (or
step-bound) exactness argument from the plan.

## Losslessness (msg #384)

Neither lever is built. L1 as specified was only ever a starting point for an exact sort checked
key for key; L2 culls exactly per view inside the union. No stale order or approximate cull was
measured or proposed.

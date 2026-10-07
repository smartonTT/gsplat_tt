# t335: no-device model of finer fill jobs and a big-tile split (iter 207)

Input: `opt/profiler/t319-fz/percore.csv` (t319 Tracy fill zones at iter 207 defaults,
yyzo-bh-04 p100a, per core mean over 29 views). Script: `model.py`, output: `out.txt`
(`python3 docs/fill-model-t335/model.py`). No device was used.

## What the data says first

- Six row-2 cores (1-2 … 6-2) are **pure big-tile cores**: their NCRISC mover runs one item, the
  >8192-record tile's subchunk (2.10-2.30 ms), then waits 0.24-0.30 ms for its 8192-record fill.
  This wait is the final flush with nothing left to overlap. The big item's gather loop is
  chunk-outer (`sort_subchunk_materialize.cpp`, `mat_ol_gather`), so its slab is complete only
  after the last pass: finer fill jobs cannot start early on it unless the gather is rewritten to
  be rank-ordered.
- On the other cores the longest single wait (`nc_dwm_us`, mean 289 µs) is most of the NCRISC
  done-wait (355 µs): it is one exposed slab fill per view, not many small stalls.
- BRISC cannot take a big item today: its buffers hold 6144 records (`kMatMover0Cap`), and
  `build_mat_worklist` pins big items to NCRISC slots.

## Results (mat makespan = max over cores of the per-core mean end; base 2.675 ms)

Frame = model gain × 0.40 (#306: 0.216 ms measured in mat+blend / 0.56 ms modelled); × 0.28
(#306's frame / model) is the low case.

| variant | mat ms | mat gain | frame ×0.40 | frame ×0.28 |
|---|---:|---:|---:|---:|
| t319 bound: every done-wait removed | 2.372 | 0.303 | 0.121 | 0.085 |
| (1) 512-record jobs, per-mover queues, streamed emit, gather as today | 2.678 | −0.003 | 0.00 | 0.00 |
| (1) + rank-ordered big gather, 1 chunk residual wait | 2.402 | 0.273 | 0.109 | 0.076 |
| (1) + rank-ordered big gather, 2 chunks residual | 2.428 | 0.246 | 0.099 | 0.069 |
| (2) big item split over both movers + host re-LPT, ideal (no dup, imbalance 0.061) | 2.271 | 0.403 | 0.161 | 0.113 |
| (2) realistic (15 % duplicated work/sync, imbalance 0.10) | 2.325 | 0.350 | 0.140 | 0.098 |
| (1)+(2) ceiling (1 chunk, ideal split, rank-ordered gather) | 1.943 | 0.731 | 0.293 | 0.205 |
| **(1)+(2) realistic (2 chunks, 15 % dup, imbalance 0.10)** | **2.025** | **0.650** | **0.260** | **0.182** |
| absolute ceiling: all mover work perfectly balanced, zero waits | 1.854 | 0.821 | 0.328 | 0.230 |

Assumptions: one 512-record fill = 26.2 µs (64.5 + 18.7/4 cycles per record at 1.35 GHz); a mover
pays 300 cycles per extra job; the split is a fluid model (global mean slot work + LPT imbalance
+ residual wait, floored by each big core's half of its big item). The LPT imbalance 0.061 ms is
what BRISC slots (no big items) show today. All numbers are on per-core means; the per-view
makespan (2.834 ms) is higher because the critical core changes from view to view.

## Gate

The realistic combined frame gain is **0.26 ms (×0.40) or 0.18 ms (×0.28)**, under the 0.30 ms
gate. Even the ceiling with an ideal split is 0.29 ms, and perfect balance with zero waits is
0.33 ms. Reaching the gate needs ≥ 46 % conversion from model to frame; #306 delivered 28-39 %.
The build would also need three non-trivial changes at once: a job protocol with per-mover queues
and streamed emit, a rank-ordered big-tile gather, and a two-mover radix sort on one core inside a
nearly full L1 (BRISC cannot hold the big buffers). The closest measured relative, splitting big
tiles into OL_MAT_SELECT parts across cores (#121), was 0.15 ms/view slower.

**Verdict: STOP. Do not build. See `docs/conclusion.md`.**

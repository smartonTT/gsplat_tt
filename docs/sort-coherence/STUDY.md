# t529: can a fixed scene skip re-sorting between views? (CPU-only study)

Task #529, 2026-10-12 (PDT). Base best-iter-224 (ad647d8e). No device work: every number
below is from the CPU model in this folder, except the device stage times, which are quoted
from the measured profile `docs/profile-t224/PROFILE.md` (#528, 377b0ab7, bh-33 p100a,
b2b 9.131 ms/view).

## Answer in short

- **30-view benchmark (b2b): no gain is available.** Consecutive bench views are 2-4 deg
  apart (18 deg on the hero <-> orb_00 wrap). Only **4% (2-18%) of (tile, gaussian) pairs
  stay in the same tile**, no tile keeps an identical list, and the previous view's order
  leaves **12.6 inversions per pair** inside each tile. That is more fix-up work than the
  device's adaptive radix sort does today. Bound for the bench: **0 ms/view** (an
  implementation would be slower).
- **Live viewer: a small gain is possible, not a big one.** At viewer-like steps
  (0.06-0.5 deg per frame) 77-97% of pairs stay in their tile and an old order leaves
  0.2-1.6 inversions per pair. That only helps the per-tile depth sort inside mat
  (~0.85 ms of mover time per view). The tile binning (sort program 1.46 ms + k2 0.65 ms)
  must still run every frame, because the pair sets change in every tile. Upper bound
  **~0.85 ms/view**, realistic **0.2-0.4 ms/view (2-4%)**, and only for slow camera motion.
- **Blending in an approximate order (no fix-up) is not exact** at any tested offset
  (0.25 deg already changes 40k pixels) and costs 0.05-4.5 dB. It fails the md5 gate. Rejected.
- **Scene-static preprocessing is already in place** for everything the sort path uses
  (cov3d, opacity, colours pinned on device; SH degree 0 so colour is view-independent).
  Nothing left to move out of the sort path.

Recommendation: do not spend a device task on cross-view sort reuse for the benchmark. If the
viewer needs a few percent later, option B2 below is the design to try.

## Method

`coherence.py` re-implements the CPU reference pipeline (`gsplat/rasterization.py`) in numpy:
EWA projection, opacity-aware 3-sigma AABB, 32x32 tiles, per-pair contribution cull at the
camera file's `contrib_floor`, per-tile depth sort. `sc_native.c` (built to `sc_native.so`)
counts per-tile inversions and blends a given order. Check of the model: the hero view in
exact order gives **42.52 dB** against `benchmarks/reference_v2/hero.png` (device: 42.51 dB).
Views and order: `benchmarks/cameras_v2.json` bicycle, hero, orb_00 .. orb_28, as
`render/run.py --back-to-back` runs them (hero first, then the orbit, wrap back to hero).

Key fact used throughout: view depth is `z = f . x + t_z` with `f` the camera forward vector
(row 3 of the world-to-camera rotation). **The depth order depends only on the view direction,
not on the camera position.** So an order computed for direction `d` is exact for every
camera looking along `d`, and its error grows with the angle between `d` and the current view
direction. (Sanity check: sorting by `f . x` gives 0.001 inversions per pair, all of them
equal-depth float ties.)

```
python docs/sort-coherence/coherence.py coherence   # 1) consecutive views + presort disorder
python docs/sort-coherence/coherence.py psnr        # 2) image impact of approximate orders
python docs/sort-coherence/coherence.py smallstep   # viewer-like steps between orb_14 and orb_15
```
Outputs: `out/coherence.json`, `out/psnr.json`, `out/smallstep.json` (images are not tracked).
Scene: bicycle, N = 6.13 M gaussians; per view 1.27-2.09 M visible, 2.0-3.1 M pairs (median
2.66 M), largest tile 20-23k pairs.

## 1. What stays the same between consecutive bench views

Per pair of consecutive views (28 orbit steps; the two wrap steps listed apart):

| metric | orbit median | orbit range | hero <-> orb_00 wrap |
|---|---|---|---|
| view direction step | 3.2 deg | 1.9-3.9 deg | 18 deg |
| camera position step | 0.055 | 0.034-0.068 | 0.31 |
| pairs of the new view also in the old view (same tile, same gaussian) | **4.4%** | 2.2-17.9% | 0.1% |
| tiles with an identical list | **0%** | 0% | 0% |
| inversions per common pair (old order vs new, inside a tile) | 1.26 | 0.45-3.49 | 0.17 |
| p99 max displacement of a common pair | 74 | 31-172 | 3 |

(#528's xview model, with a 3-sigma circle and no contribution cull, measured pair-set Jaccard
0.15 and visible-set Jaccard 0.935: same picture. The visible set is stable; the pairs are not.)

Why: a 3 deg turn shifts most of the screen by ~2 tiles (32 px), and most gaussians cover one
or two tiles, so almost every pair lands in a different tile.

Viewer-like steps (`smallstep`: cameras interpolated between orb_14 and orb_15, a 1.95 deg step):

| step per frame | pairs kept | tiles identical | inversions / common pair | prev-frame-direction order: inversions / pair, tiles already sorted, p99 displacement |
|---|---|---|---|---|
| 0.06 deg (1/32) | 97.0% | 0.1% | 0.19 | 0.19, 6.4%, 26 |
| 0.24 deg (1/8) | 88.1% | 0% | 0.71 | 0.78, 1.4%, 80 |
| 0.49 deg (1/4) | 77.0% | 0% | 1.31 | 1.56, 0.5%, 144 |
| 0.97 deg (1/2) | 56.3% | 0% | 2.11 | 3.18, 0.4%, 260 |
| 1.95 deg (full) | 17.5% | 0% | 2.12 | 6.27, 0%, 477 |

Even at 0.06 deg per frame no tile is unchanged, so a fix-up pass is always needed, and every
frame still has to re-bin.

## 2. Alternatives and their image impact

### (a) Global view-independent presort + per-view fix-up

The scene (or its compacted visible list) is stored in the order of `d . x` for a fixed
direction `d`. The device's tile binning is stable, so each tile's list then arrives in
`d`-order and only needs a fix-up. Disorder left per tile, median over the 30 views:

| presort | angle to view direction (median) | inversions / pair | tiles already sorted | p99 / max displacement |
|---|---|---|---|---|
| 26 directions (cube faces, edges, corners) | 15.6 deg | 55.2 | 0% | 2364 / 3162 |
| 128 directions | 5.9 deg | 23.6 | 0% | 1262 / 2148 |
| 512 directions | 3.6 deg | 12.0 | 0% | 746 / 1300 |
| 2048 directions | 1.7 deg | 5.7 | 0% | 374 / 662 |
| direction 0.25 deg off (synthetic) | 0.25 | 0.85 | 1.1% | 67 / 118 |
| 1 deg off | 1.0 | 3.4 | 0% | 236 / 452 |
| 4 deg off | 4.0 | 13.8 | 0% | 848 / 1486 |
| 16 deg off | 16.0 | 59.3 | 0% | 2543 / 3628 |

Inversions grow linearly with the angle, ~3.4 per pair per degree. To land under ~1
inversion per pair you need a presort within ~0.3 deg of the view, i.e. ~10^5 directions:
impossible (each is a 6.13 M permutation, 24.5 MB, and the scene streams would have to be
stored in that order, 368 MB per copy). Even 26 copies (9.5 GB of permuted scene) leave 55
inversions per pair. **Not viable.**

Image impact if the fix-up is skipped (bounded-disorder blending), hero view
(`out/psnr.json`):

| order used | vs exact CPU order | pixels changed | vs reference_v2/hero.png (exact: 42.52 dB) |
|---|---|---|---|
| 0.25 deg off | 61.4 dB | 40,165 | 42.48 |
| 1 deg off | 55.2 dB | 125,140 | 42.34 |
| 4 deg off | 49.0 dB | 314,662 | 41.71 |
| 16 deg off | 41.0 dB | 634,349 | 38.69 |
| 128-direction presort (9.7 deg) | 44.0 dB | 511,454 | 40.23 |
| 26-direction presort (21 deg) | 39.8 dB | 670,635 | 38.03 |
| previous view's direction (18 deg wrap) | 40.5 dB | 654,176 | 38.37 |

orb_14 (2.1 deg from orb_13): previous view's order 51.0 dB vs exact, 203k pixels changed;
1 deg off 54.0 dB; 4 deg off 46.6 dB. **No approximate order is bit-exact**, so all of these
break the golden md5. Rejected for the bench and for the viewer (the viewer must show the same
image as the bench).

### (b) Reuse the previous frame's order with an incremental repair

On the 30-view bench, the previous view's order leaves **12.6 inversions per pair** (median;
6.3 at the 1.9 deg step, 66 on the 18 deg wrap) and only 4% of pairs are even present in both
views. **It does not apply to the benchmark**: the views are far apart for this purpose.

It does apply to the live viewer when the camera turns slowly (table in section 1). Because
the order depends only on direction, translation alone (walking, dolly) costs nothing;
only rotation adds disorder.

### (c) Scene-static preprocessing out of the per-view path

Already done for everything the sort and blend paths consume: cov3d, opacity and colours are
uploaded once and pinned (#528 sec. 4; SH degree 0 makes colour view-independent). The
per-view N-sized cost left is pfwc's projection math (~1.2 ms fixed, #528), which is
per-view by nature. Static chunking for frustum culling was tried (#433/t493: skipped 39% of
pfwc tiles, no speedup, Morton reorder slowed blend). The 60% of gaussians never visible in the
30 views (#528) are a benchmark artifact, not a legitimate static cull. Nothing in (c) touches
the sort path, so it adds no sort saving.

## 3. Device work that could be saved

Device sort-related time per view at best-iter-224 (#528, measured, steady b2b):

| stage | ms/view | what it does | reusable across views? |
|---|---|---|---|
| k2 (pair emit) | 0.65 window | emits (gaussian, tile) pairs | no: pair sets change (4% kept on bench, 56-97% in viewer, 0% identical tiles) |
| sort program (one-launch bin: count, prefix, emit) | 1.46 window | bins pairs by tile, stable | no: same reason |
| mat: per-tile depth sort in L1 (`fz_mv_sort`, adaptive stable LSD radix, `sort_radix_tile_algo.h`) | ~0.85 per mover (mat 1.42 mean core / 2.04 max; TRISC1 `band_batch` 1.08 runs alongside) | sorts each tile's sub-chunks by depth | partly: an almost-sorted input could use a cheaper fix-up |

Cost model for the fix-up: P = 2.72 M pairs / (110 cores x 2 movers) = ~12.4k pairs per mover
in 0.85 ms at 1.35 GHz, so today's radix costs **~93 cycles per pair** (including moving the
records). An insertion fix-up costs about one L1 load/compare/store per element plus one
shift per inversion; at ~10 cycles per step it breaks even with the radix around **~8
inversions per pair**.

| option | bench b2b bound | viewer bound | work reduction factor (depth-sort part only) |
|---|---|---|---|
| A: global presort (26-2048 dirs) + fix-up | 0 (5.7-55 inv/pair, at or above break-even; memory infeasible) | 0 | <=1x |
| B1: blend in approximate order, no fix-up | not allowed (not exact, -0.05 to -4.5 dB) | not allowed | n/a |
| B2: previous-frame order + per-tile insertion fix-up (design below) | 0 (12.6 inv/pair: ~1x the radix work, plus the order bookkeeping) | <= 0.85 ms/view; realistic 0.2-0.4 ms (2-4%) at <= 0.5 deg/frame | ~4-6x at 0.2-0.8 inv/pair, ~1x at 3 deg |
| C: more static preprocessing | 0 for sort | 0 for sort | 1x |

The 0.85 ms upper bound assumes the whole mat depth sort disappears and that mat's movers,
not TRISC1 (1.08 ms), are what makes mat 1.42-2.04 ms. Only part of it is on the critical
path, so the realistic figure is lower. k2 + binning (2.1 ms, 23% of the view) cannot be
skipped under any option: no tile keeps its pair list even at 0.06 deg per frame.

### Implementation sketch for B2 (viewer only; not recommended for the bench)

1. Keep a persistent per-gaussian `rank_prev[N]` (u32, 24.5 MB DRAM) = position of each
   gaussian in the previous frame's global depth order, or a large value if it was not visible.
2. pfwc (`pfwc_compute.cpp`, `writer_pfwc*.cpp`, `gather_vis_scatter.cpp`): compact the
   visible records in `rank_prev` order instead of storage order (scatter to an N-sized slot
   array, then compact; ~+0.1-0.2 ms for the extra 24.5 MB read + slot pass). Newly visible
   gaussians go at the end and need a small sort-and-merge (visible-set Jaccard is 0.935 even at bench
   steps, so a few % of M per viewer frame).
3. k2 and the one-launch binning (`sort_bin_onelaunch.cpp`, `sort_onelaunch_algo.h`) stay as
   they are; they are already stable, so each tile's list arrives in `rank_prev` order.
4. mat (`matblend_ncrisc.cpp` / `matblend_brisc.cpp`, `sort_radix_tile_algo.h`): pick
   insertion sort keyed on (depth, gid) when a cheap pre-scan counts few descents (the older kernel
   already used insertion for n <= 16), else the radix. Keying on (depth, gid) keeps the result
   bit-identical to today's stable radix, so the md5 gate still holds.
5. Producing `rank_prev` for the next frame is the hard part: per-tile sorted lists do not
   give a global order without a 1024-way merge, and a global re-sort of the visible set
   costs more than it saves (global disorder is ~M/2 x Kendall distance, ~56 inversions per
   element even at 0.06 deg). A cheaper stand-in is a reference order for direction `d_ref`,
   refreshed in the background only when the view direction drifts > ~1 deg from `d_ref`
   (translation never invalidates it); its cost (a global sort of 1.8-6 M keys) must be
   amortized over the frames between refreshes, which only works for slow turns.

Given that step 5 eats much of the 0.2-0.4 ms and none of it helps the headline b2b metric,
B2 ranks below the levers in #528 (blend 3.67 ms, pfwc fixed 1.2 ms, binning 1.46 ms,
mat/TRISC1 0.29 ms bound).

## Files

- `coherence.py`: modes `coherence`, `presort`, `psnr`, `smallstep`, `all`.
- `sc_native.c`: per-tile inversion / max-displacement counter and an order-driven CPU blend
  (`cc -O3 -shared -fPIC -o sc_native.so sc_native.c`).
- `out/*.json`: raw results of the runs quoted above.

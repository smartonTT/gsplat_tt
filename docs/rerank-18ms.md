# Re-ranked levers at the 18.59 ms tip (task #167, 2026-10-04)

Tip: 220ba33 (iter-184), **18.59 ms/view, 53.8 FPS**, bicycle, 30 views, 1024x1024.
Board for every measured number in this file: **yyzo-bh-07, Blackhole p100a, 110 cores
(not a p150)**. No device was used for this task; every number comes from an earlier
capture, cited per row. Savings are **upper bounds derived from measured zone times**, not
measured gains. "Realistic" columns are judgment, and say so.

Sources:
- t160 Tracy, 30 views, traced with `GSPLAT_TT_OL_EMIT_PROF=1`, code e74a2dc (= tip code):
  `docs/emit-diet-t160/out/{gaps,zones,emit_parts}.txt`, untraced host stages
  `docs/emit-diet-t160/out/run-r1-base.log`.
- t150 roofline, 30 views, code 3372664 (pre-PRECULL, pre-fast-emit; used only for
  per-RISC balance of programs that t156/t160 did not restructure):
  `docs/reprofile-t150/out/iter180/roofline.txt`.
- t161 Tracy, 10 views, PRECULL 0/1/2 on b54c54e: `docs/precull-pc-t161/out/t161-m*-gaps.txt`.
- t147 per-tile blend cycles and schedule model: `docs/fuse-matblend-t147/` (06bf91a).
- Visible share 17-31% of 6.13 M gaussians: `docs/reprofile-t115/README.md` (lever G row).
- DRAM/NoC ceilings: `docs/hw-ceilings.md`.

## 1. Critical path at the tip (yyzo-bh-07 p100a)

Device timeline from t160 Tracy (mean per view). All five programs run on all 110 cores,
one after the other; nothing overlaps across programs or across views.

| # | segment | window ms | bound by (busiest RISC) | who owns it now |
|---|---|---:|---|---|
| 0 | pfwc (project + visibility + compaction + pair counts) | 2.98 | all 5 RISCs at 0.95-1.00 of window (t150); TRISC imbalance 0.22 | **nobody** |
| - | gap | 0.006 | | |
| 1 | segment K2 (TA pair scatter) | 1.48 | NCRISC 1.44 / 1.51 (t150), ~175 cycles per pair | **nobody** |
| - | **gap: host bridge before sort** | **0.67** | host | #155 |
| 2 | sort one-launch | 3.78 | BRISC | |
|   | - sort_ol_count | 0.90 makespan (0.63 mean, t150) | ~74 cycles per pair | **nobody** |
|   | - sort_ol_barrier | 0.44 makespan | waits on count imbalance (0.30, t150) | **nobody** |
|   | - sort_ol_prefix | 0.04 | | |
|   | - sort_ol_emit | 2.76 makespan (2.61 mean): issue_brec 0.51, pack/key 1.82, write 0.13, drain 0.09 | movers scalar issue | #164, #165, #166 |
| - | **gap: host worklist before materialize** | **0.36** | host: `bin_layout` 0.11 + `publish_host` 0.18 ms (untraced host timers) + upload | add-on A (see 2b) |
| 3 | materialize + SFPU cull | 3.38 | NCRISC busiest 3.75 vs mean 3.07 (t150): set by the largest item | big-tile re-sort task (being added) |
| - | gap | 0.006 | | |
| 4 | blend | 5.93 | TRISC1 SFPU issue; greedy makespan 5.86 vs mean 5.80 (t147) | #148 (counters) |
| | **device span (traced)** | **18.58** | all-core idle 1.04 | |
| | host output read-back `d2h` (untraced) | 0.21 | host, after the device finishes | **nobody** |

Untraced frame (r1): 18.56 ms = project 4.65 + sort 4.37 + blend stage 9.30 + d2h 0.21.

Stage share: per-gaussian front end (pfwc + K2) 4.46 ms (24%), pair sort 3.78 (20%),
mat + blend 9.31 (50%), host gaps + d2h 1.24 (7%).

## 2. Levers outside the in-flight set

In flight, not ranked here: #155 (bridge before sort), #164/#165 (emit scan fold / TRISC
pack), #166 (PRECULL=2 fast-emit fix + flip), #148 (blend waste counters). **#147 (fuse
materialize into blend) failed and is removed**: its calibrated model predicts ~0 gain
(cull moves onto the TRISC critical path, one slab per mover, tiles above 16384 records form
serial chains; `docs/fuse-matblend-t147/README.md`).

Gates: deep tier needs >= 3 ms/view; the standing bar is 0.3 ms (~1.6%) and 1% (0.19 ms).

| rank | lever | measured basis (p100a) | upper bound ms/view | realistic (judgment) | 3 ms gate | 0.3 ms / 1% bar |
|---:|---|---|---:|---:|---|---|
| 1 | **Chunk frustum cull before pfwc**: skip whole spatial chunks of gaussians outside the view before pfwc reads them | pfwc 2.98 ms, all RISCs busy (linear in gaussians read); visible 17-31% (t115) | 2.98 x (0.69..0.83) = **2.06-2.47** | 1.0-2.0, depends on chunk tightness (not measured) | no | yes |
| 2 | **Pair stage: K2 diet + fold the per-tile count into K2** (K2 writes the sort's count rows; sort_ol_count goes away; K2's 1-deep 64 B reads and per-page flushes get read-ahead and batching) | K2 1.48 + count 0.90 + barrier 0.44 (makespans); K2 write floor 20 MB / 260 GB/s = 0.08 | **~2.7** (2.82 - floor) | 0.8-1.5 | no | yes |
| 3 | **Big-tile re-sort shrink** (coordinator is adding it) | mat NCRISC busiest - mean 0.68 (t150); model 3.75 vs 3.03 hero, 3.33 vs 2.23 view 1 (t147); big-tile re-sort 0.108 us/record (t147 model) | **0.68-0.72** (measured imbalance) | 0.3-0.7 | no | yes |
| 4 | **Blend fixed per-tile cost** | t147 fit: blend_us = 85-90 + 0.027 rec + 0.197 live; 1024 tiles x 87 us / 110 cores | **~0.81** | 0.3-0.5; the intercept has rms 63-69 us per tile, so verify with a sub-zone split first | no | yes |
| 5 | **Add-on A standalone: device worklist** for the materialize (no totals read-back, no host bin_layout / publish_host, no worklist upload) | gap before mat 0.36 (t160 Tracy); host parts 0.11 + 0.18 = 0.29 (untraced) | **0.29-0.36** | 0.2-0.3, minus the device LPT cost (must reproduce the host LPT order exactly) | no | yes, barely |
| 6 | **Overlap the output read-back** with the next view's pfwc (double-buffered output, non-blocking `ReadShard` as found by #155) | `d2h` 0.207 (untraced host timer) | **0.21** | 0.15-0.2 | no | 1% only |
| 7 | pfwc core imbalance | TRISC busiest - mean 0.22 (t150) | 0.22 | fold into lever 1 (chunk cull changes per-core loads anyway) | no | below 0.3 |
| - | Blend LPT recalibration | greedy makespan 5.86 vs mean 5.80 (t147) | ~0.06 | dropped | no | no |
| - | Cross-frame overlap of mover-only stages (K2 + sort, TRISCs idle 5.26 ms) with TRISC-bound blend | windows above | 5.26 on paper | not feasible now: same-core overlap needs sort (~1.03 MB) and mat (~1.37 MB) L1 next to blend slabs in 1.46 MB (t147, sort v2); a spatial split saves nothing because RISC-time is conserved; adds a frame of latency | paper only | - |
| - | Dead-after-saturation records (28% of blend input, t147) | per-record mover, cull and blend cost | not derivable | saturation is only known inside the blend, after mat has paid for them | - | - |
| - | Cheaper exp / matrix-engine splat (e.g. community tt-splat: polynomial splat + order-independent blending) | no measured basis; changes the image | - | breaks the md5 / PSNR gate | - | - |

**Plain answer:** no lever outside the in-flight set clears the 3 ms deep-tier gate on
measured data. The frame is now a sum of 0.2-2.5 ms items. Levers 1 and 2 are the largest
(upper bounds 2.1-2.5 and ~2.7 ms) and both sit in the per-gaussian / per-pair front end,
which no task touches today. Levers 3-6 each clear the 0.3 ms or 1% bar only.

### 2b. Add-on A vs #155

They do not overlap in time: #155 removes the **0.67 ms idle before sort_ol_count** (host
reads P after K2; #155's 5ce3382 enqueues the sort behind K2 and lets the kernel read P on
the device). Add-on A removes the **0.36 ms idle before materialize** (host reads sort totals,
builds the bin layout and worklist, publishes, uploads). They share the file
(`render/host/sort_device.cpp`), the technique (device-side read of a count page instead of
a host round trip, non-blocking `ReadShard`) and the A/B driver. **Recommendation: do not
drop add-on A; run it as #155's follow-up after #155 lands**, bundled with lever 6 (d2h
overlap) as one "host residue" task with a kill switch per part, so the 0.3 ms gate is
judged on the sum (~0.5 ms upper bound).

## 3. Gap to the published GPU number

GPU reference: INRIA 3DGS on an RTX A6000, bicycle, **93 FPS = 10.75 ms/view at 1920x1080
(published, not measured;** arXiv:2308.04079, `opt/cpu-vs-tt-comparison.md` G1). No published
number exists at 1024x1024; scaling by pixel count gives **5.44 ms (derived from a published
number, not measured; optimistic**, because projection, tiling and sort do not shrink with
pixels).

| step | ms/view (p100a) | basis |
|---|---:|---|
| tip today | 18.59 | measured, t160 |
| in-flight work, realistic | -2.1 to -3.2 | #155 ~0.5; #166 0.3-0.7 (t162 measured -0.86 without fast emit, t162 measured +0.45 sort penalty with it); #164 <= 0.51; #165 part of 1.82; big-tile re-sort 0.3-0.7 |
| levers 1, 2, 4, 5, 6 above, realistic | -2.5 to -4.5 | section 2 |
| **reachable estimate** | **~11-14** | judgment on the above |

- Against 10.75 ms (A6000 at 1080p, published, not measured): the gap is 7.8 ms; about
  4.6-7.7 ms of it (60-95%) looks reachable on the p100a. Beating it needs the optimistic
  end of every lever.
- Against 5.44 ms (pixel-normalized, published-derived): not reachable with
  bit-identical levers. The blend alone has a TRISC floor of ~5.3-5.8 ms per core (t147: mean MATH per core
  5.80 ms on the hero view).
- A p150 has more Tensix cores than the 110-core p100a used here. Every stage above is
  parallel over cores, so a p150 run should be faster, but that is not measured; the p150
  re-anchor is still open (charter asks for p150; no p150 was free).

## 4. Proposed tasks (also in the #167 hand-off)

1. **Chunk frustum cull before pfwc** (lever 1). Step 1, device-free: on the Mac, from
   `scenes/bicycle.ply` and `benchmarks/cameras_v2.json`, measure per view the share of
   gaussians a chunk AABB test (3-sigma bounds, chunk 256/1024/4096) removes in ply order and
   in Morton order. Go only if the mean skip share x 2.98 ms >= 1 ms. Step 2: per-scene chunk
   table on the host, pfwc reader skips culled chunks, per-core work rebalanced over the
   surviving chunks. Gate: md5-identical to md5-r82new.txt (if reordering is needed, keep the
   original gid as storage index so tie order holds; else <= 1 LSB, PSNR >= 70 dB, same as
   t156), paired A/B >= 0.3 ms, Tracy shows the pfwc window shrink.
2. **Pair stage diet + count fold** (lever 2). Read-ahead of K2's lofs/box pages, batched
   pair writes without per-page flush, per-tile counts kept in RISC local memory, written as
   the sort's count rows when the K2 and sort page splits match (they both split pair pages
   across cores and movers; align them). Delete sort_ol_count's pass when the rows come from
   K2. After #155 and #166 land (same files). Gate: md5-identical, paired A/B >= 0.3 ms,
   Tracy K2 + count + barrier sum.
3. **Host residue: device worklist (add-on A) + d2h overlap** (levers 5 + 6). After #155.
   Gate: md5-identical, paired A/B >= 0.3 ms on the sum, Tracy shows the pre-mat gap < 0.05 ms.
4. **Blend fixed per-tile cost** (lever 4). Step 1: `GSPLAT_TT_BLEND_PROF` sub-zones per tile
   (acquire/init, T readback, pack, per-subchunk staging) on 2 views to confirm the 85-90 us
   intercept. Step 2: cut the largest part. Gate: md5-identical, paired A/B >= 0.3 ms.

# Lever re-rank at the ~17.4 ms tip (task #182, 2026-10-04)

No device run. All numbers come from captures already in the repo. Board: yyzo-bh-07,
Blackhole **p100a** (110 cores), bicycle, 1024x1024, 30 views. Tip = 04f86dc (#170,
K2 diet + count fold on by default).

Sources:
- Tracy, tip arm: `docs/k2-diet-t170/out/tracy2-on-{gaps,zones}.txt`. Diet-only arm:
  `tracy2-nf-*`. The base arm is named `on`.
- Untraced tip stages: `docs/k2-diet-t170/out/run-r7-base.log`.
- Host bridge and d2h: `docs/t171/README.md`.
- Per-view imbalance (older 19.7 ms tip): `docs/reprofile-t150/`.
- Blend waste: `docs/blend-waste-t148/`. Blend cost fit: `docs/fuse-matblend-t147/`.
- Previous re-rank: `docs/rerank-18ms.md` (#167).

## 1. Default chain check

The default pipeline (`GSPLAT_TT_PFWC_FUSE=1` since #122) runs five programs: **pfwc, K2,
sort_ol, mat, blend**. All five appear in the tip capture. proj_vis_scan, proj_scatter and
ta_bucket_scatter do not appear; they belong to the lever-2 chain (`PFWC_FUSE=0`).
`docs/t159/README.md` and `docs/proj-vis-scan-model/RESULT.md` (#179) describe that chain
and now carry a note saying so. Every lever below names its program, and that program is in
the default list.

## 2. Critical path at the tip

Program busy time and the gaps between programs, ms/view (Tracy, 30 views):

| program | on (tip) | gap before (on) | nf (diet only) | gap before (nf) |
|---|---:|---:|---:|---:|
| pfwc | 2.981 | - | 2.981 | - |
| K2 (k2_pairs + k2_rows) | 0.985 | 0.006 | 0.814 | 0.006 |
| sort_ol (barrier + prefix + emit; count is folded into K2) | 3.547 | **0.422** | 3.963 (with count) | 0.435 |
| mat (movers + mat_cull_mask) | 3.008 | **0.305** | 3.008 | 0.314 |
| blend | 5.868 | 0.006 | 5.857 | 0.006 |
| traced end | 17.129 | all-core idle 0.740 | 17.384 | idle 0.762 |

Untraced tip (`run-r7-base.log`), view_total **17.438** ms/view (paired means: on 17.382,
nf 17.566, off 18.270):

| host stage | ms/view | contents |
|---|---:|---|
| project | 4.111 | gather_wait 3.947 (pfwc + K2 on device), gather_result 0.101, pfwc args + enqueue 0.057 |
| sort | 4.105 | pread 0.022, bin_layout 0.118, bin_emit 3.677 (sort_ol on device), publish_host 0.175, mat enqueue 0.048, other 0.065 |
| blend | 8.880 | mat + blend windows on device |
| d2h | 0.333 | output read (t171 measured 0.197) |
| pybind | 0.030 | |

Device windows add to ~16.4 ms; the host residue is ~1.05 ms. Of that, the bridge before
mat is 0.310 (t171) and d2h is 0.2-0.33.

### Where each window goes: window minus mean per-core busy time

A zone's mean over (core, RISC) is the work an even split would take. Window minus mean is
the most that a perfect balance could recover. It also includes start/end cost outside the
zone, so it is an upper bound.

| program | window | busiest zone (busiest / mean) | window - mean | per-view busiest - mean (t150, older tip) |
|---|---:|---|---:|---:|
| pfwc | 2.981 | pfwc TRISC 2.948 / 2.765 | 0.216 | 0.222 (all 5 RISCs at 0.95-1.00 of window) |
| K2 | 0.985 | k2_pairs 0.922 / 0.863 | 0.122 | - |
| sort_ol | 3.547 | emit 2.931 / 2.479; barrier 0.407, prefix 0.373 (makespans) | emit 0.452 | emit 0.11-0.17 |
| mat | 3.008 | mat_cull_mask 2.837 / 2.520; movers 2.782 / 2.441 | 0.488-0.567 | NCRISC 0.676, TRISC 0.609 |
| blend | 5.868 | tile_blend_sfpu 5.491 / 5.358; tile_blend_load 5.172 / 5.10 | **0.510** | **TRISC 0.464**, NCRISC 0.311 |

The zones file gives the busiest core over the whole run divided by 30. That hides
per-view imbalance, because a different core can be the slow one in each view. The t150
column is the true per-view figure, but it comes from the older 19.7 ms tip.

## 3. Levers, ranked

Gate: >= 0.3 ms/view untraced. "Covered" means a running or queued task already owns the
lever.

| rank | lever | program | measured basis | upper bound ms/view | realistic (judgment) | covered by | status |
|---:|---|---|---|---:|---:|---|---|
| 1 | Emit mover speed refit | sort_ol | emit busiest 2.931 vs mean 2.479 | 0.45 | 0.3-0.5 (#177 predicts emit 3.01 -> 2.49) | **#177** queued | covered |
| 2 | Sort window fill / prefix: fewer, larger reads in the 1024-page window fill | sort_ol | prefix 0.037 -> 0.373 under the fold; ~26 MB of 64 B reads (t170) | ~0.78 (barrier + prefix makespans) | 0.2-0.4 | **#181** running | covered |
| 3 | Blend per-tile fixed cost | blend | t147 fit 85-90 us/tile x 1024 / 110 cores | ~0.81 | 0.3-0.5 | **#172** running | covered |
| 4 | **NEW A: blend tail rebalance** (true-cost claim order, split the costliest tiles) | blend | window - mean 0.510 (tip); per-view TRISC imbalance 0.464 (t150) | **0.46-0.51** | 0.2-0.35 | none (#172 is per-tile fixed cost, not order or granularity) | proposed |
| 5 | Mid-tile mover split (8192 < n <= 16384) with a dependency-aware worklist | mat | movers window - mean 0.567; NCRISC per-view 0.676 (t150) | 0.57-0.68 | model pending | **#176** queued (model first) | covered |
| 6 | **NEW B: overlap the host mat bridge with the sort emit** | sort_ol -> mat bridge (host) | bridge 0.310 untraced (t171); traced gap before mat 0.305 | **0.31** | 0.15-0.25 | none (#171 only judged replacing it with a device pass) | proposed |
| 7 | **NEW C: drop dead records before blend** (mask 0 after cull, empty dispatches) | mat / blend | #148: empty-dispatch records 0.244, wasted microblock dispatches 0.186 | **0.43** | 0.15-0.25 | none (#148 was judged against the old 3 ms gate) | proposed |
| 8 | K2 count difference array + pfwc core rebalance | K2, pfwc | count costs k2_pairs 0.16 (t170); pfwc imbalance 0.216 | 0.38 together, each < 0.3 | 0.1-0.25 | none | bundle only, below gate alone |
| 9 | Emit fold re-check | sort_ol | #164 0.12-0.15 | 0.15 | 0.1 | **#173** queued | covered, below gate |
| - | d2h without an extra round trip | host | d2h 0.197-0.333 | 0.2 | ~0 under the per-view latency metric (t171) | - | excluded |

### Excluded (shelved or rejected; no new data changes them)

| lever | task | why |
|---|---|---|
| Fuse mat into blend | #147 | shelved: L1 does not fit sort, mat and blend slabs together |
| Early sort (gap before sort_ol) | #155 | untraced gap only ~0.2 (vs 0.422 traced); measured gain 0.03 |
| Emit pack on idle TRISCs | #165 | +0.056 |
| Shared big-tile sort | #168 | -0.038 |
| Chunk frustum cull before pfwc | #169 | +0.012, fails the 1 LSB gate |
| Device worklist for mat; d2h overlap | #171 | net 0-0.2; d2h has nothing to hide under while views run one at a time |
| Big-tile split across two movers | #175 | 0.05-0.1 |
| Mat LPT calibration | #144 | -0.21 |
| proj_vis_scan levers | #179/#180 | not in the default chain |
| FPU quadratic form in blend | #111 | not md5-identical; net <= 1 ms at the old 10.6 ms blend |

### pfwc (2.98 ms, the largest program no task owns)

There is no new lever here. All five RISCs are busy for 0.95-1.00 of the window (t150). The
TRISC spends ~67 cycles per gaussian, so pfwc is compute-bound on the per-gaussian SFPU
work, not on the 40 B/gaussian of streamed input. Fewer input bytes would not help. The only
work cut tried, chunk cull (#169), lost. The core imbalance (0.22) is below the gate alone,
so it is folded into rank 8.

## 4. Top 3 new levers

### A. Blend tail rebalance (blend) — measure first, then build

- **Basis:** blend window 5.868 vs mean TRISC busy 5.358: 0.51 ms of the window is not
  average-core work. t150 measured per-view TRISC busiest minus mean at 0.464. The claim
  order comes from host LPT lists, whose cost model is the record count only
  (`sort_device.cpp` `build_lpt`). The t147 fit says the real cost is
  87 + 0.027 rec + 0.197 live us per tile, and live pixels do not track records. The largest
  tiles (max_tile_n 17.8k-23k records) can cost about 1 ms each, which sets the tail
  granularity. #167 dismissed this at ~0.06 using a hero-view model. The tip window and t150
  disagree with that.
- **Step 1 (no build, one Tracy capture or the t170 `dev30.csv`):** per view, find the spread
  of blend end times across cores and the cost of the last tile each late core claimed.
  Gate: per-view tail >= 0.3 ms.
- **Step 2 (model, no device):** replay dynamic claiming with the measured per-tile costs,
  under (a) claim order by live-pixel cost (mat_cull_mask knows each tile's live count),
  (b) splitting the top-k tiles into 32x16 halves, (c) both.
- **Step 3 (build):** whichever wins. Example: mat writes per-tile live counts and the blend
  claim order puts the largest first. md5 must stay identical, since only order and
  partitioning change.
- **Expected untraced saving:** 0.2-0.35 ms/view. **Model first: yes.**

### B. Overlap the host mat bridge with the sort emit (host, sort_ol -> mat)

- **Basis:** after sort_ol finishes, the host reads per-tile totals (0.015-0.022), builds the
  layout and LPT (0.097-0.118), builds the subchunk layout and mover worklist and uploads it
  (0.157-0.175), then enqueues mat (0.041-0.048). Total 0.310 untraced (t171). The traced
  gap before mat is 0.305. The emit runs 2.9 ms, so there is plenty of time to hide this
  work, but the host waits for the whole sort before starting.
- **Idea:** per-tile record counts are already final once K2 ends: K2 writes the count rows
  since #170, and the emit does not change them. Get the counts early, build the layout on
  the host while the emit runs, and queue the uploads and the mat launch behind sort_ol on
  the same in-order queue. #171 only judged moving this work onto the device. It did not
  consider hiding it under the emit.
- **Feasibility check first (host only, no device):** (1) can the counts be read before
  sort_ol ends without a new sync point? Options: a read on a second command queue polling
  a flag the prefix sets, or a read of K2's count rows at the K2 -> sort_ol boundary if the
  host already waits there. Note that the per-core rows are ~0.9 MB, but the prefix totals
  are one page. (2) Must the mat worklist uploads wait for the sort? They target buffers the
  sort does not touch. (3) Does the result match the current LPT order exactly, so md5 is
  unchanged?
- **Expected untraced saving:** 0.15-0.25 ms/view (0.31 minus the early read and the
  enqueue that stays). This is at or under the gate, so build only if the check shows
  >= 0.25 hidden. **Model first: yes (feasibility and timing check).**

### C. Drop dead records before blend (mat / blend)

- **Basis:** #148 counters (`GSPLAT_TT_MB_STATS=1`): records whose mask is 0 after the
  microblock cull still cost 0.244 ms/view of blend dispatch. Dispatches into saturated or
  pixel-free microblocks cost 0.186. Only 32% of pixels in dispatched microblocks are
  live. #148 closed this against the old 3 ms deep-tier gate. Under today's 0.3 ms gate the
  0.43 ms upper bound qualifies.
- **Idea:** mat_cull_mask already computes each record's mask. Have the mat movers skip
  writing records whose mask is all zero (compaction in the mover that already writes
  them), so blend never loads or dispatches them. The second half (saturated microblocks)
  is only known inside blend, so leave it out.
- **Model first:** from the #148 counters, compute (a) blend time saved by the 16.5% of
  records that are dead, minus (b) the extra mover cost per kept record from the
  compaction branch, using the t144 mover fit (0.197 us/record on NCRISC). Build only if
  the net is >= 0.3 ms or it stacks with #172 to clear the gate. md5 must be identical,
  since dead records add nothing to the image.
- **Expected untraced saving:** 0.15-0.25 ms/view. **Model first: yes.** Run it after #172
  lands, because both change blend's per-tile cost.

## 5. GPU gap

Published, not measured: 3DGS bicycle on an RTX A6000 at 1920x1080, 10.75 ms/frame.
Scaled by pixel count to 1024x1024: 5.44 ms (derived from the published number). The tip
is 17.44 ms/view untraced, about 3.2x slower. The covered levers (ranks 1, 2, 3, 5) have
upper bounds of 0.45 + 0.78 + 0.81 + 0.68 ≈ 2.7 ms. The new ones (A, B, C) add realistic
0.5-0.85 ms. Even if all of them land at their upper bounds, the tip stays near 14 ms.
Closing the GPU gap needs a structural change of more than 3 ms, and none of the measured
data points to one. The frame is now a sum of 0.2-0.8 ms items on top of pfwc, emit, mat
and blend work that is near its per-element cost.

## 6. Recommendation

Queue A step 1 (a measurement only) now. Queue B and C as model/feasibility tasks with no
device. Run C after #172. If none of A, B and C clears 0.3 ms once modeled, and #172, #176,
#177 and #181 have landed, the remaining levers are below the charter's bar. The project
should then report its conclusion and stop.

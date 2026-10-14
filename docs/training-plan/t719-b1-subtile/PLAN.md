# #719 — B1 adaptive heavy-tile split: implementation plan

Status 2026-10-14: host planner and its unit test landed on this branch
(`render/host/subtile_split.h`, `tests/unit/test_subtile_split.cpp`, PASS). No device
code is changed yet, so there are **no device results**: no md5, b2b table, JIT check
or screenshot. The device work is split into the sub-tasks in §5. They are too large
for one run.

Source: docs/training-plan/t710-lowres-blend/SIZING.md (branch ttp/t710-lowres-blend-sizing).
Base: best-iter-229 (b9171bf9) plus the D1 merge (40c10784).

## 1. What B1 does
A 32x32 tile whose estimated cost is above total / 120 becomes four 16x16 sub-tiles. A
quarter that is still above total / 120 becomes four 8x8 sub-tiles. Each sub-tile is
its parent tile plus a microblock mask (m = (my<<2)|mx, my 0..7 in 4-pixel rows, mx 0..3
in 8-pixel columns):
- a 16x16 sub-tile covers 8 microblocks;
- an 8x8 sub-tile covers 2.

Sub-tile ids come after the screen tiles, in [num_tiles, num_ids()). A split parent's
own list is empty.

## 2. Why it is lossless
- **Same floats.** Records stay parent-relative. pack_rec in sort_bin_onelaunch.cpp
  (~l.670) subtracts the parent's origin (txi*32, tyi*32). For a sub id it must take
  the parent from a sub→parent table, never the sub-tile's own origin. Every cov/mx/my
  word the blend sees is then bit-identical to today's.
- **Same records.** A record goes to sub-tile s if its cull box [mx_lo..mx_hi] x
  [my_lo..my_hi] (microblock_cull.cpp: r = sqrt(-2 log(floor/op)), half-extents
  r*sqrt(a), r*sqrt(c)) meets s's mask. The cull keeps only microblocks inside that box,
  so every record with a keep bit in s is in s's list. A conservative (larger) box is
  also fine, because the cull still decides.
- **Same order.** Buckets fill in ascending pair order and the radix is stable, so each
  sub-tile list is the parent list restricted, ties included. The unit test checks
  per-microblock sequences on random lists.
- **Saturation cadence (finding).** The blend reads its max-T live mask only every
  BLEND_T_PERIOD = 512 records (alpha_blend_compute_mb.cpp ~l.1524–1594;
  blend_device.cpp:426, 0 = off). Between reads, dead microblocks keep blending. A
  shorter sub-tile list reaches its checks at different records than the parent did. So
  at the default period, B1 output is **not** md5-identical to today's. Pixels differ
  only where T < 1/256, which is below u8 resolution in almost all cases, but it is not
  bit-exact.
  - md5 gate: run **both arms with BLEND_T_PERIOD=0**. With saturation off, both arms
    blend exactly the same records in the same order.
  - At the default period: report PSNR and max-abs u8 diff against the period-0 arm.
  - The training backward must replay the forward's own stop points (T0 oracle).

## 3. Where the split plan comes from
Per-tile counts are known only after the K2 count pass. Choices:
- (a) **Previous view's counts** (chosen). Use view N-1's K2 count rows and microblock
  touches for view N's plan (views run in training/NN order, so they are coherent).
  This is lossless, because the plan only moves records between lists, never drops one.
  The first view, or a tile that overflows its bucket, takes the existing
  overflow_retry.h path with the view's own counts (one re-run). It costs no extra
  device pass.
- (b) A device split between count and emit. This is exact for every view but adds a
  host round trip or a device planner. Use it only if (a)'s re-run rate is above about 2%.

## 4. Code map (per stage)
| Stage | File | Change |
|---|---|---|
| plan | render/host/subtile_split.h (new, done) | plan_split, masks, emit rule |
| upload | tile_assign_device.cpp | per-tile (first_id, n_sub, 16 masks) table, about 2 KB at s1, in the pfwc runtime args/L1 |
| emit | render/kernels/dataflow/pfwc_fuse.h l.179 and ~l.398 | for a split tile, emit (gid, sub_id) for every sub whose mask meets the record's microblock box; K2 count rows indexed by id; row stride = round16(num_ids) |
| sort | sort_bin_onelaunch.cpp, sort_onelaunch_layout.h | bucket count = num_ids; pack_rec uses parent(id) for the origin |
| cull | mat_cull_compute.cpp | AND the record mask (word3) with mask_of(id) |
| blend | alpha_blend_compute_mb.cpp | live mask starts at mask_of(id), so dead microblocks are skipped from record 0 |
| write | matblend_brisc/ncrisc.cpp | write only the sub-tile's rows/columns (16 or 8 px) into the u8 zero-copy image at the parent origin plus the sub offset |
| claim | blend_claim_order.h (unchanged), sort_device.cpp:1615 | pass num_ids; split parents have count 0 (unit-tested) |
| overflow | overflow_retry.h | on overflow, re-plan with the view's own counts and re-run |
| backward | (t674 T0 oracle) | same lists; replay stop points |

Bucket caps: SIZING puts the largest sub list at about 25k (s2) and about 35k (s4).
kTileCapBig (65472) covers both, which is what lets 3M s4 run.

## 5. Sub-tasks (each one device run under `ttp lock p100`, p150a bh-49)
1. **B1a emit+sort** (~$8): plan upload, pfwc_fuse emit by sub id, count rows/stride,
   sort id space, parent-origin pack_rec, with the blend temporarily full-mask.
   Gate: at BLEND_T_PERIOD=0 the md5 must equal today's at 1M s4. Expected to hold
   already, because extra records only land in other sub-lists and the blend writes the
   whole tile. Correction: with full-tile writes, sub-tiles of the same parent would
   overwrite each other. So B1a must land together with the writer change of B1b, or
   with an env switch (B1=0 by default) that keeps today's path.
2. **B1b mask+write+claim** (~$8): cull mask AND, live-mask init, sub-tile writer,
   claim over num_ids, overflow re-plan.
   Gates:
   - md5 at BLEND_T_PERIOD=0 equals today's at 1M s4, 3M s2 and s1 6.13M;
   - 30-view b2b grid: s4 {0.5M,1M,1.5M,3M}, s2 {1M,3M,6.13M}, s1 6.13M (D1 tooling:
     docs/training-plan/d1-scaling/drive.sh + remote.sh, cameras_train_nn);
   - `python3 opt/ab_jit_check.py` on both arms' JIT dirs (a JIT-HASH FAIL voids the A/B;
     under ~0.1 ms/view is noise);
   - s1 hero.png, diff image and PSNR against benchmarks/reference_v2/hero.png, plus a
     visual tile-artifact check.
   Pass bar: 1M s4 <= 2.0 ms/view (today 6.47), and 3M s4 runs.
3. **B1c backward** (only if B1b passes): the same sub lists in the backward, checked
   against the T0 gradient oracle (#674). Quality bar: published 3DGS/gsplat or LiteGS
   bicycle test PSNR/SSIM/LPIPS (labeled published). Lossless only.

Restrictions restated for every sub-task: own worktree; never edit ~/dev/gstt2; every
sync+build+run inside `ttp lock p100 -- ...` with the repo-scoped one-way rsync
(opt/sync_remote.sh); never reserve or release machines; never use bh-30; no host-key
bypass; no scanning; push only to a new ttp/* branch; no PRs, no force pushes.

## 6. D1 baseline to beat (p150a bh-49, best-iter-229, b2b ms/view)
| N | s4 | s2 | s1 |
|---|---|---|---|
| 0.5M | 4.331 | 2.560 | 2.437 |
| 1M | 6.470 | 3.708 | 3.245 |
| 1.5M | fails | 4.723 | 3.934 |
| 3M | fails | 7.288 | 5.758 |
| 6.13M | fails | fails | 7.918 |

SIZING model for B1: 1M s4 1.43; 3M s4 2.40; s2 1M/3M/6.13M 1.85/2.90/3.53 (modelled,
not measured).

## 7. B1 full (#738): corrected emit/sort design

Status 2026-10-14 05:20 PDT (#738, branch ttp/t738-b1-full-per-sub-tile-emit-and-sort-to-fi,
on top of #722's B1' at 8a9def98): routing header and unit test done
(`render/kernels/dataflow/subtile_route.h`, `tests/unit/test_subtile_route.cpp`, PASS; a
mutation that shrinks thr by 0.3 or makes ln_upper a lower bound fails it). No device code
changed yet: no gate is met (3M s4 / 1.5M s4 / 6.13M s2 still overflow).

### 7.1 Finding: the emit cannot go in pfwc_fuse.h
§4 puts "emit (gid, sub_id) for every sub whose mask meets the record's microblock box" in
the K2 (pfwc_fuse.h l.179, ~l.398). That does not fit the K2:
- the pair index space is fixed before the K2 by the tile-box areas: pair p belongs to the
  gaussian whose lofs prefix covers p, and p - lofs walks its tile box row by row. One
  pair per (gaussian, tile); a gaussian cannot emit a variable number of extra pairs
  without K1 counting them into lofs first;
- the K2 sees only the packed tile box (vis_tile.h aabb), no conic, opacity or mean, so it
  cannot compute a microblock box.

The sort bin movers (sort_bin_onelaunch.cpp) can: process_batch already loads each
gaussian's conic, mean and opacity (pack_invariants, ~l.650) next to its tile id. So the
fan-out goes there: one pair in, one record per routed sub-tile out.

### 7.2 Routing rule (done)
`subtile_route::box_mask(A, B, C, mx, my, q16, 1/floor)` gives a microblock mask that
contains every microblock the device cull can keep (thr = 2 ln(op/floor) + 0.05, upper
bound of ln from the fp32 bits, squares only, no sqrt/log; degenerate records take the
full mask). A record goes to sub k of a split tile iff `mask_of(k) & box != 0`
(`subtile_route::targets`). The test checks it against the exact boxmin cull on 20k random
records (box is 2.2x the kept microblocks on that sample) and that every microblock gets
the same cull-kept records in parent order from its sub list. mx, my are relative to the
PARENT tile origin, as pack_rec writes them.

### 7.3 Sub-tile shapes: use B1' unit codes, not 8x8
#722's blend already handles units (parent, code), code = quarter 1..4 (16x16) or band
5..20 (16x4) (`subtile_unit.h`), with the mask, live-mask and sub-region writer done. So
the host plan should split a heavy tile into 4 quarters or 16 bands with
`subtile_unit::split_code(n, k)`, not subtile_split.h's 8x8 eighths; then sub id
first_id + k is unit (parent, split_code(n, k)) and B1b needs no new writer.

### 7.4 Remaining steps (each is its own run)
1. **Sort bin (B1 route mode, compile define SORT_B1_ROUTE)**: force the plain path (fold
   off: the K2's per-tile rows cannot hold sub counts; ring_on off; TRISC writers `town`
   off). Plan table in L1 (one DRAM page read at start): per tile `first_id | n << 16`
   (0 = unsplit), per sub id its unit mask. Count step (l.370-405): for a kept pair whose
   tile is split, read its blendrec (brec_now, cached per gid; pairs are gid-major) and do
   `hp[id]++` per target; others `hp[t]++`. Rows and totals over num_ids
   (row_pages = round16(num_ids) / 16). Emit (process_batch, non-ring branch ~l.858): per
   target `c = curp[id]++`, `slot = id * tile_cap + c`, `pack_rec(..., parent)` (pack_rec
   already takes the tile whose origin it subtracts). q16 = `inv_opr & 0xFFFF`.
2. **Host (sort_device.cpp)**: bucket buffer num_ids * tile_cap * 32 B (l.2189, l.3568),
   row stride, num_tiles arg -> num_ids for the sort, materialize and tile_ranges; plan
   upload; `GSPLAT_TT_B1_ROUTE=1` env switch (default off keeps today's path bit for bit).
3. **Materialize / cull** (sort_subchunk_materialize.cpp, mat_cull_compute.cpp): iterate
   num_ids buckets; AND word3 with the unit mask (or leave it to the blend's unit mask, as
   B1' does).
4. **Blend reader** (reader_alpha_blend_mb_devcull.cpp l.247-415): in route mode, claim
   over num_ids, take (parent, code) from the uploaded id table instead of the on-device
   heavy detection; ids < num_tiles are code 0.
5. **Plan source (render.cpp ~l.537 retry loop)**: view N uses view N-1's per-id totals
   (a split parent's total = sum of its subs); on overflow, re-plan with the view's own
   totals and re-run (before growing the floor). First view: plan from its own totals
   after one overflow, as overflow_retry does today.
6. Gates as in the #738 spec: s4 3M, s4 1.5M, s2 6.13M run; md5 at BLEND_T_PERIOD=0 equal
   to B1=0 at s4 1M, s2 3M, s1; b2b via t722/drive.sh; ab_jit_check on both arms; s1
   hero + diff + visual check.

Cost note: the count step's blendrec reads for split-tile pairs are new DRAM traffic
(one 64 B page per gaussian run, the same reads the emit already makes). If they show up
in the sort time, keep the route mask per pair in a scratch plane from the count to the
emit.

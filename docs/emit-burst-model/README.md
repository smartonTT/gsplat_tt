# t637: emit-burst model for U3 (Morton reorder + cull), CPU only

**Question.** #631 bounds U3 (Morton reorder + per-chunk cull for pfwc) at 0.44 ms/view gross, but
three device runs (#470, #479, #493) lost 0.14-0.41 ms/view because Morton gids make the onelaunch
sort emit burst into one screen tile. Is there an order that keeps the emit within a ring that fits
L1 (R/D 4/8 or 8/8; 16 does not fit, t493) and nets >= 0.1 ms/view b2b?

**Answer.** Only one option clears both bars, and only on the central estimate: **u64g1**
(64-gaussian cull units, 16 units gathered per pfwc tile, lanes interleaved one gaussian per unit).
Its emit bursts stay within the 4-record ring (record-weighted p99 2, max 4 over 30 views) and it
skips 59% of pfwc slots (gross 0.63 ms). The emit still costs +0.18 ms (+0.14 calibrated), and two
costs are not yet measured (pfwc unit-gather reads, host unit list), so the net is **about
0.0-0.3 ms/view b2b, central ~0.15**. Global Morton (with or without the chunk-1024 cull) and the
strided emit read do not fit any ring that fits L1 and lose 0.3-0.5 ms net. **Recommendation:** no
full U3 A/B yet; first a cheap device probe of the pfwc unit-gather in b2b mode (u64 lists, no sort
change) and build u64g1 only if the measured pfwc saving is >= 0.35 ms/view b2b.

## Model

`model.py` rebuilds, per bench view, the exact record stream the onelaunch emit sees
(bicycle after the #536 prune, N = 5,559,909, 1024x1024, 30 views of `benchmarks/cameras_v2.json`,
C = 120 p150 cores):

- gid order: pfwc deals 1024-slot tiles strided over the cores; each core's visible gaussians get
  consecutive gids in tile then lane order; core segments are concatenated.
- pairs per gid: the gaussian's 3-sigma rect of 32 px screen tiles (t479 `imbalance.py` rects,
  same visibility model as t466/t631), row-major.
- the sort core split: equal 16-pair pages per core, mover 0 / mover 1 halves (EMIT_SPLIT 500).

`emit_sim.c` replays each mover's stream:

- `sim` (one packer, 1 unit per record): a run start (c % R == 0) waits until the run D records
  back left L1, L record-times after its last record. Burst = consecutive records of one tile with
  no gap longer than L; the burst histogram counts records.
- `sim3` (the TOWN emit, task #202): TRISC t % 3 owns tile t, 3 units per own record, batches of
  OL_PB 8 x 16 = 128 pairs, 4 batch slots in flight (`sort_ol_town.h` SLOTS), same ring wait.
  This adds what `sim` misses: a burst into one tile serialises on one TRISC while the other two idle.

**Calibration** (`calib.py`, `out-C110-v0-10.json`: t479 p100a Tracy, C = 110, views 0:10,
sort_ol_emit max-core ms/view off 1.42, ro 2.07, ro48 1.88). `sim` alone cannot fit both Morton arms
(L 5-6 fits ro 8/8 but then predicts ro48 1.49 against 1.88). `sim3` fits ro48 at any L (2.00-2.02)
because most of the Morton cost is TRISC owner imbalance, not ring wait: with no ring wait at all
(D 16) Morton still costs +0.6 ms. L = 4 is used; it over-predicts the measured Morton Δ by ~25-30%
(ro: +0.90 vs +0.65; ro48: +0.60 vs +0.46), so the model is pessimistic and the tables also give
the Δ scaled by 0.75. Cross-check against t493 (p150 b2b, c48 = mcull with ring 4/8): model
loss = emit +0.65 (0.49 calibrated) + host list 0.19 - pfwc gross 0.44 = +0.40 (+0.23 calibrated) ms; measured +0.177. t493 was latency mode with --dump-views at aiclk 800, which hides pfwc time (#464), so the measured loss is itself an upper bound on the real b2b loss; the model errs pessimistic.

## Results (30 views, C = 120, L = 4)

Calibrated 0.0986 us per record-time, N = 5,559,909; `tables.txt` is the full output.

| order | skip (slots read) | burst p50 / p99 / max (records, same tile) | records in bursts >4 / >8 | emit Δ ms R8D8 | emit Δ ms R4D8 | gross pfwc saving ms | 1-packer stall, record-times |
|---|---|---|---|---|---|---|---|
| today | 0.000 | 1 / 2 / 10 | 0.000 / 0.000 | +0.000 | -0.005 | 0.000 | 77.407 |
| u64_s8 | 0.589 | 6 / 16 / 43 | 0.622 / 0.304 | +0.353 | +0.185 | 0.626 | 2845.935 |
| u64g1_s8 | 0.589 | 1 / 2 / 5 | 0.000 / 0.000 | +0.158 | +0.157 | 0.626 | 22.618 |
| u64g32_s8 | 0.589 | 6 / 16 / 41 | 0.595 / 0.271 | +0.313 | +0.159 | 0.626 | 2725.564 |
| morton | 0.000 | 37 / 790 / 1023+ | 0.749 / 0.685 | +0.956 | +0.656 | 0.000 | 3414.846 |
| mcull | 0.418 | 37 / 791 / 1023+ | 0.749 / 0.685 | +0.944 | +0.647 | 0.444 | 3414.331 |
| mcull_s8 | 0.418 | 7 / 17 / 39 | 0.654 / 0.360 | +0.702 | +0.474 | 0.444 | 3087.303 |
| mcull_s64 | 0.418 | 12 / 78 / 200 | 0.713 / 0.579 | +0.672 | +0.429 | 0.444 | 3217.665 |
| u64 | 0.589 | 25 / 64 / 130 | 0.740 / 0.667 | +0.390 | +0.198 | 0.626 | 3152.386 |
| u64g32 | 0.589 | 19 / 32 / 96 | 0.730 / 0.648 | +0.366 | +0.186 | 0.626 | 3007.769 |
| u64g16 | 0.589 | 13 / 16 / 48 | 0.712 / 0.608 | +0.343 | +0.184 | 0.626 | 2717.436 |
| u64g1 | 0.589 | 1 / 2 / 4 | 0.000 / 0.000 | +0.183 | +0.181 | 0.626 | 18.732 |

per-view spread of emit Δ ms (R4D8): min / median / max
  today      -0.007 / -0.005 / -0.003
  u64_s8     +0.116 / +0.182 / +0.285
  u64g1_s8   +0.097 / +0.154 / +0.234
  u64g32_s8  +0.101 / +0.157 / +0.246
  morton     +0.490 / +0.652 / +0.795
  mcull      +0.506 / +0.650 / +0.867
  mcull_s8   +0.367 / +0.477 / +0.601
  mcull_s64  +0.275 / +0.424 / +0.604
  u64        +0.145 / +0.202 / +0.251
  u64g32     +0.132 / +0.188 / +0.246
  u64g16     +0.136 / +0.185 / +0.237
  u64g1      +0.123 / +0.183 / +0.234

Columns: skip = share of the N slots pfwc no longer reads (mean of views); burst = records per screen
tile per burst in the one-packer stream (record-weighted p50/p99, max); emit Δ = `sim3` max-core emit
time minus today's order at R8D8, in ms (9.86e-5 ms per record-time from the calibration); gross =
skip x 1.063 ms (t227 pfwc fit; t470 measured 0.525 b2b on p150 at chunk-1024 skip).

Options:
- `today`: pruned PLY order (the shipping path).
- `morton`: Morton order, no cull (t479 arm `ro`).
- `mcull`: Morton, chunk 1024, sphere cull (t631), survivors dealt strided (= `GSPLAT_TT_CHUNK_CULL`).
- `mcull_s8`, `mcull_s64`: as mcull, each emit mover reads its page range k-way interleaved.
- `u64`: 64-gaussian cull units (sphere per unit); each pfwc tile gathers 16 surviving units
  T = ceil(M / 16) apart; lanes unit-contiguous.
- `u64g32`, `u64g16`, `u64g1`: as u64, lanes interleaved over the 16 units in groups of 32, 16 or 1.
- `*_s8`: the option plus the 8-way strided emit read.

## Findings

1. **Global Morton is the problem, the cull is not.** `morton` and `mcull` burst identically
   (p50 37, p99 ~790, max >= 1023 records into one tile; 69% of records in bursts > 8). Emit Δ
   +0.65 ms at R4D8 against a 0.44 ms gross saving: a net loss, as #470/#479/#493 measured.
2. **Strided emit reads cut the bursts but not enough.** `mcull_s8` brings p99 to 17 (36% of records
   still in bursts > 8, needs ring 16+, which does not fit L1) and emit Δ to +0.47; `mcull_s64` is
   worse on bursts (p99 78) because the 64 sub-streams are too short to interleave tiles.
3. **Small cull units fix the spatial locality at the source.** With 64-gaussian units gathered from
   16 distant survivors per pfwc tile, the skip rises to 0.59 (finer cull than chunk 1024) and the
   burst length is set by the lane order: unit-contiguous lanes (`u64`) burst 25/64 (p50/p99),
   groups of 32/16 burst 19/32 and 13/16, and full interleave (`u64g1`) bursts 1/2, the same as
   today's order (1/2, max 10). Only g1 fits the 4-record ring.
4. **A residual +0.18 ms emit cost remains even with no bursts** (u64g1: one-packer stall 19
   record-times vs 77 today, yet `sim3` max-core emit +0.18). It is TOWN owner imbalance: the 128-pair
   batches of a spatially ordered stream cover few screen tiles, so the t % 3 tile owners get
   unequal shares. The 8-way strided read trims it only to +0.16 (`u64g1_s8`) and slightly lengthens
   bursts (max 5). The ring depth is secondary; the R8D8 and R4D8 columns differ only for bursty orders.
5. **Per view** (R4D8 emit Δ, min/median/max): u64g1 +0.12/+0.18/+0.23, mcull +0.51/+0.65/+0.87.
   The u64g1 cost is stable across the 30 views; no view breaks the ring.

## Net estimate (b2b ms/view, best option)

| item | ms/view | source |
|---|---|---|
| pfwc gross saving (skip 0.589 x 1.063) | -0.63 | model; t470 measured 0.525 b2b at chunk-1024 skip 0.42 (model 0.44) |
| sort emit, R4D8 | +0.14 to +0.18 | this model (calibrated x0.75 / raw) |
| host unit list (upload of surviving unit ids) | ~+0.19 | t493 chunk list; units are 16x more entries, could be built on device |
| pfwc reader gather (16 x 256 B reads per plane per tile instead of 1 contiguous) | +0.00 to +0.25 | not measured |
| pfwc writer mask transpose for g1 lane order | ~+0.01 | walk of a 16x64 bit-transposed mask (writer_pfwc_split.cpp set-bit walk); no L1 data shuffle |
| **net (negative = faster)** | **about 0.0 to -0.3, central ~-0.15** | |

The central case clears the 0.1 ms bar; the low end does not. The two unmeasured rows decide it, and
the gather cost can be measured on device without touching the sort (pfwc reads u64 unit lists in
today's order, b2b). If that probe shows pfwc saving >= 0.35 ms (gross minus gather), build u64g1 with
the transposed writer walk and run the full U3 A/B; otherwise shelve U3 with this model as the bound.

## Reproduce

```sh
cc -O2 -shared -fPIC -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk \
   -o /tmp/emit_sim.so docs/emit-burst-model/emit_sim.c      # (macOS 26 SDK tbd fails to link)
for v in 0:6 6:12 12:18 18:24 24:30; do EMIT_SIM_SO=/tmp/emit_sim.so EMIT_LS=4,8 \
  python3 docs/emit-burst-model/model.py scenes/bicycle.ply --views $v --jobs 6; done
for v in 0:10 10:20 20:30; do EMIT_SIM_SO=/tmp/emit_sim.so EMIT_LS=4,8 python3 docs/emit-burst-model/model.py \
  scenes/bicycle.ply --views $v --jobs 6 --opts today,u64_s8,u64g1_s8,u64g32_s8 --tag s8; done
python3 docs/emit-burst-model/analyze.py 4 > docs/emit-burst-model/tables.txt
python3 docs/emit-burst-model/calib.py docs/emit-burst-model/out-C110-v0-10.json
```

The model outputs (out-C*.json, 13 MB) are not committed; a copy is in the project archive
(tt-project/state/archive/worktrees/t637/). About 3 minutes per 6 views on a 14-core Mac (24 GB; do not run two at once).

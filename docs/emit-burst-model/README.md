# t637: emit-burst model for U3 (Morton reorder + cull), CPU only

**Question.** #631 bounds U3 (Morton reorder + per-chunk cull for pfwc) at 0.44 ms/view gross, but
three device runs (#470, #479, #493) lost 0.14-0.41 ms/view because Morton gids make the onelaunch
sort emit burst into one screen tile. Is there an order that keeps the emit within a ring that fits
L1 (R/D 4/8 or 8/8; 16 does not fit, t493) and nets >= 0.1 ms/view b2b?

**Answer.** VERDICT_PLACEHOLDER

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
loss = emit +MCULL48 + host list 0.19 - pfwc gross GROSS_MC = NET_MC ms; measured +0.177.

## Results (30 views, C = 120, L = 4)

TABLES_PLACEHOLDER

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

FINDINGS_PLACEHOLDER

## Net estimate (b2b ms/view, best option)

NET_PLACEHOLDER

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

About 3 minutes per 6 views on a 14-core Mac (24 GB; do not run two at once).

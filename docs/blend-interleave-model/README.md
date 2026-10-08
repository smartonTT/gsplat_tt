# Blend in the mat gaps: feasibility and model (t419)

Question: can TRISC1's idle time between mat cull jobs be filled with blend work, how much would
that save per view, and is it worth a device build (gate: at least 0.2 ms/view after the expected
mover speedup)?

**Answer: no. Shelve it.**
- The model gives 0.03–0.08 ms/view for a version that can actually be built.
- Even a perfect, idealised version gives at most 0.27 ms today, and only 0.13–0.16 ms once the
  movers get 30–50% faster.
- The L1 memory that blend would need while mat runs is not available.

**These are modelled numbers, not measurements.** The model is built from the #413 Tracy traces
(p100a yyzo-bh-04, 30 views, 110 cores).

## Files

- `model.py`: reads the #413 traces, works out each slab's blend cost and ready time, and simulates
  the mat+blend phase with and without gap filling.
  - Run: `python3 docs/blend-interleave-model/model.py`. It takes about 2.5 minutes and caches
    the parsed traces in `/tmp/t419_views.pkl`.
- `out/model.txt` and `out/gain-table.csv`: the output.

## What the traces show

| item | value |
|---|---|
| between-job gaps per core per view | 0.469 ms mean (0.904 max), about 9 gaps per core |
| single gap length, p10/p50/p90/p99 | 0 / 16 / 157 / 304 µs |
| blend cost per slab on the TRISCs (inferred), p10/p50/p90 | 67 / 265 / 846 µs, mean 362 |
| slabs per view | about 1100 (about 10 per core) |
| model calibration | simulated slowest-core blend end 5.95 ms vs measured 6.08 ms (−0.13 ms) |

- **Most gaps are shorter than one tile's blend.** The median gap is 16 µs. The median slab takes
  265 µs to blend.
- **Gains spread across all cores.** Each core claims blend tiles from one shared counter, so the
  cores finish within about 0.1–0.2 ms of each other. Gap-fill time on any core therefore cuts the
  frame end by about (fill time) / 110.
- **The start gap is out.** Before each core's first mat job, almost nothing is ready to blend: at
  most about 19 µs per core today and under 1 µs at s=0.5. The #413 "crit" numbers (−0.83 ms and
  −0.24 ms) counted this start gap, which is why they were too optimistic.

## Modelled gain (ms/view, mean of 30 views)

In the table:
- `s` is how much the movers shrink the mat gaps: 1.0 is today, 0.7 is 30% shorter gaps, 0.5 is
  50% shorter.
- "mover only" is what the shorter gaps alone save versus today, with no interleave.
- "gain" is what the interleave saves on top of that, at the same `s`.
- "wr deferred" means the BRISC u8 pack and image write for interleaved tiles waits until mat
  ends.

| case | s=1.0 | s=0.7 | s=0.5 |
|---|---|---|---|
| mover only (no interleave) | 0 | 0.325 | 0.541 |
| whole tiles in gaps, 32 KB slot (≤1024 records), write right away | 0.057 | 0.031 | 0.015 |
| whole tiles, 32 KB slot, wr deferred | 0.078 | 0.046 | 0.026 |
| whole tiles, 128 KB slot (≤4096 records), wr deferred | 0.076 | 0.044 | 0.026 |
| whole tiles, any size, wr deferred | 0.076 | 0.044 | 0.026 |
| same, random cost-to-slab mapping (sensitivity) | 0.140 | 0.062 | 0.027 |
| **upper bound**: split tiles, save/restore DEST at each switch, wr deferred | **0.277** | **0.162** | **0.127** |

Model settings:
- Switching between mat and blend costs 2 µs each way (LLK re-init).
- Saving and restoring DEST costs 4 µs per switch (t311 measured 2–5 µs).
- The NCRISC reader costs 5 µs per slab.
- The BRISC writer costs 30 µs per tile. This is an estimate from the scalar u8 pack code, not a
  measurement.
- Picking which tile to fill a gap with is an oracle best fit. A real scheduler would do worse.

Where the gain goes:
- **The two levers are not additive.** At s=0.5 the mover speedup takes 0.54 ms and leaves
  0.03–0.13 ms for the interleave.
- **Whole-tile filling** fills only about 0.1 ms of gap per core. Few gaps are long enough for a
  whole tile, and the movers that fill a slab are the same movers mat waits on.
- **The upper bound** fills about 90% of all between gaps with perfectly divisible work. Even then
  it misses the gate once the movers are faster, and it needs things that do not exist today (see
  below).

## Feasibility, per question

1. **Can blend run per tile, and stay bit-exact?**
   - The order can be kept. A tile's output depends only on its own records in their sorted
     order, not on which core blends it or when.
   - So the rule is:
     - Switch to blend only at a tile boundary.
     - Blend only single-subchunk tiles whose one slab is ready.
     - Leave multi-subchunk ("fat") tiles to the normal post-mat pass.
   - That keeps md5 906e0435 (11x10) and 39d84b28 (12x10).
   - Tiles need a "taken" state so the normal claim loop skips them. For example, the DEST-ready
     flag could hold epoch+1 once a gap-filler takes the tile, and the post-mat claimer skips
     those tiles.
   - Splitting a tile across gaps (the upper-bound row) needs an exact fp32 spill and restore of
     the DEST blend state. That path has never been built or tested for md5.
2. **What would it cost?**
   - **L1 is the blocker.**
     - Blend's CBs are aliased onto mat's CBs (t289: the fused program was 21 KB over L1 before
       aliasing). Blend's bulk slab buffer (512 KB, on mat CB4), output tile (on CB6) and u8 image
       (on CB20) are in use by mat whenever mat runs.
     - About 0–34 KB of L1 is free, and the Tracy build needs most of that.
     - A dedicated 32 KB slot holds 1024 records. The model shows a bigger slot changes nothing
       (0.076 vs 0.078 ms).
   - **Code size: probably fine, not checked.** Both mat and blend code are already in the fused
     binary. The interleave adds a scheduler check in mat's job loop plus an out-of-line per-tile
     blend call, roughly 0.5–1 KB. The kernel config buffer is 70,656 B and the fused build is near
     it, so a compile-only size check would be the first step if this is ever revived.
   - **Sync:**
     - All five RISCs must agree on every mode switch: NCRISC loads the slab, the TRISCs blend, and
       BRISC writes or defers. This is the KI-2 deadlock class.
     - Each switch costs LLK re-init on the TRISCs, and with splitting also a DEST save and
       restore.
     - It needs a few new semaphores or L1 mailbox words.
   - **Mover cost:** mat is limited by its movers (#413), so every interleaved slab costs NCRISC
     reader time that mat was waiting on. Writing right away costs about 30 µs of BRISC per tile.
     Deferring the write needs about 6 KB of L1 per held tile, and that L1 is not free either.
3. **Modelled gain:** see the table. The best buildable case is 0.078 ms today and 0.026–0.046 ms
   after the mover speedup. That is far under the 0.2 ms gate.

## Recommendation

- **Shelve lever L1a** (blend interleave in mat gaps). Do not build it on device.
- **Put the effort into the mover speedups instead** (sort/perm on the between gaps, smaller first
  subchunk on the start gap). In the same model:
  - 30% shorter gaps save 0.33 ms/view.
  - 50% shorter gaps save 0.54 ms/view.
- **Revisit only if both of these hold:**
  - A later change frees L1 during mat (≥128 KB that is not aliased).
  - Measured gaps stay long, with many over 100 µs, after the mover work lands.
- **If revived, the plan would be:**
  1. Run a compile-only TRISC size check.
  2. Add a "taken" ready-flag state and whole single-slab tiles only, with deferred writes.
  3. A/B it on the p150.

## Caveats

- Per-slab blend cost is inferred from the reader's 2-slot prefetch, not traced per tile. The
  random-mapping row shows the sensitivity to that inference: at most +0.06 ms, and only at s=1.0.
- Mover cuts are modelled as a uniform shrink of the start and between gaps. Real speedups may be
  uneven, but the interleave gain only falls as the gaps shrink.
- The traces come from p100a. The p150 has more cores (12x10 eth), so the per-core share of any
  fill is a little smaller there.

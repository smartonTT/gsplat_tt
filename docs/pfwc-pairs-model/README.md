# Writers emit the K2 pairs: host model (task #384, no device)

**Verdict: shelved.** The modelled net gain on bh-30 is about −0.05 ms/view at the
central cost estimate, below the 0.15 ms gate. md5 can be kept.

All numbers are **modelled, not measured**. The inputs are measured captures already in
the repo. Model: `model.py`, output: `model-out.txt`. Host test:
`tests/unit/test_pfwc_fuse.cpp` (`check_writer_pairs`).

## Idea

Today the pfwc writers write the compact streams plus a per-core counts page. K2
(`tile_assign_scatter_seg.cpp`) then reads every core's count, builds the segment
table and emits the (gid, tid) pairs. In the idea tested here:

- The writers emit the pairs themselves. Core c writes them at `c * cap + lofs`, in
  per-core fixed-capacity segments.
- The writers also count per-tile rows per *unit*. A unit is a contiguous chunk-order
  run of one core's chunks.
- `sort_ol_prefix` reads the unit rows directly, so K2 and its launch go away.

## (a, b) Order and md5: kept, by construction

Host test result: `t384 writer pairs: 96 unit cases md5-identical (U = 1, 2, 4, 8;
canonical + shuffled mover order), 72 overflow grow+rerun; parity units reorder ties in
6 scenes (25190 pairs)`.

- **Units in chunk order keep md5.** The final (tile, depth, gid) list equals the
  legacy K2 + sort order exactly, ties included, for 1, 2, 4 or 8 units per core.
  - Chunk k is written by writer `k & 1` (pfwc_wsplit). Both writers count into the
    unit's row, as two arrays summed before the prefix.
  - A pair's slot comes from its gaussian's lofs, so storage order is gaussian order
    whichever writer wrote it.
  - The prefix runs over units in (core, unit) order. Movers may take whole units in
    any order (checked with canonical and shuffled orders).
  - Half the trials force ties (depth & 3).
- **Units per writer parity break md5.** With rows (core, BRISC) and (core, NCRISC),
  the order changes exactly at pairs with the same tile and equal depth key that come
  from different-parity chunks of one core. All 6 forced-tie scenes changed; no
  difference ever appeared outside a (tile, depth) tie (checked).

## (c) Capacity and overflow

- **Cap:** each core needs `cap >= p_c`. At the hero view, max per-core pairs are about
  1.19 × the mean (t197: mean 22.4k, max 26.7k). A cap of about 1.5 × the mean pairs
  per core leaves margin: 110 × 33.6k × 8 B ≈ 30 MB of pair buffer, versus about 20 MB
  dense today.
- **Overflow:** a writer whose `pr` passes the cap stops writing pairs but keeps
  counting. It flags overflow and `p_c` in the counts page. The host grows the cap to
  the page-rounded max `p_c` and reruns, the same pattern as pair-overflow-t213. The
  test starts at the mean (overflows in 72 cases), checks the flags and the reported
  max, then grows, reruns and gets the identical order.

## (d) Cost: where the gain goes

All on bh-30, iter-209 Tracy (`docs/p150-blend-gap/out/t366-dev.csv.gz`), ms/view.

**Gross saving: 0.835 ms.** That is `sort_ol_prefix` start (2.739) minus `k2_pairs`
start (1.904): the sort could start where K2 starts today.

**1. Writer growth.**
- K2 costs 88-97 cycles per pair. Calibration: bh-30 88 (mover mean 0.726 ms),
  p100a t197 97.
- The writer would do the same per-pair loop: tid, 2 L1 stores, count increment, and
  2 NoC page writes per 16 pairs. It skips only K2's lofs/box read stream, about 1 read
  per 12 pairs.
- So the writer's per-pair cost is a fraction `f` of K2's, plausibly 0.6-0.9.
- The heaviest core carries 1.19 × the mean pairs. Its writers are already busy for
  1.67 ms (t221 split, max core), against a pfwc TRISC floor of 1.82 ms (bh-30 max).
  So almost all of the added work lands on the critical path.

**2. Sort emit, unit-locked.**
- Movers can only take whole units, so they lose the free speed-proportional split.
- The model gives each mover its bh-30 effective speed and assigns units with LPT.
  Compared with a dense split calibrated to the same speeds:
  - halves (220 units): +0.75 ms
  - quarters (440): +0.25 ms
  - eighths (880): +0.07 ms
- Unit sizes are synthetic: per-core t197 pairs, split with sd 12.7 %.

**3. Prefix.** It reads one row per unit, so it grows linearly. Today's 220 rows take
0.081 ms (max mover).

| per-pair cost f × K2 | pfwc + | NET halves | NET quarters | NET eighths |
|---:|---:|---:|---:|---:|
| 0.30 | 0.130 | −0.05 | +0.37 | +0.39 |
| 0.45 | 0.271 | −0.19 | +0.23 | +0.25 |
| 0.60 | 0.413 | −0.33 | +0.09 | +0.11 |
| **0.75** | **0.554** | −0.47 | **−0.05** | **−0.04** |
| 0.90 | 0.696 | −0.61 | −0.19 | −0.18 |
| 1.00 | 0.790 | −0.71 | −0.29 | −0.27 |

- **Break-even:** the 0.15 gate needs f ≤ ~0.52, i.e. ≤ ~50 cycles per pair in the
  writer. That would mean about half of K2's per-pair cost is its read stream. K2
  issues about 1 read per 12 pairs against 2 writes per 16, so that is unlikely.
- **Variant without writer counts:** the writers emit pairs and the sort counts them
  itself. The gain is the K2 window (0.81 ms) minus the unfolded count pass (0.79 ms on
  p100a, t170): ≈ 0.
- **Revisit only if** a K2 profile shows the read stream at ≥ 50 % of K2 time, or the
  writers' max-core busy drops by ≥ 0.4 ms.

## Side finding (bigger than this lever): the bh-30 sort emit is mover-imbalanced

- **Data.** On bh-30 (iter-209), `sort_ol_emit` per mover has median 1.63 ms and max
  2.65 ms. The slowest movers are the whole y = 3 row, both RISCs: 2.49-2.65 ms.
- **Why.** The page split uses `kMoverSpeedP150`, which t174 measured on a **p100a**
  (yyzo-bh-07). Its slow movers are NOC0 rows y = 2, 3 and NOC1 columns x = 14, 15.
- **Model.** Re-weight the pages by the bh-30 effective speeds (table speed / measured
  time). The emit makespan drops from 2.65 to 1.61 ms, a modelled gain of up to
  ~1.0 ms/view on bh-30.
  - This assumes emit time is proportional to pages / speed. It does not hold if the
    y = 3 slowness comes from the receiving side.
  - The capture predates iter-210, whose 513-page tile stride changed the DRAM layout.
  - So recheck it on an iter-210 Tracy capture first.
- **Consistent with** the iter-210 stage numbers: sort is 1.031 ms on bh-30 vs
  0.50 ms on the p100a.
- **Follow-up:** a per-board mover-speed table, measured on bh-30 with
  `opt/profiler/emit_cores.py --weights` and picked by board type.

## Reproduce

```
python3 docs/pfwc-pairs-model/model.py > docs/pfwc-pairs-model/model-out.txt
python3 docs/pfwc-pairs-model/zones.py docs/p150-blend-gap/out/t366-dev.csv.gz k2_pairs sort_ol_emit pfwc
CXXFLAGS="-Irender/kernels/dataflow -ffp-contract=off" tests/unit/run_cpp.sh tests/unit/test_pfwc_fuse.cpp
```

The test takes about 70 s; the pre-existing checks take most of it. The t384 lines are
printed to stdout, which run_cpp.sh discards; build the test by hand to see them.

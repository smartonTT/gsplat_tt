# t114: device A/B of the one-launch sort (GSPLAT_TT_SORT_ONELAUNCH, #106)

Code b20217e (base 840fff0 + scripts), yyzo-bh-07 p100a, bicycle, 30 views.
Full log: `drive-b20217e.log`.

## Correctness
- Checked run (ONELAUNCH=1, CHECK=1, 5 views): `[SORT] ONELAUNCH_CHECK bad_tiles=0`
  on all 6 frames (warm-up included), md5-identical to md5-r82new.txt.
- Watcher run (TT_METAL_WATCHER=2, SFPU_VIS=0, 3 views): no assert, bad_tiles=0.
  The watcher can't run with PFWC_VIS on: the program grows to 76128 B, over the
  70656 B kernel config buffer.
- Hang check: 3 back-to-back 30-view runs with the knob on. No hang.
- Every knob-on and knob-off run: all 30 views md5-identical to md5-r82new.txt,
  hero_vs_ref 100 dB.

## A/B (3 interleaved rounds, ms/view)

| round | base frame | on frame | base sort | on sort | base blend | on blend |
|---|---|---|---|---|---|---|
| 1 | 29.55 | 28.38 | 10.70 | 8.94 | 10.68 | 11.29 |
| 2 | 29.54 | 28.62 | 10.69 | 9.12 | 10.66 | 11.28 |
| 3 | 29.55 | 28.44 | 10.67 | 9.04 | 10.64 | 11.25 |
| mean | **29.55 (33.8 FPS)** | **28.48 (35.1 FPS)** | 10.69 | 9.03 | 10.66 | 11.27 |

Net: **-1.07 ms/view**. Sort is 1.66 ms faster, but blend is 0.61 ms slower
because the materialize now also sorts the in-budget tiles. The one-launch
program takes 8.70 ms (`[SORT] stage=ONELAUNCH onelaunch=`), plus layout
0.12 ms and host publish 0.19 ms.

**Decision: not adopted** (gate is >= 3 ms/view). The knob stays default off.

## Where the time goes (Tracy, 10 views, knob on)
- sort_ol_* program: 8.42 ms busy. Per-zone makespans: emit 8.26, count 1.01,
  barrier 0.49, prefix 0.05.
- The one-launch emit (8.26 ms) is far slower than the legacy emit at the tip
  (bin_emit 5.29 ms). The legacy emit has #100's PB batching and RING write
  coalescing; the one-launch emit does not. This gap (~3 ms) is the main reason
  the expected sort of ~5-6 ms was missed.
- sort_subchunk_mat+mat_cull_mask: 3.75 ms. Blend: 7.60 ms. Idle between programs: 1.35 ms.

## Materialize (GSPLAT_TT_MAT_STATS=1)
~1040 items on 220 slots. Big-tile cost is 1.63-1.77 M out of 2.80-2.93 M
total (~59%). max_item is 31-33 k against a mean slot of 12.7-13.3 k, and
max_ncrisc equals max_item. So one big-tile item sets the materialize
makespan, at ~2.5x the mean slot. **Big-tile materialize dominates.**
The cached sorted index / split big items follow-up is justified.

## Emit write coalescing
In the legacy emit, PB+RING (with PUBOC) cut bin_emit from 8.8 to 5.29 ms.
The one-launch emit already has PUBOC, and its 8.26 ms emit is ~3 ms over the
legacy 5.29 ms. Porting PB+RING could close most of that gap. If it does, the
one-launch net would be roughly -3.5 to -4 ms/view. This is an estimate and
needs a device run. RING needs ~256 KB of L1 staging per mover next to the
count page cache, so L1 is the risk.

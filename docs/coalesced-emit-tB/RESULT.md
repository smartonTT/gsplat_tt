# T-B (#534): coalesced bin emit, CPU model (no device)

**Verdict: NO-GO for B1's device stage.** The predicted saving is about 0.0-0.1 ms/view if B1 stays
inside the sort, and about -0.1 to +0.15 ms/view if staging moves into pfwc. Both are below the
0.4 ms bar. The only case that clears it (about 1.0 ms) assumes pfwc places 2.65M records per
view at no cost. pfwc is TRISC-SFPU-bound (t224), so that assumption does not hold.

Model: `model.py`, output `model_out.txt`. Host float64 projection
(docs/precull-t140/estimate.py) with today's PRECULL=2 pixel-centre rect (the "pc" arm of
docs/precull-t157/analyze.py). The pair list is gaussian-major in ply order, with row-major
tiles inside each rect. It is split evenly over 110 cores × 2 movers in 16-pair pages, as
`sort_onelaunch_layout.h` does. Bucket bases come from `device_prefix`. Bursts are also cut at
the bucket's 64-record (2 KB) DRAM pages. These are estimates, not measurements.

Calibration: the model's median is 2.727M pairs/view; t224 measured 2.72M on device. The model's
largest tile is 22,976 records, the same as t157.

## Per core (30 views × 110 cores = 3300 core-views)

| | mean | p95 | max |
|---|---|---|---|
| kept pairs per core (even page split) | 24,080 | | 29,056 |
| kept pairs per core if split by equal gaussian count (pfwc-style) | | | 63,837 |
| tiles touched per core | 933 | 997 | 1024 |
| records per (core, tile) run | 25.8 (median 14) | 94 | 407 |

Runs are short. Half of them hold 14 records or fewer, because each core's contiguous slice of
gaussians covers almost every tile.

## L1 needed to stage 32 B records (KB per core)

Free L1 next to pfwc's CBs is about **920 KB**:

- 1536 KB total
- minus 39.5 KB for the memory map (MEM_MAP_END 40448 B)
- minus 93 KB of kernel config (69 KB plus 24 KB for the split writer)
- minus 482 KB of pfwc CBs, read from pfwc_device.cpp in the default rec32/fuse/wsplit/vis
  config: 9 input × 3 × 4 KB, 8 output, 10 scratch, 3 conic, 6 vis, 10 odd × 2 × 4 KB, the
  CB_OP, 2 × 33.5 KB staging and the MBX

| staging | mean | p95 | max | core-views over 920 KB |
|---|---|---|---|---|
| whole (core, tile) runs, even pair split | 753 | 906 | 908 | 0 |
| whole runs, equal-gaussian split | 753 | 914 | 1995 | 147 |
| linked chunks of K = 8 | 883 | 1035 | 1049 | 1135 |
| linked chunks of K = 16 | 1028 | 1170 | 1195 | 2645 |
| linked chunks of K = 32 / 64 | 1356 / 2108 | | 1523 / 2327 | all |

- Whole-run staging needs per-tile counts before placement: a count pass, or the K2 count. With
  a pair-balanced split it fits, but only with 12 KB of headroom on the worst view (orb_10). It
  does not fit with pfwc's gaussian split.
- Chunked staging needs no count, but it does not fit at all.
- Inside the sort program, per-mover tile rings cost 1024 × (32K + 4) B: 266 KB at K = 8 (today),
  528 KB at K = 16. Today's sort uses about 2 × 567 KB of L1 per core, which leaves about
  290 KB, so the rings can grow to about K = 12. Full per-mover runs (908 KB per core) do not
  fit next to today's windows.

## Bursts and predicted write time (mean view)

Rates come from docs/hw-ceilings.md: 78 cycles per write on a mover, a chip-wide cap of
1.78-2.27 G writes/s, and 260 GB/s DRAM write on one NoC up to 512 GB/s (p150 published peak).

| emit | writes/view | mean burst | write floor ms |
|---|---|---|---|
| 32 B scatter (no ring) | 2.65M | 32 B | 1.17-1.49 |
| **today: per-mover rings, K = 8** | 463k | 5.7 rec, 183 B | **0.20-0.33** |
| K = 16 | 315k | 8.4 rec | 0.17-0.33 |
| K = 32 | 250k | 10.6 rec | 0.17-0.33 |
| whole (core, tile) runs | 142k | 18.7 rec, 597 B | 0.17-0.33 |

Burst length histograms, by record-count bins up to 1, 2, 4, 8, 16, 32 and 64, are in
`model_out.txt`.

The plan's 1.2 ms bound compared bursts with a 32 B scatter. The rings of 8 (t479) already
replaced that scatter, so today's write floor is about the same as the burst floor: both are
bandwidth-bound at 0.17-0.33 ms. Full bursts save at most about 0.04 ms of chip write time and
about 0.085 ms of mover issue time per mover. The movers have about 0.35 ms of slack (t215), so
neither is on the critical path.

## Why the emit does not shrink

The emit zone is 1.228 ms. It is set by the TRISC pack: 178.8 cycles per record, 1.05 ms net
(t215). That cost is L1 traffic: the list load, 8 blendrec loads, sub_int32, 8 ring stores and
the run words. Coalescing the writes removes only the ring and run-word bookkeeping, about
10-20 cycles per record, which is 0.06-0.12 ms.

- **B1 inside the sort:** about 0.0-0.1 ms.
- **B1 in pfwc** (place records into per-tile L1 runs as pfwc makes them, then burst after the
  prefix): this removes the sort's 1.459 ms window but adds:
  - the kept fixed prefix and barrier, about 0.27 ms
  - the burst floor, 0.17-0.33 ms
  - the placement on pfwc's TRISCs, which are SFPU-bound. At 120-160 cycles per record ×
    24.1k records per core / 3 TRISCs that is 0.71-0.95 ms.
  
  Net: 1.459 - 0.27 - 0.33 - (0.71..0.95) = **-0.1 to +0.15 ms**. It reaches 1.0 ms only if the
  placement is completely hidden. It also needs a pair-balanced pfwc split; see the L1 table.

## What would change the verdict

- A pfwc lane with at least 0.7 ms of idle that can place records. t224 shows pfwc
  TRISC-SFPU-bound, with reader waits near 0, so none is known.
- A pack that is not scalar L1 traffic per record.
- Fewer records. B2/T-A cuts pairs, and the pack scales with them.

# #365: p150 emit mover throughput (WIP — results pending)

Question: why sort bin_emit is ~2.4 ms/view on bh-30 (p150b) vs 0.57 ms on yyzo-bh-04 (p100a).
Note: the host `bin_emit` stage is the blocking CQ1 read of the K2 rows (ends when the emit ends, #356);
the device emit window itself is 2.92 ms (p150) vs 1.34 ms (p100a), per-mover mean 1.78 vs 1.18 ms.

## Hypothesis under test: DRAM bank camping of the bucket writes on 8 banks
The emit's ring flushes write <=256 B runs to the tile bucket, a 2 KB-page DRAM-interleaved buffer:
tile t's page k is global page t*512 + k (kTileCap 32768 records = 512 pages), bank (t*512 + k) % nb.
- p100a, nb = 7: 512 % 7 = 1, bank = (t + k) % 7, so each mover's writes rotate over all 7 banks.
- p150, nb = 8: 512 % 8 = 0, bank = k % 8 for every tile. A mover's slots in tile t start at its prefix
  base (cores in order), so movers early in the order (rows y=2-4) write page 0-few of every tile:
  banks 0..2 only. That matches the Tracy map (rows y=2-4 slow on every column, #358 emit-B.txt).
Fix candidate: `GSPLAT_TT_OL_TILE_PAD` / `sort_onelaunch::bucket_tile_cap`: stride = next page count
coprime with the bank count (auto: 513 on 8 banks, 512 unchanged on 7). Arms: P0 = pad 0 (pre-#365),
P1 = auto.

## Microbench (step 0)
`render/bench/kernels/p10_emit_noc.cpp`, `MB_SUITE=emit render/bench/build/risc_microbench`:
R64_d16 / R256_d8 reads (64 B-page buffer, in-bank runs), W256 / W32 writes to a 2 KB-page bucket
at stride 512 vs 513 pages, BRISC (NoC0), NCRISC (NoC1), both; one core and the 11x10 grid.
Per-core bytes per RISC tick maps: `[MBMAP]` lines in `out-*/mb.txt`.

Scripts: `drive_bh30.sh` (sync/build, bench with viewer stop/restart), `drive_p100a.sh`, `bench365.sh`.

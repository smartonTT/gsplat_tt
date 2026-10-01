# Task #105 — lever 3: sort enumerates the (gid, tid) pairs itself (not adopted)

GSPLAT_TT_FUSED_PAIRS (default 1 on this branch only, 0 = kill switch). tile_assign skips K2
(ta_bucket_scatter); sort_bin hist/emit enumerate pairs from proj_m_offs + proj_m_aabb
(SORT_FUSED_PAIRS). All runs are bicycle, 30 views, md5-identical to md5-r82new, 100 dB, on yyzo-bh-07 p100a, untraced.

## Old base (72cd487 + lever 3, 9bc4a69), 3 interleaved rounds
| | view ms (mean) | tile_assign | bin_count | bin_emit |
|---|---|---|---|---|
| on  | 31.467 | 0.005 | 1.06 | 8.92 |
| off | 32.655 | 1.44  | 0.92 | 8.84 |
Delta -1.19 ms/view (-3.6%).

## Current tip (c5fed7d, #100 emit rewrite), 4 interleaved rounds (r10 3fbd9fa, r11-r13 0b139bf)
| round | on | off |
|---|---|---|
| r10 | 29.409 | 29.485 |
| r11 | 29.297 | 29.671 |
| r12 | 29.508 | 29.426 |
| r13 | 29.333 | 29.549 |
| mean | 29.387 | 29.533 |
Delta -0.15 ms/view (noise level). Stages with it on vs off: tile_assign 0.005 vs 1.44, bin_count 1.06 vs 0.91,
bin_emit 6.31 vs 5.29.

## Why the gain is under the 3 ms gate
- K2 costs only 1.43 ms, so that is the most removing it can save.
- The count pass now enumerates the pairs, which adds back about 0.15 ms.
- After #100, the emit's batched pair reads are already hidden behind the pipeline. Enumerating pairs synchronously inside
  issue_pairs, plus a global noc_async_read_barrier on each 32-page aabb window (which also waits on in-flight
  blendrec/pair batches), adds about 1.0 ms to bin_emit. Making the fill run-based did not change that, so the cost is
  latency and the barrier, not per-pair ALU work.
- Even with a fully async, double-buffered window, the ceiling is about 1.3 ms/view.
- Making the pairs cheaper would not cut emit DRAM traffic. The gid/tid reads are a small part of what #98 measured;
  the record writes dominate.

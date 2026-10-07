# #358: p150 mover speed table — measured, loses, auto keeps the p100a table

Goal (from #356): the p150's sort bin_emit is 2.4 ms vs 0.57 ms on the p100a; re-weight the
emit mover split with a p150-measured speed table to cut ~1 ms.

## What was built
- `render/host/sort_mover_speed.h`: `kMoverSpeedP150[]` (bh-30, from
  `docs/p150-gap-t356/out/reweight-E.txt`), next to the p100a table. `mover_board()` is read once
  and shared by the sort and the K2 fold. `GSPLAT_TT_MOVER_TABLE=p100a|p150` forces a table.
- After the A/B below, auto (`mover_board_for_cluster`) returns the p100a table on every board,
  p150 included, so the default behaviour is the pre-#358 one. Test: `tests/unit/test_mover_table_t358.cpp`.

## Results (ms/view, 30 views, md5 906e0435 on every run)
Arms: A = p100a table, B = p150 table, E = even split (`GSPLAT_TT_OL_MOVER_SPEED=0`).

| board | arm | runs | ms/view | sort bin_emit |
|---|---|---|---|---|
| bh-30 p150b | A | 12.723 12.689 12.869 12.932 12.855 | 12.81 | 2.41-2.48 |
| bh-30 p150b | B | 13.473 13.561 13.472 | 13.50 | 2.78-2.83 |
| bh-30 p150b | E | 15.433 15.192 | 15.31 | 5.06 |
| yyzo-bh-04 p100a | A | 10.986 10.891 10.912 | 10.93 | 0.57 |
| yyzo-bh-04 p100a | B (auto = p100a there) | 10.851 10.884 10.977 | 10.90 | 0.56 |

Device hero (bh-30, every arm, same md5): 42.51 dB vs `benchmarks/reference_v2/hero.png`;
diff `out-p150/hero_diff10.png` checked by eye, no tile seams.

## Why the re-fit fails (Tracy EMIT capture of arm B, `out-p150-diag/emit-B.txt`)
- Emit window with the p150 table is 3.12 ms; the last mover is (11, 3) NCRISC in 25 of 31
  launches. The re-fit moved pages onto x = 11-12, y = 2-4 (2.8-3.2 ms) while x = 13-15 finish
  at 1.1-2.0 ms. The speed of a core depends on the split itself (shared NoC/DRAM contention),
  so a one-shot `v = share / time` re-fit does not converge.
- Even split costs 5.06 ms: the p100a table already does most of the balancing on the p150.
- Even a perfect balance cannot close the gap: the p150 movers are ~4x slower in aggregate
  (2.4 vs 0.57 ms for the same pages). The cause is per-mover throughput on the p150
  (DRAM/NoC path), not imbalance. Next lever: find why emit reads/writes are 4x slower on the
  p150 (DRAM bank/channel placement of the emit buffers, NoC choice per row), not re-weighting.

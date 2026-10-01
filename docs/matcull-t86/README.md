# Task #86: sort_subchunk_mat + SFPU cull attribution (written up by task #90)

#86 was cancelled after two run timeouts and an API outage; this is what its
branch (2603c40, 9abff74, d9373a6; rebased as 3895efa, 4c16332, d8117a5) and notes contain. Board: yyzo-bh-07 p100a,
bicycle views 0-9, Tracy with `GSPLAT_TT_MATCULL_PROF=1` (fine zones, default
off) at 9abff74 (= a5f2bd6 + zones). Script: `opt/profiler/matcull_breakdown.py`.

## Windows (per view)

| program | window ms | notes |
|---|---|---|
| sort_subchunk_mat | 3.54 | BRISC busy 2.67, NCRISC busy 2.78; critical mover NCRISC 7/10 views |
| SFPU cull (tile_l1_cull) | 1.31 | all 5 RISCs busy ~1.2 ms; BRISC writer is the last to finish |
| blend | 7.48 | |

## Mat: critical-mover breakdown (ms/view)

| zone | ms | what |
|---|---|---|
| mat_rd + mat_ov_rd | 0.56 + 0.86 = **1.42** | bucket reads, 64 B PACK2 pages (one NoC read per 2 records) |
| mat_sort + mat_ov_sort | 0.55 + 0.66 = **1.21** | in-L1 adaptive radix (sort_radix_tile_algo.h) |
| mat_perm + mat_ov_perm | 0.15 + 0.19 = 0.34 | L1->L1 depth permute |
| mat_gather | 0.50 | blendrec gather (sc>=1 of over-cap tiles) |
| mat_wr, mat_meta | 0.06 | slab writes are 2 KB pages already |

## Cull (ms/view, per-RISC means)

- TRISC0/1/2 `tile_mb_mask` 1.20, reader `cr_fill` (transpose to coeff tiles) 1.05,
  writer `cw_patch` (mask -> word3) 1.00. Slab DRAM read `cr_bulk` 0.11 and
  write-back `cw_wr` 0.09: the cull is NOT DRAM-bound; its window is the
  SFPU/transpose/patch pipeline.

## Change left unmeasured

- d9373a6: buf_l1_recs and the in-cap overflow region use 2 KB DRAM pages
  (64 records) instead of 64 B PACK2 pages, so the materialize bucket read is
  ceil(n/64) NoC reads instead of n/2. sort_bin writes 32 B records either way.
  Measured by task #90 (docs/matcull-t90/README.md).

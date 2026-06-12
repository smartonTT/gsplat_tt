# Iter 70 (ttw-070) — standalone tile_mb_mask path (no fuse)

**Status:** **reject** — gate PSNR OK; wall-clock and Tracy success bar not met.

## Hypothesis

Optimize standalone tile L1 cull / mask (not `MB_FUSE_TILE_L1_CULL`):

1. `microblock_cull_compute.cpp`: copy box ramps once per tile (`processed==0`), not per 32-gaussian batch.
2. Skip phased SFPU `cull_dispatch` when entire batch is `thr_pre_culled` (op≤floor).
3. `writer_tile_l1_mask.cpp`: branchless `mask_from_keep()` (same perm geometry, no `perm()` calls).

## Device (yyzo-bh-07)

| Attempt | bin | Result |
|---------|-----|--------|
| 1 (iterate, no sync) | `d3754d74d3a307b3` (= iter 68) | **Invalid** — remote kernels stale; logged keep in error |
| 2 (sync + build + verify) | `f282b2575ac4d16d` | `hero_vs_ref` **63.63**, `ms_view` **296.41** |
| Tracy 30-view | same | See zones below |

## Gate vs iter 68

| Metric | iter 68 | iter 70 (valid) | Δ |
|--------|--------:|----------------:|---|
| `hero_vs_ref` | 63.63 | 63.63 | 0 |
| `ms_view` | 296.58 | 296.41 | −0.17 ms (−0.06%) |

Need **≥2%** on `tile_mb_mask` or `tile_l1_cull_rd` or `ms_view` → **fail**.

## Tracy (30-view, `analyze_device_zones.py` sum/30)

| Zone | iter 68 (`ttw-068-tracy`) | iter 70 (`wrap_out_ttw-070`) | Notes |
|------|--------------------------:|-----------------------------:|-------|
| SFPU cull | `tile_mb_mask` **31086.85** /view | `cull_global_mb` **25773.66** /view | Same 10230 events; iter 70 trace missing `TILE_L1_CULL` zone rename |
| Reader | `tile_l1_cull_rd` **10361.62** /view | *(absent)* | Reader zone not in iter-70 CSV |

Summed core-time on compute cull zone drops ~17%, but **ms_view flat** and reader zone missing → do not claim perf keep.

## Next

- Do not fuse into blend (iter 69 LRA blocked).
- Stronger levers: measure `batch_all_pre_culled` hit rate; reader skip emit for op≤floor tiles; overlap `tile_l1_cull_rd` with blend (iter 61 hang risk); or trim blend SFPU for LRA headroom before fuse retry.

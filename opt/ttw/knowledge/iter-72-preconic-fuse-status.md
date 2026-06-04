# Iter 72 (ttw-072) — reader PRECONIC + wired MB_FUSE_TILE_L1_CULL

**Hypothesis:** Move conic (cov→A,B,C) from blend compute SFPU to reader emit + per-g scalar preconic on L1 bulk; shrink `blend_one_gaussian_math` LRA so fused `tile_l1_cull_sfpu` links. First l1_bulk subchunk runs fuse cull before tile_regs init (no DEST stash); later subchunks still load DRAM masks from standalone `tile_l1_cull` unless fuse env skips that pass.

**Env:** `MB_FUSE_TILE_L1_CULL=1` at program build (recreate blend context after toggling).

**Code:**
- `render/kernels/common/mb_cov_preconic.hpp` — scalar cov→ABC
- Reader: preconic coeff emit; box ramps CB 14/15; skip DRAM mask load when `!MB_FLAG_CONTINUE`
- Compute: PRECONIC blend; `fuse_l1_cull_subchunk` + kernel_main reorder
- Host: fuse CBs, box DRAM upload, skip `tile_l1_cull::process_frame` when fuse

**Device:** (fill after yyzo-bh-07 runs)

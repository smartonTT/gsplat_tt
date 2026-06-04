# Iter 69 (ttw-069) — fuse tile_l1_cull into blend reader

**Status:** **blocked** — kernel-config / TRISC0 LRA ceiling (same class as iter 61 / e204f63).

## Hypothesis

`MB_FUSE_TILE_L1_CULL`: skip standalone `tile_l1_cull` 3-kernel program; reader bulk-loads PACK2 subchunks; compute runs SFPU `tile_mb_mask` on same L1 residency then `cp_l1_blend`; no DRAM `cull_masks` read.

## Device tries (yyzo-bh-07)

| Try | Result |
|-----|--------|
| 1 | JIT: `trisc0.elf segment[1] +0x768 overflows limit 0x720` (full `tile_l1_cull_sfpu.hpp` + batch arrays) |
| 2 | Same LRA overflow after per-gaussian cull + cache clear |
| 3 | Same hash `2850661745767925136`, same +0x768 overflow with minimal `fuse_cull_one.hpp` (V=0 only) |

Blend compute kernel is already at LRA budget; any fused SFPU cull body exceeds TRISC0 local region even without batch stack.

## Revert

All fuse edits reverted to iter-68 anchor (`53ac306`). Standalone `tile_l1_cull` + DRAM masks path restored.

## Next lever

- **Not** monolithic fuse into `alpha_blend_compute_mb` — blocked by kernel-config ceiling.
- Options: (1) keep separate `tile_l1_cull` but overlap reader bulk with blend via single-CQ ordering (iter-61 hang risk); (2) reader-side mask from precomputed DRAM only (no ~108ms win); (3) reduce blend compute footprint first to free LRA headroom.

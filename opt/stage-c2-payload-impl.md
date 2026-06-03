# Stage C2 — per-tile contiguous blend payload (first device step)

North star: blend reads **sequential** L1/DRAM after sort; no per-splat SoA gather on the mover.

## Current hot path (resident, `blend_mb_devcull_resident`)

- **Input:** `sort_sorted_ids`, `sort_tile_ranges`, `proj_m_*` SoA in DRAM.
- **Blend reader:** for each gaussian in tile, `noc_async_read_tile` on 9 SoA pages (a,b,c,px,py,op + colors) — **random gather**, mover-bound.
- **Cull:** scalar soft-float on mover (`compute_microblock_mask`) or shelved SFPU pre-pass (`cull_masks`).

## Target payload layout (~36 B / sorted candidate)

Per depth-ordered entry `i` in tile `t` (local index `p`):

| field | bytes | notes |
|-------|------:|-------|
| cov_a, cov_b, cov_c | 12 | raw conic (same as proj_m_a/b/c) |
| mean_x, mean_y | 8 | proj_m_px/py |
| opacity | 4 | proj_m_opacity |
| cr, cg, cb | 12 | 3× proj_m_colors pages |

Optional: pack `uint32 microblock_mask` (+4) if SFPU cull stays separate; fused kernel can keep mask in L1 only.

**Tile storage:** one contiguous DRAM slab per tile, base from exclusive prefix-sum over `per_tile_count[t]` (page-aligned to 64B). Register as `blend_tile_payload` + `blend_payload_base` (mirror `cull_mask_base`).

## Kernel split (incremental)

1. **S2a — `payload_pack.cpp` (device, DONE in tree)**  
   - Inputs: `sort_sorted_ids`, `sort_tile_ranges`, `proj_m_*`, `blend_payload` + `blend_payload_base`.  
   - Per tile (LPT tile list): pipelined 9-field gather, scalar `compute_microblock_mask`, 64B row write.  
   - Env: `GSPLAT_TT_BLEND_PAYLOAD=1`. Separate `g_ctx_payload` program.  
   - Host: blend still uses devcull reader until S2b; payload pass is additive for now.

2. **S2b — blend reader variant**  
   - Read payload sequentially (64B rows, 6–9 fields used).  
   - Scalar cull can remain on mover until SFPU fused.

3. **S2c — fuse with C2+D+E** (plan §4): single `tile_render` kernel; payload never leaves L1.

## device_state registration

```text
blend_payload       DRAM  (P_kept * 36 rounded to pages)
blend_payload_base  SoA   uint32 per tile, page-aligned offset
```

## Verify

- `a003_verify.py` resident env; `hero_vs_ref >= 63.85` ×2 on 30-view.
- Byte-compare payload vs host `build_mb_payload` for one tile (`GSPLAT_TT_PAYLOAD_VERIFY=1`).

## Effort / risk

| piece | ~lines | risk |
|-------|-------:|------|
| S2a writer + host gate | 250–400 | medium (gather pattern exists in cull reader) |
| S2b reader | 150–250 | low if layout matches host |
| device_state + sizing | 80 | grow-only like `cull_masks` |

**Do not** block SFPU cull fix; C2 is parallel track after SFPU gate or in parallel.

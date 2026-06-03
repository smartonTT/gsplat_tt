# Phase B — chunk fusion boundary

**Goal (endgame §3 Phase B):** One program per splat chunk: **BRISC bulk read → TRISC FPU project + frustum cull → BRISC bucket scatter**, killing redundant DRAM passes and separate `ta` / `ta_scat` on the hot path.

**Scope:** IDEAL path only — `GSPLAT_TT_TILE_BUCKET=1`, `GSPLAT_TT_SFPU_CULL=1`, `GSPLAT_TT_RESIDENT_*`, **not** `GSPLAT_TT_FUSED_TILE`. Gate: **63.85 dB** unchanged.

---

## What merges (target architecture)

| Stage today | Tracy zone | After Phase B |
|---|---|---|
| Device project (pfwc TRISC) | `mcam`, `pfwc` | Inside **chunk TRISC** (unchanged math, co-located with read) |
| `gather_visible_scatter` count | `proj_count` | **chunk** count (or fused into one launch) |
| `gather_visible_scatter` scatter → `proj_m_*` | `proj_scatter` | **chunk_scatter** (compact SoA + blendrec) |
| `tile_assign_bbox` K1 | `ta_gauss_aabb` | **Fused into chunk scatter** (same px/py/rx/ry in L1) |
| `tile_assign_scatter` K2 | `ta_bucket_scatter` | **Direct per-tile bucket scatter** from L1 (no `proj_m_*` reread) |
| Host / device TA prefix scan | (host / scan kernels) | **Device exclusive scan** on bucket or pair counts only |

**One Finish per chunk** (or back-to-back kernels on one CQ with no host between): read tile → project+cull on TRISC → write compact record + tile overlap count (+ eventually bucket offsets).

---

## What stays separate (for now)

| Piece | Why not merged yet |
|---|---|
| **TRISC project kernels** (`project_means_cam`, pfwc compute) | Still separate programs from gather; fusion needs multi-RISC **one Program** with BRISC+TRISC handoff via CBs |
| **`gather_scan_bases` / `ta` scan chain** | Exclusive prefix over `tiles_per_gaussian` still its own single-core pass until counts are produced in-chunk |
| **`tile_assign_scatter` K2** | Pair-centric `(gid,tid)` list; Phase B end state is **bucket scatter**, not pair list + `sort_bin` histogram |
| **`sort_bin` + `sort_radix_tile`** | Phase C — per-tile L1 sort/mask/blend mega-kernel |
| **Global `CULL_SPLIT`** | Phase C — move mb-mask into per-tile L1 |

---

## Shipped in this slice (`GSPLAT_TT_CHUNK_FUSION=0` default)

**B2 partial — fuse TA K1 (AABB / `tiles_per_gaussian`) into gather scatter:**

1. On scatter pass, for each visible compact Gaussian, compute `tiles_per_gaussian[m]` with the same formula as `tile_assign_bbox.cpp` (bit-exact int32).
2. Write into resident `ta_tiles_per_gaussian` DRAM (registered in `device_state`).
3. `tile_assign_tt` skips **K1** when fusion is on, resident inputs are on, and the buffer is present; scan/K2 unchanged.

**Env:** `GSPLAT_TT_CHUNK_FUSION=1` (opt-in). Requires `GSPLAT_TT_RESIDENT_TA_IN=1` and ideal gather path. `render_full_py` publishes tile grid via `device_state::set_chunk_fusion_tile_grid` before project.

**Not in this slice:** killing `ta_scat`, fusing TRISC project into chunk, direct tile-bucket scatter from gather, device scan for bucket bases (B4).

---

## Next steps (ordered)

1. **Verify** on device: `GSPLAT_TT_CHUNK_FUSION=1` + existing ideal stack → 63.85 dB; Tracy should lose one `ta_gauss_aabb` bar (~K1 time).
2. **Fuse K2 / bucket scatter** into gather scatter (write per-tile records using `sort_bin` emit layout); drop `proj_m_*` DRAM reread for TA.
3. **Single Program** BRISC+TRISC per chunk with sub-zones `chunk_read` / `chunk_project` / `chunk_scatter`.
4. **B4:** extend `gather_scan_bases` pattern to TA/bucket offset scan without host.

---

## Tracy rename (when fully fused)

See `opt/endgame-execution-plan.md` §4: `chunk_dram_read`, `chunk_project`, `chunk_bucket_scatter` replacing `proj_count` / `proj_scatter` / `ta_gauss_aabb` / `ta_bucket_scatter`.

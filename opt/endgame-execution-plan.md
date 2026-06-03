# Endgame execution plan — fix the pipeline, reach the roofline

**Written:** 2026-06-02, after ideal-path Tracy review (`render-ideal-labeled.tracy`).

**User target (unchanged):** scene upload once → per frame: chunk-in-L1 project+frustum-cull → scatter **full splat records** to per-tile DRAM buckets → per tile: **one** bulk DRAM→L1 load → sort + mb-mask + blend **entirely in L1** → framebuffer. Host only at frame boundaries (view constants + final D2H). No gather. No per-candidate random DRAM.

**What actually shipped and helped:** ideal `TILE_BUCKET` path (63.85 dB), host `Finish` collapse / pipelining (~159 ms/view @8v vs ~300+ gather path). **What is still structurally wrong:** the **stage graph** — too many separate programs, wrong RISC roles, **global cull** before blend, DRAM ping-pong between every phase.

---

## 1. How to read your Tracy screenshot (frame 2, ~30–270 ms)

| What you see | Zone name today | What it really is | Why it looks wrong |
|---|---|---|---|
| Two orange `proj` bars ~40–70 ms | `proj` (BRISC) | **Not two projections.** Typically **count pass + scatter pass** (or two waves of `gather_visible_scatter`), both labeled `proj` because only that kernel has a stage marker. **`mcam` / `pfwc` (real projection math) run on TRISC** but are **~1–2 ms total** — invisible at your zoom. | Should be **one** bulk read + FPU work + one scatter, not two BRISC programs. |
| Green `ta` then `ta_scat` | `ta`, `ta_scat` | **K1:** per-Gaussian AABB → keep mask. **K2:** scatter **(gid,tile)** or full records to buckets. Two **separate programs** + DRAM reread of `proj_m_*`. | Should be **fused into the project chunk** (AABB + bucket scatter while splats still in L1), or at least one DRAM pass. |
| Green `sort` then `radix` | `sort`, `radix` | **Bin:** frame-wide histogram + bucket record scatter into tile pages. **Radix:** per-tile depth sort in DRAM. | Should be **per-tile**: load bucket → sort **in L1** — not a global bin pass + separate radix pass over DRAM. |
| ~10–20 ms **gaps** after `radix` | (empty) | **Host:** `host_finish_sort` / return to `render_full_py` / setup cull programs / **still not one continuous CQ** on all paths. Smaller than the old 7 s JIT cliff but still real. | Must vanish when sort→cull→blend never returns to host. |
| Cyan `cull` ~60 ms, all cores | `cull` (TRISC) | **Global SFPU pass** over **all ~3.37 M (gaussian,tile) candidates** into `cull_masks` DRAM — **not** per-tile L1 cull after local sort. | This is the single biggest structural mistake vs your design. |
| Cyan `blend` + green `blend_rd` ~55 ms | `blend`, `blend_rd` | Per-tile bucket load + SFPU composite. **Correct shape**, but pays **after** global cull + still separate reader/compute programs. | Should be **one L1-resident tile kernel** with mask + blend, no global cull pass. |

**You are not crazy:** BRISC-heavy early timeline = **dataflow/mover work dominates** because we never fused **read → FPU project → L1 → scatter** into one chunk program. TRISC **does** run for cull/blend; it does **not** run for the long orange bars because projection math was never the bottleneck — **DRAM traffic and extra programs** are.

---

## 2. Roofline (honest targets)

From `plan-high-utilization-pipeline.md` §7, hero scene:

| Bound | Order-of-magnitude |
|---|---|
| SFPU blend+cull math @ 100% util | **~7 ms** (splat-by-splat, all pairs) |
| DRAM: one scene read + one scatter + one bucket read/tile | **bandwidth-limited**, not 150 ms if done as **bulk** |
| **1 ms / frame** | Requires **~10× fewer effective ops** (early-T, hierarchical reject, fewer pairs) — **not** achievable by scheduling alone |

| Milestone | ms/view (8v steady) | Requires |
|---|---|---|
| **Now (ideal path)** | **~159** | TILE_BUCKET, pipelined Finishes |
| **A — host out + no JIT cliff** | **~140–150** | JIT warmup; sort→cull→blend no pybind return; on-device sort LPT |
| **B — fused chunk project+TA scatter** | **~80–100** | One DRAM read/chunk; kill redundant passes |
| **C — per-tile L1 mega-kernel** | **~30–50** | sort+mb-mask+blend in L1; **delete global cull** |
| **D — pair-count reduction** | **→10, then →1** | early-T, tile/splat reject, lower-prec composite |

---

## 3. What we will fix (ordered — execute in this sequence)

### Phase A — Host/driver (days, in flight)

**Goal:** No 7 s frame-1 cliffs; no host between sort and blend.

| # | Change | Done when |
|---|---|---|
| A1 | **`GSPLAT_TT_JIT_WARMUP`**: compile all ideal programs at device open (proj/ta/sort/cull/blend/bucket). | Frame-1 Tracy gap ≈ frame-2 |
| A2 | **`sort_and_bin_tt` → `render_blend_tt` without returning to `render_full_py`** — enqueue publish→cull→blend on same CQ inside C++; pybind only `host_frame_begin` / `host_frame_end`. | No gap between `radix` and `cull` on host track |
| A3 | On-device **sort histogram + LPT + page layout** (kill `host_lpt_build` + hist D2H). | `host_finish_sort_bin_cnt` ≈ 0 |
| A4 | **Tracy zone renames** (see §4) — so the next capture matches your mental model. | Next `render-ideal-labeled.tracy` |

**Not “done” until:** `render_full_py` does **not** call separate `sort_and_bin_tt()` then `render_blend_tt()` for the hot path.

---

### Phase B — Project + TA: one chunk, one read (week)

**Goal:** Orange `proj` becomes: **BRISC bulk read → TRISC FPU project + frustum cull → BRISC bucket scatter** — **one program per splat chunk**, one `Finish` per chunk (or fused chain).

| # | Change |
|---|---|
| B1 | **Fuse `means_cam` + `pfwc` + frustum cull** on TRISC inside the chunk program (FPU tiles for quadratic form where possible; SFPU only for `exp`). |
| B2 | **Fuse TA AABB** into same chunk: read splat from L1, write **full record** directly into **per-core tile bucket DRAM** (your 430 MB scatter — accepted). |
| B3 | **Kill separate `ta` + `ta_scat` programs** for the hot path. |
| B4 | Device **exclusive scan** for bucket offsets (already have `gather_scan_bases`; extend to TA scatter bases). |

**Tracy should show:** one labeled region per chunk: `chunk_read` (short BRISC) → `chunk_project` (TRISC) → `chunk_scatter` (BRISC) — **not** two `proj` bars.

---

### Phase C — Per-tile L1: sort + mask + blend (1–2 weeks, hard)

**Goal:** Replace global `cull` + separate `blend` with **one tile program**:

```
bulk_dram_to_l1(tile_bucket)   // one contiguous read
sort_keys_in_l1()                // radix/insertion, already proven in bucket path
sfpu_mb_mask_in_l1()             // was global CULL_SPLIT — moves HERE
sfpu_blend_in_l1()               // read records + mask from L1 only
write_tile_to_framebuffer()
```

| # | Change |
|---|---|
| C1 | **Delete global `CULL_SPLIT` program** from hot path — masks computed **per tile** from L1-sorted records. |
| C2 | **Fix DEST hazard** (iter-26): isolate SFPU dst between mb-mask and blend phases **inside one program** (barrier + dst reset — root-cause the 30 dB, do not give up). |
| C3 | Merge `sort_radix_tile` into tile mega-kernel for bucket path (no separate `sort` + `radix` DRAM pass). |
| C4 | `blend_rd` becomes a **short** bulk-load slice at start of tile kernel, not 93 ms coextensive with all blend math. |

**Tracy should show:** per core, one long **TRISC** block `tile_l1` (or sub-zones `tile_load`, `tile_sort`, `tile_mask`, `tile_blend`) — **no** 60 ms global `cull` row spanning all tiles before blend.

**Expected win:** cull+blend **173 ms serial SFPU → ~30–50 ms** wall (parallel across tiles, one DRAM read/tile).

---

### Phase D — Toward 1 ms (after C is correct)

| # | Change |
|---|---|
| D1 | **Early-T termination** in SFPU blend (stop pixel when T≥threshold). |
| D2 | **Hierarchical / coarse-tile reject** before pair generation (cut 11M pairs). |
| D3 | **Tighter frustum + Mahalanobis** in chunk (fewer bucket entries). |
| D4 | Profile-guided **pair budget** per tile (cap candidates, LPT already balances cores). |

---

## 4. Tracy zone renames (next capture)

| Old | New (device) |
|---|---|
| `proj` | `proj_count`, `proj_scatter` **or** `chunk_dram_read`, `chunk_bucket_scatter` after Phase B |
| `mcam`, `pfwc` | keep; ensure visible (separate short TRISC bars) |
| `ta` | `ta_gauss_aabb` |
| `ta_scat` | `ta_bucket_scatter` |
| `sort` | `sort_bin_hist` + `sort_bucket_emit` (split in code) |
| `radix` | `sort_tile_depth` |
| `cull` | `tile_mb_mask` (per-tile only, after Phase C) |
| `blend` / `blend_rd` | `tile_blend_sfpu` / `tile_blend_load` |

Host zones (keep): `host_frame_begin`, `host_finish_sort_tail`, `host_finish_tile_pipeline`, `host_frame_end` — **not** `host_stage_sort` wrapping entire device sort + blend.

---

## 5. What we stop doing immediately

1. **Stop optimizing FUSED_TILE / gather** — dead path.
2. **Stop claiming “L1-resident” while running a global `cull` pass** — that is a lie on the timeline.
3. **Stop micro-optimizing `Finish` ms** without Phase A2/C1 — scheduling wins are **spent**; structure is next.
4. **Stop soft-float on movers** — already off on ideal path; never bring `MB_DEVCULL` back.

---

## 6. Success criteria (how you know we fixed it)

| Check | Pass condition |
|---|---|
| Tracy frame 2 | No gap > **2 ms** between `sort_tile_depth` end and `tile_blend` start on **host** track |
| Tracy per tile | **One** bulk BRISC read spike, then **TRISC** dominates sort/mask/blend |
| No global `cull` row | Zone `tile_mb_mask` only inside per-tile kernel |
| Verify | **63.85 dB** held |
| ms/view @8v | **< 100** after Phase C; **< 30** stretch goal before Phase D |
| vs CPU ~100 ms | **Beat CPU** after Phase B+C |

---

## 7. Immediate worker priority (loop)

1. **Phase A** (edd42e18): JIT warmup + sort→blend no pybind return + on-device sort LPT.
2. **Phase B** spec → implement fused chunk (single worker, device-only).
3. **Phase C** DEST root-cause session (Opus if credits; else Composer with bounded probes) — **non-optional** before more scheduling tweaks.

This document supersedes incremental “host-free bridge” work that does not move Phase B or C.

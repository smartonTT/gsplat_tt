# Phase C — per-tile L1 cull + blend without the iter-26 DEST hazard

**Written:** 2026-06-02  
**Branch context:** `smarton/tt-loop`, ideal `TILE_BUCKET` path (~159 ms/view @8v, 63.85 dB)  
**Parent:** `opt/endgame-execution-plan.md` §Phase C  
**Hard constraint:** Do **not** ship `GSPLAT_TT_FUSE_BLEND=1` (`microblock_cull_compute.cpp` `#ifdef FUSE_BLEND`) until the iter-26 cull-SFPU → blend-SFPU **in-program** DEST hazard is understood and fixed. That path is ~30 dB; two-program production is immune.

---

## 1. Problem statement (what Tracy shows today)

| Stage today | Zone | Cost (hero, 8v) | Structural issue |
|---|---|---:|---|
| Global SFPU cull | `cull` | ~60 ms wall, all cores | Iterates **~3.37M (gaussian,tile)** candidates; writes `cull_masks` DRAM |
| Bucket blend load | `blend_rd` | coextensive with blend | Bulk bucket + **bulk `cull_masks` read + ~512 spin**/tile |
| Bucket blend SFPU | `blend` | ~55 ms | Correct microblock kernel (`alpha_blend_compute_mb.cpp`) |
| DRAM radix | `radix` | separate pass | Depth sort over DRAM pages, not inside tile L1 program |

**Target (Phase C):** per core, one logical `tile_l1` pipeline:

```
tile_load   — one bulk DRAM→L1 read of dense bucket (exists in reader bucket branch)
tile_sort   — stable LSD radix in L1 (exists; must become only sort for bucket tiles)
tile_mask   — SFPU Mahalanobis mask for Lb locals only (move HERE from global CULL_SPLIT)
tile_blend  — alpha_blend_compute_mb on L1 coeffs + masks (no DRAM mask gather, no spin)
```

**Expected win (from endgame plan):** delete ~60 ms global `cull` + ~84 ms aggregate spin / mask DRAM traffic → cull+blend SFPU wall **~173 ms → ~30–50 ms** (still SFPU-bound, but per-tile and one bucket read).

---

## 2. iter-26 DEST hazard — what it forbids vs what it allows

### 2.1 Root cause (pinned by `CULL_LEVEL` sweep)

Source: `opt/blend-data-movement-plan.md` §9, `microblock_cull_compute.cpp` header + `FUSE_BLEND` block.

| `CULL_LEVEL` | Cull SFPU activity | Blend output |
|---:|---|---|
| **0** | `fill_tile` only — **no `copy_tile`, no cull SFPU** | **Deterministic** (44.13 dB blend-all reference) |
| **1** | `copy_tile(BOX)` + register-only `v_if` bbox | **Non-deterministic**, ~29 dB |
| **3** | full phased `face_x` / `face_y` / `combine` on DR_QV/QH | **Non-deterministic**, ~30 dB |

**Conclusion:** Any `copy_tile` (UNPACK→DEST) in the **cull phase** of the **same** `kernel_main` as blend SFPU corrupts blend — not mask handoff, not cross-gaussian DR_R/G/B/T, not missing `hw_configure` (all falsified). `init_sfpu()` mid-kernel **deadlocks**.

**Immune path:** standalone `CULL_SPLIT` program (fresh UNPACK/MATH/PACK + DEST at launch) → separate `alpha_blend_compute_mb` program. **This is the production shape at 63.85 dB.**

### 2.2 Phase C design rule (non-negotiable)

> **One tile “mega-kernel” = one Program and one reader loop, but cull SFPU and blend SFPU must remain in separate Compute kernels or separate Programs** until a proven LLK inter-phase reset exists.

**Explicitly banned on the hot path:** `#define FUSE_BLEND` on `microblock_cull_compute.cpp` (lines 69–87, 487–519, 552–699).

**Allowed:** Reuse the **same** cull math (`cull_dispatch`, phased `face_x`/`face_y`/`combine`, `CULL_LEVEL=3`) in a **dedicated** cull compute binary; hand masks via L1 CB / record word 10 / core-local DRAM — not via shared DEST with blend in one `kernel_main`.

### 2.3 Internal cull DEST hazard (already fixed in standalone cull)

`microblock_cull_compute.cpp` lines 140–150: phased dispatch + `DR_KEEP`/`DR_KEEP_B` parity — fixes **intra-cull** +32 batch skew. **Do not regress** when forking a tile-local cull compute.

`alpha_blend_compute_mb.cpp` lines 24–26: blend uses DEST 0–5 (R,G,B,T,XRAMP,YRAMP) — disjoint from cull’s DR_BOX_OX..DR_QH when in **separate** programs.

---

## 3. Concrete code changes (C1–C4)

### C1 — Delete global `CULL_SPLIT` from the TILE_BUCKET hot path

**Goal:** Masks computed **per tile** from L1-sorted bucket records; no frame-wide pass over 3.37M candidates.

| File | Change |
|---|---|
| `blend_device.cpp` | Gate `cull::execute()` / `CULL_SPLIT` enqueue: **skip when** `tile_bucket_enabled() && !bucket_mask_enabled()` (no duplicate ROUTE C). New env `GSPLAT_TT_TILE_LOCAL_CULL=1` (default OFF until verified). |
| New `kernels/dataflow/reader_tile_l1_cull.cpp` | Fork of `reader_bucket_cull.cpp`: per LPT tile, read `(rec_start, Lb)` from `sort_bucket_meta`, bulk-load bucket slice into L1 (reuse bucket branch from `reader_alpha_blend_mb_devcull.cpp` ~609–643), push `CB_CULL_COUNTS` + coeff rows from **L1 records** (post-sort order), not proj_m gather. |
| `microblock_cull_compute.cpp` | **No FUSE_BLEND.** Optional `TILE_L1_CULL` define only for Tracy zone rename `tile_mb_mask` (`DeviceZoneScopedN`). |
| `writer_microblock_cull.cpp` or new `writer_tile_l1_mask.cpp` | Write 32-bit masks to **per-tile L1 ring** (`CB_TILE_MASKS` / `CB_BMASK`) or **record word 10** in the L1 bucket copy — **not** global `cull_masks` DRAM for the hot path. |
| `reader_alpha_blend_mb_devcull.cpp` | Bucket emit loop (~740+): read mask from L1 (`recp[10]` or `CB_BMASK[k]`), drop `cull_masks_acc` bulk read + `MB_CULL_SPIN` block (~716–737). |
| `env_config.h` | Document mutual exclusion: `TILE_LOCAL_CULL` ⇒ no `CULL_SPLIT`; `BUCKET_MASK` (sort-stage cull) must stay OFF to avoid duplicate cull (§13 blend-data-movement-plan). |

**Correctness gate:** 63.85 dB hero, `min_vs_ref` ≥ production bucket path (60.82 @8v baseline).

**Perf gate:** Tracy — no full-frame `cull` row; `tile_mb_mask` appears inside per-tile `blend` window.

---

### C2 — DEST-safe cull → blend handoff (not FUSE_BLEND)

**Goal:** Capture ~116 ms fused-program blend *timing* without iter-26 corruption.

**Preferred implementation (lowest risk):**

```
Program "tile_mask" (per frame, same LPT cores as blend):
  reader_tile_l1_cull  →  microblock_cull_compute (NO FUSE_BLEND)  →  writer_tile_l1_mask
       ↓ masks in L1 CB or rec[10]
Program "tile_blend" (existing resident blend):
  reader_alpha_blend_mb (bucket)  →  alpha_blend_compute_mb  →  writer
```

- Enqueue **mask program → blend program** on one in-order CQ (`GSPLAT_TT_SORT_BLEND_PIPE` pattern); **no** `Finish` between if host already eliminated (Phase A2).
- Masks in **L1** (iter-15 proved bit-identical transport); spin removed because consumer reads producer’s L1/CB, not post-`noc_async_read` DRAM (`lessons.md` MB_CULL_SPIN).

**Bounded C2 experiments** (device, one flag each, `devrun.sh`):

| ID | Define / env | What it tests | Pass | Fail |
|---|---|---|---|---|
| C2-A | `CULL_LEVEL=0` in **tile-local cull program only** | Pipeline plumbing, no cull `copy_tile` | Deterministic output; 44 dB blend-all | N/A for ship |
| C2-B | `CULL_LEVEL=1` tile-local cull, separate program | Bbox cull without DR_QV/QH | 63.85 dB if math sufficient | <63.6 dB → need L3 |
| C2-C | `CULL_LEVEL=3` (production math), separate programs | Full quality path | **63.85 dB**, deterministic | Any 30 dB → stop, not FUSE |
| C2-D | `FUSE_BLEND=1` negative control | Repro iter-26 | — | ~30 dB confirms hazard |
| C2-E | Two Compute kernels, one Program, semaphore between cull and blend compute | Same-process without shared `kernel_main` | 63.85 dB | 30 dB or deadlock → abandon |

**Do not retry without new evidence:** `init_sfpu()` between phases; per-gaussian `_llk_math_eltwise_unary_sfpu_start_/done_` only; MATH-only `hw_configure` (all disproven iter-26).

---

### C3 — Merge `sort_radix_tile` into the tile L1 program (bucket path)

**Goal:** For `Lb <= MB_BUCKET_FIT`, **no** separate DRAM radix pass; depth order = L1 stable LSD radix already in `reader_alpha_blend_mb_devcull.cpp` (~659–698).

| File | Change |
|---|---|
| `sort_device.cpp` | When `TILE_BUCKET` + `TILE_LOCAL_CULL` + `BIN_NODEPTH` or new `GSPLAT_TT_TILE_L1_SORT_ONLY=1`: **skip** `sort_radix_tile` enqueue for tiles that fit bucket (host/device metadata flag per tile or global “all buckets fit” fast path). |
| `sort_bin.cpp` | Scatter-only mode: records land in bucket unsorted; sort happens only in blend reader (already true for T2). |
| `reader_alpha_blend_mb_devcull.cpp` | Extract `bucket_l1_radix_sort()` to shared header; call from both reader and future `reader_tile_l1_cull` so cull sees **same order** as blend. |
| Tracy | Rename: `sort_tile_depth` only for overflow/fallback tiles. |

**Risk:** `cull_masks` post-sort indexing — **avoid entirely** on hot path by masks in record order (word 10) or `mask[k]` in L1 after sort.

**Gate:** Bit-identical vs baseline bucket+global cull for FIT=8192 hero; then FIT sweep 64/256/8192.

---

### C4 — `blend_rd` = short bulk load only

**Goal:** Reader time dominated by **one** `tile_recs[rec_start..Lb)` burst, not mask pages + spin.

| File | Change |
|---|---|
| `reader_alpha_blend_mb_devcull.cpp` | After C1+C2: remove `#elif` DRAM `cull_masks` bulk (~716–737) and `MB_CULL_SPIN`; keep `noc_async_read` bucket loop only (~634–642). |
| `blend_device.cpp` | Tracy: `DeviceZoneScopedN("tile_blend_load")` around bucket bulk; `tile_blend_sfpu` on compute. |
| CB sizing | `CB_BMASK` optional removal if masks only in `recp[10]`. |

**Gate:** `blend_rd` zone &lt; ~5 ms/tile amortized @8v (measure in labeled Tracy); total blend stage ↓ ~80+ ms vs today’s spin path.

---

## 4. Bounded test matrix (device)

**Config base (ideal path):**  
`GSPLAT_TT_FUSED_TILE=0 GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_SFPU_CULL=1 GSPLAT_TT_RESIDENT_BLEND=1 GSPLAT_TT_BUCKET_FIT=8192 GSPLAT_TT_BUCKET_CB_FENCE=1`  
**Views:** 1 for debug, 8 for ms/view. **Metric:** `hero_vs_ref` dB, stage ms from log, Tracy zone checklist §endgame §6.

| Run | Flags / knobs | Primary signal | Stop if |
|---|---|---|---|
| **P0** | Baseline (above), global `CULL_SPLIT` ON | Reference 63.85 / ~159 ms/view | — |
| **P1** | `GSPLAT_TT_TILE_LOCAL_CULL=1`, masks → `rec[10]`, global cull OFF | 63.85 dB; global `cull` gone | &lt;63.6 dB or new `cull` row |
| **P2** | P1 + `CULL_LEVEL=1` compile (tile cull program only) | Cheaper SFPU mask cost | &lt;63.6 dB |
| **P3** | P1 + `CULL_LEVEL=3` | Production-quality local cull | &lt;63.6 dB or nondeterministic (2 values/tile DPRINT) |
| **P4** | P1 + `GSPLAT_TT_FUSE_BLEND=1` | Negative control | Must **fail** ~30 dB (confirms hazard understood) |
| **P5** | P3 + `GSPLAT_TT_TILE_L1_SORT_ONLY=1` (C3) | `radix` ↓, sort ms ↓ | PSNR drift |
| **P6** | P3, `BUCKET_FIT=64` then `256` | Lb boundary (iter-22/27 lesson) | 42 dB → stop, fix assembly not algorithm |
| **P7** | P3 @8v, 30 views | `min_vs_ref` ≥ 60.82 | Per-view regression |
| **P8** | P3 + `CULL_PIPELINE=1` | Host overlap only (~1 ms) | Claiming &gt;5 ms win |

**CULL_LEVEL sweep (compile-time, tile-local cull program only):**

| Level | Code path (`microblock_cull_compute.cpp`) | Expect |
|---:|---|---|
| 0 | `fill_tile` only (~614) | Deterministic; proves handoff |
| 1 | `cull_phase_bbox` (~628–629) | Faster; may be quality-limited |
| 2 | `cull_phase_vary` (~615–620) | Pack pipeline sanity |
| 3 | `cull_dispatch` (~631) | **Ship target** |

**ROUTE C control (do not combine with P1):** `GSPLAT_TT_BUCKET_MASK=1` + global `CULL_SPLIT` → duplicate cull (~154 ms sort, §13). Use only to validate sort-stage writer (`writer_bucket_cull.cpp`), not production hot path.

---

## 5. Implementation sequence (worker order)

1. **C2-A/C2-C** — Tile-local cull as **separate program** (reader fork + writer to `rec[10]`), global cull off; prove 63.85 dB (dest hazard avoided by construction).
2. **C1** — Wire env gate, Tracy renames, delete global enqueue on hot path.
3. **C4** — Strip DRAM mask read + spin from bucket reader.
4. **C3** — Skip DRAM radix for in-FIT tiles; extract shared L1 sort helper.
5. **C2-E** (optional) — Only if separate programs leave a host `Finish` bubble &gt;2 ms; attempt two compute kernels + semaphore.

**Do not start with:** `FUSE_BLEND`, inline soft-float cull in reader (§13 trap, 416 ms blend), or `BUCKET_MASK` + `CULL_SPLIT` together.

---

## 6. Top 3 experiments (ROI order)

1. **P1 / C2-C — Per-tile bucket cull in a separate program, masks in `rec[10]`, global `CULL_SPLIT` off**  
   - **Why #1:** Removes ~60 ms global SFPU + ~3.37M gather-driven cull work; reuses proven `microblock_cull_compute.cpp` (L3, phased dispatch) without DEST hazard; eliminates spin/DRAM mask read in one step.  
   - **Files:** `reader_tile_l1_cull.cpp` (from `reader_bucket_cull.cpp`), `writer_tile_l1_mask.cpp`, `blend_device.cpp` enqueue guard, `reader_alpha_blend_mb_devcull.cpp` emit.  
   - **Success:** 63.85 dB, ms/view −50–80 ms, no global `cull` Tracy row.

2. **P3 + P5 — Full `CULL_LEVEL=3` local cull + skip DRAM `sort_radix_tile` for in-FIT bucket tiles**  
   - **Why #2:** Cuts redundant DRAM sort pass (~26 ms + host gap) once masks are in bucket order; L1 radix already proven (iter-22/27).  
   - **Files:** `sort_device.cpp`, `reader_alpha_blend_mb_devcull.cpp` (shared sort), `env_config.h`.  
   - **Success:** `radix` zone gone or minimal; sort ms ↓ without PSNR change.

3. **P2 — `CULL_LEVEL=1` bbox cull per tile (quality vs cost probe)**  
   - **Why #3:** If bbox passes 63.6 dB gate, reduces SFPU work per candidate (iter-26: level≥1 `copy_tile` is safe **across programs**); lowers tile_mask ms toward roofline ~7 ms SFPU math bound.  
   - **Files:** `microblock_cull_compute.cpp` compile flag per program binary (separate JIT spec from L3).  
   - **Success:** Same PSNR band, measurably lower `tile_mb_mask` ms; else stay on L3.

---

## 7. Success criteria (copy from endgame §6)

| Check | Pass |
|---|---|
| No global `cull` | Full-frame `cull` row absent; `tile_mb_mask` inside per-tile work |
| PSNR | 63.85 dB hero; 8v `min_vs_ref` ≥ 60.82 |
| ms/view @8v | &lt; 100 after Phase C (stretch &lt; 30 needs Phase D) |
| Host gap | &lt; 2 ms between `sort_tile_depth` end and blend start |
| DEST guard | `FUSE_BLEND=1` remains **off** default; P4 negative stays ~30 dB |

---

## 8. Key file reference map

| Role | Path |
|---|---|
| SFPU cull math + `CULL_LEVEL` + **FUSE_BLEND hazard** | `src/gsplat_tt/kernels/compute/microblock_cull_compute.cpp` |
| SFPU blend + DEST 0–5 | `src/gsplat_tt/kernels/compute/alpha_blend_compute_mb.cpp` |
| Bucket load + L1 radix + mask emit | `src/gsplat_tt/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp` |
| Global cull enqueue | `src/gsplat_tt/blend_device.cpp` (`cull::`, `CULL_SPLIT`) |
| ROUTE C sort-stage cull (reference, not hot path dup) | `reader_bucket_cull.cpp`, `writer_bucket_cull.cpp` |
| DRAM radix | `kernels/dataflow/sort_radix_tile.cpp`, `sort_device.cpp` |
| iter-26 write-up | `opt/blend-data-movement-plan.md` §9 |
| DEST lesson | `tt-workflows/knowledge/lessons.md` (iter-26 / FUSE_BLEND) |

---

## 9. Out of scope (Phase D / banked)

- `CULL_LEVEL` math reduction without quality proof (Phase D pair-count).  
- Sort/cull overlap via mask-through-radix (≤26 ms ceiling, §14).  
- LLK full inter-phase `init_sfpu` without deadlock (iter-26 banked).  
- Default-flip `TILE_BUCKET` to ON (thin margin; separate milestone).

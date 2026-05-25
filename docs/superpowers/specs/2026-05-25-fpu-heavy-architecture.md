# gsplat_tt — FPU-heavy end-state architecture

**Authored:** 2026-05-25 (post-iter-2 redirect)
**Supersedes (in part):** `2026-05-25-gsplat-tt-1ms-design.md` §2 PSNR floor + greedy-iter rejection rules.

## Why this doc exists

The user (2026-05-25) redirected the optimization loop:

> "Let's approach the optimization plan this way: come up with the ideal holistic setup at the outset, from cpu gs algo to LLK. Many optimizations probably require each other and compound, even if each step will not necessarily be faster by itself."

The previous greedy "is this single iter faster than baseline?" rejected iter-2 (basis-form, 47 dB / 106 ms) even though basis-form is **structurally required** for the FPU-heavy + 16×16-face end state.

The reference for the architectural direction is `/Users/smarton/Downloads/gs-fpu.md` ("convert your 4-face example to leverage as much fpu as possible").

## End-state architecture (target ≤1 ms @ 1024×1024)

### Algorithmic rewrite — quadratic basis form

Per-Gaussian Q is currently:
```
Q(x,y) = qxx·(x-mx)² + 2·qxy·(x-mx)·(y-my) + qyy·(y-my)²
```
which requires per-pixel `dx = x - mx`, `dy = y - my` recompute followed by 3 multiplies and 2 adds, all touching pixel-coord tiles for every Gaussian.

**Rewrite as basis form** with per-Gaussian coefficients (computed host-side):
```
Q(x,y) = A·x² + B·xy + C·y² + D·x + E·y + F
```
where
```
A = qxx
B = 2·qxy
C = qyy
D = -2·qxx·mx - 2·qxy·my
E = -2·qyy·my - 2·qxy·mx
F = qxx·mx² + 2·qxy·mx·my + qyy·my²
```

Now Q assembles entirely from **6 FPU `mul_tiles(basis_cb, gauss_cb, …)` with `acc_to_dest=true`** — each multiplies a precomputed basis tile (x², xy, y², x, y, 1) against the per-Gaussian coefficient tile and accumulates into Dst.

### Precomputed basis tiles

Six 32×32 tiles, computed ONCE per frame and held in DRAM, identical for every owner tile (tile-local coords; basis values are relative to each tile's top-left):
- `x²_tile`, `xy_tile`, `y²_tile`, `x_tile`, `y_tile`, `one_tile`

For each output tile, its per-pixel basis values can be computed from local coords in [0, 31] × [0, 31] — small magnitudes, fp32-clean.

### 16×16 face decomposition

Decompose the 32×32 owner tile into 4 × 16×16 **faces**. Each face has its own independent Gaussian stream and length:
```
count0 = #gaussians intersecting face 0
count1 = ...
count2 = ...
count3 = ...
```

Per Gaussian (within a face), the reader streams 10 tiles:
```
A, B, C, D, E, F, opacity, r, g, b
```

Where each tile is 32×32, but only the **target 16×16 face** contains non-zero values; the other 3 faces are zero. The 7th tile is a `log_mask_face` (added before exp): 0 inside the active face, large negative outside — `exp_tile` zeroes contributions from the other 3 faces.

This means a single 32×32 owner tile can hold 4 different Gaussian loops (one per face), all sharing the same DST-resident R/G/B/T accumulators, because face-mask `exp(power + mask)` ≈ 0 outside the active face.

The win: **per-face AABB cull → ~3-4× fewer Gaussians visited per face** vs. the unioned cull of the full 32×32 owner.

### DST-resident state, single acquire per owner

One `tile_regs_acquire()` per owner tile.

DST slot map (10 fp32 slots — well under the Blackhole limit):
```
dst[0] = R accumulator (R = R + contrib·color_r)
dst[1] = G accumulator
dst[2] = B accumulator
dst[3] = T accumulator (T = T·(1-alpha))
dst[4] = Q (rebuilt per Gaussian, then -0.5·Q → exp → α)
dst[5] = tmp (channel temp / alpha / 1-alpha)
dst[6] = contrib = T·α  (DST×DST → SFPU mul_binary_tile)
dst[7] = scratch
```

Init at the top of the acquire:
```cpp
copy_tile(cb_zero, 0, 0);  // R = 0
copy_tile(cb_zero, 0, 1);  // G = 0
copy_tile(cb_zero, 0, 2);  // B = 0
copy_tile(cb_one,  0, 3);  // T = 1
```

Pack at the bottom:
```cpp
pack_tile(0, cb_out_r);
pack_tile(1, cb_out_g);
pack_tile(2, cb_out_b);
```

### Operation mix (the win)

**FPU (CB×CB → DST) — the workhorse:**
- `mul_tiles(basis_cb, gauss_cb, ofs_b, ofs_g, dst[4])` with default `acc_to_dest` → builds Q in 6 FMAs.
- `add_tiles(...)` → log_mask add.

**SFPU (unavoidable):**
- `exp_tile<approx>(dst[4])` → in-place exp.
- `mul_binary_tile(dst[3], dst[4], dst[6])` → `contrib = T·α` (DST×DST).
- `add_binary_tile(dst[0], …, dst[0])` → R += contrib·color_r.
- `sub_binary_tile(dst_one, dst[4], dst[5])` → `1-α`.

**Dest-reuse (FPU one operand from DST, other from CB):**
- `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_face, IDX_OPACITY, dst[4])` → `α = exp(Q) · opacity` in-place.

## Perceptual PSNR floor (overrides spec §2)

The previous "kernel-algebra" class required ≥100 dB PSNR — an unrealistic numerical-equivalence bar that forced rejecting structurally-correct architectural changes for sub-LSB rounding noise.

**New floor: 40 dB**, plus a visual guard (no NaN holes, no color-clipping bands, no 32×32 tile-grid seams, no gross geometry shift). Validators report PSNR and visual checks; the loop accepts iters with PSNR ≥ 40 dB and clean visuals.

Rationale (user, 2026-05-25): "PSNR 40 should still be OK from a perceptual standpoint. Just keep an eye on the screenshots and diffs to make sure you don't introduce anything egregious."

## Implementation milestones

Each milestone is a coherent step that keeps the rendering correct. The loop runs them in order without stopping. If rendering breaks, **debug the code like a human** — don't reject and walk away.

### M1 — Basis-form host coefficients + FPU q-accumulation (tile-global basis)

Host (`gsplat/rasterization.py::prepare_kernel_inputs`):
- Compute (A, B, C, D, E, F) per Gaussian from (mean_x, mean_y, cov_inv_a, cov_inv_b, cov_inv_c).
- Replace the current `[mean_x, mean_y, a, 2b, c, r, g, b, opacity]` 9-scalar pack with `[A, B, C, D, E, F, r, g, b, opacity]` 10-scalar pack.

Host (basis-tile generator, in C++ host-side):
- Generate 6 basis tiles (x², xy, y², x, y, 1) once. The x, y here are **tile-local** [0,31]×[0,31] (or owner-tile global if needed).
- Push to a single DRAM buffer accessible by all cores via `TensorAccessor`.

Kernel (`alpha_blend_compute.cpp`):
- Replace stages B1-B3 (dx, dy, dx², dy², dx·dy, a·dx², c·dy², 2b·dx·dy, sum) with:
  ```cpp
  // Build Q in dst[4] via FPU FMA
  copy_tile(cb_zero, 0, 4);   // Q = 0
  mul_tiles(cb_x2,  CB_SCALARS_TILE, 0, IDX_A, 4);  // Q += A·x²
  mul_tiles(cb_xy,  CB_SCALARS_TILE, 0, IDX_B, 4);  // Q += B·xy
  mul_tiles(cb_y2,  CB_SCALARS_TILE, 0, IDX_C, 4);  // Q += C·y²
  mul_tiles(cb_x,   CB_SCALARS_TILE, 0, IDX_D, 4);  // Q += D·x
  mul_tiles(cb_y,   CB_SCALARS_TILE, 0, IDX_E, 4);  // Q += E·y
  mul_tiles(cb_one, CB_SCALARS_TILE, 0, IDX_F, 4);  // Q += F
  ```

For M1 we still spill Q to CB_POWER and the rest of the chain stays the same (still SFPU-heavy). The point of M1 is to flip Q assembly to FPU; M3 turns the whole loop DST-resident.

**Validation:** PSNR ≥ 40 dB on hero/side/top. Screenshots must look correct (no tile-grid seams, no geometry shift). Timing may regress slightly (6 mul_tiles vs. 3 + SFPU chain) — that's OK; M3 reclaims it.

### M2 — Precomputed basis tiles in DRAM

The 6 basis tiles are owner-tile-local and identical across owner tiles. Compute them in a tiny one-shot compute kernel that runs at the start of each frame and writes them to a fixed DRAM region. Reader streams them to all cores.

**Validation:** Identical pixels to M1; small timing win from removing tile-by-tile host-side regen.

### M3 — DST-resident R/G/B/T, single acquire per owner tile

Restructure the per-output-tile loop:
- Eliminate CB_COLOR_R/G/B_STATE, CB_T_STATE, CB_SAT_MASK as L1 spill targets.
- One `tile_regs_acquire()` at the start of each output tile.
- Init dst[0..3] = (0,0,0,1).
- Per Gaussian: build Q (M1), exp, alpha-clamp, contrib, then SFPU DST×DST color/T updates.
- One `tile_regs_commit/wait/release` + 3 pack_tiles at the end.

Use `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` for `α *= opacity` (one DST operand, one CB operand).

**Validation:** PSNR ≥ 40 dB; timing should drop noticeably (eliminates ~23 acquire/release barriers per Gaussian).

### M4 — 16×16 face decomposition

Host (`gsplat/rasterization.py`):
- Bin Gaussians into 4 per-face streams per owner tile based on 16×16 face AABB intersection.
- Per face: pack 10 tiles per Gaussian, plus a 7th basis tile `log_mask_face[0..3]` (computed once, 4 versions).

Kernel:
- 4 CBs for face streams: cb_f0, cb_f1, cb_f2, cb_f3.
- `process_face<cb_f0>(count0); process_face<cb_f1>(count1); …` inside the single acquire.
- `mul_tiles(cb_log_mask_f, cb_one, …)` adds the face mask before exp.

**Validation:** PSNR ≥ 40 dB; timing should drop ~2-4× from per-face culling (each Gaussian visits ~1 face on average, not all 4 quadrants of the 32×32 owner).

### M5+ — Beyond M4

Open optimization budget for: bf16 storage of (A,B,C,D,E,F) (smaller transfers, slight PSNR cost); pre-multiplied opacity·color (saves 3 muls per Gaussian); fp32 accumulator → bf16 final pack inline; tighter per-core work distribution; etc.

## Loop discipline

- **Never stop.** Broken renders are code bugs. Read the code, find the mistake, fix it, rerun.
- **40 dB floor + visual check** = KEEP. Below 40 dB or visible artifact = debug.
- **Don't reject building-block iters.** A step that's slower in isolation can still be the foundation of compound speedup — assess against the milestone, not the previous best.
- iter-2 (basis-form-tile-local) is a building block, not a REJECT. Reclassified.
- See memory: `[[feedback-never-stop-loop]]`, `[[project-psnr-floor-perceptual]]`, `[[project-fpu-heavy-architecture]]`.

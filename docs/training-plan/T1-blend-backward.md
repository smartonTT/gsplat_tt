# T1 blend backward: decomposition, oracle gate, device notes (task #677)

2026-10-13 PDT. Branch ttp/t677-training-t1-docs-training-plan-plan-md-i (merges #671 and
the T0 oracle #674). Plan: docs/training-plan/PLAN.md T1 and §3 steps 1-4.

Status: the device decomposition is written as a numpy model
(`gsplat/blend_backward_ref.py`) and passes the T0 oracle gate. The device kernel is not
written yet, so nothing is timed: no device time was used. The model is the lane-for-lane
reference the kernel will be tested against.

## 1. What the model computes

Per 32x32 tile, pairs front to back in the tile's sorted order, per pixel carrying T and
the colour accumulated so far (`C_acc`). Per live pixel:

    dalpha = T (c . g) - ((image - C_acc) . g) / (1 - alpha)     one reciprocal per live px
    G      = dalpha * alpha   (dL/dpower; 0 where alpha is clamped at 0.99)
    terms  = dalpha*w, alpha*T*g (3), G u^2, G u v, G v^2, G u, G v     (9 floats)

u, v = pixel centre minus the Gaussian's tile-local mean. The 9 terms are summed over the
32 lanes of each 4x8 microblock, then over microblocks in order. The pair epilogue turns
them into the plan's 9 grads (dmx, dmy, dA, dB, dC, dopacity, dr, dg, db):
dA = -0.5 S_uu, dB = -S_uv, dC = -0.5 S_vv, dmx = A S_u + B S_v, dmy = B S_u + C S_v.
Each pair writes them to slot `P[g] + j` (P = exclusive prefix of tile counts, j = row-major
tile index in g's rectangle); the per-Gaussian sum reads j = 0..count-1 in order. Bitwise
deterministic (test).

## 2. Oracle gate (PLAN.md T1: rel err <= 1e-2)

Metric: train_oracle.rel_error (max |d| / max(|ref|, 1e-4 max|ref|)) over visible
Gaussians; "max/max" = max abs error / group max. fp32 model vs fp64 oracle.

| case | mode | dmeans2d | dconic | dopacity | dcolors | worst | gate |
|---|---|---|---|---|---|---|---|
| t0_sh0_64x64_n50.npz | recompute | 5.2e-5 | 4.0e-4 | 2.1e-5 | 4.1e-4 | **4.1e-4** | pass |
| t0_sh0_64x64_n50.npz | frozen | 5.2e-5 | 4.0e-4 | 2.1e-5 | 4.1e-4 | 4.1e-4 | pass |
| t0_sh3_64x64_n50.npz | recompute | 2.3e-5 | 3.4e-4 | 1.4e-5 | 1.2e-5 | **3.4e-4** | pass |
| dense 64x64 n50 (7 clamped, 49 stopped px) | frozen | 2.7e-4 | 7.5e-5 | 7.7e-4 | 1.0e-4 | 7.7e-4 | pass |
| multi-tile 104x80 n120 (3x4 tiles, partial) | recompute | 7.2e-4 | 4.0e-5 | 4.5e-5 | 9.2e-5 | 7.2e-4 | pass |
| multi-tile 104x80 n120 | frozen | same | same | same | same | 7.2e-4 | pass |
| dense 64x64 n50 | recompute | 2.5e-1 | 6.0e-3 | 1.8e-3 | 1.9e-2 | 2.5e-1 | n/a (forward differs) |

Mean abs error / group max is 4e-8 to 2e-7; max abs / group max <= 1.0e-6 in every
passing row. Live pixels match the oracle exactly on the dumps (8616 = 8616) and the multi-tile case
(52806).

- *recompute* is device-faithful: alpha, the 1/255 skip and the early stop are recomputed
  in fp32 and Gaussians cover only their 3-sigma tile rectangle (gsplat_cpu::tile_assign).
- *frozen* uses the oracle's live masks, which isolates the backward. In the dense case
  the fp32 forward blends 38 fewer pixels than the fp64 oracle (47805 vs 47843: early-stop
  flips and pixels outside the 3-sigma rectangle with alpha >= 1/255). That is a forward
  difference the device forward has too, not a backward error.

Run: `python3 -m gsplat.blend_backward_ref benchmarks/train_oracle/t0_sh*_64x64_n50.npz`;
tests: `tests/unit/test_blend_backward_ref.py` (8 tests, ~70 s on the Mac).

## 3. Findings that change the plan

1. **The forward must keep the final colour in fp32.** The forward packs bf16 tiles to u8
   (img_pack_u8.h). Feeding the backward a bf16 final image gives rel err 0.20 (SH0 dump),
   2.0 (dense), 1.3 (multi-tile); u8 gives 0.36 / 3.1 / 1.5 (all fail the 1e-2 gate;
   max/max up to 4.6e-3 / 7.8e-3). `image - C_acc` cancels, so the error of the stored image
   lands directly in dalpha. Training mode needs an fp32 RGB output per pixel: 12 B/px,
   12 MiB at 1024x1024 (plan §3 step 1 assumed 8 MB for T + colour). Final T is not needed
   when the background is 0 (image already holds C + T_final * bg).
2. **Pair records stay fp32.** Rounding means2d, conic, opacity or colour to bf16 fails the
   gate by orders of magnitude (means2d in bf16 has 1/8 px steps at x = 32: rel 17.6). The
   forward already uses fp32 records (dm_fp32), so this costs nothing; the plan's "(bf16
   inputs)" note does not apply.
3. **Moments in u, v, not in tile coordinates.** Accumulating G u^2 ... G v (pixel minus
   mean) instead of G x^2 ... G (tile coordinates) avoids the cancellation in
   dmx = A (S_x - mx S) + ... for Gaussians centred far outside the tile, and needs 5 sums
   instead of 6, so the per-pair payload is the plan's 9 floats.
4. **No live mask is stored.** The backward recomputes the forward's live decision with the
   same fp32 math in the same order, so it matches the device forward exactly; only the
   live-microblock masks (slab word3) are reused for skipping.

## 4. Device kernel (next step)

Per tile, on the core and in the claim order the forward used, reusing the forward's slab
(sort_subchunk_payload, microblock mask in word3) and pfwc records:

- Dst state per pixel (fp32): T, s = C_acc . g, Ig = image . g, g (3, dL/dimage), x/y
  ramps = 8 fp32 tiles, exactly the Dst capacity in the forward's full-sync mode (the model
  carries this scalar form). The loss-gradient kernel (T7) has image and g, so it can write
  g and Ig (16 B/px); the image it dots must be the forward's fp32 colour (finding 1).
- Per live (pair, microblock): forward alpha (same ops as alpha_blend_compute_mb.cpp) plus
  roughly 25 SFPU ops (E, not counted) for dalpha (one reciprocal of 1 - alpha, as pfwc_recip_nr.h), the 9 terms
  and the 32-lane reduce (SFPU shuffle, or FPU row-reduce: T6).
- Per pair: 9-float epilogue, 36 B write to slot P[g] + j (NoC write, no atomics).
- Host: a training flag that makes the forward write the fp32 image and keeps the slab and
  per-Gaussian tile counts alive until the backward has run.

Device check before timing: the kernel's slots vs `blend_backward(...)["slots"]` on the
64x64 dumps (expect <= 1e-5 rel; the SFPU exp is the 21-bit one), then hero-view timing
alone with the traced split (PLAN.md T1).

## 5. What T2 needs from T1

- Slot array: float32 [sum(counts), 9], rows Gaussian-major at `P[g] + j`, columns
  (dmx, dmy, dA, dB, dC, dopacity, dr, dg, db), mean in pixels, conic (A, B, C) with
  power = -0.5 (A dx^2 + 2 B dx dy + C dy^2), opacity activated (after sigmoid), colour after
  the clamp (T2 applies the clamp mask and the SH backward).
- P and counts from the forward's per-Gaussian tile counts (gather_scan_bases), in the same
  Gaussian order T2 walks; T2 sums j = 0..count-1 in order (deterministic), then runs the
  projection backward and sparse Adam (oracle: gsplat/train_oracle.py, sparse_adam_step).

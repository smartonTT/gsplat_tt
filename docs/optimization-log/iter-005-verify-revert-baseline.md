# iter-005 — Verify revert of iter-003 M1 basis-form

- Class: kernel-algebra
- Track: revert-validation
- Date: 2026-05-26
- Status: KEEP / no_commit_baseline_reproduced

## Goal

Confirm that reverting the M1 basis-form change (gsplat/rasterization.py + alpha_blend.cpp + alpha_blend_host.h + kernels/compute/alpha_blend_compute.cpp at SHA 33d5e3f) restores the iter-000 baseline byte-exactly, including elimination of the user-flagged tile-grid seam artifact at Stitch's ear bottoms.

## Result

- kernel ms median: 100.34 (within noise of iter-000 baseline 99.95)
- PSNR per view: hero ∞ / side ∞ / top ∞ (bit-identical to baseline reference)
- diff10 hero/side/top: pure black (no residual)

## What was reverted

Commit 044f398:
- `gsplat/rasterization.py` — back to 9-fp32 per-Gaussian packs (qxx, qxy, qyy, mean_x, mean_y, R, G, B, opacity); no A..F coefficients.
- `backends/tt/.../alpha_blend.cpp` — back to direct-Q allocation; no CB_BASIS_X2/XY/Y2/X/Y/ONE setup.
- `backends/tt/.../alpha_blend_host.h` — back to SCALAR_PACK_BYTES = 36.
- `backends/tt/.../kernels/compute/alpha_blend_compute.cpp` — back to baseline direct-Q evaluation: dx = px - mean_x; dy = py - mean_y; Q = dx²·qxx + 2·dx·dy·qxy + dy²·qyy, then exp + alpha + blend chain.

Commit f50be70 logged the iter-004 screenshots as record of the rolled-back experiment.

## Confirms

The revert restores the iter-000 baseline byte-exactly. Loop can resume with iter-000 as `current_best` (99.95 ms, PSNR ∞). M1/M2/M3/M4 in QUEUE.json all depend on basis-form and need a new hypothesis before the next dispatch.

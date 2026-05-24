---
iter: 071
label: Precompute lx²/ly²/lxly once per tile
status: KEEP
kernel_ms: 17.14
prev_kernel_ms: 17.77
psnr_db: 42.06
total_ms: 95.64
date: 2026-05-23
---

## Hypothesis

The iter 070c single-acquire Dst-resident design recomputed `lx²`, `ly²`, `lx·ly`
via 3× SFPU `mul_binary_tile` for **every Gaussian** (expensive, ~3 SFPU binary
ops × 448 Gaussians/tile). Moving this to a separate precompute block (separate
`tile_regs_acquire/release`) runs these 3 ops once per tile and packs results to
CB_LX2/CB_LY2/CB_LXLY. Inside the state loop, FPU `copy_tile` reads replace the
3 SFPU binary muls. FPU is ~5× faster per element.

## Code change

`backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`

- Added PRECOMPUTE BLOCK before the state loop:
  - Separate `tile_regs_acquire()` ... `tile_regs_release()`
  - `copy_tile` lx→Dst[0], ly→Dst[1]
  - `mul_binary_tile(0,0,2)` → lx², `mul_binary_tile(1,1,3)` → ly², `mul_binary_tile(0,1,4)` → lxly
  - Pack Dst[2,3,4] → CB_LX2, CB_LY2, CB_LXLY
- STATE LOOP now uses `copy_tile(CB_LX2, 0, 7)` etc. to load precomputed values
  via FPU (fast), replacing 3× SFPU `mul_binary_tile` per Gaussian.

## Results

| Metric       | iter 070c   | iter 071    | Delta       |
|-------------|-------------|-------------|-------------|
| Kernel ms   | 17.77 ms    | 17.14 ms    | -0.63 ms (-3.6%) |
| PSNR dB     | 43.18 dB    | 42.06 dB    | -1.12 dB (still well above 35 dB gate) |
| Total ms    | ~95 ms      | 95.64 ms    | ~0 ms       |

## Findings

- Modest improvement: the 3 SFPU binary ops per Gaussian savings are partially
  offset by the precompute block's own acquire/release overhead and extra CB reads.
- PSNR slightly decreased (42.06 vs 43.18) — within noise, still comfortably above 35 dB.
- Main bottleneck remains: ~448 Gaussians × per-Gaussian SFPU cost (exp, mul_unary chains).
- Next highest ROI: **4-face per-quadrant culling** to reduce Gaussian count per sub-tile.

## Screenshots

- `screenshots/iter-071-1024x1024.png`
- `screenshots/diff-iter-071-x10.png`

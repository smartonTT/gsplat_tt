# Iter 000 — Baseline (Blackhole P150)

**Status:** locked (kept)
**Device:** Tenstorrent Blackhole P150 on `bh-30`
**Kernel:** `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp` (unchanged from `HEAD`)
**Date:** 2026-05-21

## Why this iteration exists

We need a fixed, reproducible starting point on the device we plan to optimize.
The optimization loop measures every subsequent iteration relative to this
baseline. The PNGs in `benchmarks/reference/` are the visual ground truth that
all future iterations are diffed against.

## How the numbers were measured

Each (scene, view) was rendered on `bh-30` with:

```bash
python scripts/render_fixed.py <scene> <view> --backend tt --warmup 2 --frames 3
```

The reported `kernel_ms` is the median of `sub_timings["blend.daemon_rt.device_kernel"]`
across the timed frames — kernel-only on-device elapsed, no host overhead.
`daemon_rt_ms` is the median round-trip per frame (kernel + DRAM upload/readback +
host→daemon stdin/stdout).

Cameras come from `benchmarks/cameras.json` (derived once via `scripts/derive_camera.py`,
opacity > 0.1 + drop top-10%-largest Gaussians, PCA-aligned orbit, OpenCV
convention). They are deterministic across iterations.

## Results (median over the timed frames)

| Scene      | View | Kernel ms | Daemon RT ms | E2E fps |
|------------|------|----------:|-------------:|--------:|
| stitch     | hero |     68.92 |       121.95 |    3.20 |
| stitch     | side |     75.66 |       158.47 |    2.69 |
| stitch     | top  |     85.38 |       143.83 |    2.93 |
| luigi      | hero |     15.11 |        34.44 |   14.31 |
| luigi      | side |     15.50 |        34.76 |   13.80 |
| luigi      | top  |     25.05 |        48.42 |   10.71 |
| strawberry | hero |    250.33 |       570.68 |    0.72 |
| strawberry | side |    247.85 |       564.61 |    0.73 |
| strawberry | top  |    230.81 |       528.91 |    0.81 |

(bicycle / `point_cloud.ply` deferred — 1.5 GB scene not yet on bh-30. We'll
add it later if its kernel cost shape differs from the other three.)

## Headline numbers (the ones the supervisor watches)

- **stitch hero kernel:** 68.92 ms → 14.5 fps kernel-only.
- **stitch hero end-to-end:** 320.5 ms → 3.2 fps in the viewer.
- 100× headline target: 0.69 ms kernel / 1450 fps.
- 10× headline target: 6.89 ms kernel / 145 fps.

For reference, the previous Wormhole baseline on the same `stitch hero` was
~151.8 ms kernel-only (≈6.6 fps kernel-only), so **Blackhole already starts ~2.2×
ahead** of Wormhole on the unchanged kernel.

## Notes on per-scene profile

- **Stitch (341K Gaussians, dark plush, lots of empty pixels):** kernel cost
  varies 69–85 ms across views — top-down has the largest visible footprint,
  costs most. Lots of opportunity for early-termination (interior pixels saturate
  fast in the dark fur).
- **Luigi (~26K Gaussians, very small scene):** sub-25 ms kernel everywhere.
  Already close to the floor of "fixed dispatch + readback" overhead. Improving
  Luigi is mostly improving the constant tail of the kernel.
- **Strawberry (~1.5M Gaussians, photographed fruit):** 230–250 ms kernel —
  3.3× more expensive than stitch on the same hardware, roughly proportional to
  the visible-Gaussians count (1.5M vs 341K ≈ 4.4×). This is the stress test for
  any change that improves throughput-per-pair.

## Files committed in this iter

```
benchmarks/cameras.json
benchmarks/reference/stitch_hero.png
benchmarks/reference/stitch_side.png
benchmarks/reference/stitch_top.png
benchmarks/reference/luigi_hero.png
benchmarks/reference/luigi_side.png
benchmarks/reference/luigi_top.png
benchmarks/reference/strawberry_hero.png
benchmarks/reference/strawberry_side.png
benchmarks/reference/strawberry_top.png
docs/optimization-log/000-baseline.md (this file)
```

No kernel or host code changed in this iter.

## Next iter

Iter 001 — block-wide early termination signalled by the reader RISC-V to the
compute RISC-V when all 32×32 = 1024 pixels in the tile have transmittance
below the early-termination threshold (e.g. T < 1e-3). Foundation for the
honest theoretical-peak calculation in iter 002.

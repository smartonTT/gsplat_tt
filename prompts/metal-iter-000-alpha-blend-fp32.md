# Metal iter-000 — fp32 alpha blend (4×8 microblock)

You are the metal-port worker for gsplat_tt_2 Phase 5. Read before coding:

- `opt/plan-amendment-001-metal-port.md`
- `opt/microblock-cpu-spec.md` §0–§3 (microblock enumeration + inner loop)
- `../gsplat_tt/docs/optimization-log/microblock-kernel-design.md` §3–§6 (TT kernel)
- `dev/tt-workflows/README.md` (bh-30 workflow, cache rules, never kill -9 daemon)

## Goal

Port alpha blend to tt-metal on Blackhole (P150, `bh-30`), **fp32 throughout**.
Prefer the **4×8 microblock-major** inner loop from the microblock proposal.
If that is too large for one iter, ship a **naive 32×32 per-tile** fp32 blend first,
then refactor to microblock-major in metal-001.

## Gates

- **Layer 2:** `python3 scripts/verify_blend_metal.py` → PSNR ≥ 80 dB vs
  `tests/fixtures/hero/blend_output.npy` (bicycle hero fixture).
- **Layer 3:** `scripts/run_iter_metal.sh 0 alpha-blend-fp32 port` → 30-view
  PSNR ≥ 60 dB vs `benchmarks/reference_v2/` when tt blend-only path is wired;
  for iter-000, Layer 2 alone is sufficient to KEEP.

## Files to touch

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
- `backends/tt/.../reader_alpha_blend.cpp`, `writer_alpha_blend.cpp`
- `backends/tt/.../alpha_blend.cpp`, `alpha_blend_host.h`
- `scripts/verify_blend_metal.py` (only if fixture I/O needs tweaks)

## Build on bh-30

```bash
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
ninja -C $TT_METAL_HOME/build metal_example_gaussian_splatting
python3 scripts/verify_blend_metal.py --backend tt
```

## Notes

- Existing kernel uses bf16 SFPU paths; switch compute to fp32 for correctness first.
- CPU project/tile_assign/sort stay on host; only blend runs on device for now.
- PSNR should be very high — almost no visible diff vs numpy. Visible tile grids = REJECT.

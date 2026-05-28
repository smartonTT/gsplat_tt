# Metal iter-001 — 4×8 microblock-major blend inner loop

Worker: Composer 2.5 on bh-30. Read:

- `opt/plan-amendment-001-metal-port.md`
- `opt/microblock-cpu-spec.md` §0–§3
- `../gsplat_tt/docs/optimization-log/microblock-kernel-design.md` §1 (TT kernel)

## Prerequisite

metal-iter-000 fp32 tile CB path must reach ≥80 dB hero PSNR (or ≥bf16 baseline 47.5 dB) before microblock refactor.

## Goal

Refactor `alpha_blend_compute.cpp` inner loop to microblock-major order using
`TT_SFPLOAD` / `TTI_SFPMAD` / `TT_SFPSTORE` + replay buffer per microblock-kernel-design §1.
Host emits `mb_header` + `mb_stream` per tile (see microblock-cpu-spec §3).

## Gates

- Layer 2: hero PSNR ≥ 80 dB
- Layer 3: 30-view PSNR ≥ 60 dB; `kernel_ms` median drop vs iter-000

## Build

```bash
scripts/sync_kernels_bh30.sh
REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 1 microblock-major port
```

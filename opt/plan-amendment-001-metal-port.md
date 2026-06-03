# Plan Amendment 001 — Metal Port (Phase 5) [SUPERSEDED]

> **SUPERSEDED by `opt/plan-amendment-002-tt-emulator-port.md` (2026-05-27).**
> The daemon + `.npy` IPC + legacy full-tile kernel approach in this amendment
> diverged from the `cpu_cpp` emulator design and produced an 8× slower blend
> with no microblock cull on device. Amendment 002 replaces it with an
> in-process pybind port that swaps one C++ kernel at a time. Keep this file
> for historical context only.

**Date:** 2026-05-27
**Supersedes:** Phase 5 deferral in `opt/plan.md` (CPU plateau reached: iter-057 sum30 ≈ 322 ms @ 1024²).

## Pivot

- **Benchmark scene:** `bicycle` (`scenes/bicycle.ply`, 6.1M Gaussians, iconic hero = Mip-NeRF360 `_DSC8679`).
- **Reference:** `benchmarks/reference_v2/` regenerated with numpy `cpu` backend @ 1024² (frozen iter-000 bicycle).
- **Correctness target:** PSNR vs numpy reference ≥ 60 dB (same north-star invariant). Metal fp32 alpha blend should aim for ≥ 80 dB on hero fixture before microblock cull lands.
- **Hardware:** `bh-30` on `tt_aus` (P150 Blackhole). Workflow: `dev/tt-workflows/README.md`.
- **Perf north star:** 1 ms/frame (1000 fps) after full port + optimization.

## Render paths (updated)

| Path | Language | Purpose |
|---|---|---|
| `cpu` | numpy | Bit-truth spec (frozen reference) |
| `cpu_cpp` | C++ | CPU perf vehicle (done) |
| `tt` | tt-metal | Metal port target |

## Metal iteration order

Same TDD pyramid as CPU port, but Layer 1 = host-side fixture tests + on-device golden compares; Layer 2 = PSNR vs numpy on hero fixture; Layer 3 = 30-frame benchmark on bh-30.

| iter | change | gate |
|---|---|---|
| metal-000 | fp32 alpha blend, 4×8 microblock-major inner loop (or fallback 32×32 tile) | hero fixture PSNR ≥ 80 dB vs numpy blend output |
| metal-001 | wire `tt` backend blend-only path (CPU project/sort, TT blend) | 30-view PSNR ≥ 60 dB, hero ≥ 65 dB |
| metal-002+ | port project → tile_assign → sort → microblock_cull in order | per-stage fixture match + 30-view PSNR ≥ 60 dB |
| metal-N+ | SFPU/bf16 optimizations, LPT dispatch, frame coherency | sum30 regression budget per iter |

Spec: `opt/microblock-cpu-spec.md` §0–§3 for algorithm; full TT kernel design in `../gsplat_tt/docs/optimization-log/microblock-kernel-design.md`.

## Iter loop

Driver: `scripts/run_iter_metal.sh` (runs on bh-30 via SSH or locally when on box).  
Log: `opt/metal-iters.jsonl`.  
Prompts: `prompts/metal-iter-NNN-*.md`.

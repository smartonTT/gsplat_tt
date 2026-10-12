# t510: FPU quadratic-form blend alpha (draft, stage 0 in progress)

Task #510, 2026-10-12 (PDT), base best-iter-223 (cd33efcd on this branch).

## Stage 0: CPU precision model (`qfmodel.py`, `run_cpu.sh`)

The model runs the gsplat CPU pipeline (project, tile assign, sort) on bicycle 1024x1024 and
blends every 32x32 tile in fp32 with the device rules (conic with -0.5 folded, exp(min(p, 0)),
alpha clamp 0.99, pixel floor 1/255, T cut at 1e-4). Only the power differs between variants:

- `base`: device order, p = A dx dx + B dx dy + C dy dy in fp32.
- QF variants: p = sum of features [u², uv, v², u, v, 1] times coefficients [A, B, C, D, E, F],
  operands rounded (RNE) to tf32 (10 mantissa bits) or bf16 (7), products and sums in fp32
  (the FPU's fp32 DEST accumulation). Origins: tile corner, tile centre (u, v in ±15.5, all
  features exact in tf32), or microblock centre (u ±3.5, v ±1.5, exact in bf16). `hilo` adds a
  second coefficient term (c - hi rounded), i.e. a second matmul pass.

Results so far: `base` hero 42.493 dB vs `benchmarks/reference_v2/hero.png` (device at
best-iter-223: 42.51 dB, docs/b2b-timeline-p150-223), so the model tracks the device.
Synthetic check (500 random gaussians, one tile, max |p - p_fp32|): tf32 corner 0.70, tf32
centre 0.19, tf32 centre hi/lo 2.4e-4, tf32 microblock 0.52 (relative 4e-3), bf16 centre hi/lo
0.16, bf16 microblock hi/lo 4.0e-3. Hero and 30-view PSNR per variant: pending (run_cpu.sh).

## Cost model: why the FPU form does not pay at today's occupancy

The blend walk is sparse: a live (gaussian, tile) record covers about 3 of the 32 microblocks
(3.06 in docs/blend-analysis, ~2.8 in t189; 2.07 per record including dead ones in t148). The
FPU computes a whole 32x32 power tile per record, ~10x more pixels than the walk uses.

Per live record, TRISC1/SFPU cycles (1.35 GHz):

| item | cycles |
|---|---|
| saved: x/y loads + 13-op front per microblock, replaced by 1 power load (14 ops x ~2.9) | ~41 |
| saved: 5 of the 7 staged coefficients (pushes overlap SFPU work under A2) | 0-22 |
| cost: one tile matmul, K=16, 8 MVMUL (8x16x16 each) x 4 fidelity phases (tf32 needs HiFi4) | 32 |
| cost: per-record coefficient operand (SFPU builds [C, Bx+E, Ax²+Dx+F] per column, ~8 ops, then DEST->srcB move) or a bilinear Yf M_g Xf form (two matmuls, 12 MVMUL x 4 + DEST->srcA move) | 12-56 |
| cost: SFPU<->FPU hand-offs on the one math thread (stall-waits drain the pipes, ~3 per record) | ~15-30 |

Saved 41-63, cost 59-118: break-even at best, about +0.3 to +0.5 ms/view slower in the middle.
The FPU-free bound (#111 ablation scaled to today's 3.40 ms/core tile_blend_sfpu, 19%) is
~0.65 ms/view; the realistic FPU costs eat all of it. LoFi bf16 (8 cycles) would fit the
budget but needs microblock-centred features and hi/lo coefficients (2 passes, per-microblock
coefficients), which brings back the coefficient build cost. A sparse per-microblock MVMUL
(8x16 output, 2 microblocks) still costs ~8 FPU cycles x fidelity + a hand-off per dispatch
against 14 ops saved.

Recommendation (pending the stage-0 numbers): shelve the device build; precision is not the
blocker, occupancy and the shared math thread are.

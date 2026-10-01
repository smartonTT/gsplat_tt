# Task #111 — lever 4 probe: FPU quadratic form for blend

**Verdict: no-go.** Taking the conic off the SFPU saves at most 2.0 ms/view, and
only if the FPU work costs nothing. A real FPU path costs part of that and cannot
be md5-identical.

## What was measured

`GSPLAT_TT_BLEND_FPU_QF_ABL=1` (default 0, compiled out) is a timing-only ablation in
`render/kernels/compute/alpha_blend_compute_mb.cpp`. The blend bodies skip
`dx, dy, A dx^2 + B dx dy + C dy^2` and the 5 coefficient stagings
(mx, my, A, B, C), and SFPLOAD a ready-made "power" from DEST instead. That is
exactly what an FPU quadratic form would hand the SFPU, with the FPU, unpack and
coefficient-tile costs set to zero. The output is wrong by design.

A fake power changes T saturation, so both comparison arms run with
`BLEND_T_PERIOD=0`: p0 = period 0, q0 = period 0 + ablation. d = default.

Setup: yyzo-bh-07 p100a, bicycle, 30 views, `render/run.py --no-ref`, 2 interleaved
rounds, tree at 12fc308 (base origin/smarton/tt-project-opt 840fff0). Raw lines: `ab.log`.

| arm | frame ms/view (r1, r2, mean) | blend ms (r1, r2, mean) |
|---|---|---|
| d  (default)        | 29.44, 29.48, **29.46** | 10.70, 10.68, **10.69** |
| p0 (period 0)       | 29.33, 29.59, **29.46** | 10.63, 10.62, **10.63** |
| q0 (p0 + ablation)  | 27.54, 27.36, **27.45** | 8.58, 8.61, **8.59** |

**q0 − p0: blend −2.03 ms, frame −2.01 ms/view.** No other stage moves beyond
noise (project 6.42–6.46, tile_assign 1.42–1.44, sort 10.61–10.83, d2h 0.21–0.28).

Default arm: hero_vs_ref = 100.00 dB against the golden, same `.so` md5 as the base.
With the flag off the ablation is not compiled in, so the default output does not change.

## Why no-go

1. **The ceiling is low.** It is 2.0 ms with the FPU free. Task #93 asked for at
   least 3 ms (blend + cull) before a deep FPU task. The cull's quadratic is part of
   a ~1.2 ms SFPU mask that is now fused into sort_subchunk_mat, so blend + cull
   tops out around 2.5 ms.
2. **The FPU is not free.** The FPU and SFPU are both issued by the math thread
   (TRISC1), and the matmul result needs DEST space that today holds the pixel
   ramps, the staging rows and the accumulators. The coefficients would have to
   arrive as srcB tiles through the unpacker, not as RISC register stagings.
   A dense (pixel × gaussian) matmul also gives up the jump walk's sparsity: the
   blend only visits the set (gaussian, microblock) pairs. Even with K padded only
   to 16 and a hi/lo coefficient split for accuracy, the saving left over is
   likely ≤ ~1 ms, which is under the 1.5 ms gate.
3. **It cannot be md5-identical.** tf32 operands with a different summation order
   give a different power than the SFPU's fp32 FMAs. Getting close needs hi/lo split
   coefficients plus tile-centered coordinates, and it would still need a golden
   refreeze and a PSNR gate.

## What the number does show

The conic plus coefficient staging is ~19% of blend (2.0 of 10.6 ms). That is more
than its static share (~16 of ~130 insns per pair body, task #78), because the
5 stagings per gaussian also go away. Any later blend rework that cuts the per-gaussian
staging (e.g. precomputed per-tile coefficients loaded by the unpacker, or fewer
staged scalars) goes after this same ~2 ms.

## Reproduce

```
bash docs/blend-fpu-qf-t111/run_all.sh <rev>   # sync+build, then one devrun chunk per round under ttp lock p100
```
`render/run.py` refuses to open the device unless launched through `devrun.sh` (TTW_DEVRUN).
The first attempt (run 323) called it directly and got no output.

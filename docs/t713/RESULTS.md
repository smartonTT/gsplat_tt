# t713: T2 segment reduce as FPU matmul — results (2026-10-14, bh-49 p150a, 110 cores)

| arm | syn K=8 ms (med) | syn K=1 ms (med) | oracle sh0 grad / Adam | oracle sh3 grad / Adam |
|---|---|---|---|---|
| t697 RISC gather (before) | 13.85 | 3.18 | 9.8e-5 / 8.2e-6 PASS | PASS |
| run 2: tf32 hi + fp32 lo, 8 matmuls | 6.29 | 2.81 | 0.110 / 0.027 FAIL | 0.135 / 0.25 FAIL |
| run 3: bf16 hi+mid+lo, 12 matmuls (ec71ee0e) | 6.21 | 2.71 | 1.17e-3 / 1.14e-3 FAIL | 2.70e-3 / 5.97e-4 PASS |

Bars: grads 1e-2, Adam 1e-3 (per-element relative, floor 1e-4 of max).
Run 3 fails sh0 only on `delta_quats` (1.14e-3); params 3.7e-6, m/v exact, m bf16 bits equal.

Finding: the reference (`t2_step`) gives identical fp32 and fp64 results, so the device's
~1e-3 per-element grad error (all five params, sh0) is real precision loss of ~10 bits, not
fp32 summation order (which would be ~1e-6). The three bf16 pieces are exact through SrcA per the
model in tests/unit/test_t2_segmm.py, so the loss is elsewhere in the matmul path: suspects are
Dst accumulation across the 12 matmuls/windows (fp32_dest_acc_en / HiFi4 really on for this op),
the pack format of the matmul output CB, or SrcB fidelity. Speed: 2.2x at K=8, 1.17x at K=1.

Raw: run3-bf16x3/, run2-tf32pair/. Log: tt-project/state/runs/2022/t713e.log.

## t726: root cause of the ~2^-10 loss, and the fix (cbb60f4a)

Root cause: the 32-bit Dst transpose (`transpose_wh_dest<true>`) moves the low 16 bits of each
value through SrcB. SrcB was still configured for the bf16 matmul operands, so those bits were
reformatted on the way and g2d came back with mantissa bits 9-14 flipped (~2^-10 relative). Found
offline from the run-3 dumps: the back-solved device g2d differs from the exact g2d only in those
bits; the matmul body, Dst accumulation and the Float32 pack are exact. Fix: set fp32 Src formats
(`reconfig_data_format(CB_K, CB_P, CB_A, CB_P)`) before the transpose. Source regression test in
tests/unit/test_t2_segmm.py fails without it.

Run 4 (fix cbb60f4a vs t697 gather 311181db), bh-49 p150a, 110 cores, 10 reps, one `ttp lock`,
separate JIT caches (ttmc-t726-fix / ttmc-t726-gather): `ab_jit_check --kernel t2_compute` JIT-HASH OK.

| arm | syn K=8 ms (med / min) | syn K=1 ms (med / min) |
|---|---|---|
| t697 RISC gather (311181db) | 13.85 / 13.85 | 3.19 / 3.18 |
| FPU segment matmul + fp32 Src transpose (cbb60f4a) | 6.15 / 6.13 | 2.76 / 2.70 |

Speedup 2.25x at K=8, 1.16x at K=1; the fix costs nothing measurable (run 3: 6.21 / 2.71).
The gate's oracle outputs were deleted on the box by a drive.sh quoting bug (fixed in 3371c380),
so sh0/sh3 were rerun separately (below). Raw: run4-fp32src/.

Run 4 oracle (3371c380, 03:59 PDT): sh0 grad 1.17e-3 / Adam 1.14e-3 FAIL, sh3 2.70e-3 / 5.97e-4 PASS,
the same to every printed digit as run 3. The fp32-Src reconfig changed nothing (the LLK's 32-bit
transpose sets its own Src formats), so the transpose diagnosis above was wrong.

## t726, second diagnosis: the FPU sums one matmul's products with limited alignment

Back-solving device g2d from the run-4 grads (docs/t726/backsolve.py; grads are linear in g2d,
lstsq residual 6e-9) and comparing with the fp64 segment sum: lanes with ONE slot are exact (to
back-solve noise), lanes with 2 or 4 slots are off by ~4096 fp32 ulps (~2^-11 relative), e.g. a
2-slot lane -2.1e-3 + -1.7e-5 gives xor 0x1000. A transpose fault would hit 1-slot lanes as well.
So hi + mid + lo of one slot accumulate exactly across matmuls in fp32 Dst, but the products of
several slots summed inside ONE matmul (one piece of every slot of the lane) lose the low bits of
the smaller ones: the FPU aligns them before adding. Per-slot pieces have per-slot exponents, so
their sum needs more alignment width than the FPU keeps.

Fix (this commit): block-float pieces. Per page and stream column, E = exponent of the column's
largest |x|; piece k is the truncation of the remainder to a multiple of 2^(E-7-8k) (< 256 steps, so
bf16 exact). All terms one matmul sums share a grid, so alignment loses nothing; the split itself
drops < 2^(E-23) per slot. Offline with exact sums (tmp model): sh0 grad 1.3e-4, sh3 3.5e-4, Adam
delta exact. Host-only change (t2_bench.cpp; the T1 producer would emit the same pieces); the
no-op transpose reconfig is reverted. Model test: tests/unit/test_t2_segmm.py::test_slot_split_shared_grid.

### t726 run 5 (d2256866, block-float pieces) on bh-49 (p150a, 110 cores), 2026-10-14 04:30 PDT

T0 oracle (python3 -m gsplat.t2_dev check, bars grad 1e-2, Adam 1e-3), both PASS, m bits equal:

| dump | worst grad | worst Adam | worst pfwc |
|---|---|---|---|
| sh0 | 1.34e-4 (run 3: 1.17e-3; t697 gather 9.8e-5) | 2.23e-5 (run 3: 1.14e-3 FAIL) | 3.7e-5 |
| sh3 | 3.80e-4 (run 3: 2.70e-3) | 8.8e-6 (run 3: 6.0e-4) | 1.0e-5 |

These match the offline exact-sum model (sh0 1.3e-4, sh3 3.5e-4), so the remaining error is the
split's own truncation, not the FPU.

Synthetic timings (10 reps each, median ms; same session, separate JIT caches):

| case | FPU segment matmul (d2256866) | t697 RISC gather (311181db) | speedup |
|---|---|---|---|
| SYN_K=8 | 6.13 (min 6.11) | 13.85 (min 13.85) | 2.26x |
| SYN_K=1 | 2.73 (min 2.69) | 3.19 (min 3.16) | 1.17x |

Both deltas (7.7 ms, 0.46 ms) are far above the ~0.1 ms noise bar. ab_jit_check: JIT-HASH OK
(t2_compute hashes differ between arms, as expected). Logs: docs/t726/out/ (text outputs only).
The block-float split is host-only and costs nothing on device (same 12 matmuls per window).

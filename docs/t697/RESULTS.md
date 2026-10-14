# t697 — T2 device param pass: results (2026-10-14, bh-49, Blackhole p150a, single chip)

Kernels: `render/train/kernels/t2_{reader,compute,writer}.cpp`, host `render/train/t2_bench.cpp`,
layout/check `gsplat/t2_dev.py`, driver `docs/t697/drive.sh` (one `ttp lock p100`, repo-scoped sync).

## T0 gradient oracle (device vs `t2_step`, bars: grads 1e-2, Adam/pfwc 1e-3)

| dump | worst grad | worst Adam (params/m/v/delta) | worst pfwc | m bf16 bits equal | pass |
|---|---|---|---|---|---|
| t0_sh0_64x64_n50 | 9.8e-5 | 8.2e-6 | 3.7e-5 | yes | PASS |
| t0_sh3_64x64_n50 (SH band 0) | 7.6e-5 | 8.8e-6 | 1.0e-5 | — | PASS |

T2 owns SH band 0 only; for sh3 the check compares band 0 (higher bands are another pass).

## Timing (synthetic, N = 1,048,576 Gaussians, 73% visible, 110 cores, 10 reps, median device ms)

| mean slots/Gaussian | slots | max per group | device ms (min / med) |
|---|---|---|---|
| 8 | 8.39 M | 27 | 13.85 / 13.97 |
| 1 | 1.05 M | 9 | 3.18 / 3.22 |

Time scales with the number of rank planes per group (kmax), about 4.7 us per plane per group
(64 lanes x 9 words copied by the reader RISC, plus the CB hand-off and the SFPU plane add).
The fixed per-group projection/backward/Adam/pfwc work is small next to it. So the RISC rank-plane
gather is the bottleneck (the design finding of the first run, now measured), and padding every
group to its kmax wastes another ~2x at mean 8.

## Next lever (not built)

Replace the RISC gather with an FPU segment reduce: the staged slot span is already rows of 16
fp32 streams (one 64 B slot = one face row), so G = Seg (64 x span, 0/1) @ Slots (span x 16)
gives the 64 x 16 segment sums in one matmul chain, no per-lane copies and no kmax padding.
Precision must be checked against the oracle (fp32 inputs at HiFi4); lossless rule (msg #384)
applies. Then the bicycle-view T1 slot check (spec step 2) on top.

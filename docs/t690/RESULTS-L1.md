# t690 L1 loss on device: results (p100 (provisional), bh-33)

Run 2026-10-13 23:42 PDT, commit 0cfef825, job t690-diag (run 1956).
Kernel: render/bench/loss_bench.cpp, fp32, unpack-to-dest fp32 on CBs 0/1/3 (default, LOSS_U2D=1).

| case | shape | u2d | loss vs oracle (rel) | dL/dimage mismatches | b2b ms | launch ms (med) |
|---|---|---|---|---|---|---|
| dump64 | 64x64x3 | 1 | 1.2e-12 | 0 (PASS, bitwise grads) | 0.0106 | 0.034 |
| syn | 822x1237x3 | 1 | 7.0e-10 | 0 (PASS, bitwise grads) | 0.1169 | 0.145 |
| dump64 | 64x64x3 | 0 | 3.6e-4 | 5 (FAIL, TF32 unpack) | 0.0100 | 0.035 |
| syn | 822x1237x3 | 0 | 2.0e-3 | 7311 (FAIL, TF32 unpack) | 0.1168 | 0.145 |

Oracle: gsplat/train_oracle.py L1 loss and dL/dimage. The earlier "hang" (run 1938) did not
reproduce with either u2d setting; stage markers show launch, readback and timed launches all
complete. Likely a stale build in the shared remote dir (the driver now resyncs on that case).
fp32 unpack-to-dest costs nothing measurable (0.1169 vs 0.1168 ms).

Not done yet: D-SSIM (11-tap separable Gaussian as banded matmul on FPU), fused fwd+bwd,
and timing on the p150 measurement box.

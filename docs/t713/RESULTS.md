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

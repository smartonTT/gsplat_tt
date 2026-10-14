# t716: T0 oracle check of the device fp32 training forward (blend-only harness)

Run 2026-10-14 02:29 PDT on bh-49 (p150a, Blackhole), commit 53615db1, via `docs/t716/drive.sh`
inside `ttp lock p100` (repo-scoped sync with opt/sync_remote.sh). Evidence in `out/`.

Harness: `render/train/blend_fwd_bench` runs the production `alpha_blend_compute_mb.cpp` with the
train-mode defines (GSPLAT_TT_BLEND_TRAIN=1) at 64x64, fed the oracle's fp32 means2d/conic and
UNORM16 opacity/colours (`gsplat/blend_forward_dev.py`). Output: fp32 R/G/B/T buffer.

## Forward vs T0 oracle (bar 1e-4 max abs)

| scene | max abs R | max abs G | max abs B | max abs final_T | vs numpy model | result |
|---|---|---|---|---|---|---|
| t0_sh0_64x64_n50 | 8.64e-6 | 7.03e-6 | 8.06e-6 | 7.05e-6 | <= 1.2e-7 | PASS |
| t0_sh3_64x64_n50 | 6.52e-6 | 7.35e-6 | 7.71e-6 | 7.05e-6 | <= 1.2e-7 | PASS |

The device output matches the numpy model of the BLEND_TRAIN math to fp32 rounding (1.2e-7); the
~8e-6 gap to the oracle is the UNORM16 quantization of opacity/colour in the PACK2 records.
No pixel hit early stop in either scene. Forward device time at 64x64: 0.159 ms min (5 reps).

## T1 backward with the device forward image (bar 1e-2 rel err)

`blend_backward_dev prep --image` = the device forward buffer (box image.f32 == local re-check).

| scene | rel err (max term) | worst term | result |
|---|---|---|---|
| t0_sh0_64x64_n50 | 9.18e-4 | dmy 1.87e-3 | PASS |
| t0_sh3_64x64_n50 | 3.12e-4 | dA 3.12e-4 | PASS |

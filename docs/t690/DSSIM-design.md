# T7 fused D-SSIM on device: design (task #690)

Loss 0.8·L1 + 0.2·(1−SSIM), fp32, exact against the T0 oracle (gsplat/train_oracle.py).
Model of the device algorithm: gsplat/train_loss.py (`loss_and_grad_device`), tests in
tests/unit/test_train_loss.py. L1 step: render/bench/kernels/loss/ + render/bench/loss_bench.cpp.

## Math (closed-form fused backward)
Target y is fixed per view: μy = G*y and E_yy = G*y² are per-view constants. Per step:
forward filters μx, E_xx = G*x², E_xy = G*(xy); per-pixel S and three partials
(dS/dμx, dS/dE_xx = −S/D, dS/dE_xy = 2S/B, times w = −0.2/(3HW), padding masked);
backward filters P, Q, R of those; dL/dx = P + 2x·Q + y·R + 0.8·sign(x−y)/(3HW).
G is symmetric, so the adjoint filter is G again.

## Filter as banded matmul on the FPU
G = G_h·Z·G_wᵀ. On 32×32 tiles each band matrix has three non-zero tile blocks
(R[−1], R[0], R[1], `band_tiles()`): row pass out[:,J] = Σk Z[:,J+k]·R[k],
column pass out[I,:] = Σk R[k]ᵀ·Z[I+k,:]. 3 tile matmuls per output tile per pass.

## Precision (measured in the model against the fp64 oracle)
FPU operands are TF32 (10-bit mantissa) even with fp32 CBs and fp32 dest.

| scheme (forward / backward filters) | 1237×822 bicycle, grad L2 rel | max err / max |grad| |
|---|---|---|
| fp32 / fp32 (reference arithmetic) | 1.2e-4 (vs real GT pair, earlier run) | 3.3e-2 |
| tf32 / tf32 (1 matmul per band tile) | 67 (unusable: E_xx − μx² cancels) | — |
| split3 / split3 (xh·gh + xl·gh + xh·gl) | 5.3e-6 | 7.9e-5 |
| split3 / tf32 | 2.0e-3 | 9.2e-3 |

Decision: split3 on all six filters (3 matmuls per band tile). tf32 on the backward filters
alone costs 2e-3 relative grad error, which is not "exact" (msg #384); rejected.

## Cost
6 maps × 2 passes × 3 band tiles × 3 (split3) = 108 tile matmuls per tile per channel;
26×39×3 = 3042 tile-channels at 1237×822. If μy and E_yy are recomputed each step instead of
cached, +36. Caching them costs 2 planes × 12.2 MB per view in DRAM (194 views: 4.7 GB).

## Partitioning: column strips, neighbour contributions pushed, no recompute
Core c owns one tile column J of one channel (39×3 = 117 strips) and walks down the 26 tile rows.
- Forward row pass needs Z[:,J±1] where Z ∈ {x, x², xy}: elementwise of x, y, so the core
  just reads x, y tiles of J−1..J+1 (no recompute). Column pass: rolling 3-tile-row window, local.
- S and the partials at column J are local.
- Backward: run the column pass first (local, rolling), then the row pass needs columns J±1.
  Instead of fetching, core J computes its own contributions to its neighbours:
  P_c[J→J'] = Pcol[:,J]·R[J−J'], and, since it already holds x, y of J±1, the combined
  2-tile contribution P_c + 2x[J']·Q_c + y[J']·R_c for J' = J−1 and J+1 (one tile each per tile
  row, not three maps). The neighbour adds the two incoming tiles to its own result.
- Exchange: v1 writes the left/right contribution planes to DRAM and a small second launch
  sums them with the L1 term (≈3×12 MB write + read, ~0.1 ms). v2 sends them to the neighbour's
  L1 over the NoC with a semaphore per tile row (2×4 KB per tile row per core) in one launch.
- Strips > cores: split each strip into two row halves with a 1-tile-row halo (forward needs
  rows I±1 for the column pass, backward I±2 → recompute 2 tile rows at each cut).

## Order of work
1. L1 kernel (done in loss_bench, checked bit-exact vs model, see out-*/).
2. One-core split3 filter kernel (μx only) checked against `filt_tiles(..., "split3")`.
3. Full forward + S + partials + backward on column strips, DRAM exchange (v1), checked against
   `loss_and_grad_device(..., "split3")` and the oracle dumps; then timing.
4. NoC exchange (v2), μy/E_yy caching, fusing the L1 term.

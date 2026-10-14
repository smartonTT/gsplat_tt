# t749: on-device block-float split — results (2026-10-14, measurement box bh-49)

Branch ttp/t749-t2-on-device-block-float-split-replace-h, kernel commit 67927d02, gate driver 4634e976.
Gather arm: t697 311181db. Separate JIT caches per arm; ab_jit_check JIT-HASH OK in all 4 rounds.

## Gate (T0 oracle, bars grad 1e-2 / Adam 1e-3)
| dump | worst grad | worst Adam | M bits equal | pass |
|---|---|---|---|---|
| sh0 | 1.336e-4 | 2.23e-5 | yes | yes |
| sh3 | 3.797e-4 | 8.84e-6 | yes | yes |
Same as #726 run 5 (host split): the device split is bit-identical in effect.

## Synthetic timing, device_ms_med (10 reps), 4 alternating rounds
| arm | K=8 r1..r4 | K=8 mean | K=1 r1..r4 | K=1 mean |
|---|---|---|---|---|
| fix (device split) | 6.511 6.523 6.483 6.481 | 6.50 (spread 0.04) | 3.622 3.573 3.604 3.600 | 3.60 (spread 0.05) |
| gather (t697) | 13.849 13.853 13.913 13.854 | 13.87 (spread 0.06) | 3.187 3.251 3.189 3.160 | 3.20 (spread 0.09) |

- K=8: fix 2.13x faster (-7.37 ms). #726 host-split fix was 6.13 ms, so the device split costs about +0.37 ms.
- K=1: fix 0.40 ms slower than gather (+12.6%), well above the ~0.1 ms noise floor. #726 host split was 2.73 ms,
  so the device split costs about +0.87 ms here (it runs per slot page; K=1 has few matmuls to hide it behind).

## Verdict
The FPU reduce with on-device split is lossless (same oracle numbers) and wins clearly at high slot counts (K=8),
but loses at K=1. For the trainer step, pick per workload: use it when the mean slots per group is well above 1;
a straight-line interpolation between the two measured points (not measured) puts the crossover near K~1.4.
Before replacing the gather, measure on the real bicycle slot distribution, or dispatch per group bucket (gather
for K=1 groups, FPU reduce for K>=2).

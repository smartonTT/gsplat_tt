# T1 device blend backward: results (#693, 2026-10-13)

Kernel: render/train/kernels/{bwd_reader,blend_bwd_compute,bwd_writer}.cpp, host
render/train/blend_bwd_bench.cpp, driver docs/t685/drive.sh (one `ttp lock p100`: preflight,
opt/sync_remote.sh, build, run, pull, check). Device: bh-33, **p100 (provisional)**; the timing
of record is p150 and still to come. Raw files: docs/t685/results/.

v1 differs from DESIGN.md: one pair per pack (the 9 per-lane term sums of the ACC tile are
packed as one fp32 tile per pair) and the host does the 32-lane reduce and the finish
(A Su + B Sv, ...). Dst holds T, BH, g (3) and the x/y ramps as fp32; records stay fp32.

## Oracle check (`blend_backward_dev check`, bar max rel err <= 1e-2, floor 1e-4 of scale)

| dump | pairs | rel err | worst term | pass |
|---|---|---|---|---|
| t0_sh0_64x64_n50 (first run, forward's `_sfpu_exp_21f_bf16_`) | 74 | 1.25e-1 | dmy 1.8e-1 | **fail** |
| t0_sh0_64x64_n50 (fp32-accurate exp) | 74 | 8.8e-5 | dC 8.8e-5 | pass |
| t0_sh3_64x64_n50 | 74 | 6.8e-6 | dop 6.8e-6 | pass |
| synth s1 256x256, N=3000 (docs/t685/synth.py) | 7,975 | 3.5e-4 | dop 5.7e-3 | pass |
| synth s2 512x512, N=12000 | 35,052 | 9.4e-4 | dmy 4.8e-3 | pass |

Finding: the forward's exp (`_sfpu_exp_21f_bf16_`, a degree-2 2^f polynomial, about 1.7e-3
relative error) is not accurate enough for gradients. The errors were uniform at about 5e-4 of
each term's scale, so slots that are small after cancellation failed (only 1-3 slots per term).
The kernel now uses the fp32-accurate exp (Juffa's algorithm, the same as tt-metal's
`_sfpu_exp_fp32_accurate_`), inlined in the kernel. The numpy model with a model of the old exp (`blend_backward_dev.exp21f`)
reproduces the failed device run term for term (rel err 0.1250 vs 0.1253 on device; dmy 0.185
both), so the cause is confirmed; tests/unit/test_blend_backward_dev.py keeps it. Consequence for training: the forward's
training mode must use the same exp, or its image and T disagree with the backward's
recomputed alpha.

The synthetic dumps (s1, s2) are timing workloads, checked against the numpy reference on the
same inputs (the image is an fp32 forward with the device rules). The slightly larger errors
there (up to 5.7e-3 on single small slots) come from fp32 summation order over many more
pairs per pixel, still under the bar.

## Timing, p100 (provisional), bh-33

Host wall time of EnqueueMeshWorkload + Finish, median of 5, after a JIT warm-up run.
Includes dispatch; excludes the host lane reduce (listed separately).

| dump | pairs | live vectors | cores | max pairs/core | device ms | host reduce ms |
|---|---|---|---|---|---|---|
| sh0 | 74 | 721 | 4 | 21 | 0.069 | 0.05 |
| s1 | 7,975 | 76,701 | 64 | 224 | 0.499 | 4.70 |
| s2 | 35,052 | 326,498 | 110 | 454 | 1.367 | 17.8 |

Slope s1 to s2: about 3.8 us per pair on the busiest core (about 10 live vectors per pair).
v1 packs and writes a 4 KB tile per pair and leaves the lane reduce to the host, so its cost
per pair and its output size (4 KB per pair, 143 MB for s2) do not carry to the hero view.
No hero-view number yet: the hero needs v2 (below) and a hero dump.

## Not done in this task (moved to the follow-up)

- Hero-view timing with a per-stage split (needs v2 and a real hero-view forward state).
- Forward training flag (fp32 image, accurate exp, slab kept alive) with the flag-off
  b2b raw_md5 1d372b94 check and the on/off A/B (opt/ab_jit_check.py on both arms).
- DRAM/L1 footprint at hero scale.

## Next steps (proposal)

1. **T1 v2**: do the 32-lane reduce on device (SFPU transpose/shuffle into one vector, or FPU
   row-reduce via a ones matmul, T6) and the 9-float finish per pair; write 36 B per pair to
   slot P[g] + j. This removes the 4 KB pack + write per pair and the host reduce. Check again
   on sh0/sh3/s1/s2, then time s2 against v1.
2. **Hero dump**: a vectorised prep (the current one loops per pair in Python) plus a hero-view
   forward state from bicycle (means2d, conic, opacity, colours, order, fp32 image, dL/dimage
   from the L1 loss against _DSC8679 at 1237x822). Time v2 on it, p100 first, then p150.
3. **Forward training flag** with the accurate exp, fp32 image output and the slab kept alive,
   so T1 consumes the forward's own pair lists (no host prep). Flag-off b2b raw_md5 must stay
   1d372b94; on/off A/B with opt/ab_jit_check.py on both arms.
4. **T2** (fused segment reduce + projection/SH0 backward + sparse Adam) against the slot
   layout above; it can start now on the oracle slots.

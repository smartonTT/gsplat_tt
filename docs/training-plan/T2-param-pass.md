# T2 param pass: segment reduce + projection/SH0 backward + Adam + pfwc epilogue (#689)

2026-10-13 PDT. Labels: **M** measured (cited), **E** estimate from measured numbers (basis
shown). No device run in this task: the p150 measurement box (#687, bh-49) was still
being set up, so nothing here is timed on device.

## 1. Reference and oracle gate (done)

`gsplat/t2_param_backward.py` is the fp32, lane-wise model of the one device pass (every op
is elementwise over Gaussians, the way the SFPU kernel runs it over 1024-row SoA tiles):

1. segmented reduce of T1's Gaussian-major slots, rank-major (plane k = slot P[g]+k of every
   Gaussian with counts[g] > k), so the fp32 sum order is j ascending, bit-identical to T1's
   own per-Gaussian sum (`test_segment_reduce_matches_t1_sum_order`);
2. projection forward recompute (pc, J, cov3d, cov2d, conic, colour clamp, visibility) and the
   analytic backward to means, log_scales, quats (with the normalisation), opacity logits and
   SH band 0 (clamp mask);
3. Adam, sparse (visible rows, SelectiveAdam / T0 `sparse_adam_step`) by default, `dense=True`
   for the A/B; fp32 params and math, moments stored bf16 (round to nearest even) or fp32;
4. epilogue: pfwc / gather_visible inputs (mx,my,mz, cov3d unique c00..c22, opacity, rgb),
   fp32 SoA padded to 1024 rows.

Gate (`python3 -m gsplat.t2_param_backward benchmarks/train_oracle/t0_sh0_64x64_n50.npz`,
`tests/unit/test_t2_param_backward.py`), T0 slots from `blend_backward_ref` with the oracle's
frozen live masks (**M**, Mac):

| check | bar | committed dump | fresh case (seed 7) |
|---|---|---|---|
| param grads, worst group (sh) | rel 1e-2 | 4.1e-4 | 1.7e-4 |
| Adam bf16 + fp32: params, updates, m, v (oracle grads in) | rel 1e-3 | 1.6e-5 | 5.9e-5 |
| end to end (slots -> params after Adam, bf16 moments) | rel 1e-3 | 1.8e-5 | pass |
| visibility mask = oracle `visible` | exact | yes | yes |

Lossless (msg #384): nothing is approximated beyond fp32 and the oracle's own bf16 moment
storage. Training quality parity (published 3DGS/gsplat or LiteGS PSNR/SSIM/LPIPS) is checked
by M1, not here.

## 2. Finding: sparse Adam saves bandwidth only after a spatial reorder

The pass is DRAM-bound (~300 SFPU ops per Gaussian is ~0.1 ms on 110 cores; the bytes below
are ms). A SoA kernel reads and writes whole blocks, so a visible-only update costs the blocks
that hold any visible row, not the visible rows. `docs/t689/tile_visibility.py` (**M**, CPU,
bicycle.ply N = 6.13M, every 8th of the 169 NN-tour training views at 1237x822, the oracle's
visibility rule; raw output `docs/t689/tile_visibility_stride8.txt`):

| order | visible rows | blocks touched, B = 32 | B = 1024 |
|---|---|---|---|
| file order | 21.0 % | 92.4 % | **100 %** |
| Morton (Z-order of means, one permutation at load) | 21.0 % | 22.3 % | **30.2 %** |

In file order the "sparse" pass moves as many bytes as the dense one. A one-time Morton
permutation of the Gaussians (an exact relabelling: same params, same renders; the forward's
depth sort does not depend on storage order) cuts the touched blocks to 30 %. The block skip
needs a per-block "any visible" flag, which pfwc's PFWC_VIS mask (128 B per 1024-row tile)
already has. Densify/prune (T8b) must keep the order (append new rows near their parent's
block, or re-sort on each densify step).

## 3. Byte budget and expected time (E)

Per touched row (SH0): read params 14 x fp32 + m, v 2 x 14 x bf16 = 112 B; write the same
112 B plus 10 pfwc streams x 4 B = 40 B (pfwc reads means straight from the master param
buffer, same SoA layout, so the 3 mean streams are not copied). 264 B per row, plus the slot
read (contiguous per block, below).

At ~420 GB/s (both movers streaming, docs/hw-ceilings.md, **M** p100; p150 published peak 512
vs p100 448 GB/s):

| N | dense / file order | Morton, sparse, B = 1024 |
|---|---|---|
| 6.13M (bicycle.ply) | 1.62 GB, ~3.9 ms | ~1.2 ms |
| 1.2M (parity) | 0.32 GB, ~0.8 ms | ~0.23 ms |

The plan's ~1.6 ms at parity N assumed visible rows only and 59 SH3 params; with SH0 and
the Morton order it is ~0.25 ms plus the slot reduce. SH3 (T4) adds 45 params per row at the same 16 B per param (fp32 r/w + bf16 m, v r/w).

## 4. Device kernel design (next task)

- Layout: Gaussians in Morton order; master params fp32 SoA (14 streams), moments bf16 SoA
  (28 streams), pfwc streams as today; all padded to 1024. Block = 1024 rows = one tile per
  stream. Cores take blocks round-robin from the list of touched blocks (PFWC_VIS mask).
- Slots: Gaussian-major means a block's slots are one contiguous span [P[g0], P[g0+1024]).
  Read it with bulk NoC reads (never one 64 B read per slot: 64 B reads cap at ~35 GB/s chip
  wide, hw-ceilings p5). T1's writer should pack the 9 terms as 36 B rows (or 9-float SoA
  per rank) rather than 64 B pages: 44 % fewer slot bytes on both sides.
- Rank planes: the reader (RISC-V, integer loads/stores in L1) scatters the span into 9 x
  kmax fp32 plane tiles (lane g = slot P[g]+k, zero past counts[g]); compute adds planes in k
  order (same sum order as the model). kmax per block is not measured yet: take the bicycle per-Gaussian
  slot-count histogram from T1 before sizing L1 (blocks over the L1 budget split by rank).
- Compute (SFPU, fp32 dest, approx off): forward recompute + backward as in the model; exp
  for scales and sigmoid, reciprocal + 2 Newton steps (pfwc_recip_nr.h) for 1/z, 1/det,
  1/|q|, Adam's 1/den; sqrt for |q| and sqrt(v). Moments: unpack bf16 to fp32, update, round
  to bf16 RNE in the SFPU (integer add 0x7FFF + lsb, mask) before packing, so storage matches
  the oracle bit for bit whatever the packer's rounding mode is.
- Writer: params + moments back in place, pfwc streams. Sparse rows: lanes with visible = 0
  write their input values back unchanged (the block is written whole).
- Check: run the device pass on the T0 dump's slots (padded to one block) and compare with
  `t2_step` (grads 1e-2, Adam 1e-3), then on a bicycle view's slots from T1.

## 5. What this changes in the plan

- T2 needs the Morton relabelling (load-time permutation of bicycle.ply, exact) to get the
  sparse-Adam saving; without it the pass is dense in bytes (~3.9 ms at 6.13M).
- The same order should help pfwc's chunk cull and L2's batch-union pre-cull, which also
  skip whole blocks (not measured here).

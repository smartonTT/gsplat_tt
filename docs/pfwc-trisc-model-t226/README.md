# pfwc TRISC compute cuts: model (task #226, no device)

All savings below are **modeled, not measured**. Inputs are measured: the T1 (math)
per-step times of the default tip 21edcf0
(`docs/pfwc-writer-split-t207/dev-t221/out/prof-both-pc_split.txt`) and the per-op
costs from t197 STEPCYC=2 (`docs/pfwc-breakdown-t197`). Model script: `model.py`,
output: `model-out.txt`. 1 ms/view of T1 time = 24.8k cycles per chunk (1.35 GHz,
54.4 chunks/core/view).

## Answer (ranked)

| # | candidate | modeled ms/view saved | md5-safe | verdict |
|---|---|---:|---|---|
| 1 | **P2: fused cov2d a/b/c + conic + radii in 2 sections (t197 lever 2)** | **0.48** (0.39-0.50) | yes, by construction | **BUILD** |
| 2 | R: radii reuse a/c only (subset of P2) | 0.25 | yes | no: below 0.3; P2 includes it |
| 3 | X: xform + recip + depth + means as one section | 0.13 (0.07-0.14) | yes | no: below 0.3 |
| 4 | vis/pop fusion into the conic section | <= 0.05 | yes | no: DEST does not fit; only copies saved |
| 5 | cut vis/precull SFPU instructions | ~0.1-0.15 (rough guess, 25% of a 0.53 ms pass) | risky (edge-exact logic) | no: below gate, and it needs an instruction count first |
| 6 | rebalance T0/T1/T2 | ~0 | n/a | no: the per-thread split is not a real imbalance (below) |
| 7 | xform on the FPU (matmul) | n/a | **no** (FPU srcA/srcB are TF32) | no |

After P2, pfwc TRISC models at 1.73 ms (1.71-1.82). That is still above the
writers (BRISC 1.12, NCRISC 1.19 ms), so the floor is TRISC and the whole 0.48
counts. Pfwc is on the critical path and TRISC-bound, so untraced savings are taken
as equal to traced ones (t221: the cov_cam SFPU cut paid in full untraced).

## Why T0/T1/T2 "imbalance" is not real

The pfwc compute kernel runs with `dst_full_sync_en = true` and unpack-to-DEST fp32
on every CB (`render/host/pfwc_device.cpp:419-452`). DEST is not double-buffered,
so each acquire/release section runs strictly in turn: unpack copies, math SFPU ops,
pack. All three walls are ~2.20-2.21 ms. A step that looks long on T0 or T2 (xform
0.49/0.75 ms) is the pack or unpack of a *neighbouring* section landing in that
step's window (T2 "a" is 0.003 ms, "radx" 0.007 ms). Only math can run SFPU ops,
only unpack can copy, only pack can pack, so there is no work to move between
threads. T1's sum of steps is the wall: cutting T1 work is the only lever.

## Cost model (fitted on measured data)

| item | cycles | source |
|---|---:|---|
| copy_tile (fp32 unpack-to-DEST) | 154 | t197 STEPCYC=2 |
| mul_unary_tile / add_unary_tile | 160 | t197 |
| binary tile op (mul/add) | ~140 | t197 (add 128, mul ~150) |
| one section sync or one pack | ~222 | fit: steps a..rady residual (3558 cycles / 16) |
| relu + sqrt + ceil on one tile | ~1460 | fit: radx/rady residual |
| SFPU instruction in an in-LREG pass | **1.02** cycles | calibrated on the t206 cov_cam pass (step 0.256 ms) |

Current steps a, b, c, conic, radx, rady (T1 0.997 ms = 24.7k cycles/chunk): 6
sections, 36 copies, 24 unary + 57 binary tile ops, 10 packs. Most of it is tile-op
dispatch, not arithmetic. Steps a, b, radx and rady each recompute j00/j02/j11/j12
from scratch (step a also computes j11/j12, which it never uses), and radx/rady
recompute a and c, which already sit in CB_TMP_A/CB_TMP_C.

## P2 design (the BUILD)

**DEST budget.** a, b and c together need cc00..cc22 (6) + inv_tz + tx + ty = 9
tiles; DEST holds 8 fp32 tiles under full sync. Folding cov_cam in makes it worse
(scales use tiles 6-7). So one section cannot hold all of lever 2. Split by output:

**S_AC** (replaces steps a, c, radx, rady). Copies (8): 0 cc00, 1 cc02, 2 cc22,
3 cc11, 4 cc12, 5 inv_tz, 6 tx, 7 ty. One SFPU loop over 32 vectors
(`#pragma GCC unroll 0`, `dst_reg++`), per vector in LREGs:
- L0 = inv; L1 = j00 = inv*fx; L2 = j02 = ((tx*(-fx))*inv)*inv;
  L3 = j11 = inv*fy; L4 = j12 = ((ty*(-fy))*inv)*inv.
- a = (((cc00*j00)*j00) + (((cc02*j00)*j02)*2)) + ((cc22*j02)*j02)) + 0.3
  (L5 accumulator, L6 term, L7 constants 2.0 / 0.3).
- c the same with cc11, cc12, cc22, j11, j12.
- Store a to slots 0 and 6, c to slots 3 and 7 (tx/ty at this vector are already
  in LREGs).

Then the existing tile ops on slots 6 and 7: `relu_tile`, `sqrt_tile`,
`mul_unary_tile(k)`, `ceil_tile`. Packs (6): 0 to CB_TMP_A, 3 to CB_TMP_C, 6 to
OCB(CB_RX) + CB_TMP_RX, 7 to OCB(CB_RY) + CB_TMP_RY.

**S_BC** (replaces steps b and conic). Copies (7): 0 cc02, 1 cc01, 2 cc12, 3 cc22,
4 inv, 5 tx, 6 ty. SFPU loop: same j's, then
b = ((((cc01*j00)*j11) + ((cc02*j00)*j12)) + ((cc12*j02)*j11)) + ((cc22*j02)*j12).
Store b to slots 1 and 7. Copy CB_TMP_A to slot 0 and CB_TMP_C to slot 2 (cc02/cc12
are dead): 9 copies in all. Run the existing `pfwc_conic_unroll<0>` on 0..2. Packs
(4): A, B, C to OCB outputs, slot 7 to CB_TMP_B (precull reads raw b).

**Bit-exactness.** Every product is a separate SFPMUL (x*y+0) and every sum a
separate SFPADD (x*1+y), in exactly the order the tile ops use. That is the t206
cov_cam pattern, md5-verified on device. With fp32 DEST, `mul_binary_tile` is plain
in0*in1 and `mul_unary_tile` / `add_unary_tile` are x*s / x+s. No SFPMAD fusion.
recip, relu, sqrt and ceil stay as the existing tile ops, so their LLK bit patterns
(e.g. sqrt(-0), the ceil fix-up) are untouched. Expected md5 46a725ab.

**LREG budget.** At most 8 live: inv, j00, j02, j11, j12, accumulator, term,
constant. inv is dead after the four j's; its LREG then holds 2.0, and 0.3 is loaded
into L7 once per vector. fx, -fx, fy and -fy use SFPLOADI pairs (fp32 values are
not bf16-exact) or the three programmable constants.

**MADs per element.** S_AC: 8 (j's) + 7 mul + 3 add (a) + 7 mul + 3 add (c) = 28
arithmetic, ~52 instructions with loads, LOADIs and 4 stores. S_BC: 8 + 8 mul +
3 add = 19 arithmetic, ~35 instructions. Today the same work is 81 tile ops of
~140-160 cycles each, plus 19 extra copies and 4 extra sections.

**Modeled new cost.** S_AC ~8.1k and S_BC ~4.8k cycles/chunk (central CPI 1.2),
against 24.7k now: -11.9k cycles = **-0.48 ms/view** (CPI 1.0: -0.50; CPI 2.0:
-0.39). T0/T2 share: they shrink with T1, since the sections are serialized. T0
loses 19 copies per chunk, and T2 loses 1 pack + 4 section handshakes.

**kcfg.** The program is 92496 B, which fits only at +24 KB (95232 B) with ~2.7 KB
headroom. P2 removes ~81 inlined tile-op calls (math) and 19 copy_tile calls
(unpack), and adds two looped SFPU passes (~52 + 35 instructions, ~0.4 KB). Net
code should shrink. If it does not fit, raise the default open to +32 KB (t221:
extra kcfg costs nothing visible). Tracy keeps needing +32 KB.

## Not built (why)

- **X (front fusion, 0.13 ms).** Steps 1-5 are only 0.341 ms of T1 and already
  pack-heavy (9 packs must stay). One section saves 12 copies, 24 tile ops and 6
  syncs, but adds 2 SFPU passes (~54 instructions/vector) plus recip. Below the gate.
  Consider it only as an add-on in the P2 task if P2 lands (same helper style).
- **vis/pop fusion.** precull needs raw a, b, c, op, rx, ry plus a temp, and vis
  needs tz, op, mx, my, rx, ry plus params and a flag slot. Neither fits next to the
  3 conic outputs in 8 tiles. Best case is saving a few copies and one sync,
  <= 0.05 ms.
- **vis SFPU cut.** The vis+precull pass is ~13.2k cycles/chunk (0.53 ms), but most
  of it is predicate logic that must stay edge-exact. Without an instruction count,
  a 25% cut (~0.13 ms) is a guess. If revisited, first split STEPCYC into precull
  and vis.
- **Rebalance.** See above: ~0.
- **xform on the FPU.** FPU srcA/srcB carry TF32 (19-bit), so fp32 means lose bits
  and the md5 changes. The xform arithmetic is also only ~0.1 ms of tile ops; its
  cost is copies and syncs, which X addresses.
- **DEST double-buffering (`dst_full_sync_en = false`).** It would let pack overlap
  math, but it caps sections at 4 fp32 tiles. cov_cam, vis and P2 all need 6-8, so
  they would split into many more sections and copies. Not pursued.

## Follow-on: 2:1 BRISC:NCRISC chunk deal (docs/pfwc-feed-model, f6251b4)

t222 measured feed slack at 0.045 ms/view and shelved feed cuts until TRISC drops
below the writers. After P2, TRISC models at ~1.73 ms, still above NCRISC 1.19 ms,
so the 2:1 deal is modeled at ~0 right after P2. Queue it gated: run it only when a
measured pfwc TRISC wall is within ~0.05 ms of the slowest writer.

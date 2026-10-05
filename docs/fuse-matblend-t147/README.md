# t147: fuse materialize into blend per core (blend lever 1): measured inputs and model

Board: yyzo-bh-07 p100a (IRD), bicycle 1024x1024. Base b54c54e (PRECULL=1 tip)
plus 805125f, which adds `GSPLAT_TT_MB_TILECYC=1` (per-tile blend MATH cycles
via DPRINT; it compiles out when off). Driver: `drive_tc.sh` / `remote_tc.sh`.
Model: `sim.py`. Outputs: `out/t147-tc.dprint`, `out/t147-model.txt`.

**Result: the fused kernel was not built.** Measured per-tile data fed into a
schedule model predicts about 0 ms/view gain for the design as specified, and
the gate is 1.5 ms. Per the spec, I stopped and did not run add-ons A and B.

## Measured

- Tip, 30 views, untraced: **19.06 ms/frame**. The blend stage (materialize +
  host gap + blend) is 9.27 ms. Hero vs golden: 100.00 dB.
- Per-tile blend MATH time (TRISC1, wall clock 1350 MHz) for views 0 and 1,
  1024 tiles per view:

| view | records | live records | blend MATH sum | mean per core | greedy makespan | tiles > 8192 | max tile |
|---|---|---|---|---|---|---|---|
| 0 (hero) | 3,236,947 | 72.3% | 638.0 ms | 5.80 ms | 5.86 ms | 100 | 25,019 |
| 1 | 2,326,963 | 72.7% | 484.5 ms | 4.40 ms | 4.46 ms | 77 | 21,115 |

- **TRISC cost model per tile** (least squares fit, both views):
  `blend_us ~ 85-90 + 0.027 * rec + 0.197 * live`, rms 63-69 us. Here
  `live` means records whose mask is still non-zero after T-saturation. A dead
  record costs the blend about 0.03 us; a live one about 0.22 us.
- Tiles above 16384 records (6-7 per view) blend at 0.106-0.119 us per record,
  about half the overall 0.197-0.208, because many of their records are dead.
  Each still takes 1.5-3.0 ms of blend.

## Model (`sim.py`, us costs)

Inputs:
- the measured per-tile blend times above;
- the t144 mover fits: whole tile NCRISC 5.5 + 0.197n, BRISC 16.1 + 0.187n,
  n > 6144 -282 + 0.2226n, big-tile sort 0.1083n + gather 0.1267 per record;
- the SFPU cull at 0.04 us per record on the TRISC, from t86/t90;
- 1.5 us per TRISC entry switch.

The fused schedule:
- each mover has one L1 slab, and CB_SLAB is also the sort's scratch;
- NCRISC claims tiles with n > 6144 first, BRISC claims tiles with n <= 6144;
- the TRISC runs the cull and blend of each slab in arrival order.

| variant (ms per frame, blend stage) | hero | view 1 |
|---|---|---|
| today: materialize LPT + 0.33 gap + blend | 9.94 (3.75 + 0.33 + 5.86) | 8.11 (3.33 + 0.33 + 4.46) |
| **fused as specified** | **9.88** | **8.22** |
| + a second slab per mover | 9.11 | 7.86 |
| + big tiles delivered already sorted | 8.68 | 7.13 |
| + host LPT on measured TRISC cost (oracle) | 8.02 | 6.55 |
| TRISC floor (mean cull + blend per core) | 6.98 | 5.25 |

Why the fused path gains so little:
1. **The cull moves onto the critical path.** In today's materialize window the
   SFPU cull (~1.2 ms per core) runs alongside the movers, which bound that
   window. Fused, it runs before every blend on the same TRISC.
2. **Each mover has one slab.** The slab is also the sort's scratch, so a
   mover cannot sort or permute the next tile or subchunk until the TRISC has
   blended the current slab. A second slab would cost +256 KB (NCRISC) and
   +192 KB (BRISC) of L1. It does not fit: the materialize program already uses
   ~1.37 MB, plus ~45 KB for the blend CBs.
3. **Big tiles form a long serial chain.** A tile above 16384 records needs a
   0.108n sort on one NCRISC before its first slab (2.3-2.7 ms). Its 3 slabs
   then run gather, cull and blend in turn on one core, an 8-10 ms chain. Today
   other cores materialize those subchunks in parallel.
4. **Pipeline fill.** The TRISC idles until the first tile has been read,
   sorted and permuted.

The model overestimates today's materialize window (hero 3.75 ms in the model,
3.34 ms traced). The fused rows use the same mover fits, so the comparison
holds roughly, but this is a model, not a device A/B of a fused kernel. The
gate is reached only with all three extra pieces. One of them does not fit in
L1, one does not exist (a big-tile pre-sort path), and the oracle LPT needs
`live` counts that the host does not have before the blend.

## What the data points to instead

1. **The materialize window is set by its largest items, not by total work.**
   Busiest mover vs mean mover in the model: hero 3.75 vs 3.03 ms, view 1
   3.33 vs 2.23 ms. Each big-tile subchunk item re-sorts the whole tile
   (0.108n). Cutting the largest item could give up to ~0.7-1.1 ms by the
   model; given the model's bias, more likely 0.3-0.7 ms. Ways to do it: split a
   large tile's sort across the core's two movers, or sort once and run gather
   items that depend on it. This matches t144's verdict.
2. The blend is already balanced: greedy makespan 5.86 ms vs mean 5.80 ms. A
   better LPT there gains nothing.
3. 28% of the records that reach the blend are dead (zero mask after
   saturation). Each still costs ~0.2 us of mover time, 0.04 us of cull and
   0.03 us of blend.

Reproduce: `ttp lock p100 -- bash docs/fuse-matblend-t147/drive_tc.sh <rev>`, then
`python3 sim.py out/t147-tc.dprint --launch 1|2 [--dbuf 1] [--big_free 1]`.

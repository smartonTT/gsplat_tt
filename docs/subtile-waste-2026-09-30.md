# Sub-tile (4x8 microblock) culling waste on bicycle (2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150: none was obtainable, see project ledger).
Scene: bicycle, 30 views, 1024x1024, `render/run.py --no-ref`. Commit base 76ffb66 + this change.
Raw logs: `opt/profiler/t9-mbstats/`.

## Instrumentation

`GSPLAT_TT_MB_STATS=1` (host env) compiles scalar MATH-thread counters into
`alpha_blend_compute_mb.cpp`; each blend core DPRINTs one `MBSTATS` line per launch
(needs `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=TR1 TT_METAL_DPRINT_FILE=...`).
`GSPLAT_TT_MB_STATS_INV_FLOOR=N` sets the "non-negligible alpha" floor to 1/N (default 255).
`profiler/mb_stats_summary.py <log> --launches 31` aggregates (31 = warmup + 30 views;
110 cores x 31 launches = 3410 lines). With the env unset the code is compiled out.

A pixel of a dispatched microblock is "live" iff `op*exp(min(power,0)) >= floor`, same
power formula and pixel centres (c+0.5, r+0.5) as the blend. A microblock is "useful" iff
it has >= 1 live pixel.

## Measured (per frame, mean over 31 launches)

| quantity | count / frame | ratio |
|---|---:|---:|
| (a) splat-tile pairs reaching blend (slab records) | 3,077,443 | 1.000 |
| (b) pairs with non-zero microblock mask | 2,928,868 | 0.952 of (a) |
| microblock slots if every pair covered the whole tile (32 x a) | 98,478,176 | 1.000 |
| microblocks kept by SFPU cull | 13,868,875 | 0.141 |
| ... skipped by T-saturation early-out | 2,575,817 | 0.026 |
| microblocks actually blended (SFPU dispatches) | 11,293,058 | 0.115 |
| (c) blended microblocks with a live pixel, floor = pipeline 1/16384 | 9,870,440 | 0.874 of blended |
| (c) blended microblocks with a live pixel, floor = GPU 1/255 | 5,993,817 | 0.531 of blended |
| live pixels / blended lanes, floor 1/16384 | 134.0 M / 361.4 M | 0.371 |
| live pixels / blended lanes, floor 1/255 | 63.3 M / 361.4 M | 0.175 |
| blended microblocks per pair-dispatch (ILP pairing) | | 1.39 |
| blended microblocks per live pair | | 3.86 |

## Timing

| run | avg ms/view | blend bucket ms | hero vs golden |
|---|---:|---:|---:|
| default (stats compiled out) | 166.6 | 63.59 | 100.00 dB (bit-identical) |
| instrumented (DPRINT + scalar pixel loop) | 1562.6 | 1459.0 | 100.00 dB |
| default, `contrib_floor` 1/16384 -> 1/255 (cameras json) | 160.0 | 57.04 | 51.97 dB |

Default blend bucket 63.59 ms matches the pre-change stage profile (63.60 ms).

## Read

1. The "32x" whole-tile waste is ~88% recovered already: only 11.5% of the 32-per-pair
   microblock slots are blended.
2. At the pipeline's own floor (1/16384), 12.6% of blended microblocks (1.42 M/frame) have no
   live pixel: this is the pure granularity loss of the continuous-box cull. 4.8% of pairs
   (148 k/frame) have mask 0 but are still streamed through blend.
3. Most of the rest is a threshold choice, not granularity: bicycle runs at
   `contrib_floor = 1/16384`, 64x below the GPU 1/255 skip. Under 1/255, 47% of blended
   microblocks contribute nothing. Measured: moving the floor to 1/255 cuts blend
   63.59 -> 57.04 ms and the frame 166.6 -> 160.0 ms (-4.0%), at 51.97 dB vs golden
   (not bit-identical, so it needs a golden re-baseline decision).
4. Blend time is not proportional to microblock work: the floor change removed at most
   5.3 M dispatches/frame for 6.55 ms, i.e. >= ~1.24 ms per million dispatches. The
   11.3 M dispatches then account for <= ~14 ms of the 63.6 ms bucket; the other ~50 ms
   is per-record streaming, cull and fixed cost.

## Estimated recoverable ms/view on this axis (bit-identical)

- Granularity loss (1.42 M useless dispatches x ~1.24 ms/M): ~1.8 ms.
- Mask-0 records streamed (4.8% of records x <= ~50 ms per-record share): <= ~2.4 ms.
- Sub-microblock lanes (63% idle at 1/16384): not recoverable; 32 lanes is the SFPU vector
  width, the same granularity a GPU warp rejects at.

Total <= ~4 ms/view (~2.5%). Recommendation: do not pursue finer-grain culling. Bigger
levers are the per-record blend cost (~50 ms), host `bin_emit` in sort (~30 ms) and
project (~37 ms). The 1/255 floor is a cheap measured 4% if the golden may be re-based.

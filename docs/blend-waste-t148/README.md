# Blend waste counters (task #148, blend lever 3)

Question: how much of the blend's per-frame cost goes to work that contributes
nothing at 1/255? Gate: queue a deep work-cut task only if >= 3 ms/view.

**Answer: ~0.43 ms/view (mean core), 0.63 ms on the worst core. Below the gate;
no work-cut task proposed.**

## Setup
- Code: 9ef7c0a (based on 0654fa7, which includes b54c54e: PRECULL=1, refreshed goldens).
  The tip has since moved to PRECULL=2 (t174), which changes sort/emit only, not the blend loop.
- Device: yyzo-bh-07 p100a, under `ttp lock p100`. 30-view bicycle set + hero (31 blend launches, 110 cores).
- Base run (no stats): 18.48 ms/frame, blend 9.30 ms/view, hero 100 dB vs golden (`out/t148-base.log`).
- Stats run: `GSPLAT_TT_MB_STATS=1`, DPRINT on TR1 (`out/t148-stats.dprint`); also 100 dB vs golden.
  The stats build reads T back before every record to get true saturation; the kernel's own
  live mask and early-out are unchanged, so the counts describe the real dispatch stream.
- Summary: `python3 profiler/mb_stats_summary.py out/t148-stats.dprint --launches 31` (`out/summary.txt`).
  Cost model: 65 cycles per microblock dispatch, 44 cycles per record, 1.35 GHz, mean over 110 cores.

## Numbers (per frame)
| item | count | share | ms/view |
|---|---:|---:|---:|
| records | 2.95 M | | |
| dead records (dispatch nothing) | 0.82 M | 27.8 % of records | 0.244 |
| ... of which mask 0 after cull | 0.49 M | 16.5 % of records | |
| microblock dispatches | 6.12 M | | |
| dispatches into already-saturated mb | 0.33 M | 5.4 % of disp | 0.145 |
| dispatches with no live pixel at 1/255 | 0.10 M | 1.6 % of disp | 0.043 |
| wasted dispatches (union) | 0.43 M | 7.0 % of disp | 0.186 |
| pair-ops fully wasted | 0.29 M | 6.0 % of pair-ops | |
| **total (dead records + union dispatches)** | | | **0.430** |

Per core (`out/percore.txt`): waste 0.34 / 0.42 / 0.63 ms (min / median / max).

## Caveats
- The counted loop cost (records x 44 + dispatches x 65) is ~3.55 ms per core, about 38 % of the
  9.30 ms blend stage; the rest is reads, setup, writeback and waits this model does not cover. Even
  if every cycle constant were 2.6x too low, the waste would be ~1.1 ms, still under 3 ms.
- Not counted as waste: dead lanes inside a dispatched microblock. Only 32 % of the pixels in
  dispatched microblocks are live (`px_live / (32*mb_disp)` = 0.325). That is SIMD granularity, not
  skippable work; cutting it would need finer-grained masking, a different lever.

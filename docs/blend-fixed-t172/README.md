# Blend per-tile fixed cost (task #172)

Question: the t147 fit (`docs/fuse-matblend-t147`) gave the blend
`us/tile ~ 85-90 + 0.027*rec + 0.197*live`. Is the 85-90 us/tile intercept a real
per-tile fixed cost (acquire/init, T read-back, staging, emit), and can it be cut?
Gate for a cut: md5-identical and paired A/B >= 0.3 ms/view.

**Answer: no. The real per-tile fixed cost on the critical thread (MATH) is ~10 us/tile,
~0.1 ms per core per view, and almost all of it is the periodic T read-back. The 85-90 us
intercept was a fit artifact: the fit had no term for the microblock dispatch count. With
that term the intercept drops to 1-3 us/tile. No cut built (upper bound ~0.1 ms < 0.3 gate).**

## Setup
- Code: d3bae7d (instrumentation) on 04f86dc (iter 186 tip, PRECULL=2, K2 diet+fold).
- Device: yyzo-bh-07 p100a, under `ttp lock p100`. Bicycle 1024x1024, views 0:2
  (3 blend launches: warm-up + 2 views), 110 cores.
- `GSPLAT_TT_MB_TILECYC=1`: MATH (TRISC1) counts wall cycles per tile split into
  records loop / T read-back / stage / tail / init / emit, DPRINTed at kernel end (`out/tc.dprint`,
  0 dropped tiles). Split + fits: `python3 tc_split.py out/tc.dprint` (`out/tc_split.txt`).
- `GSPLAT_TT_BLEND_PROF=2`: Tracy sub-zones cmp_init / cmp_stage / cmp_trb / cmp_sc_tail / cmp_emit
  on all three TRISCs (`out/zones.txt`; csv on yyzo-bh-07
  `/localdev/smarton/gstt2-t172/opt/profiler/t172-prof/chunks/0-2/`, not committed, 37 MB).
- Drivers: `drive.sh` (local), `remote_prof.sh` (remote).

## MATH per core, view 0 (launch 1)
| part | ms/core | per tile |
|---|---:|---:|
| wall | 5.784 | |
| records loop | 5.683 (98.3 %) | |
| T read-back (5143 reads, 1.1 us each) | 0.097 | ~10 us |
| init (acquire, fills, ramps) | 0.004 | 0.41 us |
| stage + tail + emit | <0.001 | 0.05 us |
| other (waits) | <0.001 | 0.05 us |

Loop fit with the dispatch count (rms 14.7 us/tile, vs 70 us without it):
`cyc_us ~ 1.0 + 0.016*rec + 0.080*live + 0.061*disp - 2.4*ntrb`.
Per core per view that is dispatches 3.66 ms (64 %), live-record staging 1.71 ms (30 %),
record scan 0.30 ms (5 %). The intercept moved into `disp`: dispatches per tile track
tile size but not linearly in records or live records.

The long cmp_trb / cmp_sc_tail / cmp_emit zones on UNPACK and PACK (80-103 us each) are
those threads waiting on MATH (read-back sync, the bulk-slot mailbox ack, tile_regs_wait).
They are off the critical path.

## Conclusion
- Per-tile fixed cost is not a lever. The biggest piece, the T read-back, is ~0.1 ms/view
  and also drives the saturation early-out, so cutting it would cost more than it saves.
- Blend MATH time is the dispatch stream: ~82 cycles per microblock dispatch and ~108 cycles
  per live record (coefficient staging). #148 found only 7 % of dispatches wasted, so a
  blend lever has to make each dispatch or each staging cheaper, not cut tile overheads.

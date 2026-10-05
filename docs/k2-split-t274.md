# Task #274: K2's idle TRISCs (L2)

## Step 1: where K2's 0.98 ms goes (measured, GSPLAT_TT_K2_PROF=1 Tracy, yyzo-bh-07 p100a)

Raw: `docs/k2-split-t274/out/t274-k2p-k2_parts.txt` (220 movers, ms/view).

| part | mean | BRISC | NCRISC | busiest |
|---|---|---|---|---|
| pair loop (k2p_pairs) | 0.817 | 0.800 | 0.835 | 0.873 |
| whole kernel (k2p_tot) | 0.846 | | | 0.905 |
| loop NoC issue + flush (riss+rdw+wiss+wfl) | 0.070 | | | |
| of which NoC wait | 0.007 | | | |
| loop compute | 0.747 | | | |

Compute is 88.3% of K2 busy time (91.7% with setup and rows); NoC wait is under 1%.
2,640,222 pairs/view, 91.9 loop cycles/pair, 84.0 compute cycles/pair. K2 is
compute-bound on two RISC-Vs per core while the three TRISCs idle.

## Route models

- (a) K2 inside the sort_ol launch: K2 end -> sort start gap 0.011 ms, K2 tail 0.063 ms;
  modeled saving 0.009-0.010 ms. Below the 0.25 ms gate: **killed**.
- (b) TRISC producer: compute share 88% >= 40% gate: **alive**. Split each mover's pages
  [job0 43% -> TRISC0 (BRISC) / TRISC2 (NCRISC)][job1 21.5% -> TRISC1, both movers in turn][mover 35.5%].
  If TRISCs run the loop at mover speed, K2 drops from ~0.85 to ~0.39 ms (~0.45 ms/view).
  Host model in `tests/unit/test_pfwc_fuse.cpp` (run_jobs) matches the plain path on all
  161 cases (45,705 jobs, 21 window fallbacks).

## Build (d71d9b6, GSPLAT_TT_K2_TRISC=1, default off)

`k2_trisc.h` (job layout), `k2_trisc_compute.cpp` (TRISC side), mover side in
`tile_assign_scatter_seg.cpp`: diet_start for the three part starts, NoC-read each job's
lofs/box window (<= 640 pages) into L1, post header + GO, run its own part, wait DONE,
NoC-write the job's pair pages and add its tile counts. Windows over the cap fall back to
the mover doing everything.

## A/B (yyzo-bh-07 p100a, untraced bicycle 30 views, swapped order, one build d71d9b6)

Mean of the 30 per-view times (`[run] view=` lines, 0.1 ms resolution each).

| round | order | A (off) ms/view | B (K2_TRISC=1) ms/view | A-B |
|---|---|---|---|---|
| 1 | B,A | 11.673 | 11.510 | 0.163 |
| 2 | A,B | 11.693 | 11.387 | 0.306 |
| 3 | B,A | 11.763 | 11.497 | 0.266 |
| mean | | 11.710 | 11.465 | **0.245** |

md5: all 6 runs SWEEP_MD5=46a725ab, 30/30 views identical to golden.

## Decision

B is faster in all 3 rounds (0.245 ms/view, 2.1%), but below the 0.3 ms keep gate and
about half the 0.45 ms model, which assumed the TRISCs run the loop at mover speed.
**Not kept as default**: GSPLAT_TT_K2_TRISC stays default off. Logs: `out/ab-r*.log`.
Next step (follow-up): a Tracy capture with K2_PROF + K2_TRISC=1 to get per-part times
(mover part vs TRISC jobs vs the DONE wait), then rebalance the 43/21.5/35.5 split to the
measured speeds. If the TRISC jobs are the long pole, a better split can reach the model.

# t291 (t274b) K2 TRISC rebalance — progress notes (run 774)

Device yyzo-bh-07 p100a, bicycle 30 views, untraced means of `[run] view=` lines (`means.sh`).

## Tracy, K2_TRISC=1, split 430/215 (commit b544d00), `out/t291-tr0-trisc_zones.txt`
ms per core per view (3300 core-views):

| job | prep (wait for window) | run | slack |
|---|---|---|---|
| B job0 (TRISC0) | 0.117 | 0.266 | 0.057 |
| B job1 (TRISC1 1st) | 0.117 | 0.130 | 0.194 |
| N job0 (TRISC2) | 0.150 | 0.271 | 0.085 |
| N job1 (TRISC1 2nd) | 0.289 | 0.136 | 0.081 |

Mover k2_pairs B 0.441 / N 0.506, core span 0.526, slowest core per view 0.626.
TRISCs are faster per page than the movers (full range ~0.62 vs 0.80-0.835 ms), but each
job waits 0.12-0.29 ms for its whole window to land before starting. Shifting the split alone
cannot help much: own part and jobs are already about balanced.

## Change: chunked window fill (commit after b544d00)
Mover posts both jobs first (sizes from a no-read diet_window pass), then fills job 1, job 0
in K2_FCHUNK (64) page steps, raising H_FILL; the TRISC's WinIo::wait_reads waits on H_FILL.
Knobs: GSPLAT_TT_K2_TJ0, _TJ1, _FCHUNK.

Untraced (one run each, md5 46a725ab, 30/30 identical to golden):
- old code B (430/215): 11.370 ms/view (chk2)
- chunked B (430/215): 11.387; chunked 500/250: 11.383 (chk3)
So no measurable gain yet (noise ~0.02-0.05).

## Open problem
Tracy capture of the chunked build (K2_TRISC=1, no K2_PROF) hung the device
("Timeout waiting for ARC msg request queue"); board reset with tt-smi -r. Untraced runs are
fine. Earlier, K2_PROF markers + TRISC jobs under Tracy hung the same way. Suspect: profiler
buffer/flush interplay with the TRISC spin waits (GO / H_FILL), not a protocol bug (md5 clean).

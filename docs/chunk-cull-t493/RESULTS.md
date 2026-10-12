# t493 chunk cull + de-bursted sort emit: bh-30 p150 b2b A/B (shelved)

Board: bh-30 p150, 12x10 grid, aiclk 800, run under the 2026-10-07 viewer exception
(viewer stopped 17:32-17:37 PDT 2026-10-11, restarted, localhost:8091 -> 200).
Tree: 9c94bea1 (feb0875b + GSPLAT_TT_CHUNK_SCATTER). "off" = default config = best-iter-223 code path
(chunk cull is off by default). Metric: render/run.py --back-to-back, 30 bicycle views, 20 passes + 1 warm-up;
one discarded warm-up run, then 4 alternating rounds (off c48 c416 c416s / reversed).

| arm | config | b2b ms/view (median of 4 run medians) | md5 (12x10 golden 39d84b28) |
|---|---|---|---|
| off | default (iter 223) | **8.787** (8.783 8.789 8.786 8.790) | 30/30 OK |
| c48 | CHUNK_CULL=1 OL_RING=4 OL_RING_DEPTH=8 | 8.964 (8.961 8.965 8.963 8.964) | FAIL 5dbc0cc9 (all 30 differ) |
| c416 | CHUNK_CULL=1 OL_RING=4 OL_RING_DEPTH=16 | does not run: CBs 1827200 B > L1 1572864 B | - |
| c416s | c416 + CHUNK_SCATTER=1 | does not run (same L1 overflow) | - |

c48 is 0.177 ms/view slower than off. The gate (>= 0.15 ms faster) misses by 0.33 ms.

Why (dump-run stage means, ms/view, latency mode, diagnostic only):
- project (pfwc + gather wait): off 0.848, c48 0.866. On the p150 the cull gives no pfwc saving
  (the 0.525 ms saving was measured on p100a 11x10).
- sort bin_emit: off 0.407, c48 0.534 (+0.127): the burst stall in the onelaunch emit is still there.
- host xview (chunk-cull bounds, pfwc setup): off 0.138, c48 0.328 (+0.19).
So even a perfect emit fix leaves c48 at or above off on this board: no saving left to bank.
The scatter arm (idea a) could not be measured because it was paired with depth 16; it only reorders
gids inside each Morton chunk, so it cannot create a pfwc saving that the p150 does not show.

Conclusion: shelve chunk cull on p150. Stop line reached for this lever.

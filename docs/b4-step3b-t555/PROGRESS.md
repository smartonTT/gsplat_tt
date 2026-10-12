# t555 — B4' step 3b step 2: fuse NT=1 town sort(N+1) into mat+blend(N) — progress

Status 2026-10-12 16:50 PDT: building blocks done and checked on device; the fusion itself
(host ordering + kernel embedding) is not written yet. No A/B, no keep/shelve verdict.

## Done (branch ttp/t555-b4-step-3b-step-2-fuse-nt-1-town-sort-n-, iter 226 merged at 53c2b331)
- a2fe6f72 `GSPLAT_TT_OL_SOLO=1` (needs `OL_TOWN_NT=1`, default off): one sort stream per core on
  BRISC; `OL_YIELD()` hooks in every sort wait (barriers, town spins) for an embedding program to
  serve the blend writer there. Unit test: solo slots == two-mover slots.
- 9a12a01b every sort scratch buffer via `OL_CBP` (CB by default, `g_ol_arena[]` under
  `OL_ARENA=1`); host arena placer `render/host/sort_ol_arena.h` + `tests/unit/test_sort_ol_arena.cpp`.
  Finding: the R=8 run ring (266,240 B) fits no single blend alias-cut tail (largest, CB6,
  251,904 B, even counting dead CB21/CB22); R=7 (233,472 B) fits. The fused sort needs R=7 or a
  ring split across two tails.

## Device check @02576361 on bh-33 (out-check/, iter-226 baseline, FOLD on)
| config | md5 (30 views) | b2b median ms/view (1 round, 20 passes) |
|---|---|---|
| base (two movers) | 906e0435 30/30 (golden) | 8.601 |
| `OL_TOWN_NT=1 OL_SOLO=1` standalone | 906e0435 30/30 (golden) | 10.479 |

Solo is bit-exact. Standalone it costs +1.88 ms/view of device time: one stream per core does both
movers' work serially plus the town waits. That is expected and is not the fused number: in the
fused design this stream runs in the blend movers' idle CB waits (3.2/3.6 ms per core per view,
docs/b4-step3b-t553), so it only needs its whole span (about 1.46 + 1.88 = ~3.3 ms) to fit inside
mat+blend(N) (~7.3 ms device). It fits on paper, but with less slack than the gate assumed (the
gate used the 0.64 ms per-mover net, not the solo span with town waits).

## Left (next run)
1. Host (sort_device.cpp, blend_device.cpp) behind a new flag, default off: enqueue pfwc(N+1)+k2(N+1)
   before mat+blend(N); compile the sort into the blend program with `OL_ARENA=1`, place the run
   ring R=7 (or split) + town mailbox with sort_ol_arena.h; KCFG_EXTRA_KB 32->45; town addresses via
   OL_TOWN_ADDR_RTA. Mind the BRISC K2 fold (iter 226) budget.
2. matblend_brisc.cpp: include the solo sort as a state machine, `OL_YIELD()` = one blend-writer poll.
3. TRISC 2 (alpha_blend_compute_mb.cpp + sort_ol_town_compute.cpp): town in PACK's MATH-wait gaps.
4. drive.sh PLAN=ab ARM=<flag>: hero + diff + PSNR, 3 alternating b2b rounds vs 8.615.

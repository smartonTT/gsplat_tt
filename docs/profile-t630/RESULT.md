# t630: FOLD=1 fixed costs and the mat+blend overhang (best-iter-227, bh-33 p100)

Task #630, ledger U6 + U9. Build: best-iter-227 (445f9e40) plus the t630 counters (2aa6b5cc), FOLD=1
(the b2b default), bicycle, 30 views. Run 2026-10-13 12:18-12:31 PDT under `ttp lock p100`.

## Headline (untraced)

`render/run.py --back-to-back`, 30 views, 20 passes: **8.454 ms/view** (median 8.457, 118.3 FPS),
raw_md5 1d372b94. The counters compile out of this build.

## How it was traced

- The FOLD=1 program is too big for the profiler kcfg. `GSPLAT_TT_FZ_OFF=1` drops the fz_* mat
  counters, and `GSPLAT_TT_KCFG_EXTRA_KB=34` makes it fit. The MATCULL_DEPTH=1 fallback was not needed.
- `GSPLAT_TT_BZ_PROF=1` turns on the per-phase counters in the blend reader, compute and writer
  (`render/kernels/dataflow/bz_prof.h`).
- Six chunks of 5 measured views each (P0..P25) were stitched with `prep506.py`. FOLD=1 chunks have
  35 programs per core, which is 2 + 11×3. The steady order per view is pfwc(N+1),
  mat+blend(N)+K2(N+1), sort(N+1), so the analysis runs with `ANA_PROGS=mat+blend,sort,pfwc`. In each
  chunk the 5th "pfwc" slot holds the final mat+blend, so all tables use medians.
- Tracing inflates the frame. The device chain adds up to about 9.48 ms against 8.45 untraced, so it
  runs about 12% long. All ms below are **traced**. Scale by about 0.89 for untraced (that factor is
  an estimate).

Scripts: `bz630.py` (per-phase counters), `zones630.py` (zone timeline), `bound630.py` (overlap
bound), `loops630.py` (TRISC1 loops). Outputs are in `out/*.txt`.

## Attribution: mat+blend window (median 6.20 ms traced)

| part | all cores (median) | critical core | note |
|---|---|---|---|
| launch skew / preamble / postamble | 0.000 / 0.001 / 0.000 | 0 | not a cost |
| **mat phase** (TRISC mat_cull_mask; NCRISC+BRISC sort_subchunk_mat) | 1.42 (per-view mean 1.45, max 2.06) | **1.71** | **serial before blend on every core**: tile_blend_load starts at 1.357 when sort_subchunk_mat ends at 1.356, and tile_blend_sfpu starts at 1.418 when mat_cull_mask ends at 1.417 |
| CB setup (c_setup / r_setup) | 0.000 / 0.005 | 0.002 / 0.003 | negligible |
| per-tile T init (c_init, 10 tiles) | 0.005 | 0.005 | negligible |
| blend SFPU (c_blend) | 4.55 | 4.44 | compute-bound. c_cntw = 0: compute never waits for counts |
| reader: claim / meta / ready / DMA | 0.003 / 0.018 / 0.007 / 0.031 | ~0.02 total | negligible |
| reader bulk wait (r_bulkw) | 4.47 | 4.37 | the reader waits for the compute to free the slab, so it is idle, not a cost |
| writer K2 pre-phase (w_pre: k2_pairs 1.99 + k2_rows 1.05 on BRISC) | 2.00 | 1.72 | hidden: the writer then waits 2.1-2.5 ms for output tiles (w_outw) |
| writer pack (w_pack, scalar BRISC, 10 tiles) | 0.65 (65 µs/tile) | 0.65 | hidden except the last tile |
| output write + barrier (w_wr + w_bar) | 0.026 | 0.21 | |
| **drain** (TRISC1 end to core end) | 0.10 | **0.25** (per-view max 0.27) | last tile's pack and write |
| end spread (window - earliest core end) | 0.23 | | blend imbalance: max - mean 0.24 |

Per view (`bound630.py`): window 6.20 = max over cores of (mat + blend + drain) 6.20. Max (blend + drain)
is 4.93. **Window minus slowest blend core = 1.49 ms (traced) = the critical core's mat phase plus
drain.** The mat phase is nearly fixed. Over the 30 views mat max barely moves with blend max
(slope 0.03), and it matches the fitted 1.42 intercept. LEDGER §2 says mat "overlaps blend". In
FOLD=1 it does not: it runs before blend on every core.

## Attribution: other fixed intercepts

| item | measured (traced, median) | reading |
|---|---|---|
| pfwc window 1.55 (pfwc zone 1.36, end spread 0.20) | launch skew 0, pre 0.001, post 0 | The 1.06 intercept is not program overhead. It is the per-view streaming of all N = 5.56M Gaussians, which is constant across views, so the fit puts it in the intercept. Bounded by the existing pfwc stream-floor lever (~0.7, ledger) |
| sort window 1.49: prefix 0.061, fill 0.082, barrier/fillb 0.08, emit 1.23-1.32, end spread 0.21 | | The 0.33 intercept ≈ the pre-emit phases (~0.14) + end imbalance (~0.21) |
| program gaps | mat+blend→sort **0.224** (min 0.201); sort→pfwc 0.007; pfwc→mat+blend 0.007 | Device idle once per view at the view boundary (host gather/enqueue, t614 A ≈ 0.25) |

## U9: SFPU instructions per Gaussian per 32-pixel microblock (BLEND_SCHED=2)

`blend_pair_body<J,PM>` (48 noinline variants, jump-table dispatch) runs `sched_single_a2` once per
live microblock of a Gaussian:

- 2 x/y loads
- replay 0..12, the front (13: 5 sfpload, 2 sfpadd, 4 sfpmul, 2 sfpmad)
- 24 inline ops: 7 sfpload, 3 sfpswap, **4 sfpnop**, 2 sfpmad, exp core (2 exexp, 2 exman, shft, cast, setexp), 1 sfpmul
- replay 13..31, the tail (19: 4 sfpload, 4 sfpstore, 5 sfpmad, 3 sfpmul/add, setcc/encc/mov, setrwc)

That is **58 issue slots per Gaussian×microblock**:

| kind | slots |
|---|---|
| sfpload | 18 (31%) |
| sfpstore | 4 |
| sfpnop | 4 |
| sfpswap | 3 |
| math / cc | ~28 |
| setrwc | 1 |

The PM=3 body (both microblocks of a pair) issues the full 58 twice, so per-Gaussian parameters are
reloaded for every microblock (L0-L7 are all in use). At the ledger's ~214 core-cycles per pair this
is about 3.7 live microblocks per pair at 1 slot/cycle. MB stats would confirm that; it was not
measured here.

## Ranked levers (≥0.1 ms/view b2b; bounds traced, ×~0.89 untraced)

| rank | lever | bound | basis |
|---|---|---|---|
| 1 | **SHELVED (#645): bound is 0.298 traced / ~0.27 untraced, not 1.26; TRISC1 does band_batch cull during mat (1.077 mean / 1.237 crit, #595). See docs/mat-overlap-t645/STEP0.md and LEDGER §1c. Mover-work cut = LEDGER U19, unsized.** ~~Overlap mat with blend~~ (blend a core's first tiles as soon as their slabs are ready, and move the rest of mat into the idle NCRISC bulk-wait and BRISC output-wait time). Or cut the fixed mat work itself: sort_subchunk_mat takes 1.36 ms on the movers whatever P is | **1.26** (window 6.20 → max(blend+drain) 4.93); realistic 0.6-1.0 | bound630.py; ready flags already exist (r_ready 0.007) |
| 2 | Mat imbalance, if mat stays serial (critical 1.71 vs mean 1.45; per-view max 2.06) | 0.26-0.6 (part of 1) | zones630 |
| 3 | Blend SFPU trim: 4 nops + 3 swaps + per-microblock parameter reloads (18 loads) in the 58-slot body | 0.3-0.5 (7-12% of a ~4.4 ms SFPU span) | loops630 / disassembly. Needs live-microblocks-per-pair stats to firm up |
| 4 | Blend drain: last tile's scalar BRISC pack (65 µs/tile) and write on the critical core | 0.15-0.25 | drain 0.25 crit. #165 (emit pack on idle TRISCs, pre-xview) was +0.056 slower; this targets only the last-tile tail |
| 5 | View-boundary device gap mat+blend→sort | ≤0.22 | program gap; overlaps ledger "view boundary ≤0.27" |
| 6 | Blend end imbalance (end spread 0.23, max-mean 0.24) | ~0.2 | bz630 |
| 7 | Sort fixed part (pre-emit 0.14 + end spread 0.21) | 0.15-0.2 | zones630; overlaps U10 |
| — | pfwc 1.06 intercept | belongs to the pfwc stream-floor lever (~0.7) | no launch/pre/post overhead measured |

Below 0.01 ms (not levers): T init, CB setup, claim/meta/ready/DMA waits, launch skew, preamble and
postamble, the sort→pfwc and pfwc→mat+blend gaps. K2 (folded) is hidden behind the writer's output
wait in this profile.

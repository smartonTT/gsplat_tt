# t606: RECFAST + VIS_FAST bundle device A/B (structural plan 4, spec 1)

bh-33 p100, 2026-10-13 02:30-02:44 PDT, one build at ab9b4077 (bf425d81 + this driver), separate
TT_METAL_CACHE per arm off /home. Arms: A = defaults, B = `GSPLAT_TT_PFWC_RECFAST=1
GSPLAT_TT_PFWC_VIS_FAST=1`, C = `GSPLAT_TT_PFWC_VIS_FAST=1` (attribution, r5 only).
Driver `drive.sh` -> `remote_ab.sh`, outputs in `out/` (run logs gzipped).
`render/run.py --back-to-back`, 30 bicycle views, 20 timed passes after 1 warm-up, ms/view:

| round | order | A | B | C | B gain |
|---|---|---|---|---|---|
| r0 (warm-up round) | A B | 8.442 | 8.370 | | 0.072 |
| r1 | A B | 8.434 | 8.368 | | 0.066 |
| r2 | B A | 8.435 | 8.366 | | 0.069 |
| r3 | A B | 8.447 | 8.370 | | 0.077 |
| r4 | B A | 8.445 | 8.376 | | 0.069 |
| r5 (attribution) | C B A | 8.444 | 8.367 | 8.422 | 0.077 (C: 0.022) |
| **mean r1-r4** | | **8.440** | **8.370** | | **0.070 (0.83 %)** |

B wins every round, but the bundle gives the same 0.07 ms/view as RECFAST alone (t599: 0.071):
VIS_FAST alone is worth 0.022 and adds nothing measurable on top of RECFAST.

md5: A 077caaba = golden 30/30. B c19db0d9, **off golden**: views 27 and 29 differ
(`out/md5-r0-dump-*.txt`); b2b raw_md5 B and C both b0ca00be vs A 1d372b94, so VIS_FAST is the
cause (RECFAST alone was bit-exact in t599). Hero (view 0) is bit-exact: B device hero
(`opt/metal-screenshots/ttw-t606-bundle/`) 42.85 dB vs benchmarks/reference_v2/hero.png, golden
match, max 0 LSB; hero and diff x10 checked by eye: no tile seams, blocks or stripes, diff only on
spoke/frame/leaf edges as in best-iter-227.

Verdict: **shelved** on both kill rules (gain 0.070 < 0.1 ms/view gate; md5 off golden). The VIS_FAST
predicate was not made exact: an exact predicate can only keep or lose VIS_FAST's 0.0-0.02 ms, so the
bundle cannot reach the gate either way. Both flags stay default 0; no golden refresh proposed.

## pfwc window profile (plan-4 spec 1 step 3, review #607, task t608)

The gain (0.070) is between 0.05 and 0.1, so step 3 runs before shelving. Device profiler
(t595 "tracy" shape, `GSPLAT_TT_B4A_FOLD=0` so pfwc, k2, sort and mat+blend are separate programs
that fit kcfg), bh-33 p100, one build at 68e841f1, one JIT cache per arm off /home, 2026-10-13
03:00-03:31 PDT (rec views 25:30 at 09:29 PDT, after an ssh drop). Arms: base = A (defaults),
rec = RECFAST only, vis = C (VIS_FAST only), new = B (bundle). One capture per arm per 5-view chunk; each capture renders 11 views and the last
5 (the b2b pass) are analysed. Views 0:5 also with `GSPLAT_TT_PFWC_STEPCYC=1` (mover and TRISC
splits) for base and new. Driver `profile/drive_prof.sh` (CHUNKS mode for views 5:30) ->
`profile/remote_prof.sh`; per-capture summaries by `profile/pfwc608.py` (per RISC) and
`profile/progwin608.py` (per program); `profile/agg608.py` joins them (`profile/agg608.txt`).
Views 0:5 in `profile/out/` (raw device csvs included), 5:30 in `profile/out30/` (summaries only;
the 20 raw csvs, 45 MB, are kept outside git in `tt-project/state/archive/worktrees/t608/out30/`).

Slowest core of the pfwc program, mean over the 30 views (ms; busy = KERNEL zone length):

| arm | pfwc window | gain vs A | b2b gain (A/B) | TRISC_0 | TRISC_1 | TRISC_2 | NCRISC | BRISC | sets the window |
|---|---|---|---|---|---|---|---|---|---|
| A base | 1.527 | | | 1.446 | 1.454 | 1.442 | **1.524** | 1.469 | NCRISC 29/30, BRISC 1 |
| rec (RECFAST) | 1.466 | 0.061 | 0.071 (t599) | 1.403 | 1.411 | 1.399 | **1.462** | 1.418 | NCRISC 28/30, BRISC 2 |
| C vis (VIS_FAST) | 1.504 | 0.023 | 0.022 | 1.422 | 1.428 | 1.418 | **1.503** | 1.446 | NCRISC 30/30 |
| B new (bundle) | 1.456 | **0.072** | **0.070** | 1.397 | 1.404 | 1.393 | **1.444** | 1.417 | NCRISC 23/30, BRISC 7 |

Across all 110 cores the last kernel to end is always a mover (NCRISC 36-37 %, BRISC 63-64 %),
never a TRISC; on the slowest core the TRISCs end 0.05-0.08 ms before NCRISC in every arm.

**The A/B was right.** The pfwc window gain matches the b2b gain for every arm (bundle 0.072 vs
0.070, VIS_FAST 0.023 vs 0.022, RECFAST 0.061 vs 0.071 from t599, another session and build), and the other three programs do not move
(30-view means k2 0.640-0.643, sort 1.435-1.443, mat+blend 5.179-5.188 ms, `profile/agg608.txt`). So the whole b2b effect is the pfwc window, and the
bundle is worth 0.07 ms/view. The shelve verdict stands.

**Why VIS_FAST adds ~0 on top of RECFAST (plan-4 predicted 0.12-0.26 for the pair).** The model
assumed that after RECFAST pfwc becomes TRISC-bound, so a VIS_FAST TRISC cut of dT >= ~0.05 would
show up in full. The profile says otherwise:

1. The movers, not the TRISCs, set the window in every arm, also after RECFAST (rec: NCRISC on the
   slowest core in 28 of 30 views). The TRISC walls are not free compute: they move with a mover-only
   change (RECFAST alone lowers the slowest core's TRISC_1 from 1.454 to 1.411 with no TRISC code
   change), so they include output backpressure from the movers. The per-core pipeline is paced by
   the movers, with the TRISCs ending one tile's worth of mover work (~0.06 ms) ahead.
2. RECFAST's record-loop cut is mostly absorbed by other mover parts. STEPCYC, slowest core, views
   0:5, base -> bundle: NCRISC `rec` 0.710 -> 0.517 (-0.19), but `wait` (color reads and TRISC
   tiles) 0.315 -> 0.367, `opn` 0.024 -> 0.065 and `rd` (reader polls, inside the other parts)
   0.271 -> 0.339; NCRISC wall only 1.461 -> 1.381 (-0.08). BRISC `fl` (record/page flushes)
   doubles, 0.083 -> 0.187, and its survivors per launch rise 7382 -> 7793 while NCRISC's drop
   7358 -> 6936, so with the bundle BRISC becomes the setter in 7 of 30 views (0 in vis, 2 in
   rec). Once the record loop is fast, the other mover's flushes and the shared reads set the
   window, not the TRISCs.
3. VIS_FAST's TRISC cut is smaller than the model's dT >= 0.1: STEPCYC TRISC_1 `vis+pops`
   0.569 -> 0.509 (-0.06) on the slowest core (views 0:5, bundle), and over 30 views VIS_FAST alone
   lowers the slowest core's TRISC_1 wall by only 0.026. It reaches the window only through the
   movers' `wait` on TRISC tiles: VIS_FAST alone gains 0.05 on views 0:5 and 25:30 and -0.01 to
   0.02 on the other chunks. On top of RECFAST the window gains 0.011 (1.466 -> 1.456; the rec
   25:30 chunk, captured 6 h later, gains only 0.022 vs ~0.08 on the other chunks, so 0.011 is an
   upper bound), and in chunks 5:15 the bundle is slower than RECFAST alone while BRISC sets
   the window.

So the plan's two-bottleneck story (each switch capped by the bottleneck the other removes) does
not hold at best-iter-227: both switches act on the same mover-paced pipeline and their gains do
not add. A TRISC lever is worth little until the movers' non-record parts (`wait`, `rd`, `opn`,
`fl`) shrink; model603's calibration point (RECFAST = 0.071 with a TRISC-bound remainder) should be
replaced by this profile before spec 2 (A' survivor compaction) is sized.

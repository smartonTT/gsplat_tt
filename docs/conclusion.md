# Conclusion: Gaussian splatting on Tenstorrent, bicycle reference (stop at iter 207)

Written by task #335 on 2026-10-07. No device was used for this document; every number names its
source. GPU numbers are **published, not measured** (there is no GPU box in this project).

## Result

| | ms/view | FPS | source |
|---|---:|---:|---|
| Project start, 30-view bicycle 1024x1024 baseline | 173.30 ± 0.26 | 5.8 | `docs/benchmark-baseline.md`, commit a3a11c6e (p100a) |
| **iter 207 (final, tag `best-iter-207`)** | **10.907** | **91.7** | `opt/ttw/iters.jsonl` iter 207, yyzo-bh-04 p100a, 3 rounds 10.923 / 10.861 / 10.937 |
| iter 207 on **p150b** (bh-30), diagnostic, not an iteration | 12.906 | 77.5 | `docs/p150-bench-bh30.md` (#346), 2 rounds 12.934 / 12.878, md5 906e0435 30/30 |
| GPU G1: INRIA 3DGS, RTX A6000, bicycle 1080p | 10.75 | 93.0 | published, not measured (REPORT.html) |
| GPU G2: G1 scaled to 1024² pixels | 5.44 | 184 | published G1, pixel-normalized; not measured |

- iter 207 is **15.9x faster** than the project's starting baseline and **1.5 % slower than G1**
  (0.16 ms/view). It is about 2x slower than G1 normalized to the same pixel count (G2).
- Quality at iter 207: md5 906e0435 on 30/30 views, hero 42.51 dB against
  `benchmarks/reference_v2/hero.png`, device screenshot checked by eye with no tile seams
  (#315, #319). The 8-bit golden match is a separate badge.
- **p150 (added by #346):** iter 207 on bh-30 (p150b, the user's viewer box, the only p150 we could
  get) runs at **12.906 ms/view (77.5 FPS)**: 1.20x G1 (10.75 ms, published, not measured) and
  1.18x the p100a's 10.907 ms. Same image (md5 906e0435 30/30, hero 42.51 dB, no seams). Blend is
  0.44 ms faster on the p150 but sort `bin_emit` is 1.85 ms slower and host stages are slower; the
  boxes also differ in host CPU (EPYC 7352 vs Ryzen 7600X), firmware and tt-metal build.
  See `docs/p150-bench-bh30.md`.
- All other numbers are on a **p100a** (Blackhole, yyzo-bh-07 then yyzo-bh-04). Absolute ms/view
  differ a little between boxes (iter 206 tip: 11.129 on bh-07, re-baselined 11.093 on bh-04), so
  every A/B ran both arms on the same box.
- Defaults at iter 207: one-launch sort v2, PRECULL=2, BLEND_SCHED=2, decode-ahead, PFWC writer
  split + NoC balance, PFWC_RECIP_NEWTON, MATBLEND_FUSE, MATCULL_TRISC_FILL, contrib floor 1/255
  with the per-pixel floor.

## Why stop here

The last open lever was recovering the fill-path shortfall of #306. #319 measured it with Tracy
fill zones (`docs/fill-zones-t319.md`): TRISC0 is only 52 % busy, the movers wait on whole-slab
fill jobs (NCRISC 0.355 ms mean), and removing every wait caps the mat gain at 0.30 ms.

#335 modelled the two fixes from that data without a device (`docs/fill-model-t335/`):

| variant | mat gain (model) | frame (x0.40 / x0.28) |
|---|---:|---:|
| finer 512-record fill jobs, per-mover queues, streamed emit | 0.00-0.27 | ≤ 0.11 / 0.08 |
| big-tile item split over both movers of its core + host re-LPT | 0.35-0.40 | 0.14-0.16 / 0.10-0.11 |
| **both, realistic** | **0.65** | **0.26 / 0.18** |
| both, ideal split | 0.73 | 0.29 / 0.20 |
| ceiling: all mover work perfectly balanced, zero waits | 0.82 | 0.33 / 0.23 |

The conversion factors are #306's: 0.216 ms measured in mat+blend for 0.56 ms modelled (0.40),
and 0.158 ms frame (0.28). The realistic combined gain is under the 0.30 ms gate, and even the
perfect-balance ceiling barely passes it at the optimistic conversion. The build would need three
risky changes at once (new job protocol, a rank-ordered big-tile gather, a two-mover radix sort in
a nearly full L1), and the closest measured relative, OL_MAT_SELECT big-tile parts (#121), was
0.15 ms slower. Every other lever is below the charter's ~1 % line or already measured slower
(table below). So the work stops at iter 207.

## How the time came down

Measured ms/view on the 30-view bicycle bench (`opt/ttw/iters.jsonl`, untraced, p100a).

| phase | iters | ms/view | what changed |
|---|---|---|---|
| Before this project (Cursor era) | 0-141 | 173 at the 2026-09-30 baseline | on-device scan bases, resident buffers, L1 subchunk sort, fused cull+blend handoff, microblock cull, PACK2 records; a persistent-kernel rewrite was gated off (#139-141) |
| Sort emit and data-mover dataflow | 142-149 | 166.6 → 145.4 | split sort stage timers (#18), blendrec prefetch (#20), emit fp32 math moved off NCRISC (no FPU), dual data mover for the emit (#22) |
| Soft-float removal | 151-155 | 121.7 → 107.8 | libgcc soft-float audit of BRISC/NCRISC (#30, #27) and the TRISC cull/blend (#39) |
| Cull and tile assignment | 156-164 | 103.2 → 61.8 | thin-Gaussian microblock keep test (#41), split tile_assign K1/K2 (#34), adaptive stable radix (#26), dual-mover gather_visible (#33), band-extent SFPU microblock cull (#59) |
| Sort tail, blend tail, on-device image | 166-175 | 58.5 → 41.4 | dual-mover sort tail (#56), dynamic blend tile claim (#60), on-device 8-bit RGB pack (#61), blend coefficients in DEST (#68), TRISC1 floor cut (#80), 2 KB record pages (#90) |
| Visibility and the one-launch sort | 176-178 | 32.6 → 24.5 | visibility predicate on device (#102), batched emit rewrite (#100), one-launch sort v2 (#121) |
| pfwc, pre-cull, blend diet | 179-190 | 21.3 → 16.2 | pfwc writer (#122), TRISC1 instruction diet (#146), dead-record pre-cull PRECULL=2 (#142, #161, #174), K2 pair-stage diet (#170), blend late claim (#188), K2-fold window fill (#181) |
| Emit balance, 2-CQ bridge, TRISC emit pack | 191-195 | 15.7 → 13.6 | emit mover balance (#196), 2-CQ bridge hiding (#198), tile-owned TRISC emit pack (#202), pfwc writer split (#221) |
| Blend schedule, decode-ahead, pfwc NoCs | 196-199 | 12.7 → 11.6 | BLEND_SCHED=2 (#229), decode-ahead on TRISC1 (#231), pfwc NoC balance (#232) |
| Fusion and TRISC fill | 205-207 | 11.66 → 10.91 | Newton reciprocal in pfwc (#266), fused mat+blend program (#273/#293, −0.52), TRISC fill of the mat cull (#306/#315, −0.18) |

The project's thrust held throughout: keep data in L1, move less over the NoC and through the
host, fuse programs, and put arithmetic on the TRISCs (SFPU/FPU) instead of the data-mover RISCs.
The two largest single steps were removing soft-float from hot loops (−38 ms combined) and the
band-extent SFPU microblock cull (−22 ms). Late gains came from overlapping stages (2-CQ bridge,
fused mat+blend) and from moving work off the movers onto idle TRISC threads.

## Where the 10.9 ms goes (Tracy, iter 207 defaults, #319 `opt/profiler/t319-fz/zones.txt`)

Per-view makespans of the device stages (traced, so slightly inflated): blend 3.86 ms, mat
(sort-subchunk materialize + TRISC cull) 2.68, pfwc 1.93, sort emit 1.28, sort town 1.18, K2
pairs 0.92. The host bridge adds d2h 0.21-0.32 and publish 0.15-0.17. Blend and pfwc dominate,
and every blend lever measured so far is either shipped or shelved.

## Shelved levers and why

| lever | expected or measured | why shelved | source |
|---|---|---|---|
| Finer fill jobs + big-tile split (t319 (b)) | 0.18-0.26 ms frame (model) | under the 0.3 ms gate; three risky changes | `docs/fill-model-t335/` |
| Lever A: blend lane during mat on idle TRISCs | 0-0.4 ms | TRISC0 now fills (52 % busy); the idle is spread over ~10 waits per view (min 0.88 ms), too fragmented; DST spill per yield | `docs/checkpoint-t311.md` (ttp/t311-stop-or-reframe), #319 |
| Fill rebalance across TRISC0/TRISC2/mover | — | TRISC0 is not the bound (52 % busy, not ≥ 85 %) | #319 |
| Radix sort on TRISC1 | unknown | needs TRISC1 idle ≥ 2 ms; measured 1.10 ms | #319 rule (d) |
| Small-first mat order | +0.011 ms (slower) | does not change any slot's total work | #303 |
| Lever B: fold K2 into sort_ol | +0.19 ms (slower) | measured slower | #298 |
| C: dynamic chunk claim for pfwc/emit/town tails | < 0.3 ms | tails 0.08-0.30 ms each | t297 |
| Host serial path (d2h, publish, pfwc dispatch) | each < 0.3 ms | below the gate each | t297, #171 |
| FPU quadratic-form blend | ≤ 0.7 ms ceiling | not md5-safe; blend has shrunk from 10.6 to 3.7 ms since the estimate | #111 |
| pfwc cov2d on SFPU | +0.256 ms (slower) | measured slower | iter 197, #228 |
| OL_MAT_SELECT big-tile parts across cores | +0.15 ms (slower) | measured slower | #121 |
| 16 B records (R16) | +1.6 ms (slower) | record size is not the emit's cost | iter 150, #23 |
| Closing inter-program idle by merging launches | −0.1 ms, flat frame | no frame gain then; later recovered by the 2-CQ bridge and fusion | iter 158, #31 |
| contrib floor 1/1024 (accuracy mode) | +0.98 ms, +4.7 dB | a quality choice for the user; available as opt-in `GSPLAT_TT_CONTRIB_FLOOR_INV=1024` | `docs/floor-ab-t260.md` |
| contrib floor 1/255 with microblock cull (no pixel floor) | −6.6 ms then | visible artifacts on thin spokes | iter 152, #28 |

## Open items that are not speed levers

- **#284 big-tile split for far views** (`docs/tile-split-t284/README.md` on ttp/t284 @ da0464ba):
  tiles over 32768 records render at full 1/255 instead of a coarser floor; far poses go from
  29.35 to 43.57 dB at +0.01 ms and the default md5 is unchanged. Review and landing were queued.
- **p150 sort gap.** The p150 run (#346) is slower than the p100a only in sort `bin_emit` and the
  host stages. Whether that is the host CPU or the chip is not known yet; a per-view stage or
  Tracy run on the p150 would tell.
- **G2 gap.** At equal pixel count the published GPU number is about 2x faster. Nothing on the
  lever list closes that; it would take a different blend formulation (for example FPU matmul
  blending that is not md5-identical), which is a new project, not a tuning step.

## Reproduce

Bench and reservation steps: `docs/benchmark-baseline.md`. Iteration log and dashboard:
`opt/ttw/iters.jsonl` → `opt/ttw/REPORT.html`. iter 207 run: `docs/iter207-t315/`.
Fill zones: `docs/fill-zones-t319.md`, `opt/profiler/t319-fz/`. Final model:
`python3 docs/fill-model-t335/model.py`.

# t188: blend late claim + record-count-descending claim order

Build spec: `docs/blend-tail-t183/README.md`. Two switches, both default on:

- `GSPLAT_TT_BLEND_LATE_CLAIM` (reader `reader_alpha_blend_mb_devcull.cpp`): the
  reader reserves a bulk ring slot before it takes the next tile from the shared
  counter, so a core has at most 2 tiles committed instead of 3.
- `GSPLAT_TT_BLEND_CLAIM_DESC` (host, `render/host/blend_claim_order.h`): the
  per-core lists are a round-robin deal of tiles in record-count-descending order
  instead of the LPT lists.

## A/B (yyzo-bh-07, Blackhole p100a, bicycle 30 views 1024x1024, untraced)

Arms: off = LPT + early claim, late = LPT + late claim, base = late + desc (default).
`view_total` ms/view from the `STAGES` line (`out/run-r*-*.log`), arm order rotated per round.

| round | off | late | base | late - off | base - off |
|---|---:|---:|---:|---:|---:|
| r1 | 17.449 | 17.047 | 16.864 | -0.402 | -0.585 |
| r2 | 17.312 | 17.068 | 16.991 | -0.244 | -0.321 |
| r3 | 17.277 | 17.048 | 17.039 | -0.229 | -0.238 |
| mean | 17.346 | 17.054 | 16.965 | **-0.292** | **-0.381** |

Blend stage (device wait for the blend program): off 8.884, late 8.610, base 8.544
ms/view; base - off is -0.346 / -0.338 / -0.336 per round, so the blend saving is
steady and the round-to-round spread is in the sort stage (off r1 sort 4.098).

All 9 runs: 30 views md5-identical to `md5-r82new.txt` (46a725ab), `out/md5-r*.txt`.

## Tracy (default arm, `out/tracy-on-*.txt`)

Blend TRISC core end, max - mean: mean **0.106 ms** (median 0.074, max 0.249),
down from 0.439 ms in t183. Blend program span 5.54 ms/view
(`out/tracy-on-gaps.txt`).

Decision: keep (-0.381 ms/view mean, over the 0.3 gate). Most of it is the late
claim (-0.29); the descending order adds -0.09.

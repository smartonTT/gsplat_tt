# Zero-copy pinned output (task #374)

`GSPLAT_TT_OUT_ZEROCOPY=1` (implies `GSPLAT_TT_OUT_PINNED=1`, default off): the
blend writer writes the u8 image into a ring of pinned, NoC-mapped host buffers
(`render/host/out_ring.h`) and `render_view` returns a numpy view of that buffer
instead of copying it out. The view holds the slot's lease, so the ring never
writes a buffer the caller still holds. Lifetime: a returned image stays valid as
long as the caller keeps it (and at least until the frame after next). With the
flag off the path is unchanged (md5 906e0435 on 30/30 views).

## A/B on yyzo-bh-04 (p100a, Ryzen 5 7600X), build 8fd193fd, bicycle, 30 views

One build, 3 alternating rounds (`drive.sh`, outputs in `out/`).

| arm | r1 | r2 | r3 | mean ms/view | d2h ms | blend span ms |
|---|---|---|---|---|---|---|
| base | 10.862 | 10.913 | 10.887 | 10.887 | 0.21-0.23 | 7.05-7.08 |
| pinned (copy) | 10.909 | 10.901 | 11.033 | 10.948 | 0.25-0.38 | 7.05-7.13 |
| zero-copy | 11.290 | 11.283 | 11.293 | 11.289 | 0.000 | 7.64-7.72 |

md5 matched golden 906e0435 on 30/30 views in all 10 runs. Zero-copy removes the
copy but the blend span grows ~0.6 ms, so the frame is ~0.40 ms (3.7%) slower.
Pinned copy mode is no faster than the base device read on this board.

Device hero with zero-copy on: `opt/metal-screenshots/t374-zerocopy-out/hero.png`,
42.51 dB vs `benchmarks/reference_v2/hero.png`, golden md5 match. hero.png and
hero_diff10.png were viewed: no tile seams.

## Why zero-copy is slower (diag rounds, `diag.sh`, `diag2.sh`, outputs in `diag/`)

Rounds 11-12 (build a963af6b): every arm allocates its pinned slots once (the
`OUT_RING new pinned slot` log stops after the first frames; `replaced=0`), so
re-pinning is not the cost. Copy mode rotating the writer through 3 pinned
buffers (`GSPLAT_TT_OUT_PIN_HOLD=2`) keeps the blend span at 7.0 ms but its copy
gets slower (d2h 0.37-0.58 ms); zero-copy (4 slots) has blend 7.64-7.73 ms.

| arm | r11 frame / blend / d2h | r12 frame / blend / d2h |
|---|---|---|
| pin | 10.83 / 7.135 / 0.208 | 10.91 / 7.057 / 0.281 |
| pinhold=2 (3 bufs) | 11.01 / 7.006 / 0.365 | 11.21 / 7.056 / 0.580 |
| zc (4 slots) | 11.25 / 7.675 / 0 | 11.26 / 7.639 / 0 |

DIAG2_PLACEHOLDER

## Recommendation

Keep both flags off by default. The whole lever is the ~0.22 ms base readback
(2%), and on this board neither pinned variant beats it.

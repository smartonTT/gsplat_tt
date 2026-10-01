# Task #100: sort_bucket_emit rewrite

Branch base: `smarton/tt-project-opt` @ 72cd487 (iter-176, 32.61 ms/view). A/B base = c4dc7e2 with
all knobs at legacy (32.84 here; same code path as 72cd487). Board: yyzo-bh-07
(Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024, untraced.

## What changed (render/kernels/dataflow/sort_bin.cpp, c4dc7e2)

Task #98 showed the emit is bound by DRAM traffic plus pack work. The rewrite
covers all five spec items, each behind a runtime knob (render/host/env_config.h):

| Knob | Default | What it does |
|---|---|---|
| `GSPLAT_TT_EMIT_PB` | 8 | Pair pages and their blendrec reads are fetched N pages per barrier, double-buffered (spec 3 and 4). 1 = legacy, one page per barrier. |
| `GSPLAT_TT_EMIT_RING` | 8 | Records are staged in per-tile L1 runs and written as one coalesced run, not as 32 B writes (spec 1). Bucket bytes are unchanged. 0 = legacy. |
| `GSPLAT_TT_EMIT_PUBOC` | 1 | The gather (gather_vis_scatter / gather_visible_scatter) publishes the packed op/color words and the depth key into blendrec[10..12]. The emit copies them, so the 16 B packoc writes, the depth page reads and most of pack_invariants are gone (spec 2). 0 = legacy. |

`pack_rec` is slimmed on every path (spec 5). Run counters that are read right
after they are written stay in RISC-local memory.

Hardening from review #103 (37eced0): the non-piped sort_subchunk_mat falls back
to `device_state::get_bucket_cull_params`; a static_assert ties
`GATHER_PART_RECS` to `kMatMover0Cap`; `GSPLAT_TT_MATCULL_DEPTH` is validated so
the coefficient CBs fit L1; the unused `t_mat0` is removed.

## Interleaved A/B (3 rounds, step order rotated; ab_driver.sh, remote_job.sh)

Means over 3 rounds, ms/view. All 21 runs md5-identical to md5-r82new.txt,
hero_vs_ref 100 dB.

| Step | Knobs | frame | vs base | project | sort | bin_emit |
|---|---|---|---|---|---|---|
| base | legacy | 32.84 | — | 6.03 | 14.38 | 8.94 |
| pb | PB=8 | 32.32 | -0.52 | 6.02 | 13.89 | 8.51 |
| ring | RING=8 | 32.18 | -0.66 | 6.03 | 13.74 | 8.33 |
| puboc | PUBOC=1 | 31.07 | -1.77 | 6.47 | 12.26 | 6.88 |
| pb+ring | PB=8 RING=8 | 31.50 | -1.34 | 6.02 | 13.09 | 7.69 |
| **all** | PB=8 RING=8 PUBOC=1 | **29.59** | **-3.25 (-9.9%)** | 6.46 | 10.68 | 5.29 |
| all4 | PB=4 RING=4 PUBOC=1 | 29.64 | -3.20 | 6.45 | 10.83 | 5.43 |

- The steps compound: alone they sum to -2.95 ms, together -3.25 ms. Once
  PUBOC removes the pack work, traffic is the limit again, so PB and RING are
  worth more.
- PUBOC moves ~0.43 ms of packing into the gather (project 6.03 -> 6.46), and
  the emit saves 2.06 ms.
- bin_emit goes from 8.94 to 5.29 ms (-3.65). The t98 traffic+pack floor was
  2.58 ms, so ~2.7 ms is still above that floor.
- Log: ab.log (this directory).

## Default-on confirm (31838f2)

Built with the new defaults and no env knobs (confirm.log): 29.64 and 29.53
ms/view, bin_emit 5.29 ms. Legacy kill switch (PB=1 RING=0 PUBOC=0): 32.80
ms/view. All three runs md5-identical over 30 views, hero_vs_ref 100 dB.

Net: 32.84 -> 29.59 ms/view (-3.25 ms, -9.9%), 33.8 FPS. Published GPU
reference: 10.75 ms/view (published, not measured).

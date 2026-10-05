# t270: a tile over the sort bucket no longer fails the frame

## Problem
`sort_onelaunch` holds at most 32768 records per tile (`kTileCap`). A tile above
that made `render_clean` throw ("device sort failed"), so the whole frame was lost
(#257 viewer log, tile 370 = 34672 records at floor 1/16384).

## Repro (host tile-count model, `tilecount.py`; CSVs here)
- 30 bench views (`bench.csv`): max 22165 / 23493 / 25407 records at floors
  1/255 / 1/1024 / 1/16384 (hero, tile 584). No overflow.
- Close / zoomed hero (`close.csv`): max 15840. Close views spread splats over
  more tiles and do not overflow.
- Pulled-back / wide hero (`far.csv`): dolly back 2 units hits 34059 / 35747 /
  38345 (tile 554); back 4 units 48838-54090; fov 90 52494-58505 (18-24 tiles over).

## Design
`render_view` catches the overflow and re-renders the whole view at a coarser
contrib floor (`render/host/overflow_retry.h`: next floor from a log model of
records vs floor, at least 2x per step, at most 8 retries, capped at 0.5). The
floor applies to every pair in the frame, so the result is uniform with no tile
seams. Views that fit never take this path, so the default kernels and output
are unchanged. Chosen over splitting a tile's records across two sort passes
because that needs sort and blend kernel changes for a case the bench never hits.
`GSPLAT_TT_TEST_TILE_CAP=N` lowers the host check to force the path in tests.

## Device results (yyzo-bh-07 p100a, `remote_t270.sh`, log in run 737)
| run | ms/view | md5 (30 views) |
|---|---|---|
| base d8d3591 (4 runs, ABBA) | 11.639 11.635 11.643 11.646 = 11.641 | 46a725ab |
| t270 6dffd07 (4 runs, ABBA) | 11.648 11.676 11.641 11.601 = 11.642 | 46a725ab |

Default cost +0.001 ms/view (gate +0.05). Hero byte-identical to base.

Overflow views:
- forced cap 20000 on the 30 bench views: all render, 16.7 ms/view mean (retries).
- far pose (back 2) at 1/255: base segfaults after "device sort failed"; t270
  retries 1/255 -> 1/96 -> 1/48 -> 1/24 and renders (28.4 ms).
- far pose at 1/16384: 5 retries to 1/26 (39.1 ms). Far pose at 1/64: 1 retry to 1/32 (18.9 ms).

## Images (`img/`, checked by eye: no tile seams or block artifacts)
- `hero.png` vs `benchmarks/reference_v2/hero.png`: 41.16 dB (`hero_diff10.png`).
- `far2_255_retried.png` vs CPU reference `far2_cpu_ref255.png` (cpu_cpp_mb, same
  pose, floor 1/255): 29.35 dB (`far2_255_diff10.png`). 1/16384 retried: 29.72 dB;
  1/64 retried to 1/32: 30.89 dB.

## Limits
The fallback is a quality step: a coarse floor trims every splat's tail, so
thin spokes and window edges lose contrast (about 29-31 dB vs the CPU reference
on the far pose). It also costs one extra render per retry. Record counts fall
slowly with the floor (35840 at 1/255, 33120 at 1/48), so the ladder must go far.
A tile split (second bucket pass for the overflowing tile) would keep full quality.

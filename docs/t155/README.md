# t155 — host bridge before the one-launch sort

Board: yyzo-bh-07 p100a (p150 IRD). Tree e41c668 (on top of 220ba33). Bicycle, 30 views.

## Change
`tile_assign_fused_k2` queues the one-launch sort right behind the fused K2
(non-blocking `ReadShard` of proj_M + host event, early enqueue, `EventSynchronize`).
The sort kernel reads P / P_pad from `ta_pairs_P` on device and computes its own page
split (`sort_ol::mover_pages`, unit-tested equal to the host split).
Kill switch `GSPLAT_TT_SORT_OL_EARLY=0`; `GSPLAT_TT_SORT_OL_EARLY_LOG=1` prints per-view
early-enqueue timing and gate inputs.

Bugs found on device and fixed: `EnqueueReadMeshBuffer(..., blocking=false)` TT_FATALs
(use `ReadShard`); the early gate used the K2's gaussian-tile count (5989) instead of the
image tile count (tiles_x*tiles_y = 1024), so it never fired at first.

## Results (yyzo-bh-07 p100a)
| | early on | early off |
|---|---|---|
| round 1 view_total (ms/view) | 18.31 | 18.45 |
| round 2 view_total (ms/view) | 18.51 | 18.44 |
| md5 vs md5-r82new.txt | 30/30 identical | 30/30 identical |

Paired gain: mean 0.03 ms/view (round 1: 0.14, round 2: -0.07). Sort host stage
4.19-4.23 vs 4.30-4.32 ms. Below the 0.3 ms acceptance bar.

Tracy (device, 30 views): gap before sort_ol_count 0.507 -> 0.011 ms; device span
18.489 -> 17.942 ms; total all-core idle 0.828 -> 0.326 ms/view (0.303 ms left before
materialize).

## Conclusion
The early enqueue does close the gap on device, but the untraced wall barely moves.
The 0.53 ms Tracy gap was mostly profiler overhead on host work: untraced, the bridge
host work (gather_result ~0.1, sort layout ~0.1, tile_assign ~0.005 ms; from
GSPLAT_TT_HOST_PROFILE VIEW_STAGES) is about 0.2 ms, and the measured gain is within
noise of that. Lesson: Tracy all-core idle between programs overstates host gaps; confirm
with an untraced paired A/B before chasing them. Not landed.

Files: out/run-r{1,2}-{base,off}.log, out/md5-*.txt, out/gaps.txt, out/roofline.txt, out/zones.txt.

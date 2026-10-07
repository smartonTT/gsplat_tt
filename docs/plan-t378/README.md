# Plan after iter-210 (task #378, 2026-10-07)

Target: the bicycle render on the p150 (bh-30) goes from 11.071 ms/view (iter-210, 6640239b,
best-iter-210) to below 10.75 ms/view. 10.75 is a published GPU number, not measured here.
This plan uses existing data only. No device was used.

## Where the time goes (p150, bh-30)

Untraced stage means for iter-210 (review #369 P1 logs, 3 rounds x 30 views, ms/view):

| stage | bh-30 p150 | p100a |
|---|---:|---:|
| wall | 11.071 | 10.935 |
| project (gather_wait = pfwc + K2) | 2.989 (2.794) | 3.034 (2.945) |
| sort (bin_emit 0.406, publish_host 0.319, sort_mat 0.106) | 1.031 | 0.50 |
| blend (mat + blend) | 6.512 | 7.082 |
| d2h | 0.458 | 0.27 |

The newest p150 Tracy capture is from iter-209 (docs/p150-blend-gap/out/timeline-bh30-209.txt).
There is none for iter-210. iter-209 device windows: pfwc 1.83, k2_pairs 0.81, emit 2.79
(before the stride fix), mat 2.83, blend tail about 3.2. The fused mat + blend runs at 98.5%
efficiency: per-core busy 2.29 + 3.68 = 5.97 ms over a 6.06 ms window on 110 cores. So it scales
with the core count.

Already in flight, not part of this plan:
- #373: pinned-output A/B on the p150.
- #374: zero-copy pinned output (up to about 0.37 ms on the p150).
- #370: p150 host path.
- Cross-view overlap: about 0.8-0.9 ms/view of host time between views.

## Proposals

| # | task | expected gain (p150) | device |
|---|---|---|---|
| 1 | iter-210 Tracy capture on bh-30 | 0 (sizes the other levers) | yes, exclusive 'viewer' |
| 2 | Ethernet dispatch, 12x10 = 120-core grid on the p150 | 0.4-0.7 ms | yes, exclusive 'viewer' |
| 3 | Emit K2 pairs from the pfwc writers (drop the K2 launch) | 0.3-0.5 ms (model first) | model, then device |
| 4 | Model: sort emit straight into the owner core's L1 | 0.3-0.7 ms if it fits L1 | no (model only) |

### 2. Ethernet dispatch (new, never tried)

bh-30 has 12 Tensix columns after harvesting (mask 0xc0) and 12 live Ethernet cores. tt-metal's
default Blackhole dispatch takes a whole Tensix column, so the grid is 11x10 = 110 cores.
tt-metal ships `core_descriptors/blackhole_140_arch_eth_dispatch.yaml`, which moves dispatch onto
Ethernet cores and keeps all Tensix columns. On bh-30 that gives 12x10 = 120 cores (+9.1%).
Mat + blend scale almost perfectly with core count: 5.97 x (1 - 110/120) = about 0.50 ms.
pfwc is TRISC-bound and adds about 0.15 ms; K2 adds about 0.07 ms.
The p100a has no Ethernet cores, so it keeps worker dispatch.
No project doc mentions Ethernet dispatch.
Risks:
- bh-30's tt-metal build may not support Blackhole Ethernet dispatch with 2 command queues.
- Per-column tables (the emit mover speed table, the pfwc NoC balance) assume 11 columns.

### 3. K2 pairs from the pfwc writers

K2 exists as its own launch because the pair segment table needs every core's count.
Lever B (#125) fused the compaction into the pfwc writer but kept K2 separate.
The only K2 fold tried so far went into sort_ol (#298) and was 0.19 ms slower.
Idea: each pfwc writer writes its pairs into its own fixed-capacity segment. Sort's prefix stage
reads the per-core counts, so the K2 launch, its 0.81 ms window and the about 0.07 ms gap before
it disappear.
The pfwc writers have headroom: per #226 they ran at 1.12-1.19 ms against a TRISC floor of
about 1.73 ms.
Risks:
- The extra writer work may push the writers above the TRISC floor.
- md5 may break if pair order changes the order of tied sort keys.
Model this before writing any kernel code.

### 4. Emit into owner L1 (model only)

The sort emit is a DRAM scatter into the tile bucket, and mat then reads the bucket back.
Writing records straight into the L1 of the core that owns the tile removes one DRAM round trip.
L1 limits:
- Records are 32 B.
- The largest tile holds about 23k records, which is about 735 KB.
- Owners hold about 840 KB of records on average.
- Blend and mat already use L1.
So this proposal is a capacity and gain model only. It goes to implementation only if the model
shows at least 0.3 ms and the data fits.

## If the levers fall short

- If #373, #374 and the cross-view overlap together bring bh-30 below 10.75 ms, proposals 2-4 add
  margin.
- If proposal 2 does not work on bh-30's tt-metal and models 3 and 4 come in below 0.11 ms, no
  lever is left worth 1%. Then update docs/conclusion.md and stop.

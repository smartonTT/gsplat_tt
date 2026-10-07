# t366: the p150 "blend gap" (bh-30 ~8.0 ms vs p100a ~7.0 ms) is not in blend

Question: why is the per-view `blend` stage ~8.0 ms on the p150 (bh-30) but ~7.0 ms on the p100a
(yyzo-bh-04) at the same code (best-iter-209)?

Answer: the blend (and mat) device work takes the same time on both boards, to within 0.01 ms.
The host `blend` stage is a wait: it runs from the enqueue of the mat+blend program to `Finish`.
Since iter-209 the host enqueues that program while the device is still in the sort emit, so the
stage also counts the rest of the emit. The emit takes 1.46 ms longer on the p150 (task #365).
That is the whole "blend gap". **Nothing in blend is ours to fix, so no code change was made.**

## Method

- 30-view Tracy captures (device + host zones) of iter-209 code on both boards, same day:
  - p150 bh-30: the #355 tree `/localdev/smarton/p150bench-t355/tree` (010939d2, iter-209 render
    code, built against the viewer's tt-metal 437bc366, used read-only), `bh30_tracy.sh` via
    `drive_bh30.sh`.
  - p100a yyzo-bh-04: the #355 tree `/localdev/smarton/gstt2-t355` (8588f1ac, iter-209 render code),
    `opt/profiler/capture_tracy.sh` with `GSTT2_REPO` set to that tree, via `drive_p100a.sh` under `ttp lock p100`.
- `blend_cores.py` (new): per frame, the window, per-core busy time (mean/min/max), and the spread of
  per-core start and end times for the mat and blend zones (frames 1-30).
- `docs/p150-gap-t356/timeline.py`: per-stage device and host timeline (frames 1-30).
- Untraced per-view stages: the #355 A/B base arms (`docs/k2-rows-view-t355/out`, 3 rounds per board).
- `memcpy_bench.py`: copy one 3 MB u8 image on each host (no device), to split d2h into host copy
  and transfer.
- The profiler adds ~0.1-0.2 ms on both boards, so only differences are used.

## Per-cause breakdown (iter-209, ms)

| candidate cause | measurement | p150 bh-30 | p100a yyzo-bh-04 | p150 - p100a |
|---|---|---:|---:|---:|
| blend compute (SFPU/FPU, incl. CB waits) | `tile_blend_sfpu` per-core busy, mean (min-max) | 3.682 (3.130-4.099) | 3.681 (3.133-4.093) | **0.00** |
| blend NoC/DRAM reads | `tile_blend_load` (reader) per-core busy, mean; window | 3.616; 4.162 | 3.616; 4.166 | **0.00** |
| blend window (first start to last end) | `tile_blend_sfpu` window | 4.166 | 4.172 | **-0.01** |
| blend per-core load balance | end spread over cores (start spread) | 0.139 (0.937) | 0.142 (0.941) | **0.00** |
| core grid / harvesting | cores running blend (mat TRISC zones) | 110 (330) | 110 (330) | same |
| mat (same merged program) | `mat_cull_mask` window, per-core mean | 2.830, 2.292 | 2.836, 2.298 | **-0.01** |
| **sort emit (device)** | `sort_ol_emit` window | 2.792 | 1.333 | **+1.46** (task #365) |
| pfwc + tile-assign K2 (device) | pfwc window; k2_pairs window | 1.831; 0.814 | 1.960; 0.969 | **-0.29** |
| emit end -> mat start (device idle) | mat start - emit end | 0.024 | 0.829 (traced: the host bridge is slower under Tracy) | not a p150 cost |
| image d2h (untraced) | `d2h` stage, mean of 3 rounds | 0.470 (0.31 / 0.55 / 0.55) | 0.226 | **+0.24** |
| of it: host copy of the 3 MB image | `memcpy_bench.py`, fresh or reused destination | 0.090-0.100 | 0.099-0.100 | 0.00 |
| host sort work (publish_host, layout) | untraced `SORT_STAGES` | 0.35, 0.083 | 0.20, 0.053 | hidden: the p150 device is still in emit |
| rest (host enqueue/finish latency on the slower CPU) | frame gap minus the rows above | | | ~+0.3 |
| **frame (untraced, #355 base arms)** | ms/view | **12.608** | **10.888** | **+1.72** |

The host `blend` stage (untraced) is 7.97 ms on the p150 and 7.04 ms on the p100a. Mat + blend on
the device take 6.06 ms on both, so the stage holds 1.91 ms (p150) vs 0.98 ms (p100a) of the device
still running the emit when the host enqueues mat+blend: +0.93 ms. That is the emit gap seen from
the host.

Timelines, ms (device: from the first pfwc start; host: from the first EnqueueProgram; frames 1-30):

| zone | p150 bh-30 | p100a yyzo-bh-04 |
|---|---|---|
| pfwc | 0.000-1.831 | 0.000-1.960 |
| k2_rows | 2.352-2.723 | 2.555-3.002 |
| sort_ol_emit | 2.837-5.628 | 3.177-4.510 |
| mat_cull_mask | 5.652-8.483 | 5.339-8.175 |
| tile_blend_sfpu | 7.546-11.712 | 7.234-11.406 |
| host_finish_blend (host clock) | 5.136-11.872 | 4.870-11.517 |

Iter-207 captures (#356, `docs/p150-gap-t356/prof`) give the same mat and blend numbers on both boards
(blend window 4.166 vs 4.169, per-core mean 3.688 vs 3.685), so this has not changed between 207 and 209.

## Conclusions

1. Blend is not slower on the p150: compute, reader, CB waits, core count and balance all match within
   0.01 ms. The 8 vs 7 DRAM banks, the harvested grid and NoC routing make no measurable difference to
   blend. There is no blend fix worth >=1 %, so per the spec nothing was changed.
2. The p150 frame gap (+1.72 ms) is: emit +1.46 (task #365), pfwc/K2 -0.29, d2h +0.24, host
   latency ~+0.3.
3. d2h: the host copy costs the same on both hosts (~0.1 ms), so the extra 0.24 ms on the p150 is the
   transfer and completion-queue path (PCIe Gen4 on bh-30 vs Gen5 on yyzo-bh-04, see docs/p150-gap.md).
   tt-metal 437bc366 has pinned memory only for host-to-device writes; reads always go through the
   completion queue. Both hosts have the IOMMU on (bh-30 129 groups, yyzo-bh-04 30 groups) and KMD
   2.9.0, so a `PinnedMemory::Create(..., map_to_noc=true)` buffer could be written by the blend
   writer directly, during blend. That would remove most of d2h: up to ~0.37 ms (~3 %) on the p150
   and ~0.13 ms (~1 %) on the p100a. Not tried here (needs a kernel writer change and review); proposed
   as a follow-up.

## Screenshot

No code changed and no new arm was measured that could become an iteration, so there is no new hero.
The Tracy runs used the iter-209 code; its device hero (42.51 dB vs `benchmarks/reference_v2/hero.png`,
md5 906e0435, no seams) is in `docs/k2-rows-view-t355`.

## bh-30 viewer window

Existing viewer reservation, no IRD reserve/extend/release. Viewer stopped 2026-10-07 23:02:58Z,
restarted 23:04:20Z and READY at 23:04:28Z (1 min 30 s), at sha 4307f076 (unchanged). Self-test
13.10 ms median, localhost:8091 answered 200.

## Files

- `blend_cores.py`, `memcpy_bench.py`, `bh30_tracy.sh`, `drive_bh30.sh`, `drive_p100a.sh`
- `out/t366-tracy-u.csv.gz`, `out/t366-dev.csv.gz`, `out/t366-T.log` (p150); `out/p100a-tracy-u.csv.gz`,
  `out/p100a-dev.csv.gz` (p100a); `out/timeline-*-209.txt`, `out/cores-*-209.txt`, `out/memcpy_bench.txt`

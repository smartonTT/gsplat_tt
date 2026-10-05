# Profile of the iter-199 tip (task #267)

Tip: smarton/tt-project-opt at best-iter-199 (render code = eb3a371; remote tree 5b6a1e7).
Board: yyzo-bh-07, Blackhole p100a, 11x10 = 110 compute cores (all used).
Scene: bicycle, 30-view sweep, defaults. All numbers measured on device; nothing estimated
except where a lever row says "est.".

Raw outputs: `docs/profile-iter199-t267/out/` (capture log, zones, gaps, roofline, untraced
logs and md5 lists). Driver: `docs/profile-iter199-t267/drive.sh`. Per-core mat/blend
analysis: `opt/profiler/matblend_cores.py`. Device idle between views: `opt/profiler/interview_gaps.py`.

## 1. Tracy kcfg auto-size check (t258 fix, d5508ae): PASS

`capture_tracy.sh` run with `GSPLAT_TT_KCFG_EXTRA_KB` unset: build and capture OK,
`[capture_tracy] DONE rc=0`, no `TT_FATAL`, `command queues 2, compute grid 11x10`,
render.tracy 8.38 MB, 29 frames parsed. The override is no longer needed.

## 2. Untraced numbers (same session)

| arm | ms/view | notes |
|---|---:|---|
| default | 11.584 (avg_frame 11.6, p50 11.7, min 10.0, max 13.2) | 30/30 md5 = r82new (46a725ab hero) |
| host-stage profile | 11.668 | md5 OK |

Host stage means (hp arm, ms/view): project 3.03 (of which gather_wait 2.93 = device
wait, pfwc rtargs 0.042, enqueue 0.038), sort 0.88 (bin_emit 0.563, publish_host 0.166,
mat 0.064, layout 0.049), blend 7.47, d2h 0.234, pfwc dispatch 0.080, head/tail < 0.01.
Python between views ~224 ms (outside the timed region).

Host work that is serial with the device inside render(): d2h 0.234 + pfwc dispatch 0.08
+ rtargs/enqueue ~0.08 ≈ 0.39 ms; the rest of the host stages are waits on the device.

## 3. Tracy breakdown (ms/view, mean of 29 views)

| # | program | window | gap before | busiest RISC mean / max | tail idle |
|---|---|---:|---:|---|---:|
| 0 | pfwc | 2.008 | 0 | BRISC 1.795/2.006, NCRISC 1.803/1.969, TRISC 1.76/1.95 | 0.21 |
| 1 | K2 (tile_assign scatter) | 0.981 | 0.006 | NCRISC 0.905/0.980, BRISC 0.870/0.948; TRISCs idle | 0.08 |
| 2 | sort_ol | 1.432 | 0.010 | emit NCRISC 1.200/1.292, emit BRISC 1.161/1.308, town TRISC 1.085/1.23 | 0.09 |
| 3 | mat + blend | 6.895 | 0.015 | see below | 0.65 |
| | **device span** | **11.348** (min 9.70, max 12.93) | 0.031 total | | ~1.03 |

Mat + blend (one Tracy segment; two back-to-back programs):

| part | mean | max |
|---|---:|---:|
| mat_cull_mask (TRISC) | 2.512 | 3.001 |
| sort_subchunk_mat BRISC / NCRISC | 2.375 / 2.491 | 2.464 / 3.001 |
| tile_blend_sfpu (TRISC1) | 3.732 | 3.814 (makespan 3.758) |
| tile_blend_load (NCRISC) | 3.664 | 3.733 |

Per-core (matblend_cores.py): mat ends at 2.512 mean / 3.001 max, but **blend starts at
3.001-3.003 on every core** (program barrier). Mean per-core wait mat-end → blend-start
= **0.489 ms**. Blend tail idle after the dynamic claim = 0.082. corr(mat_end, blend_end) ≈ 0,
so the claim already balances blend; the waste is the barrier in front of it.

Per-frame RISC busy sums: BRISC 10.14, NCRISC 10.20, TRISC 9.24 vs span 11.35. Overlap
bound (sum of per-program busiest-RISC means) 10.79.

### Change vs #230 (iter-197 tip) / t258

| program | #230 | t267 | delta | cause |
|---|---:|---:|---:|---|
| pfwc | 2.33 | 2.01 | −0.32 | t232 P2 cov2d SFPU + NoC1 writer rebalance |
| K2 | 0.98 | 0.98 | 0 | |
| sort_ol | 1.44 | 1.43 | 0 | |
| mat + blend | 7.56 | 6.90 | −0.66 | t231 decode-ahead (blend sfpu 4.42 → 3.76) |
| traced span | 12.31 | 11.35 | −0.96 | |

## 4. Critical path and idle resources

- The path is strictly serial: pfwc → K2 → sort_ol → mat → blend, each program waits for
  its slowest core. Launch gaps are negligible (0.031 total).
- Biggest measured idle: the **mat → blend barrier, 0.489 ms per core** (mat imbalance:
  mean 2.51 vs max 3.00, one slow NCRISC mover). Next: per-program tails, ~1.03 ms total
  of which 0.65 is in program 3 (mostly the same barrier).
- **K2: all three TRISCs idle for 0.98 ms** on every core. pfwc and sort_ol use all five RISCs.
- FPU: unused in every program (FPU blend ruled out by t111).
- Host: ~0.39 ms of serial host work per view (d2h + dispatch + rtargs).
- Blend itself: TRISC1 SFPU-bound (3.73 of 3.76 makespan), already dynamically balanced.

## 5. Ranked levers (≥0.3 ms/view est.; none re-proposes shelved items)

Excluded per spec: mat split, mask-0 drop, mover table/fold, blend chain walk, emit-loop
cursors, U2, more replay, #147 per-tile mat-in-blend fusion, #169, #165, #168, #171,
#148, #172, #215, #111.

**L1. Remove the mat → blend program barrier (per-tile readiness).** est. 0.30-0.49.
Launch mat and blend as one program (or blend without waiting on the mat program) and let
blend's existing atomic tile claim take only tiles whose mat output is published (per-tile
or per-chunk ready flag in L1/DRAM, set by the mat writer, polled by the blend reader).
Fast mat cores start blending at ~2.5 ms instead of 3.0. This is not #147: mat stays its
own pass with its own buffers; only the barrier goes. Bound: the 0.489 mean wait; the
slow mat core's own tiles still finish late, so the real gain is wait minus the tail
created by tiles that become ready last.
Kill gate (before building): replay the Tracy per-core mat_end and per-tile blend times
in a small model (ready time = mat end of the producing core; claim order unchanged);
kill if modeled saving < 0.25 ms or if mat + blend kernels and CBs do not fit L1/kcfg
together. After building: kill if measured < 0.3 ms or md5 changes.

**L2. Give K2's idle TRISCs work, or fold K2 into sort_ol.** est. 0.3-0.6.
K2 runs 0.98 ms on BRISC/NCRISC only. Two routes: (a) fuse K2 into the sort_ol launch so
sort's town phase (TRISC) on a tile range starts as soon as that range's K2 counts and
pairs are written (per-range ready flags, same pattern as L1), overlapping K2 movers with
town; (b) move K2's per-pair key/tile computation into a TRISC producer that hands
packed records to the movers through a CB, leaving the movers pure NoC writers.
Kill gate: first measure K2's split between compute (key/tile math, loop overhead) and
NoC wait with zone markers; kill (b) if compute < 40 % of K2 busy; kill (a) if a model
from the capture (K2 per-core end vs town start) gives < 0.25 ms.

**L3. Overlap the host serial tail with the device.** est. 0.25-0.39 (borderline).
d2h 0.234 ms runs after blend ends; pfwc dispatch + rtargs (~0.16) run before pfwc.
Read finished output rows while blend runs (blend tiles finish in claim order; a second
CQ exists), and pre-build next-view rtargs during the device span.
Kill gate: measure untraced with d2h removed (GSPLAT_TT_DUMMY-style skip) to get the true
ceiling; kill if the ceiling < 0.3 ms.

**L4. Program-tail trimming in pfwc and sort_ol by dynamic range claim.** est. 0.2-0.3.
pfwc tail 0.21 (BRISC max 2.006 vs mean 1.795), sort_ol 0.09-0.15. Both use static ranges.
An atomic chunk counter (like blend's) would cut most of it.
Kill gate: only worth it if combined with L1/L2 (same ready-flag/claim plumbing);
standalone, kill if the measured per-core max-minus-mean sum < 0.3 ms.

**L5. Throughput metric (metric question, not a kernel change).** est. large on a
throughput metric, 0 on the current latency metric. Two CQs exist; the next view's pfwc
could overlap this view's blend tail/d2h. #171 found 0 gain because the bench measures
per-view latency with ~224 ms Python between views. Needs a coordinator/user decision on
whether a frames-per-second-under-load number is reported next to latency.

## 6. Recommendation

Queue L1 first (largest measured idle, isolated change). L2 next, starting with the
zone-split measurement. L3 only after its ceiling measurement. L4 rides on L1/L2. L5 is a
question for the coordinator.

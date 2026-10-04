> **Note (task #182, 2026-10-04):** this describes the lever-2 chain (GSPLAT_TT_PFWC_FUSE=0: proj_vis_scan / proj_scatter / ta_bucket_scatter), not the default pipeline (PFWC_FUSE=1 since #122: pfwc, K2, sort_ol, mat, blend). See docs/vis-scan-t180/RESULT.md (commit e13e4f8 on ttp/t180-device-build-a-b-of-fused-multi-core-pro) and docs/rerank-17ms.md.

# Task #179: model of a multi-core proj_vis_scan (no device)

**Answer: estimated untraced saving 0.36-0.46 ms/view (central 0.39), above the 0.3 gate.
Recommend a build: fold the scan into the gather_vis_scatter program as a per-core partial
sum + one barrier/all-gather + per-mover cut, and drop the separate scan launch.** Nothing was
run on a device; all numbers below come from existing captures and probes.

## What the scan does today

- `render/kernels/dataflow/gather_vis_scan.cpp` + `vis_tile::scan_slots` (`vis_tile.h`), its own
  program on core (0,0) BRISC (`render/host/gather_visible_device.cpp:393-410`), enqueued right
  before the scatter program (`:1194-1195`). The scatter reads its slot page as its first act, so
  the whole scan sits on the device chain pfwc -> scan -> scatter -> host reads proj_M.
- Input: 5989 per-tile [visible, pairs] counts (6.13 M gaussians / 1024), 47 KB. Work: three
  serial passes over all 5989 tiles (total weight; balanced cut with two 64-bit multiplies per
  tile; bases), then 220 slot pages + 3 publish pages written.
- Measured zone: 0.396 ms/view (docs/t159, yyzo-bh-07 p100a), 0.436-0.445 in t121/t146 captures.
  It is one zone with no inner markers, so the traced time is the untraced time. In the program
  view it is merged with the scatter: 3.70 ms = scan 0.40 + boundary + scatter 3.24
  (docs/t159/gaps.txt).

## Model (docs/proj-vis-scan-model/model.py, output in model-out.txt)

Cycle costs from docs/hw-ceilings.md (p100a probes: dependent L1 load 8.26 cyc, write 77.7,
64 B DRAM read 476 exposed / 105 pipelined, divu 20.4).

1. **Calibration.** The per-tile instruction estimate of today's loops gives 76 cyc/tile ->
   351 us, against 396-445 us measured (model ~12% low). The scan is RISC latency-bound
   (~90 cyc/tile measured), not I/O-bound: the 47 KB read and 223 page writes are ~14 us.
2. **Fused multi-core scan** (sort_bin_onelaunch.cpp already uses this barrier shape):
   each core's BRISC reads its 55 strided count entries, sums them (vis, pairs, weight, last
   non-empty position), sends a 16 B record to core 0 and increments a semaphore; core 0
   multicasts the 110-record array and a release. Each mover then computes the total, walks the
   core prefixes to its cut (s * total / S), walks the 55 tiles of that core, and gets base,
   pair base and is_last. NCRISC waits on a local semaphore. Prologue cost per case:
   optimistic 22 us, central 31 us, pessimistic 50 us (sort_ol-style 109 unicast releases,
   5 us start skew, on-demand reads for the cut core).
3. **Launch removed.** One fewer program boundary. The traced gap before the scan is
   35-48 us (an upper bound); the model takes 10-40 us untraced.

| case | scan removed | boundary removed | prologue added | saving ms/view |
|---|---:|---:|---:|---:|
| optimistic | 445 us | 40 us | 22 us | 0.46 |
| central | 400 us | 25 us | 31 us | **0.39** |
| pessimistic | 396 us | 10 us | 50 us | 0.36 |

The saving should carry over untraced 1:1: the host blocks on the proj_M read right after the
scatter, so the chain from enqueue to readback gets shorter by the same amount (unlike the
Tracy gap work in #155, this is kernel time, not host bridge).

**Cheaper alternative, not recommended alone:** keep one core but rewrite the loop
(non-volatile loads, 32-bit thresholds with one divu per slot, stride walk, cut and bases in
one pass): ~23 cyc/tile -> ~121 us, saving 0.28-0.32 ms/view, right at the gate. It is worth
adding as a middle knob value in the build because it costs little and splits the gain into
"faster loop" and "parallel + no launch".

## Output safety

The compaction order does not depend on the cut (vis_tile.h comment and #99 results), and the
fused version computes the same q0/qn/base/pbase/is_last as scan_slots, so the image should be
md5-identical, not just within 1 LSB. Unit check: compare the per-slot values against
`vis_tile::scan_slots` in tests/unit/test_vis_lever2.cpp.

Build hazards found while modelling:
- **offs tail page race.** Today the scan writes the offs page holding offs[M] (all P) before
  the scatter starts; the is_last slot then pads the page holding M-1. Fused, both writes can
  land in either order and P could overwrite real offsets. The is_last mover must write that
  page itself (and only when M % 16 == 0 is it a separate page).
- proj_M / ta_pairs_P publish moves to one mover (core 0 BRISC after the barrier).
- Kernel config space: the scatter already has 32 runtime args and 24 accessors; use a
  multicast release (2 args: rectangle) rather than 109 per-core NoC coords on core 0.
- L1: +~4 KB per core (records) or +48 KB if each core stages the whole counts array; the
  scatter's CBs are ~130 KB, so both fit.

## Recommendation

Build (estimate 0.39 ms/view central, >= 0.3 gate in all three cases). Follow-up spec below.

### Follow-up: device build + A/B of the fused multi-core proj_vis_scan

In a worktree of ~/dev/gstt2 on a task branch from smarton/tt-project-opt: add knob
`GSPLAT_TT_VIS_SCAN` (0 = today's scan program, default for now; 1 = fast single-core loop in
gather_vis_scan.cpp; 2 = scan fused into gather_vis_scatter.cpp: per-core 55-tile partial sums,
16 B records gathered on core 0 via semaphore, multicast release, per-mover cut/base/is_last,
NCRISC waits on a local semaphore, no wl_vis_scan enqueue). Handle the offs tail page in the
is_last mover and the proj_M / ta_pairs_P publish on core 0. Add a host unit test that checks
both new paths' slot values against vis_tile::scan_slots on random and bicycle-like counts.
On the p150/p100a reservation, inside `ttp lock p100 -- ...`: bicycle 30-view benchmark,
paired 3 x 30 rounds per knob value (0/1/2); gate: md5 of all views equal to md5-r82new.txt
(at most 1 LSB allowed, md5-identical expected), land knob 2 as default only if the paired
saving is >= 0.3 ms/view untraced; take one Tracy capture to confirm proj_vis_scan is gone and
the scatter's prologue zone is <= 50 us. Update the status HTML with the measured result.

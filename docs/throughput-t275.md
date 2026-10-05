# Back-to-back throughput vs per-view latency (task #275)

Tip: `675bcb5` (best-iter-199 + later default-off changes; output identical: sweep md5
46a725ab, 30/30 views equal to md5-r82new). Board: yyzo-bh-07, Blackhole p100a.
Bicycle, 30 views, 1024x1024, defaults. 3 untraced rounds, all arms in each round.
Driver: `docs/throughput-t275/drive.sh`, remote `remote_time.sh`; logs and md5 lists in
`docs/throughput-t275/out/`.

## Mode

`render/run.py --back-to-back [--b2b-passes 3] [--b2b-drop]` renders the sweep with no
host work between views. Extrinsics are built before the timed window; the window holds
only `pipeline.render()` per view. One check pass runs first (its frames are md5'd after
the pass and dumped with the latency mode's file names, so the sweep md5 is the same
check), then 3 measured passes. Kept (default): the measured passes keep their frames
and must match the check pass byte for byte (exit 5 otherwise). Dropped (`--b2b-drop`):
each frame is dropped when `render()` returns, like a viewer. Reported: wall / frames.

## Results (ms)

| arm | r1 | r2 | r3 | mean | FPS | md5 |
|---|---:|---:|---:|---:|---:|---|
| latency, ms/view (primary) | 11.592 | 11.684 | 11.642 | 11.639 | 85.9 | 46a725ab, 30/30 golden |
| throughput, back-to-back, kept | 11.631 | 11.612 | 11.611 | **11.618** | 86.1 | 46a725ab, all passes identical |
| throughput, back-to-back, dropped | 11.587 | 11.593 | 11.589 | 11.590 | 86.3 | 46a725ab (check pass) |

Throughput equals latency within 0.05 ms (0.4 %). `render()` is fully synchronous: it
ends with a blocking D2H, so the ~224 ms of Python between views in latency mode costs
nothing in the timed number and removing it gains nothing. The check pass is 0.36-0.69 ms
slower per frame (12.00-12.28): it first-touches fresh output pages (30 x 3 MB kept);
later passes reuse that memory.

## What cross-view overlap could gain (estimate, not measured)

Throughput mode only pays off if work of view N+1 runs while view N still occupies
something. Inside `render()` the device is idle only while the host works:

- D2H after blend: 0.190 / 0.263 / 0.217 ms (this session's stage timer, mean 0.223).
- pfwc dispatch + rtargs + enqueue: ~0.16 ms (t267 host profile), but part of it already
  runs while the previous program executes.
- Bound: untraced wall minus traced device span = 11.584 - 11.348 = **0.24 ms** (t267).
  That is all non-device time per view, including launch gaps (0.031).

Options from the spec:

1. Next view's pfwc during this view's blend/D2H (2 CQs): blend uses all 110 cores and
   is SFPU-bound with a dynamic claim; pfwc also uses all 110 cores and all 5 RISCs. A
   second CQ adds no cores, so only the D2H window (0.22) can be overlapped, and only
   with a double-buffered output. The PCIe read then competes with pfwc's DRAM gather.
2. Host rtargs prebuild for the next view: 0.04-0.08, mostly already hidden.

Estimate: at most 0.24 ms/frame ideal, ~0.15-0.2 realistic (2 %). Below the 0.3 ms gate,
so no build follow-up is proposed. This confirms #171 (host residue/D2H overlap, shelved
at ~0) for the throughput metric too. If revisited, first measure the ceiling untraced
with the blocking D2H removed (t267 L3 kill gate).

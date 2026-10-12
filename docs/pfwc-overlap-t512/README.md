# t512: overlap frame N+1 pfwc with frame N blend tail — not built, bounded from the t506 timeline

Task #512, 2026-10-12 (PDT). Baseline best-iter-223 (5ca019ae). No device run: the measured
p150 b2b timeline of t506 (`docs/b2b-timeline-p150-223`, 15 steady periods, 120 cores, 1350 MHz)
already answers whether the lever can work and what it can save.

## Why "issue pfwc N+1 on CQ1 while blend N drains on CQ0" cannot overlap

- In b2b mode the pfwc N+1 is already queued right behind blend N (task #379 hook,
  `render/host/blend_device.cpp` ~979), so dispatch is not the gap: all-core idle at the
  blend N -> pfwc N+1 boundary is 0.015 ms.
- The lost time is blend N's **end spread** (0.146 ms): early cores sit idle until the last core
  finishes, because a program's go signal goes to all its cores at once, after every core is done
  with the previous program. t506 found 0 per-core overlaps between programs.
- A second CQ does not change this. Both queues dispatch to the same 120 worker cores and share their
  launch and kernel-config rings, so programs on the same cores in two CQs must be ordered by events
  (tt-metal does not support running them concurrently on the same cores). Putting pfwc on CQ1 at most
  moves the same serialized start to another queue.
- The only way to start pfwc N+1 on a core as soon as that core finishes blend N is to run both
  in **one program**: pfwc N+1's three kernels fused onto the tail of the mat+blend kernels, with
  their CBs, semaphores and runtime args merged, and pfwc's output buffers double-buffered against
  the ones blend N reads. That is a large build, so first: what would it save?

## Bound with the real per-core data (`bound512.py`, output `bound512.txt`)

For each steady period, the model starts core c's pfwc N+1 at blend_end_c(N) + 2 us and keeps the
measured per-core pfwc lengths. k2 N+1 starts when the last core finishes.

| model | saves (ms/view, mean of 15) | range | what it needs |
|---|---|---|---|
| per-core start, today's pfwc split | **0.070** | 0.019-0.126 | the fused blend+pfwc program |
| per-core start, slowest pfwc work on earliest blend cores | 0.204 | 0.138-0.285 | fusion plus a split that knows blend's end order in advance (not available) |
| per-core start, pfwc work perfectly divisible (dynamic claiming) | 0.322 | 0.245-0.407 | fusion plus a work-stealing pfwc |
| static per-core pfwc reweight, **no overlap** | **0.095** | 0.022-0.156 | host-side deal change only (leave-one-out weights) |

The per-core start saves only 0.070 because the cores that finish blend late also tend to have long
pfwc (correlation 0.27), and pfwc's own end spread (0.158 ms) moves to the pfwc -> k2 boundary.

pfwc per-core length (20 views x 120 cores): mean 1.449, mean max 1.607 ms. **77% of the
per-core variation is a property of the core, not of the view**. The core speed factors span
0.943-1.104; the slowest cores are (13,7) 1.104, (11,5) 1.103, (6,5) 1.089, (14,3) 1.084,
(3,2) 1.082. The t439 LPT deal weighted tiles by a CPU model of their gaussian counts, and the
mean core got slower by as much as the max fell. A per-core speed factor is a different input:
it would give the measured slow cores fewer tiles and keep the tile order.

## Verdict

**Not built; shelved as specified (CQ1).** The CQ1 mechanism cannot overlap at all. Fusing pfwc
into the blend tail has a realistic bound of 0.070 ms/view. That is in the stack band only as a
bound, and it costs a fused two-stage program with double-buffered pfwc outputs. The same data
points to two better levers (follow-ups):

1. a per-core speed-weighted pfwc deal (bound 0.095 ms/view, host only, no overlap needed), with
   the cause of the slow cores checked (NoC or DRAM distance);
2. a work-claiming pfwc fused onto the blend tail (bound 0.32 ms/view), only if (1) shows the
   pfwc balance can be reached on the device.

Reproduce: `python3 ../b2b-timeline-p150-223/prep506.py -o p.csv.gz <t506 out2>/dev-P{0,5,10,15,25}.csv.gz;
python3 bound512.py p.csv.gz` (~8 min on the Mac; the raw CSVs are in
`tt-project/state/runs/1393/wt/docs/b2b-timeline-p150-223/out2/`).

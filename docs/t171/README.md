# t171 — host residue gate: device worklist (add-on A) + d2h overlap

**Result: not built.** Gate measured at tip 8d33ffe (= 02329ca + driver) on yyzo-bh-07 p100a,
bicycle, 30 views, untraced (no Tracy). Both runs md5-identical to `md5-r82new.txt`.

## Measurement (ms/view, 30-view mean)

| stage | base run (STAGES) | host-profile run (HOST_PROFILE=1 + PER_VIEW_STAGES=1) |
|---|---|---|
| sort pread (totals read) | 0.015 | 0.023 |
| sort bin_layout (host layout + LPT + uploads) | 0.097 | 0.120 |
| sort publish_host (subchunk layout, mat worklist, uploads) | 0.157 | 0.196 |
| sort mat (materialize enqueue) | 0.041 | 0.052 |
| **host bridge before materialize** | **0.310** | 0.391 |
| d2h (output read) | 0.197 | 0.212 |
| avg_frame_ms | 18.457 | 18.612 |

Raw: `out/run-r1-base.log`, `out/run-r1-hp.log`, `out/hp-r1.txt` (VIEW_STAGES / HPGAP lines),
`out/md5-r1-*.txt`. Driver: `drive.sh`, `remote_time.sh`. num_tiles = 1024, P ≈ 3.24 M.

## Why the removable part is below the 0.3 ms gate

The nominal sum (0.31 + 0.20 = 0.51 ms) clears 0.3, but the part either change can actually
remove does not:

1. **d2h overlap (part 2): ~0 under the project metric.** `render/run.py` times each view as
   the latency of `pipeline.render()`, which must return the image. The views are not run
   back to back (the HPGAP lines show ~210-230 ms of Python/PNG work between views), so there
   is no next-view pfwc to hide the read under. Hiding d2h needs a pipelined throughput loop,
   which changes the metric, not the renderer. Chunked readback inside one view does not help
   either: blend is LPT-balanced (t167), so cores finish within a few µs of each other.
2. **Device worklist (part 1): net well below 0.31 ms.** The host bridge does the per-tile
   layout, LPT over 1024 tiles x cores, downstream metadata, subchunk layout and mover
   worklist, then ~6 small uploads. On device this becomes serial work on one RISC-V
   (a sort of 1024 costs plus a least-loaded-core scan per tile, then the subchunk and mover
   passes) followed by a NoC broadcast and a barrier before materialize can start. That
   serial device pass sits on the same critical path and plausibly costs 0.1-0.3 ms itself
   (RISC-V vs a desktop core), so the realistic net is ~0.0-0.2 ms (≤ 1%), consistent with
   the ~0.33 ms ceiling t147 gave and t155's finding that the untraced bridge is small.

Exact host-LPT order reproduction plus md5-identity would also need a sizeable kernel, so the
cost is not worth a ≤ 1% outcome under the charter.

## When to revisit

- If the frame metric moves to pipelined throughput (views back to back), part 2 becomes a
  real ~0.2 ms win and is cheap: double-buffer the output and read view i after enqueuing
  view i+1's pfwc.
- If the host bridge grows past ~0.6 ms (more tiles, higher resolution), re-run this gate
  (`docs/t171/drive.sh`).

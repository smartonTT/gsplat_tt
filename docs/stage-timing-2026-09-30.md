# Per-stage attribution of the 173 ms bicycle frame (2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150 — no p150 was obtainable).
Scene: bicycle, 30 views, 1024x1024, `python3 render/run.py --iter-dir t15-stages --no-ref`.
Build: render_clean rebuilt from `smarton/tt-project-opt` with `render/host/stage_timers.{h,cpp}`.

Before this, the bench only emitted `TTW_TIMING ms_view=` and `blend=` (both the same
avg frame time) plus the `[SORT]` line, so ~135 ms/view was unattributed.

## Measured (mean over the 30 timed views, warmup excluded)

| stage | ms | % of frame | what it is |
|---|---:|---:|---|
| head | 0.189 | 0.1% | py buffer requests + 12 MB output-image memset |
| project | 36.957 | 21.3% | fused means_cam+pfwc, then gather_visible |
| tile_assign | 21.778 | 12.6% | device tile_assign (bbox + scan + scatter) |
| sort | 46.736 | 27.0% | `sort_and_bin_tt` minus its fused cull/blend continuation |
| blend_setup | 0.026 | 0.0% | resident-blend `SetRuntimeArgs` pre-pass |
| cull | 0.008 | 0.0% | SFPU microblock cull **launch only** (see caveat) |
| blend | 63.599 | 36.7% | blend enqueue + the single `Finish` that drains the CQ |
| d2h | 0.587 | 0.3% | final bf16 image readback |
| assemble | 3.217 | 1.9% | host bf16 microblock tiles -> fp32 HWC image |
| tail | 0.004 | 0.0% | P_kept scan + stats dict + pybind return |
| **sum** | **173.102** | | |

- `view_total` (render_view entry -> return) = **173.103 ms**; residual inside
  render_view = **+0.001 ms**.
- `avg_frame_ms` (Python-side wall clock around `pipeline.render`) = **173.154 ms**;
  residual vs the stage sum = **+0.052 ms** (pure pybind/marshal). Well under the ~1 ms target.

## Instrumentation overhead

173.154 ms vs the 173.30 ms baseline on the same board (stdev 0.26 ms) — i.e. **-0.08%**,
inside run-to-run noise and far under the 0.5% budget. Quality gate unchanged:
`hero_vs_ref = 100.00 dB` (bit-identical to the golden 8-bit frame).

## Caveats when reading the table

1. **`cull` is ~0 because cull and blend are chained on one command queue.**
   `chain_cull_blend` launches the SFPU cull non-blocking; the only `Finish` is in the
   blend call. So the cull's *device* time lands inside the `blend` bucket.
2. **`blend` is the whole CQ drain, not just blend compute.** Everything piped ahead of
   it (sort publish, subchunk materialize, cull) is drained by that same `Finish`.
3. **`sort` is almost entirely HOST work.** The same run's `[SORT]` line shows
   `bin=~30-36 ms` (host Pass1+Pass2 binning), `publish=~9-10 ms` (H2D of the resident
   outputs) and `kernel=0.02 ms` (the device radix). The device is idle for the ~36 ms
   of host binning.

## Read: top optimization target

Two, in order:

1. **`sort` (46.7 ms, 27%) is the best lever** even though `blend` is larger. It is
   ~36 ms of single-purpose host CPU binning plus ~10 ms of H2D publish, with the
   device idle throughout, inside a pipeline whose whole design goal is host-free.
   Moving Pass1+Pass2 on-device (or overlapping them with the previous frame's blend)
   attacks time that is currently pure bubble.
2. **`blend` (63.6 ms, 37%)** is the largest single bucket and is real device time, but
   it is also the bucket that already absorbed every prior kernel-algebra win; per the
   iter-141 freeze, NCRISC is saturated there (~125 ms/view of NCRISC across the frame).
   It needs a data-movement change, not more fusion.

`project` (37.0 ms) and `tile_assign` (21.8 ms) are the next tier and are not yet
split into host vs device — a follow-up should push the same span discipline into
`pfwc_device.cpp` / `gather_visible_device.cpp` / `tile_assign_device.cpp`.

## How to reproduce

```
devrun.sh --no-verify --timeout 580 --tag t15-bench -- \
  "... cd /localdev/smarton/gstt2; python3 render/run.py --iter-dir t15-stages --no-ref"
```

The run prints one `TTW_TIMING stage_<name>=<ms>` per stage plus a single `STAGES ...`
summary line carrying the sum, `view_total` and both residuals. `ms_view=` and `blend=`
keep their legacy meaning (avg frame time) so the existing report tooling is unaffected.

# Sort stage split and host-binning cut (2026-09-30, task #18)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views,
1024x1024, `python3 render/run.py --no-ref`. All numbers are means over the 30
timed views; the A/B is 3 interleaved rounds (baseline, new, baseline, new, ...)
inside one device reservation.

## 1. What the 46.5 ms sort span actually is

`SortCallTimings` (previously passed as `nullptr`) is now booked into the stage
accumulator, and its lumped `bin` span is split into the pieces that run.
`run.py` prints one `TTW_TIMING stage_sort_<leaf>=` per leaf and a
`SORT_STAGES ... | sum= sort= resid=` line.

| leaf | runs on | before | after | delta |
|---|---|---:|---:|---:|
| `pread` (tile_assign P control page D2H) | host wait | 0.017 | 0.015 | -0.002 |
| `bin_count` (Pass A histogram kernel, launch + Finish) | device | 1.403 | 0.897 | **-0.505** |
| `bin_hist_d2h` (per-(core,tile) histogram D2H) | host | 0.258 | 0.251 | -0.007 |
| `bin_layout` (`host_bin_layout_from_hist`) | host | 1.983 | 0.249 | **-1.734** |
| `upload` (H2D enqueue of layout outputs + metadata) | host | 0.089 | 0.070 | -0.019 |
| `bin_emit` (Pass B scatter/emit kernel, launch + Finish) | device | 32.168 | 30.262 | **-1.907** |
| `kernel` (radix enqueue) | host | 0.027 | 0.024 | -0.003 |
| `publish_host` (publish prep + enqueues) | host, overlapped | 0.167 | 0.149 | -0.018 |
| `publish_wait` (drain of radix + publish [+ directory]) | device | 10.307 | 8.016 | **-2.291** |
| `mat` (materialize enqueue) | host | 0.032 | 0.029 | -0.004 |
| residual (sort - sum of leaves) | | 0.042 | 0.037 | |
| **sort** | | **46.493** | **39.998** | **-6.495** |

Frame: `avg_frame_ms` **173.53 ± 0.56 → 166.63 ± 0.05 ms** (runs 173.3 / 174.3 /
173.0 vs 166.6 / 166.7 / 166.6), i.e. 5.76 → 6.00 FPS. The sort stage accounts
for -6.50 ms; the other -0.42 ms is host noise in `assemble` in one baseline run.
project / tile_assign / blend are unchanged (±0.02 ms).

Quality: `hero_vs_ref = 100.00 dB` in all six runs, and a separate
`--dump-views` run of both builds produced **byte-identical 8-bit PNGs for all 30
views**.

**Correction of the task premise.** The "~30–36 ms host Pass1+Pass2 binning with
the device idle" reading was wrong (as `docs/OPTIMIZATION-PLAN.md` §1.1 already
suspected): Pass 1 (count) and Pass 2 (emit) were already device kernels. Before
this change the host-serial part of the sort stage was **~2.4 ms** (hist D2H
0.26 + layout 1.98 + upload 0.09 + small enqueues); the rest (≈44 ms) is device
time. After it, the host-serial part is **~0.65 ms**.

## 2. What changed (all bit-identical)

1. **Host layout rewrite** (`host_bin_layout_into`, `render/host/sort_device.cpp`).
   Three row-major passes over the 110 x 1024 histogram instead of a tile-outer /
   core-inner walk that touched a new 4 KB row per element; the histogram and
   the three 450 KB base tables persist across views (no per-view ~2 MB of
   allocation and page faults); LPT uses a (load, core) min-heap, which makes
   exactly the choices of the old first-minimum `std::min_element`; the dense
   `recbase` table, which the emit kernel never reads (it fed the retired
   tile_recs scatter), is no longer built or uploaded (450 KB H2D per view).
   `bin_layout` 1.98 → 0.25 ms.
2. **Subchunk directory from the host.** `sort_subchunk_directory` rebuilt the
   blend meta / payload prefix / subchunk dir on **one core**, with ~4 dependent
   DRAM read-modify-write round trips per tile, between publish and materialize.
   The host already computes the identical tables (`build_subchunk_layout`) for
   buffer sizing, so they are now uploaded (~30 KB, CQ-ordered ahead of their
   only readers: materialize, cull, blend). Kernel deleted. `publish_wait`
   10.31 → 8.02 ms.
3. **Emit reads its own histogram row.** The emit kernel re-scanned every
   tid/keep page of its range to recount the per-tile histogram the count pass
   had just produced. The count pass now writes into a dedicated `buf_bin_hist`
   (bin2d still receives the host page bases), and the emit reads its 4 KB row
   via the formerly dead runtime arg 17. `bin_emit` 32.17 → 30.26 ms (includes
   the dropped 450 KB `recbase` H2D that the emit's Finish used to drain).
4. **Count pass read batching.** Mode 0 issued two 64 B reads and a barrier per
   page; it now stages 32 pages per barrier in the (unused in mode 0)
   counting-sort regions. `bin_count` 1.40 → 0.90 ms.

## 3. Costed design: the rest of "binning on device"

What remains host-side between the count and emit kernels is **0.25 ms D2H +
0.25 ms layout + 0.07 ms upload enqueue + ~0.1 ms launch/Finish latency ≈ 0.65 ms**.

- *Multi-core device layout* (column-parallel scan over the histogram, the
  `tile_assign` scan pattern: per-slice tile totals → cross-slice prefix →
  per-(core,tile) bases; 2 extra programs). Removes the D2H + host layout from
  the device-idle path, but the host still needs `counts`, the LPT and the
  subchunk layout to set runtime args of radix / publish / materialize, so a D2H
  of per-tile data remains. Two extra program launches (~0.05–0.1 ms each) plus
  the scan itself eat most of the ≤0.6 ms prize. **Not worth doing alone.**
- *Emit computes its own bases* from the full device-resident histogram (every
  core reads 450 KB and sums 110 x 1024 counts, ~0.5–1 ms of NCRISC per core) so
  the host layout overlaps the emit. Costs more NCRISC time on the pole than it
  hides. **Rejected.**
- The old single-core device layout (`sort_bin_layout.cpp`, gated off since
  iter-127 at +10–16 ms/view) stays gated off.

Conclusion: the host bridge is at its floor (~0.65 ms of 40 ms). The sort
stage is now 99 % device time: **emit 30.3 ms, radix+publish 8.0 ms, count 0.9 ms**.

## 4. Next levers in this stage

1. **De-stall `sort_bucket_emit` (R3), expected −5 to −10 ms.** The per-pair
   loop still has the exposed-round-trip pattern the count pass had: 3 reads + a
   barrier per 16 pairs, a 64 B `blendrec` read + barrier per gaussian (~17 k per
   core per view), a depth-page read + barrier per 16 gaussians, and write
   barriers every 16 records / 16 gaussians. Batching the count pass's identical
   page reads measured 1.40 → 0.90 ms here, which calibrates one exposed NoC read
   round trip at ~0.28 µs; prefetching the page's distinct gaussians' `blendrec`
   under one barrier attacks the largest of these. Bit-identical. Needs device
   zones inside the emit to confirm the split before and after.
2. **Adaptive radix in `sort_tile_depth` (R11), expected −2 to −3 ms** of the
   8.0 ms `publish_wait`.

## How to reproduce

```
devrun.sh --no-verify --timeout 400 --tag t18-ab -- \
  "... cd <tree>; python3 render/run.py --iter-dir t18-ab --no-ref"
```

The A/B used two trees on the box: `/localdev/smarton/gstt2-t18base`
(commit `ab018fd`, instrumentation only) and `/localdev/smarton/gstt2-t18`
(`dede928`). Logs: `~/dev/gstt2/.ttw/logs/t18-ab-*.log`, `t18-dumpcmp-*.log`.

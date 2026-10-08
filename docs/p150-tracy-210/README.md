# t382: first device timeline of iter-210 on bh-30 (p150)

Commit under test: 6640239b (tag best-iter-210). Box: bh-30 (p150, tt_aus, aiclk 1350, fw 19.13.2),
the user's viewer box, used under the existing viewer reservation (no reserve/extend/release).
All numbers are measured on bh-30 in one session; compare only with bh-30 numbers.
The GPU reference 10.75 ms/view is published, not measured.

## Method

- Self-contained tree on bh-30 (`/localdev/smarton/p150bench/tree382`, synced by `sync_bh30.sh`),
  built with `nice -n 19 ionice -c3 -j nproc/2`. The viewer's tt-metal and venv are used read-only;
  nothing under `/localdev/smarton/gstt2` or `/home` was used or touched.
- `drive_bh30.sh` stops the viewer, runs `bench382.sh`, restarts the viewer (trap) and fetches outputs.
- `bench382.sh`: 3 untraced rounds of the 30-view bicycle bench (`render/run.py --no-ref --dump-views`,
  `GSPLAT_PER_VIEW_STAGES=1`), then one Tracy capture of the same 30 views
  (`python3 -m tracy -r -p -v --dump-device-data-mid-run`, `TT_METAL_DEVICE_PROFILER=1`,
  `GSPLAT_TT_PROFILE=1`), exported with `csvexport-release -u`.
- md5 of all 30 views matched `md5-golden-906e0435` in all 3 untraced rounds.
- Analysis: `ana210.py` (windows, gaps, device span, per-core busy), `../p150-gap-t356/timeline.py`
  (same format as `timeline-bh30-209.txt`), `../p150-blend-gap/blend_cores.py`. `ana210.py`
  reproduces the iter-209 numbers from the t366 capture exactly (`out/ana-bh30-209.txt`).

## Untraced bench (bh-30, same session)

| round | ms/view | project | sort | blend | d2h | between views |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 10.952 | 3.003 | 1.087 | 6.442 | 0.332 | 0.069 |
| r2 | 10.960 | 3.001 | 1.075 | 6.449 | 0.344 | 0.071 |
| r3 | 11.083 | 2.982 | 1.088 | 6.474 | 0.461 | 0.060 |
| mean | **10.998** | 2.995 | 1.083 | 6.455 | 0.379 | 0.066 |

Sort sub-stages: bin_emit 0.45-0.47, bin_layout 0.08-0.09, publish_host 0.32-0.33, sort_mat 0.09-0.11.
Project: pfwc rtargs 0.099 + enqueue 0.083 before the device starts, then gather_wait 2.80.
Traced run: 57.0 ms/view (the mid-run dump adds ~46 ms between views); sort 1.98 (bin_emit 1.25), blend 6.19, d2h 0.54.

## Device timeline, 209 vs 210 (traced, ms from pfwc start, mean of 30 views)

| zone | 209 start | 209 end | 210 start | 210 end | 210 window |
|---|---:|---:|---:|---:|---:|
| pfwc | 0.000 | 1.831 | 0.000 | 1.835 | 1.835 |
| k2_pairs | 1.904 | 2.718 | 1.905 | 2.719 | 0.814 |
| k2_rows | 2.352 | 2.723 | 2.354 | 2.725 | 0.371 |
| sort_ol_prefix | 2.739 | 2.822 | 2.740 | 2.823 | 0.082 |
| sort_ol_fill | 2.741 | 2.865 | 2.742 | 2.866 | 0.123 |
| **sort_ol_emit** | 2.837 | **5.628** | 2.838 | **4.126** | **1.288** (209: 2.792) |
| sort_ol_town | 2.855 | 5.562 | 2.856 | 4.058 | 1.203 |
| mat_cull_mask | 5.652 | 8.483 | 4.789 | 7.611 | 2.822 |
| tile_blend_load | 7.468 | 11.629 | 6.599 | 10.759 | 4.160 |
| tile_blend_sfpu | 7.546 | 11.712 | 6.675 | 10.836 | 4.161 |
| last BRISC-FW end | | 11.787 | | 10.913 | |

Gaps (210, traced; mean, min-max): pfwc end -> k2_pairs 0.071; k2_rows -> prefix 0.016;
**emit end -> mat start 0.664 (0.409-0.994)**, was 0.024 at 209; blend end -> last writer end 0.077.

Host zones (host clock, ms from the view's first EnqueueProgram): Enqueue#0/1/2 at 0.000/0.176/0.267;
CQ1 K2-rows read 3.803-3.912 (209: 3.842-4.089); bridge Finish 4.655-4.704; blend setup 4.715-4.801;
**Enqueue#3 (mat+blend) at 4.816** (209: 5.102).

Host/device clock check: the device-minus-host offset is constant per view within 0.03 ms. With it,
the mat device start lands within 0.03 ms of Enqueue#3 (after the same dispatch latency as pfwc), i.e. right on the host enqueue.

## Answers

**(a) Emit window and critical path.** Emit now takes 1.288 ms (2.838-4.126), down from 2.792 at 209.
In the traced run the device then idles 0.664 ms until mat starts, and that start is set by the host
enqueue of mat+blend: the host path after emit is K2-rows read -> bin_layout -> publish_host -> mat
args -> bridge Finish -> blend setup -> Enqueue#3. Tracy inflates this host path (sort stage 1.98 traced
vs 1.08 untraced), so the traced gap overstates it.
Untraced, the device leads instead. The untraced blend stage (from Enqueue#3 to Finish) is 6.455 ms,
0.265 ms longer than traced (6.190), where mat already started at the enqueue. So untraced, mat waits
~0.27 ms after the host enqueue for emit to finish, which puts emit end at ~4.17 ms after pfwc start
(traced: 4.126). **Untraced, emit -> mat is the critical path; the host bridge path is hidden behind
emit with only ~0.27 ms slack.** Cutting emit by more than ~0.27 ms exposes the host bridge.

**(b) Device vs wall (untraced 10.998 ms/view).** Derived from the untraced stages plus the traced
device durations (the device span itself was not measured untraced):

| part | ms/view |
|---|---:|
| host before the device starts (head, pfwc rtargs, enqueue) | ~0.19 |
| device span, pfwc start -> last writer end (~4.17 emit end + 6.05 mat+blend + 0.08 writer) | **~10.30** |
| Finish return after the device ends | ~0.06 |
| d2h (readback) | 0.379 |
| host between views (wall - view_total) | 0.066 |
| **total** | ~11.00 |

So the device idles ~0.70 ms per view (~6%): ~0.19 before pfwc, ~0.44 for Finish+d2h, ~0.07 between
views. The traced device span is 10.913 (9.52-12.50 over views); it is longer because of the
host-gated 0.66 ms gap and profiler overhead. This matches t370 (#370: ~0.8-0.9 ms/view of host wait
on bh-30 at 11.13 ms/view).

**(c) Mat+blend per-core busy (110 cores, mean of per-frame mean/min/max).**

| zone | mean | min | max | window | core-ms (x110) |
|---|---:|---:|---:|---:|---:|
| mat_cull_mask | 2.286 | 1.885 | 2.821 | 2.822 | 251.5 |
| tile_blend_sfpu | 3.682 | 3.131 | 4.093 | 4.161 | 405.0 |
| mat+blend TRISC busy | **5.968** | 5.914 | 6.046 | 6.047 | **656.5** |

Mat then blend run back to back on each core; the mat-start -> blend-end span equals the TRISC busy
(5.968 of a 6.047 window, 98.7% efficient). For the ETH-dispatch estimate: 656.5 core-ms / 120 cores =
5.47 ms, i.e. ~0.50 ms/view less if the work divides as evenly on 120 cores as on 110. Per-core busy is
unchanged from 209 (2.292 / 3.682), so the stride fix moved only emit.

## Implications

- Emit -> mat is the critical path untraced, but only ~0.27 ms ahead of the host bridge. Further emit
  gains need the host path (K2-rows read, bin_layout 0.08, publish_host 0.32, bridge Finish, enqueue)
  shortened or overlapped at the same time, for example by enqueuing mat+blend before the bridge and
  gating it with a device-side semaphore.
- ~0.70 ms/view of device idle sits outside the device span (pre-pfwc host work, Finish + d2h,
  between views). Overlapping view N+1's start with view N's readback attacks it directly.
- 120-core ETH dispatch is worth ~0.5 ms/view on mat+blend by core count alone.

## Viewer window (bh-30)

Viewer stopped 2026-10-08T00:06:53Z, restarted 00:09:28Z, READY 00:09:36Z (2 min 43 s down).
Deployed sha 8329be50 unchanged; self-test 11.53 ms median; localhost:8091 answered 200.

## Files

- `sync_bh30.sh`, `bench382.sh`, `drive_bh30.sh`: sync/build, remote bench + capture, local driver.
- `ana210.py`: windows, host zones, gaps, device span, per-core busy.
- `out/timeline-bh30-210.txt` (timeline.py format), `out/ana-bh30-210.txt`, `out/ana-bh30-209.txt`,
  `out/cores-bh30-210.txt`.
- `out/r{1,2,3}.log`, `out/md5-r{1,2,3}.txt`, `out/hero-r1.png`: untraced bench, md5, hero view.
- `out/T.log`, `out/t382.tracy`, `out/tracy-u.csv.gz`, `out/dev.csv.gz`: traced run and raw captures.

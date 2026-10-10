# Back-to-back vs latency gap (task #464)

**Question.** Why does `render/run.py --back-to-back` take ~9.4 ms/frame when the latency
metric says 7.76 ms/view at the same build? That gap caps the live viewer's frame rate (user
target is at least 117 page FPS, i.e. at most 8.55 ms/frame).

**Answer.** The gap is **pfwc device time** (the fused project and frustum-cull program),
about 1.7 ms/view. In latency mode with `--dump-views`, the PNG save between views gives the
device free time to finish the next view's prefetched pfwc (xview). That removes pfwc from the
measured window. In back-to-back mode, pfwc sits on the critical path. Host work is not the
cause: `_to_image` costs 0.002 ms and the Python residue 0.06 ms. The device is ~95% occupied
during a frame, so there are no large idle bubbles to fill. To reach 117 FPS, device work has
to shrink, mainly pfwc.

## Setup

- bh-30 (p150b, aiclk 1350, 12x10 grid), under the existing viewer reservation. The viewer was
  stopped from 12:57:38Z to 12:59:49Z. After the restart, SELFTEST took 10.19 ms and
  localhost:8091 returned 200.
- Build tree392 = 31bf48c4 (the #435/#439 build; run.py identical to HEAD apart from this
  task's additions), bicycle 1024x1024, 30 views.
- New flag `render/run.py --view-gap-ms X`: a busy-wait of X ms after each view, outside the
  timed window, in both loops. In b2b mode, each pass also prints `B2B_STAGES` (per-view stage
  means from `stage_timings()`) and `render_ms_frame` (the period without the gap). Unit tests:
  `tests/unit/test_run_back_to_back.py`.
- Scripts: `probe464.sh` (remote arms), `drive464.sh` (Mac driver: stop viewer, run, restart,
  check 8091), `vstart464.sh`. Logs: `out/logs/`.
- bh-30 load average was 7-8 even with the viewer stopped, so single passes vary by about
  ±0.5 ms (for example, B0 pass 2 at 10.35 ms).

## Numbers (ms)

| Arm | Mode | Gap | Frame/view | Render window | gather_wait (project) | Notes |
|---|---|---|---|---|---|---|
| Lref | latency + PNG dump | – | **7.788** | – | ~0.85-1.1 (project 1.096) | sort 1.21, blend 5.22; the published metric |
| B0 | b2b | 0 | **9.700** (passes 9.29 / 10.35 / 9.47) | = | **2.65** | blend 5.19, sort ~1.3-1.5, py_resid 0.06 |
| B05 | b2b | 0.5 | period 9.413 | 8.911 | 2.13 | |
| B1 | b2b | 1 | period 9.333 | 8.330 | 1.66 | |
| B2 | b2b | 2 | period 10.354 | 8.350 | 0.975 | sort noisy (1.88) |
| B3 | b2b | 3 | period 10.745 | 7.742 | 0.935 | |
| BX0 | b2b, xview off | 0 | 9.640 | = | 2.73 | |
| L0 | latency, no dump | 0 | 9.135 | – | project 2.511 | |
| L1 | latency, no dump | 1 | 8.175 | – | project 1.548 | |
| L2 | latency, no dump | 2 | 7.614 | – | project 0.997 | matches Lref |
| LX0 | latency, xview off | 0 | 9.495 | – | project 2.861 | |

Correctness: every b2b arm gives raw_md5 d9a60a3c, identical across passes. Lref is
MD5_GOLDEN_OK 39d84b28 on 30/30 views. All 12 hero PNGs are bit-identical.

### What the gap sweep shows

- **gather_wait falls 1 ms for each 1 ms of host gap**, then flattens at ~0.95 ms (gather plus
  its count D2H) once the gap reaches 1.7-2 ms. So the host waits for about 1.7 ms of pfwc work
  left over from the previous frame. This matches the #439 Tracy pfwc window of 1.874 ms
  (traced, so slightly slower).
- **The period stays at ~9.3 ms for gaps of 0-1 ms.** Adding host idle time does not lengthen
  the frame, so the device is the bottleneck at ~9.3 ms/frame and the host has ≥1 ms of slack.
- Latency mode with no dump (L0, 9.14 ms) already matches b2b. The 7.79 ms "latency" figure
  holds only when the host pauses ≥1.7 ms between views.

### xview and `_to_image`

- xview (pfwc N+1 enqueued behind blend N) saves 0.36 ms in latency mode (L0 vs LX0) and about
  0.06-0.35 ms in b2b (noisy). It hides only the host's inter-view gap, not pfwc itself,
  because pfwc stays serial on the same in-order CQ.
- With xview off, the host profile (LXH, `GSPLAT_TT_HOST_PROFILE=1`) puts the median gap from
  blend done to next pfwc enqueue at 0.311 ms: py 0.165 (c2w 0.096), pfwc dispatch 0.127, tail
  0.010, D2H and unpack ~0. **`_to_image` = 0.002 ms.** Neither is on the critical path.

### Device occupancy (Tracy capture of the same build, `docs/pfwc-deal-t439/out/dev-S.csv.gz`)

`devtimeline.py` takes the union of all cores' kernel zones per view (CPU only):

| Program | Window | Core occupancy | Idle per core |
|---|---|---|---|
| pfwc | 1.874 | 0.878 | 0.236 |
| k2 pairs/rows | 0.746 | 0.951 | 0.036 |
| sort_ol | 1.303 | 0.942 | 0.075 |
| mat + blend | 5.195 | 0.974 | 0.133 |

Idle time between programs is 0.05-0.3 ms/view. The host sync points inside a frame (hist
D2H, layout, publish_host 0.33 ms) are mostly hidden behind device work. **There is no large
idle bubble.** Running pfwc on a second CQ beside the frame could reclaim at most ~0.1-0.3 ms
(the blend tail plus gaps), unless pfwc and blend use different limits: pfwc is writer/DRAM
bound, blend is SFPU bound.

## Ranked levers

The target is ≥0.8 ms off the b2b period (9.3 to 8.5 ms) for ≥117 viewer FPS.

| # | Lever | Expected b2b saving | Confidence | Why |
|---|---|---|---|---|
| 1 | **Cut pfwc device time.** Chunk frustum cull inside pfwc only: skip whole chunks of off-screen Gaussians via an index list, without #433's blend reorder, which cost +0.72 ms of blend. | 0.4-0.7 | medium | pfwc is 1.7-1.87 ms and fully on the critical path. #433 reported ~39% of tiles skipped, but judged "pfwc no faster" from gather_wait in latency + dump, where pfwc is invisible. **Re-measure in b2b.** |
| 2 | **Balance pfwc load across cores.** Occupancy is 0.878 and max/mean 1.144. #439's LPT deal was judged in latency + dump mode, so its result is blind to pfwc. | 0.1-0.24 | medium | 0.236 ms idle per core in pfwc. Re-run #439's arm in b2b mode before any new work. |
| 3 | **Fewer pfwc output bytes** (pfwc was writer-bound in #197): narrower records, or write only survivors so gather has less to read. | 0.2-0.5 | low-medium | Also shortens gather, the ~0.95 ms floor of gather_wait. |
| 4 | **Fuse gather into pfwc / drop the count-D2H round trip** (device-side count to sort, no host read). | 0.1-0.3 | low | The 0.95 ms floor is gather device time plus one host round trip. |
| 5 | Overlap pfwc N+1 with mat/blend N on CQ1 using double-buffered pfwc outputs. | ≤0.3 (more only if DRAM-bound pfwc and SFPU-bound blend share cores well) | low | Device is ~95% occupied. Complex (buffer ping-pong, ordering events). Do it after 1-3. |
| 6 | Host sync points inside a frame (sort_bin_emit 0.45, publish_host 0.33 host ms) | <0.1 in b2b | low | Mostly hidden behind device work, per the trace. |

Levers 1+2 are the most direct route to the viewer target. They reopen #433 and #439 (both
shelved) on new evidence: both were judged with a metric that cannot see pfwc.

**Metric change (recommended):** report b2b `ms_frame` (or latency with no dump) for every
iteration next to the latency number. Latency + dump hides ~1.7 ms of pfwc and
misjudges any pfwc lever.

## Hero check

`out/hero.png` (device render, b2b arm B0, build 31bf48c4) vs `benchmarks/reference_v2/hero.png`:
**PSNR 42.51 dB**, max abs 46. `out/diff-B0-x10.png` (×10): differences follow edges only
(spokes, foliage, the bench slats). I inspected it and found no tile seams or block artifacts.
No device code changed in this task.

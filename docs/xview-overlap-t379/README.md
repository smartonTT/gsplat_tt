# Cross-view overlap (tasks #379, #386) — `GSPLAT_TT_XVIEW_OVERLAP=1`, default off

Code only so far: nothing here is measured yet. The A/B driver below produces the numbers.

## Problem

Between two views the device sits idle while the host finishes view N (waits for the
blend, reads or copies the image, returns to Python, builds view N+1's pose, crosses
pybind and enqueues view N+1's first program). #370 measured this gap at ~0.39 ms/view
on the p100a and ~0.8-0.9 ms/view on bh-30 (p150, slower host CPU and PCIe path).

## Idea

The first device program of a view, the fused project+pfwc, needs only the scene (already
resident) and the pose. If the host knows the next pose, it can queue view N+1's pfwc right
behind view N's blend, *before* it waits for and reads view N. The device then runs pfwc N+1
while the host reads and hands out image N.

## Design

Commit chain: 175eb6f6 (t379, C++ side) and this branch (t386: run.py, stubs, this doc).

**Python (render/run.py).** `CleanBackend.next_extrinsics` holds the next view's w2c (torch
4x4). The loops set it right before each render; `render_fused` takes it, clears it, and,
only with the flag on, passes it as `render_view(..., next_extrinsics=)` (float32,
C-contiguous, converted exactly like `extrinsics`, because the C++ key compares bitwise).
- Latency loop: view i is hinted view i+1; the last view and the warmup get no hint, so
  a 30-view run expects `xview_hits=29 xview_misses=0`.
- `--back-to-back`: the w2c list is built before the timed window; the last view of a
  pass is hinted the first view when another pass follows, so only the very last render
  has no hint (hits = passes*views - 1).
- `STAGES` (latency) and `B2B` lines print `xview_hits` / `xview_misses`.

**C++.**
- `render_view` (render.cpp) installs a one-shot hook via `blend_set_after_enqueue_hook`
  before `sort_and_bin_tt`; a scope guard clears it after the call, so it never outlives
  the view (and is re-installed per overflow-retry attempt).
- Frame end in the resident blend (`process_frame_mb_devcull_resident`, blend_device.cpp),
  with a hook:
  1. enqueue blend N (non-blocking);
  2. unless the output is pinned: enqueue a non-blocking `ReadShard` of `res_out` into
     `ctx.res_out_host`;
  3. `enqueue_record_event_to_host` → `frame_done`;
  4. run the hook: `enqueue_pfwc(next pose)` and `g_xview.set(key)`;
  5. `EventSynchronize(frame_done)` (waits for blend + read, not for pfwc N+1);
  6. copy the image to the caller's array (from `res_out_host`, or from the pinned slot),
     or with zero-copy hand out the pinned slot; pfwc N+1 runs on the device meanwhile.
  Without a hook the old path is unchanged (`Finish`, then a blocking read).
- `run_project` (view N+1) calls `g_xview.consume(key)`. The key (render/host/xview.h) is
  every pfwc input: scene pointers and N, the 16 w2c and 9 intrinsics floats (bitwise),
  min opacity, size, max radius, tile size, mb floor (pre-retry), cull flag. On a hit it
  skips `pfwc_tt`; on any mismatch (other pose, scene, size, floor, an overflow retry of
  view N) it runs pfwc again, so a wrong hint costs one wasted pfwc, never a wrong image.
  `device_shutdown` clears the prefetch.

**Ordering and buffer safety.** One in-order command queue: pfwc N+1 starts only after
blend N and the `res_out` read finish, so nothing view N still reads is overwritten under it.
pfwc N+1 writes the pfwc outputs, which view N no longer needs once its blend is done.
`res_out_host` is the read target until `frame_done`; if the hook throws, `pair_guard::
drain_on_throw` `Finish`es the queue first so the pending read never lands in freed memory.

**Image lifetime.** Copied output (default and `GSPLAT_TT_OUT_PINNED=1`): the caller's array is
filled after `frame_done`, as before. Zero-copy (`GSPLAT_TT_OUT_ZEROCOPY=1`): pfwc N+1 never
touches the pinned ring; view N+1's blend writes a slot the caller does not hold
(out_ring.h), so overlap does not change the zero-copy lifetime rules.

**Sync points per view with the flag on:** `EventSynchronize(frame_done)` in the blend,
plus the existing ones inside view N+1 (gather count read, sort). Nothing waits on pfwc
N+1 until view N+1's gather.

**Host-profile bookings under the flag.**
- `stagetimers xview`: host time of the hook (pfwc enqueue for N+1). It is subtracted from
  the blend's returned ms and added to the fused-path subtraction in render.cpp, so `sort`
  and `blend` exclude it.
- `d2h`: two spans, the read+event enqueue (before the hook) and the copy after the wait.
- `blend`: includes the `EventSynchronize` wait (Tracy zone `host_wait_blend_event`, was
  `host_finish_blend`).
- `hostprof::on_blend_device_done` fires after the event, `on_blend_readback_done` after
  the copy, unchanged otherwise.
- `project` of view N+1 drops by the pfwc enqueue on a hit; the device time of pfwc N+1
  is hidden behind view N's host tail. In the latency loop, per-view wall times shift
  work from view N+1 to view N; compare the mean, not single views.

## Expected gain (model, not measured)

Bounded by the smaller of (a) the host gap between views and (b) pfwc's device time:
up to ~0.3-0.4 ms/view on the p100a, up to ~0.8 ms/view on bh-30 if pfwc is at least that
long. Pinned output (#373) and zero-copy (#374) shrink the host tail, which helps the
base too; `xvpin` and `xvzc` show whether the overlap still adds on top of them.

## A/B driver (device, not run in #386)

- `docs/xview-overlap-t379/drive.sh <iter> <rev>` (Mac, repo root; start it with
  `ttp detach t379-ab -- ...`). It wraps everything in **one `ttp lock p100 --`**: ssh
  preflight, sync + build, build-ID stamp, an `xv` smoke run, 3 rotating rounds of
  `base | xv | xvpin | xvzc`, a per-arm mean (`out/summary.txt`), then the device screenshot.
- `docs/xview-overlap-t379/remote_time.sh <round> <arm>...` (on the box): untraced
  30-view bicycle run per arm; md5 of every view vs
  `docs/matblend-ready-t273/t289/md5-golden-906e0435.txt` and vs the round's base; for
  `xv*` arms `XVIEW_HITS_OK` only when `xview_hits == views-1` (else rc 7; md5 mismatch rc 6).

| arm   | env |
|-------|-----|
| base  | defaults |
| xv    | `GSPLAT_TT_XVIEW_OVERLAP=1` |
| xvpin | `GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_PINNED=1` |
| xvzc  | `GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_ZEROCOPY=1` |

**Screenshot step (required for any iteration).** `drive.sh` ends with
`opt/ttw/screenshot.sh` (same tree, `NO_SYNC=1`, env `GSPLAT_TT_XVIEW_OVERLAP=1`): the bicycle
hero rendered on the device → `opt/metal-screenshots/t379-xview-overlap/hero.png`,
`hero_diff10.png`, and a `SHOT` line with the PSNR vs `benchmarks/reference_v2/hero.png`
and the sweep md5. Someone must look at hero.png and the diff for tile artifacts (seams,
blocky or empty tiles); md5 and PSNR alone are not enough.

Restrictions for the device run: only the existing measurement reservation (no ird
reserve/release), never the viewer box, host-key checking always on, one device run at a
time under `ttp lock p100`. Always A/B all arms on the same box.

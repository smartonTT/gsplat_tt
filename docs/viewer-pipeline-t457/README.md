# Viewer frame pipelining (tasks #457, #458)

Goal (user, 2026-10-10): the live viewer at localhost:8091 shows page FPS >= 117, end to end,
with the same image as the bench.

## What changed (#457, d43350a5)

- `FastRenderer` latches the client pose: during a camera move it renders the pose latched on
  the previous frame and hints the newest client pose as the next view. The hint goes through
  `GaussianViewer._hint_next_pose` -> `backend.next_extrinsics` -> the C++ xview prefetch
  (`render/host/xview.h`), the same path as the bench's `_set_next_extr`. A static frame resets
  the latch. `GSPLAT_VIEWER_PIPELINE=0` turns it off.
- Letterboxing moved onto the sender thread, so the render thread only renders.
- HUD at 4 Hz with device, host and encode ms; a page FPS shim in the served HTML that
  reports decoded FPS to `/gsplat/pagestats` (204); the heartbeat logs it.

## Tests (#458)

`tests/unit/test_viewer_fast_renderer.py` and `tests/unit/test_viewer_session.py` in bh-30's
viewer venv: 13/13 pass. New tests cover next-pose latching (and `GSPLAT_VIEWER_PIPELINE=0`),
letterbox on the sender thread, the hint matching the bench extrinsics, shim injection,
`PageStats` parsing and staleness, and the pagestats 204.

## Device probe: 30 bench views, pipelined vs not (`opt/viewer/pipeline_probe.py`)

bh-30 (p150, eth 12x10), viewer stopped for the probe. 4 passes per mode, pass 0 is the check
pass, ms/view is the mean of passes 1-3. "Viewer path" is `FastRenderer.render_once("move")`
with a client whose camera walks the bench poses; "bench loop" is render/run.py's b2b loop
(`pipeline.render` + `_to_image`) in the same process.

| Path | Settings | Not pipelined | Pipelined (hinted) | xview hits / misses |
|---|---|---|---|---|
| Viewer path | viewer defaults | 10.208 ms (98.0 FPS) | 9.348 ms (107.0 FPS) | 119 / 0 |
| Bench loop | viewer defaults | 9.659 ms | 9.423 ms | 119 / 0 |
| Bench loop | bench (`Pipeline(.., 32, floor)`) | 9.599 ms | 9.266 ms (107.9 FPS) | 119 / 0 |
| Viewer path | bench | 9.777 ms | 9.303 ms (107.5 FPS) | 119 / 0 |

- Pipelining saves 0.86 ms/view on the viewer path (-8.4%), and the pipelined viewer path now
  matches the in-process bench loop (9.30 vs 9.27 ms).
- Same image: raw_md5 **d9a60a3c** in all 8 modes, every pass identical, pipelined or not. The
  viewer and bench settings are identical (floor, transmittance, min_opacity 1/255, max_radius 0,
  k_cap 3; K equal).
- The md5 is not the bench golden 39d84b28, but the plain bench loop in the same process gives
  the same d9a60a3c, so the difference is the viewer process's environment (its own tt-metal
  build and ETH overlay, or run.py env defaults the viewer env overrides), not pipelining.
  Follow-up filed. The same environment explains why the in-process floor is ~9.3 ms, not the
  bench's 7.7 ms (130 FPS).
- Hero (pipelined viewer path, bench settings): `hero.png`, `diff.png` (|diff| x4), PSNR
  **42.513 dB** against `benchmarks/reference_v2/hero.png`, max abs diff 46. That matches the
  42.51 dB of earlier p150 bench heroes. Checked by eye: no tile seams or block artifacts; the
  diff is fine edge detail (spokes, foliage) as before.
- Viewer downtime for the three probe runs: 44 s, 51 s, 56 s.

## Websocket sweep: page FPS end to end (`opt/viewer/ws_sweep.py`)

A viser 1.0.27 client on bh-30 sends the 30 bench poses at 240 Hz to the live viewer
(localhost:8080) for 20 s after 3 s warm-up and counts decoded JPEG frames.

| Viewer build | Decoded FPS | Interval p50 / p90 / p99 | JPEG | Decode |
|---|---|---|---|---|
| before, f1691d57 | 44.5 | 17.63 / 34.91 / 59.1 ms | 103 KB | 3.13 ms |
| after, 7137d17e (pipelining) | 43.0 | 17.72 / 35.11 / 35.69 ms | 104 KB | 3.16 ms |

Page FPS is **43**, far below the 117 target, and pipelining did not move it (only p99 got
better). The device is not the limit: the pipelined viewer path renders at 107 FPS (above).
The frame intervals sit at 17.7 ms and 35 ms, i.e. multiples of ~1/57 s. That fits viser's
`AsyncMessageBuffer`, which flushes once per `window_duration_sec` (1/60 s by default) and keeps
only the newest message per redundancy key. Every background image has the same key, so all
frames rendered within one window but the last are dropped before they reach the socket. The
page can never get much above ~57 FPS, and a window that misses a frame gives 35 ms.

Next lever: patch viser's flush window in `gsplat/viser_patches.py` (a ~1-2 ms window, or flush
at once when a background image is queued), then sweep again. Expected ceiling after that: the
pipelined device rate (~107 FPS in this viewer environment); reaching 117+ also needs the
viewer process to match the bench's 7.7 ms/view (see the md5 follow-up above).

Viewer downtime: probes 44 s, 51 s, 56 s; deploy restart a few seconds (stop + start).

## Task #459: viser flush patch and the real cap (two clients sharing the device)

- `gsplat/viser_patches.py` now pulses `AsyncMessageBuffer.flush()` on every
  `BackgroundImageMessage` (off with `GSPLAT_VIEWER_FRAME_FLUSH=0`); unit test
  `test_background_image_skips_message_window` fails without it.
- Deployed de3cc41b to bh-30 (hero 42.51 dB). Sweep: **43.0 FPS**, but intervals are now a
  steady 23.3 ms (p50/p90/p99 23.28/24.72/25.38) instead of 17.7/35 ms. So the window only
  quantized the frames; it was not what capped them.
- `ws_sweep.py` now records the viewer HUD. It reads `Sent: 82.3 FPS`, device render 11.3 ms
  at 1024x1024, encode 4.0 ms. 82 sent but 43 received per client: py-spy on the viewer shows
  **two render threads both busy** (the user's open page and the sweep client). The UI burst
  was viewer-wide, so one client's camera drag made every client render back to back, and
  the device was split between them. The camera message rate made no difference (60/120/240 Hz
  all gave 43 FPS).
- Fix in 68ae2556 (not yet deployed or measured): a camera move bursts only that client;
  settings changes still burst all clients. Test `test_camera_burst_renders_only_the_moving_client`.
  Expected: about 82 FPS for the moving page while another tab is idle.

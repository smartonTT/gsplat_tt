# t597: B4a fold in the live viewer (bh-30) — why it lowered page FPS, and the fix

Start: #587 (docs/viewer-t587/README.md): fold on cost page FPS in the viewer (115-117 vs 120.7 off),
device-call time per frame 7.98-8.07 ms on vs 7.69 off, while bench_loop gets faster with the fold.

## Finding
Under the fold, pfwc(N+1) and K2(N+1) are enqueued inside frame N's blend, so the device has no
queued work between the end of frame N and sort(N+1). Any host time the render thread spends before
it enqueues the next frame (building the hinted pose's w2c ~0.16 ms, rebuilding the camera state
~0.1 ms, K) is device idle time. With the fold off that host work hid under pfwc(N+1).
pipeline_probe (viewer stopped) shows the fold faster in every offline arm (p1: bench 8.05 vs 8.32,
viewer pattern 8.50 vs 8.72 ms/frame, hinted), so the device fold itself is fine; the live loss is
host work on the critical path. Not a pose-upload wait and not ETH dispatch serialization.

## Fix
1. 2004ac30: the render reuses the hinted pose's K and w2c (K cached per fov/size). Moved
   render_fn pose time 0.18 -> 0.004 ms but the hint grew 0.09 -> 0.16 ms: live A/B p1 still had
   fold off ahead (120.6-121.2 vs 117.8-118.5 page FPS).
2. 457b09a5: viser's camera-update callback (runs while the render thread waits on the device,
   GIL released) builds the pose's w2c and keeps the latest camera state; the render loop hints
   and renders from it. Same function, bit-identical w2c (tests/unit/test_viewer_fast_renderer.py
   test_camera_update_prepares_w2c_off_the_render_thread).
3. opt/viewer/viewer.sh: fold on by default in the viewer (VIEWER_B4A_FOLD=0 turns it off).

## Live A/B p2 (fix 2, bh-30, same 30-view pose stream, arms 0 1 0 1; out/p2/)
| arm | Mac page FPS (localhost:8091) | box rendered FPS | call ms | dev ms |
|---|---|---|---|---|
| fold off 1 | 120.35 | 120.30 | 8.27 | 8.09 |
| fold on 2  | 121.60 | 121.52 | 8.19 | 7.99 |
| fold off 3 | 119.91 | 119.83 | 8.30 | 8.13 |
| fold on 4  | 121.90 | 122.01 | 8.15 | 7.96 |

Fold on beats fold off by ~1.6 page FPS and beats #587's 120.7 off baseline.
Output: device path unchanged from p1, where pipeline_probe gave raw md5 c194c715, golden list
84695855 and hero PSNR 42.845 dB vs benchmarks/reference_v2/hero.png for both folds (out/p1/probe).

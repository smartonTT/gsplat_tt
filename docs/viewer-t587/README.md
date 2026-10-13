# t587: why the B4a fold lowers live-viewer page FPS (bh-30)

Result: fold stays OFF in the viewer (viewer.sh default unchanged; VIEWER_B4A_FOLD=1 opts in).
No fix that makes fold-on beat fold-off on the page was found within this task's budget.

## Page FPS (live viewer, Mac sweep through localhost:8091, 240 Hz pose stream, p2)

| run | fold | page decoded FPS | rendered FPS (box timeline) | call ms/frame | dev ms/frame (inside render_view) | host gap ms |
|---|---|---|---|---|---|---|
| fold0-1 | off | 120.72 | 120.72 | 8.135 | 7.688 | 0.043 |
| fold0-3 | off | 120.77 | 120.55 | 8.148 | 7.692 | 0.041 |
| fold1-2 | on  | 117.12 | 117.12 | 8.394 | 7.976 | 0.039 |
| fold1-4 | on  | 115.21 | 115.30 | 8.523 | 8.074 | 0.040 |

Timeline evidence: the gap between render calls is ~0.04 ms in both modes, no frames dropped,
every frame hinted (next-pose pfwc enqueued). Encode/send overlap fully (sent FPS = rendered FPS).
So the page is device-bound in both modes, and the fold makes the per-frame device wait
~0.3 ms LONGER in the viewer (7.69 -> 8.02 ms), the opposite of the bench (#577: 8.384 -> 8.119).

## pipeline_probe (same viewer box, viewer stopped for the probe; p1/p3)

- p1 bench_loop: fold makes it faster, 8.28 -> 8.09 ms/view; viewer path (FastRenderer.render_once) slower, 8.43 -> 8.54.
- p3 (`--short`, with GSPLAT_VIEWER_PROF section timing):

| arm | fold off ms/view | fold on ms/view |
|---|---|---|
| viewer (send) | 8.630 | 8.487 |
| viewer, no send (no letterbox/JPEG) | 8.303 | 8.511 |
| viewer (send), repeat | 8.465 | 8.376 |
| bench_loop | 9.103 | 8.939 |

render_fn host sections are the same in both modes: pose 0.18, hint 0.09, render_py 0.01, post 0.02 ms.
Stage split fold off: project 1.79 / sort 1.45 / blend 4.87; fold on: project 0.29 / sort 1.16 / blend 6.45.
Pass-to-pass noise is 0.2-0.3 ms (first passes often 9.1-9.7), so p3 cannot separate the modes;
the no-send arm agrees with the live result (fold on slower), the send arms do not.

Image: raw md5 c194c715 and golden list md5 84695855 in all arms, both folds; hero PSNR 42.845 dB vs
benchmarks/reference_v2/hero.png in both (identical output).

## Conclusion so far

- Not host gaps, not encode/send: host gap 0.04 ms, sends fully overlapped, host render_fn sections equal.
- The cost is device-side in the viewer's call pattern: with the fold, pfwc(N+1) is folded into blend(N),
  so frame N's blend carries the next frame's front stage. In the bench the next pose is known views
  ahead; in the viewer the hint arrives once per call, and the fold's longer blend lies on the critical
  path of every frame (dev 7.69 -> 8.02 ms) instead of overlapping the host's pose/hint work.
- Next lever (not done): per-frame device trace (tt-profile) of render_view with the fold in the viewer
  call pattern vs bench_loop, to see whether blend(N)+K2(N+1) waits on the hint/pose upload.

Raw outputs (probe logs, hero/diff PNGs) are archived in the run folder tt-project/state/runs/1669/t587-out/.

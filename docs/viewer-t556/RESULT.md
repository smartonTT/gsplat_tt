# Viewer deploy of best-iter-226 + page FPS through localhost:8091 (task #556)

2026-10-12, 16:28-16:55 PDT. Viewer box bh-30 (p150, tt_aus, IRD 138278, extended to 06:36 PDT 10-13).

## Deploy

`opt/ird/renew.sh viewer --rev best-iter-226`: deployed_sha 9026845799508627 (best-iter-226), over
ab8f841b (#532's adaptive-quality ladder, which was never merged to smarton/tt-project-opt, so
this deploy drops it; it only acts on a backlogged link). Build under nice/ionice, half the cores
(sync_remote.sh viewer path). No box change, launchd tunnel untouched.

## The B4a fold hangs the viewer on the p150 (ETH dispatch, 12x10 grid)

With the default (fold on), the first render that carries a next-pose hint hangs: watchdog
rc=86 after 5 s, render thread stuck inside `render_fused` (C++), then the board reads
0xffffffff over PCIe and needs `tt-smi -r`. Seen twice (16:40 and 16:46 PDT), on the first or
second client frame each time. Logs: `out/hang1_viewer_log.txt`, `out/hang1_watchdog.txt`,
`out/hang_pose_20261012T234026Z.json`, `out/hang2_viewer_log.txt`.

| Viewer config (bh-30, best-iter-226) | Hang? | Page FPS (23 s sweep) | Device ms |
|---|---|---|---|
| fold on, ETH dispatch 12x10 (default) | yes, first hinted frame | - | - |
| fold on, worker dispatch 11x10 | no; fold ran on 110/110 cores, 2533 hinted frames | 109.63 | 8.67 |
| **fold off, ETH dispatch 12x10 (now live)** | no | **118.79 median** | 8.00 |

The fold was validated only on the p100 with worker dispatch and an 11x10 grid
(docs/b4a-t554/out/run-r0-*.log: `card p100a: worker dispatch`, `compute grid 11x10`). The
hang is specific to ETH dispatch / the 12x10 grid (120 cores). Root cause not investigated here.

Live now: fold off via the new `VIEWER_B4A_FOLD=0` knob in `opt/viewer/viewer.sh` (forwards
`GSPLAT_TT_B4A_FOLD=0` to the viewer; supervisor restarts inherit it). A later
`renew.sh viewer` redeploy of a new tree does not pass it, so the fold comes back on and hangs
until the fold is fixed or gated off under ETH dispatch.

## Page FPS, fold off (live config), Mac -> localhost:8091

`opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:8091 --subprotocol viser-v1.0.27 --no-decode
--secs 23` (t532's sweep client; simplejpeg import made lazy for --no-decode):

| Run | Page FPS | Interval p50/p99 ms | Link MB/s | JPEG KB | Viewer sent FPS (median) | Device ms (median HUD) |
|---|---|---|---|---|---|---|
| 1 | 122.35 | 8.22 / 20.2 | 11.8 | 103 | 117.6 | 8.01 |
| 2 | 118.79 | 8.28 / 19.4 | 12.3 | 103 | 117.7 | 8.00 |
| 3 | 118.61 | 8.33 / 19.8 | 12.3 | 103 | 117.9 | 8.03 |
| after final restart | 118.77 | 8.27 / 19.8 | 12.3 | 103 | 117.7 | 8.02 |
| **median (runs 1-3)** | **118.79** | | | | | |

Target >= 104.5 (90% of bench 116.1 FPS): met, 1.02x bench. HUD: host 0.30 ms, encode 3.1 ms
(overlapped). For comparison, iter 224 (#532, before the ladder): 116.74 median, device 8.1-8.3
ms. No iter-225 viewer measurement is on record.

## Image

Viewer startup hero (fold off; a static render, which never folds): 42.51 dB vs
benchmarks/reference_v2/hero.png, same as the bench's iter-226 hero. Looked at by eye
(`hero_viewer.png`): no tile seams or artifacts.

## Health

localhost:8091 after the last restart: page 200, websocket 101 (viser-v1.0.27).

# Live viewer through localhost:8091 after the direct VPN tunnel (task #532)

2026-10-12, 10:55-11:15 PDT. Viewer box bh-30 (p150, IRD 138278).

## 1. Page FPS on the current build (ad647d8e = best-iter-224), new tunnel

The user moved the launchd tunnel (com.tt-project.tunnel.gsplat-viewer) off the yyz-ird hop to
bh-30.local.tenstorrent.com:46483 over the VPN (ping 51-65 ms, avg 57 ms; was ~310 ms HTTP RTT
over two hops). `opt/viewer/ws_sweep.py --url ws://127.0.0.1:8091 --subprotocol viser-v1.0.27
--no-decode --secs 23` from the Mac, 5 runs (`out/before-r*.out`):

| Run | Page FPS | Interval p50/p90/p99 ms | JPEG KB | Viewer sent FPS | Device ms |
|---|---|---|---|---|---|
| 1 | 95.04 | 7.6 / 21.6 / 51.4 | 103 | 119.4 | 8.07 |
| 2 | 116.66 | 7.7 / 16.2 / 26.0 | 103 | 116.1 | 8.22 |
| 3 | 116.76 | 8.4 / 15.0 / 25.5 | 103 | 115.3 | 8.26 |
| 4 | 116.97 | 8.0 / 15.2 / 27.6 | 103 | 116.8 | - |
| 5 | 116.74 | 7.9 / 15.3 / 27.1 | 104 | 124.4 | - |
| **median** | **116.74** | | | | |

On-box (no tunnel, #531): 116.9 FPS. Bench b2b at iter 224: 109.5 FPS (bh-33).
Target >= 117: **116.7 = 0.998x on-box, 1.07x bench, 0.3 FPS under 117**.

## 2. Which limit binds now

- Device + host: 8.1-8.3 ms device + 0.3 ms host per frame = ~8.6 ms = ~117 FPS. This is the
  cap; the page now gets 99.8% of it in 4 of 5 runs.
- Link: 116.7 FPS x 103 KB = ~12.0 MB/s through 8091 with no backlog. One ssh channel moves up
  to about window / RTT = 2 MB / 0.057 s = ~35 MB/s, so ~3x headroom. Not binding in steady state.
- Run 1 (95 FPS, viewer sent 119): a transient link dip; viser dropped the backed-up frames.
- JPEG bytes vs quality, hero view 1024x1024 4:2:0 (ttw-224 hero.png): q70 162 KB (35.1 dB),
  q50 121, q40 105 (32.4 dB), q35 98, q30 88, q25 79, q20 68 KB (29.9 dB). Lower quality
  alone saves at most ~35%.

## 3. Change (c3c4e55a)

`gsplat/nerfview_viewer.py`: send pacing plus adaptive moving-frame quality. When a frame is
ready while the previous one still sits unsent in viser's buffer (the link is the limit), the
sender waits (up to 50 ms) for viser to take it, sends the newest frame instead, and steps the
moving-frame JPEG quality down 5 (floor q20); after 60 clean frames it steps back up to q40.
Still frames (static/update) stay q70, byte-identical. `GSPLAT_VIEWER_ADAPTIVE=0` keeps q40.
With no backlog nothing changes, so steady-state frames are the same q40 as before.
Tests: `tests/unit/test_viewer_fast_renderer.py` (2 new; all 11 pass on bh-30's viewer venv).

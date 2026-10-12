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

### c3c4e55a measured (deployed 11:13 PDT), Mac via 8091, 23 s runs (`out/c3c4e55a-r*.out`)

| Run | Page FPS | Interval p50/p99 ms | JPEG KB | Viewer sent FPS | Device ms |
|---|---|---|---|---|---|
| 1 | 116.21 | 8.4 / 27.4 | 103 | 115.0 | 8.22 |
| 2 | 116.61 | 8.1 / 28.6 | 104 | 117.2 | 8.21 |
| 3 | 116.29 | 8.2 / 26.4 | 103 | 115.7 | 8.19 |
| 4 | 95.73 | 4.4 / 90.0 | 94 | 116.7 | 8.09 |
| 5 | 64.26 | 2.5 / 194.6 | 78 | 120.6 | 7.84 |
| **median** | **116.29** | | | | |

Runs 4-5 hit link dips (5-9 MB/s, bursty arrival: p50 2-4 ms, p99 90-195 ms stalls). The
quality steps engaged (median JPEG fell to 78 KB) but cannot shrink frames enough.

Controlled slow link: `ws_sweep.py --max-mbs N` caps the Mac client's read rate (2-frame
receive queue), so TCP backpressure through the tunnel reaches the viewer like a slow link:

| Cap | Page FPS | JPEG KB |
|---|---|---|
| 6 MB/s | 88.43 | 68 |
| 9 MB/s | 108.58 | 83 |

JPEG bytes vs quality and scale, hero frame as received (re-encoded from the q70 frame, so
slightly under the live sizes): full 1024x1024 q40 97 KB, q30 86, q20 65; 0.75x (768x768,
bilinear) q40 59, q30 50, q20 38 KB. At 0.75x even q40 is smaller than full-size q20 and
looks better, so the ladder should drop scale before it drops quality far.

## 4. Change (ab8f841b): moving-frame ladder with 0.75x scale

`MotionQuality.LADDER`: full q40 -> full q30 -> 0.75x q40 -> 0.75x q30 -> 0.75x q20, one rung
down per backlogged frame, one rung up after 60 clean frames. The page stretches 0.75x frames to
the canvas. Still frames unchanged (full size, q70). Also fixed an order-dependent thread-count
test flake (12/12 viewer tests pass 3x on bh-30's venv). Deployed 11:3x PDT with
`opt/ird/renew.sh viewer --rev ab8f841b` (deployed_sha ab8f841b1c15).

### ab8f841b measured, Mac via 8091, 23 s runs, 11:40-12:00 PDT (`out/ab8f841b-r*.out`)

| Run | Page FPS | Interval p50/p99 ms | JPEG KB | Viewer sent FPS | Device ms |
|---|---|---|---|---|---|
| 1 | 63.78 | 7.8 / 142.8 | 97 | 117.7 | 7.66 |
| 2 | 30.61 | 11.6 / 302.9 | 42 | 111.2 | 8.14 |
| 3 | 34.65 | 12.3 / 270.1 | 42 | 119.1 | 7.81 |
| 4 | 121.38 | 5.2 / 39.7 | 100 | 115.2 | 8.26 |
| 5 | 70.56 | 3.4 / 99.8 | 65 | 118.1 | 7.59 |
| 6 | 117.18 | 8.1 / 28.0 | 102 | 114.4 | 8.29 |
| 7 | 116.50 | 8.0 / 26.2 | 103 | 115.8 | 8.20 |
| 8 | 116.05 | 8.1 / 31.0 | 103 | 117.0 | 8.03 |
| 9 | 105.52 | 6.5 / 59.7 | 103 | 111.6 | 8.62 |
| **median** | **105.52** | | | | |

Runs 1-3 and 5 hit VPN dips: in runs 2-3 the page got 30-35 FPS of 42 KB frames (bottom rung),
i.e. the link carried only ~1.3-1.5 MB/s. No encoding short of ~11 KB frames fits 117 FPS in
that. With a clean link (runs 4, 6-8) the page gets 116-121 FPS, the device cap.

Controlled slow link (same method as section 3), A/B against c3c4e55a:

| Cap | c3c4e55a FPS (KB) | ab8f841b FPS (KB) |
|---|---|---|
| 6 MB/s | 88.43 (68) | **104.74** (54) |
| 9 MB/s | 108.58 (83) | **110.04** (89) |

Static hero (`--hero-out`, `out/hero_received_ab8f841b.jpg`): md5 acede79d, byte-identical to
c3c4e55a's and #531's checked capture (69.6 dB vs the ttw-224 hero, no seams by eye).

## 5. Conclusions and what limits it now

- Clean link: page FPS 116-121 = the device + host cap (~8.2 + 0.3 ms). Nothing more to gain in
  transport; more FPS needs a faster device render.
- Slow link: the ladder keeps 105 FPS at 6 MB/s (was 88). Below ~3 MB/s the VPN itself is the
  limit.
- Backlog shows late: pacing only sees viser's own buffer. Past it, the localhost socket to sshd
  and sshd's channel window (OpenSSH 2 MB, ~20 frames) absorb the backlog, so the ladder steps
  down only after that buffer fills; frames then arrive bursty (p99 100-300 ms in dips). A
  signal from the far end (the page acking frames, or sshd's TCP send queue on bh-30) would let
  the ladder react a full window earlier. Not done here: needs a client-side ack in viser or
  reading another process's socket state.
- A sustained link-rate probe with repeated curl fetches of the page through 8091 gave
  implausible numbers (176-332 MB/s, likely not real transfers) and was discarded.

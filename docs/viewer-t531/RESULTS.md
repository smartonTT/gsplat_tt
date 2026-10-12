# Live viewer at best-iter-224: page FPS end to end (task #531)

2026-10-12, 10:15-10:50 PDT.

## Deploy

- `opt/ird/renew.sh viewer --rev best-iter-224`: extended IRD job **138278** on **bh-30**
  (p150, tt_aus) to 14 h, new expiry **2026-10-13 00:36 PDT** (07:36Z). Box unchanged, so the
  launchd tunnel agent was not touched.
- `deployed_sha` **ad647d8e682082f6746683c5a8728d097bec68b0** (= best-iter-224), over feb0875b.
  Log: `out/renew.log`.
- Viewer self-test after restart: hero 1024x1024, single-view 9.45 ms median; viewer-path hero
  **42.51 dB** vs `benchmarks/reference_v2/hero.png` (same as the bench hero of iter 224).
- Tunnel: `http://localhost:8091/` 200, websocket upgrade 101, port 8091 held by the agent's own
  ssh (pid 7883, `com.tt-project.tunnel.gsplat-viewer`, `-L 8091:localhost:8080 bh-30`).

## Method

`opt/viewer/ws_sweep.py` (now runs on the Mac: `--subprotocol viser-v1.0.27`, no viser import;
`--no-decode`; `--hero-out`). The client connects like the page and sends the 30 bench poses
at 240 Hz (a fast continuous camera drag) for 23 s; FPS is counted over the last 20 s.
Frames are counted as they arrive (`--no-decode`): this Mac (load average ~19 from other
work) needed 40-170 ms per Python JPEG decode, which capped a decoding client at ~11 FPS
(`out/sweep-r*.out`). A browser decodes off its socket thread, so the arrival rate is the
page-side number. The user's tab was not connected during the runs (heartbeat clients=0),
so the sweep client had the device to itself.

## Results

Bench at best-iter-224: 9.131 ms/view b2b = **109.5 FPS** (bh-33, iter 224).

| Path | Runs (FPS) | Median | Ratio to bench | Interval p50/p90/p99 | Link rate |
|---|---|---|---|---|---|
| **Mac via localhost:8091 (what the page gets)** | 91.25, 94.03, 69.58 | **91.3** | **0.83** | 1.2 / 32 / 89-148 ms (bursty) | 6.9-9.4 MB/s |
| On bh-30, ws://127.0.0.1:8080 (no tunnel) | 116.93, 116.85, 116.89 | **116.9** | 1.07 | 8.6 / 9.7 / 10.1 ms | 11.8 MB/s |

Viewer HUD during the Mac runs (medians over all samples): sent **116.4 FPS**, device render
**8.06 ms**, host 0.30 ms, encode (letterbox + JPEG q40 4:2:0) **3.07 ms**, end-to-end
latency 11.8 ms. JPEG ~100 KB per frame while moving.

Frame-time split (sustained, per frame):
- Device 8.1 ms + host 0.3 ms on the render thread = the 8.6 ms frame interval on the box
  (116.9 FPS). Encode (3.1 ms) runs on the sender thread and overlaps the next frame's
  device work, so it is not on the throughput path.
- Transport: the viewer sends ~116 FPS, the Mac receives 91 (median). The missing ~25 frames/s
  are dropped in viser's send buffer (newest frame per key wins) when the socket backs up,
  which is why arrivals come in clumps (p50 1.2 ms, p90 32 ms).

HUD device render (8.1 ms) is below the bench's 9.13 ms because bh-30 is a p150 (eth
dispatch 12x10) and the bench number is from bh-33; the viewer also skips stale poses.

## Bottleneck (page FPS < 90% of bench)

**The ssh tunnel to the Mac, not the device or the encoder.** On the box the same viewer
delivers 116.9 FPS (steady 8.6 ms intervals); through localhost:8091 it delivers 91. The
tunnel's round trip is ~310 ms (HTTP time-to-first-byte through 8091: 0.31-0.35 s, vs 0.6 ms
on the box) over two ssh hops (Mac -> yyz-ird -> bh-30). A single ssh channel moves at most
about window / RTT (OpenSSH's 2 MB channel window / ~0.3 s = ~7 MB/s per hop), which fits the
measured 6.9-9.4 MB/s. At ~100 KB per frame that caps the page near 70-94 FPS. Run 3's 69.6 FPS
shows how much it varies with the link.

Fix to try next (follow-up, not done here): cut bytes per moving frame so ~117 FPS fits in
~7 MB/s, i.e. <= ~60 KB/frame: adaptive motion encoding (lower JPEG quality or a 0.75x
downscale only while the camera moves, sized from the measured send backlog), keeping the
static pass at full size q70 so the still image is unchanged. Pace sends to the link rate
instead of letting viser drop frames, so the frames that do arrive are evenly spaced.

## Image check

- `out/hero_received.jpg`: the hero view as received on the Mac through 8091 (the viewer's
  static pass, JPEG q70 4:2:0, 1024x1024, 166,033 bytes).
- Against the bench's device hero at iter 224 (`opt/metal-screenshots/ttw-224/hero.png`)
  encoded the same way on the Mac (q70 4:2:0 fastdct, 166,068 bytes): PSNR **69.6 dB**, max
  abs diff 8. The small difference is the two libjpeg builds; the pixels before encode match.
- Against `benchmarks/reference_v2/hero.png`: 34.66 dB for the received JPEG, the same as the
  bench hero after the same JPEG encode (34.66 dB). The JPEG costs the 42.51 -> 34.66 dB; the
  pre-encode viewer image is 42.51 dB (viewer self-test above), >= the 42.4 dB bar.
- Looked at by eye: no tile seams, block edges or empty tiles; only normal JPEG softening.

## Files

`out/sweep-nd-r{1,2,3}.out` (Mac, arrival rate), `out/sweep-r{1,2,3}.out` (Mac, Python decode
in the loop), `out/sweep-box.out` (on bh-30), `out/hero_received.jpg`, `out/renew.log`.

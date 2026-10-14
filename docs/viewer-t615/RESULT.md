# t615: live viewer, alternate-frame rendering (AFR) on 2 chips — page FPS

Host: bh-50 (2x p150, not bh-30), IRD job 133889, reserved 17:35 PDT, released 17:46 PDT 2026-10-13
(earlier run 4: IRD 133878, released 17:31 PDT). Viewer code: #620 AFR viewer at sha 4c9bfc26,
self-contained in bh-50:/localdev/smarton/nosync/t582/viewer, worker dispatch. Page FPS measured
through a temporary Mac tunnel (localhost:8095 via yyz-ird, strict host-key checking) with the
camera moving continuously (240 Hz pose stream, 23 s per sweep, 3 sweeps per arm).
bh-30, IRD 138278, the launchd agent and localhost:8091 were not touched.

| arm | page FPS (3 sweeps) | mean | device render (HUD) | selftest | bench b2b (#582) |
|---|---|---|---|---|---|
| 1 chip (board 0) | 70.6 / 75.4 / 67.7 | 71.2 | ~12.2 ms | 14.24 ms/frame (70.2 FPS) | 8.395 ms/view (119 FPS) |
| **2 chips** (AFR, boards 0,1) | 115.9 / 116.7 / 107.2 | 113.3 | ~10.7 ms per chip | 5.44 ms median interval (183.8 FPS) | 4.226 ms/view (236.6 FPS) |

2 chips gives 1.59x page FPS over 1 chip on the same host (run 4's 1-chip sweeps: 72.9/70.4/74.5).
Both page numbers sit well below the bench: the viewer path (host 2.4-4 ms, encode ~1.7 ms, link
~3 MB/s over the tunnel) and per-frame device time (~10.7-12.2 ms vs 8.4 ms bench) limit it.

Image check (hero view, vs benchmarks/reference_v2/hero.png): 1 chip 42.85 dB; AFR chip 0 42.85 dB,
md5 51d25cd7 identical to the 1-chip viewer; AFR chip 1 42.85 dB but NOT bit-identical: 257 of
1,048,576 pixels differ (max 13/255), scattered with no tile structure (run5/afr2-diff_chip0_chip1.png,
run5/check_strip.png; looked at, no tile artifacts). Likely cause: the two boards carry different
firmware (18.12.1 vs 19.7.0) and/or core grids; not resolved in this reservation.

Board-1-alone arm failed: `pfwc_num_cores()` raised "Proceeding could lead to undefined behavior"
when board 1 was opened on its own (run5/viewer-fail.log), so chip 1 could not be checked as a
single-chip viewer. Full logs/heroes: tt-project/state/archive/worktrees/t615/.

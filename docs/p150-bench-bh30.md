# p150 bench of iter 207 on bh-30 (task #346, 2026-10-07)

The project's first p150 number. This is a diagnostic run of the final iteration, not a new iteration:
no iters.jsonl row and no tag. bh-30 is the user's viewer box and was the only p150 we could get
(IRD reserve on bh-50 crashes). The user approved the run on 2026-10-07. It ran under the existing
viewer reservation (IRD job 135630). Nothing was reserved, extended or released.

## Result

| | ms/view (mean) | median | FPS | source |
|---|---:|---:|---:|---|
| **iter 207 on p150b (bh-30)** | **12.906** (rounds 12.934 / 12.878) | 12.9 | **77.5** | this run, measured |
| iter 207 on p100a (yyzo-bh-04) | 10.907 (10.923 / 10.861 / 10.937) | 11.0-11.1 | 91.7 | iters.jsonl iter 207, measured |
| GPU G1: INRIA 3DGS, RTX A6000, bicycle 1080p | 10.75 | | 93.0 | **published, not measured** |

- p150 / G1 = **1.20x** (2.16 ms/view slower than G1). p150 / p100a = 1.18x.
- Spread: the two rounds differ by 0.056 ms (0.4 %). Per-view times in both rounds: min 11.6-11.7, max 14.5.
- Quality from the same run: md5 **906e0435 on 30/30 views** in both rounds (the per-view list is
  identical to `docs/matblend-ready-t273/t289/md5-golden-906e0435.txt`). Device hero `hero.png`
  vs `benchmarks/reference_v2/hero.png`: **42.51 dB**. Separate badge: 8-bit golden match
  (`tests/fixtures/hero/hero_golden_8bit.png`) is true, max 0 LSB. The p150 hero is bit-identical
  to the iter-207 p100a hero (`opt/metal-screenshots/ttw-207/hero.png`).
- Seam check: I looked at `hero.png` and `hero_diff10.png` (|hero - reference_v2| x 10). No tile
  or microblock seams, no blocky or empty tiles, no stripes or colour shifts. The diff shows only
  thin high-gradient edges (spokes, frame, bench slats, foliage), same as iter 207 on the p100a.

## Why the p150 is slower than the p100a here

Per-stage means over 30 views (`STAGES` / `SORT_STAGES` lines, round 1 on each box):

| stage | p150b bh-30 | p100a bh-04 | delta |
|---|---:|---:|---:|
| project | 2.974 | 3.041 | -0.07 |
| sort | 3.060 | 0.881 | **+2.18** |
| - sort bin_emit | 2.418 | 0.571 | **+1.85** |
| - sort publish_host | 0.333 | 0.169 | +0.16 |
| blend (fused mat+blend) | 6.273 | 6.711 | **-0.44** |
| d2h | 0.552 | 0.243 | +0.31 |
| frame | 12.934 | 10.923 | +2.01 |

- The device-heavy stages are as fast or faster on the p150: blend is 0.44 ms faster and project
  0.07 ms faster. Both boxes run the same compute grid (11x10 = 110 cores, `[DEV]` line in the logs);
  the p150 has all 8 DRAM banks (DRAM harvest mask 0x0) where the p100a has 7 (0x20).
- The loss is in sort `bin_emit` (+1.85 ms, 4.2x) plus host-side stages (publish_host 2x, d2h 2.3x).
- The two boxes differ in more than the chip. Host CPU: AMD EPYC 7352 (Zen 2, 24 cores) on bh-30
  vs Ryzen 5 7600X (Zen 4) on bh-04. Firmware: 19.13.2 vs 19.12.0. tt-metal: both 437bc366, but
  bh-30 runs the viewer's own build (`/localdev/smarton/viewer/tt-metal`, built by
  `opt/viewer/setup_box.sh`) and bh-04 runs a Tracy-enabled build. Also, `mutagen-agent`
  (devsync) used about 130 % CPU on bh-30 just after the run.
- This run does not show whether `bin_emit` is slower because of the host or the chip. It needs a
  per-view Tracy or `GSPLAT_PER_VIEW_STAGES=1` run on the p150 (see follow-ups in the hand-off).
- Viewer cross-check: the viewer's own selftest on bh-30 gives 13.12 ms median (746e9e9d), in line.

## Setup and exact command

- Tree: tag `best-iter-207` = commit b28a6c76, synced with `opt/sync_remote.sh` into
  `bh-30:/localdev/smarton/p150bench/tree` (the `SHA` file there holds the annotated tag object 03ad3530).
  It links the viewer's venv and scenes read-only and uses `TT_METAL_HOME=/localdev/smarton/viewer/tt-metal` (437bc366).
  The JIT cache is a copy of the viewer's cache in `/localdev/smarton/p150bench/cache`. bh-30's
  `/localdev/smarton/gstt2` was not used and `/localdev/smarton/viewer` was not changed.
- Bench command, same as the iter-207 base arm (`docs/iter207-t315/remote_time.sh`), no env overrides:
  `TT_METAL_CACHE_RENDER=$P/cache timeout 330 python3 render/run.py --no-ref --iter-dir t346-rN --dump-views t346-rN-dump`
  with `TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1`. Warmup is the usual hero render (16.3 s in r1 with JIT, 1.7 s in r2).
- Scripts: `docs/p150-bench-t346/drive.sh` (Mac: stop viewer, run, restart viewer, fetch) and
  `docs/p150-bench-t346/bench_remote.sh` (box). Logs and md5 lists: `docs/p150-bench-t346/out/`.
  Machine-readable result: `docs/p150-bench-t346/p150.json` (feeds the REPORT.html line).

## Viewer downtime

- Before: viewer pid 6569 (started 13:33 UTC, opt tip 746e9e9d, port 8080). The user reported it
  stopped rendering at 19:44:20 UTC. This task's build ran 19:47:27-19:48:10 UTC, after that stall
  (sync_remote.sh at its default `-j 16` on 24 cores, normal priority, about 45 s; the viewer was still up then).
- Stopped 19:48:39 UTC (`opt/viewer/viewer.sh stop`). Bench 19:49:15-19:50:05. Restarted 19:50:05
  (`opt/viewer/viewer.sh start`, same tree, port and env). Downtime about 1 min 26 s.
- After: pid 10623, selftest median 13.12 ms (76.2 FPS), hero 42.51 dB vs reference_v2, READY on 8080.
  From the Mac, http://localhost:8091 gives 200 and a websocket upgrade gives 101 through
  com.tt-project.tunnel.gsplat-viewer. A test client on the box (`docs/p150-bench-t346/ws_render_check.py`)
  sent one camera pose and got 168 frame-sized messages in 10 s, and viewer.log grew with new frames.

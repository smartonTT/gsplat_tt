# t470: b2b A/B of the #433 chunk-cull arms (off / ro / cull) on p150

Headline metric: back-to-back ms/frame (`render/run.py --back-to-back`, 30 bicycle views, 1024x1024).

## Setup

- Host: bh-30 (p150b, aiclk 1350), the only free p150, used under the 2026-10-07 viewer exception:
  viewer stopped 13:58:39Z, bench 13:59:14Z-14:02Z, viewer restarted by the bench's exit trap
  (selftest 10.22 ms, localhost:8091 -> 200 at 14:02:59Z).
- Tree: tree470 = opt tip 13ce253e device code (equal to this branch's HEAD), env as
  docs/b2b-gap/probe464.sh run(): `TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1`.
- Arms: off (default); ro = `GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0` (Morton reorder, all
  chunks kept); cull = `GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_LOG=1`.
- One smoke run per arm (kernel compile), then 3 rounds with the arm order rotated
  (off ro cull / ro cull off / cull off ro). Each run: 1 check pass + 3 measured passes;
  `--dump-views` writes PNGs from pass 0 only, outside the measured passes.
- Scripts: drive470.sh (Mac, locks), bench470.sh (remote), run_t470.py. Raw logs in out/.

## Results (ms/frame b2b, 3 measured passes per run)

| run | off | ro | cull |
|---|---|---|---|
| r1 | 10.081, 9.614, 9.477 (med 9.614) | 10.799, 10.431, 10.182 (med 10.431) | 10.181, 9.870, 9.671 (med 9.870) |
| r2 | 9.914, 9.561, 9.286 (med 9.561) | 10.694, 10.395, 10.165 (med 10.395) | 10.059, 10.088, 9.649 (med 10.059) |
| r3 | 10.117, 9.494, 9.295 (med 9.494) | 10.678, 10.320, 10.181 (med 10.320) | 10.036, 9.709, 9.750 (med 9.750) |
| **median of 9** | **9.561** | **10.395** | **9.870** |
| mean / min / max | 9.649 / 9.286 / 10.117 | 10.427 / 10.165 / 10.799 | 9.890 / 9.649 / 10.181 |

All 9 runs: passes identical to pass 0 (`identical_across_passes=yes`). Every ro pass is slower
than every off and cull pass. Single off and cull passes overlap (the first pass of each run is the
slowest), but off has the lower median in every round (by 0.26, 0.50 and 0.26 ms).

- pfwc saving (ro - cull): **0.525 ms/frame**
- reorder cost (ro - off): **0.834 ms/frame**
- net cull vs today (cull - off): **+0.309 ms/frame (slower)**

## Stage breakdown (B2B_STAGES, median of the 9 measured passes, ms)

| stage | off | ro | cull | ro-off | cull-ro |
|---|---|---|---|---|---|
| period | 9.561 | 10.395 | 9.870 | +0.834 | -0.525 |
| project (gather_wait) | 2.616 (2.613) | 2.952 (2.949) | 2.379 (2.376) | +0.336 | -0.573 |
| sort | 1.535 | 1.509 | 1.581 | -0.026 | +0.072 |
| sort_bin_emit | 0.435 | 0.439 | 0.334 | +0.004 | -0.105 |
| blend | 5.207 | 5.669 | 5.503 | +0.462 | -0.166 |
| xview | 0.117 | 0.173 | 0.320 | +0.056 | +0.147 |
| py_resid | 0.058 | 0.058 | 0.058 | 0 | 0 |

The cull keeps about 60% of the 5989 chunks per view (e.g. 3494-3704 of 5989 on the logged views).
The reorder costs time in two places: pfwc is 0.34 ms slower with Morton gids, and blend is
0.46 ms slower. The cull wins back 0.57 ms of pfwc (0.24 ms below off) and 0.17 ms of blend.
The cull arm also runs with `GSPLAT_TT_CHUNK_LOG=1` (as the spec says). Its xview is 0.15 ms
higher than ro; part of that may be the per-view log line, so the true cull cost may be a little lower.

## md5 (30 views, dump from pass 0 of each b2b run)

| arm | r1 | r2 | r3 | golden 39d84b28 |
|---|---|---|---|---|
| off | 39d84b28 | 39d84b28 | 39d84b28 | OK (30/30) |
| ro | dd46d8cb | dd46d8cb | dd46d8cb | differs (tie order), stable |
| cull | 5dbc0cc9 | 5dbc0cc9 | 5dbc0cc9 | differs (tie order), stable |

Each arm matches itself across all three rounds. Hero images (out/hero-<arm>.png): PSNR against
benchmarks/reference_v2/hero.png is 42.51 dB for all three arms. ro and cull differ from off on
254 and 252 pixels, by at most 4/255 per channel (tie order). No config is proposed to keep, so no
iteration hero/diff card was made.

## Gate verdict

ro - cull = 0.525 ms/frame, above the 0.3 ms gate, so the chunk-cull lever stays open. But as it
stands the cull loses to today's default by 0.31 ms/frame b2b. The reorder it needs costs 0.83 ms
(pfwc +0.34, blend +0.46), more than the 0.53 ms the cull saves. Do not turn it on.

Follow-up (proposed): remove the reorder cost and keep the cull.
1. Sort tie-break on the original gid, so records with equal depth come out in today's order
   whatever the gid numbering. That should bring back the golden md5 39d84b28 and any blend cost
   that comes from the tie order.
2. Find out why blend and pfwc slow with Morton gids (blend +0.46, pfwc +0.34 ms b2b here; #433
   saw +0.72 ms blend in latency mode). Likely suspects are the record address pattern in blend's
   reads and the pfwc gather locality. Profile per-core blend time off vs ro.
If both reorder costs go away, the cull's pfwc saving vs off (0.24 ms of project, plus 0.10 ms of
sort_bin_emit) gives about 0.3 ms/frame, around 3% at 9.56 ms. Any config that is kept needs a
device hero.png, a diff against benchmarks/reference_v2/hero.png, PSNR and a visual check for tile
artifacts.

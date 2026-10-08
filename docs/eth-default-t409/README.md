# Task #409: ETH dispatch (12x10) is the p150 default (iteration 214)

## What changed
- `render/eth_default.py` runs before the device opens (called from `render/run.py`):
  - on a p150/p300 it makes or reuses the `opt/eth` overlay (`<repo>/tmp/ttm-eth12`, own JIT
    cache `-cache/{prod,render}`) and sets `GSPLAT_TT_DISPATCH=eth`;
  - on a p100a, on overlay failure, or with `GSPLAT_TT_DISPATCH=worker` it runs worker and logs why.
- `render/host/dispatch_select.h`: unset/`auto` resolves to eth only on an ETH card *and* when
  `TT_METAL_RUNTIME_ROOT` has the overlay marker `.gsplat-eth-overlay`; otherwise worker. Other
  entry points (the viewer) therefore stay on worker unless they run with an overlay.
- `opt/md5_golden.py`: the md5 check picks the golden by the grid in the `[DEV]` log line
  (11x10 `906e0435`, 12x10 `39d84b28`) and fails on a mismatch or an unknown grid.
- The C++ log reads `dispatch eth (eth (forced))` because the Python launcher already resolved
  the mode; the `[eth] card p150b: eth dispatch, overlay ...` line above it gives the reason.

Tests: `tests/unit/run_cpp.sh tests/unit/test_dispatch_select.cpp`, `pytest tests/test_eth_default.py`,
`python3 opt/test_md5_golden.py`.

## A/B on bh-30 (p150b, viewer reservation IRD job 135630), build 50900f21
Bicycle, 30 views, 1024x1024, zero-copy output on in both arms, 3 alternating rounds
(`drive409.sh` -> `bench409.sh`; logs in `out/`).

| round | worker 11x10 (ms/view) | eth default 12x10 (ms/view) |
|---|---|---|
| r1 (W,D) | 8.839 | 8.427 |
| r2 (D,W) | 8.857 | 8.450 |
| r3 (W,D) | 8.906 | 8.360 |
| mean | **8.867** (112.8 FPS) | **8.412** (118.9 FPS) |

Delta -0.455 ms/view (-5.1%). Stage means worker -> eth: project 1.136 -> 1.085, sort
1.060 -> 1.173 (bin_emit noise), blend 6.448 -> 5.895, d2h 0.010 -> 0.010, xview 0.122 -> 0.146.

GPU reference 10.75 ms/view is **published, not measured**: eth default is 1.28x faster,
worker 1.21x.

md5: worker `906e0435` 30/30 in 3/3 rounds (11x10 golden), eth `39d84b28` 30/30 in 3/3 rounds
(12x10 golden), all `MD5_GOLDEN_OK`. MATBLEND_PROGRAM fz=1 zerocopy=1, XVIEW_HITS 29/30 in all runs.

## Screenshot
`opt/metal-screenshots/ttw-214/hero.png` (= `out/hero-r1-D.png`) and `hero_diff10.png`:
PSNR vs `benchmarks/reference_v2/hero.png` 42.51 dB (worker hero 42.51 dB). Pixel-identical to
the t397 eth hero; saved as golden `tests/fixtures/hero/hero_golden_8bit_12x10.png`. Visual
check: no seams, no band along the 12th core column; tile-boundary pixel columns mean |err|
0.873 vs interior 0.875.

## Viewer
Stopped 2026-10-08T02:19:21Z, restarted 02:22:20Z, READY and localhost:8091 -> 200 at
02:22:29Z. The viewer stays on worker dispatch (its tree may not get an overlay).

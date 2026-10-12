# t554 result: B4a fold (K2(N+1) in fused mat+blend(N)) default on — KEEP, iter 226

Commit under test 69ee8b28 (624032db #552 fold + default on; `GSPLAT_TT_B4A_FOLD=0` disables).
Board bh-33 (Blackhole p100, measurement reservation), one sync + build, all runs inside
`ttp lock p100` (docs/b4a-t554/drive.sh). Code review notes: REVIEW.md (no blocking finding).

## Back-to-back (headline), render/run.py --back-to-back, 30 bicycle views 1024x1024

Run medians of 20 measured passes (ms/view):

| round | order   | on (default) | off (FOLD=0) |
|-------|---------|--------------|--------------|
| warm  | off,on  | 8.593        | 8.989        |
| 1     | off,on  | 8.601        | 8.990        |
| 2     | on,off  | 8.615        | 8.988        |
| 3     | off,on  | 8.645        | 8.991        |
| **median r1-3** |  | **8.615** | **8.990** |

-0.375 ms/view (-4.2%), 111.2 -> 116.1 FPS. Gate (>= 0.1 ms) passed. The on arm shows a few
slow passes (up to 8.887) that the off arm does not; the run medians are still 0.35-0.39 ms apart.

B2B_STAGES medians (on vs off): project 0.222 vs 2.099, sort 0.865 vs 1.003, blend 7.255 vs 5.700
(K2 time moves from the project stage into blend, where it overlaps the blend on BRISC).

Latency (secondary, no --dump-views, round 1 only): 8.644 vs 8.958 ms/view.

## Correctness

- Sweep md5 over 30 views (default config): 906e0435, 0 of 30 views differ.
- GSPLAT_TT_B4A_VERIFY=3 (fold vs real K2 on the first 3 folded views), latency and b2b: 3/3 OK.
- Kill switch: the FOLD=0 arm never logged a `[B4A] fold` line (no FOLD_NOT_OFF).
- Hero (default config, 69ee8b28): opt/metal-screenshots/ttw-226/hero.png, diff x10
  hero_diff10.png, PSNR 42.51 dB vs benchmarks/reference_v2/hero.png; bit-exact vs
  tests/fixtures/hero/hero_golden_8bit.png (max 0 LSB). Looked at both: no tile seams, blocky or
  empty tiles; diff only on thin edges (spokes, frame, bench slats, foliage), same as iter 225.

Measured on p100 only, not on p150. Published GPU reference 10.75 ms/view (published, not measured).

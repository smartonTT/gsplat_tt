# t536: B3 lever 4, static opacity prune (order-keeping) — A/B on bh-33

Measured 2026-10-12 11:52-11:56 PDT on bh-33 (Blackhole p100, measurement reservation),
tree 186d6f3b (best-iter-224 ad647d8e + this change), one build, under `ttp lock p100`.

## What changed
- `render/opacity_prune.py`: at load, drop gaussians with opacity < min_opacity (bicycle: keeps
  5,559,909 of 6,131,954, -572,045, -9.3%). Survivors are placed so pfwc's strided-deal sequence
  order lists them in the old relative order, so storage ids (tie key, record address) map
  monotonically and the image is unchanged bit for bit.
- `GSPLAT_TT_OPACITY_PRUNE` default on; `=0` restores the old path. New pybind `pfwc_num_cores`.
- The ply-order chunk frustum cull was not built: docs/pfwc-chunk-cull-t466 measured 0 skippable
  chunks in ply order on all 30 views, even at chunk size 64 (ply order is spatially random).

## Result (render/run.py --back-to-back, 30 bicycle views 1024x1024, 1 check + 1 warm-up + 20 passes)
| arm | round 1 | round 2 | round 3 | median |
|---|---|---|---|---|
| default (prune on) | 8.998 | 8.999 | 8.998 | **8.998** |
| off (=0, = best-iter-224) | 9.132 | 9.129 | 9.131 | 9.131 |

Warm-up round: 9.001 vs 9.131. Delta **-0.133 ms/view (-1.5%)**, 109.5 -> 111.1 FPS.
B2B_STAGES medians (def vs off): project 2.103 vs 2.231, sort 0.952 vs 0.979, blend 5.758 vs 5.740.
Latency mode (no dumps, secondary): 8.888/8.876/8.889 vs 9.019/9.013/9.026.

Inside the 0.2-0.4 realistic band's lower edge; above the 0.1 ms keep bar.

## Correctness
- `--dump-views` sweep of the default arm: md5 906e0435 on 30/30 views (0 differ).
- Device hero (default arm): bit-exact vs tests/fixtures/hero/hero_golden_8bit.png,
  PSNR vs benchmarks/reference_v2/hero.png 42.51 dB
  (opt/metal-screenshots/ttw-225/hero.png, hero_diff10.png).
- Looked at hero and diff x10: no tile seams, blocky or empty tiles, stripes or colour shifts;
  diff only on thin edges (spokes, frame, bench slats, foliage), same as iter 224.
- Unit test tests/unit/test_opacity_prune.py (sequence order vs a SeqMap reference).

Raw summaries in out/*.out (drivers: drive.sh, remote_ab.sh).

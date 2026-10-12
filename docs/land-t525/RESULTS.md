# t525: landing A/B for #519 MAT_EARLY_CULL (default on), iteration 224

Tree c3269ce7 (ttp/t513 f0b2e31e + default on), bh-33 (Blackhole p100, measurement reservation),
one build, 2026-10-12 09:41-09:46 PDT. Driver: `docs/land-t525/drive.sh` under `ttp lock p100`.
Arms: def = GSPLAT_TT_MAT_EARLY_CULL unset (on), off = GSPLAT_TT_MAT_EARLY_CULL=0.
render/run.py --back-to-back defaults (1 check + 1 warm-up + 20 measured passes), 30 bicycle views 1024x1024.

| round | def b2b ms/view | off b2b ms/view | def latency | off latency |
|---|---|---|---|---|
| r0 (warm-up, discarded) | 9.130 | 9.367 | - | - |
| r1 | 9.130 | 9.365 | 9.000 | 9.260 |
| r2 | 9.131 | 9.369 | 9.004 | 9.259 |
| r3 | 9.131 | 9.367 | 9.010 | 9.263 |
| **median** | **9.131** | **9.367** | 9.004 | 9.260 |

B2B: -0.236 ms/view (-2.5%), 106.8 -> 109.5 FPS. Latency (no dumps, secondary): -0.256 ms/view.
B2B_STAGES medians over 66 passes (def vs off): project 2.233 vs 2.232, sort 0.998 vs 0.893, blend 5.713 vs 6.056.
The culling moves into the mat step (sort +0.105) and the blend shrinks by 0.343.

Image: 30-view sweep md5 906e0435 (0 of 30 views differ), b2b raw_md5 5a438f5a in both arms, identical across passes.
Hero (default arm) bit-exact vs hero_golden_8bit, PSNR vs benchmarks/reference_v2/hero.png 42.51 dB
(opt/metal-screenshots/ttw-224/). Visual check: no tile seams, blocky or empty tiles, stripes or colour
shifts; the x10 diff is dark except thin edges (spokes, frame, bench slats, foliage, rooftops), same as iter 223.
Keep gate (>= 0.1 ms/view b2b, md5 unchanged, no seams): pass.

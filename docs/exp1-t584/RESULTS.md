# t584: degree-1 minimax exp in the blend (BLEND_EXP1) — device A/B

Same tree 2e0d8462, GSPLAT_TT_BLEND_EXP1=1 (new) vs 0 (base), separate JIT caches, bh-33 p100
(measurement box), 2026-10-12 22:52-22:58 PDT. Drive: `drive.sh`; logs in `out/`.

| | base (EXP1=0) | new (EXP1=1) |
|---|---|---|
| b2b ms/view, run medians r1-r4 (ABBA) | 8.588, 8.602, 8.590, 8.608 | 8.460, 8.452, 8.455, 8.453 |
| median | 8.596 | 8.454 (-0.142, -1.6%) |
| 11x10 sweep md5 | 906e0435 (golden) | 077caaba (new golden) |
| hero PSNR vs reference_v2 | 42.51 dB | 42.85 dB (model: 42.86) |

Visual check of `opt/metal-screenshots/ttw-227/hero.png` and `hero_diff10.png`: no tile seams,
banding or blocky tiles; diff only on thin edges as before. Verdict: KEEP (iter 227), default on.
The 12x10 (eth, p150) golden with EXP1 on is not captured yet; `opt/md5_golden.py` reports "no golden".

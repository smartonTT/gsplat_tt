# t523: combined b2b A/B of the stacked small levers

Box: bh-33 (IRD 138447, card p100a, aiclk 1350), 2026-10-12 09:52-09:58 PDT, head cf0c1258
(best-iter-223 + merges of t514 GAP_CUT, t517 PFWC_DEAL, t519 MAT_EARLY_CULL, t513; all off by default).
render/run.py --back-to-back, 30 bicycle views, 20 passes + 1 warmup, 6 rotating rounds per arm.

Arms:
- off: best-iter-223 defaults.
- ec: GSPLAT_TT_MAT_EARLY_CULL=1. Same code path as best-iter-224 defaults (224 = 223 + this flag
  on by default + default-off #511 probes); its 9.133 matches #525's 9.131 landing number.
- st: ec + vf48 bundle (PFWC_VIS_FAST=1 OL_RING=4 OL_RING_DEPTH=8) + GAP_CUT=1 + PFWC_DEAL=range.

| arm | round medians (ms/view) | median |
|-----|-------------------------|--------|
| off | 9.365 9.366 9.365 9.366 9.366 9.369 | 9.366 |
| ec  | 9.130 9.133 9.132 9.134 9.133 9.133 | 9.133 |
| st  | 9.044 9.053 9.048 9.021 9.055 9.027 | 9.046 |

Paired per-round deltas:
- st - ec (stack on top of best-iter-224): -0.086 -0.080 -0.084 -0.113 -0.078 -0.106, median **-0.085 ms/view** (0.9%).
- st - off: median -0.319; ec - off: median -0.233.

Correctness: off and ec MD5_GOLDEN_OK 906e0435 30/30. st list 0a790b06: view27_orb_26 and
view29_orb_28 differ (the same two views vf48 changed in #494); the other 28 views are bit-identical.
Hero is bit-identical in all three arms (md5 86524912), PSNR 42.51 dB vs
benchmarks/reference_v2/hero.png. hero-st.png looked at: no seams or tile artifacts.

## Conclusion
The whole remaining stack saves 0.085 ms/view over best-iter-224, below the 0.1 ms/view gate, and it
breaks the golden md5 on two views. Not kept: no new iteration, no tag. The individual levers were each
at or below noise (#494 vf48 -0.068, #517 deal 0.00, #514 gap cut small), and stacking them adds up to
under 1%. Proposal: stop small-lever work on this pipeline; further gains need a structural change
(a different stage split or algorithm), not more flags.

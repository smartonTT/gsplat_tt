# t494: PFWC_VIS_FAST=1 + OL_RING=4 + OL_RING_DEPTH=8 — shelved

Board: bh-30 p150 (viewer box, under the 2026-10-07 viewer exception), grid 12x10, tree at 4c965f79
(= iter 223 defaults, host C++ unchanged). 2026-10-11 17:44-17:47 PDT, viewer stopped only for the bench.

b2b ms/view (render/run.py --back-to-back, 30 views, 20 passes + 1 warmup), run medians, rounds A B B A A B B A:

| round | off (iter 223) | vf48 |
|---|---|---|
| r1 | 8.779 | 8.712 |
| r2 | 8.781 | 8.703 |
| r3 | 8.780 | 8.719 |
| r4 | 8.782 | 8.714 |
| median | 8.7805 | 8.713 |

Delta 0.068 ms/view (0.77%), below the 0.1 ms/view gate: not made default, no best-iter tag.

Correctness: off = MD5_GOLDEN_OK 39d84b28 (12x10) 30/30. vf48 list e646dd67: only view27_orb_26 and
view29_orb_28 differ (the known 1-LSB VIS_FAST drift), other 28 views identical. Hero byte-identical in
both arms (md5 c07dfb8d), PSNR 42.51 dB vs benchmarks/reference_v2/hero.png; looked at, no tile seams.

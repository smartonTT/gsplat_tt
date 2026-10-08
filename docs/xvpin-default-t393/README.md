# t393: cross-view overlap + pinned output as the default (iter 212; first measured as iter 211 before the t395 merge)

Board: yyzo-bh-04 (Blackhole p100a, measurement reservation IRD job 244892).
Code: 56f75707 (opt tip iter 210 + t388 merged; xv + pinned output on by default).
Method: same binary, 1 warm-up + 3 rotating rounds x 30 bicycle views at 1024x1024
(`drive.sh`). The old arm is the opt-out env `GSPLAT_TT_XVIEW_OVERLAP=0 GSPLAT_TT_OUT_PINNED=0`.

| arm | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| old (opt-out) | 10.879 | 11.016 | 11.001 | 10.965 |
| new default | 8.978 | 9.071 | 8.998 | **9.016** (-1.949 ms, -17.8%) |

Stage means (ms): project 3.06 -> 1.14, sort 0.52 -> 0.50, blend 7.07 -> 7.01, d2h 0.25 -> 0.22, xview 0 -> 0.07.

Correctness: md5 906e0435 on 30/30 views in all 7 runs; XVIEW_HITS_OK hits=29 misses=0
in all 4 new-default runs. Device hero at the default config: `opt/metal-screenshots/ttw-212/hero.png`,
PSNR 42.51 dB vs `benchmarks/reference_v2/hero.png`, golden match (max LSB 0). Visual check:
no tile seams, grid lines or blocky/empty tiles; the x10 diff shows only thin edges.

p150 (bh-30, #394, t388 tree with flags by env): xvpin 9.176 vs base 11.027 ms/view (-16.8%).
Published GPU reference: 10.75 ms/view (published, not measured).

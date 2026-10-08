# t393: cross-view overlap + pinned output as the default (iter 212; first measured as iter 211 before the t395 merge)

Board: yyzo-bh-04 (Blackhole p100a, measurement reservation IRD job 244892).
Code: dd5e523c (t393 merged with opt tip 42f92a56: t395 fused mat+blend zero-copy option, t374 OUT_PIN_HOLD;
xv + pinned output on by default). Output: `out-t404/`. The pre-merge run (56f75707) is in `out/`.
Method: same binary, 1 warm-up + 3 rotating rounds x 30 bicycle views at 1024x1024
(`drive.sh`). The old arm is the opt-out env `GSPLAT_TT_XVIEW_OVERLAP=0 GSPLAT_TT_OUT_PINNED=0`.

| arm | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| old (opt-out) | 10.906 | 10.867 | 10.865 | 10.879 |
| new default | 8.924 | 8.982 | 8.978 | **8.961** (-1.918 ms, -17.6%) |
| new default + `GSPLAT_TT_OUT_ZEROCOPY=1` (1 round) | 8.709 | | | 8.709 |

Stage means (ms): project 3.03 -> 1.09 (pfwc gather_wait 2.93 -> 1.08), sort 0.49 -> 0.45,
blend 7.10 -> 7.08, d2h 0.21 -> 0.25, xview 0 -> 0.06. Zero-copy round: d2h 0.003, blend 7.04.

Correctness: md5 906e0435 on 30/30 views in all 8 runs (warm-up, 3 old, 3 default, zero-copy);
XVIEW_HITS_OK hits=29 misses=0 in all 5 overlap runs. Device hero at the default config:
`opt/metal-screenshots/ttw-212/hero.png`, PSNR 42.51 dB vs `benchmarks/reference_v2/hero.png`,
golden match (max LSB 0). Visual check: no tile seams, grid lines or blocky/empty tiles;
the x10 diff shows only thin edges.

Pre-merge (#393, 56f75707): 9.016 vs 10.965 ms/view (-17.8%).
p150 (bh-30, #394, t388 tree with flags by env): xvpin 9.176 vs base 11.027 ms/view (-16.8%).
Published GPU reference: 10.75 ms/view (published, not measured).

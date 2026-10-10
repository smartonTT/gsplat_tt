# #465: back-to-back re-baseline (p150 + p100), 2026-10-10

Headline metric since #465: back-to-back (sustained) ms/view, `render/run.py --back-to-back
--b2b-passes 3` over the 30 bicycle views at 1024x1024, median of all measured passes.
Latency without `--dump-views` is the secondary number.

Builds: A = best-iter-216 (`e13e9f6b`), B = opt tip `c7232e14` + this branch's run.py b2b
print (`13ce253e`). One build per tree, both trees run the same run.py. 3 alternating rounds
(A/B, B/A, A/B); each round runs b2b A/B, then latency A/B. Scripts: `drive465.sh`,
`bench465.sh` (step mode keeps each p100 devrun call under the 600 s reservation ceiling),
`sync_bh30_t465.sh`, `vstart465.sh`. Logs: `out-p100/`, `out-p150/`.

| board | build | b2b ms/view (median of 9 passes) | passes | latency ms/view, no dump (median; runs) |
|---|---|---|---|---|
| p150 bh-30 | A best-iter-216 | **9.574** (104.4 FPS) | 10.036, 10.071, 9.310, 9.334, 9.661, 9.306, 9.904, 9.574, 9.347 | 9.147 (9.156, 9.139, 9.147) |
| p150 bh-30 | B tip | **9.397** (106.4 FPS) | 10.054, 9.500, 9.300, 9.929, 9.397, 9.422, 9.297, 9.385, 9.261 | 9.134 (9.124, 9.181, 9.134) |
| p100 yyzo-bh-04 | A best-iter-216 | **9.821** (101.8 FPS) | 9.841, 9.821, 9.818, 9.848, 9.818, 9.825, 9.822, 9.820, 9.818 | 9.796 (9.783, 9.796, 9.797) |
| p100 yyzo-bh-04 | B tip | **9.818** (101.9 FPS) | 9.840, 9.812, 9.818, 9.813, 9.818, 9.813, 9.828, 9.835, 9.841 | 9.798 (9.804, 9.798, 9.793) |

GPU reference: 10.75 ms/view (RTX A6000, published, not measured). Best b2b: p150 1.14x,
p100 1.09x faster than that published figure.

Reading:
- The p100 is very steady (pass spread 0.04 ms); tip and 216 are the same (-0.003 ms).
- On bh-30 the passes spread 9.26-10.07 ms; the first pass of a run is often about 10.0 ms.
  The tip's -0.18 ms against 216 sits inside that spread. It is not a kept lever and is not
  claimed as a gain.
- The earlier latency figures (iter 216: p100 8.043, p150 7.705 ms/view) were taken with
  `--dump-views`, which hides about 1.7 ms of pfwc device time (#464). Without the dump,
  latency is 9.80 (p100) and 9.14 (p150) ms/view, close to b2b.

Image check (device render, the hero from each b2b run's untimed pass 0):
- p150: sweep md5 `39d84b28` (12x10 golden) on 30/30 views in all 6 b2b runs; hero PSNR vs
  `benchmarks/reference_v2/hero.png` 42.51 dB; golden badge: bit-exact vs
  `hero_golden_8bit_12x10.png`. `opt/metal-screenshots/ttw-217` (A), `ttw-218` (B).
- p100: sweep md5 `906e0435` on 30/30 views in all 6 b2b runs; hero PSNR vs reference_v2
  42.51 dB; golden badge: bit-exact vs `hero_golden_8bit.png`. `ttw-219` (A), `ttw-220` (B).
- Looked at the hero and the x10 diff: no tile seams, blocky or empty tiles; the diff is only
  on thin edges (spokes, frame, bench slats, foliage). The hero is identical across A and B.

Dashboard: iters 217-220 in `opt/ttw/iters.jsonl` (decision `rebaseline`).

Machines: p100 yyzo-bh-04 under the measurement reservation (IRD job 244892, renewed to
2026-10-11T03:40Z; IRD caps at 14 h, so 24 h cover needs another renewal). p150 bh-30 (the
user's viewer box) under the existing viewer reservation, because bh-50 could not be
reserved (ird ValueError). The viewer was stopped only for the bench (13:35-13:38Z) and
restarted (sha 45e6ab92); localhost:8091 answered 200 afterwards.

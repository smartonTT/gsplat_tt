# Task #586: viewer at best-iter-227 + 12x10 md5 golden with BLEND_EXP1 on

2026-10-12, bh-30 (p150, ETH dispatch, 12x10 = 120 cores), viewer reservation IRD 138278 only
(extended by `opt/ird/renew.sh viewer`; no other reserve or release). Measurement box and p100 lock untouched.

## Viewer deploy

`opt/ird/renew.sh viewer --rev best-iter-227` deployed 445f9e40 (best-iter-227, includes #577's
e8fd8e7f: the viewer keeps the B4a fold off) over 614abe85. It is the newest best tag, so no newer
build had to be chosen. The viewer runs with `GSPLAT_TT_B4A_FOLD=0` and EXP1 on (tree default).
Viewer self-test: hero median 9.12 ms.

## Page FPS through localhost:8091

`opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:8091 --subprotocol viser-v1.0.27 --no-decode --secs 23`
from the Mac, 3 runs (`viewer/r*.out`). Page 200, websocket upgrade 101.

| run | page FPS (received) | interval p50 ms | HUD device ms (median) | HUD sent FPS (median) |
|---|---|---|---|---|
| r1 | 121.03 | 8.19 | 7.85 | 119.8 |
| r2 | 120.97 | 8.18 | 7.86 | 119.5 |
| r3 | 120.92 | 8.10 | 7.83 | 119.5 |

Median page FPS 120.97, up from 118.9 at 614abe85 (#577, fold off). Bench b2b headline for
best-iter-227: 8.454 ms/view (118.3 FPS, bh-33 p100). The page FPS is higher than the bench b2b
because bh-30 runs ETH dispatch on 120 cores (b2b 8.24 ms below).

## 12x10 md5 golden (EXP1 on)

`drive.sh best-iter-227 bench.sh r1` built the tag in p150bench/tree586 (nice -n 19, ionice -c3,
half the cores; its SHA file holds the tag object a2348cad, tree = commit 445f9e40), stopped the
viewer 23:41:33 PDT, ran `bench.sh`, and restarted it (ready 23:44:09 PDT, localhost:8091 -> 200).

- 30-view sweep, tree defaults: list md5 **84695855** (30 views), `[DEV] dispatch eth ..., compute grid 12x10`.
- Same sweep with the fold off (the viewer's setting): 84695855, identical.
- raw_md5 in every b2b run: c194c715 (pairs with 84695855).
- Device hero vs benchmarks/reference_v2/hero.png: 42.85 dB (same as iter 227 on bh-33). I looked at
  `out/r1/hero.png` and `out/r1/hero_diff10.png`: no tile seams or blocky tiles; the diff shows only on
  thin edges (spokes, frame), as in iter 227.
- b2b (20 passes, ABBA): fold off 8.245 / 8.242, fold on 7.959 / 7.954 ms/view.

`opt/md5_golden.py` now has `"12x10": "84695855"` (list in `md5-golden-84695855.txt`);
39d84b28 stays in `GOLDENS_EXP1_OFF`. `opt/test_md5_golden.py` checks the new golden passes for
12x10 and that the pre-227 12x10 list now fails.

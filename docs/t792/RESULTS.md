# t792: T2 heavy-row split on by default, kcap 8 confirmed (iter 232)

Run: `docs/t792/drive.sh` at 1ae0aa7c, one `ttp lock p100` session on bh-49 board 0 (Blackhole p150a),
2026-10-14 15:31-15:41 PDT. Raw outputs: `docs/t792/out/remote/`, driver log in run 2235 (`t792.log`).

## kcap A/B (hero dump, 1014 tiles, 4.48 M records, 10 reps per arm, alternating A B B A A B B A)

| round | kcap 8 T2 med (ms) | kcap 16 T2 med (ms) | kcap 8 chain med | kcap 16 chain med |
|---|---|---|---|---|
| 1 | 8.8466 | 9.1784 | 90.1589 | 90.5139 |
| 2 | 8.8488 | 9.2716 | 90.1512 | 90.5638 |
| 3 | 8.8172 | 9.2062 | 90.2199 | 90.5173 |
| 4 | 8.8416 | 9.2107 | 90.1585 | 90.4862 |
| median | **8.844** | **9.209** | **90.159** | **90.516** |

kcap 8 wins every round; T2 -0.364 ms, chain -0.357 ms (bar 0.1 ms). Verdict: default `T2_KCAP=8`.
Split off (T2_SPLIT=0, same tree, 3 reps): T2 98.404 ms, chain 179.713 ms. Defaults (split on, kcap 8):
T2 8.916 / chain 90.217 in the 3-rep check arm; -89.5 ms/view on the trainer backward chain.

## Lossless checks

- Hero, defaults vs T2_SPLIT=0 (`check_g2d.py same`): params_out, m_out, v_out, pfwc, grads all bit-equal.
- T0 oracle crops (`gsplat.train_step_dev check`), defaults and T2_KCAP=1 (every row heavy):
  sh0 t1/t2/t0 pass; sh3 t1/t2 pass, overall pass, t0 ungated and false as before the split (#784 RESULTS).
- `opt/ab_jit_check.py`: kcap 8 vs 16 same kernel hashes (runtime-arg only), JIT-HASH OK; split vs base
  JIT-HASH OK.

## Render (unchanged by this trainer change)

Orbit b2b 3 runs x 20 passes: ms_frame_median 7.299, 7.301, 7.299 (median 7.299); latency 7.259 ms/view.
raw md5 c194c715, sweep md5 84695855 (0 of 30 views differ), hero.png bit-identical to iter 231,
PSNR 42.85 dB vs benchmarks/reference_v2/hero.png. Hero and x10 diff checked by eye: no tile artifacts.

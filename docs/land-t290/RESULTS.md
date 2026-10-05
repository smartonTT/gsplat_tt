# t290: review of #266 and landing PFWC_RECIP_NEWTON default-on

Review: PASS. The step-2 1/tz path uses the same SFPU start/done + approx_recip + 2 Newton
steps as the existing conic path; tests/unit/test_pfwc_recip.cpp passes; `ttp checks` passed on 7bcba92.

Device: yyzo-bh-07 (Blackhole p100a), bicycle 30 views 1024x1024, commit 7bcba92, untraced,
3 rotated rounds (out/run-r*-*.log, out/md5-r*-*.txt).

| arm | r1 | r2 | r3 | mean ms/view | sweep md5 |
|---|---|---|---|---|---|
| default (NEWTON=1) | 11.657 | 11.640 | 11.673 | 11.657 | 906e0435 |
| GSPLAT_TT_PFWC_RECIP_NEWTON=0 | 11.629 | 11.640 | 11.715 | 11.661 | 46a725ab |

Delta -0.004 ms/view (noise). Hero md5 86524912; PSNR vs benchmarks/reference_v2/hero.png
42.51 dB (old default 41.16). Device screenshot: opt/metal-screenshots/ttw-205/ (looked at: no
tile seams; diff on thin edges only). Hero golden refreshed; old one archived as
tests/fixtures/hero/archive/hero_golden_8bit_pre-t290-newton.png, and iters 196-204 now point at it.

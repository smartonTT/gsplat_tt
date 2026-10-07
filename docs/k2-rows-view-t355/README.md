# t355: early-path K2 rows read through a 4 KB-page view (GSPLAT_TT_K2_ROWS_VIEW)

Change (8749e9d8, 8588f1ac): the host reads the early-path K2 count rows on CQ1 through a view of
`buf_k2_rows` with 4 KB pages instead of 14,080 pages of 64 B, then unshuffles the interleaved
bank layout on the host (unit test in tests/unit/test_sort_onelaunch.cpp). Default on;
`GSPLAT_TT_K2_ROWS_VIEW=0` restores the 64-B-page read.

## p100a A/B (yyzo-bh-04, 3 alternating rounds, same build #124 sha 8588f1ac, untraced, 30 views)

| arm | ms/view (r1 / r2 / r3, mean) | sort | bin_emit | blend |
|---|---|---:|---:|---:|
| view on (base) | 10.887 / 10.904 / 10.873 = **10.888** | 0.545 | **0.203** | 7.041 |
| view off (64-B pages) | 10.903 / 10.911 / 10.892 = **10.902** | 0.905 | **0.588** | 6.687 |

- bin_emit drops 0.385 ms (-65 %), but ms/view moves only -0.014 ms (noise). The saved host time
  reappears as blend wait (+0.354 ms): on the p100a the device is the critical path and the
  64-B read was overlapped. The gain is expected where the read is slow (p150: bin_emit 2.42 ms).
- md5: all 6 arms ALL_VIEWS_IDENTICAL to the 906e0435 golden list (30/30).

## Device hero (p100a, 8588f1ac defaults)

`opt/metal-screenshots/t355-k2view/hero.png`, diff `hero_diff10.png` (|hero - reference_v2| x 10).
Sweep md5 906e0435 (0 of 30 differ). PSNR vs `benchmarks/reference_v2/hero.png`: **42.51 dB**.
Golden badge (hero_golden_8bit): match, max 0 LSB. Looked at both images: no tile or
microblock seams, no blocky or empty tiles; the diff shows only thin edges (spokes, frame, slats).

## p150 A/B (bh-30)

`p150_drive.sh` / `p150_remote.sh`: same A/B on bh-30 under the existing viewer reservation,
holding `ttp lock viewer`; viewer stopped for the bench only. Results: see below once run.

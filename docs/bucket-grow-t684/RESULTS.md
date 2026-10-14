# t684: cost of the grown sort tile bucket (65472 vs 32768 records)

Box: bh-33 (p100, 7 DRAM banks), under `ttp lock p100`, 2026-10-13 22:10-22:19 PDT.
Code: b895b386 (best-iter-229 base + test-only knob `GSPLAT_TT_TEST_PREGROW_TILE_CAP=1`,
which starts the sort bucket at 65472 records per tile instead of 32768).
Workload: `render/run.py --back-to-back`, benchmarks/cameras_v2.json (orbit, 30 views),
20 passes, warm-up round 0 discarded, 4 rotated rounds per arm.

| arm  | b2b ms/view per round (median)  | median of medians | spread |
|------|---------------------------------|-------------------|--------|
| base (32768) | 8.054, 8.057, 8.059, 8.059 | 8.058 | 0.005 |
| grow (65472) | 8.034, 8.035, 8.037, 8.036 | 8.035 | 0.003 |

Delta grow - base = -0.023 ms/view (per round -0.020..-0.023). Output md5 identical in
both arms (1d372b94), 0 overflows on the orbit in either arm.

Arm validity: grow logs show `bucket pre-grown to 65472 records`, base logs show
`bucket tile stride 512 pages`. The cap is a runtime arg only, so
`opt/ab_jit_check.py` (a48aa40a) reports JIT-HASH FAIL (all 15 kernels identical):
that is the expected result here, since no kernel source differs between arms; the
check is meant for kernel-edit gates. The arm difference is proven by the log lines.

Screenshot (grow arm): opt/metal-screenshots/t684-grow/hero.png, diff hero_diff10.png,
PSNR 42.85 dB vs benchmarks/reference_v2/hero.png, golden match (max 0 LSB).
Looked at both: no tile artifacts, diff is the usual edge-only pattern (spokes, foliage).

Verdict: noise. The grown bucket costs nothing per view (it is 0.023 ms faster,
below the 0.1 ms/view floor of docs/structural-plan/ab-noise-floor.md). Training-view
numbers from #667 are not inflated by the one-time grow. No per-tile overflow split
is needed, so it was not sized.

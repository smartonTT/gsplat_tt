# t485: b2b keep mode stops re-pinning output slots

Fix (b6250059): `render/run.py --back-to-back` keep mode sets
`GSPLAT_TT_OUT_ZEROCOPY_SLOTS=n_views+2` (32 for bicycle) before the first
render unless the variable is set. Measured passes then reuse ring slots
instead of replacing 26 of 30 per pass (aligned_alloc + PinnedMemory::Create +
OUT_RING fprintf, booked as `sort_cont_prep`). `--b2b-drop` keeps the 4-slot default.

A/B on p100a (yyzo-bh-04, 11x10 grid), commit 811121cd, one build, order
A B B A A B, `GSPLAT_B2B_ALL_STAGES=1`, 20 passes each (check pass excluded).
A = `GSPLAT_TT_OUT_ZEROCOPY_SLOTS=4` (old behaviour), B = new default.

| run | ms/view | pass sd (us) | pass min / max | cont_prep med / max (ms) | sort med (ms) | new pinned slots |
|-----|---------|--------------|----------------|--------------------------|---------------|------------------|
| A1  | 9.406 | 4.7 | 9.398 / 9.416 | 0.120 / 0.253 | 0.568 | 578 |
| B1  | 9.405 | 3.9 | 9.398 / 9.412 | 0.005 / 0.030 | 0.456 | 33 |
| B2  | 9.402 | 4.2 | 9.395 / 9.412 | 0.005 / 0.012 | 0.493 | 33 |
| A2  | 9.407 | 6.7 | 9.395 / 9.429 | 0.133 / 0.572 | 0.638 | 578 |
| A3  | 9.408 | 5.7 | 9.395 / 9.423 | 0.132 / 0.288 | 0.625 | 578 |
| B3  | 9.401 | 4.2 | 9.394 / 9.410 | 0.005 / 0.011 | 0.478 | 33 |

Mean A 9.407, B 9.403 ms/view (-0.004 ms, in the noise: the re-pin was hidden
behind the device). Pass-to-pass sd 5.7 -> 4.1 us, worst pass 9.429 -> 9.412.
`sort_cont_prep` 0.128 -> 0.005 ms median.

Output: raw_md5 5a438f5a in all six runs; B `--dump-views` run md5 906e0435 =
11x10 golden on 30/30 views. Hero (out/hero-r9-dump-B.png) is byte-identical
across all 8 runs; PSNR 42.51 dB against benchmarks/reference_v2/hero.png;
diff image out/diff-r9-dump-B.png (x8 gain) checked by eye: no tile seams.

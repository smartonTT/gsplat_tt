# t481: pfwc output / gather lever at iter 222 -> skip the dead fp32 colour reads (GSPLAT_TT_PFWC_SKIP_RGB)

## Where the spec's two variants stand at iter 222 (from code and the iter-221 Tracy, opt/profiler/ttw-221/zones.txt)
- Gather is already fused into pfwc (#125 PFWC_FUSE): pfwc writes the M-compact segments and the
  counts table; the old gather kernels do not run. The "gather_wait" bucket is pfwc + K2 + the proj_M read.
- pfwc records were already shrunk to 32 B (#467 REC32, iter 221): a,b,c,mx,my,u01,u23,dep. Going lower
  needs fp16/fixed-point means or conics, which changes the image (PSNR margin 42.51 vs 42.4 gate). Not tried.
- The survivor count round trip (lever 4) is off the device critical path in b2b (#467 note).
- Iter-221 p150 device makespans per view: pfwc 1.70 ms, k2_pairs 0.76 ms (iter 222 moved K2 to TRISCs), sort emit 1.22,
  mat_cull 2.16, blend 3.72.

## Bytes pfwc moves per Gaussian (all N, not only survivors), default chain
| stream | B/Gaussian |
|---|---|
| reader: means 3 + cov3d 6 + opacity 1 fp32 tiles | 40 |
| writer: opacity fp32 tile (classify) | 4 |
| writer: colour r, g, b fp32 tiles | **12 (dead with REC32 + UNORM16 packs)** |
| writer: UNORM16 packs q01, q23 | 8 |
| total reads | 64 -> 52 with SKIP_RGB (-19%) |
| writes per survivor | 32 rec + 12 dep/offs/aabb |

With REC32 the record carries only the pre-packed UNORM16 colour words, so the fp32 colour tiles feed
only the NaN-scene fallback (pub01 == 0, where the writer packs on device). The writer still read all
three per chunk. GSPLAT_TT_PFWC_SKIP_RGB=1 reads them only when pub01 == 0. Output is bit-identical by
construction. REC32 dropped 32 B/survivor of writes and cut pfwc+K2 ~0.17 ms in b2b (#471), so pfwc does
respond to DRAM traffic.

## A/B
drive.sh / remote_ab.sh (copies of docs/k2-b2b-t469): one ttp lock p100 around sync + build + warm-up
+ 3 alternating rounds (off on / on off / off on) of render/run.py --back-to-back and latency, then a
--dump-views md5 pass per arm. Box: the existing measurement reservation (yyzo-bh-04, p100a 11x10).

## Results (2026-10-10, yyzo-bh-04 p100a 11x10, one build 2b55e3a6, 20 b2b passes + 1 warm-up per run)

| round | b2b off median ms/view | b2b on median ms/view | delta |
|---|---|---|---|
| r0 (warm-up) | 9.406 | 9.350 | -0.056 |
| r1 | 9.409 | 9.351 | -0.058 |
| r2 | 9.410 | 9.352 | -0.058 |
| r3 | 9.407 | 9.350 | -0.057 |
| **median r1-r3** | **9.409** | **9.351** | **-0.058 (-0.6%)** |

Pass-to-pass spread within a run is about 0.01 ms, so the 0.058 ms saving is real and repeatable.
Latency mode (secondary, no --dump-views): avg 9.4 -> 9.3, p50 9.5 -> 9.4 in all three rounds.
Correctness: --dump-views md5 pass on both arms: MD5_GOLDEN_OK 906e0435 (11x10 golden), 30/30 views,
per-view md5s identical between arms. Device hero (out/hero-on.png) vs benchmarks/reference_v2/hero.png:
42.51 dB; diff image out/hero-on-diff.png (|diff| x8). Looked at both: no seams or tile-grid lines, diff
follows edges only, same as baseline.

## Verdict: not kept (below the 0.1 ms keep gate)

The saving is 0.058 ms/view b2b, under the >=0.1 ms gate, so the flag stays default off on this branch
and no iteration is added. It is free and bit-identical, so it is a good rider: bundle it with the next
pfwc-side lever and A/B them together. 12 B/Gaussian of reads (19% of pfwc reads) bought ~0.06 ms, which
says pfwc reads are not the main pfwc cost now; the remaining pfwc time is more likely compute or the
writer's survivor scatter than read bandwidth.

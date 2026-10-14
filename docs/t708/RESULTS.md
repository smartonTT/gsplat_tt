# t708/t712: forward training flag (GSPLAT_TT_BLEND_TRAIN) on the p150 box

Run 2026-10-14 01:24-01:34 PDT, bh-49 (Blackhole **p150a**, label 'p150'), commit 530de93d,
driver docs/t708/drive.sh (one `ttp lock p100`, opt/sync_remote.sh). Orbit cameras_v2,
30 views, 1024x1024, b2b 20 passes. Raw outputs: docs/t708/out/.

Fix needed to compile the flag-on kernel (530de93d): with BLEND_T_PERIOD 0 (forced in train
mode) the TRISC_PACK readback loop's `g_seen % kBlendTPeriod` failed -Werror=div-by-zero
even behind `kBlendTPeriod != 0u`; it now divides by kBlendTPeriodDiv (never 0).

## (a) flag off is unchanged

Flag-off b2b raw_md5 = **c194c715** in all 5 runs (r0-r4). That is the p150 value of
best-iter-229 (docs/p150-rebaseline-t687, iter 230). The spec's 1d372b94 is the p100 (bh-33)
value (docs/b2b-floor-t614); the p150 output differs from p100's, not from the baseline.
Device sweep: 30 views, sweep_md5 84695855, same as iter 230.

## (b) A/B, 4 rounds per arm, order alternated (r1 off,on; r2 on,off; r3 off,on; r4 on,off)

| arm | r1 | r2 | r3 | r4 | median | spread (max-min) |
|---|---|---|---|---|---|---|
| off (fused matblend, render rules) | 7.405 | 7.406 | 7.409 | 7.408 | **7.407** | 0.004 |
| on (GSPLAT_TT_BLEND_TRAIN=1) | 12.923 | 12.926 | 12.922 | 12.919 | **12.923** | 0.007 |

ms/view b2b medians. Flag on costs +5.52 ms/view (+74%). That cost mixes three things: the
unfused program (train mode turns the fused matblend path off), the fp32-accurate exp with
no T early-out, and the 16 KB/tile fp32 writes. Not split here.

JIT check (`ab_jit_check.py --kernel alpha_blend_compute_mb --kernel writer_alpha_blend`):
**JIT-HASH FAIL**, by the tool's rule, because both kernels are *missing* from the off arm
(not because the hashes match). The off arm runs the fused program (matblend_compute,
matblend_brisc, matblend_ncrisc); the on arm runs alpha_blend_compute_mb,
writer_alpha_blend, reader_alpha_blend_mb_devcull, mat_cull_compute and
sort_subchunk_materialize. So the arms run different kernel sets, and this is not an A/A
run; the md5s also differ (on: 54d7c762). The FAIL flags the tool's check, which assumes
both arms compile the same kernel names.

## (d) device screenshots (reference: benchmarks/reference_v2/hero.png)

| state | hero | PSNR vs reference_v2 | sweep (30 views) |
|---|---|---|---|
| off | opt/metal-screenshots/t708-off/hero.png | 42.85 dB | md5 84695855, all differ 0 of 30 |
| on | opt/metal-screenshots/t708-on/hero.png | 42.52 dB | md5 94113ad6, all differ 0 of 30 |

Looked at both heroes and both 10x diffs: no tile seams, no grid, no blank or shifted
tiles. The diffs have the same structure (spokes, frame and leaf edges); flag on is slightly
brighter in the sky and treetops, where the 1e-4 stop rule and no early-out change the
saturated pixels. The golden md5 badge is separate (golden_match false, max 13 LSB, as for
the baseline).

## dump sanity (docs/t708/check_dump.py, docs/t708/out/dump-check.json)

train-v0.bin (view 0, hero): 16,777,216 bytes = 1024 pages of 16 KB = the 32x32 screen
tiles of 1024x1024; 4 fp32 tiles R, G, B, T per page in microblock layout.
- all values finite; RGB in [0.0016, 0.9999]; |T| <= 1 (T max 0.946).
- full fp32: >99.99% of words have non-zero low 16 bits (not bf16 in fp32 slots).
- vs the device u8 PNG of the same run: max 1 LSB (73.3% of pixels equal, 0 over 1 LSB);
  the PNG goes through bf16 and truncation, so 1 LSB is expected.
- T < 0 (blend stopped on the 1e-4 rule): 404,944 pixels (38.6%); their |T| is under
  0.01 (T min -0.00999), as the rule requires (T (1 - 0.99) < 1e-4 means T < 0.01).

## (c) T0 gradient oracle with the device fp32 forward: not run

The render pipeline cannot run the T0 scene so that the oracle check means anything:
render/run.py renders a .ply through the device projection (its own lowpass, k_cap 3
bounds, pruning, UNORM16 packs), so the blend's 2D inputs differ from the oracle's fp64
`means2d`/`conic`/`opacity`/`colors`. `blend_backward_dev prep` with the oracle's 2D
params and the device image would mix two forwards and fail for reasons that are not the
backward's. Proposed as its own task (follow-up): a blend-only forward harness (like
render/train/blend_bwd_bench) that feeds the oracle's fp32 2D params to the BLEND_TRAIN
compute kernel at 64x64 and returns the R/G/B/T buffer, then T1 prep with image = that
buffer and the oracle check at 1e-2 (sh0 and sh3).

Known lossless gap: the forward's records carry opacity and colour as UNORM16 (per-scene
packs, task #122/#481), so the training forward is not bit-exact against fp32 parameters
(16-bit quantisation). The lossless rule (user msgs #384/#387)
needs fp32 (or an exactness argument) for op/colour in train mode before training quality
is judged against published 3DGS/gsplat or LiteGS bicycle numbers.

# t717: train mode carries fp32 opacity/colour (lossless gap)

Run 2026-10-14 02:30-02:37 PDT, bh-49 (Blackhole **p150a**, label 'p150'), commit 0a572e51,
driver docs/t717/drive.sh (one `ttp lock p100`, opt/sync_remote.sh). Orbit cameras_v2,
30 views, 1024x1024, b2b 20 passes. Raw outputs: docs/t717/out/.

## Change (9286c250)

With GSPLAT_TT_BLEND_TRAIN=1 the host uploads `scene_train_opc` (fp32 op, r, g, b; 16 B per
gaussian). The per-scene packs then hold only the cull opacity code (rounded up, so the cull
stays conservative) and the scene index in rec[7]. The blend reader gathers each bulk record's
fp32 entry (64 B DRAM pages, 64 records per batch) into the unfused program's idle CB 9; the
compute uses those words directly (SFPU_UNORM 0, DECODE_AHEAD 0). Flag-off code is unchanged.
Unit test: tests/unit/test_blend_train_t717.cpp.

## (a) flag off is unchanged

Flag-off b2b raw_md5 = **c194c715** in all 5 runs (r0-r4), the p150 value of best-iter-229
(iter 230, t708). Sweep md5 84695855, same as before.

## (b) A/B, 4 rounds per arm, order alternated (r1 off,on; r2 on,off; r3 off,on; r4 on,off)

| arm | r1 | r2 | r3 | r4 | median | spread |
|---|---|---|---|---|---|---|
| off (fused matblend) | 7.403 | 7.406 | 7.405 | 7.404 | **7.405** | 0.003 |
| on, fp32 op/colour (t717) | 14.943 | 14.944 | 14.946 | 14.943 | **14.944** | 0.003 |
| on, UNORM16 packs (t708, same box, 01:24 PDT) | 12.923 | 12.926 | 12.922 | 12.919 | 12.923 | 0.007 |

ms/view b2b medians. fp32 op/colour costs **+2.02 ms/view** over the UNORM16 train mode
(+15.6%), and train mode now costs +7.54 ms/view over flag off. Train-mode raw_md5 0c8d83a1
in all 5 on runs (t708: 54d7c762). STAGES (shot run): blend 13.60 of 14.92 ms/view, so the
added cost is in the blend, where the reader's per-batch gather (64 random 64 B NoC reads,
then a barrier, before the bulk slot is pushed) sits on the critical path.

JIT check (`ab_jit_check.py`, docs/t717/out/jit-check.txt):
- off vs on: JIT-HASH FAIL by the tool's rule because the blend kernels are missing from the
  off arm (it runs the fused matblend program), as in t708. Not an A/A run.
- t708 on vs t717 on: **JIT-HASH OK**: alpha_blend_compute_mb, reader_alpha_blend_mb_devcull
  and writer_alpha_blend all compiled to new hashes, so the on arm ran the edited kernels.

## (c) device screenshots (reference: benchmarks/reference_v2/hero.png)

| state | hero | PSNR vs reference_v2 | sweep (30 views) |
|---|---|---|---|
| off | opt/metal-screenshots/t717-off/hero.png | 42.85 dB | md5 84695855, differ 0 of 30 |
| on (fp32) | opt/metal-screenshots/t717-on/hero.png | 42.52 dB | md5 cc822a42, differ 0 of 30 |

Looked at the on hero and its 10x diff: no tile seams, grid, blank or shifted tiles; the diff
shows the usual spokes, frame and leaf edges. t717-on vs t708-on hero (u8): max 1 LSB,
99.84% of pixels equal, PSNR 80.4 dB, as expected for UNORM16 vs fp32 inputs.

## (d) fp32 dump (docs/t708/check_dump.py, docs/t717/out/dump-check.json): PASS

train-v0.bin (view 0): 1024 pages of 16 KB; all finite; RGB in [0.0016, 0.9999]; |T| <= 1;
vs the device u8 PNG max 1 LSB (73.3% equal, 0 over 1 LSB).

## Not done here

T0 gradient oracle with the device forward: still needs the blend-only forward harness
proposed in t708 (follow-up there). The gather cost is a follow-up (carry fp32 op/colour in
the materialized records, or prefetch the next batch's gather behind compute).

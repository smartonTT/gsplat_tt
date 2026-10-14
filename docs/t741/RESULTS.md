# Task #741/#745: fp32 op/colour side bucket (GSPLAT_TT_TRAIN_OPC_SIDE) device A/B

Box: bh-49, Blackhole p150a (measurement reservation). Commit d5d30d31. Driver
docs/t741/drive.sh + remote.sh, outputs docs/t741/out/. Run 2026-10-14 09:00-09:10 PDT.

## A/B (b2b ms/view, orbit cameras_v2, 30 views x 20 passes, 4 alternated rounds)
| arm | env (on top of GSPLAT_TT_BLEND_TRAIN=1) | rounds | mean |
|---|---|---|---|
| on | default (PFWC_REC32=1) | 14.697 14.702 14.700 14.699 | 14.700 |
| aa | same as on (A/A) | 14.704 14.705 14.699 14.685 | 14.698 |
| rec32off | GSPLAT_TT_PFWC_REC32=0 | 14.902 14.907 14.902 14.903 | 14.904 |
| side | rec32off + GSPLAT_TT_TRAIN_OPC_SIDE=1 | 14.923 14.925 14.922 14.924 | 14.924 |

A/A gap 0.002 ms. rec32off costs +0.20 ms/view (64 B blendrec), side adds +0.02 on top:
side is +0.22 ms/view slower than the default. Host stage split (pass 10): blend wait
on 13.43-13.45, rec32off 13.63-13.65, side 13.63-13.66; sort 0.51-0.54 in all arms.

## Checks
- Lossless: fp32 train dump of view 0, md5 3fada021 for on, rec32off and side.
  docs/t708/check_dump.py vs each dump run's PNG: DUMP_CHECK PASS (max 1 LSB) for all three
  (out/check_dump-{on,r32,side}.txt).
- T0 oracle: tests/unit/test_train_oracle.py, test_train_opc_writer_t730.py,
  test_train_opc_side_t741.py: 18 passed, 1 skipped.
- JIT (opt/ab_jit_check.py on reader_alpha_blend_mb_devcull, sort_bin_onelaunch,
  sort_ol_town_compute; out/jit-check.txt): on vs side OK and rec32off vs side OK (all three
  kernels rebuilt); on vs aa identical (A/A). on vs rec32off: reader unchanged (expected,
  REC32 only changes the sort kernels).
- L1: the +128 KB/mover ring fits (the side arm built and ran; no CB/L1 overflow).
- Screenshot (side): opt/metal-screenshots/t741-side/hero.png + hero_diff10.png, PSNR
  42.52 dB vs benchmarks/reference_v2/hero.png, 30-view sweep md5 cc822a42, 0 of 30 views
  differ. Looked at it: no tile artifacts; the diff shows only the usual edge/spoke structure.

## Open question
The side blend time equals rec32off's, so either the reader falls back to the per-record
gather (stamps missing) or the windowed side read costs as much as the gather. A diagnostic
run (PHASE=diag in drive.sh) answers it: GSPLAT_TT_T727_DIAG=2 drops the gather fallback, so
the side dump keeps md5 3fada021 only if every subchunk took the side read.

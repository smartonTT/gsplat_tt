# Task #755: 3-slot op/colour ring (GSPLAT_TT_TRAIN_OPC_RING3) device A/B

Box: bh-49, Blackhole p150a (measurement reservation). Commit d732621e. Driver
docs/t755/drive.sh + remote.sh, outputs docs/t755/out/. Run 2026-10-14 10:19-10:26 PDT.

## What was tried
- (a) "issue the next subchunk's op/colour reads before the current compute" was already the
  default: CB_BUCKET_BULK and CB_TRAIN_OPC are 2-slot double buffers, so the reader gathers one
  subchunk ahead. RING3 makes both 3 slots (CB_TRAIN_OPC with 32 B pages, the gather staging
  moved to its own CB 15), so the reader can run two subchunks ahead.
- (b) moving the read to another RISC was tried in t734 (writer gather): +0.12 ms/view, loses.

## A/B (b2b ms/view, orbit cameras_v2, 30 views x 20 passes, 4 alternated rounds)
| arm | env (on top of GSPLAT_TT_BLEND_TRAIN=1) | rounds | mean |
|---|---|---|---|
| on | default (2-slot) | 14.698 14.698 14.698 14.701 | 14.699 |
| aa | same as on (A/A) | 14.698 14.701 14.702 14.702 | 14.701 |
| r3 | GSPLAT_TT_TRAIN_OPC_RING3=1 | 15.214 15.219 15.220 15.220 | 15.218 |

A/A gap 0.002 ms. r3 is +0.52 ms/view slower than the default. Host stage split: blend
13.41-13.45 (on) vs 13.93-14.05 (r3); the extra time is all in the blend kernel. A deeper
ring does not hide the op/colour wait; it makes the reader slower (more, smaller CB
transactions on CB_TRAIN_OPC and the extra staging CB). Not root-caused further.

## Checks
- Lossless: fp32 train dump of view 0, md5 3fada021 for on and r3 (out/dump-md5.txt),
  identical to t741's on/rec32off/side dumps. The driver's check_dump call used the wrong PNG
  path (FileNotFoundError, out/check_dump-*.txt); the r3 buffer is bit-identical to the
  buffer t741 checked with DUMP_CHECK PASS, so the check carries over.
- T0 oracle + unit: tests/unit/test_train_oracle.py, test_train_opc_writer_t730.py,
  test_train_opc_side_t741.py, test_train_opc_ring3_t755.py: 21 passed, 1 skipped.
- JIT (opt/ab_jit_check.py, out/jit-check.txt): on vs r3 reader_alpha_blend_mb_devcull
  hashes differ (JIT-HASH OK, the r3 arm ran the rebuilt reader); sort_bin_onelaunch
  identical (not edited). on vs aa: all 13 kernels identical (A/A).
- Screenshot (r3): opt/metal-screenshots/t755-r3/hero.png + hero_diff10.png, PSNR 42.52 dB
  vs benchmarks/reference_v2/hero.png, 30-view sweep md5 cc822a42, 0 of 30 views differ.
  Looked at both images: no tile artifacts, the diff is the usual edge/spoke pattern.

## Verdict
Shelved. GSPLAT_TT_TRAIN_OPC_RING3 stays default off (+0.52 ms/view, lossless).

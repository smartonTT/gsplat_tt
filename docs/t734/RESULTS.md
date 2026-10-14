# Task #734: device A/B of the writer-side fp32 op/colour gather (#730)

Box: bh-49, Blackhole p150a, board 0 (measurement reservation). Commit 600c109b
(= #730 330a9819 + driver). Driver docs/t734/drive.sh, outputs docs/t734/out/.
Run 2026-10-14 04:36-04:45 PDT.

## Verdict: loses, default stays off
| arm | env (on top of GSPLAT_TT_BLEND_TRAIN=1) | b2b ms/view, 4 rounds | mean | spread |
|---|---|---|---|---|
| on | (t724 default) | 14.688 14.678 14.681 14.681 | 14.682 | 0.010 |
| aa | same as on (A/A) | 14.681 14.681 14.677 14.677 | 14.679 | 0.004 |
| wr | GSPLAT_TT_TRAIN_OPC_WRITER=1 | 14.797 14.797 14.807 14.802 | 14.801 | 0.010 |
| wrb | wr + GSPLAT_TT_TRAIN_OPC_BATCH=128 | 14.780 14.781 14.767 14.781 | 14.777 | 0.014 |

Rounds alternated (on/wr/aa/wrb, wrb/aa/wr/on, wr/on/wrb/aa, aa/wrb/on/wr); 30 views x
20 passes each. A/A gap 0.003 ms, so wr +0.119 and wrb +0.095 ms/view are real
slowdowns, not noise. Bound was 13.26 (#727 d2, no gather).

## Checks
- Lossless: fp32 train dump of view 0, md5 3fada021 for both on and wr (byte-identical).
  docs/t708/check_dump.py: DUMP_CHECK PASS for both (out/check_dump-*.txt).
- T0 oracle: tests/unit/test_train_oracle.py + test_train_opc_writer_t730.py pass
  (15 passed, 1 skipped); device output is byte-identical, so oracle inputs are unchanged.
- JIT (opt/ab_jit_check.py, reader_alpha_blend_mb_devcull + writer_alpha_blend):
  on vs wr OK, on vs wrb OK (both kernels rebuilt with new hashes); on vs aa all 13
  kernels identical (expected for the A/A). out/jit-check.txt.
- Screenshot (wr): opt/metal-screenshots/t734-wr/hero.png + hero_diff10.png,
  PSNR 42.52 dB vs benchmarks/reference_v2/hero.png (same as #727). Looked at it: no tile
  artifacts; the diff shows only the usual edge/spoke structure.

## Why it does not help, and what next
Moving the 64 B random gather from NCRISC/NoC1 to BRISC/NoC0 did not hide any of the
1.06 ms gather; it added ~0.1 ms. So the cost is not reader issue time that another
RISC can overlap: the random small DRAM reads themselves (latency/request rate) are
what the slab pipeline waits for, wherever they are issued, and the writer handoff adds
CB 14 traffic and delays tile writes.

Recommendation: build the full side slab (docs/t730/RESULTS.md, est. +0.3-0.5 ms,
net ~1 ms/view). It is the only lever left here that removes the random gather instead
of moving it. Drop the writer-gather path (or leave it off) once the slab lands.

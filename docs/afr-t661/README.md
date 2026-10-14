# t661: why AFR chip 1 differs from chip 0 on bh-50

t615 (docs/viewer-t615/RESULT.md) found the AFR hero from chip 1 differing from chip 0 in 257 of
1,048,576 px (max 13/255), both at 42.85 dB vs benchmarks/reference_v2/hero.png. Chip 0 matched the
single-chip render (51d25cd7).

## Root cause: core count, not firmware

| board | firmware | harvest | stage grid in t615 |
|---|---|---|---|
| 0 | 18.12.1 | 0x0 | 13x10, capped to 12x10 = 120 cores |
| 1 | 19.7.0 | 0xc0 (2 Tensix rows) | 11x10 = 110 cores |

Several stages partition work by core count, so the rounding or tie order changes for a few pixels.
This was shown on one chip in docs/eth-dispatch-t397: 10x10, 11x10 and 12x10 each give a different md5.

Device proof, from t627 on bh-50 (IRD 133915, run 3, ttp/t627 fe9cf434, both cards capped to 11x10):

| hero | card | image md5 | PSNR vs reference_v2 |
|---|---|---|---|
| hero-1chip-b2b | 0 (fw 18.12.1) | eb8d8a50 | 42.845 dB |
| hero-chip1 | 1 (fw 19.7.0) | eb8d8a50 | 42.845 dB |

Card 0 vs card 1: 0 px differ. raw_md5 is 1d372b94 on both cards, which is the 11x10 golden. The
diff vs the reference (hero-chip1_diff10.png) shows the usual edge and spoke structure, with no tile
seams. The firmware difference does not change the bits.

## Fix (9653fd24)

- `gsplat/afr_pool.py`: each worker reports its stage grid (new binding `render.stage_grid()`).
  If the chips differ, the pool restarts the larger ones with GSPLAT_TT_GRID_X/Y set to the smallest
  grid, so every chip renders the same bits. Round-robin AFR is paced by the slower chip, so this
  should cost about nothing. GSPLAT_VIEWER_AFR_SAME_GRID=0 turns the cap off.
- `opt/viewer/viewer.sh stop` now waits for the AFR workers listed in viewer.afr.pids. In t615, a
  leftover worker held board 1, so the board-1-alone arm failed.
- Unit tests: tests/unit/test_viewer_afr.py (14 passed, 2 skipped), tests/unit/test_afr.py (12 passed).

## Still to do on device

`docs/afr-t661/verify.sh` (with JOB=<own IRD job on bh-50>, after deploying the head with
docs/viewer-t615/deploy.sh). Arms: b0, b1, AFR (cap on), AFR (cap off), and b0 at 11x10.
compare.py prints the md5, PSNR and pixel diffs.

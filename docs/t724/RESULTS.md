# t724: hide the fp32 op/colour cost in train-mode blend (bh-49 p150a, 2026-10-14)

Commit cb904694 (branch ttp/t724-..., from t717 a63b4dff). Driver docs/t724/drive.sh, outputs docs/t724/out/.

## Finding

t717's train mode (fp32 op/colour) also switched decode-ahead off (DA 2 -> 0) and moved the
op/colour decode to TRISC1. The reader gather was already double-buffered (2 x 256 KB slots).
cb904694 keeps DA 2 in train mode: TRISC0 stages the fp32 op/colour as SFPLOADI lo/hi words in
128 B ring slots, TRISC1 stores them into the same DEST slots blend_stage_coeffs used (same
values by construction). `GSPLAT_TT_BLEND_DECODE_AHEAD=0` gives the t717 path (arm on0).

## b2b ms/view (30 bicycle views, 20 passes, 4 rounds, order alternated)

| arm | r1 | r2 | r3 | r4 | mean | raw_md5 |
|---|---|---|---|---|---|---|
| off (flag off) | 7.405 | 7.408 | 7.406 | 7.403 | 7.406 | c194c715 (10/10) |
| on0 (train, DA 0 = t717) | 14.944 | 14.945 | 14.944 | 14.944 | 14.944 | 0c8d83a1 |
| on (train, DA 2, t724) | 14.670 | 14.667 | 14.672 | 14.668 | 14.669 | 0c8d83a1 |

on vs on0: **-0.275 ms/view** (-1.8%), spread within each arm <= 0.005 ms, so well above the
0.1 ms noise floor. t717's UNORM16 train mode was 12.923, so 1.75 of the 2.02 ms remain.
Flag off unchanged (7.406, md5 c194c715 = best-iter-229 on p150).

## Lossless checks

- fp32 train buffer dump of view 0: md5 3fada021... for both DA 2 and DA 0 (bit-identical).
  docs/t708/check_dump.py: PASS (finite, |T| <= 1, max 1 LSB vs the device PNG, 0 px over 1 LSB),
  docs/t724/out/dump-check.json. The .bin dumps (16 MB each) are not committed.
- Train-mode raw_md5 0c8d83a1 in all 8 on/on0 runs, same as t717's.
- T0 gradient oracle with the device forward: not run here; it needs the device-forward harness
  of #716. The forward output is bit-identical to t717's fp32 path, so #716's result covers it.

## JIT check (opt/ab_jit_check.py @a48aa40a, docs/t724/out/jit-check.txt)

- on0 vs on: JIT-HASH OK (alpha_blend_compute_mb and reader_alpha_blend_mb_devcull differ).
- off vs on / on0: reported "FAIL" only because the off arm runs a different blend pipeline
  (matblend_*; alpha_blend_compute_mb is not built). This is not an A/A run.

## Screenshots (device, vs benchmarks/reference_v2/hero.png)

| arm | hero | diff | PSNR |
|---|---|---|---|
| off | opt/metal-screenshots/t724-off/hero.png | hero_diff10.png | 42.85 dB (sweep md5 84695855) |
| on | opt/metal-screenshots/t724-on/hero.png | hero_diff10.png | 42.52 dB (sweep md5 cc822a42) |

t724-on hero.png is byte-identical to t717-on. Looked at it: no tile artifacts; the diff
follows edges (spokes, foliage), no tile grid.

## What remains

1.75 ms/view of the fp32 cost is not decode placement. Not profiled yet: next is a zone profile
of the on arm (reader gather wait vs TRISC0 staging vs TRISC1 stores), then, if the reader is
the wait, carry fp32 op/colour in the materialized sorted records so reads are contiguous.

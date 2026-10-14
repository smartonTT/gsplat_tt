# #803 (F3 prep): exact per-pair alpha cull of the training pair lists

Code only, Mac, no device. Lever 3 of docs/training-plan/t799/PLAN.md (training only; render B2
stays shelved, #538/#666).

## What
`GSPLAT_TT_TRAIN_PAIR_CULL=1` (default off) makes the training prep drop a (Gaussian, 32x32 tile)
pair when its largest alpha over the tile's pixel centres is below alpha_min (1/255):
`gsplat/blend_backward_dev.py` `pair_alive` (exact conic minimum over the tile rectangle, as
#799's pair_model.py) and `pair_lists(..., live)`. `prep` (T1 records), `ref_slots_fast` (T1
reference slots), `forward_fast` and so the #748 chain (`train_step_dev prep`, T2 P/counts) all
use it; counts, P and the slot word number only the kept pairs, so the device T1/T2 need no change.
Lossless (msg #384): a pair is dropped only if it misses alpha_min by more than a power margin of
1e-4 + 4e-6 x (magnitude sum of the fp32 tile-centred power polynomial), >10x its fp32 error;
non positive-definite conics are always kept. `pair_lists(...)["kept"]` maps culled slots back
to the uncut numbering.

## Tests (tests/unit/test_train_pair_cull.py, numpy only)
On a random scene (31% of pairs dropped) and both T0 oracle dumps (sh0, sh3), cull on vs off:
forward image bit-equal; kept T1 slots (fp32 and fp64 refs) and tie flags bit-equal, dropped
slots exactly zero; per-Gaussian slot sums bit-equal; T2 grads and Adam params bit-equal (sh0);
device-kernel emulator on the culled lists vs culled reference < 1e-4. Brute force over every
pixel of every pair: no dropped pair reaches 1/255 in fp32, every pair 1% under it is dropped.
Culled records are the uncut ones (same bits, slot word renumbered); every mask-0 pair is dropped.
Env flag default off.

## Hero pair counts (Mac model, `hero_pairs.py` -> `hero_pairs.json`)
Bicycle hero `_DSC8679`, 1237x822, rows with opacity >= 1/255, code the prep runs:

| list | jclamp off | jclamp 1.3 (#797) |
|---|---:|---:|
| 3-sigma rect (today) | 4,478,225 | 3,300,477 |
| opacity-aware rect (lever 4, counted only) | 4,002,455 | 2,981,999 |
| exact alpha cull (this flag) | 3,163,378 (-29.4%) | 2,639,120 (-20.0%) |

pair_model.py (#799): 2,637,695 tight + jclamp; this cull keeps 1,425 more (the fp32 margin).
Longest row stays 1,014 pairs; tile max 23,886 -> 20,261, mean 3,255 -> 2,603 (jclamp on).
Device A/B (pairs, T1, T2, forward) is F3b.

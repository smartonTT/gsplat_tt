# #785: T1 blend backward, LPT load balance by body count (device A/B, hero)

p150 bh-49 (measurement reservation, inside `ttp lock p100`), one build, same kernels in all arms,
own TT_METAL_CACHE per arm. Driver `docs/t785/ab.sh`; raw outputs `out/remote/`. Hero view
_DSC8679 at 1237x822, 110 cores, 4 rotated rounds x 8 reps per arm.

- base: contiguous row split by pairs (current bench, no `cores.u32`)
- LPT: `python3 -m gsplat.t1_balance_dev plan` (whole tiles, LPT by modelled cycles, #775)
- split: `plan --split --kmax 2` (81 heavy tiles split into 2 microblocks, 80,191 partial slots
  merged by `t1_balance_dev merge` in fixed part order, on the host, not timed)

## Hero T1 time (warm median per round, ms)

| arm | r1 | r2 | r3 | r4 | mean | vs base | modelled (#775) |
|---|---|---|---|---|---|---|---|
| base | 82.415 | 82.411 | 82.413 | 82.415 | 82.41 | - | 82.61 |
| LPT | 51.948 | 51.924 | 51.953 | 51.954 | 51.95 | -30.47 ms (-37.0%) | 51.39 |
| LPT + split k<=2 | 50.885 | 50.879 | 50.857 | 50.887 | 50.88 | -31.54 ms (-38.3%) | 50.09 |

Round spread is under 0.03 ms in every arm; the arms do not overlap. Ideal (total work / 110
cores) is ~49.8 ms.

## Per-core busy spread (device stats, round 1, cycles)

| arm | busiest/mean total | min/mean total | busiest/mean body | busiest core (pairs) | sum of all cores |
|---|---|---|---|---|---|
| base | 1.655 | 0.520 | 1.809 | 107 (41,398) | 7.388e9 |
| LPT | 1.043 | 0.978 | 1.054 | 72 (42,640) | 7.390e9 |
| split | 1.018 | 0.985 | 1.028 | 76 (44,638) | 7.415e9 |

Total work is unchanged; it is only redistributed. Modelled busiest/mean bodies were
1.778 / 1.051 / 1.023; measured body ratios are 1.809 / 1.054 / 1.028.

## Exactness and oracle

- LPT: the 9 output floats per slot are bit-identical to base on sh0, sh3, s1, s2 and all 4 hero
  rounds (same9). Base hero is bit-identical to #747/#767, whose hero oracle passed (3.28e-3).
- split: not bit-identical (summation order of the 2 parts): hero 2,848 of 4,478,225 slots differ,
  max abs 0.0625; small dumps max abs <= 6e-8.
- Device T0 oracle (f32, bar 1e-2), all arms pass: sh0 3.79e-5 (all arms), sh3 1.28e-4 / 1.28e-4 /
  1.17e-4, s1 1.036e-3 / 1.036e-3 / 1.037e-3, s2 1.64e-3 (all arms).
- Hero oracle, merged split slots (round 4, `check --fast`, Mac): rel_err 0.00328, 55108 tie slots skipped, rel_err_all 0.0278, pass. Base/#747: 3.28e-3, 55,108 ties, 0.0278.

## JIT check

`opt/ab_jit_check.py` reports A/A (JIT-HASH FAIL) for blend_bwd_compute and bwd_reader, base vs
LPT and base vs split. That is by design here: the lever is host data (per-core row boundaries
in `cores.u32` and the row order of the prep dir), not kernel code, so all arms run the same
kernels. The arms differ where it matters: `cores.u32` md5s differ per arm (`out/remote/cores-md5.txt`),
`timing.json` shows `balance` 0/1/1 and busiest-core pairs 52,187 / 44,263 / 46,640, and the
per-core stats above. (ELF md5s differ per arm only because each arm has its own cache dir.)

## Trainer ms/view

There is no end-to-end on-device trainer ms/view yet; T1 blend backward is the measured stage.
The T1 saving per hero view is 30.47 ms (LPT) or 31.54 ms (split, before the merge cost).

## Recommendation

Keep LPT: lossless (bit-identical), -30.5 ms (-37%) on the hero T1, close to the model. Shelve
the split for now: another -1.07 ms (-2%), but it is not bit-identical and needs the partial-slot
merge in T2 on device, whose cost is not measured yet.

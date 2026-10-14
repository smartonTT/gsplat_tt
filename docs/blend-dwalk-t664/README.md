# #664 R1 — BLEND_DWALK: walk DEST with load addr mods, not INCRWC

Step 0 (paper, no device) for the iter-229 TRISC1 blend body (BLEND_SCHED=2, EXP1, SWAP_FILL).

## Slot count per Gaussian x 32-px microblock (issue slots on the SFPU/RWC port)

| part | iter 229 | DWALK |
|---|---|---|
| pushed x/y loads | 2 | 2 |
| front replay (slots 0-12) | 13 | 13 |
| pushed middle (7 loads, 13 math incl. 3 swaps, pads filled) | 20 | 20 |
| INCRWC chain to D = 2 IX (steps <= 14; 0-5 per IX, mean 85/32) | 2.66 | 0 |
| tail replay (6 loads, 4 stores, 8 math) | 18 | 18 |
| SETRWC D = 0 | 1 | 0 (last store's ADDR_MOD clears D) |
| **total** | **56.66** | **53** (-3.66) |

LREG use is unchanged (L0-L7; L9 0, L10 1, L11 -1, L12-14 EXP1 constants): the same ops in the
same order with the same registers, so the output is bit-identical (md5 077caaba expected).

How: IX = 16a + 4b + c (a <= 1, b <= 3, c <= 3, a+b+c <= 7). Middle load k post-increments D
by 32 (ADDR_MOD_6), 8 (ADDR_MOD_5) or 2 (ADDR_MOD_4) for the first a, next b, next c loads, else
ADDR_MOD_7 (0). Its immediate is BS_S(slot) - D_before_k (>= 384 - 62, no wrap), so it reads the
same DEST row. D ends at 2 IX = the tail base. The tail's final SFPSTORE uses ADDR_MOD_1
(dest clr), so the tail recording is 18 slots (replay(13, 18)) and the SETRWC goes. The four
ADDR_MODs are set per tile in blend_sched_record (copy_tile init sets 0/2/3, fill init sets 7;
nothing in the blend loop sets 1/4/5/6). Model test: tests/unit/test_blend_dwalk_t664.py.

Loop orders considered and rejected on paper:
- pixel state resident across Gaussians (T, R, G, B in LREGs): 4 more LREGs live for the whole
  record; the body already uses all 8, would need the 7 param loads spilled. No slot gain.
- Gaussian params resident across microblocks: 9 params, 8 LREGs. No.
- 2-Gaussian pair body: swap pads already filled by SWAP_FILL; no further slot.
- C0 into a programmable const LREG via SFPCONFIG: 1 slot, L11 is the compiler's -1. No.
- folding the K99 clamp or the ln2 prescale into params: not bit-identical.

Projection: 3.66 slots x ~0.041 ms/view per slot ~= 0.15 ms/view b2b, if the loop stays SFPU
issue-bound (t219 F, t646 SWAP_FILL -0.163 ms for 4 NOP slots).

## Device A/B (bh-33 p100)

`ttp detach t664 -- docs/blend-dwalk-t664/drive.sh <rev>`; arms base (defaults, DWALK 0) /
new (GSPLAT_TT_BLEND_DWALK=1) / aa (base again, same JIT cache), 4 rotated b2b rounds.
Results: see RESULT below (filled in after the run).

## RESULT (t679, from the t664 drive on bh-33 p100, 2026-10-13 19:34-19:57 PDT)

Code: 33e38f1f (= best-iter-229 + DWALK, parent is best-iter-229 exactly), on
ttp/t679-r1-at-229. One build, arms differ only by GSPLAT_TT_BLEND_DWALK (JIT define).
render/run.py --back-to-back, 30 bicycle views, 20 passes, 4 rotated rounds (r0 = warm-up).

| round | base (DWALK 0) | new (DWALK 1) | aa (base again) |
|---|---|---|---|
| r1 | 8.068 | 7.955 | 8.059 |
| r2 | 8.063 | 7.960 | 8.065 |
| r3 | 8.061 | 7.947 | 8.059 |
| r4 | 8.059 | 7.949 | 8.063 |
| mean | **8.063** | **7.953** | 8.062 |
| spread (max-min) | 0.009 | 0.013 | 0.006 |

b2b delta new - base: **-0.110 ms/view (-1.4 %, 124.0 -> 125.7 FPS)**. A/A delta -0.001,
A/A spread 0.006, so the delta is ~10x the same-day noise. Every round of new is below every
round of base and aa. Paper projection was -0.15 (got 73 % of it).
Latency (no --dump-views, secondary): 7.994 -> 7.887 ms/view (-0.107).
JIT check (opt/ab_jit_check.py): JIT-HASH OK (matblend_compute base 8820795738948860400,
new 5512001582297800429; only the blend kernel differs).
Correctness: raw_md5 1d372b94 in every arm; 30-view sweep md5 077caaba (= best-iter-229),
0 of 30 views differ; hero opt/metal-screenshots/ttw-t664/hero.png + hero_diff10.png,
PSNR 42.85 dB vs benchmarks/reference_v2/hero.png, golden match, max 0 LSB. Hero inspected
by eye: no tile seams or block artifacts; diff shows only edge detail (spokes), as before.
Outputs: docs/blend-dwalk-t664/out/ (drive.log, r1-r4.out, lat.out, shot.out, jit-*).

Verdict: KEEP (needs review; then flip the default to 1 and land as the next iteration).

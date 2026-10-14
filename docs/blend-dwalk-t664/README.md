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

# t646: blend SFPU body trim (U9) — fill the pad NOPs, size pair param sharing

Base: best-iter-227 kernel (BLEND_SCHED 2, EXP1 1). Source lever: #630 U9
(docs/profile-t630/RESULT.md on ttp/t630-fold1-fixed-cost 00055b83).

## Step 0 (repeat check)

Not a repeat. structural-plan-2/3 do not list it; t205 (`docs/blend-loop-model-t205/README.md`)
listed "replace the swap-pad nops with useful loads" as untried ("unknown if the pad is a data or
structural hazard"). t219 only reordered for MAD stalls and kept every swap pad.

## Change (commit 951e2cd2, knob `GSPLAT_TT_BLEND_SWAP_FILL`, default 1)

The EXP1 middle had 4 SFPNOPs: one after each of the 3 SFPSWAPs (min(power,0), max(xlog2,0),
min(alpha,0.99)) and one between the xlog2 MAD and the swap that reads it. The 7 loads of the
middle (C0 OP K99 FL CR CG CB) now sit in those slots, each into an LREG that is dead at that
point. Every swap and MAD result still has one slot before its reader. Same ops, same operands:
24 -> 20 middle ops, 58 -> 54 SFPU slots per live microblock, and 4 fewer RISC pushes (the
middle is pushed, not replayed).

The swaps cannot go: they are the three clamps. A conditional (setcc/mov/encc) is 3 ops vs 2.

Check: `docs/exp1-t584/check_exp1.py` (run by `tests/unit/test_blend_exp1.py`) expands all 32
A2 singles with FILL 1 and runs them symbolically against FILL 0 (`schedule.run`): 32/32 same
DEST writes and constant LREGs, 4 ops fewer, no sfpnop, no op touching a swap's LREGs right after
it, MAD stall count unchanged. Mutations (CB load before the MUL that reads OP; EXEXP right after
swap 2) fail it.

Open question the device answers: whether the pad after SFPSWAP is a structural hazard (any
following op) rather than a data hazard. md5 30/30 golden on the FILL 1 arm decides.

## Pair param sharing: sized, not built

- Calls with both microblocks live (PM=3): t148/t189 measured 1.275 microblocks per body call,
  so about 27.5% of calls are pairs (about 0.6 pairs per live record at 2.8 microblocks/record).
- What a pair could share: 12 param loads per second microblock (MX MY A B C in the front, C0 OP
  K99 FL CR CG CB in the middle). Only 8 working LREGs exist (L9-L14 hold constants), and the
  body uses all 8, so they cannot all stay live. t219 level 1 (shared-load pair, 109 ops vs 120)
  was ~1.0 ms/view slower than A2 because pushed ops cost more than replayed ones.
- Upper bound: 12 slots x 0.6 pairs / (2.8 x 54 slots) per live record = about 4.8% of blend SFPU
  slots, about 0.2 ms traced if the body were purely SFPU-issue bound; it is not (t230: A2 loses
  60-100 cycles per live record to TRISC1 issue), and sharing needs a different replay layout with
  more pushes. Not worth building: below the bar after the push cost.

## Device A/B

bh-33 (p100 lock), commit 1da40a62, both arms built from the same tree; FILL 0 is the
best-iter-227 kernel. Log: `out/` (run logs per round), driver log run 1793 `t646ab.log`.

- md5: both arms `MD5_GOLDEN_OK list 077caaba = golden (30/30 views)`; b2b raw_md5 1d372b94 in
  every round of both arms, identical across passes. So the swap pad is a data hazard only: the
  filled loads after SFPSWAP are safe.
- Hero shot (FILL 1): `opt/metal-screenshots/ttw-t646-swapfill/hero.png`, diff `hero_diff10.png`,
  PSNR 42.85 dB vs `benchmarks/reference_v2/hero.png`, golden_match true, max_lsb_vs_golden 0.
  Looked at: no tile seams or block artifacts; the diff is the usual fine-edge residue (spokes,
  foliage).
- b2b ms/view (render/run.py --back-to-back, 30 views, 20 passes), warm round r0 then 4 ABBA rounds:

| round | FILL 0 (base) | FILL 1 (new) |
|---|---|---|
| r0 (warm) | 8.460 | 8.302 |
| r1 | 8.459 | 8.305 |
| r2 | 8.449 | 8.309 |
| r3 | 8.484 | 8.320 |
| r4 | 8.462 | 8.320 |
| mean r1-r4 | 8.4635 | 8.3135 |

Delta **-0.150 ms/view b2b (-1.77%)**, 118.2 -> 120.3 FPS. Every new round beats every base
round (spread within an arm 0.035 ms). Latency-mode avg (dump runs, secondary): 8.6 -> 8.4 ms.

Larger than the 4/58 slot share alone predicts on the blend phase (~0.07 ms if blend were
SFPU-issue bound): the 4 removed pushes also cut TRISC1 issue work, which t230 named as the
A2 limiter.

## Verdict

Keep. Candidate for the next best-iter (BLEND_SWAP_FILL default 1). Pair param sharing not built
(sized above, below the bar).

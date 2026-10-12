# B4' step 3a (#546): NT=1 town sort on TRISC 2 — standalone A/B

Commit 81d15953 (code 16ea0176), bh-33 (p150, IRD measurement reservation), 2026-10-12 14:04-14:14 PDT.
Base: ttp/t544 @507a0e60 (alias fifo cut on). Driver ran inside `ttp lock p100`.

## What changed
- `sort_ol_town` takes `GSPLAT_TT_OL_TOWN_NT` (default 3). NT=1 runs the per-record emit on TRISC 2 only;
  TRISC 0/1 exit at once. Movers loop over NT mailboxes.
- `OL_TOWN_ADDR_RTA=1`: mailbox/arena addresses come from runtime args (a1=0 = one stream), so the
  town can later sit inside the fused mat+blend program at offsets the host picks.
- Town protocol unit test covers NT=3 and NT=1 (fails on the old header with NT=1).

## Correctness (NT=1)
- 30-view sweep md5 **906e0435** (30/30 identical to golden), hero golden_match=true, max LSB 0.
- Hero PSNR vs benchmarks/reference_v2/hero.png: **42.51 dB**.
- Device screenshot: `opt/metal-screenshots/ttw-t546/hero.png`, diff `hero_diff10.png`. Checked by eye:
  no tile seams or block artifacts; diff shows only the usual thin edges (spokes, foliage).

## b2b ms/view (render/run.py --back-to-back, 30 views, 20 passes each, alternating)
| round | NT=3 (default) | NT=1 |
|---|---|---|
| r0 | 8.999 | 10.667 |
| r1 | 9.002 | 10.664 |
| r2 | 9.001 | 10.664 |
| r3 | 9.003 | 10.667 |
| **median** | **9.002** | **10.666** |

NT=1 standalone costs **+1.66 ms/view** (blend stage +1.53 ms, where the one-launch sort's device time
lands; sort_bin_emit +0.11 ms host). Stage breakdown r2: NT=3 project 2.10 / sort 0.83 / blend 5.89;
NT=1 project 2.11 / sort 0.96 / blend 7.42.

## Verdict
Step 3a is a building block, not a lever: **shelved as a standalone setting** (default stays NT=3).
It is correct (md5 30/30) and gives the RTA-addressed one-thread town that fusion needs.

The number matters for step 3b: the town emit on one TRISC adds about 1.6 ms of device time over
three TRISCs. Fusing sort(N+1) into mat+blend(N) only wins if that single-TRISC town work hides in
TRISC 2 idle time during blend(N). Blend uses TRISC 2 as its packer, so step 3b must first measure
TRISC 2 idle time during blend (per-core cycle zones) before building the fusion. If idle time is
well under ~2.5 ms per view, the >= 0.4 ms gate is out of reach with NT=1 on TRISC 2 and the
alternative is the town on NCRISC/BRISC spare time or a lighter per-record emit.

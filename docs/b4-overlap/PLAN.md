# B4 cross-frame persistent overlap: sizing note (task #591)

2026-10-13 00:40 PDT. Sizing only, no device run. Base: **best-iter-227**, 8.454 ms/view b2b
(bh-33 p100, 110 cores; md5 golden 077caaba). Inputs are existing profiles and results, cited
below; every gain here is a model estimate, not a measurement.

**Verdict: shelve B4 (the remaining part). Expected b2b gain is below the 0.1 ms/view gate for
every design that differs from the shelved L4 and SIB.** No code was built and no device time used.

## What of B4 is left

B4 was "k2 + sort of view N+1 inside mat+blend of view N" (#539 re-bound, 9ceb255f).

- k2(N+1) in blend(N) = **B4a, landed** (iter 226, -0.375 ms; GSPLAT_TT_B4A_FOLD default on).
- sort(N+1) in blend(N) = **B4' / SIB, shelved** (#573: 11.5 vs 8.67 ms; static CBs and kcfg full).
- one persistent sort(N) + pfwc(N+1) + mat+blend(N) launch = **L4, shelved** (#579: code 246-250 KB
  vs a 105 KB kcfg ring, 2.37x; combined BRISC stack margin < 256 B).

The fold fixes the order `sort(N) -> pfwc(N+1) -> mat+blend(N)+k2(N+1)`: sort(N+1) needs k2(N+1),
which ends with blend(N), so sort and blend stay serial. What the overlap can still hide is the
**program boundaries**: end spread pfwc 0.105 + sort 0.092 + mat+blend 0.146 + gaps 0.040 =
**0.383 ms/view ceiling** (t224 profile, docs/structural-plan-2/PLAN.md). Everything else is
SFPU-bound core work that overlap does not shrink (t224: blend TRISC1 ~97 % busy, pfwc TRISC-bound
t484).

## Designs that differ from L4 and SIB, sized

| # | design | how it differs from L4 / SIB | ceiling | expected | blocker |
|---|---|---|---|---|---|
| V1 | **Disjoint-core pipeline**: pfwc(N+1) on core group A, sort+mat+blend(N) on group B, as separate programs on tt-metal sub-devices; pfwc output triple-buffered in DRAM | no combined kernel: each program keeps today's code size and stack | 0.12 (pfwc boundary leaves the critical path) | **+0.07 at best** (`model.py`: a = 20 cores, period 8.381) | fewer cores per stage; sort town layout re-tune for 90 cores; sub-device sync cost not counted |
| V2 | **Barrierless hand-off**: each core starts pfwc(N+1) as soon as its own sort(N) ends, pfwc chunks claimed dynamically | per-core early start instead of a merged program | 0.105 (sort end spread + one gap) | **<= 0.05** | needs a tt-metal dispatcher change (the go signal waits for every worker of the previous program); pfwc rebalancing already moved 0 untraced (t439 LPT deal: max/mean 1.144 -> 1.096, +0.011 ms/view) |
| V3 | **Resident per-frame state** across frames | keeps data, not code, on device | - | **0 on the bench** | gaussians 368 MB/view vs 165 MB total L1, which static CBs already fill (t565, t302); sort reuse shelved (#529, 0 % identical tiles); mat depth-order reuse is viewer-only (#539: 0.2-0.4 ms at slow motion, nothing at bench motion) |
| V4 | pfwc(N+2) on the blend(N) tail (#539 item 4) | - | 0.32 | - | the same kcfg limit as L4: pfwc (95 KB code) + mat+blend (97 KB) in one program |

V1 model (`model.py`, output in `model_out.txt`): 888 core-ms of work per view, pfwc 17.7 %.
Moving pfwc to its own group removes its boundary from B but gives B 20 fewer cores, so only the
pfwc spread (0.105) minus core quantization is won. It assumes perfect linear scaling and free
cross-sub-device sync; both are optimistic (blend per-core imbalance grows at 90 cores, and the
sort's onelaunch town layout is tuned to the full grid). Per-view pair jitter of +-5 % does not
change it, because group A has slack at the best split.

## Budgets (for V1, the only one that fits)

- **kcfg**: unchanged per program. Largest is mat+blend + K2 fold, 96,944 B code at 226, about
  95.7 KB with the t574 fix (in 227), under the 101,376 B go line (#579 elfscan). V1 adds no code.
- **L1**: unchanged per core. The extra pfwc output buffer is in DRAM (third proj_m_blendrec copy).
- **BRISC stack**: unchanged per kernel. #579 measured matblend_brisc (+K2 fold) at 368 B margin
  with the t574 fix; the t577 K2_CNT16 change (in 227) cuts the K2 count row by about 2 KB more.
  Both are above the 256 B line. Not re-measured with brisc_stack_check.py on the 227 cache,
  since no build was made.

## Why stop here

V1 is the only design that fits L1, kcfg and the stack, and its best case (+0.07 ms/view) is below
the 0.1 gate before its unmodelled costs. V2 is smaller and needs a firmware change outside this
repo. V3 and V4 give nothing on the bench or hit the L4 limit. With k2 hidden (B4a) and sort
fixed in series by the fold, the B4 family is exhausted on one chip. The throughput lever that is
left is the second chip (L2 AFR), which is measured separately.

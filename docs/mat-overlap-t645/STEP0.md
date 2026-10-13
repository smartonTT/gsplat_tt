# t645 step 0: overlap mat with blend per core in FOLD=1 — stopped (re-queue, no new evidence)

Task #645 asked to overlap each core's mat phase with its blend, sized from #630
(`ttp/t630-fold1-fixed-cost` 00055b83, `docs/profile-t630/RESULT.md`): overlap bound 1.26 ms traced,
realistic 0.6-1.0 ms/view. Step 0 (user msgs #299/#300) checks the lever against earlier results
before any build. No code changed, no device time used.

## Why the 1.26 ms bound does not hold

#630 takes the bound as window minus max(blend + drain), as if the mat phase were pure waiting.
It is not. The mat phase is mostly TRISC compute, the same TRISC that is the blend bottleneck
(c_blend is compute-bound, c_cntw = 0):

- #595 (`ttp/t595-profile-227` 8692f3f1, `docs/profile-t227/PROFILE.md` §1, same best-iter-227 build):
  TRISC1 in mat = 1.420 ms mean / 2.039 max, of which **band_batch (SFPU cull) 1.077 / 1.237**.
  TRISC1 idle in mat = 0.331 ms mean.
- Overlap only helps where the TRISC would otherwise wait. Filling every core's TRISC1 mat idle with
  its own blend work shortens the slowest core 5.235 -> 4.937 ms: **upper bound 0.298 ms/view traced**
  (~0.27 untraced). #595 labels this "same lever as spec B".
- Moving the rest of mat's mover work (sort_subchunk_mat) into the NCRISC bulk-wait does not
  change this: the cull of those items still runs on TRISC1, so the TRISC total (cull + blend) is
  unchanged and the bound is the same 0.30.

## Earlier results on this exact lever

| task | lever | result |
|---|---|---|
| #303 (`docs/lane-model-t303.md`) | blend lane during mat | 0.27 ms realistic, under its 0.4 gate |
| #419 (`ttp/t419-blend-interleave-model` b1bc4af2) | blend in TRISC1 mat gaps (L1a) | **shelved**: whole-tile fill 0.078 ms, idealised split 0.277; median gap 16 µs vs ~265 µs per tile; L1 blocker: blend CBs alias mat CBs, only 0-34 KB free while mat runs |
| #595 spec B (best-iter-227) | mat fill jobs + big split / T1 idle fill | **killed**: ceiling 0.125-0.179 ms/view (direct 0.020), upper bound 0.298 |
| #595 spec C | mat band cull on FPU | killed: 0.037 ms/view |

#630 adds one correction: LEDGER §2 said mat "overlaps blend"; in FOLD=1 it runs before blend on
every core. That does not change the sizing above: #303, #419 and #595 all already modelled mat as
serial before blend per core.

## Conclusion

Re-queue of a shelved lever with no new evidence that the earlier result was wrong. Real bound
<= 0.30 ms/view traced (realistic 0.08-0.18), with an unsolved L1 blocker. Not built.
What #630 does point at is cutting the mover work itself (sort_subchunk_mat ~1.36 ms whatever P is;
#419 sized mover-only cuts at 0.33 / 0.54 ms for 30 / 50 % shorter gaps), which is a different lever.

# t727: where the fp32 train-mode blend cost goes (bh-49 p150a, 2026-10-14)

Code: 5df7e221 (t724 path + gather knobs, defaults unchanged). Driver docs/t727/drive.sh,
outputs docs/t727/out. b2b = render/run.py --back-to-back, 30 bicycle views, 20 passes,
GSPLAT_TT_BLEND_TRAIN=1. Rounds r0-r4 alternated (r0 counted as warm round, r1-r4 = A/B).

| arm | what it changes | r1-r4 ms/view b2b | vs on |
|---|---|---|---|
| on  | t724 default (64 reads per barrier) | 14.681 (14.684 14.688 14.681 14.670) | - |
| b2k | 2048 reads per barrier | 14.661 (14.641 14.725 14.639 14.638) | -0.02 |
| sd  | skip mask-0 records | 14.641 (14.639 14.638 14.643 14.643) | -0.04 |
| sdb | 2048 per barrier + skip mask-0 | 14.600 (14.607 14.601 14.596 14.597) | -0.08 |
| d1  | diag: no DRAM reads, L1 copy kept (output wrong) | 13.624 (r0-r2) | -1.06 |
| d2  | diag: no gather and no copy (output wrong) | 13.261 (r0-r2) | -1.42 |
| t717 UNORM16 train (t724 RESULTS) | | 12.923 | -1.76 |

Breakdown of the 1.76 ms/view fp32 over UNORM16:
- random 64 B DRAM gather of fp32 op/colour on the blend reader (BRISC): ~1.06 ms
- scalar L1 copy staging -> side CB (4 words per record): ~0.36 ms
- compute side (TRISC0 staging 2 SFPLOADI words per value in 128 B slots, TRISC1 stores,
  wider ring): ~0.34 ms (d2 vs UNORM16)

Cause: the reader is the critical engine in train mode. CB_TRAIN_OPC is already double
buffered (2 x 4096-record slots), so the gather does not hide behind compute; it adds
straight to the reader's per-subchunk time. Barrier count is not the problem (b2k: noise)
and dead records are few (sd: noise). The per-read issue/transfer cost of ~one random
64 B read per record is.

Knobs b2k/sd/sdb: all under the 0.1 ms noise floor; defaults left unchanged.
JIT check (docs/t727/out/jit-check.txt): reader_alpha_blend_mb_devcull hash differs
on vs sdb/sd/b2k (JIT-HASH OK).

Lossless: fp32 dump view 0 md5 3fada021 for on and sdb (= t724/t717 reference).
Shots: hero 42.52 dB vs benchmarks/reference_v2/hero.png for on and sdb, sweep md5
cc822a42 both, diff image edge-structured only, no tile artifacts seen.

Zone profile: the TT_METAL_DEVICE_PROFILER runs failed before rendering on this tt-metal
build (idle_erisc.elf segment 0x4668 B overflows the 0x4490 B region when the profiler is
on), so the breakdown above comes from the diag arms, not zones.

Fix not built in this task: removing the gather means carrying fp32 op/colour in the
sorted data (a 16 B side slab per record written by sort_subchunk_materialize, read
contiguously like the 32 B slab). Every materialize path (in-budget L1 permute, overflow
sc==0 / sc>=1 gather, onelaunch) must emit it, and the in-budget path gets its records
from buf_l1_recs, so the op/colour must come from bin_emit (which walks gaussians in id
order, so contiguous) or be gathered again in materialize. Bound: up to ~1.42 ms/view
(d2), minus what the extra slab write and read cost. Queued as a follow-up.

# #365: p150 emit mover throughput

Question: why sort `bin_emit` is ~2.4 ms/view on bh-30 (p150b) vs 0.57 ms on yyzo-bh-04 (p100a).

**Answer: it was ours.** The tile bucket's tile stride was 512 pages of 2 KB (1 MiB). The p150 has
8 DRAM banks, and 512 % 8 = 0, so page k of every tile sits in the same bank (k % 8), at the same
128 KB-aligned offset inside that bank. The emit's ring flushes (<=256 B writes) then pile onto a few
banks and collide inside them. The p100a has 7 banks and 512 % 7 = 1, so its writes rotate.
Fix: the tile stride is now the next page count coprime with the bank count (513 pages on 8 banks,
512 unchanged on 7). Measured on bh-30: **12.896 -> 11.094 ms/view (-1.80 ms, -14.0%)**,
`bin_emit` 2.43 -> 0.62 ms, the same as the p100a (0.57-0.58). The p100a does not change (no
regression; the auto rule keeps 512 there).

Note: the host `bin_emit` stage is the blocking CQ1 read of the K2 rows. It ends when the device
emit ends (#356).

## Per-cause table (ms/view = median of the per-run means over 30 views, >=3 runs per arm, same box)

| Cause tested | Arms | bh-30 p150b ms/view | bh-30 bin_emit | p100a ms/view | p100a bin_emit | Verdict |
|---|---|---|---|---|---|---|
| DRAM bank placement of the bucket writes (tile stride) | P0 = pad 0 (512 pages, pre-#365) vs P1 = auto (513 on 8 banks) | **12.896 vs 11.094** (runs 12.81/12.90/12.97 vs 11.09/11.10/11.08) | **2.43 vs 0.62** | 10.956 vs 10.913 (same stride, noise) | 0.579 vs 0.569 | **The cause. Fixed, default on.** |
| Same, forced pad on 7 banks | P0 vs F1 = pad 1 (513 pages) | (auto is already 513) | - | 10.956 vs 10.859 (-0.9%, runs 10.97/10.86/10.84) | 0.579 vs 0.566 | Under 1%, within noise: not forced on 7 banks |
| Read transfer size (64 B vs 256 B runs) | microbench R64_d16, R256_d8 | per core p150 >= p100a (table below) | - | - | - | Not p150-specific |
| Write size (32 B vs 256 B) | microbench W32 vs W256 at stride 513 | 0.38 vs 2.08 B/tick (NoC0) | - | 0.39 vs 2.00 | - | Same on both boards |
| NoC0 vs NoC1 per row/column | microbench maps | same gradients as p100a | - | - | - | Not p150-specific |
| Rows y=2-4 read stalls | microbench R256 rows | rows 2-4 0.88-0.92 B/tick | - | 0.78-0.82 | - | Same pattern on both; #358's late y=2-4 cores were the write camping (they own the first pages of every tile) |
| Outstanding depth, mover placement | not A/B'd in the kernel | - | - | - | - | Not needed: after the fix the emit matches the p100a (0.62 vs 0.58 ms) |

md5: 906e0435 on 30/30 views in every run: bh-30 P0 x3, P1 x3; p100a P0 x3, P1 x4, F1 x3.

## Microbench (step 0): bytes per RISC tick, single-RISC sets, 11x10 grid, all cores at once

`render/bench/kernels/p10_emit_noc.cpp`, `MB_SUITE=emit render/bench/build/risc_microbench`.
Reads: random 64 B / 256 B runs inside one bank of a 64 B-page buffer, 16 / 8 outstanding per
barrier. Writes: 256 B / 32 B into a 2 KB-page bucket at page `t*stride + k`, 8 / 16 per flush.
AI clock 1350 MHz on both boards (read-only sysfs `tt_aiclk`; tick rate 1350.2 vs 1350.1 MHz).
Full per-core maps: `[MBMAP]` lines in `out-p150/mb.txt`, `out-p100a/mb.txt`;
`python3 mbmap_summary.py out-p150/mb.txt` prints the row/column means.

| Probe | NoC | p150 mean (min-max) | p100a mean (min-max) | p150 agg GB/s | p100a agg GB/s |
|---|---|---|---|---|---|
| R64_d16 | 0 / 1 | 0.37 (0.24-0.60) / 0.28 (0.26-0.33) | 0.34 / 0.25 | 35 / 38 | 31 / 33 |
| R256_d8 | 0 / 1 | 1.14 (0.88-1.52) / 1.01 (0.94-1.10) | 1.09 / 0.90 | 131 / 140 | 116 / 121 |
| W256 stride 512 | 0 / 1 | **0.40 (0.13-1.78) / 0.39 (0.16-2.23)** | 1.35 / 1.49 | **20 / 23** | 83 / 103 |
| W256 stride 513 | 0 / 1 | 2.08 (0.96-4.23) / 2.73 (1.46-4.22) | 2.00 / 2.48 | 143 / 216 | 130 / 198 |
| W32 stride 512 | 0 / 1 | **0.05 / 0.05** | 0.27 / 0.29 | **3.2 / 3.4** | 20 / 23 |
| W32 stride 513 | 0 / 1 | 0.38 / 0.38 | 0.39 / 0.39 | 33 / 33 | 36 / 38 |

Single core, no contention: reads are equal (R256 2.67 vs 2.60 B/tick), and so are writes at stride 513
(W256 4.23 vs 4.03). At stride 512 even one p150 core drops to 3.22 (p100a 4.10), far below one bank's
bandwidth, so each write takes longer inside the DRAM, not just in a queue.
Row/column shape (both boards alike): NoC0 rises with y (p150 R256 0.91 at y=2 to 1.51 at y=11),
NoC1 falls with x (p150 W256 s513 4.21 at x=1 to 1.64 at x=13).

Reading it: the p150 reads as fast as the p100a; only the stride-512 writes collapse (5-7x).
The microbench spreads its movers evenly over the 8 banks and still collapses at stride 512, so the loss
is not only uneven bank load. Every tile's page k is at the same 128 KB-aligned offset inside its
bank, so writes to different tiles likely conflict inside the DRAM (GDDR6 bank/row). We cannot observe
that directly. The p100a's own s512 drop (1.35 vs 2.00) is an artifact of this probe: its tile step
of 7 equals the p100a's bank count, so each core stays on one bank. The real kernel shows no such loss
(F1 arm, -0.9%, noise).

## Fix (on this branch)

- `render/host/sort_onelaunch_layout.h`: `bucket_tile_cap(cap, nbanks, pad)`; `static_assert`s:
  7 banks unchanged, 8 banks 513 pages, the materialize big-path limit still holds.
- `render/host/sort_device.cpp`: applies it to the default cap, logs `bucket tile stride N pages, B DRAM banks`.
- `render/host/env_config.h`: `GSPLAT_TT_OL_TILE_PAD` (unset = auto, 0 = pre-#365 stride, n = n extra pages).
- `tests/unit/test_bucket_stride_t365.cpp`: auto, pad 0, forced pad, big cap.

## Acceptance

- md5 906e0435 on 30/30 views in all 16 runs (bh-30: 6; p100a: 9 in `out-p100a-ab`, 1 in `out-p100a`).
- Device-rendered hero (bh-30 P1, `out-p150/hero-r1-P1.png`; p100a `out-p100a/hero-r1-P1.png`, the same
  md5 86524912): **42.51 dB vs `benchmarks/reference_v2/hero.png`**, max abs diff 46, 0.822% px > 8
  (`out-*/psnr.txt`). Diff x10: `out-p150/hero_diff10.png`. I looked at it: only edge residue on the
  spokes, bench slats and foliage, the same as iter 209. No tile seams, no straight tile-grid lines.
- bh-30 viewer: stopped 2026-10-07T22:58:16Z, restarted 23:00:51Z, ready 23:00:59Z, localhost:8091 -> 200.
  Microbench and all 6 bench runs ran inside that window (`docs/.../drive_bh30.sh`, nice/ionice build).
  The viewer still runs its own deployed sha 4307f076, without this fix.

## What is left on the p150 (P1 vs p100a, stage means)

bh-30 P1 11.089 vs p100a 10.888 ms/view (+0.20). Device stages are now faster on the p150 (blend 6.31 vs
6.67, project 2.99 vs 3.04). The rest is host time on bh-30: d2h 0.45 vs 0.22, publish_host 0.35 vs 0.19,
pfwc_rtargs 0.10 vs 0.05, sort_mat 0.10 vs 0.06 (+0.47 ms together). Next lever: the bh-30 host path
(d2h and publish_host, ~2x slower than on yyzo-bh-04). The emit gap itself is closed.

Scripts: `drive_bh30.sh` (sync/build, viewer stop/bench/restart), `drive_p100a.sh` (under `ttp lock p100`),
`bench365.sh` (arms P0, P1, P2, F<n>; MB=1 runs the microbench).

# L2: alternate-frame rendering (AFR) on 2 chips — result (task #582)

**2 chips (AFR): 4.226 ms/view back-to-back (236.6 FPS)** over the 30 bicycle views, against
**8.395 ms/view (119.1 FPS) for one chip on the same host and build: 1.99x**. Published GPU G1
reference: 10.75 ms/view (published, not measured), so 2 chips are 2.54x faster than it.
This is a side result. The single-chip b2b ms/view stays the headline. It is not an iteration and
gets no best-iter tag.

## Setup

- Host: **bh-50** (2x Blackhole p150, IRD cluster tt_bgd), reached only through yyz-ird with
  strict host-key checking. Not bh-30, not the measurement box (bh-33).
- IRD job **132985**, reserved 2026-10-12 23:24 PDT (2 h). Released by the driver at
  **23:39 PDT**: ird printed `132985 on bh-50: Reservation released!` and the job was gone from
  `ird list` at 23:39:09 PDT (`out/release.txt`). Earlier attempts: 132917, 132945, 132949, 132971,
  each released (outputs in `out/attempt*`).
- Code: commit `ed027bb4` (this branch: #581's AFR host path `031fbbd3` merged with best-iter-226,
  plus `opt/md5_golden.py` accepting `[afrN]`-prefixed `[DEV]` lines). Same tree and build for
  all arms. The 11x10 grid; tree synced by `opt/sync_remote.sh` (built .so md5 68ee5af7).
- Driver: `docs/l2-2chip/drive.sh` (local; `ttp lock p100`, ssh preflight, then
  `ttp detach --remote bh-50`) runs `docs/l2-2chip/remote_drive.sh` on the box.

## Commands

All with `GSPLAT_TT_DISPATCH=worker TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P150`:

- A (one chip, id 0): `TT_VISIBLE_DEVICES=0 GSPLAT_TT_DEVICE_ID=0 python3 render/run.py --no-ref --back-to-back`
- B (2 chips AFR): `GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=0,1 python3 render/run.py --no-ref --back-to-back`
- C (chip 1 alone): as A with `TT_VISIBLE_DEVICES=1`
- Correctness, separate untimed runs: A in latency mode with `--dump-views`, B with `--back-to-back --dump-views`.
- Latency (secondary): A without `--back-to-back` and without `--dump-views`.

Order: dumps, a discarded warm-up round (A, B), then rounds r1 = A B C, r2 = B A C, r3 = A B C,
then 3 latency runs. Every step exited 0 (`out/remote/steps.txt`).

## Numbers (ms/view, per pass = that run's ms_frame_median)

| arm | r1 | r2 | r3 | median | FPS |
|---|---|---|---|---|---|
| A one chip (0), b2b | 8.406 | 8.395 | 8.390 | **8.395** | 119.1 |
| **B 2 chips AFR, b2b** | 4.221 | 4.227 | 4.226 | **4.226** | **236.6** |
| C one chip (1), b2b | 8.394 | 8.387 | 8.392 | 8.392 | 119.2 |
| A latency mode (secondary) | 8.405 | 8.420 | 8.415 | 8.415 | 118.8 |

AFR has no separate latency mode: each view's latency is one chip's (the A row).

MB1, each chip's own-view time inside AFR (`afr_chip_ms_own_view`, chip 0, chip 1):
r1 8.443 / 8.332, r2 8.454 / 8.336, r3 8.452 / 8.345. Chip 0 runs about 0.7% slower than alone
(8.45 vs 8.395), chip 1 about 0.6% faster than alone (8.34 vs 8.392). The --dump-views AFR run
gave 4.220 ms/view, the same as the timed runs.

## Correctness

| run | views | list md5 | golden (11x10) |
|---|---|---|---|
| A one chip, `--dump-views` | 30 | 906e0435 | OK 30/30 |
| B 2 chips AFR, `--dump-views` | 30 | 906e0435 | OK 30/30 |

Per-view lists: `out/remote/md5-dumpA.txt`, `out/remote/md5-dumpB.txt` (identical).
`afr_hero_equal=yes` in every AFR run (both chips render the hero identically). Every timed run
also printed its raw output md5: A and the AFR total 5a438f5a, chip 1 alone 50b707c3.

## Screenshot (device, 2-chip mode, this commit)

- `out/remote/hero-afr-chipA.png` (chip 0 in AFR), `out/remote/hero-afr-chipB.png` (chip 1 in
  AFR), `out/remote/hero-1chip.png`: byte-identical (md5 86524912, the same as view00 in the dump).
- Diff x10 against `benchmarks/reference_v2/hero.png`: `out/remote/hero-afr-chipA_diff10.png`.
- **PSNR vs benchmarks/reference_v2/hero.png: 42.51 dB** for all three (gate 42.4).
- Looked at the AFR hero and its diff myself: no tile seams, blocks or stripes. The diff is only on
  thin edges (spokes, bike frame, bench slats, foliage), plus a very faint broad smudge left of
  center, the same in the single-chip hero.

## What limits 2-chip scaling, against the 4.4-4.6 ms/view bound

The #578 bound (4.4-4.6) assumed bh-33's 8.615 single-chip b2b. On bh-50 one chip runs 8.395, and 2
chips run 4.226 = 8.395 / 1.987. AFR is within 0.7% of perfect 2x here. What is left:

1. Each chip still takes a whole view (8.33-8.45 ms). AFR multiplies throughput and leaves latency
   alone: a viewer frame still takes about 8.4 ms from pose to image. Going under that needs
   split-frame work across chips (or faster single-chip kernels, which AFR then doubles).
2. Host sharing: chip 0 is about 0.7% slower in AFR than alone (one host process drives both
   workers; host prep and readback for the two chips overlap). That is the whole gap to 2x.
3. 30 views split 15/15, so no imbalance from an odd count; the run's total is the slower chip.

Code status: the AFR host path is off by default (`GSPLAT_TT_AFR` unset) and does not change the
single-chip path (A's md5 and time match the single-chip runs). This branch merges cleanly with
origin/smarton/tt-project-opt (now at iter 227). Before it lands it needs the independent review
and a single-chip CPU/md5 check on the merged tree, because iter 227 changed the blend and its golden
(077caaba).

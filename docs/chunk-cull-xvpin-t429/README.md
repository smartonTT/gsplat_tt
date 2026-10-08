# Task #429: chunk frustum cull before pfwc, re-modelled under xvpin (CPU only)

No device runs. Inputs: #169 (`ttp/t169-chunk-frustum-cull-before-pfwc` 9f362f9f,
`docs/chunk-cull-t169`), t226/t232 pfwc per-RISC times, #407/#412 xvpin Tracy
(`docs/xvpin-tracy/README.md`), #423 plan, #397/#409 per-grid goldens. Model: `model.py`
(imports the #169 `skipshare*.py`, copied here), output `model-out.txt`. Scene
`scenes/bicycle.ply` (md5 3745b7a6, N = 6,131,954, 5989 chunks of 1024), 30 views of
`benchmarks/cameras_v2.json`, 1024x1024.

## Verdict

**Passes the 0.15 ms/view gate on the model, but not md5-exact. Build only as a staged A/B with an
early stop, and with a justified per-grid golden change** (spec below). Modelled net saving
**~0.37 ms/view (range ~0.15-0.5)**. The biggest unknown is what the Morton reorder alone costs on
today's pfwc/k2/sort. #169 never measured that cleanly (its reorder-only probe timed out), so
stage A measures it first.

## (a) Share of gaussians removed (Morton order, conservative test K = 1.5, 0 visible lost)

| chunk | mean skip share of N | min-max over 30 views |
|---|---|---|
| 256 | 0.501 | |
| **1024 (pfwc unit)** | **0.394** | 0.351-0.424 |
| 4096 | 0.258 | |

On average 27.7% of N is visible. 0 visible gaussians are lost in all 30 views. With 1024-chunks,
3631 of 5989 chunks are kept on average (3447-3890; hero 3704, the same as #169's device log).
Dealt strided over 120 cores, that is at most 30.7 chunks per core (29-33) instead of 50, so a core
does 0.614x the chunks.

## (b) pfwc device time saved

pfwc is TRISC-bound and its cost is per chunk. The SFPU runs all 1024 lanes, so a culled chunk
saves its full TRISC time. Measured per core (t232 pr0F, traced, 110 cores): TRISC 1.72-1.78;
BRISC rec 0.80 + rd 0.34 + fl 0.13 = 1.27; NCRISC rec 0.78 + rd 0.43 + fl 0.10 = 1.31; window 1.891
(#412: 1.891 window, 1.766 mean). Scaled to 120 cores (x 110/120): TRISC ~1.60, writers ~1.20,
window ~1.73.

After the cull (x 0.614 chunks per core):
- TRISC: 1.60 x 0.614 = 0.98.
- Writers: rd and the per-lane `classify_tile` scale with chunks. The visible-record writes do not
  (culled chunks hold no visible gaussian). Bounds: 0.91 if half of `rec` is classify, 1.05 if all of
  `rec` is per visible.
- Visible-per-core imbalance (max/mean) barely changes: 1.237 full scene vs 1.223 survivors (range
  1.14-1.35). So keep the ~0.12 ms end-of-program tail.

New window ≈ 1.12-1.17 traced against ~1.73: **-0.56 to -0.61 traced, ~-0.45 to -0.55 untraced**
(#423 puts pfwc at ~1.5 ms/view untraced at iteration 214). pfwc is on the serial device chain and
the device is the whole critical path (all-core idle 0.041 ms/view), so this saving reaches the frame.
The floor moves from TRISC to the writers. That is why the saving is ~0.5 and not 0.39 x 1.5 = 0.59
of untraced pfwc.

## (c) Device-side costs that xvpin does not hide

| item | ms/view | basis |
|---|---|---|
| sort/bin_emit slower on Morton-ordered gids | +0.11 (0-0.2) | #169 measured, pre-xvpin, p100a; not re-measured on today's emit |
| per-view survivor-list upload (~3.6k ids, ~15 KB CQ write between blend N and pfwc N+1) | +0.01-0.02 | estimate |
| reader/writers fetching their tile-list page | < 0.005 | one 1-page read per core |
| -Os reader (#169 had to drop to -Os to fit 69 KB kcfg) | 0 | today's pfwc program is ~96 KB under the +24/+32 KB kcfg open (t226/t232), so the kernels stay at O2 |
| Morton reorder's effect on pfwc writers / k2 / mat | **unknown** | #169: reorder plus -Os, no skip = project +0.86. The reorder-only probe hit rc 124. This is the main risk. |
| md5-exact tie-break in the mover sort (only if required) | +0.05-0.15 | estimate: one extra key store plus an equal-run scan per tile on the mat critical path |

Net: 0.5 - 0.11 - 0.02 ≈ **0.37 ms/view** with a golden change, **~0.27** if made md5-exact. Both
assume the reorder itself is close to free. If it costs more than ~0.3 ms/view, the lever is dead.

## (d) md5: an equal-key gid tie-break is NOT bit-exact

The per-tile radix (`sort_radix_tile_algo.h`) is stable on `depth_bits` (fp32 bits of tz), so ties
keep emit order. Emit order is the compact order of `vis_tile::SeqMap` (`vis_tile.h:122`): scene
chunk c goes to core c % C, slot c / C, and compact order is (core, slot, lane). So the golden tie
order depends on the core count C. That is the mechanism behind 11x10 906e0435 ≠ 12x10 39d84b28,
which #397 found but did not explain.

CPU check over 30 views (`model.py`; visible gaussians, fp32 tz, pairs with equal keys whose
3-sigma tile ranges overlap):

| per view, mean | count |
|---|---|
| tied pairs sharing a 32x32 tile (16x16) | 8443 (6677); range 6146-10869, every view |
| of these, exact duplicates (equal means) | 0 |
| of these, in the same 1024-chunk | 4.8 |
| order differs: base C=120 vs base C=110 | 4074 (every view) |
| order differs: ascending gid vs base C=120 (39d84b28) | 4095 (every view) |
| order differs: ascending gid vs base C=110 (906e0435) | 4022 (every view) |
| order differs: Morton survivors (the #169 path) vs base C=120 | 4228 |

Tied pairs almost never share a chunk, so any reorder flips about half of them. A gid tie-break
therefore matches neither golden. It would give a new, grid-independent golden. Caveat: the
device's exact fp32 op order for camera z is not reproduced, so the count of coincidental ties is
statistical. The conclusion does not depend on it: thousands of flips per view against ~0 needed.

An exact match is possible only by tie-breaking on the baseline compact rank
r_C(gid) = ((gid/1024) % C, (gid/1024) / C, gid % 1024), with C = 120 for 39d84b28 and 110 for
906e0435. That needs the original gid in the record, plus an equal-run pass after the radix in the
mover sort (+0.05-0.15 ms/view estimated), and it only reproduces an order that is itself an
artifact of the core count.

**Golden change, justified:** #397/#409 already accepted a per-grid golden for the same kind of
tie-order change (hero 242 px differ, max 13 LSB, 82 dB eth vs worker; PSNR vs reference 42.513 vs
42.512). #169's image change was the same kind (max 19 LSB, min 72.47 dB vs golden). Recommendation:
no tie-break, a new deterministic golden per grid. Accept it only if PSNR vs
`benchmarks/reference_v2/hero.png` stays within 0.02 dB of 42.51 and the visual check is clean.

## (e) Host cull cost

#169's C++ `chunk_survivors` plus setup cost +0.11 ms/view on the box (stage `project_pfwc_chunkcull`).
The numpy version here takes 11 ms/view on the Mac, which is numpy overhead for 5989 x 8 corners and
not representative. Under xvpin the host waits 6.53 ms/view on the blend event (#407), and pfwc N+1
is enqueued behind blend N. The cull for view N+1 runs in that window before the enqueue: ~0.11 of
~6.5 ms, a margin of ~60x. Hidden. The per-chunk table is per scene and cached (#169).

## Build-and-A/B spec (self-contained; for the coordinator to queue)

Title: chunk frustum cull before pfwc on the xvpin/ETH 12x10 stack, staged A/B.

Restrictions (binding, restated): work in your own git worktree of ~/dev/gstt2, never in
~/dev/gstt2 itself. Push only to a new ttp/* branch, and land on smarton/tt-project-opt only after
review. No PRs, no force pushes, no rebasing or amending of pushed commits, no deleting or moving
branches or tags. Wrap every device sync+build+run in `ttp lock p100 -- ...`. Use only the existing
measurement reservation; never reserve or release. If a p150 is needed and bh-30 is the only one,
use it only under the 2026-10-07 viewer exception (hold resource 'viewer', stop the viewer only for
the bench, restart it, tell the user). On bh-30, build with nice -n 19, ionice -c3 and -j at half
the cores. Run ssh-preflight first and never disable host-key checking. Never search / or the home
folder.

1. Port #169 (`origin/ttp/t169-chunk-frustum-cull-before-pfwc`): `run.py` `morton_reorder`, then
   `pfwc_device.cpp` `chunk_table` / `chunk_survivors`, the per-core tile list in one DRAM page per
   core, and `PFWC_TILE_LIST` in the reader. The fused writer is now `writer_pfwc_split.cpp`
   (BRISC/NCRISC by chunk parity k, with a prefix chain k -> k+1). Both roles must take chunk ids from
   the list, with the parity still on the list position k. Double-buffer the list buffer, because
   pfwc N+1 is queued while N may still read it, or write it in order on the same CQ. Keep all
   kernels at O2; raise the kcfg open if needed. Env flags: `GSPLAT_TT_CHUNK_CULL` (default 0),
   `GSPLAT_TT_CHUNK_SKIP=0` (reorder only), `GSPLAT_TT_CHUNK_K=1.5`.
2. Stage A (stop check): reorder only (`CHUNK_CULL=1 CHUNK_SKIP=0`) vs off, ETH 12x10, 3 alternated
   rounds x 30 views, same box for both arms. Record frame, project, sort, mat+blend stage timers.
   **Stop and shelve if reorder-only costs > 0.3 ms/view on the frame**, with the per-stage split.
3. Stage B: full cull vs off, 3 alternated rounds x 30 views, same box. Keep if the frame mean is
   ≥ 0.15 ms/view faster and every round is faster. Log the kept chunk counts (`GSPLAT_TT_CHUNK_LOG=1`;
   expect ~3631 mean, hero 3704). If the gain is below the model, take one 11x10 worker Tracy
   (`ana407.py`) of both arms and split pfwc TRISC/BRISC/NCRISC, k2 and sort/emit.
4. Image (required for the iteration): render the bicycle hero on the device at the candidate commit
   and config. Save hero.png, the diff image and PSNR against `benchmarks/reference_v2/hero.png`
   (label that reference). Expect ~42.51 dB ± 0.02. Report the pixel count and max LSB versus the
   current 12x10 golden hero as a separate badge. Someone must look at hero.png and the diff for tile
   artifacts. Check determinism: md5 list identical in 3/3 rounds, which becomes the new per-grid
   golden (12x10, and 11x10 if run). Update the golden check (`docs/eth-default-t409`) only in the
   landing task.
5. If kept: add an iters.jsonl entry with the screenshot, rebuild REPORT.html, tag best-iter-<N>.
   If not: shelve with the stage A/B numbers.

## Files

- `model.py`, `model-out.txt`: this model (python3 with numpy; ~5 min on the Mac).
- `skipshare.py`, `skipshare_safe.py`: copied unchanged from #169.

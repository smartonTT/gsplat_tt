# Single-frame split across 2 p150 chips: sizing (task #616)

Written 2026-10-13. Model only, no device runs. Baseline is best-iter-227: 8.451 ms/view b2b on one
chip (bh-33). AFR on 2 chips is 4.226 ms/view (bh-50, t582).

**Verdict: GO.** A left/right tile split with pfwc run on both chips models at about 6.2 ms per
frame including the host join (range 5.7 to 7.1 ms). That is a 27% cut (range 16% to 33%) against the
15% bar. b2b on 2 chips is 5.6 to 6.7 ms/view: better than one chip (8.45) but worse than AFR
(4.23). So the split is a latency mode. AFR stays the throughput mode.

## 1. What splits and what is duplicated

| stage (1-chip ms) | split by tile? | on 2 chips |
|---|---|---|
| pfwc projection + cull (1.50) | no. Each chip must project all N gaussians to know which touch its half | run on both chips. Only the pair count and the kept set shrink: 1.50 -> 1.32 |
| K2 pair emit (0.27) | yes, emit only pairs whose tile is in the chip's window | 0.27 -> 0.16 |
| sort, per-tile bins (1.44) | yes, bins are per tile | 1.44 -> 0.99 |
| mat + blend (5.21) | yes, fully per tile | 5.21 -> 3.27 (slowest chip) |

Slowest chip, left/right static split, means over the 30 views (model scaled to 8.451).

**Duplicate pfwc vs split the gaussians and exchange.** Splitting the gaussians saves at most about
0.5 ms of pfwc's per-gaussian fixed cost. But each chip then has to ship the projected records that
touch the other half: about 0.42M records x 32 B = 13-14 MB per direction per view. Through the
host that is 1.4-2.8 ms. Over ETH it would be 0.3-1.4 ms plus sync, and the ETH link between the
two boards is not known to be cabled. It also breaks the storage-id order that keeps depth ties
bit-exact. Rejected: duplicate pfwc.

**Join.** Each chip writes only its own tiles. Both write into one 1024x1024 RGB8 image (3 MB) at
full row stride. If one host buffer can be pinned for both devices (`PinnedMemory::Create` takes a
user host buffer, blend_device.cpp:814), the join is free. If not, each chip has its own pinned
buffer and the host copies one half across: 1.5 MB, about 0.1-0.2 ms. The host also has to start
both chips and wait for both. Budget 0.1-0.3 ms for sync and join, plus about 0.7% for two worker
processes sharing the host (seen in AFR).

**Bit-exactness: use a tile window, not a shifted viewport.** A sub-viewport with a shifted
principal point (cx - 512) changes fp32 rounding of the projected means, so pixels can change.
Keep the full 1024x1024 camera on both chips and give each chip a tile window
(tiles tx0..tx1, ty0..ty1). Then:
- the projection arithmetic is identical
- each gaussian's bbox is clipped to the window
- each tile sees the same gaussians in the same storage-id order (compaction is monotonic)

The split line must be on a tile edge (multiple of 32 px). Then the md5 of the joined image must
equal the one-chip golden.

## 2. Load balance (bicycle, 30 views)

The model uses the exact per-tile pair counts from the host (`tiles616.py`, `bbox616.py`) and an
LPT schedule of tiles onto 110 cores per chip, which includes the heaviest tile. The heaviest tile
holds about 20k bbox pairs, against a mean of 2.6k. On half a frame it is about 1.3x a core's mean
load, so it is counted.

| partition | latency, fixed per tile | 0.3 ms fixed per view | fixed per view | imbalance (total time) |
|---|---|---|---|---|
| left/right, static at col 16 | 5.77 (max 6.12) | 5.98 | 6.88 | 1.04 |
| left/right, previous view's best line | 5.66 | 5.93 | 6.83 | 1.03 |
| left/right, best line per view (oracle) | 5.55 | 5.80 | 6.68 | 1.01 |
| top/bottom, static at row 16 | 6.11 (max 6.81) | 6.33 | 7.17 | 1.17 |
| top/bottom, previous view's best line | 5.74 | 5.97 | 6.90 | 1.05 |
| interleaved rows (1 tile), pfwc on all M | 5.92 | 6.19 | 7.13 | 1.05 |
| interleaved rows, pfwc culled to chip | 5.74 | 6.01 | 6.95 | 1.06 |
| cols1 / checker / rows2 / rows4 interleave | 5.74-5.96 | 6.01-6.22 | 6.95-7.17 | 1.02-1.08 |

All values are ms, device time only (no join/sync).

- Pair share of the left half on a static split ranges from 23% to 62% (mean 44%). Top/bottom
  ranges from 32% to 81%. Part of the work is duplicated or fixed, so the time imbalance is much
  smaller than that.
- Interleaving balances pairs well but gains nothing over left/right. It also needs more kernel
  changes: a mask in the pfwc pair count, in K2 emit and in the tile-id to origin mapping. Each
  gaussian also touches both chips more often.
- Moving the split line using the previous frame's per-chip device ms gains about 0.1 ms. The
  30 bench views are far apart, so this underrates it for the viewer, where consecutive frames are
  close. It is a phase-2 option.

**Which fixed cost halves?** The one-chip mat+blend fit is 1.421 + 1.443P ms. The model brackets
how much of the 1.421 halves with the tiles:
- all of it (optimistic)
- all but 0.3 ms per view (central)
- none of it (pessimistic)

The t595 profile has the fixed part spread over per-tile setup and claims, so the central case is
the expected one.

## 3. Modelled time

| | one chip | 2-chip split (central) | range | AFR, 2 chips |
|---|---|---|---|---|
| single-frame latency | 8.451 | ~6.2 (5.98 device + 0.2 join/sync) | 5.7-7.1 | 8.45 (no latency gain) |
| b2b ms/view | 8.451 | 5.84 (+ host) | 5.62-6.68 | 4.226 (measured) |

Assumptions:
- the stage fits from t595 hold for each half
- pfwc fixed cost per view stays fixed on each chip
- the 2-chip host overhead is as in AFR (+0.7%) plus 0.1-0.3 ms join/sync
- each chip runs the same claim scheduler on 110 cores

The model's one-chip time is 8.616 ms against 8.451 measured, so all values are scaled by 0.981.
Model M and P match the device counts to 1-3% (hero view M 1.8807 vs 1.8799).

Not modelled:
- dispatch differences between the two hosts' chips (bh-50 one-chip 8.395 vs bh-33 8.451)
- PCIe contention from both chips writing to the host at once (3 MB per frame, small)

## 4. Go/no-go

Bar: latency cut >= 15% and b2b no worse than one chip.

- Central: 6.2 ms, 27% cut. b2b 5.8 < 8.45. **Clears.**
- Pessimistic (no fixed mat+blend cost halves, plus 0.3 ms host): 7.1 ms, 16%. Still clears, but
  only just.

**GO.** Main risks:
1. how much of mat+blend's per-view fixed cost halves
2. pinning one host buffer on two devices (fallback: a 1.5 MB copy)
3. the host round trip that starts both chips in latency mode
4. on a half frame, the heaviest tiles bind more often. A follow-on is to split the biggest tile
   across two cores, but only if the measured run shows it binding.

## Build-and-measure spec (ready to queue)

Binding rules for this task:
- Own git worktree of ~/dev/gstt2 on a new ttp/* branch. Never edit ~/dev/gstt2. No PRs, no force
  push, no rebase or amend of pushed commits, no deleting or moving branches or tags.
- Device runs only on the 'p150-2board' reservation: one 2-board p150 IRD host other than bh-30,
  reached via the yyz-ird jump host. Never use bh-30 or the measurement reservation. Strict host-key
  checking on every ssh (the user's ssh config sets it off for bh-*: override it). No scanning, no
  exploratory logins. Only the 2-chip measurement task reserves or releases the box. If none is
  free, wait. Release it as soon as the measurement is done.
- One device run at a time. Wrap each sync+build+run in `ttp lock p150-2board -- ...`.

Code changes:
1. **Tile window.** Add `tile_x0, tile_x1, tile_y0, tile_y1` to `PfwcVisParams`. The default is the
   whole screen, which leaves the one-chip path unchanged.
   - pfwc fused writer (pfwc_fuse.h, around lines 179 and 398): clip each bbox to the window before
     counting and emitting pairs. A gaussian whose clipped bbox is empty is not visible on this chip.
   - Emit window-local tile ids (`(ty - ty0) * wx + (tx - tx0)`) so sort bins and the blend tile
     list cover only the window.
2. **Tile origin.** In the blend reader/writer (`reader_alpha_blend_mb_devcull.cpp`, the
   `tile_id % tiles_x` origin and the tile id passed through `CB_TILE_Q`), map the local id back to
   the global origin. Write at the full-image row stride, so both chips write disjoint tiles of one
   1024x1024 RGB8 image.
   - Pass the screen tile count and window to the claim order (blend_claim_order.h) and mat as
     runtime args.
   - Keep `max_radius` at the full-image value (512); do not derive it from the window.
3. **Host.** Reuse render/afr.py's coordinator: one worker process per chip.
   - In split mode the coordinator sends both workers the same view with their window.
   - The time of a pass is max(t1) - min(t0) over the two chips.
   - Output: try one host buffer pinned on both devices first. If `PinnedMemory::Create` refuses
     the second mapping, use one pinned buffer per chip and copy one half across, and report the
     copy time.
4. **Flags:**
   - `GSPLAT_TT_SPLIT=1`
   - `GSPLAT_TT_SPLIT_AXIS=x` (y optional)
   - `GSPLAT_TT_SPLIT_AT=16` (tile column)
   - optional `GSPLAT_TT_SPLIT_ADAPT=1`: move the line by one tile column per frame toward
     balancing the previous frame's per-chip device ms
   - `GSPLAT_TT_AFR_DEVICES=0,1`
   - The default is off, and with it off the one-chip path must be byte-identical.
5. Tests:
   - a host unit test that the window clip keeps exactly the window's pairs (pair count per tile
     equal to the one-chip count for tiles inside)
   - a test that split off leaves the one-chip md5 unchanged

Checks (all required):
- **md5.** The joined 30-view md5 equals the one-chip golden of the same build and grid on the same
  host. Run the one-chip arm first and use its md5. Known values at 227: 906e0435 on bh-50,
  84695855 on 12x10 ETH, 077caaba on 11x10 p100a.
- **Hero.** A hero.png from the device, plus a diff image and PSNR against
  benchmarks/reference_v2/hero.png (>= 42.4 dB; 227 gives 42.51).
- **Seam.** Look at the image, around column 512 and the adaptive line, for seams or tile
  artifacts. md5 and PSNR alone do not count.
- **Arms**, all on the same 2-board host:
  - A: one chip, b2b and latency
  - B: split, latency (no --dump-views)
  - C: split, b2b
  - D: AFR b2b, for reference
- **Per chip**, report device ms per stage and imbalance.
- **Results.** Report results as '2 chips (split)' next to the one-chip headline and '2 chips
  (AFR)'. Add a row to the status report.

Keep it if: latency cut >= 15% measured against arm A, md5 equal, PSNR >= 42.4 dB, no seams.

## Files

- `tiles616.py`: host model per view (projection and cull at the 1/255 floor, as in t270). Writes
  the exact per-tile pair histograms and per-partition M to `tiles616.npz` and `tiles616.json`.
- `bbox616.py`: bbox pairs per tile and bbox row/column histograms, written to `bbox616.npz`.
- `analyze616.py`: stage fits, LPT on 110 cores, all partitions. Writes `model616.txt`.

To rerun: `python3 tiles616.py; python3 bbox616.py; python3 analyze616.py` in this folder. The first
two take about 5 min each and read scenes/bicycle.ply.

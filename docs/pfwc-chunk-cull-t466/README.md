# t466: pfwc-only chunk cull that keeps today's gaussian order (CPU only, no device)

**Question.** #464 ranked a pfwc-only chunk frustum cull as lever 1 for the back-to-back (b2b) period:
skip whole off-screen chunks in pfwc through a per-view index list, with no reorder of the scene, so
sort, mat and blend see exactly today's records (md5 39d84b28 unchanged). Does such a list skip anything?

**Answer: no.** In today's (PLY / training) order, **no pfwc tile can be skipped on any of the 30
bicycle views**, by the conservative test or even by an exact one. The 1024 gaussians of a pfwc tile
are scattered over the whole scene, so every tile holds at least one visible gaussian. A cull that
skips work needs spatially compact chunks, which means processing gaussians in another order. pfwc's
writer compacts the visible gaussians in processing order, and that order is the gid that sort
(tie-break) and blend (record address) use. So any skip either changes the downstream order (the
#433 path, md5 changes, +0.72 ms of blend in latency mode) or needs a writer that puts the records
back into today's order. The env-flag code the spec asked for (`GSPLAT_TT_PFWC_CHUNK_CULL=1`, an index
list over today's tiles) was **not written**: it would skip 0 tiles on every view.

## Model

`cullshare.py` (repo root: `python3 docs/pfwc-chunk-cull-t466/cullshare.py <ply>`; output
`cullshare-out.txt`). Bicycle, `scenes/bicycle.ply` md5 3745b7a6, N = 6,131,954, 5989 tiles of 1024,
30 views of `benchmarks/cameras_v2.json`, 1024x1024. Chunk test = `render/host/chunk_cull.h`
(box of mean ± 1.5·rho per tile, rho = 3·sqrt(trace cov3d) ≥ 3 sigma_max, 5 px pad, k_near 0.2).
Visibility = the #140/#169 per-gaussian model (z > 0.01, 3-sigma rect meets the image,
opacity > 1/255, det > 0).

Three tile orders:
- **ply**: today's order (what an index list without reorder would cull).
- **morton**: the #169/#433 global Morton reorder (changes the gids, so the downstream order).
- **core**: each pfwc core keeps its own gaussians (tiles c, c+120, ... as today, 12x10 grid), Morton-sorted
  inside the core. The record set per core segment is unchanged, but the writer would still have
  to restore the order inside the segment.

Columns: `cull` = share of tiles the conservative test removes; `empty` = share of tiles with no
visible gaussian (the upper bound for any per-tile test); `lost` = visible gaussians inside culled
tiles (must be 0).

| order | tiles culled, min / median / max | empty-tile bound, median | visible lost (30 views) |
|---|---|---|---|
| **ply (today, no reorder)** | **0.000 / 0.000 / 0.000** | **0.000** | 0 |
| morton (global, #433) | 0.350 / 0.402 / 0.424 | 0.644 | 0 |
| core (per-core Morton) | 0.023 / 0.030 / 0.124 | 0.450 | 0 |

Gaussian shares equal the tile shares (whole tiles). Visible share per view: 0.208 / 0.287 / 0.342.
An earlier quick pass with ply-order chunks of 64 and 256 gaussians also culled 0.000 on all 30 views.

`subbox.py` checks whether a tighter test (a tile culled only when all its sub-boxes of S gaussians
pass) closes the gap to the empty bound (every 5th view):

| view | morton S1024 | S256 | S64 | core S1024 | S256 | S64 |
|---|---|---|---|---|---|---|
| hero | 0.382 | 0.391 | 0.396 | 0.028 | 0.061 | 0.118 |
| orb_04 | 0.371 | 0.383 | 0.390 | 0.023 | 0.032 | 0.069 |
| orb_09 | 0.354 | 0.364 | 0.368 | 0.024 | 0.036 | 0.081 |
| orb_14 | 0.398 | 0.406 | 0.412 | 0.094 | 0.121 | 0.136 |
| orb_19 | 0.408 | 0.418 | 0.423 | 0.111 | 0.129 | 0.141 |
| orb_24 | 0.417 | 0.426 | 0.431 | 0.046 | 0.086 | 0.135 |

(`subbox-out.txt`; S1024 = today's `chunk_cull.h` test.) The `build_ms` column in `cullshare-out.txt`
is the numpy model's time, not the C++ list cost below.

The gap between `cull` and `empty` is mostly not box granularity: the K·rho pad, the
conservative side planes and the opacity floor (low-opacity gaussians are invisible but still in the
box) keep it open.

## Host cost of the list

`chunk_cull::survivors` over 5989 tile boxes: **0.24 ms/view** single thread on the Mac (M-series,
`-O2`, all tiles kept, worst case); #433 measured `pfwc_setup` 0.117 ms/view on the bench host. The
per-scene table build is ~0.3 s once. With xview the list build sits on the host before the pfwc
enqueue; #464 measured ≥1 ms of host slack per b2b frame, so it overlaps device work.

## What this means for lever 1

- An order-preserving cull of today's tiles saves nothing. Lever 1 as written in #464 is closed.
- Per-core Morton (keeps each core's segment set) culls only 2-12% with the conservative test, and
  needs a new writer (records to a scratch in processing order, an L1 visibility bitmap of the
  core's ~51k gaussians, then a rank-ordered copy into today's segment). Not worth it at ≤12%.
- The only path with a large skip share is the global Morton reorder (#433, ~40% of tiles). It
  changes the gid order. #433 judged it in latency + `--dump-views`, which hides pfwc (#464), so
  **its pfwc saving in b2b was never measured**. The next step is to measure the existing #433 code in
  b2b mode. It needs no new code (all env flags are in HEAD, 3c346c2b's base c7232e14).

## Device A/B for the next task (existing #433 code, b2b)

Same bench as #464 (`docs/b2b-gap/probe464.sh` `run` helper, p150, 30 views, `--no-ref`), 3 alternated
rounds, `ttp lock p100 -- ...` around each sync+build+run:

```
# off   (default)
.venv/bin/python3 render/run.py --no-ref --back-to-back --iter-dir t-off
# ro    (Morton reorder, no tile skipped: the reorder's own cost)
GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0 .venv/bin/python3 render/run.py --no-ref --back-to-back --iter-dir t-ro
# cull  (reorder + skip ~40% of tiles)
GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_LOG=1 .venv/bin/python3 render/run.py --no-ref --back-to-back --iter-dir t-cull
```

Read `TTW_TIMING b2b_ms_frame` and the `B2B_STAGES` line (project/gather_wait, sort, blend). pfwc
saving in b2b = ro − cull; reorder cost = ro − off. Expected md5: off d9a60a3c (raw) / 39d84b28 (golden);
ro and cull differ from golden (tie order) but must be identical across rounds. Gate: if cull − ro
saves < 0.3 ms/frame b2b, close the chunk-cull lever for good. If it saves more, the follow-up is to
remove the reorder's blend cost (original-gid tie-break in sort, then find out why blend slows with
Morton gids), not an order-preserving cull. Device runs need a hero.png, diff and visual check per
the charter if anything is kept.

## Tests

`tests/unit/test_pfwc_chunk_cull_order.py` (synthetic scene, no device): random order culls nothing,
Morton culls >30% with 0 visible lost and only empty tiles culled, per-core Morton keeps every core's
gaussian set. `tests/unit/test_chunk_cull.cpp` (existing, conservative vs per-gaussian rects) passes.

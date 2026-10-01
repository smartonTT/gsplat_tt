# Task #90: sort_subchunk_mat + SFPU cull, fused (continues #86)

Board **yyzo-bh-07 (Blackhole p100a)**, bicycle 30 views 1024x1024, untraced
`render/run.py`, every run md5-identical on all 30 views to `md5-r82new.txt`,
hero_vs_ref 100 dB. Base = a5f2bd6 code (remote tree gstt2-t85 @4d71d96, render/
identical). Logs: `t90job*.log` here. #86's own findings: `docs/matcull-t86/README.md`.

## What changed

1. **2 KB record pages** (#86's d9373a6, rebased as d8117a5; first measured here): buf_l1_recs and the
   in-cap overflow region use 2 KB DRAM pages, so the materialize bucket read is
   one NoC read per 64 records instead of per 2.
2. **Host timer split** (absorbs #87): `GSPLAT_TT_SPLIT_BLEND=1` adds a Finish
   after sort_subchunk_mat and after the cull, so STAGES shows `mat`, `cull` and
   `blend` as separate program windows. Default off: one Finish drains all three
   and `blend` holds mat + cull + blend (the booking that misled #83). New
   top-level `mat` bucket; tests/unit/test_stage_timer_keys.py checks the keys.
3. **Fused mat + cull** (`GSPLAT_TT_FUSE_MATCULL`, default on; `=0` restores the
   separate tile_l1_cull program): the SFPU band cull runs inside the mat program.
   Each mover, after depth-sorting a slab in L1, transposes every 128 records into
   an fp32 coefficient tile on its own CB pair (NCRISC 8/9, BRISC 24/25) and
   patches the returned mask into word3 before the slab's single DRAM write.
   One compute kernel (`mat_cull_compute.cpp`) serves both movers in arrival
   order: UNPACK polls both coefficient CBs, MATH/PACK get the stream id through
   the TRISC mailboxes; a tile with word 1023 != 0 ends a stream. The gather
   items now also build their part in the L1 slab and write 2 KB pages (were 32 B
   writes per record). The cull program's slab DRAM read + write-back and its
   launch are gone.
4. `opt/sync_remote.sh` no longer streams the LFS hero fixtures (git-lfs smudged
   ~330 MB) or committed profiler captures/screenshots: archive 1.1 GB -> 52 MB,
   sync + rebuild ~1.5 min. This is what made #86's and the first #90 sync time out.

## Results (ms/view, avg_frame_ms; interleaved rounds)

| config | rounds | mean | blend bucket (mat+cull+blend) |
|---|---|---|---|
| base a5f2bd6 | 43.27, 43.20, 43.20, 43.02 | **43.17** | 12.31-12.42 |
| + 2 KB rec pages (FUSE=0) | 42.18, 41.96, 41.98, 41.99 | **42.03** (-1.14) | 11.24-11.27 |
| + fused mat+cull (default: FOLD=0, depth 2) | 41.31, 41.39, 41.34, 41.57 | **41.40** (-1.77 vs base) | 10.64-10.68 |
| fused, FOLD=1 (fill inside the permute) | 41.55, 41.49, 41.63, 41.33, 41.48, 41.45 | 41.48 | 10.67-10.76 |
| fused, depth 4 | 41.41 | | 10.74 |

Final: **43.17 -> 41.40 ms/view (-1.77, -4.1%), 23.2 -> 24.2 FPS**; the blend
bucket (mat + cull + blend device time) 12.4 -> 10.7 ms. In the one job that
interleaved base, 2 KB pages and the final kernel (job1, round 1): 43.27 / 42.18
/ 41.31. Published GPU reference: 10.75 ms/view (published, not measured).

Split timers (`GSPLAT_TT_SPLIT_BLEND=1`, each adds ~0.1 ms of drain):

| | mat | cull | blend |
|---|---|---|---|
| FUSE=0 (2 KB pages) | 2.51 | 1.34 | 7.58 |
| FUSE=1 | 3.24-3.27 (mat+cull) | - | 7.53-7.58 |

The 2 KB pages took mat from 3.54 ms (#86 Tracy) to ~2.4 ms. Fusing saves
~0.6 ms of the 3.85 ms mat + cull. The fused program is ~0.75 ms longer than
mat alone: the movers are throttled by the SFPU pace during their permute (a
128-record batch takes the SFPU longer than a mover needs to permute and fill
it). Folding the coefficient fill into the permute (`GSPLAT_TT_MATCULL_FOLD=1`)
and 4 instead of 2 batches in flight (`GSPLAT_TT_MATCULL_DEPTH=4`) did not
change the window (3.26-3.27 ms), which fits a throughput-bound SFPU.

## Next lever

- **sort_bucket_emit (~8 ms/view, the largest program)**: SORT_STAGES bin_emit
  8.8 ms. It writes every 32 B record to its tile bucket with its own NoC write
  (`flush_recs` in sort_bin.cpp, ~31K scattered 32 B DRAM writes per core).
  Staging records per destination tile in L1 (local counting sort by tile, then
  one write per tile run, ~1 KB) would cut the transaction count ~30x.
- **Fused cull, remaining ~0.75 ms**: the movers wait for the SFPU during the
  permute. Either start the cull on the unsorted bucket while the radix sort
  runs (the mask depends only on the record; key gather first, then patch word3
  in the bucket, non-blocking service points inside the radix passes), or cut
  the band-cull SFPU instruction count (B2 reloads nw/nz/dr/dl from DEST for each
  of the 4 columns of a band).

Commits (after the rebase onto d17df77): 3e731c6 timer split, 83858a9 fused
mat+cull + sync fix, 488d19e/fed6c28 fold option, the iter-175 commit (ledger + report). The
remote logs name the pre-rebase SHAs b3bf125, dc920de, f33ce05, eac0ded (same
render/ code).

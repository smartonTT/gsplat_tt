# t284: full-quality overflow fallback (grown tile bucket + big-tile rank select)

## Problem
#270 re-renders a view whose largest tile holds more than 32768 records (the
one-launch sort bucket, `kTileCap`) at a coarser contrib floor. The far hero
pose (dolly back 2) then drops to 29.35 dB vs its CPU reference at 1/255.

## Where the 32768 limit really is
- The blend already walks a fat tile as a depth-ordered sequence of 8192-record
  subchunks and carries T across them; it has no per-tile record limit.
- The mat worklist already gives a big tile one item per subchunk.
- The limits are (a) the DRAM bucket slot, `tile_cap` records per tile (a
  runtime arg of emit and mat), and (b) the mat big-tile path, which loads all
  N keys into CB_BSORT (33024 u32) and radix-sorts N (key, index) pairs in
  CB_BUCKET (3N u32, and the u16 radix histogram caps N at 32768).

## Design
1. Host (`sort_device.cpp`): the bucket capacity is a process state,
   `kTileCap` (32768) by default. When a frame overflows and the largest tile
   fits `kTileCapBig` (65472, a multiple of the 64-record page, under the u16
   bin limit), the capacity grows to `kTileCapBig` and stays there (the DRAM
   bucket buffer grows once, 1024 tiles x 65472 x 32 B = 2.1 GB). `render_view`
   then re-renders the view at the SAME floor. Only when the tile is above
   `kTileCapBig` (or `GSPLAT_TT_TEST_TILE_CAP` forces it) does #270's coarser
   floor retry run, as the last resort.
2. Mat kernel (`sort_subchunk_materialize.cpp`): a tile over 32768 records takes
   a new branch. Its keys are streamed into CB_BUCKET (8192-record chunks staged
   in CB_SLAB), then `sort_ol::select_ranks_big` finds only the item's stable
   depth ranks [lo, hi): the #124 bin histograms narrow the key range of ranks
   lo and hi-1 (one more level than #124), the candidates (<= 32768) are
   collected in index order and radix-sorted. The result is the same ids as a
   stable sort of the whole tile, so the image is the full-quality render.
   The existing gather / cull / slab emit then runs unchanged.
3. Tiles of <= 32768 records, i.e. every bench view, run exactly the old code;
   the default path changes only by the new branch and the cap now being a
   variable.

Rejected: depth-sliced re-render of the view (needs a far-plane cull and a
transmittance output, i.e. pfwc and blend changes, and 3 passes); sub-tile
split (the blend works on 32x32 tiles).

## Tests
- `tests/unit/test_sort_onelaunch_v2.cpp`: `select_ranks_big` against
  `std::stable_sort` for N up to 65472 (random, clustered and tied keys).
- `tests/unit/test_overflow_retry.cpp`: `grown_tile_cap` decision.

## Results (drive 2, commit 4bd3d2c3 vs base 6a7dae5, yyzo-bh-07 p100a, untraced)
- Default path: sweep md5 906e0435 (the post-#290 golden) on 30 views in all
  8 arms, base and new. Paired ABBA + BAAB, mean of 30 views per arm:
  base 11.663, new 11.664 ms/view (+0.001, gate +0.05).
- Far pose (hero dollied back 2, tile 554 = 35888 records): bucket grown to
  65472, re-rendered at the same 1/255 floor. PSNR vs
  `docs/tile-overflow-t270/img/far2_cpu_ref255.png` 43.57 dB (#270's floor
  fallback: 29.35 dB). Frame 15.6 ms (first view pays the grow + retry; the
  bucket then stays grown). Tile 554's own PSNR is 35.6 dB, the same as its
  spoke-region neighbours (28-35 dB); no tile seams in the image or the x10
  diff (`img/far2_255_grown.png`, `img/far2_255_diff10.png`).
- Far pose at 1/16384: 39376 records, grown, 18.9 ms.
- Pulled back 4: 50704 records, grown, 17.8 ms; image looks clean
  (`img/far4_255_grown.png`).
- Last resort (`GSPLAT_TT_TEST_TILE_CAP=20000`): no grow; the floor retry runs
  1/255 -> 1/16 -> 1/5 -> 1/2, the tile still holds 29488 > 20000, so the view
  raises "device sort failed" (as #270 does when every floor overflows).
- Device hero (defaults): md5 906e0435, 42.51 dB vs
  benchmarks/reference_v2/hero.png, golden match, no seams.

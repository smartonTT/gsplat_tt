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

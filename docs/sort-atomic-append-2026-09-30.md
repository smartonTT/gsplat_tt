# Atomic fixed-capacity bucket append (task #24, R8 / candidate 5.3)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views,
1024x1024, `python3 render/run.py --no-ref`.

## 1. What the sort stage did before

Per view, in order, with the host waiting on the device twice:

1. `sort_bin` count pass: per-(core, tile) histogram of the kept pairs.
2. Host bridge: 450 KB histogram D2H, prefix sums / page layout / LPT, then
   ~1.4 MB of per-(core, tile) base tables H2D.
3. `sort_bin` scatter pass: each pair's 32 B record into its tile's bucket at
   the prefix-sum slot (tiles of 8193..16384 records into a separate overflow
   region), plus a page-aligned (depth key, gaussian id) layout of every pair.
4. `sort_tile_depth` radix over every tile's (key, id) list, then
   `sort_publish` compacting it into `sort_sorted_ids`.
5. `sort_subchunk_materialize`: for tiles of up to 16384 records, read the
   bucket, stable-radix it by depth in L1, write the sorted slab; for larger
   tiles, gather each record from `blendrec` in `sort_sorted_ids` order.

The keys/ids layout, the radix and the publish (steps 3-4) exist only for the
last case. `sort_sorted_ids` has no other reader: the cull and blend readers
take its address but read only `sort_tile_ranges` and the slab directory.

## 2. What it does now

One launch, `sort_bin_atomic.cpp`, on both data movers of every core:

1. **count** — each mover counts its page range per tile (BRISC `[lo, mid)`,
   NCRISC `[mid, hi)`), keeping its gid/tid/keep pages in L1.
2. **reserve** — NCRISC does one NoC atomic fetch-and-add per (core,
   non-empty tile) on the tile's counter (a 4 KB L1 buffer the host zeroes).
   The returned old value is the core's first slot in the tile. It writes
   `(count << 16) | first slot` per tile as its row of a chunk table.
3. **emit** — both movers pack their records into `tile * 32768 + slot`
   (fixed capacity 32768 = the old `MAX_TILE_ENTRIES` limit), prefetching the
   next page's `blendrec`/depth pages while packing the current one.

The host reads the 4 KB of per-tile totals (that read is the only wait), builds
the LPT, ranges and slab directory from them, and launches the materialize. No
count pass, no histogram, no layout upload, no keys/ids, no radix, no publish.

**Determinism.** Cores reach a counter in any order, so a bucket holds the
cores' chunks in arrival order, and a stable sort of that by depth key would
order equal keys differently from run to run. The materialize first rebuilds
the canonical order — core 0's chunk, core 1's, ... — from the chunk table
(110 reads per tile), then runs the same stable radix. Inside a core the order
is BRISC's pages then NCRISC's, as before. The result is the prefix-sum
layout's order exactly. `tests/unit/test_sort_atomic_bucket.cpp` checks this
on random scenes with many equal keys, and shows that sorting arrival order
directly would differ on most tiles.

**Tiles over 16384 records** (5-18 per view) no longer gather from `blendrec`:
their full record set is in the bucket. The materialize extracts their keys in
16384-record chunks, sorts indices, and gathers each subchunk's records from
the bucket. It fits the existing L1 buffers (key array and scratch in the slab
buffer, indices in the radix buffer).

**Overflow.** A tile over 32768 records drops the excess in the emit and the
host fails the frame with a message — the same limit, and the same response,
as the old path's `MAX_TILE_ENTRIES` check.

`GSPLAT_TT_SORT_ATOMIC=0` keeps the old path in the same build (A/B).

## 3. Results (yyzo-bh-07, Blackhole p100a) — REJECTED as built

Ported to the tip (a58c917, without the unlanded task #35 split): the
materialize block now uses the adaptive radix (`sort_radix_tile::sort_pairs`)
on keys/slots gathered in canonical order, and tiles over 16384 records fill
each output subchunk from coalesced 16384-record chunk reads instead of a
random gather. Commits e1f86c5..d50703a on ttp/t24 (not landed).

**Speed (the frames that completed).** Sort stage on the atomic path is one
launch plus a 4 KB read: 6.6-6.8 ms per view (emit 6.2-6.4, layout 0.13,
publish host 0.2) against 15.6 ms for the legacy tip (count + host layout +
scatter + radix + publish). Up to ~8 ms/view of sort time is on the table.

**Correctness: not usable.** The host check `GSPLAT_TT_SORT_ATOMIC_CHECK=1`
reads the chunk table back. In 1-12 tiles per frame (of 1024) two cores got
overlapping chunks, or a gap was left, while the chunk lengths still summed to
the tile's counter total. So the counters are right but some NoC
fetch-and-adds (`noc_fast_atomic_increment<..., program_ret_addr=true>`,
`INCR_GET`) returned a wrong old value. Tried, none fixed it: one return slot
per 16 B, one counter per 16 B, per-core staggered tile order, one atomic in
flight at a time, L1 cache invalidate after the barrier. The frame after a bad
layout hangs the device (every 30-view atomic run timed out, on the old base
and on the tip). tt-metal has no in-tree user of the atomic return path.

Old base (548c58c + #35) legacy for reference: 94.8 / 95.1 / 94.9 ms/view.

**Next:** keep the design but reserve without atomic return values: a
device-side prefix sum over cores (each core writes its per-tile counts, global
semaphore barrier, each core sums a slice of tiles over cores, second barrier,
each core reads its bases). The buckets are then in canonical order, so the
materialize needs no chunk table either.

Also found on the way: `opt/sync_remote.sh` runs the remote cmake without
`TT_METAL_HOME` (configure fails in a non-interactive ssh) and prints the
error with `tail -30 f1 f2`, which GNU tail rejects, so a stale host .so
was used silently (exit code lost when piped). The watcher trips on
`gather_visible_scatter.cpp` (BRISC reads a runtime arg index out of bounds).

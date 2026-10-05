# Task #106 — lever 1: one-launch device sort (`GSPLAT_TT_SORT_ONELAUNCH`)

Code only, no device run yet. Default off; with the knob unset the build, the
programs, the runtime args and the frame are the same as the base (c5fed7d,
which includes #100's emit rewrite 2e72691).

## What it replaces

At the tip (2e72691, #100: 29.59 ms/view, 33.8 FPS) the sort stage is
10.68 ms: count launch → host reads the per-(core, tile) histogram and lays
out pages (~1.3 ms, #85) → upload → emit launch (5.29 ms) → radix launch →
publish. Everything but the emit is 5.39 ms. (#93 ranked this lever at
72cd487: 32.6 ms/view, sort 14.2 ms, emit 8.8 ms.)

With the knob on, `sort_bin_onelaunch.cpp` does it in **one launch** (both
data movers of all 110 cores):

1. **Count.** Each mover counts its kept pairs per tile (BRISC pages
   [lo, mid), NCRISC [mid, hi), same split as the legacy emit).
2. BRISC sums both movers' counts into the core's row and writes it to DRAM.
   **Barrier 1**: every core increments a semaphore on core 0; core 0 waits for
   109, then increments a release semaphore on each core (unicast, NoC coords
   from the host). No atomic return values (the #24 failure).
3. **Device prefix sum.** Core c owns row pages c, c+110, ...: it reads that
   page of every core's row, writes each core's exclusive prefix (its first
   cursor per tile), the tile totals, and the padded totals (sum of
   ceil16 per core, equal to the legacy `tile_pad`). **Barrier 2.**
4. BRISC reads its base row; NCRISC's cursors start at base + BRISC's count.
5. **Emit.** Each kept pair's 32 B record (same packing as `sort_bin.cpp`)
   goes to slot `tile * 32768 + cursor` of a fixed-capacity bucket
   (1 MiB per tile, 1 GiB for 1024 tiles, 2 KB pages). Excess records are
   dropped and the host fails the frame, like the legacy MAX_TILE_ENTRIES check.
   With `GSPLAT_TT_EMIT_PUBOC=1` (#100's default) the record's op/color/depth
   words are copied from the gather-published blendrec[10..12], as in #100's
   emit; no depth reads. #100's PB batching and RING write coalescing are not
   ported (follow-up).

The host then reads only the 8 KB totals rows, builds `bucket_meta`, the LPT
(from the padded totals, so the blend schedule is unchanged), the tile ranges
and the subchunk directory. **No host bin layout, no layout upload, no radix
launch, no publish launch.**

**Why the output is byte-identical.** Every bucket holds its tile's kept
pairs in ascending pair order (core order, then BRISC's pages, then
NCRISC's), which is the order the legacy prefix layout feeds its stable radix.
The materialize sorts each bucket with the same stable radix on the same key
word, so the sorted order matches. `tests/unit/test_sort_onelaunch.cpp` checks
this against a model of the legacy layout.

**Materialize** (`sort_subchunk_materialize.cpp`, compiled with
`SORT_ONELAUNCH` only when the knob is on; args 16/17 are added only then):

- Tiles up to the mover's whole-tile capacity (NCRISC 16384, BRISC 6144) use
  one item: read the bucket, sort in L1, then permute (with the fused cull)
  and write every subchunk. This is the legacy in-cap overflow path.
- Bigger tiles get one item per subchunk, on NCRISC only. Each item reads the
  keys in 16384-record chunks, does an index radix of the whole tile, then
  fills its own subchunk from coalesced chunk re-reads and culls it.
- `build_mat_worklist(..., onelaunch=true)` makes these items. Big items carry
  `big=true`, so they never go to BRISC.

## Files

- New: `render/kernels/dataflow/sort_bin_onelaunch.cpp`,
  `render/host/sort_onelaunch_layout.h` (host model + `check_prefix`),
  `tests/unit/test_sort_onelaunch.cpp`.
- Changed:
  - `render/host/sort_device.cpp`: knob, program build, the frame block in
    `sort_resident_pairs`, materialize args.
  - `render/host/sort_mover_split.h`: the `onelaunch` worklist option.
  - `render/kernels/dataflow/sort_subchunk_materialize.cpp`: the
    `#if SORT_ONELAUNCH` block.
  - `tests/syntax_stub/`: semaphore stubs; `check.sh` covers the new files.
- `sort_bin.cpp` is untouched (#100 and #105 are editing it).

## Expected savings

#93's upper bound was ~7 ms/view at 72cd487. #100 has since cut the legacy
emit from 8.8 to 5.29 ms, so less is left at the tip:

- #24 measured the same one-launch shape (count, bases, emit into 1 MiB
  buckets; atomics for the bases, no PUBOC) at 6.2-6.4 ms per launch plus
  0.33 ms host. Ours replaces the atomics with two barriers and a
  one-page-per-core prefix (tens of µs) and copies the published words
  (PUBOC). Expected sort stage: **~5-6 ms vs 10.68 ms**.
- Moved into the materialize: sorting in-budget tiles (the radix launch's
  job), and big tiles (over 16384 records) sorted once per subchunk item.
  Estimate +0.5-1.5 ms.
- **Net: -3 to -5 ms/view** (29.6 → ~25-26.5 ms/view, ~38-40 FPS), to be
  measured. Porting RING into this emit is worth up to ~1.2 ms more (#98's
  write-coalescing bound).

## Local checks (Mac, no device)

- `CXXFLAGS=-Irender/host bash tests/unit/run_cpp.sh
  tests/unit/test_sort_onelaunch.cpp`: PASS on 4 random frames (1.2 M
  pairs, odd tile counts, more cores than row pages, forced drops), plus
  check_prefix corruption detection and 20×2 worklist rounds. Two mutations
  fail it: NCRISC's cursor without BRISC's count, and the legacy worklist.
- Existing: `test_sort_tail_dual_mover` 60/60, `test_sort_bin_dual_mover` 0
  mismatches, `test_sort_radix_tile` 0 fails, `test_gather_dual_mover` 0
  mismatches.
- `tests/syntax_stub/check.sh`: 21/21 ok, including `sort_device.cpp`,
  `sort_bin_onelaunch.cpp` (with and without `EMIT_PUBOC`), and the
  materialize with and without `SORT_ONELAUNCH`.
- `python3 -m pytest tests/spec -q`: 36 passed, 2 skipped, 2 xfailed.

## Device A/B plan (follow-up task)

All steps on yyzo-bh-07 p100a, inside `ttp lock p100 -- ...`, with the
`opt/sync_remote.sh` flow.

1. **First run, checked.** Run `GSPLAT_TT_SORT_ONELAUNCH=1
   GSPLAT_TT_SORT_ONELAUNCH_CHECK=1 TT_METAL_WATCHER=2` for 5 views. Expect
   `[SORT] ONELAUNCH_CHECK bad_tiles=0` on every frame and no watcher asserts.
   Then 30 views without the watcher: md5-identical to the knob-off dump and
   to `md5-r82new.txt`, hero_vs_ref 100 dB.
2. **Hang check.** 3 consecutive 30-view runs with the knob on.
3. **A/B.** 3 interleaved rounds of knob 0/1, 30 bicycle views each. Report
   ms/view, FPS, and the per-stage sort and blend times. Watch the
   `[SORT] stage=ONELAUNCH ... onelaunch=` time (expect ~5-6 ms) and the blend
   stage (the materialize now sorts in-budget tiles).
4. **Attribution.** One Tracy capture (docs/gaps-t85/remote_tracy.sh):
   - The count, emit, radix and publish programs should be gone.
   - `sort_ol_barrier` zones give the barrier cost; `sort_ol_count`,
     `sort_ol_emit` and `mat_ol_*` give the split.
   - `GSPLAT_TT_MAT_STATS=1` gives the big-item makespan (`max_ncrisc`). If
     big tiles set the pace, the fix is a per-tile sorted index cached in
     DRAM by the first subchunk item.
5. **Adopt** if the gain is at least 3 ms/view and all 90 views are
   identical: default on, `=0` as kill switch, ledger row, report
   regenerated, review task.

Likely first-run problems:

- A hang at a barrier: wrong NoC coords (args 27+), or a semaphore not
  re-armed. Check with the watcher's waypoint dump.
- `bad_tiles>0`: a DRAM write not visible before the barrier
  (`noc_async_write_barrier` placement).
- Wrong pixels with check OK: the materialize branch (bucket page math
  `tile * 512`, or a big-path chunk offset).

## Follow-ups after the A/B

- Port #100's PB batching and RING write coalescing into the one-launch
  emit (up to ~1.2 ms, #98). RING needs per-tile L1 staging, about 256 KB per
  mover, so it is tight next to the count pass's page cache.
- If big tiles set the materialize pace, cache a per-tile sorted index in
  DRAM from the first subchunk item instead of re-sorting per item.

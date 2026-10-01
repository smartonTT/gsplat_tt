# Task #106 — lever 1: one-launch device sort (`GSPLAT_TT_SORT_ONELAUNCH`)

Code only, no device run yet. Default off; with the knob unset the build, the
programs, the runtime args and the frame are the same as 72cd487.

## What it replaces

At 72cd487 (32.6 ms/view, sort stage 14.2 ms) the sort stage is: count launch
→ host reads the per-(core, tile) histogram and lays out pages (~1.3 ms, #85)
→ upload → emit launch (8.8 ms, DRAM bound, #98) → radix launch → publish.

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

The upper bound is ~7 ms/view (#93): 32.6 → ~25.6 ms/view, 30.7 → ~39 FPS.

#24 measured the same one-launch shape (with atomics) at **6.6-6.8 ms/view**
for the sort stage, against 15.6 ms legacy. Our version adds two 110-core
barriers and a prefix pass of one 64 B page per core (tens of µs).

The materialize now also sorts in-budget tiles, which the radix launch used
to do. So part of the radix cost moves into the blend chain, and big tiles
(over 16384 records) are sorted once per subchunk. Realistic net:
**-4 to -7 ms/view**, to be measured.

## Local checks (Mac, no device)

- `tests/unit/test_sort_onelaunch.cpp`: PASS on 4 random frames (1.2 M
  pairs, odd tile counts, more cores than row pages, forced drops), plus
  check_prefix corruption detection and 20×2 worklist rounds. Two mutations
  fail it: NCRISC's cursor without BRISC's count, and the legacy worklist.
- Existing: `test_sort_tail_dual_mover` 60/60, `test_sort_bin_dual_mover` 0
  mismatches, `test_sort_radix_tile` 0 fails, `test_gather_dual_mover` 0
  mismatches.
- `tests/syntax_stub/check.sh`: all ok, including `sort_device.cpp`,
  `sort_bin_onelaunch.cpp`, and the materialize with and without
  `SORT_ONELAUNCH`.
- `pytest tests/spec`: see the hand-off.

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
   `[SORT] stage=ONELAUNCH ... onelaunch=` time (expect ~6-7 ms) and the blend
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

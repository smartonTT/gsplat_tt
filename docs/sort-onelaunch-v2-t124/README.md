# t124: one-launch sort v2 (lever A), code prep, no device

Task #124. Lever A from docs/reprofile-t115/README.md. Base 9bab3d4 (tip a03bd1e +
t115 docs). **No device run yet: every speedup below is unmeasured.** The work is
behind `GSPLAT_TT_SORT_ONELAUNCH=1`, which stays default off until the device A/B.

## Why
#114 measured the v1 one-launch sort at -1.07 ms/view (29.55 -> 28.48), not adopted.
Two causes (docs/sort-onelaunch-t114/results.md):
1. Its emit took 8.26 ms against 5.29 ms for the legacy emit, which has #100's PB
   batching and RING write coalescing. v1 did not have them.
2. Materialize: one big-tile item of ~33k cost against a mean slot of ~13k set
   the makespan.

## What changed (all only with GSPLAT_TT_SORT_ONELAUNCH=1)
- **Emit (sort_bin_onelaunch.cpp), `OL_PB` / `OL_RING`.**
  - Pair pages are read in batches of PB. Batches inside the count-phase window
    use the cached planes in place. Batches outside it use 3 staging buffers,
    issued two batches ahead.
  - Blendrec pages are read through 2 rings, so reads overlap processing.
  - With RING=R, records go into per-tile runs of R slots in L1 (CB 13). A full
    run is one R*32 B NoC write inside one record page. A mover's first and last
    run of a tile are clipped to its own cursors, so movers never overlap. The
    final partial runs are written at the end.
  - Rings are on only if num_tiles <= 1024 and tile_cap % 64 == 0. Otherwise the
    v1 per-record writes run.
  - PUBOC (copying the gather-published op/color/depth) is unchanged from v1.
- **Materialize (sort_subchunk_materialize.cpp), `OL_MAT_SELECT`.**
  - A big tile's subchunk is split into parts of 4096 records (OL_MAT_PART). Each
    part is its own worklist item.
  - An item reads all keys of the subchunk. Up to 3 rounds of 1024-bin key
    histograms narrow the key ranges of its first and last rank. It radix-sorts
    only the keys between those ranges and gathers its 4096 records.
  - The output is the same ids as a full sort (stable ranks). The unit test checks
    this against std::stable_sort.
  - Host cost model per part: `cnt/4 + 2*recs`. It is not calibrated; calibrate it
    with the mat_ol_* zones (step 5).
- **Knobs (render/host/env_config.h).**
  - `GSPLAT_TT_OL_PB`: 1/2/4/8/16, default 8.
  - `GSPLAT_TT_OL_RING`: 0/2/4/8, default 8.
  - `GSPLAT_TT_OL_MAT_SELECT`: default 0 since task #121 (select was 0.15 ms/view slower on bicycle; see docs/lever-a-t121/results.md).
  - `GSPLAT_TT_OL_WIN_PAGES`: 32..2048 in steps of 32. Default 1024 with the ring,
    else 1536 (as in v1).
  - **Kill switch back to v1:** `GSPLAT_TT_OL_PB=1 GSPLAT_TT_OL_RING=0 GSPLAT_TT_OL_MAT_SELECT=0`.
- **Host.**
  - The sort program prints one stderr line once:
    `[SORT] ONELAUNCH v2 OL_PB=.. OL_RING=.. OL_WIN_PAGES=.. OL_MAT_SELECT=.. cb_bytes/mover=.. shared=..`.

## Risks to check on device
- **L1.** PB=8, RING=8, WIN=1024 is ~493 KB of CBs per mover, so ~1.0 MB per core for 2 movers plus the shared CBs (the window, 192 KB,
  and the ring, 260 KB, are most of it). This has not been checked against the real L1 limit.
  - An overflow fails loudly at program creation (CB/L1 clash).
  - Fallbacks, in order: `GSPLAT_TT_OL_WIN_PAGES=768`, then `512`, then
    `GSPLAT_TT_OL_RING=4`.
- **Kernel code size.** The select path adds code to the materialize kernel. Watch
  for a kernel config buffer overflow, as in #114's watcher note.
- **Select work.** The test's worst case is 151% candidates per part (spread keys
  with outliers). Real depth keys may differ. GSPLAT_TT_MAT_STATS=1 prints the
  worklist max_item / mean_slot from the model only, so the device zones are the
  real check.

## Tests run (Mac, no device)
- `tests/syntax_stub/check.sh`: all ok, including 4 new v2 entries (emit PB8/RING8
  PUBOC WIN1024; PB16/RING2 without PUBOC; materialize OL_MAT_SELECT; with FUSE_CULL).
- `CXXFLAGS="-Irender/host -Irender/kernels/dataflow -Isrc" tests/unit/run_cpp.sh tests/unit/test_sort_onelaunch_v2.cpp`: PASS.
  - Select equals std::stable_sort for 5 key kinds (random, few distinct, all
    equal, sorted, spread + outliers) and n in {1, 17, 4097, 16385, 25700, 32768}.
    Every 4096-record part of each subchunk is checked. Max candidates: 151% of the
    part.
  - Ring model: 6 movers share tiles, R in {2,4,8,16}. Checks that the bucket image
    equals per-record writes, that no write leaves a mover's own cursors or crosses
    a page, that runs are at most R long, and that counts match.
  - Worklist on a bicycle-like case (1014 tiles, 110 cores): parts cover each tile
    exactly, big items go to NCRISC, max_item <= 1.5x mean. Model: 16384 vs 13509
    (1.21x). The max is now a whole-tile prepack item, not a big tile.
  - Mutation check: 7 of 8 mutants killed. The survivor (`>=` on r_lo) only adds
    candidates and stays correct.
- Still pass: test_sort_onelaunch, test_sort_radix_tile, test_sort_tail_dual_mover,
  test_sort_bin_dual_mover, test_sort_bin_fp32.
- `python3 -m pytest tests/spec -q`: 32 passed, 6 skipped, 2 xfailed.

## Device A/B (yyzo-bh-07 p100a, bicycle, 30 views)
Scripts are in this folder, adapted from #114: drive.sh, remote_job.sh and
remote_tracy.sh. The remote tree is /localdev/smarton/gstt2-t124. The md5
reference is /localdev/smarton/t82_scripts/md5-r82new.txt.

Run from the repo root of a worktree at the pushed commit:

```
tt-project/harness/bin/ssh-preflight yyzo-bh-07
ttp lock p100 -- bash docs/sort-onelaunch-v2-t124/drive.sh <sha> chk
ttp lock p100 -- bash docs/sort-onelaunch-v2-t124/drive.sh <sha> ab
ttp lock p100 -- bash docs/sort-onelaunch-v2-t124/drive.sh <sha> tracy
```

1. **chk.** ONELAUNCH=1 with CHECK=1 on 5 views, then the same with
   TT_METAL_WATCHER=2 and SFPU_VIS=0 on 3 views. Pass means:
   - `bad_tiles=0` on every frame;
   - ALL_VIEWS_IDENTICAL;
   - the `[SORT] ONELAUNCH v2 OL_PB=8 OL_RING=8 OL_WIN_PAGES=1024 OL_MAT_SELECT=1`
     line is present;
   - no CB/L1 or kernel-size error.

   If L1 fails, rerun with `GSPLAT_TT_OL_WIN_PAGES=768` added to every arm.
2. **ab.**
   - Hang check: 3 back-to-back 30-view runs with ONELAUNCH=1.
   - 3 interleaved rounds of base (ONELAUNCH off) vs on (v2 defaults).
   - Round 4: kill (v1) and nosel (v2 emit only). Round 5: v1sel (select only).
     These split the gain between the emit and materialize changes.
   - One MAT_STATS run.
   - Every arm must report ALL_VIEWS_IDENTICAL.
3. **tracy.**
   - A 10-view Tracy run with v2 defaults. Read the `sort_ol_emit` makespan
     (v1: 8.26 ms; target about 5-6 ms) and sort_subchunk_mat.
   - A second run with GSPLAT_TT_MATCULL_PROF=1 for the mat_ol_keys / sort /
     gather / wr zones. Use these to calibrate `cnt/4 + 2*recs` in
     sort_mover_split.h.

**Gate:** adopt (default on) only at >= 3 ms/view below base, md5-identical.
Record results in docs/sort-onelaunch-v2-t124/results.md and the status HTML.

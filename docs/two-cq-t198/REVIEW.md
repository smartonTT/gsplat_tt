# Review of task #198: 2-CQ bridge hiding (task #203)

Reviewer: task #203, independent. Head reviewed: 7ab606a (commits 8889862, e7d25e0, 64d951e, plus the docs and driver commits).
Scope: read the code for correctness, check the recorded device evidence, and run the unit test. No device run: no bug was suspected.

## Verdict: PASS

No blocking problems found.

## What was checked

1. **CQ0/CQ1 event ordering** (`tile_assign_device.cpp`)
   - CQ1 path:
     - CQ0 records `k2_done` right after the K2.
     - The early sort is enqueued on CQ0.
     - CQ1 waits for `k2_done`, then reads proj_M with a blocking read.
     - So all CQ1 work runs after the K2 and after everything before it on CQ0, including the previous view's mat and blend.
   - No-CQ1 path: a non-blocking `ReadShard` of proj_M, then `enqueue_record_event_to_host`, then the sort enqueue, then `EventSynchronize`. `mread` is valid before it is read.
2. **Buffer lifetimes and hazards across the two queues** (`sort_device.cpp`)
   - The bridge swaps `ctx->cq` to CQ1. During the swap, the helpers only allocate buffers and write to them: `publish_sort_downstream_metadata`, `upload_resident_tile_ranges`, `prepare_subchunk_buffers`, `upload_subchunk_directory`, the bucket_meta / tile_ids writes, and the mat work list. None of them enqueues a program, so no CQ1 program can overlap the CQ0 sort on the same cores.
   - None of those buffers is a sort input. The sort reads gids, tids, keep, depth, blendrec and the K2 rows, and writes bucket, bases and totals.
   - In fold mode the sort never writes the K2 count rows (`!fold` guards the step-1b write), so the CQ1 read of those rows while the sort runs is safe.
   - Grow-only reallocations in the bridge free only buffers that no queued work still uses.
   - `Finish(CQ1)` runs before the mat is enqueued on CQ0. This also covers the empty work-list case.
   - `CqSwap` restores `ctx->cq` on every return path.
   - Non-blocking writes copy host data at enqueue time, so the host vectors can go out of scope.
3. **Kernel `P == 0xFFFFFFFF` path** (`sort_bin_onelaunch.cpp`)
   - The P page goes into the mover's own H row. Mover 1 later overwrites it with BRISC's count row. Mover 0 never reads its own hp under fold.
   - The K2 publishes `P_pub = min(P, p_cap)` with a write barrier. The early sort therefore stays in bounds even when the pairs overflow.
   - Speed sums: K2 mover m gets `acc[2c+m]` and `acc[2c+m+1]`. The sort's kol (mover 1) gets `acc[2c+1]` and `acc[2c+2]`, and kol0 (mover 0) gets `acc[2c]` and `acc[2c+1]`. These match.
   - The coordinator NoC list moved from arg 28 to arg 30. Both host paths were updated: the non-early path pads with `0, 0`.
4. **Pair-overflow fallback** (`mread[2] != 0`)
   - The early sort read the clamped pairs, which is safe.
   - `Finish(CQ0)` drains it before `ensure_pair_buffers` regrows (frees) the pair buffers.
   - The K2 rerun takes the normal path, and the rows publish `early = cq1 = false`, so the sort runs at its usual place.
   - With host_free, the old hard fail is kept.
   - Not exercised on device: bicycle does not overflow.
5. **Kill switches**
   - `GSPLAT_TT_SORT_OL_EARLY=0` also forces `mat_cq1()` off.
   - It also restores:
     - 1 CQ with the same `create_unit_mesh` arguments as before;
     - the blocking proj_M and `ta_pairs_P` reads;
     - the zeroed depths vector;
     - the device totals read;
     - `bridge_cq = nullptr`.
   - `MAT_CQ1=0` with early on: the sort is early, but the totals and uploads go on CQ0.
   - On device, every arm (off, early, both, defaults, `=0`) gives md5 46a725ab over all 30 views.
   - Small differences when the switch is off, none of which changes the output:
     - an extra `[DEV]` log line;
     - `take_k2_count_rows` and `ol_frame = false` now run before the P read (stricter, since rows are taken every view);
     - `t_e0` starts earlier, so `bin_emit_ms` now includes the host noc_xy and speed-bounds computation.
6. **`proj.depths` removal**
   - `sort_and_bin_tt` ignores `depths` (`(void)depths`), so passing a null `data()` is safe.
   - `num_visible` is only set by the count-only readback. `render.cpp` falls back to it only when `depths` is empty.
7. **Device grid**: the 2-CQ device keeps the 11x10 compute grid (`[DEV]` lines in the run logs).

## Evidence checked

- `tests/unit/test_two_cq_t198.cpp`: PASS (run locally on this head).
- `out-tip/md5-rt{1,2,3}-{base,both}.txt` and `md5-rts-defchk.txt`: 7 files of 30 lines each, all hash to 46a725ab, the same as md5-r82new.
- Logs:
  - 93 x `MAT_CQ1 host totals vs device: 0 mismatches`;
  - 155 x `ONELAUNCH_CHECK bad_tiles=0`;
  - `early=1 why=0` in every early run.
- Spot check, run-rt1: view_total 15.605 (off) vs 15.185 (on). This matches the claimed paired mean of -0.418 ms/view.

## Non-blocking notes

- In the no-CQ1 path, `ReadShard(..., false)` writes into the stack vector `mread`. If `enqueue_sort()` throws (for example, the program build fails), the stack unwinds while that read is still pending. The completion reader could then write into freed memory. This only happens on a hard-fail path. Draining CQ0 in a catch, or falling back to a blocking read, would close it.
- The overflow fallback has no device test. A forced-overflow smoke with a small pair ceiling would cover it.
- Measured on yyzo-bh-07 (p100a), the same board as the earlier iterations.

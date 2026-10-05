# t185: drop mask-0 records in mat before blend (model only, no device)

**Verdict: do not build.** Net saving is +0.03 to +0.16 ms/view at best (compaction skipped on the
mat-critical item), and -0.13 to +0.09 ms/view if every item compacts. Gate is 0.3 ms/view.

## Where the mask comes from
The fused mat+cull (`render/kernels/dataflow/sort_subchunk_materialize.cpp`, `patch_batch`) writes
the 32-bit microblock mask into word3 of each slab record in place. The blend loop
(`render/kernels/compute/alpha_blend_compute_mb.cpp`, `process_tile_l1_blend`) reads `rec[3]` and
skips the record if the mask is 0. So the movers know the mask and could compact the slab.

## Blend saving (the only gain)
- Dead (mask-0) records on the PRECULL=2 tip: 0.550 M (view 0) and 0.398 M (view 1), 19.0 % of
  records (`docs/blend-fixed-t172/out/tc_split.txt`). 30-view PRECULL=1: 0.488 M, 16.5 % (t148).
- A dead record only pays the per-record scan on MATH: 0.0115-0.0161 us (t172 fits, 16-22 cycles).
  t148's 0.244 ms used an assumed 44 cycles/record and also counted live-but-saturated records.
- Saving: **0.04-0.08 ms/view** with the measured cost, 0.12-0.16 ms with the 44-cycle assumption.
  Fewer DRAM bytes (17.6 MB/view less write in mat and read in blend) are off the critical path
  (MATH is 98 % busy in the loop; mat slab writes are ~0.06 ms in t86).

## Mat cost
- Compaction copies the 7 other words of every kept record (~81 %) to its new slot after the mask
  arrives: ~0.010-0.030 us/record extra, against 0.059 us/record for today's cull+write.
- On the item that sets the mat window (one 14-16k-record whole tile on one NCRISC,
  `docs/mat-split-sort-model/RESULT.md`) that is +0.14-0.48 ms traced, +0.07-0.24 untraced,
  which cancels or exceeds the blend saving.
- Skipping compaction on that item keeps the window, but adds +0.13-0.40 ms to the mean mover
  against 0.618 ms traced slack, and the blend reader then needs per-subchunk counts from mat
  (~0.015 ms if not hidden).

## md5
Not automatically identical. The T read-back runs every 512 records of `g_seen`, which counts dead
records too (`BLEND_T_PERIOD`). Dropping records moves the read-backs and changes which saturated
microblocks are skipped. The record has no spare bit, so keeping md5 needs marker records: mat
keeps (or inserts) one mask-0 record at each original 512-boundary and the blend reads back T on a
mask-0 record instead of on the counter. That is extra mat and blend code for a ~0.1 ms gain.

## Reproduce
`python3 docs/mask0-drop-t185/model.py` (output: `model.txt`).

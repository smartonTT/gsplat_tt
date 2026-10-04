# t184 — model: hide the host mat bridge under the sort_ol emit

No device run. Code read at f39a023 (tip of smarton/tt-project-opt). Numbers come from
`docs/t171/README.md` (untraced bridge) and `docs/rerank-17ms.md` on 21a922b (traced gaps).

**Answer: feasible and md5-safe, but the hidden time is about 0.25-0.30 ms/view untraced
(central ~0.28). That is at or below the 0.3 ms gate, so it is not recommended for a build now.**

## What the bridge is today (`render/host/sort_device.cpp`, one-launch path)

After `EnqueueMeshWorkload(wl_onelaunch)` the host does a blocking read of the 8 KB totals
buffer (`buf_ol_totals`), which returns only when sort_ol has ended. Then, on the host:
per-tile layout + `build_lpt(pad_counts)` + downstream metadata (bin_layout 0.097 ms),
subchunk layout + `build_mat_worklist` + subchunk uploads (publish_host 0.157), then
`launch_subchunk_materialize` (0.041). The device idles for all of it: ~0.30 ms untraced
(t171), 0.305 traced. Separately, a 64 B P read (`pread`, 0.015) runs before sort_ol.

## (1) Getting the per-tile totals before sort_ol ends

- The totals are a pure function of the K2 count rows. In fold mode the sort_ol prefix
  (`render/kernels/dataflow/sort_bin_onelaunch.cpp`, step 3) computes, per tile t,
  `tot[t] = sum_c (row[2c][t] + row[2c+1][t])` and
  `tot[stride+t] = sum_c round_up(row[2c][t] + row[2c+1][t], 16)`. The emit never changes
  them. The host can compute the same integers from `buf_k2_rows`.
- Those rows are final before sort_ol is enqueued: `tile_assign_device.cpp` already does a
  blocking `projM` read right after the K2 (the existing K2 -> sort_ol sync), and the fold
  decision (`why == 0`) is made on the host before the sort_ol launch.
- The rows are 110 cores x 2 movers x 64 pages x 64 B = 0.9 MB. At the measured D2H rate
  (t171: 3 MB image in 0.197 ms, ~15 GB/s) that is ~0.06-0.08 ms, plus ~0.05 ms to sum
  225 K words on the host.
- **Single CQ (read the rows at the existing sync, before enqueuing sort_ol):** works, but
  the ~0.12 ms read + sum lands on the critical path. Net ~0.17 ms. Not enough.
- **Second command queue (recommended form if built):** enqueue sort_ol on CQ0, then read
  the rows on CQ1 (blocking read on CQ1 only; #155 found non-blocking
  `EnqueueReadMeshBuffer` TT_FATALs). No kernel change, no flag, no new sync on CQ0. On
  Blackhole both 1 and 2 CQs use the same dispatch column
  (`tt_metal/core_descriptors/blackhole_140_arch.yaml`, same 11x10 compute grid for the
  2x-harvested p150), so CQ1 costs no worker cores. The change is in
  `device_state.cpp`: `create_unit_mesh(..., num_command_queues=2, ...)`.
  Polling an 8 KB totals page on CQ1 would also work, but needs a frame stamp written after
  barrier 2. Reading the K2 rows is simpler.

## (2) Queuing the uploads and the mat launch before sort_ol ends

- None of the bridge's buffers is a sort_ol argument (sort_ol takes gids, tids, keep, depth,
  blendrec, bucket, count rows, base rows, totals). The bridge writes bucket_meta,
  tile_ids, lpt_meta, tile_counts, P_kept, cull_mask_base, tile_ranges, subchunk meta,
  prefix and dir, and mat_work. The previous view's blend has finished (its d2h is
  blocking). Allocations are grow-only and steady-state does not grow.
- On CQ0, every buffer write inserts a dispatch wait for running workers
  (`tt_metal/impl/buffers/dispatch.cpp`, `issue_wait`). So CQ0 writes queued behind sort_ol
  are safe, but they run serially after sort_ol: ~11 writes, ~90 KB, an estimated
  0.03-0.08 ms that stays exposed.
- Better: issue the writes on CQ1 as well, `Finish(CQ1)` on the host (this wait happens
  while sort_ol runs), then enqueue mat on CQ0. Mat then follows sort_ol back to back. The
  traced back-to-back gaps on the in-order CQ are 0.006 ms (pfwc -> K2, mat -> blend).
- The host timeline after the sort_ol enqueue: CQ1 read ~0.1, sum ~0.05, bridge ~0.25,
  mat enqueue ~0.04, about 0.45 ms in total, against an emit window of ~3.5 ms. The slack
  is ~3 ms, so the host can also wait ~0.4 ms before the CQ1 read to keep it clear of the
  prefix phase. The prefix also reads these rows and is contention-sensitive (t170).

## (3) LPT order and md5

`build_lpt`, `build_subchunk_layout` and `build_mat_worklist` are deterministic functions of
`counts` and `pad_counts`. The `n > cap` and `MAX_TILE_ENTRIES` checks use the same values.
Computing them from the K2 rows gives the same integers that the device prefix writes, so
the LPT order, the worklist and the md5 are unchanged. A cheap
`GSPLAT_TT_SORT_ONELAUNCH_CHECK`-style debug compare (host sums against `tot`) would prove
this on device. The path must stay limited to `fold == true`; otherwise it falls back to
today's totals read.

## Estimate (untraced, ms/view)

| item | ms |
|---|---:|
| device gap sort_ol -> mat today (t171 bridge without pread; traced 0.305) | ~0.30 |
| left over: mat launch back to back | -0.006 to -0.01 |
| left over if the uploads stay on CQ0 (no second-CQ writes) | (-0.03 to -0.08) |
| CQ1 read contention with the emit (0.9 MB vs >100 MB emit traffic) | 0 to -0.03 |
| **hidden, 2-CQ read + 2-CQ writes** | **0.25-0.29** |
| optional: drop the redundant `pread` (K2's projM read already returns P in mread[1] and an overflow word in mread[2]; check P_pad/P_true use first) | +0.015-0.02 |
| **best case** | **~0.27-0.31** |

#171's caveat (run.py times each view's latency) does not hurt this lever. The idle gap is
inside one view's `render()`, so hiding it shortens the measured latency directly, unlike
d2h overlap.

## Recommendation

**Do not build now.** The central estimate (~0.28, ~0.30 with the pread drop) sits at the
0.3 ms gate. #155 is a close precedent: it hid ~0.2 ms of untraced host bridge before
sort_ol with an early enqueue, and the paired untraced gain came out at 0.03 ms (rounds
+0.14 / -0.07). Realized gains for host-gap hiding have come in well under the model, and
a ~0.28 effect would take several paired rounds to separate from the ±0.1 noise.

Revisit if any of these becomes true:
- a second CQ lands for another reason (the remaining work is ~100 host lines, no kernel
  change);
- the bridge grows past ~0.45 ms (more tiles, higher resolution);
- it is bundled with another sub-gate host lever that also needs CQ1, so the combined
  effect clears 0.3 ms.

Caveat: the tt-metal semantics were read from the local vendored checkout
(`backends/tt/tt-metal`). The board's `/localdev/smarton/tt-metal` may differ in detail.

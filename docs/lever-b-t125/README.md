# Lever B (task #125): visible-gaussian compaction and TA pairs fused into the pfwc writer

Code only, not yet run on a device. Switch: `GSPLAT_TT_PFWC_FUSE` (0 = default and kill
switch, the lever 2 path; 1 = fused). It needs `GSPLAT_TT_SFPU_VIS != 0`; any other value
throws.

## What changes with `GSPLAT_TT_PFWC_FUSE=1`

Per frame, the lever 2 chain (tip a03bd1e) is: pfwc (writes 10 tiles, mask and per-tile counts to
DRAM) -> host Finish -> gather scan -> gather scatter (compaction) -> proj_M read -> tile_assign
K2. The fused chain is:

1. **pfwc** with `writer_pfwc_fuse.cpp`. The tiles are dealt strided (core c owns tiles c, c+C, ...),
   which is the legacy compaction order (`vis_tile::SeqMap`). The writer classifies each tile in L1
   and writes every visible gaussian directly to the compact streams (`proj_m_depth`, the 64 B
   `proj_m_blendrec`, `proj_m_offs`, `proj_m_aabb`) at storage index `seg_base(c) + j`, where
   `seg_base(c) = SeqMap.first(c) * 1024` (`pfwc_fuse.h`). It then writes one counts page
   `[m_c, p_c]` per core. None of the pfwc tiles, the mask or the per-tile counts go to DRAM. There
   is no host Finish after pfwc.
2. **Segment K2** (`tile_assign_scatter_seg.cpp`, launched from the gather stage through
   `tile_assign_fused_k2`). Every mover reads the counts table, builds the segment table and emits
   its page range of pairs `(gid = storage index, tid)` in the legacy gaussian-major order. Core 0
   publishes `ta_pairs_P` and `proj_M`. The host makes one blocking read of `proj_M`, which is the
   stage's only sync. If the pairs overflow, the pair buffers grow and the K2 reruns. In host-free
   mode, P is clamped to `pair_ceiling()` as before and sort hard-fails.
3. **tile_assign_tt** skips its own K2 (`fused_ready`). Sort and blend are unchanged: they read
   depth and blendrec by gid, and those buffers are `padded_n` long.

The storage index is a monotone map of the legacy dense gid, so the pair order and therefore the
image should be md5-identical.

What the fused path removes: the gather scan and scatter programs (3.76 ms gather in t115), the
pfwc tile, mask and count writes, the tile_assign K2 launch, and one host Finish. What it adds:
pfwc writes 4 compact streams, and the segment K2 does the same work as the legacy K2. The t115
estimate is 3.8-5.6 ms/view gross.

Fallbacks before the enqueue (with a log line, running the lever 2 program):
- the gather outputs or scene colors are missing;
- there are more than 128 pfwc cores;
- the fused program fails to build.

After a fused pfwc there is no fallback, because the pfwc_* tiles were not written. The gather
throws if the chain is not resident.

## Files

- `render/kernels/dataflow/pfwc_fuse.h`: segment layout, segment table, K2 page split and pair walker. This header is shared by the kernels and the host test.
- `render/kernels/dataflow/writer_pfwc_fuse.cpp`: the fused pfwc writer (CB 40 staging).
- `render/kernels/dataflow/tile_assign_scatter_seg.cpp`: the segment K2.
- `render/kernels/dataflow/reader_pfwc.cpp`: PFWC_VIS reads arg 12 = tile stride. The lever 2 host passes 1.
- Host: `pfwc_device.cpp` (fused program and args), `gather_visible_device.cpp` (`gather_visible_fuse_prepare`, fused branch), `tile_assign_device.cpp` (`tile_assign_fused_k2`, K2 skip), `vis_mode.h` (`pfwc_fuse_mode`), `render.cpp`.
- `tests/unit/test_pfwc_fuse.cpp`: the host model. It runs the legacy compaction and K2 against the fused writer and segment K2 (161 cases), checking random visibility, core counts 1/3/13/110, K2 core counts, the dual split, and the overflow clamp. The checks cover segment overlap, M, P, every pair page written exactly once, pair gid and tid after the gid map, padding, and depth by gid.

## Checks run (Mac, no device)

- `tests/syntax_stub/check.sh`: all 29 OK, covering the new kernels with and without defines, the host drivers, and render.cpp.
- `CXXFLAGS="-Irender/kernels/dataflow -ffp-contract=off" tests/unit/run_cpp.sh tests/unit/test_pfwc_fuse.cpp tests/unit/test_vis_lever2.cpp`: PASS, PASS.
- `pytest tests/spec`: 32 passed, 6 skipped, 2 xfailed.

## Device A/B (#122)

The driver is `docs/lever-b-t125/drive.sh`. It runs on yyzo-bh-07 p100a (not bh-30), with each
step under `ttp lock p100`, and reuses the t115 remote scripts with `T115_TREE=/localdev/smarton/gstt2-t122`:

```
ttp detach t122-ab -- bash docs/lever-b-t125/drive.sh <sha>
# retry_when: grep -q CHAIN_DONE $TTP_RUN_DIR/t122-ab.log  (or test -f $TTP_RUN_DIR/t122-ab.rc)
```

1. Sync and build `<sha>`.
2. Run 3 interleaved untraced rounds of `base` (kill switch) against `fuse` (`GSPLAT_TT_PFWC_FUSE=1`). Each round is 30 bicycle views, with md5 compared against `md5-r82new.txt`.
3. Run the host-profile arm of the fused path.
4. Capture 30 views with Tracy on the fused arm. Compare it with the t115 capture: pfwc window, segment K2 window, gather window gone, and gaps.
5. List the ELF sizes of the pfwc kernels.

Gates:
- The fused arm must be `ALL_VIEWS_IDENTICAL` in every round.
- The net gain must be at least 3 ms/view, as the median over the 3 rounds.
- There must be no `TT_THROW` or `TT_FATAL` in the run logs.

If the gates pass, make 1 the default (keeping 0 as the kill switch) and update the status HTML.

If something fails:
- **Kernel config buffer overflow** (a TT_THROW about program or kernel config size at the first fused launch): move the RECHECK soft-float path in `vis_tile::classify_tile` out of line, or into a separate noinline section, then rerun.
- **md5 differs**: rerun the fused arm with `TT_METAL_WATCHER=2` for 5 views. Then compare `proj_M` and `ta_pairs_P` against the kill-switch arm. M and P must be equal.
- **Hang**: check that the CB 40 layout in `writer_pfwc_fuse.cpp` fits `FUSE_CB_BYTES`, and that the segment K2 CB fits `(nseg + 6) * 64` bytes.

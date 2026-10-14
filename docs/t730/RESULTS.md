# Task #730: fp32 op/colour cost in train mode (status: code done, device A/B pending)

Bound (#727, bh-49 p150, b2b): the reader's random 64 B gather + L1 copy cost
1.42 ms/view (14.681 on vs 13.261 with no gather; gather 1.06, copy 0.36).

## Side slab from sort/materialize: sized, not built in this run
A 16 B fp32 [op,r,g,b] slab per record in slab order would turn the gather into a
contiguous read. The data is already at hand without a random gather: pfwc blendrec
words 5..8 are the fp32 op/colour, and the onelaunch emit walks them gaussian-major.

Extra cost estimate (from measured p150 zones, docs/b2b-timeline-p150-223 anaM;
P = 2.0-2.7 M pairs/view):

| stage | added work | est. ms/view |
|---|---|---|
| onelaunch emit (ring + flush) | +16 B write per pair (+50% of 32 B records) | ~0.10 |
| materialize perm/rd/wr (nc_perm 0.19, nc_rd 0.096) | +50% bytes | ~0.15 |
| blend reader | contiguous 16 B/record read | ~0.15 |
| total | | ~0.3-0.5 |

Net ~1.0 ms/view against the 1.42 bound, so it does not eat the gain. Scope: TRISC
ring pack (sort_ol_town_compute), sort_bin_onelaunch ring/scratch/flush paths,
materialize in-budget + overflow (sc==0, sc>=1) + OL_MAT_SELECT paths, host buffers
in sort_device.cpp, reader. That is several runs of work.

## Built first: gather on the blend writer (GSPLAT_TT_TRAIN_OPC_WRITER=1)
Cheaper lever, same bytes (lossless by construction): the reader (NCRISC/NoC1, on
the slab critical path) queues {slab L1 address, count} on CB 14; the writer
(BRISC/NoC0, idle between tiles) gathers into CB_TRAIN_OPC and pushes it. The
writer loop is non-blocking (serves gathers, writes a tile only when its colour is
there). Bound: the d2 arm (13.26 ms/view) if the writer fully hides the gather.
Default off until the device A/B.

Unit test: tests/unit/test_train_opc_writer_t730.py (loop replay + wiring).
Device A/B, md5 3fada021, oracle, JIT check and screenshot: not run yet (follow-up).

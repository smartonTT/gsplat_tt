# t538: lever B2 (exact ellipse-tile pair test): device cost probe

Written 2026-10-12 (PDT). Base best-iter-225 (b612d4ed, 8.998 ms/view b2b on bh-33).

## Why a cost probe first

T-A (#533, docs/ellipse-pairs-tA/RESULT.md on ttp/t533-ellipse-pairs-tA) counts 0.183 M dead
pairs per view (6.8 %). Blend already skips them (mat's band cull gives them a zero mask), so
removing them saves only k2 + sort + mat per-record work: about 0.21 ms/view gross (estimate).

Where the test can run on today's pipeline:

- **K2 (TRISC jobs + movers), scalar.** K2 only sees each gaussian's lofs (pair offset) and
  packed AABB word (`vis_tile::aabb_pack`: 10 + 10 + 10 bits + tag, no spare bits). The conic and
  mean live in blendrec, which K2 never reads. B2 here needs: pfwc to write a new per-gaussian
  conic stream; the movers to read it into the TRISC job windows (K2_TCAP = 640 pages per
  window already, ~168 KB per job, 4 jobs per core); a dead-pair sentinel tid that K2 leaves out
  of its tile counts and sort's emit skips. P (the pair array) keeps its size, so K2 still walks
  every slot: only sort emit + mat gain.
- **pfwc, SFPU.** Exact counts at the source would also cut K2, but K2 enumerates pair p as
  (gaussian, dx, dy) of a rectangle; a non-rectangular tile set needs a per-row span table per
  gaussian (variable length, up to 32 rows), a new stream and a new K2 walk. Days of work.

Either way the per-pair test in K2 is scalar integer code (no FPU on the RISC-Vs): T-A
estimates ~40 cycles per tested pair, ~0.14 ms/view, against ~0.21 ms gross. That makes the
test's real device cost the deciding number, and it is cheap to measure without building
the lever.

## The probe (GSPLAT_TT_ELLIPSE_PROBE=1, profiling only)

`pfwc_fuse::EllipseProbe` (render/kernels/dataflow/pfwc_fuse.h) runs T-A's exact test in
integer form (two clamped face candidates of the conic over the tile's pixel-centre box,
quarter-pixel fixed point) on every pair of a gaussian whose rect is >= 2 x 2 tiles (the only
rects with dead pairs: a 1-wide or 1-high rect's ellipse crosses every tile it spans). The
parameters come from the box, so the probe does not pay the conic stream reads the real lever
needs: its cost is a **lower bound** on the lever's K2 cost. The keep flag goes to a sink;
pairs, counts and the image are unchanged. Host define: K2_ELLIPSE_PROBE on the K2 movers and
the TRISC compute kernel (render/host/tile_assign_device.cpp).

## Result

(pending: docs/ellipse-pairs-t538/out/)

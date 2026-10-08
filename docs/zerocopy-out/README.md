# Zero-copy pinned output (tasks #374, #395)

`GSPLAT_TT_OUT_ZEROCOPY=1` (implies `GSPLAT_TT_OUT_PINNED=1`, default off): the
blend writer writes the u8 image into a ring of pinned, NoC-mapped host buffers
(`render/host/out_ring.h`) and `render_view` returns a numpy view of that buffer
instead of copying it out. The view holds the slot's lease, so the ring never
writes a buffer the caller still holds. Lifetime: a returned image stays valid as
long as the caller keeps it (and at least until the frame after next). With the
flag off the path is unchanged (md5 906e0435 on 30/30 views).

## Fix (review #391, task #395)

Zero-copy passes no image buffer to the sort->blend continuation
(`image_out == nullptr`), and `sort_device.cpp` enabled the fused mat+blend
program (iter 206, `GSPLAT_TT_MATBLEND_FUSE`) only when `image_out != nullptr`.
So the zero-copy arm in #374 silently ran the unfused materialize, then the
blend: +0.6 ms blend span and sort `mat` 0.063 vs 0.037 ms. Both "has output"
gates now use `sort_blend_has_output()` (`render/host/sort.h`, unit test
`tests/unit/test_sort_blend_has_output.cpp`). The blend logs
`MATBLEND_PROGRAM fz=<0|1> zerocopy=.. pinned=.. frame=N` on the first frame and
on every change; every run below logged `fz=1` at frame 0 and never changed.

## A/B on yyzo-bh-04 (p100a, Ryzen 5 7600X), build 1197ea1e, bicycle, 30 views

One build, 3 alternating rounds (`drive.sh` with `O=docs/zerocopy-out/out-t395
SHOT_NAME=t395-zerocopy-out`, outputs in `out-t395/`), all under one
`ttp lock p100`.

| arm | r1 | r2 | r3 | mean ms/view | blend span ms | d2h ms | sort mat ms |
|---|---|---|---|---|---|---|---|
| base | 10.951 | 10.965 | 10.945 | 10.954 | 7.067 | 0.263 | 0.038 |
| pinned (copy) | 10.899 | 10.908 | 11.009 | 10.939 | 7.073 | 0.261 | 0.038 |
| zero-copy | 10.660 | 10.705 | 10.675 | **10.680** | 7.069 | 0.000 | 0.036 |

md5 matched golden 906e0435 on 30/30 views in all 10 runs (smoke + 9).
Zero-copy's blend span now matches base; removing the copy saves 0.274 ms/view
(-2.5%) vs base and 0.259 ms vs pinned copy mode. Pinned copy mode is no faster
than the base device read on this board.

Device hero with zero-copy on (commit 1197ea1e):
`opt/metal-screenshots/t395-zerocopy-out/hero.png`, diff
`hero_diff10.png`, 42.51 dB vs `benchmarks/reference_v2/hero.png`, golden md5
match (max 0 LSB). Both images viewed: no tile seams, blocky or empty tiles;
the diff only lights thin edges (spokes, frame, bench slats, foliage).

## Superseded: #374 numbers (fused program off in the zero-copy arm)

`out/` (build 8fd193fd) and `diag/` rounds 11-14 ran zero-copy without the
fused program, so their zero-copy rows (11.29 ms/view, blend 7.64-7.72 ms) and
the explanation given then ("spreading writes over more pinned pages") are
wrong for zero-copy: the +0.6 ms was the unfused mat+blend.

The copy-mode diag stands on its own (it kept the fused program): rotating the
writer through 3 or 4 pinned buffers (`GSPLAT_TT_OUT_PIN_HOLD=2/3`) keeps the
blend span at ~7.0-7.1 ms but the copy gets slower (d2h 0.37-0.58 vs ~0.2-0.28
ms with one hot buffer), because the copy reads a colder source.

| copy-mode arm (diag) | r11 frame / blend / d2h | r12 | r13 | r14 |
|---|---|---|---|---|
| pin (1 buf) | 10.83 / 7.135 / 0.208 | 10.91 / 7.057 / 0.281 | 10.92 / 7.076 / 0.262 | 10.85 / 7.076 / 0.191 |
| pinhold=2 (3 bufs) | 11.01 / 7.006 / 0.365 | 11.21 / 7.056 / 0.580 | | |
| pinhold=3 (4 bufs) | | | 11.20 / 7.099 / 0.540 | 11.21 / 7.088 / 0.557 |

## Recommendation

Zero-copy gains 2.5% of the frame on the p100a (iter 211, flag on). The flag
stays off by default here: it changes what `render_view` returns (a strided
view into a pinned ring slot instead of a fresh array), and the cross-view
overlap default (#388/#393, xv + pinned output) also touches the output path.
Next: A/B xv + zero-copy against xvpin on the same box, then decide the default.

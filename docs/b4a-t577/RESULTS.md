# t577: B4a fold under p150 ETH dispatch (viewer path)

## Root cause and fix
The fold runs K2 (tile_assign_scatter_seg.cpp) inside matblend_brisc.cpp, and K2 kept a 4 KB
uint32 count row on the 8 KB BRISC stack. Under ETH dispatch the firmware takes 80 B more local
memory, so the static margin was 48 B (128 B under worker dispatch). With the uncounted firmware
and poll frames the stack ran into g_b4a_fn and K2_YIELD jumped to a wild PC: the #562/#566 hang.

- 1fcbb6e3: under B4A_K2 the count row is uint16 (K2_CNT16, 2 KB less stack), counted in chunks
  of <= 4095 pages (no wrap), flushed and summed into the L1 row between chunks.
  Host test tests/unit/test_pfwc_fuse.cpp covers chunks 1/3/4095.
- 722cdd7f: ETH gate lifted (dispatch_select.h b4a_fold_on true for both dispatch kinds); the
  viewer deploy uses the tree default (fold on).
- 8b40551e: opt/brisc_stack_check.py skips firmware/ ELFs (a fresh JIT cache's firmware brisc.elf
  has no __stack_base and made run.py exit 9); test in opt/test_brisc_stack_check.py.

## bh-30, ETH dispatch, 12x10 grid (build ab2ae01e), 2026-10-12 21:48-22:18 PDT
Viewer stopped for the run and restarted after it (localhost:8091 -> 200).
- Stack margins (fresh cache): tile_assign_scatter_seg 352 B, matblend_brisc 1024 B, others >= 3648 B (out/b2/stack.txt).
- Fold verify (GSPLAT_TT_B4A_VERIFY=3) OK on 3 views, latency and b2b.
- 30-view md5 list 39d84b28 = the ETH 12x10 golden (out/b2/md5-dump.txt). 906e0435 is the
  bench-grid golden; the 12x10 ETH grid has its own golden (docs/b2b-rebaseline-t465).
- Hero: out/b2/hero.png, diff out/b2/hero_diff10.png, PSNR 42.51 dB vs
  benchmarks/reference_v2/hero.png. Looked at both: no tile seams; mean diff on 16/32 px tile
  boundaries (0.85-0.87) equals the image mean (0.875).
- Soak: 4 x 1000-pass b2b runs, 1632 s (27 min), no hang, rc 0 (out/b2/soak.txt, logs gzipped).

b2b (render/run.py --back-to-back, 30 views, 20 passes), per-run median ms/view, alternating:

| arm | runs | median |
|---|---|---|
| fold off (GSPLAT_TT_B4A_FOLD=0) | 8.389, 8.384, 8.383 | 8.384 |
| fold on | 8.121, 8.109, 8.119 | 8.119 |

Fold on: -0.265 ms/view (-3.2%) on the viewer box (-0.305 ms by run mean).

## bh-33, bench path (worker dispatch), 2026-10-12 21:57-22:02 PDT, under ttp lock p100
Arms: base = best-iter-226 (90268457), new = a0478ed6 (this branch, includes the #574 stack fix 9cb7385b).
- md5 906e0435 on 30/30 views, identical between arms; hero opt/metal-screenshots/ttw-t577m/hero.png,
  diff hero_diff10.png, 42.51 dB, golden match (max 0 LSB). Looked at: no seams.
- Stack margins new: matblend_brisc 1104 B, tile_assign_scatter_seg 432 B (base 112 / 176 B).

| round | base | new |
|---|---|---|
| r0 (warm) | 8.603 | 8.630 |
| r1 | 8.608 | 8.605 |
| r2 | 8.613 | 8.627 |
| r3 | 8.595 | 8.629 |
| median r1-r3 | 8.608 | 8.627 |

+0.019 ms/view vs best-iter-226, the same as #574's stack fix alone (+0.024), which is already on
smarton/tt-project-opt; the K2_CNT16 change adds nothing measurable on the bench path.
Bench headline does not improve: no new best-iter tag.

## Verdict
Keep: land on smarton/tt-project-opt. The viewer gets the fold (-0.27 ms/view device time).

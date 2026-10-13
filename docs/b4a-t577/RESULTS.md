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

## Live viewer, bh-30, 614abe85 deployed (renew.sh viewer), 2026-10-12 22:49-23:05 PDT
Page FPS through localhost:8091 (`opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:8091
--subprotocol viser-v1.0.27 --no-decode --secs 23`), same build, same session; the arm switched by
restarting the viewer with/without VIEWER_B4A_FOLD=0. Outputs in viewer/.

| arm | runs (page FPS) | median | HUD device ms | HUD sent FPS |
|---|---|---|---|---|
| fold on (r1-r3) | 108.73, 114.20, 114.95 | 114.20 | 8.00-8.20 | 113-118 |
| fold off (off1-off3) | 118.57, 118.89, 119.01 | **118.89** | 8.04-8.09 | 116-118 |
| fold on (r4-r6) | 115.33, 115.50, 115.44 | 115.44 | 8.10-8.19 | 113-118 |

The fold ran on every hinted frame (120/120 cores, no `prepared=1 ... ran=0` line, about 8,000
frames), with no hang. It still costs about 3.5 page FPS (-3%) in the viewer, where the b2b bench on
the same box shows it 0.27 ms/view faster. Likely cause (not profiled): with the fold, the next
frame's pfwc is enqueued ahead of this frame's blend, so the device is idle during the viewer's
host gap between frames, which pfwc used to fill when it ran behind the blend. The bench's b2b loop
has almost no host gap, so it only sees the saving.

Change: opt/viewer/viewer.sh keeps the fold off in the viewer unless VIEWER_B4A_FOLD=1 (empty or
unset = off, so renew.sh redeploys keep it off); tests in opt/ird/test_renew.sh case 13.

## Verdict
Fix kept (landed on smarton/tt-project-opt at 614abe85): the fold no longer hangs under ETH dispatch.
Fold in the live viewer: shelved (-3% page FPS measured); the viewer runs fold off. No new best-iter.

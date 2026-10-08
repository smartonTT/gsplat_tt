# t437: why a few cores are slow in fused pfwc at ETH 12x10

CPU-only analysis of the #427 Tracy traces (branch ttp/t427-eth-tracy acff8061,
`out/dev-E12-c{0,10,20}.csv.gz`, `out/dev-E11-c*.csv.gz`; p150, bicycle 1024x1024, 30 views,
1350 MHz, all numbers traced unless marked). No device run, no code change.

**Answer.** The slow cores are slow because of what they get, not where they sit. The strided
deal (tile t -> core t mod C) gives some cores up to 24% more visible gaussians than the mean,
and their writers (record emission per visible gaussian) back-pressure TRISC. The pattern is the
same in every view but moves when the grid changes. A static weighted deal with equal tile counts
per core is modeled at **+0.116 ms/view traced (about +0.10 untraced) at 12x10**, and up to +0.22
if device per-tile counts capture the stable residual. That is at the 0.1 ms gate, so a device
A/B is proposed (below). At 11x10 the same fix is only worth +0.04 to +0.10.

## Method

| script | does |
|---|---|
| `ana437.py` | per core per view: pfwc-program BRISC/NCRISC kernel zones and TRISC0/1/2 pfwc zones, relative to program start -> `core_view-dev-E1{2,1}.csv` |
| `stats437.py` | slow vs median tables, view stability -> `cores-*.txt`, `end-*.npy`, `t1-*.npy` |
| `model437.py` | CPU visibility model of the bicycle scene for the 30 views (z>0.2, opacity>=1/255, 3-sigma AABB, 32-px tiles) -> `tilework.npz` (visible and pairs per 1024-gaussian tile per view) |
| `fit437.py` | per-view-centred core end vs tiles / visible / pairs per core -> `coef-*.npy`, `pred-*.npy` |
| `sim437.py` | predicted max-core end under alternative deals (fit + measured residual kept per core) -> `sim-dev-E1{2,1}.txt` |

Program end = max(BRISC, NCRISC) end, not the TRISC end (writers finish last).

## Per-RISC breakdown, 12x10 (ms from program start, mean of 30 views)

| core (phys, c) | TRISC0 | TRISC1 | TRISC2 | BRISC end | NCRISC end | BRISC tail us | NCRISC tail us | end | slowest in N views |
|---|---|---|---|---|---|---|---|---|---|
| slow (3,2) c=2 | 1.756 | 1.763 | 1.752 | 1.810 | 1.894 | 47 | 131 | 1.895 | 21 |
| slow (11,5) c=42 | 1.774 | 1.782 | - | 1.808 | 1.852 | 27 | 71 | 1.852 | 3 |
| slow (6,5) c=41 | - | 1.773 | - | 1.804 | 1.850 | 32 | 78 | 1.851 | 1 |
| slow (13,7) c=68 | - | 1.746 | - | 1.767 | 1.836 | 22 | 91 | 1.837 | |
| slow (4,2) c=3 | - | 1.734 | - | 1.765 | 1.832 | 31 | 98 | 1.833 | |
| slow (14,3) c=21 | - | 1.784 | - | 1.777 | 1.823 | -7 | 39 | 1.823 | |
| median (13,10) c=104 | - | 1.612 | - | 1.603 | 1.635 | -8 | 24 | 1.636 | |
| fast (2,10) c=97 | - | 1.521 | - | | | | | 1.545 | |
| fast (1,11) c=108 | - | 1.528 | - | | | | | 1.536 | |

("-": within 0.01 of TRISC1.) Tail = writer end minus TRISC1 end. Starts are within 0.8 us on
all cores (no launch skew). Mean core end 1.654, mean max-core end 1.898 (max/mean 1.147); per
view, the slowest core has TRISC1 1.781 vs mean 1.607 (+0.174) and writer tail 118 vs 47 us
(+0.071). NCRISC (reader + odd-chunk writer) ends last in 91% of core-views and on the slowest
core in all 30 views.

11x10: slowest (15,2) c=10: TRISC1 1.824, BRISC end 1.886 (tail 63 us), end 1.887, slowest
in 17 views; next (3,5) c=35 1.872, (1,3) c=11 1.870, (14,6) c=53 1.864; median (5,5) c=37
TRISC1 1.718, end 1.755; fast (4,9)/(3,9) about 1.69. Mean max-core 1.891, mean core 1.766
(max/mean 1.071). **12x10 does not beat 11x10 on pfwc (1.898 vs 1.891).**

## Reader-, compute- or writer-bound

Writer-bound. The compute kernel has no data-dependent branches, so per-core TRISC
differences are CB stalls. TRISC1 duration tracks the end (per-view corr 0.948) and the writer
tail adds to it (corr 0.693). The cost that varies by core is record emission per visible
gaussian. Reader DRAM wait is not measured in these traces (no step counters); #197 measured
the reader barrier at 0.098 ms with the reader idle 95%, and reader work is identical per tile
(10 input tiles), so it cannot explain a per-core spread that follows visible counts.

## Placement or content

Content.
- Same cores every view: view-to-view corr of per-core end 0.90; split halves 0.978 (12x10),
  0.985 (11x10).
- Not physical: the same physical core correlates 0.22 between 12x10 and 11x10. 12x10's slow
  cores rank 39/83/73/30/63/46 of 110 at 11x10, and 11x10's rank 49/63/73/54/29/12 of 120 at
  12x10. Column and row means are flat (harvested cols 7, 10 make no difference).
- Visible gaussians per core (CPU model) explain it: R^2 0.706 (12x10) and 0.621 (11x10);
  tiles + visible 0.734 / 0.662; pairs alone 0.10 / 0.12. Slope 0.050 us per visible gaussian
  (12x10). Visible per core max/mean 1.24 (12x10, mean 14157) and 1.21 (11x10). Model top-6
  [41,42,2,43,68,40] vs measured [2,42,41,68,3,21]. The extra-tile remainder cores (c < rem)
  are also slower (11x10 1.797 vs 1.742; 12x10 1.657 vs 1.628).
- The residual (std 41.5 us per core-view, 12x10) is stable across views (split-half 0.987)
  but not physical (0.166 by physical core across grids): it is per-tile content the CPU model
  misses (the model overcounts pairs by ~1.5x vs device P). Worst core (3,2) c=2 has +89 us
  residual at 12x10 and -13 us on the same physical core at 11x10.

## What a better deal buys (sim437.py, mean max-core end over 30 views, traced)

| deal | 12x10 | gain | 11x10 | gain |
|---|---|---|---|---|
| strided (today, measured) | 1.898 | 0 | 1.891 | 0 |
| LPT, oracle weights from the same view | 1.776 | +0.123 | 1.861 | +0.030 |
| LPT, previous view's weights | 1.775 | +0.123 | 1.862 | +0.030 |
| LPT static, hero weights | 1.777 | +0.122 | 1.861 | +0.030 |
| **equal-count LPT static, hero weights** | **1.782** | **+0.116** | 1.855 | +0.037 |
| same, if the residual is captured too (device counts) | 1.676 | +0.223 | 1.791 | +0.101 |
| random static tile shuffle | 1.841 | +0.057 | 1.950 | -0.058 |
| snake (boustrophedon) strided | 1.968 | -0.069 | 1.927 | -0.036 |
| contiguous blocks | 2.486 | -0.588 | 2.374 | -0.483 |

Weight per tile = a + b * visible, with the fitted a, b. A static per-scene table is as good
as per-view weights (per-tile cost is view-stable across this orbit), so no per-frame readback
is needed. The equal-count variant keeps 49/50 tiles per core, so `pfwc_fuse::seg_base` and the
segment table stay as they are. Traced -> untraced: #427 put 0.14 traced at about 0.12
untraced, so +0.116 traced is about +0.10 untraced, and +0.22 about +0.19.

## Chunk frustum cull (#433)

#433 was shelved (chunk cull + Morton reorder: blend +0.72 ms/view, pfwc not faster), so the
finding stands on the current stack. If a chunk cull ever lands, per-tile fixed cost shrinks and
visible-count imbalance becomes a larger share of pfwc, so a weighted deal matters more, not
less.

## Verdict

Cause found: visible-count imbalance under the strided deal, writer-bound. Fix: a static
equal-count weighted tile->core table. Model: +0.10 untraced (CPU weights) to +0.19 (device
per-tile counts) at 12x10. That meets the 0.1 ms gate, but only just on the conservative model,
and it carries two risks: the compaction (gid) order is core-major (each core writes its own
segment), so changing which core owns a tile changes the gid order and the md5, and #433 showed
gid reordering can slow blend. Proposed as one device A/B with blend watched, not as a sure win.

## Proposed A/B (follow-up task)

- Host: compute a per-scene tile->core table once at scene load. Weights from device per-tile
  visible counts of the hero view (vc per tile is already computed in `writer_pfwc_split.cpp`
  classify_tile; dump it once behind an env, or derive it from the compacted output). Equal-count
  LPT: sort tiles by weight, deal in rounds of C, each to the least-loaded core; remainder
  tiles to the cores that get the extra tile today.
- Kernels: `reader_pfwc.cpp` / `writer_pfwc_split.cpp` read `t` from a per-core tile list (<=50
  u16 entries, runtime args or an L1 table) instead of `chunk_start + k * stride`, behind
  `GSPLAT_TT_PFWC_DEAL=lpt` (default off; =0 keeps strided).
- Run on p150 at ETH 12x10, 30 views, both arms on the same box, untraced avg_frame_ms plus one
  traced run for per-program pfwc and blend windows. Gate: total -0.1 ms/view untraced, and blend
  not slower by more than the pfwc gain.
- md5 will change (gid order): create new per-grid goldens with `opt/md5_golden.py` for the new
  arm and check the strided arm still gives 39d84b28.
- Screenshot: render the bicycle hero view on device at the candidate commit and config, save
  hero.png, diff image and PSNR against `benchmarks/reference_v2/hero.png` (label that reference),
  and look at the image for tile artifacts.
- Restrictions for that task: own worktree of ~/dev/gstt2 on a ttp/* branch; device work wrapped
  in `ttp lock p100 -- ...`; no ird reserve/release unless it is the reserving task; bh-30 only
  under the viewer rules; ssh preflight first; never disable host-key checking.

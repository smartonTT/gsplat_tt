# Lever 2: visibility predicate + tile rectangle on the SFPU in pfwc (task #99)

**Measured and adopted in task #102 (iter-176): 41.42 -> 32.61 ms/view, 30.7 FPS, all views
md5-identical; see [RESULTS.md](RESULTS.md). `GSPLAT_TT_SFPU_VIS` now defaults to 1; `=0` is
the kill switch.** The text below is the task #99 prep write-up.

Code only, no device runs. Branch `ttp/t99-prep-lever-2-visibility-predicate-tile-a`, based on
the 41.40 ms/view tip 08f9200. Everything is behind `GSPLAT_TT_SFPU_VIS` (default 0 at the time =
the legacy path, unchanged), so one build A/Bs both paths. The kernels are **not compiled yet**:
there is no tt-metal / sfpi toolchain on the Mac. They pass a syntax-only compile against
stub headers (below). The first device run will probably surface some sfpi API details.

## What changes (GSPLAT_TT_SFPU_VIS=1)

| today (programs per view) | lever 2 |
|---|---|
| pfwc (SFPU) | pfwc + step 11.5: predicate + K1 rectangle on the SFPU, 2 more word tiles; writer builds the mask + per-tile counts |
| proj_count (soft-float predicate on 220 movers) | **gone** |
| gather_scan_bases | gather_vis_scan: balanced cut, slot bases, M, P, ta_pairs_P, offs[M] |
| proj_scatter (12 SoA/AoS streams, equal tile counts per mover) | gather_vis_scatter: depth + blend record + offs + rectangle, cut by cost |
| ta_gauss_aabb (K1) + scan1 + scan_bases + Finish + P read + scan2 | **gone** (P comes with the project stage's M read) |
| ta_bucket_scatter (K2, soft-float rectangle per gaussian) | K2 with `TA_K2_AABB`: reads offs + the packed rectangle |

- **pfwc** (`project_pfwc_compute.cpp`, `PFWC_VIS`): mean_x/mean_y/rx/ry are also packed to
  scratch CBs; the opacity is a 10th reader stream. `pfwc_vis_one<V>` evaluates
  `gather_pred::visible_bits` and the `tile_assign_bbox.cpp` rectangle per lane and writes two
  fp32-safe words (`render/kernels/dataflow/vis_tile.h`): tpg word (0, TAG|w*h, or RECHECK) and
  aabb word (TAG | min_x | min_y<<10 | (w-1)<<20). Params are staged once per tile in DEST slot
  6; intermediates reuse consumed input rows (sfpi cannot spill vector registers).
- **writer_pfwc_vis.cpp**: per tile resolves RECHECK words with the exact soft-float code, builds
  the proj_count mask layout (128 B per tile) and counts [visible, pairs]; writes the counts of its
  tile range to a dense 1 KB-page buffer at the end. It runs in the writer's idle time.
- **gather_vis_scan.cpp** (1 core): reads the counts, cuts the legacy compaction sequence (strided
  tiles, core-major) into 220 slot ranges at equal cost (`visible + 24 * nonempty + 1`, env
  tunable). The compact order is the sequence order, so **outputs do not depend on the cut**.
  Publishes proj_M = [M, P], ta_pairs_P (same layout as tile_assign_scan_bases), offs[M] = P.
- **gather_vis_scatter.cpp**: writes only what the default chain reads (proj_m_depth for sort, the
  64 B blend record) plus proj_m_offs and proj_m_aabb for K2; px/py/rx/ry/a/b/c/op/colors are dead
  in the default path (blend `(void)`s them, K1 is gone, the TA cull is off). Pages are released
  with `noc_async_writes_flushed` (one barrier at the end).
- **tile_assign**: K1, the three scans, the scan Finish and the P read are skipped; K2 is built with
  `TA_K2_AABB`.

## Bit-exactness

- Predicate: every test is the sign/zero of one rounded fp32 sum or difference, which has the sign
  of the exact value, so it is exact for finite inputs under any faithful rounding
  (`fl(mx - rx)` near the image edge is exact: rx is an integer, |mx| < 2^24 there).
  inf/NaN inputs (any of tz, op, mx, my, rx, ry) give RECHECK, resolved by the writer.
- Rectangle: `floor(clamp(q))` with the 2^23 trick + fix-up equals the K1 `(int)` + clamp for any
  faithful rounding. The only rounding that matters is `fl(m + r)` for the right/bottom edge; lanes
  whose `q = fl(m + r) / 32` is within tau = 2^-12 of an integer are also sent to RECHECK, so the
  result is exact even if SFPMAD is not nearest-even (0.15% of visible gaussians on a natural random
  scene, ~2-3k per view, ~25 per writer).
- `tests/unit/test_vis_lever2.cpp` models `pfwc_vis_one` lane by lane (in the kernel's order) and
  checks, on 1.4 M gaussians with planted ties at tile borders, image edges, k_near, min_opacity,
  max_radius, +-0, inf and NaN: every word equals the soft-float reference, with a nearest-even
  adder and with a round-toward-zero adder (0 mismatches; with tau = 0 the RTZ adder breaks
  ~600 words, so the test is sensitive). It also runs the whole new pipeline (writer masks/counts ->
  scan -> scatter -> K2 on the packed rectangle) against the legacy one (proj_count slot order, K1,
  exclusive scan, K2) for 4 core counts x 3 cut settings x 40 scenes: M, P, compact order,
  offs[0..M] and every (gid, tid) pair identical, each compact slot written once (480 pipelines).
- On device, `GSPLAT_TT_SFPU_VIS=2` adds a per-view cross-check: the legacy count pass (mask, M),
  the legacy K1 + scans (offs[0..M]) and a host K1 of every packed rectangle; it prints
  `[VIS-CHECK] gather ... OK|MISMATCH` and `[VIS-CHECK] tile_assign ... OK|MISMATCH`.

## Expected saving

Stage numbers (untraced, yyzo-bh-07 p100a): gather count 2.75 + scatter 3.78 ms (#19/#75);
tile_assign scan bucket 3.08 (K1 + scan1 + scan_bases + Finish) and k2 bucket 4.36 (scan2 + K2,
#19); proj_scatter makespan 4.15 vs per-core mean 2.59 (#83 Tracy).

| item | ms/view |
|---|---:|
| proj_count removed | -2.75 |
| K1 + scan1 + scan_bases + Finish removed | -3.08 |
| scan2 + TA P read removed | -0.4 to -0.5 |
| scatter balanced and slimmer (12 -> ~4 streams per element) | -1.0 to -1.5 |
| K2 reads 1 attribute stream, no soft-float rectangle | -0.1 to -0.3 |
| pfwc: +1 input stream, +4 packs, +6 unpacks, SFPU step, +2 output tiles | +0.4 to +0.6 |
| gather_vis_scan vs gather_scan_bases | +0.05 |
| **net** | **-5.5 to -7** |

Expected: 41.40 -> ~34.5-36 ms/view (~28-29 FPS). Upper bound if pfwc grows only 0.3 ms: ~-7.5.

## Device measurement spec (follow-up task)

Board yyzo-bh-07 p100a (not a p150), bicycle, 30 views, the existing reservation; never
reserve/release; every sync + run inside one `ttp lock p100 -- ...`.

1. Sync both trees (base 08f9200, candidate = this branch's tip):
   `ttp lock p100 -- opt/sync_remote.sh yyzo-bh-07 /localdev/smarton/gstt2-t99b 08f9200` and
   `ttp lock p100 -- opt/sync_remote.sh yyzo-bh-07 /localdev/smarton/gstt2-t99 <tip>`
   (check exit codes; 3 = stale .so). Copy `docs/lever2-t99/remote_job.sh` to
   `/localdev/smarton/t99_scripts/`.
2. Correctness first: `remote_job.sh 0 check legacy` -> expect `VIS-CHECK OK=60 MISMATCH=0`
   (30 gather + 30 tile_assign lines) and `ALL_VIEWS_IDENTICAL` vs md5-r82new.txt for both runs.
   Then `remote_job.sh 0 vis` -> `ALL_VIEWS_IDENTICAL`; hero_vs_ref 100 dB from `run.py` (ref run).
3. Timing: 3 interleaved rounds `remote_job.sh <r> base vis` (r = 1..3). Report mean ms/view,
   FPS, and the project / tile_assign stage buckets per round. Accept if all 90 candidate views are
   md5-identical and the mean gain >= 3 ms.
4. Attribution (one round each): `vis_nobal` (balance share), `vis_w8`, `vis_w64` (cost model),
   `vis_notau` (if md5-identical, SFPMAD is nearest-even and tau could be dropped).
   One Tracy capture of `vis` (docs/gaps-t85/remote_tracy.sh pattern): proj_count, ta_gauss_aabb and
   the TA scans gone; proj_scatter makespan close to its per-core mean; pfwc window growth.
5. If it wins: make `GSPLAT_TT_SFPU_VIS=1` the default (keep `=0` as kill switch), ledger row +
   `opt/REPORT.html` regeneration, push to smarton/tt-project-opt after the in-project review.

Likely first-run fixes (kernels never compiled):
- sfpi names in `pfwc_vis_one` / `pfwc_vis_cell` (`exexp`, `vInt >= int`, `vUInt << 20`,
  `reinterpret`, nested `v_if`); a register-allocation error means staging one more value in DEST.
- A hang points at the PFWC_VIS CB wiring (reader pushes CB 30; compute packs CB 31-36; writer
  pops 9-16, 35, 36) or at the scatter's slot page / mask reads.
- `[VIS-CHECK] gather` mismatch -> predicate compare semantics; `tile_assign` offs mismatch with
  gather OK -> scan/scatter offs; aabb-only mismatch -> rounding (raise tau) or the packing.

## Local checks run

- `tests/unit/test_vis_lever2.cpp` (new): 0 mismatches (command in the file header).
- Existing: test_gather_visible_mask PASS, test_gather_dual_mover 0/2400, test_tile_assign_dual_mover
  0/1353; `pytest tests/spec`: 36 passed, 2 skipped, 2 xfailed (401 s, Mac).
- Syntax-only compiles against stub tt-metal / sfpi headers (`tests/syntax_stub/check.sh`):
  the new/changed dataflow kernels (both variants), the compute kernel with and without
  `PFWC_VIS`, and the four host files.

## Files

New: `render/kernels/dataflow/{vis_tile.h, writer_pfwc_vis.cpp, gather_vis_scan.cpp,
gather_vis_scatter.cpp}`, `render/host/vis_mode.h`, `tests/unit/test_vis_lever2.cpp`,
`tests/syntax_stub/`, `docs/lever2-t99/`.
Changed: `project_pfwc_compute.cpp`, `reader_pfwc.cpp`, `tile_assign_scatter.cpp` (all `#ifdef`),
`pfwc.h`, `pfwc_device.cpp`, `gather_visible.h`, `gather_visible_device.cpp`,
`tile_assign_device.cpp`, `render.cpp`.

Follow-on (lever 3 in #93): with every visible gaussian carrying its packed rectangle and pair
offset, the sort emit can enumerate pairs itself and K2 / the gid-tid list disappear.

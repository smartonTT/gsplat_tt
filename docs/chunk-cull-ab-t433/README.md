# t433: chunk frustum cull before pfwc on the iter-215 stack — device A/B (shelved)

Branch `ttp/t433-chunk-frustum-cull-before-pfwc-on-xvpin-`, code at 9d7c340b (port of #169 onto iter 215,
behind `GSPLAT_TT_CHUNK_CULL`, off by default), driver at be4b6f91 (`drive.sh`, `remote_tracy.sh`).
Box: yyzo-bh-04 (p100a), bicycle, 1024x1024, 30 views, 3 alternated rounds of off / ro / cull, same build
(cpp #124, bin 971ba4aa2e19a355). Raw output in `out/`.

Arms:
- **off**: default (no reorder, no cull).
- **ro**: `GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0` — Morton reorder of the scene and tile list, no tiles skipped.
- **cull**: `GSPLAT_TT_CHUNK_CULL=1` — reorder plus skip of off-frustum 1024-gaussian tiles (3513-3704 of 5989 kept on the logged views, ~38-41% culled).

## Result: avg_frame_ms (device-measured)

| round | off | ro | cull | ro-off | cull-off |
|---|---|---|---|---|---|
| 1 | 8.262 | 8.977 | 8.952 | +0.715 | +0.690 |
| 2 | 8.255 | 8.986 | 8.954 | +0.731 | +0.699 |
| 3 | 8.286 | 9.007 | 9.026 | +0.721 | +0.740 |

Stage A gate (reorder cost > 0.3 ms/view -> shelve): **failed**, the reorder alone costs +0.72 ms/view.
Stage B gate (cull >= 0.15 ms/view faster every round): **failed**, cull is +0.69..+0.74 ms/view slower.
Verdict: **shelved**. Nothing lands; no iteration recorded in iters.jsonl.

## Where the time goes (stage timers, ms/view, round means)

| stage | off | ro | cull |
|---|---|---|---|
| project (pfwc) | 1.096 | 1.139 | 1.129 |
| &nbsp; gather_wait | 1.090 | 1.133 | 1.119 |
| &nbsp; pfwc_setup (host) | 0.003 | 0.020 | 0.117 |
| sort | 0.479 | 0.471 | 0.483 |
| &nbsp; bin_emit | 0.191 | 0.185 | 0.191 |
| blend (mat+blend) | 6.578 | 7.242 | 7.130 |
| xview | 0.057 | 0.077 | 0.172 |

- The whole loss is in **blend** (+0.66 ms ro, +0.55 ms cull). The Morton gid order slows the
  mat/blend pass; sort and bin_emit do not move (the +0.11 bin_emit seen in #169 does not show up here).
- **pfwc does not get faster with the cull**: gather_wait is 1.12 ms (cull) vs 1.13 (ro) vs 1.09 (off),
  although ~39% of tiles are skipped. #429's model assumed pfwc time scales with the tiles processed;
  on this stack it does not, so the modelled ~0.37 ms saving is not there even before the blend loss.
  The host tile-list build adds ~0.1 ms (pfwc_setup) and the xview stage grows by ~0.1 ms.
- Why blend slows under the reorder is not resolved here. Likely candidates: the per-tile gaussian
  record gather in blend loses DRAM locality with Morton gids (training order clusters gaussians that
  land in the same screen tile), or the reorder changes how depth ties fall per tile. Neither was tested.

Tracy of the cull arm (views 0:10, `out/dev-cull-c0.csv.gz`, zone-overhead-inflated, cull arm only, no
off-arm capture to compare): TRISC_1 pfwc mean 1408 us/view-core, tile_blend_sfpu 3723 us,
NCRISC tile_blend_load 3648 us, sort_ol_emit 1758 us, mat_cull_mask 1845 us.

## Correctness

md5 of the 30-view sweep, identical in all 3 rounds per arm:
- off `906e0435` (= current 11x10 golden), ro `dc5e3171`, cull `5f542ec0` (both differ from golden on
  30/30 views, as expected from the gid tie-break change; deterministic).
- Device hero (cull arm, `opt/metal-screenshots/t433/hero.png`, diff `hero_diff10.png`): PSNR 42.51 dB
  vs `benchmarks/reference_v2/hero.png`, max 13 LSB vs the 8-bit golden. Looked at both images: no tile
  seams or block artifacts; the diff shows only the usual edge/spoke structure.

## If revisited

Only worth it if (1) pfwc becomes truly per-tile-bound (it is not today) and (2) the cull can be done
without renumbering gaussians (e.g. a per-view tile skip list over the original order with spatially
coherent chunks built by an index indirection only in pfwc), so blend keeps today's record order.

# Lever 2 device results (task #102, iter-176)

Board yyzo-bh-07 (Blackhole p100a, not a p150), bicycle, 30 views at 1024x1024, untraced.
Base 08f9200 (41.40 ms/view tip). Candidate: the t99 branch plus the t102 build fixes, rebased
onto abc2840 as 40c629d. Logs are in `logs/`.

## Build fixes needed on device

- The VIS-CHECK `rd` lambdas take the `shared_ptr` by value, because `EnqueueReadMeshBuffer`
  needs a non-const reference.
- The PFWC_VIS program came to 83 KB, over the 70.6 KB kernel config buffer (trisc1 was 44 KB).
  `pfwc_vis_one<V>` is now instantiated for 16 vectors and runs twice per tile. Between the two
  runs the DEST counter moves on by 16 vectors in 4-row `INCRWC` steps (the field only covers
  [-8, 7] rows). trisc1 is now 29.8 KB. The program totals about 66 KB (writer brisc 15.6 KB,
  trisc0 10.3, trisc1 29.8, trisc2 6.0, reader 4.4), which leaves only ~4 KB of headroom.
- A host check: every PFWC_VIS core's count pages must fit the 16 KB CB_VCNT staging.

## Correctness

- `GSPLAT_TT_SFPU_VIS=2` (check mode) at ec1d90c: [VIS-CHECK] OK=62, MISMATCH=0 over all 30
  views, and ALL_VIEWS_IDENTICAL. This compares the mask/M against proj_count, offs against
  K1 + scans, and the K1 rectangle against the host.
- All 90 timed candidate views and every variant were md5-identical to the reference. hero_vs_ref
  was 100 dB.

## Timing (view_total ms/view)

| round | base 08f9200 | vis (=1) |
|---|---:|---:|
| 1 | 41.374 | 32.584 |
| 2 | 41.507 | 32.634 |
| 3 | 41.385 | 32.624 |
| **mean** | **41.42** | **32.61** (-8.81, -21.3%, 30.7 FPS) |

Verification runs: at 9284057 the default env gave 32.58 and `=0` gave 41.48. At the rebased tip
40c629d the default env gave 32.62. All were md5-identical.

| stage | base | vis |
|---|---:|---:|
| project | 8.69 | 6.02 (pfwc_finish 2.00 -> 2.48, gather_wait 6.45 -> 3.33) |
| tile_assign | 7.57 | 1.44 (scan_finish 3.09 + k2 4.36 -> k2 1.40) |
| sort | 14.24 | 14.23 |
| blend | 10.65 | 10.67 |

## Variants (one round each, all md5-identical)

| variant | ms/view |
|---|---:|
| vis_nobal (legacy per-core halves) | 32.72 |
| vis_w8 (tile weight 8) | 32.56 |
| vis_w64 (tile weight 64) | 32.57 |
| vis_notau (edge tau off) | 32.62 |

The balanced cut and its cost model are worth at most 0.1 ms. With tau off, the md5s are
unchanged and the time is the same, so SFPMAD rounds to nearest-even on these views. Tau stays on
as insurance because it costs nothing.

## Tracy (views 0-9, vis)

proj_count, ta_gauss_aabb and the TA scans no longer appear. Traced ms/view by program:

| program | ms/view |
|---|---:|
| pfwc | 2.46 |
| proj_vis_scan + proj_scatter | 3.32 (scan 0.40, on a single core) |
| ta_bucket_scatter | 1.40 |
| sort_bin_hist | 0.87 |
| sort_bucket_emit | 7.97 |
| sort_tile_depth | 2.52 |
| subchunk_mat + cull | 3.17 |
| blend | 7.52 |

Inter-program idle is 2.40 ms/view: proj -> TA 0.82, sort_bin_hist -> emit 1.34.

proj_scatter is unbalanced by location, not by work. The per-RISC mean is 1.96 ms against a max
of 2.92 ms (balance 0.67), and the slowest RISCs sit together around cores (2-4, 2-3). That points
to NoC/DRAM contention, which fits the balance variants having no effect.

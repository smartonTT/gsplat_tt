# Task #198: 2-CQ bridge hiding (early sort_ol + CQ1 sort->mat bridge)

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024, untraced unless noted.
Model and plan: docs/two-cq-model-t195/README.md section 3.

## What changed

- `GSPLAT_TT_SORT_OL_EARLY` (default 1; 0 = off, which also turns off `GSPLAT_TT_MAT_CQ1`).
  The one-launch sort is enqueued on CQ0 right after the fused K2. It no longer waits for the host.
  - The kernel reads P from the `ta_pairs_P` page (runtime arg 11 = 0xFFFFFFFF).
  - Its mover ranges come from `pfwc_fuse::k2_range_speed`, the same split K2 used.
  - The host P read (`pread`) and the M-float depths vector are gone; `proj.num_visible = M`.
- `GSPLAT_TT_MAT_CQ1` (default 1; 0 = off). The device opens 2 command queues. CQ1:
  1. waits on an event recorded after K2;
  2. reads projM and the K2 count rows;
  3. the host computes the per-tile totals (`sort_onelaunch::totals_from_k2_rows`);
  4. bin_layout, publish and the mat uploads run on CQ1, then `Finish(CQ1)`;
  5. mat and blend are enqueued on CQ0 behind the sort.

  All of this host work hides under the device sort. The pair-overflow hard fail is kept (`mread[2] != 0` falls back).
- Code: `render/host/{sort_device,tile_assign_device,device_state,gather_visible_device}.cpp`,
  `render/kernels/dataflow/sort_bin_onelaunch.cpp`, `render/host/env_config.h`; unit test
  `tests/unit/test_two_cq_t198.cpp`.

## Why the first device chain produced no smoke md5 (run 538)

There was no crash or hang. devrun refuses any `--timeout` above its 600 s reservation ceiling. The chain asked for
800 s and 1100 s, so no device run happened, and the md5 gate then found no files. Fix (e7d25e0): smoke 400 s,
rounds 300 s, Tracy 540 s, plus a per-run `timeout` (`RUN_TO`) inside `remote_time.sh`.

## Results

md5: all 35 runs (smoke with `GSPLAT_TT_SORT_ONELAUNCH_CHECK=1`, every timing arm, all three bases)
are identical to md5-r82new.txt (46a725ab).

### Project tip (d78a457, includes #196 bulk blendrec reads), env arms, 4 paired rounds (`out/`)

| arm   | view_total | project | sort  | bin_emit | pread | blend  |
|-------|-----------:|--------:|------:|---------:|------:|-------:|
| off   | 15.680     | 4.096   | 2.818 | 2.477    | 0.021 | 8.545  |
| early | 15.477     | 3.986   | 2.722 | 2.410    | 0.001 | 8.547  |
| both  | 15.216     | 3.994   | 0.875 | 0.563    | 0.001 | 10.090 |

Paired view_total, ms/view:
- early - off: -0.165 -0.068 -0.279 -0.297, mean -0.202
- both - off: -0.462 -0.368 -0.477 -0.548, mean **-0.464** (sd 0.074)

Blend grows because blend's host wait now also covers the sort time that used to be spent in the sort stage.
The host stage split moves around; only view_total counts.

### Defaults on (64d951e) vs `GSPLAT_TT_SORT_OL_EARLY=0`, 3 paired rounds (`out-tip/`)

| arm           | view_total | project | sort  | blend  |
|---------------|-----------:|--------:|------:|-------:|
| off (=0)      | 15.639     | 4.097   | 2.782 | 8.542  |
| on (defaults) | 15.221     | 3.994   | 0.866 | 10.094 |

Paired: -0.420 -0.431 -0.404, mean **-0.418** ms/view (sd 0.014). 15.22 ms/view = 65.7 FPS.

### Tracy, 30 views, same build (`out-tip/tracy-{toff,tip}-*`)

| gap (ms/view)          | off   | on    |
|------------------------|------:|------:|
| pfwc -> K2             | 0.006 | 0.006 |
| K2 -> sort_ol          | 0.450 | 0.010 |
| sort_ol -> mat         | 0.260 | 0.007 |
| mat -> blend           | 0.006 | 0.006 |
| all-core idle per view | 0.723 | 0.029 |

Device busy per program is unchanged (pfwc 2.981, K2 0.985, sort_ol 2.397 vs 2.396, mat 3.009, blend 5.54).
The traced device span per view drops from 15.627 to 14.943 ms.

### Old base (t194 tip 4d591f7, before #196), 4 paired rounds (`out-t194base/`)

off/early/both 16.354 / 16.289 / 16.088 ms/view; both - off mean only -0.266 ms/view.
A same-build Tracy (`out-t194diag/`) found the cause: sort_ol was slower when started early. sort_ol busy was
2.973 off, 3.101 early, 3.299 both. The per-mover emit diagnostic (`movers_diag.py`) put the extra time on the
row y=2 BRISC movers, whose per-gaussian 64 B blendrec reads were request-bound. Why an early start slowed
those reads was not isolated (concurrent host reads and uploads are a candidate, untested).
#196 replaced those reads with one bulk read per DRAM bank. On the tip the slowdown is gone:
- emit makespan 2287.4 off vs 2287.5 on;
- mover medians 2124.8 vs 2127.0 us;
- the same movers finish last in both arms (`out-tip/movers-diag.txt`).

## Decision

LANDED with both switches on by default. The paired gain (-0.418 to -0.464 ms/view) clears the 0.3 ms gate,
and every run is md5-identical.

Commits: 8889862 (implementation, default off), 2a07e9f / fb69198 / e7d25e0 (drivers, summary, timeout fix),
87dadab (old-base results and diagnosis), 64d951e (defaults on), plus the docs/ledger commit.

## Left over

- Untraced view_total (15.22) is still about 0.28 ms above the traced device span (14.94). That gap is the
  per-view image readback plus next-view setup on the host. Overlapping the readback with the next view's
  pfwc (for example on CQ1) is the next host-gap lever.
- The `kMoverSpeedP150` mover speed table was calibrated before #196. #196 narrowed the per-mover rate spread to
  0.96-1.06, so the table may now skew the ranges. A re-calibration check is cheap.

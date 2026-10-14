# Device zone profiler on the p150 with ETH dispatch (task #731)

## Problem
With `TT_METAL_DEVICE_PROFILER=1`, `render/run.py` on the p150a (bh-49, ETH dispatch 12x10)
died before rendering, in `pfwc_num_cores`:

    idle_erisc/idle_erisc.elf: segment[0] [0x6e90,+0x4668) overflows region:0 limit of 0x4490 bytes

(docs/t727/out/run-r5-prof.log on ttp/t727 @0efacafc). The profiler instruments the
cq_prefetch kernel that runs on an idle ETH core. Instrumented, it is 0x4668 B. The default
ETH overlay (`opt/eth/make_overlay.sh`, 32 KB idle-ERISC link bound) leaves 0x4490 B, which is
0x1d8 B short. #427 had already found the fix (`ETH_IERISC_KB=36`, a profiler-only overlay),
but only bench427.sh used it by hand, so every later profiled run on a p150 hit the overflow.

## Fix (recipe)
`render/eth_default.py` now does this by itself. When `TT_METAL_DEVICE_PROFILER` is set (not
empty, not `0`), `setup()`:
- uses the overlay `<GSPLAT_TT_ETH_OVERLAY or <repo>/tmp/ttm-eth12>p36`, made by
  `ETH_IERISC_KB=36 opt/eth/make_overlay.sh` (marker line `ierisc_kb=36`), and reuses it when
  the marker matches;
- uses its own JIT cache, `<GSPLAT_TT_ETH_CACHE>p36` or `<overlay>-cache`, so ELFs linked with
  the 36 KB bound never mix with the default cache;
- replaces a 32 KB overlay that `opt/eth/env.sh` put in `TT_METAL_RUNTIME_ROOT` (it logs that),
  and keeps an env overlay that already has a bound of 36 KB or more.
Without the profiler nothing changes: same 32 KB overlay, same cache, same ELFs.

So a profiled run needs no overlay setting. Verified command (bh-49, p150a, 12x10 ETH dispatch):

    GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_BLEND_PROF=1 TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 \
    TT_METAL_PROFILER_DIR=<out dir> TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 \
    GSPLAT_TT_PROFILE_READ_EVERY=4 \
      python3 render/run.py --no-ref --cameras benchmarks/cameras_v2.json \
        --back-to-back --view-range 0:4 --b2b-passes 2 --b2b-warmup 0

`TT_METAL_PROFILER_MID_RUN_DUMP=1` is needed: a plain `run.py` without it wrote no device CSV
(the `tracy` wrapper would do it too, but it needs `loguru`, which the bh-49 venv lacks).
`GSPLAT_TT_PROFILE_READ_EVERY=4` reads the profiler buffers every 4 renders so they do not
fill up. The device CSV is `profile_log_device.csv` under `<out dir>`. Aggregate it with
`python3 opt/profiler/analyze_zones.py <csv> <n_renders>` (here 12: 2 timed passes of 4 views
plus the 4-view check pass).

## Result (b3e8f99a, 2026-10-14 08:40 PDT, bh-49)
The run finished, rc 0, with the 36 KB overlay (`[eth] ... overlay .../tmp/ttm-eth12p36`,
marker `ierisc_kb=36`) and no overflow. Device CSV: 852,134 rows, 11,520 `tile_blend_*` zone
rows. B2B under the profiler: 50.38 ms/frame (the profiler adds a lot of overhead; this is
not a speed number). Sample from docs/t731/out/prof-zones.txt (`per_view` = sum over cores,
`pv_makespan` = longest core per view, ms):

    zone                      total_ms  per_view  maxcore_ms pv_makespan    pairs  riscs
    tile_blend_sfpu            28308.9  2359.076        79.3       6.611     4320  TRISC_0,TRISC_1,TRISC_2
    tile_blend_load             9262.9   771.912        77.9       6.492     1440  NCRISC
    rd_bulk_wait                9027.7   752.310        77.1       6.429    26906  NCRISC
    wr_wait_out                 8639.3   719.940        76.3       6.357    12288  BRISC
    cmp_emit                    7139.1   594.926        77.1       6.422    36864  TRISC_0,TRISC_1,TRISC_2
    pfwc                        5854.2   487.848        17.3       1.440     4680  TRISC_0,TRISC_1,TRISC_2
    mat_cull_mask               5160.9   430.072        24.1       2.006     4320  TRISC_0,TRISC_1,TRISC_2

Full table, run log and sync log: docs/t731/out/.

Other ways out, not taken: worker dispatch (`GSPLAT_TT_DISPATCH=worker`) also avoids the ETH
kernel, but it changes the compute grid (11x10 instead of 12x10), so the zones would not match
the bench. The 36 KB bound does not change render bits: #427 showed the same 30-view md5
(39d84b28) through the 32 KB and the 36 KB overlay, untraced.

Driver: `ttp detach t731-prof -- docs/t731/drive.sh <rev>` (one `ttp lock p100`: preflight,
repo-scoped sync + build, `docs/t731/remote.sh prof`, CSV pulled back to docs/t731/out).
Tests: tests/test_eth_default.py (the two profiler tests fail on the old eth_default.py).

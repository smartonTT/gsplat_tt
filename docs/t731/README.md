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

So a profiled run needs no extra setting:

    GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_BLEND_PROF=1 TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 \
      python3 render/run.py --no-ref --cameras benchmarks/cameras_v2.json --back-to-back --view-range 0:4

The device CSV lands in `$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`
(the overlay's `generated/` links there). Aggregate it with
`python3 opt/profiler/analyze_zones.py <csv> <n_views>`.

Other ways out, not taken: worker dispatch (`GSPLAT_TT_DISPATCH=worker`) also avoids the ETH
kernel, but it changes the compute grid (11x10 instead of 12x10), so the zones would not match
the bench. The 36 KB bound does not change render bits: #427 showed the same 30-view md5
(39d84b28) through the 32 KB and the 36 KB overlay, untraced.

Driver: `ttp detach t731-prof -- docs/t731/drive.sh <rev>` (one `ttp lock p100`: preflight,
repo-scoped sync + build, `docs/t731/remote.sh prof`, CSV pulled back to docs/t731/out).
Tests: tests/test_eth_default.py (the two profiler tests fail on the old eth_default.py).

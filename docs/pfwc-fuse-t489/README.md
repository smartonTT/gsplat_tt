# Task #489: fewer pfwc compute acquires (GSPLAT_TT_PFWC_ACQ_FUSE)

Lever: `pfwc_fuse_sfpu.h`. Bit 0 runs pfwc steps 1-5 (transform, 1/tz, depth, means) in one
DEST acquire instead of seven. Bit 1 runs cov_cam and S_AC in one acquire. Same rounding order,
bit-identical. Bundled with GSPLAT_TT_PFWC_SKIP_RGB=1 (#481). Both are now default on
(`render/host/env_config.h`); `GSPLAT_TT_PFWC_ACQ_FUSE=0 GSPLAT_TT_PFWC_SKIP_RGB=0` restores the old path.

## A/B (p100a, 11x10 grid, bicycle 30 views, commit d866a6c, 2026-10-10)

SKIP_RGB=1 in all arms. One warm-up, then 3 alternating rounds (r0 warm-up shown too).
`render/run.py --back-to-back`, 20 passes per run, median ms/frame:

| round | off (ACQ_FUSE=0) | on (=3) | p (=1, bit 0 only) |
|---|---|---|---|
| r0 (warm) | 9.353 | 9.236 | 9.259 |
| r1 | 9.346 | 9.240 | 9.258 |
| r2 | 9.347 | 9.241 | 9.256 |
| r3 | 9.348 | 9.241 | 9.260 |
| **median r1-r3** | **9.347** | **9.241 (-0.106)** | 9.258 (-0.089) |

B2B_STAGES project (pfwc gather wait): 2.36 -> 2.25 ms. Latency mode (secondary, no dump):
p50 9.4 -> 9.3 ms. Bit 0 gives most of the gain (-0.089); bit 1 adds -0.017.

Gate (>= 0.1 ms/view b2b): passed, narrowly. Relative to the old default (SKIP_RGB=0), add
#481's ~0.058 ms (measured separately, docs in branch ttp/t481).

## Correctness

- raw_md5 5a438f5a identical in every b2b run of every arm.
- Dump runs: `MD5_GOLDEN_OK grid 11x10: list 906e0435 = golden (30/30 views)` for off, on and p.
- Device hero (`out/hero-on.png`, ACQ_FUSE=3) vs benchmarks/reference_v2/hero.png: 42.51 dB.
  Diff x10 in `out/hero_diff10.png`: looked at, edge noise only (spokes, foliage), no tile seams.

Driver: `drive.sh` / `remote_ab.sh`; logs in `out/`.

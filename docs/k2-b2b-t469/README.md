# t469: b2b A/B of K2 on idle TRISCs (#274)

Question: keep `GSPLAT_TT_K2_TRISC=1` (#274) under the back-to-back headline metric? #274 had
measured -0.245 ms/view in latency mode and was shelved only for missing the old 0.3 ms gate
(#468 triage).

Setup: yyzo-bh-04 (Blackhole p100a, existing measurement reservation, worker dispatch 11x10),
tree = opt tip c7232e14 + no-ff merge of t274 (a9dd06e), one build (8db9efa9), bicycle 30 views
1024x1024. `drive.sh` holds `ttp lock p100` for sync + build + a discarded warm-up (r0) + 3
alternating rounds (off/on, on/off, off/on) of `render/run.py --no-ref --back-to-back` (1 check
pass + 3 timed passes) and latency (`--no-ref`, no `--dump-views`), then one `--dump-views` pass
per arm checked with `opt/md5_golden.py`. Outputs in `out/`.

## Back-to-back ms/view (headline)

| round | off (=0) passes | off | on (=1) passes | on | delta |
|---|---|---|---|---|---|
| r0 warm-up (not counted) | 9.862 9.868 9.862 | 9.864 | 9.567 9.572 9.569 | 9.569 | -0.295 |
| r1 | 9.867 9.854 9.858 | 9.860 | 9.556 9.562 9.555 | 9.557 | -0.303 |
| r2 | 9.874 9.850 9.862 | 9.862 | 9.558 9.558 9.554 | 9.557 | -0.305 |
| r3 | 9.874 9.877 9.861 | 9.871 | 9.559 9.563 9.556 | 9.559 | -0.312 |

Paired median delta (r1-r3): **-0.305 ms/view (-3.1%)**, 101.4 -> 104.6 FPS. Gate -0.15: pass.
All passes identical across passes (raw md5 5a438f5a in both arms).

Where: B2B_STAGES `project` (pfwc gather wait) 2.88 -> 2.58 ms; sort and blend unchanged. K2 runs
inside pfwc, which #464 showed is on the b2b critical path, so the full K2 saving shows in b2b.

## Latency (secondary)

avg_frame_ms off 9.8/9.8/9.8 vs on 9.5/9.5/9.5 (p50 10.0 vs 9.7).

## Image check

`--dump-views` md5 list 906e0435 = 11x10 golden on 30/30 views in both arms (MD5_GOLDEN_OK).
(39d84b28 is the 12x10 ETH golden for p150; this box runs the 11x10 worker grid.)

## Decision

Keep: default flipped to on in `render/host/tile_assign_device.cpp` (`=0` restores the mover-only
K2); `tests/unit/test_k2_trisc_default.py`. Measured on p100a only; not yet measured on p150 with
ETH dispatch (12x10).

## Re-A/B on the merged opt tip (iter 221 REC32 on)

The opt tip moved (iter 221: REC32 default on, also in pfwc), so the A/B was re-run on
ed3d3baf = opt 48d942a0 + t274 + default flip. Same box (yyzo-bh-04, p100a 11x10), one build
(so md5 c0933598), env unset (default, K2 on TRISCs) vs `GSPLAT_TT_K2_TRISC=0`. Outputs in
`out-merged/`.

| round | off (=0) passes | off | default (on) passes | on | delta |
|---|---|---|---|---|---|
| r0 warm-up (not counted) | 9.718 9.703 9.704 | 9.704 | 9.420 9.408 9.411 | 9.411 | -0.293 |
| r1 | 9.735 9.707 9.704 | 9.707 | 9.418 9.415 9.426 | 9.418 | -0.289 |
| r2 | 9.708 9.706 9.704 | 9.706 | 9.417 9.409 9.406 | 9.409 | -0.297 |
| r3 | 9.710 9.714 9.702 | 9.710 | 9.409 9.408 9.403 | 9.408 | -0.302 |

Paired median delta (r1-r3): **-0.297 ms/view (-3.1%)**, 103.0 -> 106.3 FPS. Median of the 9
measured passes 9.707 vs 9.409. B2B_STAGES per-pass medians: project (pfwc gather wait)
2.716 -> 2.417, sort 0.607 -> 0.596, blend 6.290 -> 6.305.

Latency (secondary, no dumps): off 9.669/9.605/9.647 vs on 9.376/9.373/9.353 ms/view.

Image: `--dump-views` md5 906e0435 = 11x10 golden on 30/30 views in both arms. Device hero
(default arm, dump pass): bit-exact vs `hero_golden_8bit.png`, PSNR vs
`benchmarks/reference_v2/hero.png` 42.51 dB. Looked at the hero and the x10 diff: no tile seams,
blocky or empty tiles, no color shift; diff only on thin edges (spokes, frame, bench slats,
foliage). Kept as iter 222.

# t266: accurate 1/z in pfwc (GSPLAT_TT_PFWC_RECIP_NEWTON)

**Verdict: free and a clear accuracy gain. Recommend default-on.** Cost is within noise
(-0.05 ms/view, gate was +0.1). Hero PSNR vs `benchmarks/reference_v2/hero.png`:
+1.35 dB at floor 1/255, +5.92 dB at 1/1024. Not tagged; the landing review decides.

## Root cause
pfwc step 2 computed 1/tz with `recip_tile()`, which in this tt-metal build is the
legacy_compat path (`_reciprocal_compat_<3>`). It is off by up to 1.48e-3 relative for z just
below a power of two: this is #260's 1.5e-3 residual. The knob replaces it with an SFPARECIP
seed plus 2 Newton steps (`render/kernels/compute/pfwc_recip_nr.h`). The conic determinant
reciprocal already used 2 Newton steps; its error came from 1/z feeding the Jacobian, so it is
unchanged in code but fixed by the 1/z fix (see table). `tests/unit/test_pfwc_recip.cpp` models
both paths: compat 1.48e-3 vs Newton 1.21e-7 max relative error.

## Results (yyzo-bh-07 p100a, commit 373f3f3, bicycle, cameras_v2, untraced, 30 views)

ms/view, mean of the 30 per-view times, arm order rotated per round:

| round | off | NEWTON=1 |
|---|---|---|
| r1 | 11.737 | 11.647 |
| r2 | 11.627 | 11.653 |
| r3 | 11.737 | 11.663 |
| mean | 11.700 | 11.654 (-0.046, noise) |
| floor 1/1024 (r1) | 12.560 | 12.580 |

Dump compare (hero, `GSPLAT_TT_DUMP_PROJ` + `docs/floor-ab-t260/compare_proj.py` vs cpu_cpp):

| field | off p99 | NEWTON p99 |
|---|---|---|
| means x (px) | 0.409 | 0.000122 |
| means y (px) | 0.216 | 0.000092 |
| conic a/b/c rel | 1.7e-3 / 3.0e-3 / 1.3e-3 | 1.0e-5 / 1.9e-5 / 0.9e-5 |
| means x, rows > 0.5 px | 8823 | 27 (row-match ambiguities, same in both) |

Acceptance (a) p99 < 0.05 px: met. Full json: `out/cmp-{off,nr}/proj_compare.json`.

PSNR of the device hero vs `benchmarks/reference_v2/hero.png` (identical in all 3 rounds):

| floor | off | NEWTON=1 | gain |
|---|---|---|---|
| 1/255 (default) | 41.163 | 42.512 | +1.35 dB |
| 1/1024 | 45.828 | 51.748 | +5.92 dB |

(The spec's expected gains of <=+0.2 / <=+0.9 dB were unmeasured estimates; measured is larger.)

## Golden badge (md5, separate from PSNR)
| config | hero md5 | 30-view sweep md5 |
|---|---|---|
| off, 1/255 | c4a040a5 | 46a725ab (= iter-199 golden) |
| NEWTON, 1/255 | 86524912 | **906e0435** (new golden if default-on) |
| off, 1/1024 | | 07efa52a |
| NEWTON, 1/1024 | | 166ba32d |
Stable over 3 rounds.

## Screenshot and visual check
- `img/hero_nr_255.png`, `img/hero_nr_1024.png`: device heroes, NEWTON=1.
- `img/diff10_vs_refv2_{nr,off}_{255,1024}.png`: |hero - reference_v2| x10.
- Looked at the full images and a full-res crop of the bike: no tile or microblock lines. The
  diff sits on content edges (spokes, frame), from the contribution floor. Seam ratios
  (`docs/floor-ab-t260/seams.py`, ~1.0 = none): nr_255 tile32 x 1.022 y 1.001, mb 1.005/1.008;
  nr_1024 tile32 1.019/1.001, mb 1.015/1.009. Same as off.

## Reproduce
`drive.sh` (Mac, under `ttp lock p100`) syncs, builds and runs `remote_time.sh` arms on the box.

# Task #260: contrib_floor accuracy A/B (bicycle, p100a)

yyzo-bh-07 (Blackhole p100a, not a p150), commit 9427a47 (iter-199 tip 46e9e17 plus a
default-off projection dump, `GSPLAT_TT_DUMP_PROJ`). Runtime knobs only; each arm uses a copy of
cameras_v2.json with only `contrib_floor` changed. 3 rotated untraced rounds of the 30-view
bench; ms/view = mean of the per-view walls. Hero rendered on the device; PSNR is 8-bit vs
`benchmarks/reference_v2/hero.png`. Iterations 200-204 (diagnostic) in opt/ttw/iters.jsonl.

| Arm | Config | ms/view | vs A | PSNR vs reference_v2 | sweep md5 |
|---|---|---|---|---|---|
| A | 1/255, pixel floor on (default) | 11.650 | - | 41.16 dB (golden match) | 46a725ab |
| B | 1/1024 | 12.630 | +0.98 | 45.83 dB | 07efa52a |
| C | 1/4096 | 13.522 | +1.87 | 46.71 dB | 27ae3e6d |
| D | 1/16384, PIXEL_FLOOR=0, SCHED=0 | 15.253 | +3.60 | 46.79 dB | 7beae7a2 |
| E | 1/16384, pixel floor on | 14.374 | +2.72 | 46.78 dB | 1b5426ce |

D as specified does not compile with the default BLEND_SCHED=2 (`#error` - the SCHED bodies
need the pixel floor), so D runs SCHED=0; E is the SCHED=2 equivalent.

Artifacts: no arm shows tile or microblock seams (`seams.py`: error steps across 32-px and
8x4 edges are within 3% of non-edge steps, same as the golden). A's diff has broad faint haze
(large low-alpha gaussians culled); B removes most of it; C/D/E leave only thin high-gradient
edges (spokes, frame, foliage).

Recommendation: keep A as the speed default. If an accurate mode is wanted, B (1/1024) buys
+4.7 dB for +0.98 ms; C adds only +0.9 dB for another +0.9 ms; D/E are dominated by C.
Changing the default is the user's call.

## Residual (~47 dB) diagnosis: projection

`compare_proj.py` maps the device pfwc records (dump of proj_m_blendrec/depth/aabb, first
frame = hero) to source ids (per-core segment + exact fp32 opacity/colour key) and compares
with the cpu_cpp project. Unambiguous rows (1,877,808; 2,149 duplicate keys dropped):
`out/proj_compare_unambig.json`.

- Opacity, colour: bit-exact. Depth: max rel 4.6e-7 (exact for ordering purposes).
- Means: mean abs err 0.025 / 0.013 px (x / y), p99 0.41 / 0.22 px, max 0.84 px. The worst
  rows sit at depths just below a power of two (1.9916, 3.997, 7.999, 15.995) and far off
  centre: a ~1.5e-3 relative error in the device 1/z (reciprocal approximation), scaled by
  the off-centre distance.
- Conic (stored as -a/2, -b, -c/2): p99 rel 1.7e-3, max rel 2e-2.
- Visible set: device 1,879,959 vs CPU 1,883,790 (CPU-only 5,992, device-only 10).
- Device aabb tile rect equals the CPU radius bbox for 90-96% of rows (the device rect is
  tighter, not a precision issue).
- Depth order within tiles: 1,844 adjacent swaps in 385 tiles (of 3.35 M pairs).

model.py (t251, variants dev = 1/255 floors, dev_pre156 = no floor) on three input sets
(`run_model.sh`, results in `out/model_*/model_psnr.json`; image-to-image PSNR computed from
the gitignored `model_images.npz`):

| inputs | rows | dev vs ref | dev vs golden | pre156 vs ref |
|---|---|---|---|---|
| cpu (cpu_cpp project, all CPU-visible) | 1,883,790 | 42.49 | 46.42 | 66.66 |
| cpu_devset (CPU values, matched device rows) | 1,877,798 | 37.46 | 37.98 | 39.10 |
| dev (device pfwc values, same rows) | 1,877,798 | 37.01 | 38.61 | 38.49 |

Same-row comparison, model image vs model image (8-bit PSNR):

| pair | floor 1/255 | no floor |
|---|---|---|
| dev vs cpu_devset (device projection numerics only) | 46.59 | 47.02 |
| cpu_devset vs cpu (row set only) | 38.73 | 39.09 |

Reading:
- With CPU inputs and no floor the model reproduces the reference (66.7 dB), and with the
  floor it lands at 42.5 dB vs ref (device: 41.16). The model is a faithful blend replay.
- Swapping in the device projection values on the same rows changes the image by 46.6-47.0
  dB: this matches the ~47 dB residual. **The residual is the device projection numerics**
  (mostly the ~1.5e-3 relative 1/z error moving off-centre means by up to 0.84 px, plus the
  conic error), not the blend.
- The row-set arm (38.7 dB) is NOT a device loss: the real device render is 41.16 dB vs ref,
  better than this. It mostly reflects the dump matching (2,149 rows with duplicate
  opacity/colour keys dropped, plus ~3.8 k real cull differences, M 1,879,959 vs 1,883,790,
  whose share cannot be separated with this dump). A follow-up would need the device to
  write the source id into the record.
- Fix direction: one Newton step on the device 1/z (and the conic inverse) in pfwc. Expected
  gain cannot exceed the ~47 dB term, i.e. ~+0.2 dB on the default (41.16) and
  ~+0.9 dB on the 1/1024 arm (45.83), at a small pfwc cost.

# Device golden vs reference_v2: where and why they differ (task #251)

Status: partial (run 676). Sections 1-2 are final. Sections 3-5 wait on the CPU
model (`docs/ref-golden-diverge-t251/model.py`; output in
`tt-project/state/runs/676/model/model_psnr.json`).

## 1. What made each image

- `benchmarks/reference_v2/hero.png` (md5 21a7fd8c...): the fp32 C++ CPU renderer
  (`backends/cpu_cpp`, `opt/cpu-vs-tt/render_cpu.py`), hero view, 1024x1024, fov 50,
  contrib_floor 1/16384, no per-pixel floor, T early-out per tile at 1e-4,
  output `(img*255).astype(uint8)` (truncate). First committed d6ab9cc (2026-05-28),
  re-vendored unchanged in 8f7c34f. Byte-identical to `opt/cpu-vs-tt/mac_hero.png`.
- `tests/fixtures/hero/hero_golden_8bit.png` (sweep md5 46a725ab): the device render
  since iter-156: contrib_floor 1/255 for tile cull AND a per-pixel floor
  (alpha < 1/255 -> 0, `BLEND_PIXEL_FLOOR`), T early-out per 8x4 microblock at
  T < 1/256 checked on a bf16 copy of T every 512 records, UNORM16 opacity/colour,
  21-bit exp, output fp32 -> bf16 tile -> floor(x*255).

## 2. Where they diverged

One step, not a slow pile-up. 8-bit PSNR vs reference_v2 (measured this task):

| image | PSNR | max diff | mean diff |
|---|---|---|---|
| current golden (iter-156 on, also t156/t174 archives) | 41.16 dB | 58 | -0.46 (darker) |
| pre-iter156 golden (floor 1/16384) | 46.79 dB | 52 | +0.12 |
| x86 cpu_cpp render | 82.36 dB | 4 | - |

The drop is iter-156 (task #41, commit 685f77b, 2026-09-30): floor 1/16384 -> 1/255
plus the per-pixel floor, -5.6 dB for -4.5 ms/view (107.75 -> 103.23 then). Task #41's
own numbers: mask-only 1/255 was 51.8-52.6 dB vs the old golden (but had seams);
with the pixel floor 42.2-43.2 dB. So the pixel floor (dropping sub-1/255 haze
everywhere, the GPU 3DGS rule) is the bulk of the loss, not the tile cull.
Note: the dashboard field `hero_vs_ref` has meant "vs the golden" since iter 132;
before that it was float PSNR vs a fresh cpu_cpp_mb render (~63.85 dB). Neither is
reference_v2.

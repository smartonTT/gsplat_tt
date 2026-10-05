# Device golden vs reference_v2: where and why they differ (task #251)

Status: final (run 697).

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

## 3. Error attribution (CPU model, modelled not measured on device)

Method: `docs/ref-golden-diverge-t251/dump_inputs.py` projects the hero view with the
fp32 cpu_cpp `project` (same camera as reference_v2: `benchmarks/cameras_v2.json`,
1024x1024, fov 50). `model.py` blends those inputs in fp32 torch and switches the
device blend rules on one at a time (tile cull floor, per-pixel floor, 8x4 microblock
T early-out on a bf16 T copy every 512 records, UNORM16 opacity/colour, bf16 output
tile). 8-bit PSNR on 64 sample tiles spread over the frame (`tiles_s64.txt`, results
`model_psnr_s64.json`). The model's `ref` arm reproduces reference_v2 at 65.8 dB, so
it is a faithful stand-in. Sample bias: the real golden is 41.76 dB on these tiles vs
41.16 dB full frame (~0.6 dB optimistic).

Note: the first run (run 676) used `tests/fixtures/hero/*.npz`; those inputs are a
different view (their own `blend_output.npy` is 12.3 dB vs reference_v2), so all its
numbers were ~12 dB and are discarded.

| source | PSNR vs ref if only this differs | gain if made exact (from golden) | where it shows | stage |
|---|---|---|---|---|
| per-pixel floor alpha<1/255 -> 0 (on top of 1/255 cull) | 42.9 dB (with cull) | the bulk: 42.9 -> 55.1 when removed | everywhere, image darker (mean -0.44 levels), 3.3x worse on high-gradient edges, bright regions worst (faint haze in front of bright surfaces dropped) | blend (`BLEND_PIXEL_FLOOR`) |
| tile/microblock cull at 1/255 (vs 1/16384) | 55.1 dB alone | ~+10 dB once the pixel floor is gone | same haze, smaller | K2/mat cull (`contrib_floor`) |
| device projection/blend numerics not in the model (residual) | 47.3 dB (golden vs model `dev`), 47.8 (pre-156 golden vs model) | caps every fix below at ~47 dB | zero-mean, ~5% of pixels off by >=2 levels, 3x on high-gradient edges, NOT on tile borders (0.39 vs 0.37), max 31-37 levels | pfwc (projection) / sort / blend, unattributed |
| bf16 output tile before 8-bit | 57.0 dB alone (vs 65.8) | ~0 today; matters only after the two above | uniform, rounding | blend pack (`CB_OUT` Float16_b) |
| T early-out (8x4 block, bf16 T, every 512, eps 1/256) | 66.6 dB alone | 0 | none | blend |
| UNORM16 opacity/colour | 65.7 dB alone | 0 | none | pfwc writer / blend decode |
| 0.99 alpha clamp, 0.3 blur, truncating 8-bit output | same in both paths | 0 | - | - |
| exp21f (21-bit exp) | not modelled; kernel comment says >=60 dB vs fp32 | ~0 | - | blend |

Lower floors (tile and pixel floor together, model, bf16 output):

| floor | PSNR (model) | records vs today | combined with 47.3 dB residual |
|---|---|---|---|
| 1/255 (today) | 42.9 | 1.000 | 41.6 (golden measured: 41.76 on sample) |
| 1/1024 | 51.8 | +8.8% | ~46.0 |
| 1/4096 | 56.7 | +13.9% | ~46.8 |
| 1/16384, no pixel floor (pre-156) | 57.1 | +17.3% | 46.8 (pre-156 golden measured: 46.79 full frame) |

So: the golden's 41.2 dB is two things of similar size in MSE terms after a fix.
(a) The 1/255 floors (iter-156) cost ~5.6 dB and explain the step. (b) A ~47 dB
device-numerics residual was already there before iter-156 and is still there.
Fixing the floor alone gets back to ~46-47 dB; going higher needs (b) found first.
The residual's pattern (zero mean, edge-concentrated, no tile seams, a few large
outliers) points at small geometry differences (means/conic/radius from pfwc) or
depth-order swaps, not at blend math; finding it needs a device dump of the pfwc
outputs for the hero view compared to cpu_cpp `project` (step 1 of the A/B below).

## 4. Fixes ranked by PSNR gain per ms (cost is modelled)

Cost data: iter-196 timings (blend 8.17 of 12.67 ms/view), iter-198 tip 11.94 ms/view,
task #230 busy split (pfwc 2.33, K2 0.98, sort_ol 1.44, mat+blend 7.56), and the
iter-156 measurement (17.3% fewer pairs saved 9.1% of blend time, -4.5 ms of 107.75).
Model: blend+mat cost grows 0.5-1.0x the pair growth, K2+sort grow 1x the pair growth.

| rank | fix | PSNR (modelled, full frame) | cost ms/view (modelled) | knob? |
|---|---|---|---|---|
| 1 | contrib_floor 1/1024 (pixel floor stays on, at 1/1024) | 41.2 -> ~45.5 | +0.4 to +0.9 | yes, runtime: `contrib_floor` in the camera json; no code |
| 2 | contrib_floor 1/4096 (pixel floor on) | -> ~46.2 | +0.6 to +1.4 | yes, same |
| 3 | pre-156 rule: 1/16384, `GSPLAT_TT_BLEND_PIXEL_FLOOR=0` | 46.79 (measured image) | +0.8 to +1.7 | yes; dominated by rank 2 |
| 4 | find and fix the ~47 dB device-numerics residual | up to ~+6-10 dB on top of 1-3 | unknown until found | diagnostic first |
| 5 | fp32 (or direct u8) output instead of bf16 tile | +0 now; ~+1-2 dB only after 4 | ~0 to +0.1 (pack 2x bytes) | needs code, compile define |
| - | pixel floor off, cull stays 1/255 | model 55 dB but brings back microblock-line seams (task #28/#41) | ~0 | rejected for seams |
| - | T early-out, UNORM16, exp21f, 0.99 clamp | 0 | - | not worth it |

Rank 1 is best gain per ms (~5-10 dB per ms). The ~1% rule does not apply here: this
is an accuracy-for-speed trade, so it should be measured as two modes (fast = today,
accurate = 1/1024) and the user picks the default.

## 5. Device A/B spec (follow-up)

See the hand-off follow-up "device A/B: contrib_floor accuracy mode + pfwc residual dump".

```text
Device A/B (p100, bicycle): accuracy mode for contrib_floor, plus a pfwc dump to find the ~47 dB residual. Background: docs/ref-golden-diverge-t251.md on ttp/t251-ref-golden-diverge. The golden (md5 46a725ab) is 41.16 dB vs benchmarks/reference_v2/hero.png; the 1/255 tile+pixel floor (iter-156) costs ~5.6 dB and an older device-numerics residual caps everything at ~47 dB.
Rules (binding): work in your own git worktree of ~/dev/gstt2, never edit ~/dev/gstt2; new branch ttp/<id>-floor-ab, push with no force/rebase/amend; never touch main or smarton/tt-project-opt; no PRs. Run ssh-preflight first; wrap every sync+build+run in `ttp lock p100 -- ...`; never reserve or release the machine (use the existing reservation only). Git identity Steve Marton <smarton@tenstorrent.com> via git config --local.
Arms (runtime knobs only, one build of the current tip): A = today (cameras_v2 contrib_floor 1/255, pixel floor on); B = contrib_floor 1/1024 (pixel floor on); C = contrib_floor 1/4096 (pixel floor on); D = contrib_floor 1/16384 with GSPLAT_TT_BLEND_PIXEL_FLOOR=0. Use a copy of cameras_v2.json with only contrib_floor changed; do not edit benchmarks/cameras_v2.json. Per arm: 3 rotated untraced rounds of the 30-view bench, ms/view mean; then render the hero view ON THE DEVICE (never the CPU reference), save hero.png, an x10 diff image and 8-bit PSNR against benchmarks/reference_v2/hero.png (label that reference), and look at the image for tile/microblock-line artifacts (state what you saw). Modelled expectation: B ~45.5 dB at +0.4..0.9 ms, C ~46.2 dB at +0.6..1.4 ms, D 46.79 dB at +0.8..1.7 ms; A 41.16 dB.
Diagnostic (same session): dump the device pfwc outputs for the hero view (means_2d, conic/cov2d, depth, radii, visible mask) and compare to cpu_cpp project from docs/ref-golden-diverge-t251/dump_inputs.py: report max/mean abs and relative error per field and the count of depth-order swaps within tiles. Feed the device-dumped inputs into docs/ref-golden-diverge-t251/model.py (T251_VARIANTS=dev,dev_pre156) to confirm whether projection explains the ~47 dB residual.
Delivery: add one iteration to opt/ttw/iters.jsonl per measured arm with its hero.png, diff and PSNR vs reference_v2 (the golden-match badge stays separate). Do not change the default or save a new golden in this task: report the table (ms/view, PSNR, artifacts) and recommend a default; the coordinator asks the user, since this trades speed for accuracy. If an arm ships later, its new golden md5 is saved then and the old golden is kept in tests/fixtures/hero/archive. Hand off with the table, the residual diagnosis and the branch head.
```

# Task #580 (L5/L6): CPU PSNR model of blend precision variants — result

2026-10-12 22:35 PDT. CPU only, no device runs. Model: `model.py` (t251/t260 blend replay with
bit-level emulation of the device `blend_exp21f`), saving: `summarize.py` (t205 cycle units,
110 cores at 1.35 GHz). Full 1024x1024 hero view, all 1024 tiles, 3 input sets x 11 variants.
Table: `out/summary.txt`; raw numbers: `out/model_*/model_psnr.json`.

Gate (from the spec): propose a device A/B only for a variant with predicted PSNR >= 42.4 dB
vs `benchmarks/reference_v2/hero.png` and >= 0.1 ms/view modelled saving.
`pred` = measured device 42.51 dB (best-iter-226) + modelled delta vs base on cpu inputs.

| variant | pred dB | d vs base | modelled saving ms/view (lo..hi) | verdict |
|---|---|---|---|---|
| exp degree-1, minimax (`exp1`) | 42.86 | +0.35 | +0.048..+0.139 | borderline: device A/B proposed (see below) |
| exp degree-1, least squares (`exp1ls`) | 42.45 | -0.06 | +0.044..+0.131 | no (same saving, worse PSNR than exp1) |
| T_EPS 1/128 (`teps128`) | 42.50 | -0.01 | +0.055..+0.095 | no-go (saving < 0.1 ms) |
| T period 1024 (`per1024`) | 42.51 | 0.00 | -0.624..-0.358 | no-go (slower: more microblock evals) |
| T period 2048 (`per2048`) | 42.51 | 0.00 | -1.019..-0.587 | no-go (slower) |
| alpha floor 2/255 (`floor2`) | 37.74 | -4.77 | +0.418..+0.651 | no-go (PSNR far below 42.4) |
| teps128 + per2048 + floor2 | 37.74 | -4.77 | -0.227..-0.086 | no-go |
| all (incl. exp1) | 38.02 | -4.48 | -0.103..-0.043 | no-go |

Checks: `base_exact` (torch.exp instead of the emulated exp21f) changes PSNR by < 0.01 dB, so
the current degree-2 exp costs nothing in quality. `base_nochk` (no T early-out) shows the
current period-512 readback is worth about 1.1-2.0 ms/view; making it coarser only loses.

## L6 (Newton step on device 1/z and conic inverse)

Already landed and on by default (t266/t290), so the cpu-input rows above stand for today's
device. On device-dumped pfwc rows (iter 199, pre-#266) vs the same rows with exact projection
values, the Newton step is worth +0.45 dB for every variant except the floor-2/255 ones
(+0.29..+0.31). It does not interact with exp1/teps: the deltas match within 0.01 dB. The
t260 estimate (~+0.2 dB) was conservative. Nothing left to do for L6.

## Verdict

- Alpha floor 2/255, T_EPS 1/128 and coarser T period: closed, no-go.
- L6: already landed; confirmed worth ~+0.45 dB on the device-row model.
- Degree-1 exp (minimax, `exp1`): passes PSNR (42.86 predicted), saving 0.048..0.139 ms/view,
  i.e. 0.6-1.6 % of the 8.6 ms b2b headline. The low end fails the 0.1 ms gate, the high end
  passes. Caveats: the +0.35 dB comes from its +1 % mean bias offsetting the floor haze, not
  from better accuracy (vs the current device output it is 54.6 dB, vs golden 54.3 dB instead of
  57.7), and on device rows the gain shrinks to +0.15 dB. Still >= 42.4 either way.
  Judgment: the model cannot decide which end of the saving range holds (it depends on whether
  the dropped MAD sits on the SFPU critical path), and the change is one function
  (`blend_exp21f` -> one MAD), so a single device A/B is cheaper than more modelling.
  Proposed A/B, with a kill rule: keep only if b2b ms/view (30 bicycle views, `--back-to-back`)
  drops by >= 0.1 ms, device hero PSNR vs reference_v2 >= 42.4 dB, and the hero screenshot shows
  no banding or tile artefacts. Otherwise shelve degree-1 exp for good.

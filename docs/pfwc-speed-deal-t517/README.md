# t517: per-core speed-weighted pfwc deal → order-keeping range deal (built, A/B null: shelved)

Task #517, 2026-10-12 (PDT). Tree: this branch (from t512 c720c864 = best-iter-223 + off-by-default
levers). Env-gated, default off.

## Why the slow pfwc cores are slow (step 1)

`ana517.py` maps the t506/t512 per-core pfwc length factors (20 views, p150 12x10) to logical
cores and compares them with the load of each core's strided tile set (t439 hero CPU model,
`../pfwc-deal-t439/bicycle-hero-w.txt`):

- corr(core factor, strided tile-set weight) = **0.92**; tile-set weight range 0.956..1.061.
- After the tile-set fit the residual "core speed" is 0.963..1.044, std 0.013 (factor std 0.033).
- No row/column (NoC or DRAM distance) pattern: row means 0.976..1.033, column means 0.993..1.017.
  The slow cores sit in clusters of consecutive logical ids (e.g. y3 x4..x8, y5 x6..x10), i.e.
  neighbouring tile sets with dense content.

So the "property of the core" in t512 is the core's fixed strided tile set (the same tiles every
view), not the hardware. Factors: `core-factor-t506.txt` (logical c = y*12 + x).

## The lever (step 2): `GSPLAT_TT_PFWC_DEAL=range`

`pfwc_deal::range_cuts` cuts the legacy compaction sequence (SeqMap: legacy core c' owns tiles
c', c'+C, ...) into C contiguous runs of equal weight; core c takes run [q0[c], q0[c+1]) and writes
its segment at storage index q0[c]*1024. Because the runs are contiguous in sequence order, the
storage order — and so the sort order and the image — is exactly the legacy one (unit test: the
concatenated lists equal the legacy sequence). Within a run the tiles stay strided, so DRAM
locality is kept (unlike t439's LPT, which reordered gids and changed 280 hero pixels).

- Host (`pfwc_device.cpp`): per-core tile list through the #433/#439 `PFWC_TILE_LIST` path, run
  length as num_chunks, q0[c]*1024 as seg_base. Weights from `GSPLAT_TT_PFWC_DEAL_W`.
- Device: the split writer stores seg_base+1 in counts word T_SB; `pfwc_fuse::seg_table` uses it
  (0 = the SeqMap base, as before), so the segment K2 accepts unequal runs. Default path writes the
  same base it computed before.
- Weights `bicycle-w517.txt` (`make_w517.py`): tile t gets f_c * w_t / mean(w over core c's tiles)
  — the measured per-core factor sets each legacy set's total, the hero model spreads it.
  Predicted per-core max/mean 1.104 → 1.019; run lengths 45..53 tiles (strided 49..50).
- Caveat: the factors were measured on the same 30-view orbit that is benchmarked (in-sample),
  and the weights are a static, scene-specific file.

## A/B (step 3): null, shelved

bh-33 (IRD 138447), 2026-10-12 09:30 PDT, tree 228fd0e1, `render/run.py --back-to-back` over the 30
bicycle views, 1 warm-up + 5 alternating rounds (20 passes each), `drive517.sh` under `ttp lock p100`.
The box reports **card=p100a, grid 11x10 (110 cores)**, not the p150 12x10 the factors came from.

| arm | per-round ms/view (b2b) | median |
|---|---|---|
| off | 9.375 9.369 9.380 9.362 9.368 | **9.368** |
| range | 9.372 9.373 9.366 9.367 9.361 | **9.366** |

- Gain 0.002 ms/view (noise; spread 0.02). Below the 0.04 stack line → **shelved**, stays default off.
- Stage medians (100 passes/arm): project 2.169 → 2.232, sort 1.072 → 1.011, blend 5.860 → 5.917,
  period 9.365 → 9.366. pfwc overlaps the previous view's blend in b2b; moving its tail does not
  move the period.
- md5 906e0435 = golden 30/30 in both arms; the 30 dumped views are byte-identical between arms
  (order-keeping deal, as designed). Device hero `out/hero-rg.png` PSNR 42.51 dB vs
  `benchmarks/reference_v2/hero.png`; diff `out/diff-rg.png` (x8). Looked at both: no tile artifacts,
  diff is the usual spoke/edge residue.
- Caveat: on the 110-core p100 the per-core factors (measured on 120 p150 cores) do not map to the same
  tile sets. Since the factors are 92% tile-set content, the weights reduce to the per-tile hero model,
  which is grid independent, so the deal still equalized modelled load. A p100-specific factor file is
  not worth a run: the b2b period did not move at all, so pfwc per-core balance is off the critical path.

# t517: per-core speed-weighted pfwc deal → order-keeping range deal (built, A/B not run yet)

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

## A/B (step 3): not run

`drive517.sh` / `bench517.sh` (bh-33, 5 alternating b2b rounds off/rg + one --dump-views run per arm
for md5 and the device hero). At about 03:45 PDT `ssh-preflight bh-33` returned auth_failed (publickey
denied); t514's driver on bh-33 had its connection closed by the remote host at 02:10 PDT
(09:10Z), so the measurement reservation looks expired/released.
Run: `ttp detach t517 -- ttp lock p100 -- docs/pfwc-speed-deal-t517/drive517.sh <rev>`.

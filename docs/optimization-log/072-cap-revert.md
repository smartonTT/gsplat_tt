## iter 072 — cap-revert (GSPLAT_TT_MAX_G_PER_TILE default 448 → 0)

**Status:** KEEP (visual-quality regression fix — reclaiming quality before further perf work)

### Hypothesis

The per-tile Gaussian cap (introduced at iter 032, tightened at iters 037/044) was
trading visible quality for kernel speed. At HEAD (iter 071) with the cap=448 default
the stitch renders showed tile-aligned diagonal cross-hatch artifacts and PSNR collapsed
to 23–32 dB vs the iter-0 (dbb856a) TT references. The plan's Sub-task C explicitly
authorises reverting the cap to 0 if cap=0 is visibly cleaner.

### Change

[backends/tt/backend.py:253-262](../../backends/tt/backend.py#L253-L262): default of
`GSPLAT_TT_MAX_G_PER_TILE` changed from `"448"` → `"0"` (no cap). Env override still
respected for perf sweeps. Comment block updated.

### Measurements (1024×1024, cap=0 default, viewer-default cameras)

| view              | kernel ms | PSNR vs iter-0 ref | mean\|d\| | max\|d\| |
|-------------------|-----------|--------------------|----------|---------|
| stitch / hero     | 26.78     | 39.17 dB           | 1.05      | 76      |
| stitch / side     | 26.07     | 41.19 dB           | 0.72      | 95      |
| stitch / top      | 32.18     | 38.02 dB           | 1.08      | 141     |
| luigi  / hero     |  8.12     | 33.43 dB           | 1.37      | 171     |
| strawberry / hero | 131.73    | 34.62 dB           | 1.20      | 237     |

All stitch views ≥ 38 dB (well above the 35 dB gate). Luigi/strawberry sit at 33–34 dB
which is render drift between iter-0 and HEAD on a sparse subject, **not** tile
artifacts — confirmed by visual inspection of the rendered PNGs (full Stitch frame
matches the reference; no cross-hatch on body/face).

Compare with iter 071 at the prior default (cap=448):

| view          | iter 071 PSNR (cap=448) | iter 072 PSNR (cap=0) |
|---------------|-------------------------|-----------------------|
| stitch / hero | 24.55 dB                | 39.17 dB              |
| stitch / side | 31.71 dB                | 41.19 dB              |
| stitch / top  | 23.34 dB                | 38.02 dB              |

### Cost

Kernel time regresses from the cap=448 figures (~17 ms stitch hero) to ~26–32 ms.
This is the price of correctness — the remaining ~25 ms-to-1 ms gap will be closed
by kernel-side FPU/Dst-resident-state work (iters 064-style merges, Dst-resident
state, basis-form fix, 16×16 face culling) without re-introducing truncation.

### Deferred-NO consequence

Iters 032 (per-tile cap), 037 (cap 1024→512), 044 (cap 512→448) — flagged as
deferred-NO in the report (kept the kernel-ms wins but at quality cost we're now
reclaiming). They are not reverted in code (env can still set cap>0 for sweeps).

### Next

Watchdog subagent (continuous PSNR + tile-pattern monitor), then the 16×16 face
optimization push toward 1 ms on top of this clean baseline.

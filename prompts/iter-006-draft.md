# iter-006 worker prompt

You are the iter-006 worker. The full forward pipeline now runs in C++ end-to-end with a stable C++ baseline of **sum30 ≈ 3257 ms** (blend dominates at 79 ms median). iter-006 starts Phase 3 (algorithmic iters) by introducing **per-microblock ellipse culling** in numpy. The C++ port follows in iter-007.

## Read first (mandatory)

1. `/Users/smarton/dev/gstt2/opt/plan.md`
2. `/Users/smarton/dev/gstt2/opt/microblock-cpu-spec.md` ← **THE SPEC** (read end to end)
3. `/Users/smarton/dev/gstt2/prompts/worker.md`
4. `/Users/smarton/dev/gstt2/gsplat/rasterization.py` lines 178-493 (`get_tile_assignments`, `sort_and_bin`, `alpha_blend` — the existing per-tile pipeline)
5. `/Users/smarton/dev/gstt2/backends/cpu/backend.py` (the numpy backend you'll extend)
6. `/Users/smarton/dev/gstt2/backends/__init__.py` (registry)

## Iter spec

ITER: 006-microblock-cull-numpy
GOAL: Implement `microblock_cull` and `alpha_blend_microblock` in numpy. Register a new backend `cpu_mb` that runs `project → tile_assign → sort → microblock_cull → alpha_blend_microblock`. Add pytest tests for the four microblock invariants. Render the 30-view benchmark with `cpu_mb` and confirm min PSNR vs `benchmarks/reference_v2/` ≥ 60 dB and drop rate < 5%.

This iter is **about correctness, not speed**. Numpy is slow; sum30 with `cpu_mb` may be 200 s — that's OK. The C++ port is iter-007. What matters: ONE: the cull math is correct (PSNR invariant), TWO: the data structures (mb_header, mb_stream) are exactly what iter-007 will port.

## Files to touch

- `gsplat/rasterization.py` — add `microblock_cull(...)` and `alpha_blend_microblock(...)` (NEW functions; do NOT modify the existing `get_tile_assignments`, `sort_and_bin`, or `alpha_blend`)
- `backends/cpu/backend.py` — add a `CpuMicroblockBackend` class (subclass `CpuBackend` or a separate class). Override `blend()` to internally call `microblock_cull` then `alpha_blend_microblock`. Keep project/tile_assign/sort identical.
- `backends/__init__.py` — register `"cpu_mb"` → `CpuMicroblockBackend`
- `scripts/render_30frame.py` — one-line additive change: also merge `res.sub_timings` into the per-frame `row` (so `blend.microblock_drop_pct` lands in `timing.jsonl`). Do NOT change the existing fields.
- `tests/spec/test_microblock.py` — NEW pytest file (create `tests/spec/` if missing, plus `tests/__init__.py` and `tests/spec/__init__.py` if needed) with 4 invariant tests + 1 PSNR test

## Algorithm spec

### `microblock_cull` — signature

```python
def microblock_cull(
    means_2d: torch.Tensor,             # (M, 2) — visible Gaussian centers
    covs_2d: torch.Tensor,              # (M, 2, 2)
    opacities: torch.Tensor,            # (M,)
    sorted_gaussian_ids: torch.Tensor,  # (P,) int64
    tile_ranges: torch.Tensor,          # (num_tiles, 2) int64
    tiles_x: int,
    tiles_y: int,
    tile_size: int = 32,
    contrib_floor: float = 15.0 / 255.0,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns:
      mb_header : (num_tiles, 32, 2) int64 — (offset, count) per microblock,
                  where offset indexes into mb_stream. Tiles with no entries
                  have all-zero rows.
      mb_stream : (L_prime,) int64 — depth-sorted GLOBAL gaussian ids
                  (NOT local — to keep alpha_blend_microblock simple). Same
                  semantics as `sorted_gaussian_ids` but per-microblock and
                  with the cull applied.
      stats     : {"pairs_in": P, "pairs_out": L_prime, "drop_pct": <float>}
    """
```

### `microblock_cull` — implementation

For each tile (loop ty in tiles_y, tx in tiles_x):
1. `start, end = tile_ranges[tile_id]`. If `start == end`, mb_header[tile_id] stays zeros, no entries appended.
2. `tile_g_ids = sorted_gaussian_ids[start:end]` — the depth-sorted Gaussians for this tile (length L).
3. `tx_tile = tx * tile_size`, `ty_tile = ty * tile_size`.
4. For each microblock m in 0..31:
   - `mb_origin_x = (m & 3) * 8`, `mb_origin_y = (m >> 2) * 4`
   - `mb_ox = tx_tile + mb_origin_x`, `mb_oy = ty_tile + mb_origin_y`
   - For each Gaussian g in `tile_g_ids`:
     - `(a, b, c) = covs_2d[g, 0, 0], covs_2d[g, 0, 1], covs_2d[g, 1, 1]`
     - `det = max(a*c - b*b, 1e-6)`
     - `mean_x, mean_y = means_2d[g]`
     - `cx = clamp(mean_x, mb_ox, mb_ox + 8)` — note "mb_ox + 8" (microblock is 8 pixels wide)
     - `cy = clamp(mean_y, mb_oy, mb_oy + 4)` — "mb_oy + 4" (microblock is 4 pixels tall)
     - `dx, dy = cx - mean_x, cy - mean_y`
     - `m2 = (c*dx*dx - 2*b*dx*dy + a*dy*dy) / det`
     - `keep = opacities[g] * exp(-0.5 * m2) >= contrib_floor`
   - The set of g's that pass — in their original depth order — becomes `mb_stream[mb_header[tile_id, m, 0] : mb_header[tile_id, m, 0] + mb_header[tile_id, m, 1]]`.

You can VECTORIZE this within a tile: stack the 32 microblocks' (mb_ox, mb_oy) into shape (32,), broadcast against (L,) Gaussians to get a (32, L) keep-mask, then for each m extract the kept Gaussians in order. The naive per-tile python loop is fine for the numpy spec — just don't write it as O(num_tiles × 32 × L × 32×8) (that would be hours per frame).

**One critical detail:** the `mb_stream` is built **per tile, per microblock, in raster order over m** AND **in depth order within m**. That gives the exact stream layout iter-007 ports to C++.

### `alpha_blend_microblock` — signature

```python
def alpha_blend_microblock(
    means_2d: torch.Tensor,           # (M, 2)
    covs_2d: torch.Tensor,            # (M, 2, 2)
    colors: torch.Tensor,             # (M, 3)
    opacities: torch.Tensor,          # (M,)
    mb_header: torch.Tensor,          # (num_tiles, 32, 2) int64
    mb_stream: torch.Tensor,          # (L_prime,) int64
    image_height: int,
    image_width: int,
    tile_size: int = 32,
) -> torch.Tensor:
    """Microblock-major alpha compositing.

    Inner loop per tile, per microblock:
      For each pixel in the microblock (4 rows × 8 cols):
        For each Gaussian g in mb_stream[mb_header[tile, m, 0] : ...]
          (which are already in depth order from microblock_cull):
            evaluate alpha, composite.

    Microblocks within a tile share the per-tile (T, accum) state by
    indexing into the tile's pixel array — i.e., each microblock writes
    its 4×8 sub-region of the full tile output.

    Returns: (H, W, 3) float32 image in [0, 1].
    """
```

The blend inside each microblock follows the same per-pixel scalar form as `alpha_blend`:
- T initialized to 1.0 (per pixel within the microblock — independent of other microblocks because each pixel is in exactly one microblock)
- `power = -0.5 * (ci_a*dx*dx + 2*ci_b*dx*dy + ci_c*dy*dy)`
- `alpha = min(opacity * exp(min(power, 0)), 0.99)`
- `T *= (1 - alpha)`
- `accum += alpha * T_prev * color`
- Early termination check (numpy literal `0.0001`): `if transmittance.max() < 0.0001: break`

**Match `alpha_blend`'s ordering exactly** — power, alpha-clamp, T update, accumulation order — so PSNR vs per-tile reference is ≥ 60 dB (typically infinite up to fp accumulation order, but per-tile vs per-microblock orderings within a microblock are identical because the depth order is preserved by Invariant 2).

### CpuMicroblockBackend

```python
class CpuMicroblockBackend(CpuBackend):
    """Numpy backend with microblock culling between sort and blend.

    All four upstream stages identical to CpuBackend; blend() overridden to
    consume the microblock data structures.
    """
    def blend(self, means_2d, covs_2d, colors, opacities,
              sorted_gaussian_ids, tile_ranges, image_height, image_width):
        tiles_x = (image_width + 31) // 32
        tiles_y = (image_height + 31) // 32
        mb_header, mb_stream, stats = microblock_cull(
            means_2d, covs_2d, opacities,
            sorted_gaussian_ids, tile_ranges,
            tiles_x, tiles_y, tile_size=32,
        )
        image = alpha_blend_microblock(
            means_2d, covs_2d, colors, opacities,
            mb_header, mb_stream,
            image_height, image_width,
        )
        # Emit stats as sub-timings keys for the harness to pick up.
        return image.numpy(), {"microblock_drop_pct": stats["drop_pct"]}
```

## Pytest invariants

`tests/spec/test_microblock.py` — at least these 5 tests:

1. **mb_stream length:** `sum(mb_header[t, m, 1] for t in tiles for m in 0..31) == len(mb_stream)`.
2. **Per-microblock depth monotonicity:** for each non-empty `(tile, m)` slice, the depths of its Gaussians are non-decreasing.
3. **Mask completeness (Invariant 1):** for 32 randomized synthetic Gaussians on a 64×64 image, every pixel where the *true* per-Gaussian alpha exceeds `contrib_floor` lies within an active `(g, m)` entry. Brute-force check.
4. **Drop rate < 5%:** with the hero fixture, `stats["drop_pct"] < 5.0`.
5. **End-to-end PSNR ≥ 60 dB:** load the hero fixture's `blend_inputs.npz`, run `microblock_cull` + `alpha_blend_microblock`, compute PSNR vs `tests/fixtures/hero/blend_output.npy`. Must be ≥ 60 dB.

## Layer gates

```bash
source scripts/_env.sh

# Layer 1 — none new (no C++)
ctest --test-dir build --output-on-failure -j   # must still pass at 29/29

# Layer 2 — pytest invariants
$LOCAL_PY -m pytest tests/spec/test_microblock.py -x -v

# Also rerun every existing verify_stage so we know cpu_cpp / cpu unchanged
for stage in project tile_assign sort blend; do
  $LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage $stage
done

# Layer 3 — 30-view render with the NEW backend
rm -rf /tmp/iter006
$LOCAL_PY scripts/render_30frame.py --backend cpu_mb --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter006 --warmup 1

$LOCAL_PY -c "
from PIL import Image; import numpy as np; from pathlib import Path; import json
ms=[]
for f in sorted(Path('benchmarks/reference_v2').glob('*.png')):
    a=np.asarray(Image.open(f).convert('RGB'),np.float64)/255
    b=np.asarray(Image.open(Path('/tmp/iter006')/f.name).convert('RGB'),np.float64)/255
    mse=float(np.mean((a-b)**2))
    p=float('inf') if mse<=0 else 10*np.log10(1/mse)
    ms.append((f.stem,p))
ms.sort(key=lambda x:x[1])
print(f'PSNR min={ms[0][1]:.2f}  mean={sum(p for _,p in ms)/len(ms):.2f}  max={ms[-1][1]:.2f}  (gate 60 dB)')
rows=[json.loads(l) for l in open('/tmp/iter006/timing.jsonl').read().splitlines() if l.strip()]
print(f'sum_total_ms={sum(r[\"total_ms\"] for r in rows):.1f}  (numpy backend, slow — that is fine)')
drops=[r.get('blend.microblock_drop_pct',-1) for r in rows]
print(f'drop pct: min={min(d for d in drops if d>=0):.2f} max={max(drops):.2f} mean={sum(d for d in drops if d>=0)/sum(1 for d in drops if d>=0):.2f}')
"
```

## BUDGET

5 attempts. If you can't get PSNR ≥ 60 dB after 5 tries, `git checkout -- .` and surface "BLOCKED ON: <root cause>".

Most likely failure modes:
- **PSNR ~30 dB**: per-microblock T initialization wrong (you reset T per microblock inside the tile, but the math requires per-pixel T initialized to 1.0 once per pixel for the entire image — since each pixel is in exactly ONE microblock, the per-microblock T scope is the right scope for the pixels in that microblock; but make sure you don't share T across microblocks).
- **Drop rate ≥ 5%**: per-tile cull and per-microblock cull don't agree. Most often: wrong microblock origin formula (you used 8 rows × 4 cols instead of 8 row-bands × 4 col-groups — re-read spec §1).
- **mb_stream wrong length**: forgot to mask out the gaussians dropped by Mahalanobis cull at the microblock level. Each `(g, m)` pair MUST satisfy the keep test.

## SUMMARY

End with:

```
SUMMARY: iter-006 status=PASS|FAIL pytest=N/M render30_min_psnr_dB=X.X drop_pct_max=Y.YY sum_total_ms=Z.Z
```

Do not commit. Begin.

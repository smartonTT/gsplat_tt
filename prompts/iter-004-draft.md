# iter-004 worker prompt (draft)

You are the iter-004 worker for the gsplat_tt_2 sprint. iter-002 and iter-003
landed (project + tile_assign + sort are in C++). Your job is the BIG one:
port `alpha_blend` to C++. After this iter, the full forward pipeline runs in
C++ end-to-end — and 95%+ of the speedup the sprint will see comes from this
iter.

## Read first (mandatory)

1. `/Users/smarton/dev/gsplat_tt_2/opt/plan.md`
2. `/Users/smarton/dev/gsplat_tt_2/opt/microblock-cpu-spec.md` (for context; the microblock structure is NOT used in iter-004 — only iter-008 onward — but read it so you know what's coming)
3. `/Users/smarton/dev/gsplat_tt_2/prompts/worker.md`
4. `/Users/smarton/dev/gsplat_tt_2/gsplat/rasterization.py` lines 374-493 (`alpha_blend` — the algorithm spec)
5. `/Users/smarton/dev/gsplat_tt_2/src/gsplat_cpu/thread_pool.h` (use this to parallelize per-tile)
6. `/Users/smarton/dev/gsplat_tt_2/src/gsplat_cpu/project.{h,cpp}` + `tile_assign.{h,cpp}` + `sort.{h,cpp}` (style template + numpy↔C++ binding pattern)
7. `/Users/smarton/dev/gsplat_tt_2/scripts/verify_stage.py` (Layer 2 gate; blend uses PSNR ≥ 60 dB)

## Iter spec

ITER: 004-cpp-alpha-blend
GOAL: Port `gsplat.rasterization.alpha_blend` to C++. Inner per-tile loop is scalar (one pixel at a time) — microblock-major iteration is iter-008, NOT here. Parallelism: one tile per ThreadPool task, std::thread pool wide. After this iter, the full forward pipeline (project → tile_assign → sort → blend) runs in C++ via `--backend cpu_cpp`.

## Files to touch

- `src/gsplat_cpu/blend.h` / `.cpp`        (new — `blend(...)`)
- `src/CMakeLists.txt`                     (add blend.cpp; link `gsplat_cpu` against any pthread/atomics if needed — likely already in place)
- `backends/cpu_cpp/pybind_module.cpp`     (add `m.def("blend", ...)`)
- `backends/cpu_cpp/backend.py`            (override `blend()`)
- `tests/unit/test_blend.cpp`              (new — Catch2 tests; see below)
- `tests/CMakeLists.txt`                   (add test_blend.cpp)

## C++ function signature

In `src/gsplat_cpu/blend.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct BlendResult {
    std::vector<float> image;  // H * W * 3, row-major, fp32, in [0, 1]
};

BlendResult blend(
    const float*   means_2d,             // M * 2
    const float*   covs_2d,              // M * 4 (a, b, b, c — symmetric, off-diag duplicated)
    const float*   colors,               // M * 3
    const float*   opacities,            // M
    const int64_t* sorted_gaussian_ids,  // P  (int64 to match iter-003 sort output)
    const int64_t* tile_ranges,          // num_tiles * 2 (start, end)
    std::size_t M, std::size_t P,
    int image_height, int image_width,
    int tile_size,                       // 32
    ThreadPool& pool                     // shared per-process pool (constructed once)
);

} // namespace gsplat_cpu
```

## Algorithm spec (mirror numpy — line for line)

For each tile (parallelized over the pool):
1. Read `(start, end) = tile_ranges[tile_id]`. If `start == end`, skip (output remains 0).
2. Determine tile pixel bounds: `py_start = ty * tile_size; px_start = tx * tile_size`. Clamp `py_end / px_end` at image bounds (edge tiles may be partial).
3. Inverse covariance: precomputed ONCE per blend call across all M Gaussians (NOT per-tile, not per-iteration). Numpy does this. C++ does the same — allocate a length-`4*M` fp32 array up front (or `M` `Mat2f`), fill, then read inside the per-tile inner loop. This is ~0.5 ms on stitch (60k visible) and amortizes across all tiles.
   - `det = max(a*c - b*b, 1e-6f)`
   - `cov_inv = [c/det, -b/det, -b/det, a/det]` (only 3 unique entries; you can pack as `(c/det, -b/det, a/det)`)
4. Per-tile scratch: `T[tile_h * tile_w] = 1.0f`, `accum[tile_h * tile_w * 3] = 0.0f`. Stack allocation is fine — 32*32*4 = 4KB for T and 32*32*3*4 = 12KB for accum, comfortably below the 8MB default stack on macOS. Per-tile, NOT per-thread global — one tile at a time per task.
5. Iterate `idx ∈ [start, end)`:
   - `g = sorted_gaussian_ids[idx]` (int64 — careful with the index type)
   - For each pixel (i, j) in the tile, with global coords `(px = px_start+j+0.5f, py = py_start+i+0.5f)`:
     - `dx = px - means_2d[g*2 + 0]; dy = py - means_2d[g*2 + 1]`
     - `power = -0.5f * (ci_a*dx*dx + 2.0f*ci_b*dx*dy + ci_c*dy*dy)` where `(ci_a, ci_b, ci_c) = cov_inv[g]`
     - `gw = std::exp(std::min(power, 0.0f))` — numpy uses `np.exp(np.minimum(power, 0.0))`; the min is for numerical safety when `power` slightly overflows zero due to fp error.
     - `alpha = std::min(opacities[g] * gw, 0.99f)` — numpy uses `np.clip(x, None, 0.99)`, which is upper-only. There is NO lower-bound clamp (alpha never goes negative because gw≥0 and opacity≥0). Match: no `std::max(..., 0.0f)` needed; a single `std::min(..., 0.99f)` is correct.
     - `t = T[i*tile_w + j]; at = alpha * t`
     - `accum[(i*tile_w + j)*3 + c] += at * colors[g*3 + c]` for c in {0, 1, 2}
     - `T[i*tile_w + j] = t * (1.0f - alpha)`
   - After processing this Gaussian for the whole tile, compute `tmax = max over T[]`. If `tmax < 0.0001f`, break out of the for-idx loop. (numpy uses 0.0001 exactly — match the literal so PSNR stays infinite for visible Gaussians.)
6. Write `accum[(i*tile_w+j)*3 + c]` to `image[(py_start+i)*W*3 + (px_start+j)*3 + c]`.

### Do NOT add these (matches numpy exactly)
- **No per-pixel alpha-skip** (numpy doesn't have `if alpha < 1/255: continue` — the comment in `rasterization.py:476-479` explicitly says it's removed). Adding this in C++ will drop PSNR below 60 dB.
- **No fp64 accumulator** — numpy uses fp32 throughout. Use fp32 for T, accum, all multiplies. (C++ may benefit from fp64 accumulation per tile later for image quality, but iter-004 is "mirror the spec".)
- **No microblock subdivision** — that's iter-008. Iter-004 is a pure scalar per-pixel loop, full tile.

## Performance notes (you can defer these to iter-005 / iter-008 BUT keep an eye on them)

- For iter-004 scope: **scalar inner loop** (one pixel at a time, three components). No SIMD, no microblock-major iteration. The goal is **correctness first** (PSNR ≥ 60 dB vs numpy reference); speed is iter-005's concern.
- A per-tile std::array<float, 3*32*32> on the STACK is ≤ 12 KB — cheap and avoids heap traffic. The transmittance T is another 32×32×4 = 4 KB.
- Inverse covariance precompute: do it ONCE per blend call over all M Gaussians, NOT per-tile. The numpy reference also does this.
- Early-term check `max(T) < 1e-4`: compute as a tile-wide reduction every Gaussian iteration. For iter-004 a simple linear scan is fine; iter-008 will replace this with per-microblock early-term.

## CpuCppBackend.blend (Python side)

```python
def blend(self, means_2d, covs_2d, colors, opacities,
          sorted_gaussian_ids, tile_ranges, image_height, image_width):
    # Force contiguous + correct dtypes — torch -> numpy on a non-contig
    # slice can produce strided arrays that confuse the pybind11 buffer view.
    img = self._mod.blend(
        np.ascontiguousarray(means_2d.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(covs_2d.detach().cpu().numpy().reshape(-1, 4), dtype=np.float32),
        np.ascontiguousarray(colors.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(opacities.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(sorted_gaussian_ids.detach().cpu().numpy(), dtype=np.int64),
        np.ascontiguousarray(tile_ranges.detach().cpu().numpy(), dtype=np.int64),
        int(image_height), int(image_width), 32,
    )
    # The C++ side returns a (H, W, 3) fp32 numpy array. Match the numpy
    # backend's contract (returns (image, sub_timings_dict)).
    return img, {}
```

The C++ side allocates a shared `gsplat_cpu::ThreadPool` (lazily, on first blend call, sized to `hardware_concurrency()`) and reuses it across all calls — DO NOT construct a new pool per frame (that's a ~ms allocation per call, way too much).

Sketch:
```cpp
// pybind_module.cpp
static gsplat_cpu::ThreadPool& global_pool() {
    static gsplat_cpu::ThreadPool pool(0);  // default = hardware_concurrency()
    return pool;
}

m.def("blend", [](...) {
    return gsplat_cpu::blend(..., global_pool());
});
```

## Catch2 unit tests

`test_blend.cpp`:
1. Single tile, single Gaussian at tile center, opacity=1.0, color=(1,0,0), large radius — the tile interior should be ~red. Verify center pixel ≈ (1, 0, 0) within 1e-3.
2. Two Gaussians, second on top, opacity=1.0 — front-to-back compositing means second Gaussian's color dominates pixel center.
3. Transmittance early-termination: 100 stacked opaque Gaussians at the same position; max(T) should hit <1e-4 quickly; only first few iterations actually contribute.
4. Empty tile (no entries) — output region is all zeros.
5. Tile that straddles image edge — only the in-bounds sub-region is written; image dims respected.

## Layer gates

```bash
# Layer 1
ctest --test-dir build --output-on-failure -j

# Layer 2 (PSNR gate, threshold 60 dB)
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage blend

# Layer 3 — full end-to-end render
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter004 --warmup 1
$LOCAL_PY -c "
from PIL import Image; import numpy as np
from pathlib import Path
ms = []
for f in sorted(Path('benchmarks/reference_v2').glob('*.png')):
    a = np.asarray(Image.open(f).convert('RGB'), np.float64)/255
    b = np.asarray(Image.open(Path('/tmp/iter004')/f.name).convert('RGB'), np.float64)/255
    mse = float(np.mean((a-b)**2))
    p = float('inf') if mse <= 0 else 10*np.log10(1/mse)
    ms.append(p); print(f'{f.stem:14s} psnr={p:.1f} dB')
print(f'MIN_PSNR={min(ms):.2f} dB  (gate: 60.0 dB)')
"
```

Also record sum-of-30-ms by reading the timing.jsonl in `/tmp/iter004/`:
```bash
$LOCAL_PY -c "
import json
rows = [json.loads(l) for l in open('/tmp/iter004/timing.jsonl').read().splitlines() if l.strip()]
total = sum(r['total_ms'] for r in rows)
blend = sum(r.get('blend_ms', 0) for r in rows)
print(f'sum_total_ms = {total:.1f}, sum_blend_ms = {blend:.1f}')
"
```

## BUDGET

5 build attempts. If stuck after 5, `git checkout -- .` and surface BLOCKED ON line. The blend stage is the hardest of the four — PSNR < 60 dB after the port usually means fp accumulation order is wrong (e.g. early-term check too aggressive, or color += order swapped).

## SUMMARY

```
SUMMARY: iter-004 status=PASS|FAIL build=ok|fail ctest=N/M verify_blend=psnr=X.X|fail render30_min_psnr_dB=X.X sum_total_ms=X.X
```

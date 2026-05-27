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
    const float* means_2d,             // M * 2
    const float* covs_2d,              // M * 4 (a, b, b, c)
    const float* colors,               // M * 3
    const float* opacities,            // M
    const int32_t* sorted_gaussian_ids,// P
    const int64_t* tile_ranges,        // num_tiles * 2 (start, end)
    std::size_t M, std::size_t P,
    int image_height, int image_width,
    int tile_size,                     // 32
    ThreadPool& pool                   // shared per-process pool (constructed once)
);

} // namespace gsplat_cpu
```

## Algorithm spec (mirror numpy)

For each tile (parallelized over the pool):
1. Read `(start, end) = tile_ranges[tile_id]`. If `start == end`, skip.
2. Determine tile pixel bounds: `py_start = ty * tile_size; px_start = tx * tile_size`. Clamp `py_end / px_end` at image bounds (edge tiles may be partial).
3. Precompute inverse covariance per ENTRY (or per gaussian; entries reference gaussians, so per-gaussian cache is fine and matches numpy):
   - `det = max(a*c - b*b, 1e-6f)`
   - `cov_inv = [c/det, -b/det, -b/det, a/det]`
4. Allocate `T[tile_h, tile_w] = 1.0`, `accum[tile_h, tile_w, 3] = 0.0` (stack or per-thread scratch — see notes).
5. Iterate `idx ∈ [start, end)`:
   - `g = sorted_gaussian_ids[idx]`
   - For each pixel (i, j) in the tile, with global coords `(px = px_start+j+0.5, py = py_start+i+0.5)`:
     - `dx = px - means_2d[g, 0]; dy = py - means_2d[g, 1]`
     - `power = -0.5 * (cov_inv[g, 0,0]*dx² + 2*cov_inv[g, 0,1]*dx*dy + cov_inv[g, 1,1]*dy²)`
     - `gw = std::exp(std::min(power, 0.0f))`
     - `alpha = std::clamp(opacities[g] * gw, 0.0f, 0.99f)`
     - `accum[i, j, c] += alpha * T[i, j] * colors[g, c]` for c in {0, 1, 2}
     - `T[i, j] *= (1 - alpha)`
   - After processing this Gaussian for the whole tile, if `max(T) < 1e-4` for the entire tile, break out of the for-idx loop.
6. Write `accum[i, j, c]` to `image[(py_start+i)*W*3 + (px_start+j)*3 + c]`.

## Performance notes (you can defer these to iter-005 / iter-008 BUT keep an eye on them)

- For iter-004 scope: **scalar inner loop** (one pixel at a time, three components). No SIMD, no microblock-major iteration. The goal is **correctness first** (PSNR ≥ 60 dB vs numpy reference); speed is iter-005's concern.
- A per-tile std::array<float, 3*32*32> on the STACK is ≤ 12 KB — cheap and avoids heap traffic. The transmittance T is another 32×32×4 = 4 KB.
- Inverse covariance precompute: do it ONCE per blend call over all M Gaussians, NOT per-tile. The numpy reference also does this.
- Early-term check `max(T) < 1e-4`: compute as a tile-wide reduction every Gaussian iteration. For iter-004 a simple linear scan is fine; iter-008 will replace this with per-microblock early-term.

## CpuCppBackend.blend (Python side)

```python
def blend(self, means_2d, covs_2d, colors, opacities,
          sorted_gaussian_ids, tile_ranges, image_height, image_width):
    img = self._mod.blend(
        means_2d.numpy(), covs_2d.numpy().reshape(-1, 4),
        colors.numpy(), opacities.numpy(),
        sorted_gaussian_ids.numpy(), tile_ranges.numpy(),
        int(image_height), int(image_width), 32,
    )
    return img.reshape(image_height, image_width, 3), {}
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

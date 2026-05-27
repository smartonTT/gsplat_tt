# iter-008 worker prompt

You are the iter-008 worker. iter-007 just landed the C++ `microblock_cull` (bit-identical to numpy on hero). iter-008 is the perf payoff iter: **port `alpha_blend_microblock` to C++** (microblock-major inner loop) and finally drop the blend stage from ~80 ms to ~20 ms median per frame.

## Read first

1. `/Users/smarton/dev/gsplat_tt_2/opt/microblock-cpu-spec.md`
2. `/Users/smarton/dev/gsplat_tt_2/opt/plan.md`
3. `/Users/smarton/dev/gsplat_tt_2/prompts/worker.md`
4. `/Users/smarton/dev/gsplat_tt_2/gsplat/rasterization.py` lines 600-684 (`alpha_blend_microblock` numpy reference)
5. `/Users/smarton/dev/gsplat_tt_2/src/gsplat_cpu/blend.{h,cpp}` (your per-tile scalar template — heavy reuse)
6. `/Users/smarton/dev/gsplat_tt_2/src/gsplat_cpu/microblock_cull.{h,cpp}` (style)
7. `/Users/smarton/dev/gsplat_tt_2/backends/cpu_cpp/{backend.py,pybind_module.cpp}`

## Iter spec

ITER: 008-cpp-alpha-blend-microblock
GOAL: Port `alpha_blend_microblock` to C++ in `src/gsplat_cpu/blend_microblock.{h,cpp}`. Inner per-microblock loop is scalar (one pixel at a time within each 8×4 microblock); per-microblock T scope; per-microblock early-term. ThreadPool: one task per tile (inside each task, iterate the 32 microblocks sequentially — keeping per-tile temporal locality matters more than per-microblock parallelism at 4×8 granularity). `CpuCppBackend(microblock=True).blend(...)` now calls C++ blend_microblock instead of numpy. Layer 3 PSNR remains ≥ 60 dB (target same as iter-007: min 60.34 dB).

## Files to touch

- `src/gsplat_cpu/blend_microblock.{h,cpp}` — NEW
- `src/CMakeLists.txt` — add blend_microblock.cpp
- `backends/cpu_cpp/pybind_module.cpp` — add `m.def("blend_microblock", ...)` binding (mirrors `blend`'s binding pattern; share the global ThreadPool)
- `backends/cpu_cpp/backend.py` — replace `from gsplat.rasterization import alpha_blend_microblock` path with `self._mod.blend_microblock(...)` when `microblock=True`
- `tests/unit/test_blend_microblock.cpp` — NEW Catch2 tests (~6 cases)
- `tests/CMakeLists.txt` — add test_blend_microblock.cpp

## C++ signature

```cpp
// src/gsplat_cpu/blend_microblock.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct BlendResult {  // can be the same one as in blend.h; share if convenient
    std::vector<float> image;  // H * W * 3 fp32
};

BlendResult blend_microblock(
    const float*   means_2d,            // M * 2
    const float*   covs_2d,             // M * 4 (a, b, b, c)
    const float*   colors,              // M * 3
    const float*   opacities,           // M
    const int64_t* mb_header,           // num_tiles * 32 * 2 (offset, count)
    const int64_t* mb_stream,           // L_prime
    std::size_t M, std::size_t L_prime,
    int image_height, int image_width,
    int tile_size,                      // 32
    ThreadPool& pool
);

} // namespace gsplat_cpu
```

## Algorithm — mirror numpy exactly

Per task (one tile):
1. tx, ty, py_tile, px_tile compute (same as blend.cpp).
2. Per-tile inverse covariance NOT needed up front — `microblock_cull` already filtered the (g, m) pairs. But you could still precompute `cov_inv` once across all M and read it per-Gaussian inside the inner loop (saves recomputing det/inv). Numpy precomputes globally (line 619-625); match. So: precompute `cov_inv[M*3]` once outside the per-tile parallel loop, on the main thread, before dispatching tasks.
3. Per tile, iterate `m in 0..31`:
   - `off = mb_header[tile_id*32 + m].offset, cnt = mb_header[tile_id*32 + m].count`
   - if cnt == 0, continue.
   - `mb_ox = (m & 3) * 8`, `mb_oy = (m >> 2) * 4`
   - `py_start = py_tile + mb_oy`, `px_start = px_tile + mb_ox`
   - `py_end = min(py_start + 4, image_height)` — edge tiles get clipped
   - `px_end = min(px_start + 8, image_width)`
   - if `py_start >= py_end || px_start >= px_end` continue (entirely off-image)
   - `mb_h = py_end - py_start`, `mb_w = px_end - px_start`
   - Allocate `T[mb_h * mb_w]` initialized to 1.0f and `accum[mb_h * mb_w * 3]` initialized to 0.0f. Both on the stack — max 4*8*4 = 128 bytes for T, 4*8*3*4 = 384 bytes for accum.
   - For `idx in [off, off+cnt)`:
     - `g = mb_stream[idx]`
     - For each pixel (i, j) in mb_h × mb_w:
       - `px = px_start + j + 0.5f`, `py = py_start + i + 0.5f`
       - `dx = px - means_2d[g*2]`, `dy = py - means_2d[g*2+1]`
       - Use precomputed cov_inv[g] = (ci_a, ci_b, ci_c)
       - `power = -0.5f * (ci_a*dx*dx + 2.0f*ci_b*dx*dy + ci_c*dy*dy)`
       - `gw = expf(min(power, 0.0f))`
       - `alpha = min(opacities[g] * gw, 0.99f)` (upper-only clamp; same as blend.cpp)
       - `t = T[i*mb_w + j]; at = alpha * t`
       - `accum[(i*mb_w + j)*3 + c] += at * colors[g*3 + c]` for c in {0, 1, 2}
       - `T[i*mb_w + j] = t * (1.0f - alpha)`
     - After processing this Gaussian: compute `tmax = max over T[]`. If `tmax < 0.0001f`, `break` out of the idx loop.
   - Write the microblock's `accum` into the output image: `image[(py_start+i)*W*3 + (px_start+j)*3 + c] = accum[(i*mb_w + j)*3 + c]`.

## Critical fidelity notes (must match numpy bit-by-bit through fp accumulation noise)

- **Pixel centers `+0.5f`** (same as iter-004 blend)
- **Alpha clamp upper-only `min(..., 0.99f)`** (no lower bound, same as iter-004)
- **Early-term threshold `0.0001f`** (same literal)
- **Per-microblock T scope** — T initialized to 1.0 PER microblock (not per tile). This is the key delta from iter-004's blend.
- **Order of fp ops** — `ci_a*dx*dx + 2.0f*ci_b*dx*dy + ci_c*dy*dy` in that exact order. Power then min then exp.
- **No alpha-skip** — even sub-floor contributions still update T (the cull already did the dropping).
- **fp32 throughout** — no fp64. (Per-microblock T accumulates over ≤ ~50 gaussians at the worst point, well within fp32 range.)

The PSNR target is identical to iter-007: min 60.34 dB, mean 63.26 dB. If you see significantly different numbers (especially LOWER), you have a bug.

## Pybind binding

```cpp
// pybind_module.cpp
static gsplat_cpu::ThreadPool& global_blend_mb_pool() {
    static gsplat_cpu::ThreadPool pool(0);
    return pool;
}

m.def("blend_microblock", [](
    py::array_t<float> means_2d,
    py::array_t<float> covs_2d,
    py::array_t<float> colors,
    py::array_t<float> opacities,
    py::array_t<int64_t> mb_header,
    py::array_t<int64_t> mb_stream,
    int image_height,
    int image_width,
    int tile_size
) {
    // ... extract pointers/shapes, call gsplat_cpu::blend_microblock, return (H, W, 3) array
});
```

You can reuse the same global pool as `blend` if convenient — they don't run concurrently within one frame.

## CpuCppBackend.blend wiring

```python
def blend(self, means_2d, covs_2d, colors, opacities,
          sorted_gaussian_ids, tile_ranges, image_height, image_width):
    if not self._microblock:
        # existing iter-004 path
        return <existing>

    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    mb_header_flat, mb_stream, stats = self._mod.microblock_cull(
        np.ascontiguousarray(means_2d.detach().cpu().numpy(), np.float32),
        np.ascontiguousarray(covs_2d.detach().cpu().numpy().reshape(-1, 4), np.float32),
        np.ascontiguousarray(opacities.detach().cpu().numpy(), np.float32),
        np.ascontiguousarray(sorted_gaussian_ids.detach().cpu().numpy(), np.int64),
        np.ascontiguousarray(tile_ranges.detach().cpu().numpy(), np.int64),
        int(tiles_x), int(tiles_y), 32, float(self._mb_contrib_floor),
    )
    # iter-008: C++ blend now consumes the cull output directly
    img = self._mod.blend_microblock(
        np.ascontiguousarray(means_2d.detach().cpu().numpy(), np.float32),
        np.ascontiguousarray(covs_2d.detach().cpu().numpy().reshape(-1, 4), np.float32),
        np.ascontiguousarray(colors.detach().cpu().numpy(), np.float32),
        np.ascontiguousarray(opacities.detach().cpu().numpy(), np.float32),
        np.ascontiguousarray(mb_header_flat, np.int64),
        np.ascontiguousarray(mb_stream, np.int64),
        int(image_height), int(image_width), 32,
    )
    return img, {
        "microblock_drop_pct": stats["drop_pct"],
        "microblock_work_reduction_pct": stats["work_reduction_pct"],
    }
```

## Catch2 tests in `tests/unit/test_blend_microblock.cpp`

At least 6 tests:
1. Single tile, single Gaussian at tile center (mean=(16, 16)), large opacity, mb_stream has the Gaussian in microblock m that contains pixel (16, 16) — output pixel near center is approximately the Gaussian's color.
2. Empty tile (all mb_header counts zero) → output region is all zeros.
3. Edge tile (tile that straddles image boundary) → only in-bounds region is written.
4. Two Gaussians with different depths in one microblock, both opacity 1.0 → front-to-back compositing: nearer Gaussian's color dominates.
5. Per-microblock early-term: stuff a microblock with 100 opaque Gaussians at same position → only the first few iterations contribute, T_max reaches 0 quickly.
6. Hero-fixture round-trip equivalence: load `microblock_cull_inputs.npz` + `microblock_cull_outputs.npz`, run C++ `blend_microblock`, compare to numpy `alpha_blend_microblock` output (computed via the existing pytest path). Must match within PSNR ≥ 60 dB.

For test 6, you can either:
- Run numpy alpha_blend_microblock inline in the Catch2 test via cnpy + a baked numpy reference image (add `blend_microblock_output.npy` to the hero fixtures), OR
- Skip the test 6 inside Catch2 and rely on Layer 2 verify_stage instead

If you can dump a numpy reference image once into the fixtures, the inline Catch2 test is cleaner. Add `blend_microblock_output.npy` to the fixtures (re-using your `dump_microblock_fixture.py` script — extend it).

## Gates

```bash
source scripts/_env.sh

# Layer 1
cmake --build build -j && ctest --test-dir build --output-on-failure -j
# expect 34 + ~6 new = 40/40

# Layer 2 — still all stages
for s in project tile_assign sort blend; do
  $LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage $s
done
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp_mb --stage microblock_cull
# Optional new: --stage blend_microblock if you wire it.

# Layer 3
rm -rf /tmp/iter008
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp_mb --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter008 --warmup 1

$LOCAL_PY -c "
from PIL import Image; import numpy as np; from pathlib import Path; import json
ms=[]
for f in sorted(Path('benchmarks/reference_v2').glob('*.png')):
    a=np.asarray(Image.open(f).convert('RGB'),np.float64)/255
    b=np.asarray(Image.open(Path('/tmp/iter008')/f.name).convert('RGB'),np.float64)/255
    mse=float(np.mean((a-b)**2))
    p=float('inf') if mse<=0 else 10*np.log10(1/mse)
    ms.append((f.stem,p))
ms.sort(key=lambda x:x[1])
print(f'PSNR min={ms[0][1]:.2f}  mean={sum(p for _,p in ms)/len(ms):.2f}  max={ms[-1][1]:.2f}  (gate 60 dB)')
rows=[json.loads(l) for l in open('/tmp/iter008/timing.jsonl').read().splitlines() if l.strip()]
total=sum(r['total_ms'] for r in rows)
blend=sorted(r.get('blend',0) for r in rows)
print(f'sum_total_ms={total:.1f}  blend_median={blend[len(blend)//2]:.2f}ms  blend_p95={blend[int(len(blend)*0.95)]:.2f}ms')
print(f'vs baseline 3257ms ({3257/total:.2f}x speedup)')
"
```

Expected: PSNR distribution same as iter-007 (min 60.34, mean 63.26 dB). Sum_total_ms should drop to ~1000-1500ms (3-4x speedup over per-tile baseline). Blend median should drop to ~20-30ms.

## BUDGET

5 build attempts. If you hit a PSNR cliff (say < 50 dB), the most likely cause is forgetting per-microblock T reset, OR writing into the wrong image offset (off-by-one in `(py_start+i)*W*3 + (px_start+j)*3 + c`). Diagnostic: dump `/tmp/blend_mb_dbg.npy` and compare to the numpy reference (`alpha_blend_microblock` on the same inputs).

## SUMMARY

```
SUMMARY: iter-008 status=PASS|FAIL ctest=N/M render30_min_psnr_dB=X.X sum_total_ms=Y.Y blend_median_ms=Z.Z speedup_vs_baseline=W.Wx work_reduction_pct=V.V
```

Plus 6-12 bullets. Do not commit. Begin.

# iter-002 worker prompt (draft — supervisor finalizes before dispatch)

You are the iter-002 worker for the gsplat_tt_2 sprint.

## Read these first (mandatory)
1. `/Users/smarton/dev/gsplat_tt_2/opt/plan.md`
2. `/Users/smarton/dev/gsplat_tt_2/prompts/worker.md`
3. `/Users/smarton/dev/gsplat_tt_2/gsplat/rasterization.py` lines 28-175 (`project_gaussians` — the algorithm spec you are porting)
4. `/Users/smarton/dev/gsplat_tt_2/gsplat/utils.py::build_covariance_3d` (the helper `project_gaussians` calls in step 1)
5. `/Users/smarton/dev/gsplat_tt_2/backends/cpu_cpp/backend.py` (the class you are extending)
6. `/Users/smarton/dev/gsplat_tt_2/scripts/verify_stage.py` (your Layer 2 gate)

## Iter spec

ITER: 002-cpp-project
GOAL: Port `gsplat.rasterization.project_gaussians` to a C++ function. CpuCppBackend.project must now call the C++ implementation instead of inheriting Backend.project's numpy default. tile_assign / sort / blend still delegate to numpy.

FILES TO TOUCH:
- `src/gsplat_cpu/types.h`            (new — minimal `Vec2f`, `Vec3f`, `Mat3f`, `Quatf` POD structs + helpers; nothing fancy, no Eigen)
- `src/gsplat_cpu/project.h`          (new — declares `project(...)` with raw float pointer inputs)
- `src/gsplat_cpu/project.cpp`        (new — implementation)
- `src/CMakeLists.txt`                (add project.cpp to the gsplat_cpu library)
- `backends/cpu_cpp/pybind_module.cpp` (add `m.def("project", ...)` exposing the C++ function to Python, returning numpy arrays via py::array_t)
- `backends/cpu_cpp/backend.py`       (override `project()` to call `self._mod.project(...)`)
- `tests/unit/test_project.cpp`       (new — Catch2 tests for the math primitives + a smoke test using the hero fixture)
- `tests/CMakeLists.txt`              (add test_project.cpp)

## C++ function signature

In `src/gsplat_cpu/project.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

struct ProjectResult {
    // All vectors are sized M (visible Gaussians), packed in input-Gaussian order
    // (i.e. valid_mask[i] tells you which input row was kept; outputs are in
    // the order kept-Gaussians appear in valid_mask).
    std::vector<float> means_2d;          // M * 2,  layout [x, y, x, y, ...]
    std::vector<float> covs_2d;           // M * 4,  layout [a, b, b, c, ...]  (full 2x2)
    std::vector<float> depths;            // M
    std::vector<float> radii;             // M * 2,  layout [rx, ry, ...]
    std::vector<uint8_t> valid_mask;      // N, true/false per input
};

ProjectResult project(
    const float* means,        // N * 3   (world space xyz)
    const float* scales,       // N * 3
    const float* rotations,    // N * 4   quaternion (w, x, y, z)
    const float* extrinsics,   // 16     row-major 4x4 (world -> camera)
    const float* intrinsics,   // 9      row-major 3x3
    std::size_t N,
    int image_height,
    int image_width,
    const float* opacities,    // nullable; if non-null, opacity-aware k and min_opacity cull
    float min_opacity          // default 1.0f/255.0f
);

} // namespace gsplat_cpu
```

## Algorithmic spec (mirror numpy exactly)

Read `gsplat/rasterization.py::project_gaussians`. The C++ port must reproduce every step:

1. **Step 1 — Build 3D covariance per Gaussian.**
   For each i: quaternion `(w, x, y, z)` → rotation matrix R3 (3x3). Then
   `Σ_3D = R3 · diag(scales²) · R3ᵀ`. Use the same quaternion-to-matrix formula as `gsplat.utils.build_covariance_3d` (read that function).

2. **Step 2 — Camera-space transform.**
   `t = means[i] @ R_extr^T + t_extr` where `R_extr = extrinsics[:3, :3]`, `t_extr = extrinsics[:3, 3]`. Same exact math as numpy.

3. **Step 3 — Frustum near-plane cull.**
   `valid[i] = (t.z > 0.2f)`. Use `0.2f` exactly (numpy's `near = 0.2`).

4. **Step 4 — Screen-space center.**
   `means_2d.x = fx * t.x / t.z + cx; means_2d.y = fy * t.y / t.z + cy`.

5. **Steps 5+6 — Jacobian + 2D covariance.**
   Jacobian (2x3):
   ```
   J = [[fx/tz, 0,     -fx*tx/tz²],
        [0,     fy/tz, -fy*ty/tz²]]
   ```
   Then `Σ_cam = R_extr · Σ_3D · R_extrᵀ`, `Σ_2D = J · Σ_cam · Jᵀ`. Finally add the low-pass filter: `Σ_2D[0,0] += 0.3f; Σ_2D[1,1] += 0.3f`.

6. **Step 7 — Per-axis AABB radii with opacity-aware k.**
   `a = Σ_2D[0,0]; c = Σ_2D[1,1]`. If `opacities` non-null:
   ```
   arg  = max(opacities[i] * (255.0f / 15.0f), 1.0f)
   k    = min(sqrtf(2.0f * logf(arg)), 3.0f)
   ```
   else `k = 3.0f`. Then `rx = ceilf(k * sqrtf(max(a, 0))); ry = ceilf(k * sqrtf(max(c, 0)))`. Match `np.ceil` semantics (round toward +inf).

7. **Screen-edge cull, max-radius cull, opacity cull** — exactly as numpy.

## Output layout in CpuCppBackend.project (Python side)

`backends/cpu_cpp/backend.py::CpuCppBackend.project` must return (after the numpy/torch wrapping):

- `means_2d`: torch.float32 (M, 2)
- `covs_2d`:  torch.float32 (M, 2, 2)  — reshape the C++ 4-stride per row into 2x2
- `depths`:   torch.float32 (M,)
- `radii`:    torch.float32 (M, 2)
- `valid_mask`: torch.bool   (N,) — pack the C++ uint8 mask into bool

Same dtype, shape, and ORDER as `gsplat.rasterization.project_gaussians` returns.

## Catch2 unit tests (Layer 1)

1. `test_project.cpp` — quaternion→rotation: hand-crafted unit quaternions (identity, 90° about each axis); compare R3 to expected; max-abs diff < 1e-6.
2. `test_project.cpp` — Jacobian against analytic formula at handcrafted (tx, ty, tz).
3. `test_project.cpp` — full project on a 3-Gaussian synthetic input (means at simple coordinates, fx=fy=1, principal point at origin, identity extrinsics) — verify means_2d, depths, radii match by-hand calculation to ~1e-5.
4. `test_project.cpp` — opacity-aware k: opacities = {0.01, 0.5, 1.0} produce k = {clamp(sqrt(2*ln(0.17)), max=3)=…, sqrt(2*ln(8.5)), min(sqrt(2*ln(17)), 3)} respectively.
5. `test_project.cpp` (optional but recommended) — load the hero fixture inputs (`tests/fixtures/hero/project_inputs.npz` requires a small npz reader; if none, hardcode a few representative Gaussian inputs).

## Layer 2 gate

```bash
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage project
```

All five output checks must `pass:true` (means_2d 1e-4, covs_2d 1e-5, depths 1e-5, radii 1e-3, valid_mask exact).

## Layer 3 gate (smoke)

```bash
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter002 --warmup 0
```

The 30 PNGs from cpu_cpp must match the 30 PNGs in `benchmarks/reference_v2/` to PSNR ≥ 60 dB per view. Spot-check with:
```python
from PIL import Image; import numpy as np
for view in ['hero', 'orbit_220', 'chal_close', 'chal_far']:
    a = np.asarray(Image.open(f'benchmarks/reference_v2/{view}.png').convert('RGB'), np.float64) / 255
    b = np.asarray(Image.open(f'/tmp/iter002/{view}.png').convert('RGB'), np.float64) / 255
    mse = float(np.mean((a - b)**2))
    psnr = 10 * np.log10(1.0/mse) if mse > 0 else float('inf')
    print(f'{view}: psnr={psnr:.1f} dB')
```

Since tile_assign/sort/blend still use numpy, ONLY project's fp accumulation order can differ — should be near-infinite PSNR in practice.

## BUDGET

5 attempts to get Layer 1 + Layer 2 + Layer 3 all green. If after 5 attempts the C++ project gives wrong results vs numpy, surface "BLOCKED ON: <specific stage failing>" and revert the working tree to HEAD with `git checkout -- .`.

## SUMMARY line (parsed by supervisor)

```
SUMMARY: iter-002 status=PASS|FAIL build=ok|fail ctest=N/M verify_project=psnr_eq_inf|psnr=X.X|fail render30_min_psnr_dB=X.X
```

# iter-009 worker prompt

You are the iter-009 worker. The cpu_cpp_mb pipeline now does sum30=1706ms with blend at 26ms median. The new dominant chunks are:

| stage | median ms | share |
|------:|----------:|------:|
| project | 17.1 | 30% |
| blend | 26.0 | 46% |
| sort | 9.2 | 16% |
| tile_assign | 4.0 | 7% |

iter-009 attacks **project**. Sub-profile: project_prepare 4 ms (C++), build_covariance_3d 4 ms (torch), R cov3d R.T 4 ms (torch.matmul), J cov_cam J.T 0.7 ms (torch.bmm), project_finalize 4 ms (C++). **~9 ms of project lives in the Python/torch glue path.** Port it to C++ to drop project to ~5–7 ms median.

## Read first
1. `/Users/smarton/dev/gstt2/opt/plan.md`
2. `/Users/smarton/dev/gstt2/prompts/worker.md`
3. `/Users/smarton/dev/gstt2/src/gsplat_cpu/project.{h,cpp}` (existing C++ project)
4. `/Users/smarton/dev/gstt2/backends/cpu_cpp/backend.py` (current orchestration — note `torch.matmul` and `torch.bmm` lines in `project()`)
5. `/Users/smarton/dev/gstt2/gsplat/utils.py` (`quat_to_rotation_matrix`, `build_covariance_3d` — your contract)
6. `/Users/smarton/dev/gstt2/gsplat/rasterization.py` lines 1-177 (`project_gaussians` numpy reference — your top-level contract)
7. `/Users/smarton/dev/gstt2/scripts/verify_stage.py` (Layer 2 gate; project tolerance is 0)

## Iter spec

ITER: 009-cpp-project-full
GOAL: Move `build_covariance_3d` (R, S, RS, RSRS^T) AND `cov_cam = R @ cov3d @ R^T` AND `cov2d = J @ cov_cam @ J^T` from the Python wrapper into C++ `project_prepare` (or a refactored `project_full`). The Python wrapper just calls one C++ function and gets `(means_2d, covs_2d, depths, radii, valid_mask)`. Use **fp64 intermediates** in the 3×3 / 2×2 matmuls so the result is **bit-identical** to the existing numpy fixture (`verify_stage --stage project` keeps `max_abs_diff = 0`).

## Files to touch

- `src/gsplat_cpu/project.{h,cpp}` — add `project_full(...)` that runs the entire stage. Keep the existing `project_prepare`/`project_finalize` symbols if you want; they're tiny and the existing Catch2 tests can keep using them. The new `project_full` is what the Python wrapper calls.
- `backends/cpu_cpp/pybind_module.cpp` — add `m.def("project_full", ...)` binding
- `backends/cpu_cpp/backend.py` — simplify `project()` to call `self._mod.project_full(...)` (remove `build_covariance_3d`, `torch.matmul`, `torch.bmm`)
- `tests/unit/test_project.cpp` — add at least 2 new tests: `project_full` on a 1-Gaussian fixture (verify cov2d matches by-hand math); `project_full` on the hero fixture (load `project_inputs.npz`, run, verify cov2d matches `project_outputs.npz` with `max_abs_diff == 0`).
- `tests/CMakeLists.txt` — no change unless you add a new test file
- `src/CMakeLists.txt` — no change if you only edit project.{h,cpp}

## Algorithm spec — `project_full`

Mirror the existing `CpuCppBackend.project()` orchestration but entirely in C++:

```cpp
struct ProjectFullResult {
    std::vector<float> means_2d;       // N * 2
    std::vector<float> covs_2d;        // N * 4 (a, b, b, c) — to match downstream
    std::vector<float> depths;         // N
    std::vector<float> radii;          // N * 2
    std::vector<uint8_t> valid_mask;   // M  (bool, 1 byte per element)
    int N;                              // visible count == sum(valid_mask)
};

ProjectFullResult project_full(
    const float* means,        // M * 3
    const float* scales,       // M * 3
    const float* rotations,    // M * 4 (q0, q1, q2, q3)
    const float* extrinsics,   // 4 * 4 (row-major; numpy stores it that way)
    const float* intrinsics,   // 3 * 3
    std::size_t M,
    int image_height,
    int image_width,
    const float* opacities,    // M; pass nullptr for no opacity-aware k
    float contrib_floor        // default 1.0f/255.0f
);
```

### Per-Gaussian pipeline (all fp64 intermediates, fp32 output):

For each Gaussian g in 0..M:
1. **Camera transform** (you can use Accelerate `cblas_sgemm` to batch this; iter-002 did): `mean_cam = R_extr @ mean + t_extr` where R_extr = upper-left 3×3 of extrinsics, t_extr = column 3.
2. **Near-plane cull**: if `mean_cam.z < near` (use existing constant from project.cpp), mark `valid_mask[g] = 0`, skip the rest.
3. **Screen-space xy**: `mean_2d.x = fx * mean_cam.x / mean_cam.z + cx; mean_2d.y = fy * mean_cam.y / mean_cam.z + cy`. Off-screen cull on radii-padded bounds done later.
4. **Jacobian J** (2×3, fp64 internally):
   ```
   inv_z = 1/mean_cam.z
   J[0,0] = fx*inv_z; J[0,1] = 0; J[0,2] = -fx*mean_cam.x*inv_z²
   J[1,0] = 0; J[1,1] = fy*inv_z; J[1,2] = -fy*mean_cam.y*inv_z²
   ```
5. **R quat→matrix** (3×3, fp64):
   ```
   R[0,0] = 1 - 2*(q2²+q3²); R[0,1] = 2*(q1*q2 - q0*q3); R[0,2] = 2*(q1*q3 + q0*q2);
   R[1,0] = 2*(q1*q2 + q0*q3); R[1,1] = 1 - 2*(q1²+q3²); R[1,2] = 2*(q2*q3 - q0*q1);
   R[2,0] = 2*(q1*q3 - q0*q2); R[2,1] = 2*(q2*q3 + q0*q1); R[2,2] = 1 - 2*(q1²+q2²);
   ```
6. **cov3d = R * diag(s²) * R^T** (3×3, fp64). Closed-form: `cov3d[i,j] = sum_k R[i,k]*s[k]²*R[j,k]`. Direct scalar form is ~12 mults per matrix entry → 12 × 6 = 72 mults per Gaussian (only 6 unique entries due to symmetry).
7. **cov_cam = R_extr * cov3d * R_extr^T** (3×3, fp64). Another 27 multiplies (or use 6-unique-entry form, 18 mults).
8. **cov2d = J * cov_cam * J^T** (2×2, fp64). 12 mults; symmetric.
9. **cov2d += diag(0.3, 0.3)** (the dilation term, matches existing code).
10. **Cast cov2d to fp32**, write into covs_2d[g * 4 + ...].
11. **Compute radii** from cov2d eigenvalues (existing project_finalize logic — port it inline).
12. **Set valid_mask[g] = 1** if all the bounds + near-plane + opacity-aware k checks pass.

### Why fp64 intermediates

torch.bmm + torch.matmul use fp64 reduction internally on CPU (the BLAS calls go through fp64 accumulators for fp32 inputs in Accelerate's sgemm). To match the numpy fixture (which uses torch.bmm via iter-002), the C++ must also accumulate in fp64. Direct scalar fp32 matmul drops the LSB and was the iter-002 worker's reason for delegating to torch.bmm in the first place. By using fp64 for ALL matmul intermediates inside project_full, we get the same result as torch + Accelerate.

Cost: ~140 fp64 mults per Gaussian × 233k Gaussians = 32 Mops. At 5 GFlop/s scalar fp64, that's ~6 ms. Add quat→R, Jacobian, near-plane cull, radii: total project ~5–7 ms. **Target: project drops from 17 ms to ~5 ms median.**

### Threading

Use the existing `gsplat_cpu::ThreadPool`. One task per chunk of `M / num_threads` Gaussians, since per-Gaussian work is small (~70 ns at 6 ms / 233k Gaussians) and you want to amortize task overhead.

Alternative: skip threading entirely — at 5 ms total for the stage, threading overhead might eat the savings. Try both and use whichever is faster. If you go single-threaded, document the choice in your summary.

## Gates

```bash
source scripts/_env.sh

# Layer 1
cmake --build build -j && ctest --test-dir build --output-on-failure -j
# expect 40 existing + ~2 new = 42/42

# Layer 2 — project tolerance is 0 (exact equality)
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage project
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage tile_assign
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage sort
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage blend
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp_mb --stage microblock_cull

# Layer 3
rm -rf /tmp/iter009
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp_mb --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter009 --warmup 1

$LOCAL_PY -c "
from PIL import Image; import numpy as np; from pathlib import Path; import json
ms=[]
for f in sorted(Path('benchmarks/reference_v2').glob('*.png')):
    a=np.asarray(Image.open(f).convert('RGB'),np.float64)/255
    b=np.asarray(Image.open(Path('/tmp/iter009')/f.name).convert('RGB'),np.float64)/255
    mse=float(np.mean((a-b)**2))
    p=float('inf') if mse<=0 else 10*np.log10(1/mse)
    ms.append((f.stem,p))
ms.sort(key=lambda x:x[1])
print(f'PSNR min={ms[0][1]:.2f}  mean={sum(p for _,p in ms)/len(ms):.2f}  max={ms[-1][1]:.2f}  (gate 60 dB)')
rows=[json.loads(l) for l in open('/tmp/iter009/timing.jsonl').read().splitlines() if l.strip()]
total=sum(r['total_ms'] for r in rows)
print(f'sum_total_ms={total:.1f}  vs prev best 1706  ({1706/total:.2f}x speedup)')
for k in ('project','tile_assign','sort','blend'):
    vs = sorted(r.get(k,0) for r in rows)
    print(f'  {k:12s} median={vs[len(vs)//2]:6.2f}ms')
"
```

Expected: PSNR distribution unchanged (min 60.34 dB). project median drops from 17ms to ~5-7ms. Sum_total_ms drops to ~1300-1400ms (~1.2-1.3x speedup over iter-008).

## BUDGET

5 build attempts. If verify_stage --stage project fails at `max_abs_diff > 0`:
1. Print which output array fails (means_2d / covs_2d / depths / radii / valid_mask) and the max-diff magnitude.
2. If covs_2d differs by 1-2 ULP: you have fp64-vs-fp32 mismatch somewhere in the matmul chain. Trace each intermediate (R, cov3d, cov_cam, cov2d) through fp64 and verify against Python's `torch.matmul(torch.matmul(R, cov3d), R.T)` for a hand-picked Gaussian.
3. If means_2d / depths differ: bug in the extrinsics indexing (column 3 vs row 3 — iter-002 hit this).
4. If valid_mask differs: opacity-aware-k threshold drift due to fp precision; relax `contrib_floor=1.0f/255.0f` if needed (must match Python's call).

If you're close but not zero-diff, and the iter-008 baseline PSNR holds at ≥ 60 dB on the 30-view render, you can fall back to keeping the matmul chain in Python (status quo). In that case, surface `BLOCKED ON: bit-identical to torch.bmm requires Accelerate cblas_sgemm batching — Python keeps ~9ms`. Don't ship a regression that breaks Layer 2.

## SUMMARY

```
SUMMARY: iter-009 status=PASS|FAIL ctest=N/M verify_project=pass|fail render30_min_psnr_dB=X.X sum_total_ms=Y.Y project_median_ms=Z.Z
```

Plus 6-12 bullets. Do not commit.

Begin.

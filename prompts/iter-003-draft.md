# iter-003 worker prompt (draft)

You are the iter-003 worker for the gsplat_tt_2 sprint. iter-002 just landed —
project_gaussians is now in C++. Your job is to port BOTH `get_tile_assignments`
(including the per-pair Mahalanobis cull) AND `sort_and_bin` to C++. After this
iter, only `alpha_blend` is left in numpy.

## Read first (mandatory)

1. `/Users/smarton/dev/gsplat_tt_2/opt/plan.md`
2. `/Users/smarton/dev/gsplat_tt_2/prompts/worker.md`
3. `/Users/smarton/dev/gsplat_tt_2/gsplat/rasterization.py` lines 178-371 (`get_tile_assignments` + `sort_and_bin`)
4. `/Users/smarton/dev/gsplat_tt_2/scripts/verify_stage.py` (Layer 2 gate for both `tile_assign` and `sort`)
5. `/Users/smarton/dev/gsplat_tt_2/src/gsplat_cpu/project.{h,cpp}` (from iter-002 — style template)
6. `/Users/smarton/dev/gsplat_tt_2/backends/cpu_cpp/pybind_module.cpp` (where to add bindings)
7. `/Users/smarton/dev/gsplat_tt_2/backends/cpu_cpp/backend.py` (where to add `tile_assign` and `sort` overrides)

## Iter spec

ITER: 003-cpp-tile-assign-cull-sort
GOAL: Port `gsplat.rasterization.get_tile_assignments` (AABB tile range, repeat-interleave to per-pair, per-pair Mahalanobis cull) AND `gsplat.rasterization.sort_and_bin` (composite-key sort by (tile_id, depth), per-tile range build) to C++. CpuCppBackend.tile_assign and CpuCppBackend.sort now call the C++ implementations. After this iter, the alpha_blend stage is the only one still on the numpy fallback.

## Files to touch

- `src/gsplat_cpu/tile_assign.h` / `.cpp`  (new — `tile_assign(...)`)
- `src/gsplat_cpu/sort.h` / `.cpp`         (new — `sort(...)`)
- `src/CMakeLists.txt`                     (add the two new .cpp files to gsplat_cpu)
- `backends/cpu_cpp/pybind_module.cpp`     (add `m.def("tile_assign", ...)` and `m.def("sort", ...)`)
- `backends/cpu_cpp/backend.py`            (override `tile_assign` and `sort`)
- `tests/unit/test_tile_assign.cpp`        (new — Catch2 tests)
- `tests/unit/test_sort.cpp`               (new — Catch2 tests)
- `tests/CMakeLists.txt`                   (add the two new test files)

## C++ function signatures

In `src/gsplat_cpu/tile_assign.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

struct TileAssignResult {
    std::vector<int64_t>  gaussian_ids;        // (P,)
    std::vector<int64_t>  tile_ids;            // (P,)
    std::vector<int64_t>  tiles_per_gaussian;  // (M,)
};

TileAssignResult tile_assign(
    const float* means_2d,            // M * 2
    const float* radii,               // M * 2  (rx, ry per row)
    std::size_t M,
    int image_height,
    int image_width,
    int tile_size,                    // 32 in practice
    const float* covs_2d,             // nullable; M * 4 (a, b, b, c) when provided
    const float* opacities,           // nullable; M
    float contrib_floor               // default 15.0f/255.0f when caller wants the per-pair cull
);

} // namespace gsplat_cpu
```

In `src/gsplat_cpu/sort.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

struct SortResult {
    std::vector<int64_t> sorted_gaussian_ids;   // (P,)
    std::vector<int64_t> tile_ranges;           // num_tiles * 2  (start, end pairs, row-major)
};

SortResult sort_and_bin(
    const int64_t* gaussian_ids,      // P
    const int64_t* tile_ids,          // P
    const float*   depths,            // M (indexed by gaussian_ids[i])
    std::size_t P,
    std::size_t M,
    int tiles_x,
    int tiles_y
);

} // namespace gsplat_cpu
```

## Algorithm spec — tile_assign

Mirror `get_tile_assignments` line-for-line:

1. AABB tile range per Gaussian:
   - `tile_min_x = clamp(int((means_2d.x - rx) / tile_size), 0, tiles_x - 1)`
   - `tile_max_x = clamp(int((means_2d.x + rx) / tile_size), 0, tiles_x - 1)`  (same for y)
   - `widths = tile_max_x - tile_min_x + 1`; same for heights
   - `tiles_per_gaussian[m] = widths * heights`

2. Repeat-interleave to a flat pair list of length `P = sum(tiles_per_gaussian)`.
3. Flat tile id per pair: `tile_ids[p] = (tile_min_y[m] + dy) * tiles_x + (tile_min_x[m] + dx)` where (dx, dy) enumerate the AABB grid (row-major).
4. Per-pair Mahalanobis cull (only when `covs_2d` and `opacities` are both non-null):
   - For each pair: `a = covs_2d[g, 0, 0], b = covs_2d[g, 0, 1], c = covs_2d[g, 1, 1], det = max(a*c - b*b, 1e-6)`
   - `tx_tile = (tile_id % tiles_x) * tile_size; ty_tile = (tile_id / tiles_x) * tile_size`
   - `cx = clamp(px, tx_tile, tx_tile + tile_size); cy = clamp(py, ty_tile, ty_tile + tile_size)`
   - `dx_c = cx - px; dy_c = cy - py`
   - `m2 = (c*dx_c² - 2*b*dx_c*dy_c + a*dy_c²) / det`
   - keep iff `opacities[g] * std::exp(-0.5 * m2) >= contrib_floor`
5. After culling, recompute `tiles_per_gaussian` as a length-M zero-initialized vector, then `++tiles_per_gaussian[gaussian_ids[i]]` for each surviving pair. Equivalent to `torch.bincount(gaussian_ids, minlength=M)` — Gaussians with no surviving pairs MUST keep their slot at 0 (do not skip them in the output).

NB on dtypes: numpy returns int64 for all three arrays (mixed-type promotion via `arange(P).long()`). The C++ result MUST also be int64 so `verify_stage --stage tile_assign` value-compares cleanly against the int64 fixture.

## Algorithm spec — sort

Mirror `sort_and_bin` exactly:

1. Compose `int64` sort key per pair: `key[i] = (int64(tile_ids[i]) << 32) | uint32_bits(depths[gaussian_ids[i]])`.
   - Use `std::bit_cast<uint32_t, float>` (C++20) or memcpy aliasing. The float bit pattern for positive floats is monotonic in the float value — that's why we can sort fp32-depth as int.
   - All visible depths are > 0.2 (near plane) so the sign-bit is always 0; no special handling needed.

2. Sort the index permutation by key. Use `std::iota` + `std::sort` on a `std::vector<size_t>` index array with a custom comparator on the precomputed key array (don't re-derive the key inside the comparator — that wastes ~30% on stitch).

3. `sorted_gaussian_ids[i] = gaussian_ids[perm[i]]`. Also produce `sorted_tile_ids[i] = tile_ids[perm[i]]` as a scratch buffer for the range build.

4. Build `tile_ranges`: a length-`(tiles_x*tiles_y)*2` int64 array, zero-initialized. Walk the sorted `tile_ids`; when the tile changes from `t_prev` to `t_curr`, set `tile_ranges[t_prev*2 + 1] = i` and `tile_ranges[t_curr*2 + 0] = i`. Also set `tile_ranges[sorted_tile_ids[0]*2 + 0] = 0` and `tile_ranges[sorted_tile_ids[P-1]*2 + 1] = P`. Tiles with no pairs keep their default `(0, 0)` — that's correct because numpy uses `torch.zeros(num_tiles, 2)`.

NB on dtypes: numpy returns int64 for both `sorted_gaussian_ids` and `tile_ranges`. C++ must match. Pybind11 will pass numpy arrays as `py::array_t<int64_t>` to `tile_assign(...)` (since iter-003's `tile_assign` now returns int64), so the downstream `sort()` binding receives int64 buffers naturally.

## CpuCppBackend.tile_assign / .sort (Python side)

```python
def tile_assign(self, means_2d, radii, image_height, image_width,
                tile_size=32, covs_2d=None, opacities=None,
                contrib_floor=15.0/255.0, sub_timings=None):
    gids, tids, tpg = self._mod.tile_assign(
        means_2d.numpy(), radii.numpy(), int(image_height), int(image_width),
        int(tile_size),
        covs_2d.numpy() if covs_2d is not None else None,
        opacities.numpy() if opacities is not None else None,
        float(contrib_floor),
    )
    return torch.from_numpy(gids), torch.from_numpy(tids), torch.from_numpy(tpg)

def sort(self, gaussian_ids, tile_ids, depths, tiles_x, tiles_y, sub_timings=None):
    sgids, tranges = self._mod.sort(
        gaussian_ids.numpy(), tile_ids.numpy(), depths.numpy(),
        int(tiles_x), int(tiles_y),
    )
    return torch.from_numpy(sgids), torch.from_numpy(tranges)
```

Match the numpy signature exactly (sub_timings is accepted but ignored — we'll wire C++-side timing in a later iter).

## Catch2 unit tests

`test_tile_assign.cpp`:
1. Single Gaussian at tile center, radius < tile_size → exactly 1 tile assigned.
2. Single Gaussian straddling 4 tiles → exactly 4 tiles assigned.
3. Per-pair Mahalanobis cull: hand-crafted Gaussian far enough from a tile that exp(-0.5·m²) · ω < contrib_floor → that pair dropped.
4. `tiles_per_gaussian.sum() == gaussian_ids.size()` invariant.

`test_sort.cpp`:
1. Three pairs in the same tile with depths {0.5, 0.3, 0.7} → sorted order has depths {0.3, 0.5, 0.7}.
2. Pairs across 3 tiles → tile_ranges has 3 non-zero ranges with correct (start, end) boundaries.
3. Tiles with no pairs → tile_ranges[t] == (0, 0).
4. Composite key correctness: handcrafted test that two pairs with the same tile_id sort by depth, NOT by gaussian_id.

## Layer gates

```bash
# Layer 1
ctest --test-dir build --output-on-failure -j

# Layer 2 (exact integer match required for both stages)
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage tile_assign
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage sort

# Layer 3
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter003 --warmup 0
# PSNR check: min PSNR vs benchmarks/reference_v2/ must be >= 60 dB
```

## BUDGET

5 build attempts. If stuck after 5, `git checkout -- .` and surface BLOCKED ON line.

## SUMMARY

```
SUMMARY: iter-003 status=PASS|FAIL build=ok|fail ctest=N/M verify_tile_assign=pass|fail verify_sort=pass|fail render30_min_psnr_dB=X.X
```

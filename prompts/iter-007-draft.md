# iter-007 worker prompt

You are the iter-007 worker for the gstt2 sprint. iter-006 just landed the numpy spec for per-microblock ellipse culling at `mb_contrib_floor = 1/16384`. Your job: **port `microblock_cull` to C++**, producing bit-identical `(mb_header, mb_stream)` output. The microblock-major C++ blend is iter-008; for iter-007 you wire C++ cull + numpy blend so we can prove the cull is correct in isolation.

## Read first (mandatory)

1. `/Users/smarton/dev/gstt2/opt/microblock-cpu-spec.md` — algorithm contract
2. `/Users/smarton/dev/gstt2/opt/plan.md`
3. `/Users/smarton/dev/gstt2/prompts/worker.md`
4. `/Users/smarton/dev/gstt2/gsplat/rasterization.py` lines 501-597 (`microblock_cull` numpy reference — your contract)
5. `/Users/smarton/dev/gstt2/src/gsplat_cpu/{tile_assign,sort,project,blend}.{h,cpp}` — style template
6. `/Users/smarton/dev/gstt2/backends/cpu_cpp/{pybind_module.cpp,backend.py}` — binding + override pattern
7. `/Users/smarton/dev/gstt2/scripts/verify_stage.py` — Layer 2 gate (you will add a `microblock_cull` stage)

## Iter spec

ITER: 007-cpp-microblock-cull
GOAL: Port `microblock_cull` to C++ in `src/gsplat_cpu/microblock_cull.{h,cpp}`. Output `(mb_header, mb_stream, stats)` is **bit-identical** to the numpy reference. Wire a `microblock=True` constructor flag on `CpuCppBackend` that, when set, runs the C++ cull after sort and the **numpy** `alpha_blend_microblock` for the blend. Register a top-level `"cpu_cpp_mb"` backend that constructs `CpuCppBackend(microblock=True)`. Layer 2 verify_stage gets a new "microblock_cull" runner that checks C++ vs numpy on the hero fixture.

## Files to touch

- `src/gsplat_cpu/microblock_cull.{h,cpp}` — NEW
- `src/CMakeLists.txt` — add microblock_cull.cpp to gsplat_cpu
- `backends/cpu_cpp/pybind_module.cpp` — add `m.def("microblock_cull", ...)` binding
- `backends/cpu_cpp/backend.py` — add `microblock: bool = False` to `CpuCppBackend.__init__`; when True, override `blend()` to do C++ cull + numpy mb_blend
- `backends/__init__.py` — register `"cpu_cpp_mb"` → `lambda: CpuCppBackend(microblock=True)` (or however your registry handles kwargs)
- `tests/unit/test_microblock_cull.cpp` — Catch2 tests (NEW)
- `tests/CMakeLists.txt` — add test_microblock_cull.cpp
- `scripts/capture_reference.py` — add a `dump_microblock_cull_fixture()` call to write `microblock_cull_inputs.npz` + `microblock_cull_outputs.npz` for the hero view. ALSO regenerate the existing fixture so the new file lands in `tests/fixtures/hero/`.
- `scripts/verify_stage.py` — add a `microblock_cull` stage runner that checks `mb_header` and `mb_stream` against the fixture (exact equality)

## C++ function signature

In `src/gsplat_cpu/microblock_cull.h`:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct MicroblockCullResult {
    std::vector<int64_t> mb_header;   // num_tiles * 32 * 2  (row-major: tile, m, [offset|count])
    std::vector<int64_t> mb_stream;   // L_prime (sum of all counts)
    int64_t pairs_in;
    int64_t pairs_out;                // == L_prime
    int64_t pairs_dropped_in_all_mb;  // for drop_pct stat
};

MicroblockCullResult microblock_cull(
    const float*   means_2d,            // M * 2
    const float*   covs_2d,             // M * 4 (a, b, b, c)  — same layout as blend
    const float*   opacities,           // M
    const int64_t* sorted_gaussian_ids, // P
    const int64_t* tile_ranges,         // num_tiles * 2 (start, end)
    std::size_t M, std::size_t P,
    int tiles_x, int tiles_y,
    int tile_size,                      // 32
    float mb_contrib_floor,             // default 1.0f / 16384.0f
    ThreadPool& pool
);

} // namespace gsplat_cpu
```

## Algorithm — mirror numpy line-for-line

For each tile (parallelize over the pool):
1. `start, end = tile_ranges[tile_id]`. If `start == end`, leave header zeros and produce no stream entries.
2. `tile_g_ids = sorted_gaussian_ids[start..end]`. Let `L = end - start`.
3. `tx_tile = tx * tile_size`, `ty_tile = ty * tile_size`.
4. Compute `keep[L][32]` boolean mask via §2 closest-point formula:
   - `det = max(a*c - b*b, 1e-6)`
   - `ci_a = c/det`, `ci_b = -b/det`, `ci_c = a/det`
   - For each microblock m in 0..31:
     - `mb_ox = tx_tile + (m & 3) * 8`
     - `mb_oy = ty_tile + (m >> 2) * 4`
     - `cx = clamp(mean_x, mb_ox, mb_ox + 8)`
     - `cy = clamp(mean_y, mb_oy, mb_oy + 4)`
     - `dx, dy = cx - mean_x, cy - mean_y`
     - `power = -0.5 * (ci_a*dx*dx + 2*ci_b*dx*dy + ci_c*dy*dy)`
     - `gw = exp(min(power, 0))`
     - `alpha_peak = min(opacities[g] * gw, 0.99f)`
     - `keep[l][m] = alpha_peak >= mb_contrib_floor`

5. Per-tile mb_header[m] = (offset, count); per-microblock kept gaussians go into mb_stream in depth order.

## Important: bit-identical to numpy

The numpy reference uses fp32 throughout (means, covs, opacities are float32 in the fixture). Match exactly:
- **`det = max(a*c - b*b, 1e-6f)`** — use the explicit `f` suffix. Numpy's `np.maximum(..., 1e-6)` upcasts to fp64 then downcasts. Mismatch here = 1 ULP per call. To be safe, do all intermediates in fp32 (cast to double only where numpy explicitly does).
- **`exp(min(power, 0.0f))`** — `std::exp` on negative arguments. fp32 `expf` is fine.
- **Order of fp ops** — within the §2 quadratic, evaluate `ci_a*dx*dx + 2*ci_b*dx*dy + ci_c*dy*dy` in that order. The 2.0 multiplier is on `ci_b*dx*dy`, NOT on `2*dx*dy` separately. Numpy: `ci_a * dx * dx + 2.0 * ci_b * dx * dy + ci_c * dy * dy`.

Threading: one task per tile via the existing `gsplat_cpu::ThreadPool`. Per-task local result, then concat at the end on the main thread. Or — preferable — preallocate `mb_header` (size known up front: `num_tiles * 32 * 2`) and have each task write its own tile's slice directly. For `mb_stream`, each task can compute its (kept count per m) → prefix-sum across tasks → second pass that writes the stream. Or just collect per-task `std::vector<int64_t>` chunks and concat on main.

Pick whichever is simpler; the bit-identical requirement is what matters, not perf (iter-008 is the perf iter).

## Parallel data path: `CpuCppBackend(microblock=True)`

```python
class CpuCppBackend(Backend):
    def __init__(self, microblock: bool = False, mb_contrib_floor: float = 1.0/16384.0):
        ...
        self._microblock = microblock
        self._mb_contrib_floor = mb_contrib_floor

    def blend(self, means_2d, covs_2d, colors, opacities,
              sorted_gaussian_ids, tile_ranges, image_height, image_width):
        if not self._microblock:
            # Existing C++ tile-major path.
            return <existing impl>

        # New: C++ cull + numpy mb-blend.
        tiles_x = (image_width + 31) // 32
        tiles_y = (image_height + 31) // 32
        mb_header_np, mb_stream_np, stats = self._mod.microblock_cull(
            np.ascontiguousarray(means_2d.detach().cpu().numpy(), np.float32),
            np.ascontiguousarray(covs_2d.detach().cpu().numpy().reshape(-1, 4), np.float32),
            np.ascontiguousarray(opacities.detach().cpu().numpy(), np.float32),
            np.ascontiguousarray(sorted_gaussian_ids.detach().cpu().numpy(), np.int64),
            np.ascontiguousarray(tile_ranges.detach().cpu().numpy(), np.int64),
            int(tiles_x), int(tiles_y), 32, float(self._mb_contrib_floor),
        )
        from gsplat.rasterization import alpha_blend_microblock
        image = alpha_blend_microblock(
            means_2d, covs_2d, colors, opacities,
            torch.from_numpy(mb_header_np.reshape(tiles_x*tiles_y, 32, 2)),
            torch.from_numpy(mb_stream_np),
            image_height, image_width,
        ).numpy()
        return image, {
            "microblock_drop_pct": stats["drop_pct"],
            "microblock_work_reduction_pct": stats["work_reduction_pct"],
        }
```

Register `"cpu_cpp_mb"` in `backends/__init__.py`:
```python
class CpuCppMbBackend(CpuCppBackend):
    def __init__(self): super().__init__(microblock=True)
REGISTRY["cpu_cpp_mb"] = CpuCppMbBackend
```
(Or any equivalent that lets `get_backend("cpu_cpp_mb")` work.)

## Hero fixture capture

Add to `scripts/capture_reference.py`:

```python
def dump_microblock_cull_fixture(
    fixtures_dir: Path, backend, blend_inputs: dict, mb_contrib_floor: float
):
    from gsplat.rasterization import microblock_cull
    tiles_x = (blend_inputs["W"] + 31) // 32
    tiles_y = (blend_inputs["H"] + 31) // 32
    mb_header, mb_stream, stats = microblock_cull(
        torch.from_numpy(blend_inputs["means_2d"]),
        torch.from_numpy(blend_inputs["covs_2d"]),
        torch.from_numpy(blend_inputs["opacities"]),
        torch.from_numpy(blend_inputs["sorted_gaussian_ids"]),
        torch.from_numpy(blend_inputs["tile_ranges"]),
        tiles_x, tiles_y, 32, mb_contrib_floor=mb_contrib_floor,
    )
    np.savez_compressed(fixtures_dir / "microblock_cull_inputs.npz",
        means_2d=blend_inputs["means_2d"],
        covs_2d=blend_inputs["covs_2d"],
        opacities=blend_inputs["opacities"],
        sorted_gaussian_ids=blend_inputs["sorted_gaussian_ids"],
        tile_ranges=blend_inputs["tile_ranges"],
        tiles_x=tiles_x, tiles_y=tiles_y,
        mb_contrib_floor=mb_contrib_floor,
    )
    np.savez_compressed(fixtures_dir / "microblock_cull_outputs.npz",
        mb_header=mb_header.numpy(), mb_stream=mb_stream.numpy(),
    )
```

Call it from `dump_per_stage_fixtures` after the blend fixture. Re-run `capture_reference.py --rebuild-fixtures` (or whatever the existing flag is — check the file) to regenerate.

**Don't regenerate other fixtures** if it would change them. If the existing capture_reference.py only handles the iter-000 path, just add a small helper script `scripts/dump_microblock_fixture.py` that loads `tests/fixtures/hero/blend_inputs.npz` and writes the two new files. Either works.

## Catch2 tests in `tests/unit/test_microblock_cull.cpp`

5 tests minimum:
1. Single tile, single Gaussian at tile center (mean=(16, 16)), σ small → exactly 1 microblock keeps it (the one containing pixel (16, 16) — that's m where mb_ox=16, mb_oy=16, i.e., `(16/8)*1 + (16/4)*4 = ...`; compute m from (16, 16) carefully).
2. Single tile, single Gaussian at tile center with σ large (>16 pixels) → keep_count > 1 (gaussian touches many microblocks). Exact count depends on σ + floor — assert ≥ 4 keeps.
3. Single tile, two Gaussians with different depths → mb_stream entries for any kept microblock m maintain the original depth order.
4. Two tiles, gaussian only in tile 0 → mb_header[tile_1] is all zeros.
5. Hero fixture round-trip: load tests/fixtures/hero/microblock_cull_inputs.npz, run C++ microblock_cull, compare against microblock_cull_outputs.npz — exact equality on both arrays.

## verify_stage.py update

Add to TOLS:
```python
"microblock_cull": {"mb_header": 0, "mb_stream": 0},
```

Add to RUNNERS:
```python
def run_microblock_cull(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "microblock_cull_inputs.npz")
    out = np.load(fixtures_dir / "microblock_cull_outputs.npz")
    # Call backend's mod directly — there isn't a method on Backend for this.
    # Backend MUST be cpu_cpp_mb (or cpu_cpp with microblock=True).
    mod = backend._mod
    mb_header_flat, mb_stream, _ = mod.microblock_cull(
        np.ascontiguousarray(inp["means_2d"], np.float32),
        np.ascontiguousarray(inp["covs_2d"].reshape(-1, 4), np.float32),
        np.ascontiguousarray(inp["opacities"], np.float32),
        np.ascontiguousarray(inp["sorted_gaussian_ids"], np.int64),
        np.ascontiguousarray(inp["tile_ranges"], np.int64),
        int(inp["tiles_x"]), int(inp["tiles_y"]), 32, float(inp["mb_contrib_floor"]),
    )
    return {
        "mb_header": (mb_header_flat.reshape(int(inp["tiles_x"]) * int(inp["tiles_y"]), 32, 2), out["mb_header"]),
        "mb_stream": (mb_stream, out["mb_stream"]),
    }
```

## Gates

```bash
source scripts/_env.sh

# Generate fixtures (one-time; commit them)
$LOCAL_PY scripts/dump_microblock_fixture.py  # if you made it a separate script
# OR run capture_reference.py with the right flag

# Layer 1
cmake --build build -j && ctest --test-dir build --output-on-failure -j
# expect 29 existing + ~5 new = 34/34

# Layer 2
$LOCAL_PY scripts/verify_stage.py --backend cpu_cpp_mb --stage microblock_cull
# AND verify the existing four stages still pass on cpu_cpp:
for s in project tile_assign sort blend; do
  $LOCAL_PY scripts/verify_stage.py --backend cpu_cpp --stage $s
done

# Layer 3
rm -rf /tmp/iter007
$LOCAL_PY scripts/render_30frame.py --backend cpu_cpp_mb --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter007 --warmup 1

$LOCAL_PY -c "
from PIL import Image; import numpy as np; from pathlib import Path; import json
ms=[]
for f in sorted(Path('benchmarks/reference_v2').glob('*.png')):
    a=np.asarray(Image.open(f).convert('RGB'),np.float64)/255
    b=np.asarray(Image.open(Path('/tmp/iter007')/f.name).convert('RGB'),np.float64)/255
    mse=float(np.mean((a-b)**2))
    p=float('inf') if mse<=0 else 10*np.log10(1/mse)
    ms.append((f.stem,p))
ms.sort(key=lambda x:x[1])
print(f'MIN={ms[0][1]:.2f}  MEAN={sum(p for _,p in ms)/len(ms):.2f}  MAX={ms[-1][1]:.2f}  (gate 60 dB)')
rows=[json.loads(l) for l in open('/tmp/iter007/timing.jsonl').read().splitlines() if l.strip()]
print(f'sum_total_ms={sum(r[\"total_ms\"] for r in rows):.1f}  (still slow because blend is numpy)')
"
```

Expected: bit-identical PSNR distribution to iter-006 (min 60.34, mean 63.26, max 69.74 dB).

## BUDGET

5 build attempts. If stuck, `git checkout -- .` and surface BLOCKED ON line.

Most likely failure: fp ordering mismatch causing mb_header[(tile, m, 1)] off by 1 in some tile. Diagnostic: dump the first mismatched tile's `(L, 32)` keep mask from both numpy and C++, find the index where they differ, print the (a, b, c, opacity, mean) → trace the math by hand.

## SUMMARY

```
SUMMARY: iter-007 status=PASS|FAIL ctest=N/M verify_microblock_cull=pass|fail render30_min_psnr_dB=X.X sum_total_ms=Z.Z work_reduction_pct=Y.Y
```

Plus 6-12 bullets. Do not commit. Begin.

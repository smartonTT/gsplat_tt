# Ref `_gsplat_cpu` cmake at stage2 tip (2026-06-04)

## Symptom
Gate at branch tip: `hero_vs_ref≈47`, `ms_view≈105` with fresh `render/build-tt` but stale/wrong `_gsplat_cpu`.

## Cause
`src/CMakeLists.txt` (GSPLAT_WITH_TT) still listed `src/gsplat_tt/*_device.cpp` and `OVERRIDE_KERNEL_PREFIX=src/gsplat_tt/`. Stage2 moved drivers to `render/host/` and kernels to `render/kernels/`. Missing `env_config.h` / `blend.h` / `sort.h` / `render_blend.cpp` at tip broke ref rebuild.

## Fix
- Build `gsplat_tt` from `render/host/*.cpp` + `src/gsplat_tt/{render_blend,mb_payload,blend_host}.cpp`.
- `OVERRIDE_KERNEL_PREFIX="${REPO}/render/"`.
- Restore runtime `env_config.h` (+ `blend.h`, `sort.h`) from gate-green `8833462` for pybind ref path (`resident_blend_enabled()` etc.).

## Rebuild (bh-07)
```bash
cmake -G Ninja -S src -B build-avx2-tt -DGSPLAT_SIMD_AVX2=ON -DGSPLAT_WITH_TT=ON
cmake --build build-avx2-tt -j16
cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release
cmake --build render/build-tt -j16
```

# Plan: 1ms/frame, whole pipeline resident on TT device

North star (user directive, 2026-05-29): **no python/host in the render loop; the
entire Gaussian-splat pipeline resident on the Tenstorrent Blackhole device.**
Only the camera matrix goes in per frame; only the final image comes out. Loop
autonomously toward 1ms: attack the biggest bottleneck, verify, checkpoint,
move on. Don't get bogged down — revert and go to the next bottleneck; come back
to re-optimize later.

## The single most important finding (M56)

`blend_ms` (and every stage timer) wraps the whole HOST call, not just the
device kernel. Bracketing the device call with `distributed::Finish`
(`GSPLAT_TT_MB_TIMING=1`) proves the **device blend kernel is only ~23ms**. The
rest of "blend" is host-side CPU marshaling: microblock cull, gaussian-major
stream build, a 200MB coeff payload, and a 113ms per-row upload. The device was
never the bottleneck. The path to 1ms is **eliminating host marshaling and the
per-frame host<->device round-trips**, not tuning kernels.

## Current per-view timing (bicycle, steady state)

| stage | ms | notes |
|---|---|---|
| project | ~242 | device pfwc + D2H readback + host repack (under investigation) |
| tile_assign | ~87 | host CPU pair expansion (~3.2M pairs) (under investigation) |
| sort | ~8 | host CPU sort by (tile, depth) |
| blend | ~450 | host cull ~135 + gmajor build ~131 + upload ~113 + device exec ~23 |
| **total** | **~812** | target: **1** |

Diagnostic env: `GSPLAT_TT_MB_TIMING=1` prints `[BLEND_HOST]` (host phase split)
and `[BLEND_SPLIT]` (upload/exec/readback device split).

## Target architecture (resident pipeline)

Per frame: upload camera (tiny) -> device does project -> tile_assign -> sort ->
cull+coeff -> blend, all reading/writing DRAM-resident buffers, -> read back
image. The 200MB coeff stream never exists: the blend kernel computes conic
coeffs (A=-0.5 c/det, B=b/det, C=-0.5 a/det, tile-local center, opacity, rgb)
on-device from compact per-gaussian attributes {a,b,c,meanx,meany,opacity,rgb}
plus resident sorted per-tile gaussian-id lists.

Key data fact: the per-(tile,gaussian) coeff row is per-GAUSSIAN data (conic +
color) replicated across tiles; only the tile-local center and microblock mask
differ per tile. So resident per-gaussian attrs (~M*36B) + compact (id,mask)
pairs (~3.2M*8B=26MB) replace the 200MB stream.

## CRITICAL REFRAMING (investigators, M2-era)

`render_full_tt` is bound to `render_full_py` (pybind_module.cpp:1602) and runs
**100% on host CPU**: project (`project_full_fused`), tile_assign, sort are all
CPU; only the ~23ms blend kernel runs on-device (fed by host cull + 200MB
upload). The "TT port" is barely started.

- PROJECT 242ms = pure host CPU over 6.13M gaussians (NOT a D2H round-trip).
  Device kernels `transform_means_cam_tt` + `pfwc_tt` ALREADY EXIST
  (project_device.cpp, pfwc_device.cpp) and register outputs in `device_state`,
  but are only reachable via the diagnostic `GSPLAT_TT_DEVICE_PROJECT=1` Python
  path — NOT wired into `render_full_tt`. Wiring them in (+ device/host compact
  to M) is the highest-ROI aligned win: 242ms -> ~10-50ms.
- tile_assign 87ms (host; Mahalanobis cull dominates on heavy views), sort 8ms
  (host, well-optimized). Recommendation: don't port sort (low ROI); host-side
  streamed tile_assign is a cheaper near-term win than a device port; do device
  project first. Device tile_assign/sort come later (M5/M6).
- Scene: N=6.13M gaussians, M~1.88M visible (hero), ~3.2M pairs, 1024 tiles.

## Milestones (re-prioritized each loop by measured bottleneck)

- [x] M0 Revert big-page (restored 39.88dB; per-row upload ~113ms noted for later).
- [x] M1 Drop redundant 200MB coeff_payload zero-init+copy; upload mb_coeff_stream
  directly. blend 597->450ms, ms/view 967->812, PSNR 40.2dB. (commit 3495194)
- [ ] M2 Fuse cull+coeff+gaussian-major emit into one parallel host pass
  (eliminate the separate 131ms stream-build + 128MB p.coeff intermediate).
  IN PROGRESS (subagent). Target blend ~450->~320ms.
- [ ] M3 Keep PROJECT outputs (means_2d, covs_2d, colors, opacities) RESIDENT in
  DRAM; eliminate project D2H + blend re-upload. (pending project investigation)
- [ ] M4 Move microblock cull + coeff computation ON-DEVICE (consume resident
  attrs + sorted lists; no 200MB, no host cull). Big win + big risk.
- [ ] M5 Device tile_assign (binning) with resident pair lists.
- [ ] M6 Device sort (per-tile depth) with resident sorted lists.
- [ ] M7 Fuse into a single resident program; camera in, image out; no python loop.

## Loop protocol (autonomous)

1. Measure: run `a003_verify.py --views 1` with `GSPLAT_TT_MB_TIMING=1`; read the
   stage table + the [BLEND_HOST]/[BLEND_SPLIT] splits. Identify biggest cost.
2. Attack it (prefer changes that move toward residency over throwaway host opts).
   Dispatch subagents for well-scoped implementation; supervise + verify.
3. Verify on bh-30: PSNR must stay >= ~39.5dB (vs cpu_cpp_mb ref); timing must
   improve. If a change breaks correctness or bogs down, REVERT and move to the
   next bottleneck (come back later).
4. Checkpoint: commit; add a metal-iters.jsonl entry; regenerate REPORT.html.
5. Repeat.

## bh-30 run recipe

```
bash scripts/sync_to_bh30.sh   # local -> bh-30
ssh bh-30 'cd /localdev/smarton/gstt2 && export TT_METAL_HOME=/localdev/smarton/tt-metal && \
  GSPLAT_WITH_TT=ON GSPLAT_SIMD_AVX2=ON BUILD_DIR=build-tt bash scripts/build_cpu_cpp.sh; \
  .venv/bin/tt-smi -r; sleep 2; \
  env TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  MESH_DEVICE=P100 TT_METAL_ARCH_NAME=blackhole TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache \
  GSPLAT_TT_BLEND_MODE=2 GSPLAT_TT_MB_KERNEL=1 GSPLAT_TT_MB_TIMING=1 \
  PYTHONPATH=/localdev/smarton/gstt2:/localdev/smarton/gstt2/backends/cpu_cpp/build-tt \
  .venv/bin/python scripts/a003_verify.py --views 1 --iter-dir <name> --out opt/<name>.json'
```
Notes: a benign `double free` in tt-metal's atexit ShmResourceTracker fires AFTER
the SUMMARY prints — key on the `SUMMARY` line, not the process exit code. Reset
the device (`tt-smi -r`) before each run to avoid a wedged device.

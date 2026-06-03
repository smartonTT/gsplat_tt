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
- [x] M3 Wire device project (transform_means_cam_tt_no_download + pfwc_tt) into
  render_full_tt via `project_via_device()`; gated opt-in `GSPLAT_TT_DEVICE_PROJECT=1`,
  default path unchanged. VERIFIED correct (PSNR 39.68dB, layout-identical
  ProjectResult, config-guarded to k_cap==3.0/!isoellipse w/ CPU fallback).
  FINDING: standalone it is a REGRESSION (project 220->400ms) — D2H-bound
  (H2D means + 147MB cov3d_unique, D2H 4 arrays, host repack) while tile_assign
  still reads on host. Kept as resident scaffolding; the win only lands with M3b.
- [ ] M3b Keep project outputs (mean_2d/cov2d/depth/radii + compact colors/opac)
  RESIDENT in DRAM (skip the pfwc D2H + cov3d_unique re-upload); requires a device
  consumer (M5 tile_assign) to read them over NoC. This is where device project pays off.
- [ ] M4 Eliminate the 200MB per-(gaussian,tile) coeff stream. DESIGN DONE
  (read-only subagent). Per-row 64B = 10 coeff + mask(word10) + 5 pad; A,B,C,
  opacity,rgb are PER-GAUSSIAN (identical across a gaussian's tiles), only the
  tile-local center (mx,my) and 32-bit mask are per-PAIR. Plan: split into a
  resident per-gaussian attr table (M*64B: a,b,c,mean_x,mean_y,opacity,rgb) +
  a compact per-tile (gaussian_id, mask) pair stream (~3.2M*8B=26MB), reader
  gathers attrs by id over NoC and derives tile-local center from mean - tile
  origin (needs tiles_x as a runtime arg). Upload 205MB -> 26MB.
  STAGED ROLLOUT (smallest-correct-first, each behind an env gate + revertible):
    A. Pack mask into the unused word[5], drop pad words (validate mask move;
       64B keeps 0 upload win but de-risks; 48B needs the amendment-003 align fix).
    B. Split: attr table + (id,mask) pairs; reader gathers + assembles the SAME
       64B CB row so the COMPUTE KERNEL IS UNCHANGED (lowest risk). ~179MB less
       H2D/frame + host stops emitting 10 coeff floats/row. <- the real win.
    C. On-device conic: kernel derives A,B,C from raw {a,b,c}; attrs shrink.
    D. On-device microblock cull (host emits only sorted per-tile g lists).
    E. Resident attrs via M3b (project writes buf_attrs; blend reads resident).
  RISKS: NoC random gather by id (~3.2M/frame, non-sequential), 64B DRAM page
  alignment (48B caused a silent zero-row bug before), depth-order MUST stay
  sorted (don't reorder by id), numerical parity (keep centered conic, det floor
  1e-6). Do NOT combine B and C in one change. Gate PSNR >= ~39.5dB each step.
- [ ] M4-legacy Move microblock cull + coeff computation ON-DEVICE (consume resident
  attrs + sorted lists; no 200MB, no host cull). Big win + big risk.
- [ ] M5 Device tile_assign (binning) with resident pair lists. DESIGN DONE
  (read-only subagent). Keep CPU's Gaussian-centric structure (NOT tile-centric:
  that is O(M*num_tiles)=1.9B tests). Pipeline: K1 per-gaussian AABB+count ->
  prefix-sum -> K2 disjoint-range scatter of (gid,tid) pairs (NO atomics) ->
  K3 per-gaussian m2_thresh precompute -> K4 per-pair Mahalanobis cull (rectangle
  min m^2, NOT L-inf clamp; det floor 1e-6) -> K5 compaction. New
  src/gsplat_tt/tile_assign_device.cpp + 6 kernels + gather_visible_device.cpp
  (N-indexed pfwc outputs -> M-compact). Gate GSPLAT_TT_DEVICE_TILE_ASSIGN=1,
  CPU fallback. Staged S0 infra -> S1 AABB-only (pair-count parity) -> S2 full cull
  (P'=3,212,720, PSNR>=39.5) -> S3 register ta_* resident -> S4 resident inputs
  (needs M3b gather) -> S5 on-device prefix (host->~0) -> S6 fused cull-scatter ->
  S7 device sort handoff. Prefix-sum: start as tiny host scan on M counts (~1-2ms),
  migrate to core-partial+device scan. Pairs ~51MB DRAM (preallocate P_max).
  WIN fully lands only with M3b resident project + M6 device sort (else pair D2H
  negates the 87ms). Do NOT combine S2 (cull correctness) with S4 (residency).
- [ ] M6 Device sort (per-tile depth) with resident sorted lists. DESIGN DONE
  (read-only subagent). KEY: sort's value is RESIDENCY (eliminate ~77MB/frame
  pair D2H + sorted H2D), NOT the ~8ms compute (device radix won't beat tuned
  AVX2). Algorithm = bin pairs by tile_id (count -> prefix-sum -> stable
  chunk-order scatter) then per-tile STABLE LSD radix (4x8-bit) on
  bitcast(depth) front-to-back; insertion sort for n<=16. Output:
  sorted_gaussian_ids[P] (global concat of per-tile segments) + tile_ranges
  (start,end per tile). Staged: S0 register sort outputs in device_state + CPU
  sort writes them resident (zero PSNR risk, immediate blend-read win) -> S1
  GSPLAT_TT_DEVICE_SORT=1 host-bin + DEVICE per-tile radix (LPT tile->core,
  skip empty tiles to avoid CB deadlock; worst tile ~11k fits L1 ~264KB) ->
  S2 device count+scatter (host prefix on 1024 counts ~0.01ms) -> S3 full device.
  GSPLAT_TT_SORT_VERIFY=1 asserts byte-identical vs CPU. New
  src/gsplat_tt/sort_device.cpp + reader/compute(radix)/writer kernels. Port the
  device kernel AFTER M5 tile_assign residency (else 8ms saved but 51MB D2H
  remains). Reuse blend LPT (compute_lpt_assignment) + pfwc no-download pattern.
- [ ] M7 Fuse into a single resident program; camera in, image out; no python loop.

## Residency integration pass (DESIGN DONE — the actual speedup)

Each ported stage today is gated independently and pays a full host<->device
boundary tax (D2H then H2D), so enabling one alone REGRESSES (proven: device
project alone = 400ms). The win is keeping intermediates RESIDENT in DRAM
(device_state keys) and reading them stage-to-stage over NoC. Removes ~400MB+/
frame of transfers; hero DRAM budget ~400-500MB peak (alias blend_attrs<->proj_m_*,
drop pfwc_* after gather). CRITICAL PATH: on-device gather_visible (N->M compact,
replaces host project_finish) -> resident tile_assign in/out -> resident sort ->
resident blend. Hierarchical gates under master GSPLAT_TT_RESIDENT=1
(+ per-stage GSPLAT_TT_RESIDENT_{PROJECT,GATHER,TA_IN,TA_OUT,SORT,BLEND}), CPU
fallback, GSPLAT_TT_RESIDENT_VERIFY=1 for per-stage parity. Standardize device ids
on uint32; gids are M-local in increasing-N order (sort uses depths[gid]).
STAGED: R0 key registry/device_state grow -> R1 pfwc null-D2H in fused path ->
R2a host-finisher writes into registered proj_m_* (validate keys/PSNR) ->
R2b NEW gather_visible_device.cpp kernel (valid_mask+prefix+compact on device) ->
R3 tile_assign reads proj_m_* over NoC (drop M H2D) -> R4 register ta_gids/ta_tids
(drop pair D2H) -> R5a CPU sort writes sort_* resident -> R5b device radix ->
R6 blend reads resident attrs+ids (drop per-frame upload) -> R7 unified resident
dispatch in render_full_py -> R8 on-device prefix/compaction -> R9 single fused
program. R2b (device gather_visible) is the next critical-path port after sort.

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

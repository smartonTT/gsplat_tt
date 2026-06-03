# render/ — archived alternate algorithms

This file records the genuine *alternate algorithms* (not debug/probe/dump
scaffolding) that were removed from `render/` during the "actually clean"
pass. Pure debug/verify/dump cruft was deleted without an entry here — only
recoverable alternate code paths are listed.

Recovery sources:
- **Pre-clean `render/` source:** git commit `1f783d9` (every kernel/host file
  below existed there with the `#ifdef`/runtime gates intact).
- **Full production variants:** `src/` (the production reference the clean
  renderer was extracted from) still carries these paths behind the original
  `GSPLAT_TT_*` env flags.

To recover a single file as it was before the clean pass:

```
git show 1f783d9:render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp > /tmp/old_reader.cpp
```

---

## Kernel alternate paths

### FUSE_BLEND — fused cull+blend in the cull compute kernel
- **What:** `microblock_cull_compute.cpp` could, under `FUSE_BLEND`, run the
  alpha-blend inline in the cull kernel (helpers `blend_one_gaussian_math`,
  `dispatch_blend_guarded`, `blend_tile_gaussians` + their `CB_*`/`DR_*`
  constants) instead of handing records to the separate blend program.
- **Why removed:** never defined by the host; production splits cull and blend
  into separate programs. Dead `#ifdef`.
- **Recover:** `1f783d9:render/kernels/compute/microblock_cull_compute.cpp`.

### CULL_LEVEL ladder (0/1/2) + cull_bbox / cull_vary
- **What:** the cull compute kernel had an `if constexpr (CULL_LEVEL …)` ladder
  selecting progressively cheaper culls (`CULL_LEVEL==0` none, `==1` bbox-only
  `cull_bbox`, `==2` `cull_vary`, `>=3` full `cull_dispatch`).
- **Why removed:** production bakes `CULL_LEVEL>=3` (full dispatch); the lower
  levels were a development accuracy/speed sweep.
- **Recover:** `1f783d9:render/kernels/compute/microblock_cull_compute.cpp`.

### MB_BLEND_EARLYOUT / MB_BLEND_TAILSKIP — blend transmittance shortcuts
- **What:** `alpha_blend_compute_mb.cpp` experiments that ended a tile's blend
  early once transmittance fell below a threshold (`MB_EO_BLK`, `MB_EO_EPS`,
  the `CB_HS`/`HS_*` half-state CBs and `CB_FB_*` feedback CBs).
- **Why removed:** never defined by the host; they change accumulation order so
  they are NOT bit-identical to the reference blend.
- **Recover:** `1f783d9:render/kernels/compute/alpha_blend_compute_mb.cpp`.

### GATHER_EMIT_TPG / CHUNK_FUSION — tiles-per-gaussian fusion gather
- **What:** `gather_visible_scatter.cpp` could emit a `tiles_per_gaussian`
  (TPG) SoA buffer (`GATHER_EMIT_TPG`) feeding a chunk-fusion tile-grid path;
  host side allocated `ta_tiles_per_gaussian` and pushed `fusion_tx/ty/ts`
  runtime args under `env_config::chunk_fusion_enabled()`.
- **Why removed:** `CHUNK_FUSION` is unset in production (always-false gate).
- **Recover:** `1f783d9:render/kernels/dataflow/gather_visible_scatter.cpp` +
  `render/host/gather_visible_device.cpp`.

### SoA per-gaussian gather (non-AoS blend reader path)
- **What:** `reader_alpha_blend_mb_devcull.cpp` `issue_chunk_reads()` (the
  7–9-page Structure-of-Arrays gather) and the `compute_microblock_mask()`
  soft-float CPU-style microblock cull were the non-`MB_BLEND_AOS` /
  non-`MB_SFPU_CULL` reader variants.
- **Why removed:** production runs the AoS one-page gather (`MB_BLEND_AOS`) and
  the SFPU-precomputed cull mask (`MB_SFPU_CULL`), so both helpers had zero call
  sites once the live macros were inlined.
- **Recover:** `1f783d9:render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp`.

---

## Host alternate paths

### Host gather (R2a) + CPU project reference
- **What:** `gather_visible_device.cpp` `host_gather=true` path read the
  resident `pfwc_*` buffers back, ran `gsplat_cpu::project_finish_with_cov2d_radii`
  on the host (`cpu_reference`), and uploaded the compact result
  (`write_proj_m_from_host`). `run_verify` compared the device gather against
  this CPU reference.
- **Why removed:** `render_clean` always runs the multi-core device gather
  (`host_gather=false`, `verify=false`); these were unreachable.
- **Recover:** `1f783d9:render/host/gather_visible_device.cpp`, or `src/`.

### Host sort (S0) + host bin layout (S1) fallbacks
- **What:** `sort_device.cpp` `sort_and_bin_tt` had an S0 path
  (`gsplat_cpu::sort_and_bin` on host then publish resident) and an S1 path
  (host Pass-1 counts / Pass-2 scatter binning → device radix → host compaction)
  with a "tile too large → CPU fallback" branch, plus `verify_vs_cpu`.
- **Why removed:** production runs the resident-pairs device-binning stage
  (`RESIDENT_PAIRS=1`), which returns first; the S0/S1 host paths were
  unreachable. Unsupported inputs now hard-fail (`set_fail()` → `render.cpp`
  throws) rather than silently computing on the host.
- **Recover:** `1f783d9:render/host/sort_device.cpp`, or `src/`.

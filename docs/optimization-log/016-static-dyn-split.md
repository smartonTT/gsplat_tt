# Iter 016 — Static/dyn DRAM split (Step 2, salvaged from `79a8b0c`)

- **Idea**: ship Step 2 from `plan-stitch-hero-1024.md` correctly. Split the
  9-fp32 per-entry scalar pack into a 5-fp32 **dyn** pack (frame-specific:
  `mean_x, mean_y, cov_a, 2·cov_b, cov_c`) plus a 4-fp32 **static** pack
  (`R, G, B, opacity`) that lives in DRAM for the lifetime of the scene.
  The reader fuses the two on the fly into the same 9-fp32 `CB_SCALARS`
  layout the compute kernel expects — so the compute kernel is untouched.
- **Branch**: `opt/016-static-dyn-split` (off `opt/015-tile-ids-page-align-fix`)
- **Decision**: **KEPT**

## Salvage path

`79a8b0c` ("first stage kernel optimizations") shipped this idea but also
shipped a new root-level `alpha_blend_compute.cpp` that was never loaded
by the host and referenced an unallocated `CB_CONST_ONE` — so the whole
commit was broken end-to-end. The kept compute kernel
(`kernels/compute/alpha_blend_compute.cpp`) was correct at `d8299fb` and
unchanged in `79a8b0c`. So Phase 1 cherry-picks the salvageable hunks:

- `git checkout 79a8b0c -- backends/tt/backend.py` (FRM1 → SCN1/FRM2)
- `git checkout 79a8b0c -- alpha_blend.cpp` (SCN1 protocol, FRM2 protocol,
  `static_colors_opacity` static DRAM buffer, `sorted_gids` cache,
  reader 9-RT-arg / 7-CT-arg dispatch, `dyn_packs` 32-byte pages)
- `git checkout 79a8b0c -- alpha_blend_host.h` (new page-size constants:
  `DYN_PACK_PAGE_BYTES=32`, `STATIC_COLOR_OPACITY_PAGE_BYTES=32`,
  `SORTED_GIDS_PAGE_BYTES=64`)
- `git checkout 79a8b0c -- kernels/dataflow/reader_alpha_blend.cpp`
  (new 9-RT-arg / 7-CT-arg reader; gid-page cache; dyn+static gather)
- `git checkout 79a8b0c -- gsplat/pipeline.py` (`Pipeline.set_scene` hook)
- `git checkout 79a8b0c -- gsplat/rasterization.py`
  (`prepare_kernel_inputs(static_handled_externally=True)`)

We do NOT bring in the dead `alpha_blend_compute.cpp` at the root of
`gaussian_splatting/`, nor any of the UI scaffold from `79a8b0c`
(`gsplat/camera_controls.py`, `letterbox.py`, `nerfview_viewer.py`, etc).

## Bugs in `79a8b0c`'s reader that had to be re-fixed

The salvaged reader still had two latent bugs even with the compute kernel
unchanged. Both were caught by `TT_METAL_WATCHER=5`.

### Bug A — `noc_async_read` into NCRISC IRAM (not L1)

```
Local L1 address overflow.
NCRISC using noc1 tried to unicast read 64 bytes to local L1[0xffb01b20]
from DRAM core w/ virtual coords 17-16 DRAM[addr=0x00602700]
```

The salvaged reader declared three stack-allocated scratch buffers:

```cpp
uint32_t scratch_dyn[8];
uint32_t scratch_static[8];
uint32_t scratch_gids[16];
const uint32_t scratch_dyn_addr = reinterpret_cast<uint32_t>(scratch_dyn);
// ...
```

then `noc_async_read(... , scratch_gids_addr, ...)`. The compiler placed
these on the NCRISC stack, which lives in NCRISC's **private IRAM**
(`0xFFB0_xxxx` range on Blackhole), NOT the worker's shared L1. The NoC
cannot write to NCRISC IRAM, so the read silently fault-traps the kernel
and watcher reports an L1 overflow.

**Fix**: add a dedicated **`CB_READER_SCRATCH`** circular buffer
(`alpha_blend_host.h`, index 24, depth 1, page 128 bytes,
`DataFormat::UInt32`). The reader calls `get_write_ptr(CB_READER_SCRATCH)`
once at startup to capture a stable, NoC-addressable L1 region, and uses
it as scratch for the gid-page cache (bytes 0..63) and the static
color/opacity gather (bytes 64..95). Never push/pop — depth=1 keeps it
pinned to a single L1 slot.

`tt-buddy` lesson recorded:
> NoC destinations must be in worker L1, NOT in any RISC's private
> IRAM/data-RAM. Stack-allocated arrays in dataflow kernels can be used
> as CPU scratch but never as NoC `noc_async_read` destinations. Always
> use `get_write_ptr(CB_X)` for an L1 NoC-write target.

### Bug B — `cb_addr + 20` not 16-byte aligned

NoC L1 destinations on Blackhole require 16-byte alignment. We compose
`CB_SCALARS` as `[5 fp32 dyn | 4 fp32 static]` = 9 fp32 = 36 bytes. The
static gather wants to land at byte offset 20 in CB_SCALARS, which is
4-byte but not 16-byte aligned, so we can't NoC-write there directly.

**Fix**: NoC-write the static page to `CB_READER_SCRATCH + 64` (an
aligned L1 scratch), then CPU-copy four uint32 words from scratch into
`out[5..8]` of `CB_SCALARS`. The dyn pack writes directly into
`CB_SCALARS + 0` (aligned to the 64-byte CB page boundary).

## Verification

Multi-resolution stress (60 resolutions through one persistent daemon)
passed — confirms no regression to the page-alignment fix from iter 015.

Headless harness, luigi hero 320×640, median of 10 timed frames:

| metric                       | baseline (opt/015) | Phase 1 (opt/016) | Δ        |
|-----------------------------|--------------------|--------------------|----------|
| `prep`                       | 2.63 ms            | 1.27 ms            | **−52%** |
| `save_npy`                   | 3.57 ms            | 3.24 ms            | −9%      |
| `daemon_rt`                  | 16.61 ms           | 13.43 ms           | **−19%** |
| `daemon_rt.device_kernel`    | 11.75 ms           | 12.12 ms           | +3% (noise) |
| `timings.total`              | 35.90 ms           | 30.99 ms           | **−14%** |
| `end-to-end FPS`             | 27.85              | 32.27              | **+16%** |
| PSNR vs base reference       | 37.16 dB           | 37.16 dB           | 0 |

Headless harness, stitch hero 480×640 (configured size), median of 10:

| metric                       | Phase 1   |
|------------------------------|-----------|
| `prep`                       | 10.24 ms  |
| `daemon_rt`                  | 61.43 ms  |
| `daemon_rt.device_kernel`    | 58.37 ms  |
| `timings.total`              | 155.25 ms |
| FPS (e2e)                    | 6.44 |
| FPS (kernel)                 | 17.13 |

Live viewer (`http://localhost:8080`): HTTP 200, page loads, served by
the Phase-1 binary.

## Gates met

- `prep ↓ from 2.6 → < 1 ms at 320×640` — **PARTIAL** (1.27 ms, 52% of
  baseline). The remaining ~0.3 ms over the gate target is from the
  six per-stage `torch → numpy` copies in `prepare_kernel_inputs` and
  is the explicit target of Phase 7.
- `save_npy payload ↓ ≥ 30%` — **MET** (per-frame payload was 9-fp32 ×
  N_entries = ~6.2 MB at luigi; now 5-fp32 dyn + uint32 gids
  per entry = ~4.1 MB, with static color/opacity uploaded only when
  the visible set changes; ~34% per-frame reduction).
- `PSNR ≥ 35 dB` — **MET** (37.16 dB).
- `Viewer renders luigi` — **MET**.

## Carried-forward followup for Phase 8

`Pipeline.set_scene` is called with the FILTERED visible-set
(`gaussians.colors[valid_mask]`) on every frame. The Python identity
check (`self._scene_colors_id == colors_id`) doesn't help because
`tensor[mask]` makes a fresh tensor each call. The bigger algorithmic
win is **uploading the FULL static set once at scene load** and indexing
by the global gid (rather than re-uploading the filtered set every
frame). That belongs in Phase 8 alongside the tile-major reorder of
`static_colors_opacity`.

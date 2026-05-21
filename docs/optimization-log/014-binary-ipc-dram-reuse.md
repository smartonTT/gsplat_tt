# Step 014 — Binary IPC + DRAM buffer reuse

- **Idea**: Replace per-frame `.npy` file IPC with a binary stdin/stdout protocol and cache the six DRAM `MeshBuffer`s across frames so allocation and bulk zero-fill disappear from the hot path.
- **Branch**: `smarton/optimization`
- **Decision**: **KEPT**

## Changes

### Host (`backends/tt/backend.py`)

- Switched daemon `subprocess.Popen` to `text=False`; after `READY\n` all IPC is binary.
- Frame request: 24-byte `FRM1` header + four raw `float32` payloads (packs, offsets, px, py) written via `memoryview` (no extra copy).
- Frame response: 16-byte `OK11` header + raw `float32` HWC image read directly into a preallocated numpy array.
- `QUIT\n` remains a text line on shutdown.

### Daemon (`alpha_blend.cpp`)

- Added `BufferCache` in `DeviceContext` holding the six DRAM buffers plus `last_frame_tile_nonempty`.
- Geometric ×1.5 growth when `total_entries`, `offsets_count`, or `tile_ids_bytes` exceed current caps; `px`/`py`/`output` reallocated only on resolution change.
- **Selective output zero-fill**: on resolution change, zero the entire output buffer once. On subsequent frames, zero only tiles that were nonempty last frame and empty this frame (1→0 flip), using `enqueue_write_shards` with `BufferRegion` per tile slot. Tiles that stay nonempty are overwritten by the writer kernel; tiles that stay empty retain prior zeros.

## Bench (stitch hero @ 1024×1024, median of 10 timed frames)

| Metric | Step 0 (ms) | Step 1 (ms) | Δ |
|---|---:|---:|---|
| `sub.blend.prep` | 473.85 | 245.31 | −48% |
| `sub.blend.save_npy` | 21.82 | 80.27 | +268%† |
| `sub.blend.load_npy` | 2.58 | 8.27 | +220%† |
| `sub.blend.daemon_rt` | 256.85 | **195.14** | **−24%** |
| `sub.blend.daemon_rt.device_kernel` | 106.47 | **102.53** | **−3.7%** |
| **`timings.total`** | **981.93** | **880.98** | **−10.3%** |

† Sub-stage labels retained for benchmark compatibility. `save_npy` now includes the ~73 MB stdin payload write (previously split across `.npy` disk I/O + a short text line). `load_npy` reads ~12 MB from stdout instead of `np.load` from `/tmp`. Net end-to-end win is dominated by eliminated disk I/O and per-frame DRAM allocation.

## Visual gate

- **exit 0 (clean-keep)** vs Step 0 reference: PSNR ∞, SSIM 1.0000 (pixel-identical).
- max-abs-diff 0/0/0; mean-abs-diff 0.0 LSB.

## Notes

Selective zero-fill avoids the ~6.3 MB full-buffer `EnqueueWriteMeshBuffer` on every frame while preserving correctness: `compute_lpt_assignment` skips empty tiles, so any tile transitioning from having Gaussians to none would otherwise retain stale DRAM pixels when buffers are reused. Tracking per-tile occupancy across frames and issuing small region writes only on 1→0 flips reduces device-side upload work with no visual change.

# Sort: on-device compact + publish (eliminate host bridge)

## Host work today (resident path `sort_resident_pairs`)

| Step | Where | ~ms (warm) |
|------|--------|------------|
| Device bin + radix | `sort_bin.cpp` + `sort_radix_tile` | bin ~8, kernel ~7 |
| **D2H** `buf_out` | `sort_device.cpp:621-624` | **~8** |
| **Host compact** | scatter `out_aligned` → `sorted_gaussian_ids` | **~4** |
| **H2D publish** | `publish_resident()` → `sort_sorted_ids` + `sort_tile_ranges` | **~10** |
| `Finish()` barriers | between each phase | bubbles |

Binning/radix are already on-device. The **~23 ms bridge** is: read aligned pairs back, compact on CPU, write into separate resident buffers.

## Target

- Writer kernel reads `buf_out` (page-aligned per-tile layout from binning) + `buf_tmeta` / `tile_ranges`, writes **directly** into `sort_sorted_ids` at the contiguous tile slice (same layout host compact produces).
- **No** `result.sorted_gaussian_ids` host vector in hot path when `GSPLAT_TT_SORT_DEVICE_PUBLISH=1`.
- `tile_ranges` already known on host from binning metadata — can stay a small H2D of ranges only, or device-write `sort_tile_ranges` too.

## Minimal kernel sketch (`sort_publish.cpp`, BRISC writer)

Per tile `t` from LPT list:

- `start, end` from `tile_ranges` (or tmeta)
- `pstart = pstart_elem[t]` from binning hist (already in device memory via `buf_bin2d` / keys layout)
- For `k in [0, count)`: `sorted_ids[start+k] = out_aligned[pstart+k]`

Single-core or sharded by tile like radix.

## Host changes (~120–180 lines)

1. Allocate `sort_sorted_ids` once; on device-publish path skip D2H+compact loop.
2. Gate: `GSPLAT_TT_SORT_DEVICE_PUBLISH=1` (default off).
3. Keep host path for verify/fallback.

## Verify

- `GSPLAT_TT_SORT_VERIFY=1` byte-compare vs CPU sort (existing).
- `a003_verify.py` 30-view, `hero_vs_ref >= 63.85` ×2.

## Risk

- **Low** if layout matches host compact exactly (use same `pstart_elem`, `counts`).
- **Medium** if tile_ranges must be device-written — still small.

## Not in scope

- Removing all `Finish()` (persistent kernels — separate track).

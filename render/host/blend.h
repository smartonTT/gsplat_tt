#pragma once

#include <cstdint>
#include <vector>

namespace gsplat_tt {

// RESIDENT device microblock-cull blend (GSPLAT_TT_RESIDENT_BLEND gate; requires
// RESIDENT_GATHER + DEVICE_SORT + MB_DEVCULL). Identical device kernel + conic/
// microblock-cull math as blend_mb_devcull_from_payload, but the reader gathers
// every gaussian's attributes from the device-resident per-component SoA
// proj_m_* buffers by id and consumes the resident sort_sorted_ids +
// sort_tile_ranges instead of host-built+uploaded attr/id payloads. This drops
// the host attr-table build, the host id-list build, and the ~127MB/frame upload.
// The only host input is the per-tile candidate count (for LPT tile->core load
// balancing). Returns false (device_ok=false) if any required resident buffer is
// missing, so the caller can fall back to the uploaded devcull path.
// LPT + per-tile counts come from resident sort_* buffers (no host tile_ranges
// scan). Writes the final hero image into image_out (pre-zeroed, H*W*3).
// cull_ms_out / blend_ms_out (optional): when non-null, receive the de-lumped
// SFPU cull-pass ms and blend-pass ms separately (the return value is their
// sum). Used by the sort-blend continuation so render_full_py can report SORT,
// CULL and BLEND as distinct stages. Measurement-only; no effect on the image.
double blend_mb_devcull_resident(
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    float* image_out,
    bool* device_ok,
    double* cull_ms_out = nullptr,
    double* blend_ms_out = nullptr,
    // Saturation epsilon forwarded to the blend compute kernel (viewer
    // "Transmittance threshold" slider). 0 => kernel keeps its compile-time
    // default (iter-107 baseline).
    float transmittance_threshold = 0.0f);

void device_shutdown();

// Force-create resident blend/cull MeshWorkload contexts (JIT compile only).
void blend_warmup_resident_contexts();

}  // namespace gsplat_tt

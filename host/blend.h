#pragma once

#include <cstdint>
#include <vector>

namespace gsplat_tt {

// Resident microblock blend: tile L1 cull (PACK2 + subchunk dir) then blend from
// sort_sorted_ids + proj_m_blendrec gather. Returns device_ok=false if a required
// resident buffer is missing (no CPU/upload fallback in render_clean).
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
    double* blend_ms_out = nullptr);

void device_shutdown();

// Force-create resident blend/cull MeshWorkload contexts (JIT compile only).
void blend_warmup_resident_contexts();

}  // namespace gsplat_tt

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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
// scan). Writes the final 8-bit RGB image into image_out (H*W*3 bytes,
// uint8(clip(x,0,1)*255), packed on device).
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
    uint8_t* image_out,
    bool* device_ok,
    double* cull_ms_out = nullptr,
    double* blend_ms_out = nullptr,
    // Saturation epsilon forwarded to the blend compute kernel (viewer
    // "Transmittance threshold" slider). 0 => kernel keeps its compile-time
    // default (iter-107 baseline).
    float transmittance_threshold = 0.0f);

// Task #374 (GSPLAT_TT_OUT_ZEROCOPY=1): the last blended frame, still in the
// pinned buffer the blend writer wrote (rows of `pitch` bytes, W*3 used); the
// blend skips the host copy when image_out is null. `owner` is the caller's
// lease: the ring never writes a buffer whose lease is still held, so the image
// stays valid as long as the caller keeps `owner` (and at least until the frame
// after next even if the ring has to replace slots). Empty when off.
struct OutImageView {
    std::shared_ptr<const void> owner;
    const uint8_t* data = nullptr;
    std::size_t pitch = 0;
};
OutImageView blend_out_zerocopy_last();

void device_shutdown();

// Task #90: true when the SFPU microblock cull runs inside the sort_subchunk_mat
// program (sort_device.cpp; GSPLAT_TT_FUSE_MATCULL=0 restores the separate
// tile_l1_cull program). The blend then skips its own cull pass.
bool sort_matcull_fused();
// Task #306: GSPLAT_TT_MATCULL_TRISC_FILL (needs the fused cull and the
// one-launch sort): the TRISCs fill the cull tiles and patch the masks.
// Task #315: unset = on, =0 = off.
bool sort_matcull_trisc_fill();

// Force-create resident blend/cull MeshWorkload contexts (JIT compile only).
void blend_warmup_resident_contexts();

}  // namespace gsplat_tt

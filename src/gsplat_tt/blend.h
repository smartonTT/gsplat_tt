#pragma once

#include <cstdint>
#include <vector>

namespace gsplat_tt {

// In-process TT blend (tt-001a). All buffers are row-major fp32 / uint32 as
// produced by prepare_microblock_payload in Python (to be moved to C++ later).
double blend_from_payload(
    const std::vector<float>& packs_f32,
    const std::vector<float>& offsets_f32,
    const std::vector<float>& px_f32,
    const std::vector<float>& py_f32,
    const std::vector<float>& coeff_f32,
    const std::vector<uint32_t>& mb_header_u32,
    const std::vector<uint32_t>& mb_stream_u32,
    int image_height,
    int image_width,
    std::vector<float>& image_out);

// Microblock-major (4x8) device blend (amendment-003 step 3). Consumes the
// pre-gathered, de-referenced microblock payload (counts + per-tile coeff-stream
// offsets + mb-major coeff rows) built host-side from MbPayload. xramp/yramp are
// generated internally. Returns kernel-only elapsed ms.
double blend_mb_from_payload(
    const std::vector<uint32_t>& mb_counts,        // num_tiles * 32
    const std::vector<uint32_t>& mb_coeff_off,     // num_tiles + 1 (rows)
    const std::vector<float>& mb_coeff_stream,     // total_pairs * 10 (host pads to 48B)
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    std::vector<float>& image_out);

// Device microblock-cull blend (GSPLAT_TT_MB_DEVCULL gate). The host ships only
// compact per-gaussian attributes + per-tile depth-sorted gaussian-id lists; the
// device reader gathers each gaussian's attributes, computes the conic + the
// 32-bit microblock-coverage mask on-core (bit-identical to the host cull), and
// the SFPU compute kernel blends exactly as the standard mb path. This removes
// the host conic/cull and the ~200MB per-frame coeff-row upload.
//   attrs:   M * kMbAttrLanes fp32 (per visible gaussian, 64B page)
//   ids:     concatenated per-tile depth-sorted compact gaussian ids (uint32)
//   ids_off: num_tiles + 1 prefix-sum offsets into ids
double blend_mb_devcull_from_payload(
    const std::vector<float>& attrs,
    const std::vector<uint32_t>& ids,
    const std::vector<uint32_t>& ids_off,
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    std::vector<float>& image_out);

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
double blend_mb_devcull_resident(
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    float* image_out,
    bool* device_ok);

void device_shutdown();

bool device_ready();

// Force-create resident blend/cull MeshWorkload contexts (JIT compile only).
void blend_warmup_resident_contexts();

}  // namespace gsplat_tt

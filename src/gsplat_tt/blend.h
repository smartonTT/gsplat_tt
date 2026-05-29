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

void device_shutdown();

bool device_ready();

}  // namespace gsplat_tt

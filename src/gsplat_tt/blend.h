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

void device_shutdown();

bool device_ready();

}  // namespace gsplat_tt

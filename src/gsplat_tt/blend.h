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

void device_shutdown();

bool device_ready();

}  // namespace gsplat_tt

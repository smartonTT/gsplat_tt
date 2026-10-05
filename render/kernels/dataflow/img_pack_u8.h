// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Integer-only helpers for the blend writer's on-device 8-bit image pack
// (task #61). Shared by writer_alpha_blend.cpp and the host-side model test
// (tests/spec/test_img_pack_u8.py), so both run the exact same code.
//
// bf16 -> u8: matches the host path it replaces bit for bit,
//   u8 = uint8(float32(clip(float32(bf16), 0, 1)) * 255.0f)   (truncation)
// A bf16 in [0,1) is (128+m) * 2^(e-134) with an 8-bit significand, so the
// fp32 product with 255 has a <=16-bit significand and is exact: the result is
// ((128+m)*255) >> (134-e) with no rounding. e >= 127 clips to 255; negatives,
// denormals and e < 119 (v*255 < 1) give 0.
//
// Tile layout: the blend packs each colour channel as a 32x32 bf16 tile whose
// slot dev = r*32 + c holds image pixel (i, j) of the screen tile with
//   i = (r>>2)*4 + (r&1)*2 + (c>>4)
//   j = ((r>>1)&1)*16 + (c&1)*8 + ((c>>1)&7)
// (the microblock permutation, mb_perm_img_of_dev() in blend_device.cpp).

#pragma once

#include <cstdint>

namespace img_pack_u8 {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ROW_BYTES = TILE_DIM * 3;  // one tile row of RGB u8

inline uint32_t bf16_to_u8(uint32_t b) {
    const uint32_t e = (b >> 7) & 0xFFu;
    if ((b & 0x8000u) != 0u || e < 119u) {
        return 0u;
    }
    if (e >= 127u) {
        return 255u;
    }
    const uint32_t sig = 128u + (b & 0x7Fu);
    return ((sig << 8) - sig) >> (134u - e);
}

// Scatter one bf16 channel tile (1024 values, as 512 u32 words) into a
// 32-row x 96-byte RGB staging block in image-row order.
inline void pack_channel(const volatile uint32_t* src, volatile uint8_t* stage, uint32_t ch) {
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        const uint32_t i0 = (r >> 2) * 4u + (r & 1u) * 2u;
        const uint32_t j0 = ((r >> 1) & 1u) * 16u;
        volatile uint8_t* row0 = stage + i0 * ROW_BYTES + j0 * 3u + ch;
        volatile uint8_t* row1 = row0 + ROW_BYTES;
        const volatile uint32_t* w = src + r * (TILE_DIM / 2u);
        for (uint32_t c2 = 0; c2 < TILE_DIM / 2u; c2++) {
            const uint32_t v = w[c2];
            volatile uint8_t* p = ((c2 >> 3) ? row1 : row0) + (c2 & 7u) * 3u;
            p[0] = static_cast<uint8_t>(bf16_to_u8(v & 0xFFFFu));   // c even
            p[24] = static_cast<uint8_t>(bf16_to_u8(v >> 16));      // c odd: j + 8
        }
    }
}

}  // namespace img_pack_u8

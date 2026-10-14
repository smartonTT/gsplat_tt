// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Task #708: forward training flag (host env GSPLAT_TT_BLEND_TRAIN=1, default
// off). The blend then runs the training forward (BLEND_TRAIN in
// alpha_blend_compute_mb.cpp: fp32-accurate exp, gsplat's exact alpha/T
// rules, no T early-out, SCHED 0 bodies, unfused program) and also writes the
// fp32 R, G, B, T accumulators per screen tile to a train buffer that stays
// alive until the next forward, for T1 (device blend backward). Layout: one
// 16 KB page per screen tile, page ty * image_tiles_x + tx, 4 fp32 32x32 tiles
// (R, G, B, T) in the blend's microblock layout (img_pack_u8.h permutation).
// T < 0 marks a pixel whose blend stopped on the 1e-4 rule; |T| is gsplat's
// final T. GSPLAT_TT_BLEND_TRAIN_DUMP=<path> also reads it back after each
// forward and writes it to <path> (raw pages, no header).

#include <cstdint>
#include <cstdlib>

namespace gsplat_tt::blend_train {

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_BLEND_TRAIN");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

inline const char* dump_path() {
    const char* e = std::getenv("GSPLAT_TT_BLEND_TRAIN_DUMP");
    return (e != nullptr && e[0] != '\0') ? e : nullptr;
}

constexpr uint32_t kCb = 17;                          // CB_TRAIN_OUT (fp32, 4 tiles)
constexpr uint32_t kTileBytes = 32u * 32u * 4u;
constexpr uint32_t kPageBytes = 4u * kTileBytes;      // R, G, B, T

// Task #717: lossless op/colour in train mode. The per-scene packs (#122) then
// carry no UNORM16 colour: p01 = the cull's opacity code only (rounded UP, so
// the microblock cull never drops what the fp32 opacity would keep), p23 = the
// scene index. The blend reader follows rec[7] (scene index) to the fp32
// [op, r, g, b] entry of the "scene_train_opc" buffer (16 B per gaussian, 64 B
// DRAM pages) and hands those words to the compute bit for bit.
constexpr uint32_t kOpcBytes = 16u;                   // fp32 op, r, g, b
constexpr uint32_t kOpcPageBytes = 64u;               // 4 entries per DRAM page
constexpr uint32_t kOpcReaderArg = 28u;               // reader runtime arg index

// Task #724: decode-ahead ring slot in train mode (alpha_blend_compute_mb.cpp
// DA_SLOT_BYTES with BLEND_TRAIN): 18 SFPLOADI words for 9 fp32 values, mask
// and marker kind in the last two words. Flag off keeps 64 B slots.
inline uint32_t da_slot_bytes() { return enabled() ? 128u : 64u; }

// Smallest UNORM16 code whose decode is >= v (v not NaN): the round-to-nearest
// code (nearest) or the next one, saturating at 65535 (decode 1.0).
inline uint32_t cull_op16_from_nearest(uint32_t nearest) {
    return nearest >= 65535u ? 65535u : nearest + 1u;
}

}  // namespace gsplat_tt::blend_train

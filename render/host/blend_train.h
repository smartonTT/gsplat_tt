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

}  // namespace gsplat_tt::blend_train

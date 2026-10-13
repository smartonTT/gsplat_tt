// SPDX-License-Identifier: Apache-2.0
// Task #623: single-frame split across 2 chips. The tile window this process
// renders (GSPLAT_TT_SPLIT=1); off = the full screen and every kernel arg is
// exactly the one-chip one. render_view sets it per view; the pfwc, sort, mat
// and blend hosts read it when they build kernel args.
#pragma once

#include <cstdint>

#include "../kernels/dataflow/tile_window.h"

namespace gsplat_tt::split_window {

struct Window {
    bool on = false;
    uint32_t x0 = 0, y0 = 0, w = 0, h = 0;      // tiles, window
    uint32_t screen_tiles_x = 0, screen_tiles_y = 0;  // tiles, full image
};

inline Window& current() {
    static Window w;
    return w;
}

// The value for a kernel's tiles_x slot (tile id -> pixel origin): the plain
// tiles_x when off, else tile_window::pack(w, x0, y0).
inline uint32_t origin_arg(uint32_t tiles_x) {
    const Window& w = current();
    return w.on ? tile_window::pack(w.w, w.x0, w.y0) : tiles_x;
}

// Image row length in tiles (pitch) and image rows in tiles: the full screen.
inline uint32_t image_tiles_x(uint32_t tiles_x) {
    const Window& w = current();
    return w.on ? w.screen_tiles_x : tiles_x;
}
inline uint32_t image_tiles_y(uint32_t tiles_y) {
    const Window& w = current();
    return w.on ? w.screen_tiles_y : tiles_y;
}

}  // namespace gsplat_tt::split_window

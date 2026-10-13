// SPDX-License-Identifier: Apache-2.0
// Task #623: single-frame split across 2 chips. The tile window this process
// renders (GSPLAT_TT_SPLIT=1); off = the full screen and every kernel arg is
// exactly the one-chip one. render_view sets it per view; the pfwc, sort, mat
// and blend hosts read it when they build kernel args.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

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

// The window from GSPLAT_TT_SPLIT (split) and GSPLAT_TT_SPLIT_WINDOW (win,
// "x0,x1,y0,y1" in tiles, half-open: tiles [x0, x1) x [y0, y1)). split unset or
// "0": off. Throws std::invalid_argument on a malformed or empty window, or one
// outside the screen or the tile_window packing (10-bit fields).
inline Window parse(const char* split, const char* win, uint32_t tiles_x, uint32_t tiles_y) {
    Window w;
    if (split == nullptr || *split == '\0' || std::strcmp(split, "0") == 0) return w;
    if (win == nullptr || *win == '\0')
        throw std::invalid_argument("GSPLAT_TT_SPLIT=1 needs GSPLAT_TT_SPLIT_WINDOW=x0,x1,y0,y1");
    long v[4];
    const char* s = win;
    for (int i = 0; i < 4; ++i) {
        char* end = nullptr;
        v[i] = std::strtol(s, &end, 10);
        if (end == s || (i < 3 && *end != ',') || (i == 3 && *end != '\0'))
            throw std::invalid_argument(std::string("GSPLAT_TT_SPLIT_WINDOW: bad value '") + win +
                                        "' (want x0,x1,y0,y1 in tiles)");
        s = end + 1;
    }
    if (v[0] < 0 || v[2] < 0 || v[1] <= v[0] || v[3] <= v[2] || v[1] > static_cast<long>(tiles_x) ||
        v[3] > static_cast<long>(tiles_y) || tiles_x > tile_window::FIELD || tiles_y > tile_window::FIELD)
        throw std::invalid_argument(std::string("GSPLAT_TT_SPLIT_WINDOW: '") + win +
                                    "' is empty or outside the " + std::to_string(tiles_x) + "x" +
                                    std::to_string(tiles_y) + " tile screen");
    w.on = true;
    w.x0 = static_cast<uint32_t>(v[0]);
    w.y0 = static_cast<uint32_t>(v[2]);
    w.w = static_cast<uint32_t>(v[1] - v[0]);
    w.h = static_cast<uint32_t>(v[3] - v[2]);
    w.screen_tiles_x = tiles_x;
    w.screen_tiles_y = tiles_y;
    return w;
}

inline Window from_env(uint32_t tiles_x, uint32_t tiles_y) {
    return parse(std::getenv("GSPLAT_TT_SPLIT"), std::getenv("GSPLAT_TT_SPLIT_WINDOW"), tiles_x, tiles_y);
}

inline bool same(const Window& a, const Window& b) {
    if (!a.on || !b.on) return a.on == b.on;
    return a.x0 == b.x0 && a.y0 == b.y0 && a.w == b.w && a.h == b.h &&
           a.screen_tiles_x == b.screen_tiles_x && a.screen_tiles_y == b.screen_tiles_y;
}

// Grid passed to the sort/blend driver: the window's tiles when on.
inline uint32_t grid_tiles_x(uint32_t tiles_x) { return current().on ? current().w : tiles_x; }
inline uint32_t grid_tiles_y(uint32_t tiles_y) { return current().on ? current().h : tiles_y; }

}  // namespace gsplat_tt::split_window

// SPDX-License-Identifier: Apache-2.0
// Task #623: single-frame split across 2 chips. A chip renders a window of
// screen tiles; tile ids downstream of pfwc are window-local
// ((ty - y0) * w + (tx - x0)). Kernels that turn a tile id into a pixel
// origin take a packed "origin arg" in their existing tiles_x slot:
//   FLAG | w | x0 << 10 | y0 << 20
// A plain value (no FLAG) is the full screen {w = a, x0 = 0, y0 = 0}, so the
// one-chip path passes exactly the same args as before.
#pragma once

#include <cstdint>

namespace tile_window {

constexpr uint32_t FLAG = 0x80000000u;
constexpr uint32_t FIELD = 0x3FFu;

struct Origin {
    uint32_t w, x0, y0;
};

inline uint32_t pack(uint32_t w, uint32_t x0, uint32_t y0) { return FLAG | w | (x0 << 10) | (y0 << 20); }

inline Origin decode(uint32_t a) {
    if (a & FLAG) return Origin{a & FIELD, (a >> 10) & FIELD, (a >> 20) & FIELD};
    return Origin{a, 0u, 0u};
}

// pfwc fused writer args 18/19 (tiles_x / tiles_y). Full-screen tile counts
// stay in the low bits; a window adds x0, wx (arg 18) and y0, wy (arg 19).
// wx == 0 means no window, so the off path is the plain tiles_x / tiles_y.
inline uint32_t pack_pfwc_x(uint32_t tiles_x, uint32_t x0, uint32_t wx) { return tiles_x | (x0 << 10) | (wx << 20); }
inline uint32_t pack_pfwc_y(uint32_t tiles_y, uint32_t y0, uint32_t wy) { return tiles_y | (y0 << 11) | (wy << 21); }

struct PfwcWin {
    uint32_t tiles_x, tiles_y;  // full screen
    uint32_t x0, y0, wx, wy;    // wx == 0: off (window = full screen)
};

inline PfwcWin decode_pfwc(uint32_t a18, uint32_t a19) {
    PfwcWin r;
    r.tiles_x = a18 & FIELD;
    r.x0 = (a18 >> 10) & FIELD;
    r.wx = (a18 >> 20) & FIELD;
    r.tiles_y = a19 & 0x7FFu;
    r.y0 = (a19 >> 11) & FIELD;
    r.wy = (a19 >> 21) & 0x7FFu;
    if (r.wx == 0u) {
        r.x0 = 0u;
        r.y0 = 0u;
        r.wx = r.tiles_x;
        r.wy = r.tiles_y;
    }
    return r;
}

}  // namespace tile_window

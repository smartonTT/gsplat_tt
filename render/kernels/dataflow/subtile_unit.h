// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// subtile_unit.h — task #722 (B1', docs/training-plan/t719-b1-subtile/PLAN.md):
// blend work units of a heavy 32x32 tile. Header-only, tt-metal free (device
// kernels and tests/unit/test_subtile_unit.cpp).
//
// GSPLAT_TT_B1 splits a tile whose record count passes total / num_cores into
// units. A unit is its parent tile plus a microblock mask: its core walks the
// parent's materialized slab and blends only the masked microblocks, so every
// pixel sees the records it sees today, in the same order (lossless).
//
// Unit word (reader -> writer, CB_TILE_Q): tile_id | code << 16.
//   code 0       whole tile (32x32)
//   code 1..4    16x16 quarter q = code - 1 (qx = q & 1, qy = q >> 1); same masks
//                as render/host/subtile_split.h quarter_mask
//   code 5..20   16x4 band b = code - 5 (qx = b & 1, my = b >> 1): microblocks
//                (my, 2qx) and (my, 2qx + 1). 16 px wide, so the writer's row
//                segments start at byte 0 or 48 (16 B aligned); 8 px wide units
//                (subtile_split.h eighth_mask) would start at 24 B multiples.
// Microblock m = (my << 2) | mx: my 0..7 (4-pixel rows), mx 0..3 (8-pixel columns).
#pragma once

#include <cstdint>

namespace subtile_unit {

constexpr uint32_t kCodeShift = 16u;
constexpr uint32_t kTileBits = 0xFFFFu;
constexpr uint32_t kQuarter0 = 1u;
constexpr uint32_t kBand0 = 5u;
constexpr uint32_t kNumCodes = 21u;

inline constexpr uint32_t word(uint32_t tile, uint32_t code) { return tile | (code << kCodeShift); }
inline constexpr uint32_t tile_of(uint32_t w) { return w & kTileBits; }
inline constexpr uint32_t code_of(uint32_t w) { return w >> kCodeShift; }

inline constexpr uint32_t quarter_mask(uint32_t q) {
    // rows my 4qy..4qy+3, columns mx 2qx, 2qx+1: 0x33 per 2 rows.
    return 0x3333u << ((q & 1u) * 2u) << ((q >> 1) * 16u);
}

inline constexpr uint32_t band_mask(uint32_t b) { return 0x3u << ((b & 1u) * 2u) << ((b >> 1) * 4u); }

inline constexpr uint32_t mask(uint32_t code) {
    return code == 0u ? 0xFFFFFFFFu
         : code < kBand0 ? quarter_mask(code - kQuarter0)
         : band_mask(code - kBand0);
}

// Output region in the tile: first row, row count, first byte of a 96 B RGB row
// (3 bytes per pixel) and byte count.
inline constexpr uint32_t row0(uint32_t code) {
    return code == 0u ? 0u : code < kBand0 ? ((code - kQuarter0) >> 1) * 16u : ((code - kBand0) >> 1) * 4u;
}
inline constexpr uint32_t rows(uint32_t code) { return code == 0u ? 32u : code < kBand0 ? 16u : 4u; }
inline constexpr uint32_t col_byte0(uint32_t code) {
    return code == 0u ? 0u : code < kBand0 ? ((code - kQuarter0) & 1u) * 48u : ((code - kBand0) & 1u) * 48u;
}
inline constexpr uint32_t col_bytes(uint32_t code) { return code == 0u ? 96u : 48u; }

// Units of a split tile: n = 4 (quarters) or 16 (bands); unit k's code.
inline constexpr uint32_t split_code(uint32_t n, uint32_t k) { return n == 4u ? kQuarter0 + k : kBand0 + k; }

// Split plan, computed identically by every blend reader from the view's own
// per-tile record counts (exact, no previous-view estimate):
//   thr = total / num_cores; a tile is heavy if count > thr and count >= min_split;
//   heavy tiles go to the heavy list (count-descending, ties by ascending tile id,
//   at most cap entries: the rest stay whole, which only costs balance);
//   mode 1: 4 quarters each; mode 2: 16 bands if count > 4 * thr, else 4 quarters.
// Heavy entry: min(count, 0xFFFF) << 16 | tile.
inline constexpr uint32_t heavy_entry(uint32_t count, uint32_t tile) {
    return ((count < 0xFFFFu ? count : 0xFFFFu) << 16) | tile;
}
inline constexpr bool is_heavy(uint32_t count, uint32_t thr, uint32_t min_split) {
    return count > thr && count >= min_split;
}
inline constexpr uint32_t n_units(uint32_t count, uint32_t thr, uint32_t mode) {
    return (mode >= 2u && count > 4u * thr) ? 16u : 4u;
}
// Insert into the descending list h[0..n) (capacity cap). Tiles arrive in
// ascending id order, so equal counts keep ascending ids.
template <typename H>
inline void heavy_insert(H h, uint32_t& n, uint32_t cap, uint32_t count, uint32_t tile) {
    const uint32_t e = heavy_entry(count, tile);
    const uint32_t c = e >> 16;
    if (n == cap) {
        if (cap == 0u || (h[n - 1u] >> 16) >= c) return;
        --n;
    }
    uint32_t i = n;
    while (i > 0u && (h[i - 1u] >> 16) < c) {
        h[i] = h[i - 1u];
        --i;
    }
    h[i] = e;
    ++n;
}

}  // namespace subtile_unit

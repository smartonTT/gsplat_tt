// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Lever 2 (task #99, GSPLAT_TT_SFPU_VIS): code shared by the SFPU visibility /
// tile-AABB path. Header-only and device-agnostic: the data-mover kernels
// (writer_pfwc_vis, gather_vis_scan, gather_vis_scatter, tile_assign_scatter)
// and tests/unit/test_vis_lever2.cpp all use it.
//
// pfwc (project_pfwc_compute.cpp built with PFWC_VIS) evaluates the gather
// visibility predicate and the tile_assign K1 rectangle on the SFPU and packs
// two words per gaussian into two fp32 tiles. Every word is +0 or a normal
// float, so pack, unpack and DRAM keep the bits:
//   tpg word : 0 = not visible; TAG | tpg = visible, tpg = w * h >= 1;
//              RECHECK = an input was inf/NaN: the pfwc writer re-evaluates the
//              gaussian with the soft-float code (exact_words below).
//   aabb word: TAG | min_x | min_y << 10 | (w - 1) << 20, valid when visible.
// TAG | payload with payload < 2^29 has a biased exponent in [128, 191].
#pragma once

#include <cstdint>

#include "dm_fp32.h"
#include "gather_visible_pred.h"

namespace vis_tile {

constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t MASK_WORDS = TILE_ELEMS / 32;

constexpr uint32_t TAG = 0x40000000u;
constexpr uint32_t RECHECK = 0x20000000u;  // 2^-63: normal, used by no other word
constexpr uint32_t PAYLOAD = 0x1FFFFFFFu;
constexpr uint32_t XY_BITS = 10;
constexpr uint32_t XY_MASK = (1u << XY_BITS) - 1u;
constexpr uint32_t W_SHIFT = 2 * XY_BITS;
constexpr uint32_t W_MASK = 0x1FFu;
// Grid limits of the packing (min_x, min_y < 1024, w <= 512). The host keeps
// the legacy path for bigger grids.
constexpr uint32_t MAX_TILES_X = 512;
constexpr uint32_t MAX_TILES_Y = 1024;

// Per-tile counts written by the pfwc writer: [visible, pairs] per 1024-gaussian
// tile, dense, in 1 KB DRAM pages (128 tiles per page).
constexpr uint32_t COUNT_WORDS = 2;
constexpr uint32_t COUNTS_PAGE_BYTES = 1024;
constexpr uint32_t COUNTS_TILES_PER_PAGE = COUNTS_PAGE_BYTES / (COUNT_WORDS * 4);

inline uint32_t aabb_pack(uint32_t min_x, uint32_t min_y, uint32_t w) {
    return min_x | (min_y << XY_BITS) | ((w - 1u) << W_SHIFT);
}
inline uint32_t aabb_min_x(uint32_t p) { return p & XY_MASK; }
inline uint32_t aabb_min_y(uint32_t p) { return (p >> XY_BITS) & XY_MASK; }
inline uint32_t aabb_w(uint32_t p) { return ((p >> W_SHIFT) & W_MASK) + 1u; }

// Predicate + grid parameters, fp32 bits where the soft-float code wants bits.
struct Params {
    uint32_t k_near = 0, min_opacity = 0, img_w = 0, img_h = 0, max_radius = 0;
    uint32_t tiles_x = 1, tiles_y = 1;
    uint32_t tile_shift = 0;  // tile_size == 1 << tile_shift
    float inv_tile = 1.0f;    // 1.0f / tile_size, as tile_assign_bbox.cpp computes it
};

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// The exact words for one gaussian: gather_pred::visible_bits (the proj_count
// predicate), then the tile_assign_bbox.cpp K1 rectangle with the same dm_fp32
// calls. Returns the tpg word; *aabb gets the aabb word when visible.
inline uint32_t exact_words(uint32_t tz, uint32_t op, uint32_t mx, uint32_t my, uint32_t rx,
                            uint32_t ry, const Params& p, uint32_t* aabb) {
    using dm_fp32::SIGN;
    if (!gather_pred::visible_bits(tz, op, mx, my, rx, ry, p.k_near, p.min_opacity, p.img_w,
                                   p.img_h, p.max_radius))
        return 0u;
    const int tx1 = static_cast<int>(p.tiles_x) - 1, ty1 = static_cast<int>(p.tiles_y) - 1;
    const uint32_t s = p.tile_shift;
    const int min_x = clampi(dm_fp32::add_mul_pow2_to_int(mx, rx ^ SIGN, s, p.inv_tile), 0, tx1);
    const int max_x = clampi(dm_fp32::add_mul_pow2_to_int(mx, rx, s, p.inv_tile), 0, tx1);
    const int min_y = clampi(dm_fp32::add_mul_pow2_to_int(my, ry ^ SIGN, s, p.inv_tile), 0, ty1);
    const int max_y = clampi(dm_fp32::add_mul_pow2_to_int(my, ry, s, p.inv_tile), 0, ty1);
    const uint32_t w = static_cast<uint32_t>(max_x - min_x + 1);
    const uint32_t h = static_cast<uint32_t>(max_y - min_y + 1);
    *aabb = TAG | aabb_pack(static_cast<uint32_t>(min_x), static_cast<uint32_t>(min_y), w);
    return TAG | (w * h);
}

// pfwc writer: classify one output tile of n_el real gaussians (the rest is
// padding past N). Resolves RECHECK words in place with exact_words, builds the
// 1024-bit visibility mask (bit il % 32 of word il / 32, the proj_count layout)
// and counts the visible gaussians and their (gaussian, tile) pairs.
// get_inputs(il, tz, op, mx, my, rx, ry) supplies the bits of element il.
template <class GetInputs>
inline void classify_tile(volatile uint32_t* tpg, volatile uint32_t* aabb, uint32_t n_el,
                          volatile uint32_t* mask, const Params& p, GetInputs&& get_inputs,
                          uint32_t* vis_count, uint32_t* pair_count) {
    uint32_t vc = 0, pc = 0;
    for (uint32_t w = 0; w < MASK_WORDS; w++) {
        uint32_t bits = 0;
        const uint32_t il0 = w * 32;
        const uint32_t il1 = (il0 + 32 < n_el) ? il0 + 32 : n_el;
        for (uint32_t il = il0; il < il1; il++) {
            uint32_t t = tpg[il];
            if (t == RECHECK) {
                uint32_t tz, op, mx, my, rx, ry, ab = 0;
                get_inputs(il, tz, op, mx, my, rx, ry);
                t = exact_words(tz, op, mx, my, rx, ry, p, &ab);
                tpg[il] = t;
                if (t != 0u) aabb[il] = ab;
            }
            if (t & TAG) {
                bits |= 1u << (il - il0);
                pc += t & PAYLOAD;
            }
        }
        vc += static_cast<uint32_t>(__builtin_popcount(bits));
        mask[w] = bits;
    }
    *vis_count = vc;
    *pair_count = pc;
}

// The gather's compaction order (tasks #33 / #75, kept byte-identical): tiles are
// dealt to cores in a stride (core c owns c, c + C, c + 2C, ...) and compact
// indices follow those lists core by core. Position q of that sequence is tile
// c + k * C for the (c, k) of locate(q).
struct SeqMap {
    uint32_t C = 1, base = 0, rem = 0, big = 0;
    void init(uint32_t num_tiles, uint32_t num_cores) {
        C = num_cores;
        base = num_tiles / num_cores;
        rem = num_tiles % num_cores;
        big = rem * (base + 1u);
    }
    uint32_t count(uint32_t c) const { return base + (c < rem ? 1u : 0u); }
    uint32_t first(uint32_t c) const { return c * base + (c < rem ? c : rem); }
    void locate(uint32_t q, uint32_t* c, uint32_t* k) const {
        if (q < big) {
            *c = q / (base + 1u);
            *k = q - *c * (base + 1u);
        } else {  // base > 0 here: q < num_tiles and big == num_tiles when base == 0
            const uint32_t r = q - big;
            const uint32_t d = r / base;
            *c = rem + d;
            *k = r - d * base;
        }
    }
};

// Per-slot parameters, one 64 B page per (core, mover) slot = 2 * core + mover.
constexpr uint32_t SLOT_WORDS = 16;
constexpr uint32_t S_BASE = 0;     // compact index of the slot's first visible gaussian
constexpr uint32_t S_IS_LAST = 1;  // 1: last slot with a visible gaussian (pads the tail)
constexpr uint32_t S_Q0 = 2;       // first sequence position
constexpr uint32_t S_QN = 3;       // number of sequence positions
constexpr uint32_t S_PBASE = 4;    // pair offset of the slot's first visible gaussian

struct ScanArgs {
    uint32_t num_tiles = 0, num_cores = 1, movers = 2;
    uint32_t balance = 1;          // 0: the legacy per-core halves, 1: weighted cut
    uint32_t tile_weight = 0;      // weight of a tile with >= 1 visible gaussian
    uint32_t empty_weight = 0;     // weight of every tile (mask read)
    uint32_t split_permille = 500; // balance == 0: mover 0's share of a core's list
};

// gather_vis_scan: from the per-tile [visible, pairs] counts, cut the sequence
// into num_cores * movers contiguous slot ranges (the compaction order does not
// depend on the cut) and give each slot its compact base, pair base and is_last.
// balance == 1 cuts at equal shares of sum(visible + tile_weight * any + empty).
// Writes SLOT_WORDS words per slot to out; *M = visible total, *P = pair total.
inline void scan_slots(const volatile uint32_t* counts, const ScanArgs& a, volatile uint32_t* out,
                       uint32_t* M, uint32_t* P) {
    SeqMap sm;
    sm.init(a.num_tiles, a.num_cores);
    const uint32_t S = a.num_cores * a.movers;
    auto vis_of = [&](uint32_t t) { return counts[t * COUNT_WORDS + 0]; };
    auto pairs_of = [&](uint32_t t) { return counts[t * COUNT_WORDS + 1]; };
    auto weight = [&](uint32_t t) {
        const uint32_t v = vis_of(t);
        return v + (v ? a.tile_weight : 0u) + a.empty_weight;
    };
    // 1. Cut points: out[s][S_Q0] = first sequence position of slot s.
    if (!a.balance) {
        for (uint32_t c = 0; c < a.num_cores; c++) {
            const uint32_t n = sm.count(c), q0 = sm.first(c);
            const uint32_t h0 = (a.movers == 1) ? n : (n * a.split_permille + 500u) / 1000u;
            out[(c * a.movers) * SLOT_WORDS + S_Q0] = q0;
            if (a.movers == 2) out[(c * a.movers + 1) * SLOT_WORDS + S_Q0] = q0 + h0;
        }
    } else {
        uint64_t total = 0;
        for (uint32_t t = 0; t < a.num_tiles; t++) total += weight(t);
        uint64_t prefix = 0;
        uint32_t s = 1, q = 0;
        out[S_Q0] = 0;
        for (uint32_t c = 0; c < a.num_cores; c++) {
            const uint32_t n = sm.count(c);
            for (uint32_t k = 0; k < n; k++, q++) {
                while (s < S && prefix * S >= static_cast<uint64_t>(s) * total) {
                    out[s * SLOT_WORDS + S_Q0] = q;
                    s++;
                }
                prefix += weight(c + k * a.num_cores);
            }
        }
        for (; s < S; s++) out[s * SLOT_WORDS + S_Q0] = a.num_tiles;
    }
    // 2. Bases in slot order, is_last.
    uint32_t m = 0, pr = 0, last = 0;
    bool any = false;
    for (uint32_t s = 0; s < S; s++) {
        const uint32_t q0 = out[s * SLOT_WORDS + S_Q0];
        const uint32_t q1 = (s + 1 < S) ? out[(s + 1) * SLOT_WORDS + S_Q0] : a.num_tiles;
        out[s * SLOT_WORDS + S_BASE] = m;
        out[s * SLOT_WORDS + S_PBASE] = pr;
        out[s * SLOT_WORDS + S_QN] = q1 - q0;
        out[s * SLOT_WORDS + S_IS_LAST] = 0;
        if (q1 > q0) {
            uint32_t c, k;
            sm.locate(q0, &c, &k);
            for (uint32_t q = q0; q < q1; q++) {
                const uint32_t t = c + k * a.num_cores;
                const uint32_t v = vis_of(t);
                if (v) {
                    last = s;
                    any = true;
                }
                m += v;
                pr += pairs_of(t);
                if (++k == sm.count(c)) {
                    c++;
                    k = 0;
                }
            }
        }
    }
    if (any) out[last * SLOT_WORDS + S_IS_LAST] = 1;
    *M = m;
    *P = pr;
}

}  // namespace vis_tile

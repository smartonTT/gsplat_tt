// SPDX-License-Identifier: Apache-2.0
// Task #555 (B4' step 3b): placement of the one-launch sort's scratch buffers in
// the fused mat+blend program's L1 arena. During blend the mat CBs are dead
// except the heads blend aliases (CB5 MB_COUNTS, CB6 OUT, CB20 IMG_U8, kept at
// the alias-cut sizes); the sort of the next view (sort_bin_onelaunch.cpp with
// OL_ARENA=1, after the B4a K2) puts its scratch in the tails.
// Region r of the arena: [base(cb_r) + skip_r, base(cb_r) + skip_r + bytes_r),
// base(cb) = the mat CB's L1 base (the kernel reads it from its CB interface).
// A buffer's placement packs region << 24 | byte offset in the region; the
// kernel's address is base(cb_r) + skip_r + offset. First fit, largest buffer
// first, ALIGN-byte aligned (NoC DRAM reads on Blackhole need 64 B).
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gsplat_tt::sort_ol_arena {

inline constexpr uint32_t kAlign = 64u;
inline constexpr uint32_t kUnplaced = 0xFFFFFFFFu;
inline constexpr uint32_t kMaxOffset = 1u << 24;

struct Region {
    uint32_t cb;     // mat CB whose storage holds the region
    uint32_t skip;   // bytes kept for blend at the CB's base (alias cut), ALIGN multiple
    uint32_t bytes;  // usable bytes after skip
};
struct Buf {
    uint32_t id;     // scratch buffer id (sort CB id, + 16 for mover 0)
    uint32_t bytes;
};

inline uint32_t align_up(uint32_t v) { return (v + kAlign - 1u) / kAlign * kAlign; }

// One packed placement per buffer (same order as bufs), or an empty vector if
// the buffers do not fit.
inline std::vector<uint32_t> place(const std::vector<Region>& regions, const std::vector<Buf>& bufs) {
    std::vector<uint32_t> order(bufs.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return bufs[a].bytes > bufs[b].bytes; });
    std::vector<uint32_t> used(regions.size(), 0u), out(bufs.size(), kUnplaced);
    for (const uint32_t i : order) {
        const uint32_t need = align_up(bufs[i].bytes);
        for (uint32_t r = 0; r < regions.size(); ++r) {
            if (regions[r].skip % kAlign != 0u) return {};
            if (used[r] + need <= regions[r].bytes && used[r] < kMaxOffset) {
                out[i] = (r << 24) | used[r];
                used[r] += need;
                break;
            }
        }
        if (out[i] == kUnplaced) return {};
    }
    return out;
}

inline uint32_t region_of(uint32_t packed) { return packed >> 24; }
inline uint32_t offset_of(uint32_t packed) { return packed & (kMaxOffset - 1u); }

}  // namespace gsplat_tt::sort_ol_arena

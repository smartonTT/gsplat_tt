// Task #190: the tail-chained blend walk (alpha_blend_compute_mb.cpp,
// BLEND_CHAIN_WALK 1 and 2) visits the same table slots in the same order as
// the jump walk (dispatch_blend_jump), and the branch-free form ends on an end
// slot (4 * pair + 0) inside the 64-entry table.
//   tests/unit/run_cpp.sh tests/unit/test_blend_chain_walk.cpp
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "render/kernels/compute/blend_chain_walk.h"

namespace {

// dispatch_blend_jump: index 2b + pm, b = even bit of the lowest live pair.
std::vector<uint32_t> jump_walk(uint32_t mask) {
    std::vector<uint32_t> v;
    while (mask != 0u) {
        const uint32_t b = static_cast<uint32_t>(__builtin_ctz(mask)) & ~1u;
        v.push_back(2u * b + ((mask >> b) & 3u));
        mask &= ~(3u << b);
    }
    return v;
}

// The call site plus blend_chain_body<J, PM>; returns false on a bad index.
bool chain_walk(uint32_t mask, int variant, std::vector<uint32_t>& v) {
    v.clear();
    uint32_t idx = blend_chain_index(mask, 0u);
    for (;;) {
        if (idx >= 64u) return false;
        if ((idx & 3u) == 0u) return variant == 2;  // end stub
        v.push_back(idx);
        const uint32_t j = idx / 4u;
        if (j + 1u >= 16u) return true;  // pair 15 returns
        const uint32_t rest = mask >> (2u * j + 2u);
        if (variant == 1 && rest == 0u) return true;
        idx = blend_chain_index(rest, j + 1u);
        if (variant == 1 && (idx & 3u) == 0u) return false;  // must not happen
    }
}

}  // namespace

int main() {
    uint64_t checks = 0, bad = 0;
    std::vector<uint32_t> c;
    auto check = [&](uint32_t mask) {
        if (mask == 0u) return;  // the call site skips dead records
        const std::vector<uint32_t> j = jump_walk(mask);
        for (int variant = 1; variant <= 2; ++variant) {
            ++checks;
            if ((!chain_walk(mask, variant, c) || c != j) && bad++ < 10) {
                std::printf("mismatch mask=%08x variant=%d\n", mask, variant);
            }
        }
    };
    // Every pair-occupancy pattern with every pm (1..3) of its lowest pair,
    // its highest pair and a mixed fill; every mask of up to 3 pairs below.
    for (uint32_t occ = 1; occ < (1u << 16); ++occ) {
        for (uint32_t fill = 0; fill < 3u; ++fill) {
            uint32_t m = 0u;
            uint32_t k = 0u;
            for (uint32_t p = 0; p < 16u; ++p) {
                if (occ & (1u << p)) {
                    m |= (((fill + k * 7u + p) % 3u) + 1u) << (2u * p);
                    ++k;
                }
            }
            check(m);
            check(m | 3u);
            check(m | 0xC0000000u);
        }
    }
    for (uint32_t a = 0; a < 32u; ++a) {
        for (uint32_t b = a; b < 32u; ++b) {
            for (uint32_t d = b; d < 32u; ++d) {
                check((1u << a) | (1u << b) | (1u << d));
            }
        }
    }
    std::mt19937 rng(190);
    for (int it = 0; it < 2000000; ++it) {
        check(rng() & rng());
    }
    check(0xFFFFFFFFu);
    check(0x80000000u);
    check(0x00000001u);
    // End slots of the branch-free form: rest == 0 -> 4 * base_pair.
    for (uint32_t p = 1; p < 16u; ++p) {
        ++checks;
        if (blend_chain_index(0u, p) != 4u * p && bad++ < 10) {
            std::printf("end slot p=%u -> %u\n", p, blend_chain_index(0u, p));
        }
    }
    std::printf("%llu checks, %llu bad\n", static_cast<unsigned long long>(checks),
                static_cast<unsigned long long>(bad));
    return bad == 0 ? 0 : 1;
}

// SPDX-License-Identifier: Apache-2.0
//
// Task #190: table index of the tail-chained blend mask walk. The 32-bit
// microblock mask is 16 pairs of 2 bits; the body table holds pair p with
// 2-bit value pm at index 4 * p + pm. Slot 4 * p + 0 (pm 0 is never live)
// holds the end stub, which only returns.
//
// rest = mask >> (2 * base_pair): the pairs from base_pair on. The result is
// the first live pair in rest, or the end slot 4 * base_pair when rest is 0:
// ctz(0) is 32 (the TRISC's Zbb ctz) and c & 30 maps it to 0, so the walk
// needs no branch, and the shift stays below 32.
#pragma once

#include <cstdint>

#if defined(__has_builtin)
#if __has_builtin(__builtin_ctzg)
#define BLEND_CHAIN_CTZG 1
#endif
#endif

inline uint32_t blend_chain_index(uint32_t rest, uint32_t base_pair) {
#if defined(BLEND_CHAIN_CTZG)
    const uint32_t c = static_cast<uint32_t>(__builtin_ctzg(rest, 32));
#else
    const uint32_t c = rest != 0u ? static_cast<uint32_t>(__builtin_ctz(rest)) : 32u;
#endif
    const uint32_t b = c & 30u;  // even bit of the pair; 32 -> 0
    return 4u * base_pair + 2u * b + ((rest >> b) & 3u);
}

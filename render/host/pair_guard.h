// Task #213: two small helpers for the fused K2 (tile_assign_device.cpp), kept
// device-free so tests/unit/test_pair_guard.cpp can check them on the host.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace gsplat_tt::pair_guard {

// Calls f(); if it throws, calls drain() before the exception leaves, so a
// non-blocking device read into a stack buffer completes before the stack
// unwinds (the no-CQ1 proj_M ReadShard in tile_assign_fused_k2).
template <class F, class D>
auto drain_on_throw(F&& f, D&& drain) -> decltype(f()) {
    try {
        return f();
    } catch (...) {
        drain();
        throw;
    }
}

// Bytes to allocate per pair buffer (gid/tid/keep) for p_bytes of pairs: at
// least the static pair ceiling, or test_cap pairs instead when nonzero (a
// small test_cap forces the K2 pair-overflow regrow). Whole 1024-pair pages.
inline std::size_t pair_alloc_bytes(std::size_t p_bytes, uint32_t ceiling, uint32_t test_cap) {
    const uint32_t floor_pairs = test_cap != 0 ? test_cap : ceiling;
    const std::size_t floor_bytes = (static_cast<std::size_t>(floor_pairs) + 1023u) / 1024u * 1024u * 4u;
    return std::max(p_bytes, floor_bytes);
}

}  // namespace gsplat_tt::pair_guard

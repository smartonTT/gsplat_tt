#include "gsplat_cpu/sort.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gsplat_cpu {

namespace {

int64_t compose_sort_key(const int64_t tile_id, const float depth) {
    const uint32_t depth_bits = std::bit_cast<uint32_t>(depth);
    return (tile_id << 32) | static_cast<int64_t>(depth_bits);
}

}  // namespace

SortResult sort_and_bin(
    const int64_t* gaussian_ids,
    const int64_t* tile_ids,
    const float* depths,
    const std::size_t P,
    const std::size_t M,
    const int tiles_x,
    const int tiles_y) {
    (void)M;

    const int num_tiles = tiles_x * tiles_y;
    SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (P == 0) {
        return result;
    }

    std::vector<int64_t> keys(P);
    for (std::size_t i = 0; i < P; ++i) {
        keys[i] = compose_sort_key(tile_ids[i], depths[static_cast<std::size_t>(gaussian_ids[i])]);
    }

    std::vector<std::size_t> perm(P);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&keys](const std::size_t a, const std::size_t b) { return keys[a] < keys[b]; });

    result.sorted_gaussian_ids.resize(P);
    std::vector<int64_t> sorted_tile_ids(P);
    for (std::size_t i = 0; i < P; ++i) {
        const std::size_t src = perm[i];
        result.sorted_gaussian_ids[i] = gaussian_ids[src];
        sorted_tile_ids[i] = tile_ids[src];
    }

    int64_t prev_tile = sorted_tile_ids[0];
    result.tile_ranges[static_cast<std::size_t>(prev_tile) * 2 + 0] = 0;

    for (std::size_t i = 1; i < P; ++i) {
        const int64_t curr_tile = sorted_tile_ids[i];
        if (curr_tile != prev_tile) {
            result.tile_ranges[static_cast<std::size_t>(prev_tile) * 2 + 1] =
                static_cast<int64_t>(i);
            result.tile_ranges[static_cast<std::size_t>(curr_tile) * 2 + 0] =
                static_cast<int64_t>(i);
            prev_tile = curr_tile;
        }
    }

    result.tile_ranges[static_cast<std::size_t>(prev_tile) * 2 + 1] = static_cast<int64_t>(P);

    return result;
}

}  // namespace gsplat_cpu

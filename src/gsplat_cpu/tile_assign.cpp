#include "gsplat_cpu/tile_assign.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

namespace {

int clamp_int(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

float clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

}  // namespace

TileAssignResult tile_assign(
    const float* means_2d,
    const float* radii,
    const std::size_t M,
    const int image_height,
    const int image_width,
    const int tile_size,
    const float* covs_2d,
    const float* opacities,
    const float contrib_floor) {
    TileAssignResult result;
    result.tiles_per_gaussian.assign(M, 0);

    if (M == 0) {
        return result;
    }

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;

    std::vector<int> tile_min_x(M);
    std::vector<int> tile_max_x(M);
    std::vector<int> tile_min_y(M);
    std::vector<int> tile_max_y(M);
    std::vector<int> widths(M);
    std::vector<int> heights(M);

    std::size_t P = 0;
    for (std::size_t m = 0; m < M; ++m) {
        const float px = means_2d[m * 2 + 0];
        const float py = means_2d[m * 2 + 1];
        const float rx = radii[m * 2 + 0];
        const float ry = radii[m * 2 + 1];

        tile_min_x[m] =
            clamp_int(static_cast<int>((px - rx) / static_cast<float>(tile_size)), 0, tiles_x - 1);
        tile_max_x[m] =
            clamp_int(static_cast<int>((px + rx) / static_cast<float>(tile_size)), 0, tiles_x - 1);
        tile_min_y[m] =
            clamp_int(static_cast<int>((py - ry) / static_cast<float>(tile_size)), 0, tiles_y - 1);
        tile_max_y[m] =
            clamp_int(static_cast<int>((py + ry) / static_cast<float>(tile_size)), 0, tiles_y - 1);

        widths[m] = tile_max_x[m] - tile_min_x[m] + 1;
        heights[m] = tile_max_y[m] - tile_min_y[m] + 1;
        result.tiles_per_gaussian[m] = static_cast<int64_t>(widths[m]) * static_cast<int64_t>(heights[m]);
        P += static_cast<std::size_t>(result.tiles_per_gaussian[m]);
    }

    result.gaussian_ids.reserve(P);
    result.tile_ids.reserve(P);

    for (std::size_t m = 0; m < M; ++m) {
        for (int dy = 0; dy < heights[m]; ++dy) {
            for (int dx = 0; dx < widths[m]; ++dx) {
                result.gaussian_ids.push_back(static_cast<int64_t>(m));
                result.tile_ids.push_back(static_cast<int64_t>((tile_min_y[m] + dy) * tiles_x +
                                                                (tile_min_x[m] + dx)));
            }
        }
    }

    if (covs_2d != nullptr && opacities != nullptr && P > 0) {
        std::vector<int64_t> kept_gids;
        std::vector<int64_t> kept_tids;
        kept_gids.reserve(P);
        kept_tids.reserve(P);

        for (std::size_t p = 0; p < P; ++p) {
            const int64_t g = result.gaussian_ids[p];
            const int64_t tile_id = result.tile_ids[p];

            const float a = covs_2d[static_cast<std::size_t>(g) * 4 + 0];
            const float b = covs_2d[static_cast<std::size_t>(g) * 4 + 1];
            const float c = covs_2d[static_cast<std::size_t>(g) * 4 + 3];
            const float det = std::max(a * c - b * b, 1e-6f);

            const float px = means_2d[static_cast<std::size_t>(g) * 2 + 0];
            const float py = means_2d[static_cast<std::size_t>(g) * 2 + 1];

            const float tx_tile =
                static_cast<float>(tile_id % static_cast<int64_t>(tiles_x)) *
                static_cast<float>(tile_size);
            const float ty_tile =
                static_cast<float>(tile_id / static_cast<int64_t>(tiles_x)) *
                static_cast<float>(tile_size);

            const float cx = clamp_float(px, tx_tile, tx_tile + static_cast<float>(tile_size));
            const float cy = clamp_float(py, ty_tile, ty_tile + static_cast<float>(tile_size));
            const float dx_c = cx - px;
            const float dy_c = cy - py;
            const float m2 =
                (c * dx_c * dx_c - 2.0f * b * dx_c * dy_c + a * dy_c * dy_c) / det;
            const float contrib = opacities[static_cast<std::size_t>(g)] * std::exp(-0.5f * m2);

            if (contrib >= contrib_floor) {
                kept_gids.push_back(g);
                kept_tids.push_back(tile_id);
            }
        }

        result.gaussian_ids = std::move(kept_gids);
        result.tile_ids = std::move(kept_tids);

        std::fill(result.tiles_per_gaussian.begin(), result.tiles_per_gaussian.end(), 0);
        for (const int64_t g : result.gaussian_ids) {
            ++result.tiles_per_gaussian[static_cast<std::size_t>(g)];
        }
    }

    return result;
}

}  // namespace gsplat_cpu

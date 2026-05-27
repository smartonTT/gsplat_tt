#include "gsplat_cpu/blend_microblock.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gsplat_cpu {

namespace {

constexpr int kNumMicroblocks = 32;

void blend_microblock_tile(
    const int tile_id,
    const int tiles_x,
    const int tile_size,
    const int image_height,
    const int image_width,
    const float* means_2d,
    const float* colors,
    const float* opacities,
    const int64_t* mb_header,
    const int64_t* mb_stream,
    const float* cov_inv_a,
    const float* cov_inv_b,
    const float* cov_inv_c,
    float* image_out) {
    const int ty = tile_id / tiles_x;
    const int tx = tile_id % tiles_x;
    const int py_tile = ty * tile_size;
    const int px_tile = tx * tile_size;
    const int py_end_tile = std::min(py_tile + tile_size, image_height);
    const int px_end_tile = std::min(px_tile + tile_size, image_width);

    const std::size_t header_base =
        (static_cast<std::size_t>(tile_id) * static_cast<std::size_t>(kNumMicroblocks)) * 2;

    for (int m = 0; m < kNumMicroblocks; ++m) {
        const std::size_t hdr_idx = header_base + static_cast<std::size_t>(m) * 2;
        const int64_t off = mb_header[hdr_idx + 0];
        const int64_t cnt = mb_header[hdr_idx + 1];
        if (cnt == 0) {
            continue;
        }

        const int mb_ox = (m & 3) * 8;
        const int mb_oy = (m >> 2) * 4;
        const int py_start = py_tile + mb_oy;
        const int px_start = px_tile + mb_ox;
        const int py_end = std::min(py_start + 4, py_end_tile);
        const int px_end = std::min(px_start + 8, px_end_tile);
        if (py_start >= py_end || px_start >= px_end) {
            continue;
        }

        const int mb_h = py_end - py_start;
        const int mb_w = px_end - px_start;
        const int npix = mb_h * mb_w;

        float T[4 * 8];
        float accum[4 * 8 * 3];
        for (int k = 0; k < npix; ++k) {
            T[k] = 1.0f;
        }
        for (int k = 0; k < npix * 3; ++k) {
            accum[k] = 0.0f;
        }

        for (int64_t idx = off; idx < off + cnt; ++idx) {
            const int64_t g = mb_stream[static_cast<std::size_t>(idx)];
            const float ci_a = cov_inv_a[static_cast<std::size_t>(g)];
            const float ci_b = cov_inv_b[static_cast<std::size_t>(g)];
            const float ci_c = cov_inv_c[static_cast<std::size_t>(g)];
            const float mx = means_2d[static_cast<std::size_t>(g) * 2 + 0];
            const float my = means_2d[static_cast<std::size_t>(g) * 2 + 1];
            const float opacity = opacities[static_cast<std::size_t>(g)];
            const float cr = colors[static_cast<std::size_t>(g) * 3 + 0];
            const float cg = colors[static_cast<std::size_t>(g) * 3 + 1];
            const float cb = colors[static_cast<std::size_t>(g) * 3 + 2];

            for (int i = 0; i < mb_h; ++i) {
                const float py = static_cast<float>(py_start + i) + 0.5f;
                for (int j = 0; j < mb_w; ++j) {
                    const float px = static_cast<float>(px_start + j) + 0.5f;
                    const float dx = px - mx;
                    const float dy = py - my;
                    const float power =
                        -0.5f * (ci_a * dx * dx + 2.0f * ci_b * dx * dy + ci_c * dy * dy);
                    const float gw = std::exp(std::min(power, 0.0f));
                    const float alpha = std::min(opacity * gw, 0.99f);
                    const int ij = i * mb_w + j;
                    const float t = T[ij];
                    const float at = alpha * t;
                    accum[ij * 3 + 0] += at * cr;
                    accum[ij * 3 + 1] += at * cg;
                    accum[ij * 3 + 2] += at * cb;
                    T[ij] = t * (1.0f - alpha);
                }
            }

            float tmax = 0.0f;
            for (int k = 0; k < npix; ++k) {
                tmax = std::max(tmax, T[k]);
            }
            if (tmax < 0.0001f) {
                break;
            }
        }

        for (int i = 0; i < mb_h; ++i) {
            const int gy = py_start + i;
            for (int j = 0; j < mb_w; ++j) {
                const int gx = px_start + j;
                const int out_base = (gy * image_width + gx) * 3;
                const int ij = i * mb_w + j;
                image_out[out_base + 0] = accum[ij * 3 + 0];
                image_out[out_base + 1] = accum[ij * 3 + 1];
                image_out[out_base + 2] = accum[ij * 3 + 2];
            }
        }
    }
}

}  // namespace

BlendResult blend_microblock(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* mb_header,
    const int64_t* mb_stream,
    const std::size_t M,
    const std::size_t L_prime,
    const int image_height,
    const int image_width,
    const int tile_size,
    ThreadPool& pool) {
    (void)L_prime;

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;
    const int num_tiles = tiles_x * tiles_y;

    BlendResult result;
    result.image.resize(static_cast<std::size_t>(image_height) *
                            static_cast<std::size_t>(image_width) * 3,
                        0.0f);

    std::vector<float> cov_inv_a(M);
    std::vector<float> cov_inv_b(M);
    std::vector<float> cov_inv_c(M);
    for (std::size_t i = 0; i < M; ++i) {
        const float a = covs_2d[i * 4 + 0];
        const float b = covs_2d[i * 4 + 1];
        const float c = covs_2d[i * 4 + 3];
        const float det = std::max(a * c - b * b, 1e-6f);
        cov_inv_a[i] = c / det;
        cov_inv_b[i] = -b / det;
        cov_inv_c[i] = a / det;
    }

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        pool.submit([&, tile_id]() {
            blend_microblock_tile(tile_id, tiles_x, tile_size, image_height, image_width,
                                  means_2d, colors, opacities, mb_header, mb_stream,
                                  cov_inv_a.data(), cov_inv_b.data(), cov_inv_c.data(),
                                  result.image.data());
        });
    }
    pool.wait();

    return result;
}

}  // namespace gsplat_cpu

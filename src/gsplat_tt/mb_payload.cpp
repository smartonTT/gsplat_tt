#include "gsplat_tt/mb_payload.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

#include "gsplat_cpu/thread_pool.h"

namespace gsplat_tt {

namespace {

constexpr int kNumMb = kMbNumMicroblocks;  // 32

// Per-tile staging produced in parallel, concatenated serially afterward.
struct TileOut {
    std::vector<float> coeff;             // L * kMbCoeffLanes
    std::vector<uint32_t> stream;         // L'
    std::array<uint32_t, kNumMb * 2> header{};  // (offset,count) per mb
    int64_t dropped = 0;
    int64_t kept = 0;
};

// Per-(tile) cull + basis conversion. Mirrors cull_and_blend_tile's
// constrained-min Mahalanobis cull exactly so the kept set is identical;
// emits the basis-form coeff rows + microblock stream instead of blending.
void build_tile(
    int tile_id,
    int tiles_x,
    int tile_size,
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    float mb_contrib_floor,
    bool cull_disabled,
    TileOut& out) {
    const int64_t start = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 0];
    const int64_t end = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 1];
    const int64_t L = end - start;
    out.header.fill(0);
    if (L <= 0) return;

    const int ty = tile_id / tiles_x;
    const int tx = tile_id % tiles_x;
    const float tx_tile = static_cast<float>(tx * tile_size);
    const float ty_tile = static_cast<float>(ty * tile_size);

    out.coeff.assign(static_cast<std::size_t>(L) * kMbCoeffLanes, 0.0f);

    // Per-microblock kept local-index lists (depth order preserved by walking
    // l = 0..L-1 in sorted order). 32 small vectors.
    std::array<std::vector<uint32_t>, kNumMb> mb_keep;

    for (int64_t l = 0; l < L; ++l) {
        const int64_t g = sorted_gaussian_ids[static_cast<std::size_t>(start + l)];
        const float a = covs_2d[static_cast<std::size_t>(g) * 4 + 0];
        const float b = covs_2d[static_cast<std::size_t>(g) * 4 + 1];
        const float c = covs_2d[static_cast<std::size_t>(g) * 4 + 3];
        const float det = std::max(a * c - b * b, 1e-6f);
        const float ci_a = c / det;
        const float ci_b = -b / det;
        const float ci_c = a / det;
        const float opacity = opacities[static_cast<std::size_t>(g)];
        const float mean_x = means_2d[static_cast<std::size_t>(g) * 2 + 0];
        const float mean_y = means_2d[static_cast<std::size_t>(g) * 2 + 1];
        const float mx_local = mean_x - tx_tile;
        const float my_local = mean_y - ty_tile;

        // Conic (centered) form — numerically stable. The expanded basis form
        // (A x^2+B xy+C y^2+D x+E y+F) suffers catastrophic fp32 cancellation
        // for tight gaussians (conic ~1e6 => individual terms ~1e8 that must
        // cancel to a small power near the center). The SFPU loses that
        // precision and the weight degenerates to binary noise. Centered form
        // keeps all near-center terms small:
        //   power = A dx^2 + B dx dy + C dy^2,  dx=x-mx, dy=y-my
        // where A=-0.5 ci_a, B=-ci_b, C=-0.5 ci_c (algebraically identical).
        float* row = &out.coeff[static_cast<std::size_t>(l) * kMbCoeffLanes];
        row[0] = -0.5f * ci_a;                                  // A
        row[1] = -ci_b;                                         // B
        row[2] = -0.5f * ci_c;                                  // C
        row[3] = mx_local;                                      // gaussian center x (tile-local)
        row[4] = my_local;                                      // gaussian center y (tile-local)
        row[5] = 0.0f;                                          // unused
        row[6] = opacity;
        row[7] = colors[static_cast<std::size_t>(g) * 3 + 0];
        row[8] = colors[static_cast<std::size_t>(g) * 3 + 1];
        row[9] = colors[static_cast<std::size_t>(g) * 3 + 2];

        if (opacity <= mb_contrib_floor) {
            ++out.dropped;
            continue;  // sentinel: peak alpha below floor everywhere.
        }
        const float log_thresh = std::log(mb_contrib_floor / opacity);  // <= 0
        const float rd = std::sqrt(-2.0f * log_thresh);
        const float x_half = rd * std::sqrt(std::max(a, 0.0f));
        const float y_half = rd * std::sqrt(std::max(c, 0.0f));

        const float bb_x_min = mean_x - x_half;
        const float bb_x_max = mean_x + x_half;
        const float bb_y_min = mean_y - y_half;
        const float bb_y_max = mean_y + y_half;
        const int mx_lo = std::max(0, static_cast<int>(std::floor((bb_x_min - tx_tile) / 8.0f)));
        const int mx_hi = std::min(3, static_cast<int>(std::floor((bb_x_max - tx_tile) / 8.0f)));
        const int my_lo = std::max(0, static_cast<int>(std::floor((bb_y_min - ty_tile) / 4.0f)));
        const int my_hi = std::min(7, static_cast<int>(std::floor((bb_y_max - ty_tile) / 4.0f)));
        if (mx_lo > mx_hi || my_lo > my_hi) {
            ++out.dropped;
            continue;
        }

        const float thresh_m2 = -2.0f * log_thresh;
        const float ci_a_safe = std::max(ci_a, 1e-12f);
        const float ci_c_safe = std::max(ci_c, 1e-12f);
        bool kept_any = false;
        for (int my = my_lo; my <= my_hi; ++my) {
            const float mb_oy = ty_tile + static_cast<float>(my * 4);
            const float v_lo = mb_oy - mean_y;
            const float v_hi = v_lo + 4.0f;
            const bool y_inside = (v_lo <= 0.0f) && (0.0f <= v_hi);
            const float v_fix = (v_lo > 0.0f) ? v_lo : v_hi;
            for (int mx = mx_lo; mx <= mx_hi; ++mx) {
                const float mb_ox = tx_tile + static_cast<float>(mx * 8);
                const float u_lo = mb_ox - mean_x;
                const float u_hi = u_lo + 8.0f;
                const bool x_inside = (u_lo <= 0.0f) && (0.0f <= u_hi);

                float m2_min;
                if (cull_disabled || (x_inside && y_inside)) {
                    m2_min = 0.0f;
                } else {
                    float m2_v = std::numeric_limits<float>::infinity();
                    if (!x_inside) {
                        const float u_fix = (u_lo > 0.0f) ? u_lo : u_hi;
                        float v_star = -ci_b * u_fix / ci_c_safe;
                        v_star = std::clamp(v_star, v_lo, v_hi);
                        m2_v = ci_a * u_fix * u_fix + 2.0f * ci_b * u_fix * v_star
                               + ci_c * v_star * v_star;
                    }
                    float m2_h = std::numeric_limits<float>::infinity();
                    if (!y_inside) {
                        float u_star = -ci_b * v_fix / ci_a_safe;
                        u_star = std::clamp(u_star, u_lo, u_hi);
                        m2_h = ci_a * u_star * u_star + 2.0f * ci_b * u_star * v_fix
                               + ci_c * v_fix * v_fix;
                    }
                    m2_min = std::min(m2_v, m2_h);
                }
                if (m2_min <= thresh_m2) {
                    const int m = (my << 2) | mx;  // canonical enumeration
                    mb_keep[static_cast<std::size_t>(m)].push_back(static_cast<uint32_t>(l));
                    kept_any = true;
                    ++out.kept;
                }
            }
        }
        if (!kept_any) ++out.dropped;
    }

    // Flatten per-microblock lists into header + stream.
    uint32_t off = 0;
    for (int m = 0; m < kNumMb; ++m) {
        const auto& v = mb_keep[static_cast<std::size_t>(m)];
        out.header[static_cast<std::size_t>(m) * 2 + 0] = off;
        out.header[static_cast<std::size_t>(m) * 2 + 1] = static_cast<uint32_t>(v.size());
        out.stream.insert(out.stream.end(), v.begin(), v.end());
        off += static_cast<uint32_t>(v.size());
    }
}

}  // namespace

MbPayload build_mb_payload(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t /*M*/,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int /*image_height*/,
    int /*image_width*/,
    float mb_contrib_floor,
    gsplat_cpu::ThreadPool& pool,
    bool cull_disabled) {
    const int num_tiles = tiles_x * tiles_y;
    MbPayload p;
    p.num_tiles = num_tiles;
    p.tiles_x = tiles_x;
    p.tiles_y = tiles_y;
    p.tile_size = tile_size;
    p.pairs_in = static_cast<int64_t>(P);

    std::vector<TileOut> outs(static_cast<std::size_t>(num_tiles));

    // Parallel per-tile build (LPT-ish: just round-robin via atomic counter).
    std::atomic<int> next_idx{0};
    const std::size_t W = std::max<std::size_t>(1, pool.size());
    for (std::size_t w = 0; w < W; ++w) {
        pool.submit([&]() {
            for (;;) {
                const int t = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= num_tiles) return;
                build_tile(t, tiles_x, tile_size, means_2d, covs_2d, colors,
                           opacities, sorted_gaussian_ids, tile_ranges,
                           mb_contrib_floor, cull_disabled,
                           outs[static_cast<std::size_t>(t)]);
            }
        });
    }
    pool.wait();

    // Serial concat into flat buffers + per-tile offsets.
    p.coeff_tile_off.assign(static_cast<std::size_t>(num_tiles) + 1, 0);
    p.mb_stream_tile_off.assign(static_cast<std::size_t>(num_tiles) + 1, 0);
    p.mb_header.assign(static_cast<std::size_t>(num_tiles) * kNumMb * 2, 0);

    std::size_t total_rows = 0, total_stream = 0;
    for (int t = 0; t < num_tiles; ++t) {
        const TileOut& o = outs[static_cast<std::size_t>(t)];
        total_rows += o.coeff.size() / kMbCoeffLanes;
        total_stream += o.stream.size();
    }
    p.coeff.reserve(total_rows * kMbCoeffLanes);
    p.mb_stream.reserve(total_stream);

    for (int t = 0; t < num_tiles; ++t) {
        const TileOut& o = outs[static_cast<std::size_t>(t)];
        p.coeff_tile_off[static_cast<std::size_t>(t)] =
            static_cast<uint32_t>(p.coeff.size() / kMbCoeffLanes);
        p.mb_stream_tile_off[static_cast<std::size_t>(t)] =
            static_cast<uint32_t>(p.mb_stream.size());
        p.coeff.insert(p.coeff.end(), o.coeff.begin(), o.coeff.end());
        p.mb_stream.insert(p.mb_stream.end(), o.stream.begin(), o.stream.end());
        std::copy(o.header.begin(), o.header.end(),
                  p.mb_header.begin() + static_cast<std::size_t>(t) * kNumMb * 2);
        p.pairs_dropped_all_mb += o.dropped;
        p.pairs_kept_per_mb += o.kept;
    }
    p.coeff_tile_off[static_cast<std::size_t>(num_tiles)] =
        static_cast<uint32_t>(p.coeff.size() / kMbCoeffLanes);
    p.mb_stream_tile_off[static_cast<std::size_t>(num_tiles)] =
        static_cast<uint32_t>(p.mb_stream.size());
    return p;
}

void blend_from_mb_payload_cpu(
    const MbPayload& p,
    int image_height,
    int image_width,
    float transmittance_threshold,
    float* image_out) {
    for (int t = 0; t < p.num_tiles; ++t) {
        const int ty = t / p.tiles_x;
        const int tx = t % p.tiles_x;
        const int px_tile = tx * p.tile_size;
        const int py_tile = ty * p.tile_size;
        const float* coeff_base =
            &p.coeff[static_cast<std::size_t>(p.coeff_tile_off[static_cast<std::size_t>(t)]) * kMbCoeffLanes];
        const uint32_t* stream_base =
            p.mb_stream.data() + p.mb_stream_tile_off[static_cast<std::size_t>(t)];
        const uint32_t* hdr = &p.mb_header[static_cast<std::size_t>(t) * kNumMb * 2];

        for (int m = 0; m < kNumMb; ++m) {
            const uint32_t off = hdr[m * 2 + 0];
            const uint32_t cnt = hdr[m * 2 + 1];
            if (cnt == 0) continue;
            const int col_group = m & 3;
            const int row_band = m >> 2;
            const int px_start = px_tile + col_group * 8;
            const int py_start = py_tile + row_band * 4;
            const int px_end = std::min(px_start + 8, image_width);
            const int py_end = std::min(py_start + 4, image_height);
            if (px_start >= px_end || py_start >= py_end) continue;

            float T[4 * 8];
            float acc[4 * 8 * 3];
            for (int k = 0; k < 32; ++k) T[k] = 1.0f;
            for (int k = 0; k < 32 * 3; ++k) acc[k] = 0.0f;

            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t gidx = stream_base[off + i];
                const float* row = coeff_base + static_cast<std::size_t>(gidx) * kMbCoeffLanes;
                const float A = row[0], B = row[1], C = row[2];
                const float mx = row[3], my = row[4];
                const float opacity = row[6];
                const float cr = row[7], cg = row[8], cb = row[9];
                for (int r = 0; r < 4; ++r) {
                    const float y = static_cast<float>(row_band * 4 + r) + 0.5f;
                    for (int cc = 0; cc < 8; ++cc) {
                        const float x = static_cast<float>(col_group * 8 + cc) + 0.5f;
                        const float dx = x - mx;
                        const float dy = y - my;
                        const float power = A * dx * dx + B * dx * dy + C * dy * dy;
                        const float weight = std::exp(std::min(power, 0.0f));
                        const float alpha = std::min(opacity * weight, 0.99f);
                        const int ij = r * 8 + cc;
                        const float tt = T[ij];
                        const float at = alpha * tt;
                        acc[ij * 3 + 0] += at * cr;
                        acc[ij * 3 + 1] += at * cg;
                        acc[ij * 3 + 2] += at * cb;
                        T[ij] = tt * (1.0f - alpha);
                    }
                }
                float tmax = 0.0f;
                for (int k = 0; k < 32; ++k) tmax = std::max(tmax, T[k]);
                if (tmax < transmittance_threshold) break;
            }

            for (int r = 0; py_start + r < py_end; ++r) {
                const int gy = py_start + r;
                for (int cc = 0; px_start + cc < px_end; ++cc) {
                    const int gx = px_start + cc;
                    const int ij = r * 8 + cc;
                    const std::size_t ob = (static_cast<std::size_t>(gy) * image_width + gx) * 3;
                    image_out[ob + 0] = acc[ij * 3 + 0];
                    image_out[ob + 1] = acc[ij * 3 + 1];
                    image_out[ob + 2] = acc[ij * 3 + 2];
                }
            }
        }
    }
}

}  // namespace gsplat_tt

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Chunk frustum cull before pfwc (task #169, ported to the split writer in #433).
// Header-only so tests/unit/test_chunk_cull.cpp can check it without tt-metal.
//
// run.py Morton-orders the scene (GSPLAT_TT_CHUNK_CULL=1), so each 1024-gaussian
// pfwc tile is spatially compact. Per scene: the box of mean +- K rho over each
// tile, rho = 3 sqrt(trace cov3d) >= 3 sigma_max, and rho_max. Per view a tile is
// skipped when its box is behind z = k_near, or beyond one side plane x = U0 z
// (5 px pad) with rho_max / max(zmin, k_near) <= s_K: no gaussian's 3-sigma rect
// can then reach the image (docs/chunk-cull-t169/README.md in #169).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chunk_cull {

struct Table {
    const float* means = nullptr;
    const float* cov = nullptr;
    std::size_t N = 0;
    float K = 1.5f;
    std::vector<double> box;  // per tile: lo xyz, hi xyz, rho_max
    std::vector<uint8_t> ok;  // 0: non-finite member, never skipped
};

// means: N x 3, cov: N x 6 (xx xy xz yy yz zz), tile_elems gaussians per tile.
inline void build(Table& T, const float* means, const float* cov, std::size_t N, uint32_t num_tiles,
                  uint32_t tile_elems, float K) {
    T.means = means; T.cov = cov; T.N = N; T.K = K;
    T.box.assign(static_cast<std::size_t>(num_tiles) * 7, 0.0);
    T.ok.assign(num_tiles, 1);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}, rmax = 0.0;
        const std::size_t i0 = std::size_t{t} * tile_elems;
        const std::size_t i1 = std::min<std::size_t>(N, i0 + tile_elems);
        bool ok = true;
        for (std::size_t i = i0; i < i1; ++i) {
            const float* c = cov + 6 * i;
            const double rho = 3.0 * std::sqrt(std::max(0.0, double(c[0]) + c[3] + c[5]));
            if (!std::isfinite(rho)) ok = false;
            rmax = std::max(rmax, rho);
            for (int d = 0; d < 3; ++d) {
                const double m = means[3 * i + d];
                if (!std::isfinite(m)) ok = false;
                lo[d] = std::min(lo[d], m - K * rho);
                hi[d] = std::max(hi[d], m + K * rho);
            }
        }
        double* b = &T.box[std::size_t{t} * 7];
        for (int d = 0; d < 3; ++d) { b[d] = lo[d]; b[3 + d] = hi[d]; }
        b[6] = rmax;
        T.ok[t] = ok ? 1 : 0;
    }
}

// Tiles that may hold a visible gaussian, ascending. r: row-major world-to-camera
// rotation, tr: translation.
inline void survivors(const Table& T, const float r[9], const float tr[3], float fx, float fy, float cx,
                      float cy, float W, float H, float k_near, std::vector<uint32_t>& out) {
    constexpr double PAD = 5.0;  // px: 3 sqrt(0.3) dilation, ceil, rounding
    const double K = T.K;
    struct Side { int axis; double sgn, U0, sK; };
    Side sides[4] = {{0, 1.0, (W - cx + PAD) / fx, 0}, {0, -1.0, (cx + PAD) / fx, 0},
                     {1, 1.0, (H - cy + PAD) / fy, 0}, {1, -1.0, (cy + PAD) / fy, 0}};
    for (Side& sd : sides) {
        const double uk = std::sqrt(std::max(0.0, K * K * (1 + sd.U0 * sd.U0) - 1));
        sd.sK = (uk - sd.U0) / std::sqrt(1 + uk * uk);
    }
    const uint32_t n = static_cast<uint32_t>(T.ok.size());
    out.clear();
    for (uint32_t t = 0; t < n; ++t) {
        const double* b = &T.box[std::size_t{t} * 7];
        bool cull = false;
        if (T.ok[t]) {
            double pc[8][3];
            double zmin = 1e300, zmax = -1e300;
            for (int k = 0; k < 8; ++k) {
                const double p[3] = {b[(k & 1) ? 3 : 0], b[(k & 2) ? 4 : 1], b[(k & 4) ? 5 : 2]};
                for (int a = 0; a < 3; ++a)
                    pc[k][a] = r[3 * a] * p[0] + r[3 * a + 1] * p[1] + r[3 * a + 2] * p[2] + tr[a];
                zmin = std::min(zmin, pc[k][2]);
                zmax = std::max(zmax, pc[k][2]);
            }
            cull = zmax <= k_near;
            const double s = b[6] / std::max(zmin, double(k_near));
            for (const Side& sd : sides) {
                if (cull || s > sd.sK) continue;
                bool all = true;
                for (int k = 0; k < 8 && all; ++k) all = sd.sgn * pc[k][sd.axis] - sd.U0 * pc[k][2] > 0.0;
                cull = all;
            }
        }
        if (!cull) out.push_back(t);
    }
}

// The survivors dealt strided over the pfwc cores (core c: list[c], list[c + C],
// ...), so each core's count stays <= the full-scene SeqMap count and its segment
// base still holds. Chunk k of core c keeps its parity (split writer role k & 1).
inline uint32_t list_count(std::size_t S, uint32_t c, uint32_t num_cores) {
    return c < S ? static_cast<uint32_t>((S - 1 - c) / num_cores + 1) : 0u;
}

// One page of `words` uint32 per core, core c's ids at c * words.
inline void deal(const std::vector<uint32_t>& surv, uint32_t num_cores, uint32_t words,
                 std::vector<uint32_t>& host) {
    host.assign(std::size_t{words} * num_cores, 0u);
    for (std::size_t i = 0; i < surv.size(); ++i) host[(i % num_cores) * words + i / num_cores] = surv[i];
}

}  // namespace chunk_cull

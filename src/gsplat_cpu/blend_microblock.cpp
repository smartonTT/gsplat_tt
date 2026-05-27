#include "gsplat_cpu/blend_microblock.h"

#include "gsplat_cpu/simd_exp.h"
#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define GSPLAT_HAS_NEON 1
#else
#define GSPLAT_HAS_NEON 0
#endif

namespace gsplat_cpu {

namespace {

constexpr int kNumMicroblocks = 32;

#if GSPLAT_HAS_NEON
// SIMD inner loop for the canonical 4x8 microblock (mb_h=4, mb_w=8).
// Lays accum out in SoA (r[32], g[32], b[32]) and processes 4 pixels at a time
// (8 NEON groups: 4 rows × 2 columns of 4).
//
// Matches the scalar reference up to ~6e-6 relative error (well within the
// >=60 dB PSNR North Star at fixed contrib_floor).
struct MbAccum {
    float r[32];
    float g[32];
    float b[32];
    float t[32];
};

inline void init_mb_accum(MbAccum& a) {
    const float32x4_t ones = vdupq_n_f32(1.0f);
    const float32x4_t zeros = vdupq_n_f32(0.0f);
    for (int k = 0; k < 32; k += 4) {
        vst1q_f32(&a.r[k], zeros);
        vst1q_f32(&a.g[k], zeros);
        vst1q_f32(&a.b[k], zeros);
        vst1q_f32(&a.t[k], ones);
    }
}

inline void apply_gaussian_neon(
    MbAccum& a,
    float ci_a, float ci_b, float ci_c,
    float mx, float my,
    float opacity,
    float cr, float cg, float cb,
    int px_start, int py_start) {
    const float A = -0.5f * ci_a;
    const float B = -ci_b;
    const float C = -0.5f * ci_c;
    const float32x4_t A_v = vdupq_n_f32(A);
    const float32x4_t op_v = vdupq_n_f32(opacity);
    const float32x4_t alpha_cap = vdupq_n_f32(0.99f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t cr_v = vdupq_n_f32(cr);
    const float32x4_t cg_v = vdupq_n_f32(cg);
    const float32x4_t cb_v = vdupq_n_f32(cb);

    static const float dx_offs_lo[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    static const float dx_offs_hi[4] = {4.5f, 5.5f, 6.5f, 7.5f};
    const float32x4_t dx_off_lo = vld1q_f32(dx_offs_lo);
    const float32x4_t dx_off_hi = vld1q_f32(dx_offs_hi);

    const float px_base = static_cast<float>(px_start) - mx;
    const float32x4_t dx_lo = vaddq_f32(vdupq_n_f32(px_base), dx_off_lo);
    const float32x4_t dx_hi = vaddq_f32(vdupq_n_f32(px_base), dx_off_hi);
    const float32x4_t dx_lo_sq = vmulq_f32(dx_lo, dx_lo);
    const float32x4_t dx_hi_sq = vmulq_f32(dx_hi, dx_hi);

    for (int i = 0; i < 4; ++i) {
        const float py = static_cast<float>(py_start + i) + 0.5f;
        const float dy = py - my;
        const float y_term = C * dy * dy;
        const float xy_coef = B * dy;
        const float32x4_t y_term_v = vdupq_n_f32(y_term);
        const float32x4_t xy_coef_v = vdupq_n_f32(xy_coef);

        // power = y_term + xy_coef*dx + A*dx*dx
        float32x4_t pwr_lo = vfmaq_f32(y_term_v, xy_coef_v, dx_lo);
        pwr_lo = vfmaq_f32(pwr_lo, A_v, dx_lo_sq);
        float32x4_t pwr_hi = vfmaq_f32(y_term_v, xy_coef_v, dx_hi);
        pwr_hi = vfmaq_f32(pwr_hi, A_v, dx_hi_sq);

        const float32x4_t pwr_lo_bound = vdupq_n_f32(-30.0f);
        pwr_lo = vmaxq_f32(vminq_f32(pwr_lo, zero), pwr_lo_bound);
        pwr_hi = vmaxq_f32(vminq_f32(pwr_hi, zero), pwr_lo_bound);

        const float32x4_t gw_lo = simd_exp_f32x4(pwr_lo);
        const float32x4_t gw_hi = simd_exp_f32x4(pwr_hi);

        const float32x4_t alpha_lo = vminq_f32(vmulq_f32(op_v, gw_lo), alpha_cap);
        const float32x4_t alpha_hi = vminq_f32(vmulq_f32(op_v, gw_hi), alpha_cap);

        const int row = i * 8;
        const float32x4_t t_lo = vld1q_f32(&a.t[row]);
        const float32x4_t t_hi = vld1q_f32(&a.t[row + 4]);
        const float32x4_t at_lo = vmulq_f32(alpha_lo, t_lo);
        const float32x4_t at_hi = vmulq_f32(alpha_hi, t_hi);

        float32x4_t r_lo = vld1q_f32(&a.r[row]);
        float32x4_t g_lo = vld1q_f32(&a.g[row]);
        float32x4_t b_lo = vld1q_f32(&a.b[row]);
        r_lo = vfmaq_f32(r_lo, at_lo, cr_v);
        g_lo = vfmaq_f32(g_lo, at_lo, cg_v);
        b_lo = vfmaq_f32(b_lo, at_lo, cb_v);
        vst1q_f32(&a.r[row], r_lo);
        vst1q_f32(&a.g[row], g_lo);
        vst1q_f32(&a.b[row], b_lo);

        float32x4_t r_hi = vld1q_f32(&a.r[row + 4]);
        float32x4_t g_hi = vld1q_f32(&a.g[row + 4]);
        float32x4_t b_hi = vld1q_f32(&a.b[row + 4]);
        r_hi = vfmaq_f32(r_hi, at_hi, cr_v);
        g_hi = vfmaq_f32(g_hi, at_hi, cg_v);
        b_hi = vfmaq_f32(b_hi, at_hi, cb_v);
        vst1q_f32(&a.r[row + 4], r_hi);
        vst1q_f32(&a.g[row + 4], g_hi);
        vst1q_f32(&a.b[row + 4], b_hi);

        // T_new = T - at
        vst1q_f32(&a.t[row],     vsubq_f32(t_lo, at_lo));
        vst1q_f32(&a.t[row + 4], vsubq_f32(t_hi, at_hi));
    }
}

inline float max_t_neon(const MbAccum& a) {
    float32x4_t m = vld1q_f32(&a.t[0]);
    for (int k = 4; k < 32; k += 4) {
        m = vmaxq_f32(m, vld1q_f32(&a.t[k]));
    }
    return vmaxvq_f32(m);
}
#endif  // GSPLAT_HAS_NEON

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

#if GSPLAT_HAS_NEON
        // SIMD fast path for full 4x8 microblocks (the common case interior to the image).
        if (mb_h == 4 && mb_w == 8) {
            MbAccum a;
            init_mb_accum(a);

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

                apply_gaussian_neon(a, ci_a, ci_b, ci_c, mx, my,
                                    opacity, cr, cg, cb,
                                    px_start, py_start);

                if (max_t_neon(a) < 0.0001f) {
                    break;
                }
            }

            for (int i = 0; i < 4; ++i) {
                const int gy = py_start + i;
                for (int j = 0; j < 8; ++j) {
                    const int gx = px_start + j;
                    const int out_base = (gy * image_width + gx) * 3;
                    const int ij = i * 8 + j;
                    image_out[out_base + 0] = a.r[ij];
                    image_out[out_base + 1] = a.g[ij];
                    image_out[out_base + 2] = a.b[ij];
                }
            }
            continue;
        }
#endif  // GSPLAT_HAS_NEON

        // Scalar fallback (partial microblocks at image edges, or no NEON).
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

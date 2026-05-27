#include "gsplat_cpu/cull_and_blend.h"

#include "gsplat_cpu/simd_exp.h"
#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
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

// Per-Gaussian packed cull record. Sized to fit a 64-byte cache line
// so the cull inner loop touches one line per Gaussian id instead of
// 5-6 (means_2d / log_thresh / x_half / y_half / cov_inv arrays were
// previously each a separate strided read).
//   mx,my            screen-space mean
//   ci_a/b/c         cov inverse (per-microblock power coefficients)
//   log_thresh       log(mb_contrib_floor / opacity); >= 0 = drop sentinel
//   x_half/y_half    BB half-extents
//   opacity          alpha multiplier (used by blend, kept here for locality)
//   _pad             pad to 48 bytes; combined with the AoS colors record
//                    (12 bytes) every kept-Gaussian fits in one cache line
//                    pair at most.
struct alignas(8) GaussianCullRec {
    float mx, my;
    float ci_a, ci_b, ci_c;
    float log_thresh;
    float x_half, y_half;
    float opacity;
    float _pad;  // total 40 bytes, pad to 40-aligned (no need to grow to 64)
};
static_assert(sizeof(GaussianCullRec) == 40, "GaussianCullRec must be 40 bytes");

#if GSPLAT_HAS_NEON
struct MbAccum {
    alignas(16) float r[32];
    alignas(16) float g[32];
    alignas(16) float b[32];
    alignas(16) float t[32];
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

// iter-045: per-microblock blend kernel that holds the 8 transmittance
// vectors (acc.t[0..31] as 8 float32x4_t) live in NEON registers across
// every Gaussian in the microblock instead of round-tripping through
// memory on every apply. Each Gaussian eliminates 8 vld1q (t) and 8 vst1q
// (t') — ~16 NEON ops × kn Gaussians per microblock = O(kn × 16) ops
// saved. RGB stays in memory because together r/g/b/t (32 regs) would
// exhaust the entire NEON register file and force spills.
//
// max_t check still happens every 4 Gaussians (iter-035). It now reads
// directly from the live t registers via vmaxq reductions — no memory
// fetch required.
struct TVec8 {
    float32x4_t t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi;
};

inline void load_t_regs(const MbAccum& acc, TVec8& tv) {
    tv.t0_lo = vld1q_f32(&acc.t[0]);
    tv.t0_hi = vld1q_f32(&acc.t[4]);
    tv.t1_lo = vld1q_f32(&acc.t[8]);
    tv.t1_hi = vld1q_f32(&acc.t[12]);
    tv.t2_lo = vld1q_f32(&acc.t[16]);
    tv.t2_hi = vld1q_f32(&acc.t[20]);
    tv.t3_lo = vld1q_f32(&acc.t[24]);
    tv.t3_hi = vld1q_f32(&acc.t[28]);
}

inline void store_t_regs(MbAccum& acc, const TVec8& tv) {
    vst1q_f32(&acc.t[0],  tv.t0_lo);
    vst1q_f32(&acc.t[4],  tv.t0_hi);
    vst1q_f32(&acc.t[8],  tv.t1_lo);
    vst1q_f32(&acc.t[12], tv.t1_hi);
    vst1q_f32(&acc.t[16], tv.t2_lo);
    vst1q_f32(&acc.t[20], tv.t2_hi);
    vst1q_f32(&acc.t[24], tv.t3_lo);
    vst1q_f32(&acc.t[28], tv.t3_hi);
}

inline float max_t_regs(const TVec8& tv) {
    const float32x4_t m01_lo = vmaxq_f32(tv.t0_lo, tv.t1_lo);
    const float32x4_t m01_hi = vmaxq_f32(tv.t0_hi, tv.t1_hi);
    const float32x4_t m23_lo = vmaxq_f32(tv.t2_lo, tv.t3_lo);
    const float32x4_t m23_hi = vmaxq_f32(tv.t2_hi, tv.t3_hi);
    const float32x4_t m_lo = vmaxq_f32(m01_lo, m23_lo);
    const float32x4_t m_hi = vmaxq_f32(m01_hi, m23_hi);
    return vmaxvq_f32(vmaxq_f32(m_lo, m_hi));
}

inline void apply_gaussian_t_regs(
    MbAccum& acc,
    TVec8& tv,
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
    const float32x4_t pwr_lo_bound = vdupq_n_f32(-30.0f);
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

    auto row_body = [&](int i, float32x4_t& t_lo, float32x4_t& t_hi) __attribute__((always_inline)) {
        const float py = static_cast<float>(py_start + i) + 0.5f;
        const float dy = py - my;
        const float y_term = C * dy * dy;
        const float xy_coef = B * dy;
        const float32x4_t y_term_v = vdupq_n_f32(y_term);
        const float32x4_t xy_coef_v = vdupq_n_f32(xy_coef);

        float32x4_t pwr_lo = vfmaq_f32(y_term_v, xy_coef_v, dx_lo);
        pwr_lo = vfmaq_f32(pwr_lo, A_v, dx_lo_sq);
        float32x4_t pwr_hi = vfmaq_f32(y_term_v, xy_coef_v, dx_hi);
        pwr_hi = vfmaq_f32(pwr_hi, A_v, dx_hi_sq);

        pwr_lo = vmaxq_f32(vminq_f32(pwr_lo, zero), pwr_lo_bound);
        pwr_hi = vmaxq_f32(vminq_f32(pwr_hi, zero), pwr_lo_bound);

        const float32x4_t gw_lo = simd_exp_f32x4_fast(pwr_lo);
        const float32x4_t gw_hi = simd_exp_f32x4_fast(pwr_hi);

        const float32x4_t alpha_lo = vminq_f32(vmulq_f32(op_v, gw_lo), alpha_cap);
        const float32x4_t alpha_hi = vminq_f32(vmulq_f32(op_v, gw_hi), alpha_cap);

        const int row = i * 8;
        const float32x4_t at_lo = vmulq_f32(alpha_lo, t_lo);
        const float32x4_t at_hi = vmulq_f32(alpha_hi, t_hi);

        float32x4_t r_lo = vld1q_f32(&acc.r[row]);
        float32x4_t gg_lo = vld1q_f32(&acc.g[row]);
        float32x4_t bb_lo = vld1q_f32(&acc.b[row]);
        r_lo = vfmaq_f32(r_lo, at_lo, cr_v);
        gg_lo = vfmaq_f32(gg_lo, at_lo, cg_v);
        bb_lo = vfmaq_f32(bb_lo, at_lo, cb_v);
        vst1q_f32(&acc.r[row], r_lo);
        vst1q_f32(&acc.g[row], gg_lo);
        vst1q_f32(&acc.b[row], bb_lo);

        float32x4_t r_hi = vld1q_f32(&acc.r[row + 4]);
        float32x4_t gg_hi = vld1q_f32(&acc.g[row + 4]);
        float32x4_t bb_hi = vld1q_f32(&acc.b[row + 4]);
        r_hi = vfmaq_f32(r_hi, at_hi, cr_v);
        gg_hi = vfmaq_f32(gg_hi, at_hi, cg_v);
        bb_hi = vfmaq_f32(bb_hi, at_hi, cb_v);
        vst1q_f32(&acc.r[row + 4], r_hi);
        vst1q_f32(&acc.g[row + 4], gg_hi);
        vst1q_f32(&acc.b[row + 4], bb_hi);

        t_lo = vsubq_f32(t_lo, at_lo);
        t_hi = vsubq_f32(t_hi, at_hi);
    };
    row_body(0, tv.t0_lo, tv.t0_hi);
    row_body(1, tv.t1_lo, tv.t1_hi);
    row_body(2, tv.t2_lo, tv.t2_hi);
    row_body(3, tv.t3_lo, tv.t3_hi);
}
#endif  // GSPLAT_HAS_NEON

// Per-worker scratch buffer for the cull pass. One contiguous uint32_t array
// of size 32 * L is partitioned into 32 microblock slots; per-tile reset
// just resets the per-microblock offsets, the heap buffer survives across
// tiles. Eliminates ~256 * 32 std::vector heap allocations per frame.
//
// uint32_t (vs int64_t in iter-023) halves the per-worker working set: at
// the stitch hero scene's max-Gaussian tile (~11k entries), 32 microblock
// slots = 32 * 11k = 352k entries -> 1.4 MB (uint32) vs 2.8 MB (int64).
// 1.4 MB still spills L1 (M-Pro: 192 KB) but fits L2 comfortably; the
// sequential per-microblock read pattern in the blend loop is now twice
// as cache-bandwidth-efficient.
struct TileScratch {
    std::vector<uint32_t> kept_flat;          // flat buffer, capacity grown as needed
    std::array<int32_t, kNumMicroblocks + 1> offsets{};  // offsets[m] = start of mb m in kept_flat
    int32_t stride{0};                       // capacity per microblock
};

// Per-tile fused cull + blend kernel.
// In-tile per-microblock kept-id buffer never escapes the thread; mb_stream
// global allocation is avoided entirely. The blend loop is bit-identical to
// blend_microblock's SIMD path because the per-microblock Gaussian-id order
// equals the depth-sorted order of mb_stream.
void cull_and_blend_tile(
    const int tile_id,
    const int tiles_x,
    const int tile_size,
    const int image_height,
    const int image_width,
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const float mb_contrib_floor,
    const GaussianCullRec* gauss_rec,
    float* image_out,
    int64_t* tile_dropped_count,
    int64_t* tile_kept_count,
    TileScratch& scratch) {
    const int64_t start = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 0];
    const int64_t end = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 1];
    if (start == end) {
        return;
    }

    const int ty = tile_id / tiles_x;
    const int tx = tile_id % tiles_x;
    const float tx_tile = static_cast<float>(tx * tile_size);
    const float ty_tile = static_cast<float>(ty * tile_size);
    const int py_tile = ty * tile_size;
    const int px_tile = tx * tile_size;
    const int py_end_tile = std::min(py_tile + tile_size, image_height);
    const int px_end_tile = std::min(px_tile + tile_size, image_width);

    const int64_t L = end - start;

    // Use the per-worker TileScratch buffer: one contiguous int64_t array of
    // size 32*L, partitioned into 32 microblock slots of stride L. Counters
    // track how many ids have been pushed into each slot. Avoids ~32 heap
    // allocations per tile (= 8192 per frame on a 256-tile image).
    const int32_t stride = static_cast<int32_t>(L);
    const std::size_t needed =
        static_cast<std::size_t>(kNumMicroblocks) * static_cast<std::size_t>(L);
    if (scratch.kept_flat.size() < needed) scratch.kept_flat.resize(needed);
    scratch.stride = stride;
    for (int m = 0; m < kNumMicroblocks; ++m) {
        scratch.offsets[static_cast<std::size_t>(m)] = m * stride;
    }
    // offsets[m] starts at the slot base, incremented as ids are pushed.
    // The slot for microblock m holds ids in [m*stride, offsets[m]).

    int64_t dropped = 0;
    int64_t kept_total = 0;

    for (int64_t l = 0; l < L; ++l) {
        const int64_t g = sorted_gaussian_ids[static_cast<std::size_t>(start + l)];
        const GaussianCullRec& rec = gauss_rec[static_cast<std::size_t>(g)];

        const float log_thresh = rec.log_thresh;
        // log_thresh >= 0 sentinel = "Gaussian below mb_contrib_floor".
        if (log_thresh >= 0.0f) {
            ++dropped;
            continue;
        }

        const float mean_x = rec.mx;
        const float mean_y = rec.my;
        const float ci_a = rec.ci_a;
        const float ci_b = rec.ci_b;
        const float ci_c = rec.ci_c;
        const float x_half = rec.x_half;
        const float y_half = rec.y_half;

        const float bb_x_min = mean_x - x_half;
        const float bb_x_max = mean_x + x_half;
        const float bb_y_min = mean_y - y_half;
        const float bb_y_max = mean_y + y_half;

        const int mx_lo = std::max(0, static_cast<int>(std::floor((bb_x_min - tx_tile) / 8.0f)));
        const int mx_hi = std::min(3, static_cast<int>(std::floor((bb_x_max - tx_tile) / 8.0f)));
        const int my_lo = std::max(0, static_cast<int>(std::floor((bb_y_min - ty_tile) / 4.0f)));
        const int my_hi = std::min(7, static_cast<int>(std::floor((bb_y_max - ty_tile) / 4.0f)));

        if (mx_lo > mx_hi || my_lo > my_hi) {
            ++dropped;
            continue;
        }

        bool kept_any = false;
        for (int my = my_lo; my <= my_hi; ++my) {
            const float mb_oy = ty_tile + static_cast<float>(my * 4);
            const float cy = std::clamp(mean_y, mb_oy, mb_oy + 4.0f);
            const float dy_c = cy - mean_y;
            for (int mx = mx_lo; mx <= mx_hi; ++mx) {
                const float mb_ox = tx_tile + static_cast<float>(mx * 8);
                const float cx = std::clamp(mean_x, mb_ox, mb_ox + 8.0f);
                const float dx_c = cx - mean_x;
                const float power_c =
                    -0.5f * (ci_a * dx_c * dx_c + 2.0f * ci_b * dx_c * dy_c + ci_c * dy_c * dy_c);
                if (power_c >= log_thresh) {
                    const int mb = (my << 2) | mx;
                    scratch.kept_flat[static_cast<std::size_t>(
                        scratch.offsets[static_cast<std::size_t>(mb)]++)] =
                        static_cast<uint32_t>(g);
                    kept_any = true;
                    ++kept_total;
                }
            }
        }

        if (!kept_any) {
            ++dropped;
        }
    }

    // Blend each microblock from the in-tile kept list.
    for (int m = 0; m < kNumMicroblocks; ++m) {
        const int32_t slot_base = m * stride;
        const int32_t slot_end = scratch.offsets[static_cast<std::size_t>(m)];
        const int32_t kn = slot_end - slot_base;
        if (kn == 0) continue;
        const uint32_t* kg_data = scratch.kept_flat.data() + slot_base;

        const int mb_ox_i = (m & 3) * 8;
        const int mb_oy_i = (m >> 2) * 4;
        const int py_start = py_tile + mb_oy_i;
        const int px_start = px_tile + mb_ox_i;
        const int py_end = std::min(py_start + 4, py_end_tile);
        const int px_end = std::min(px_start + 8, px_end_tile);
        if (py_start >= py_end || px_start >= px_end) continue;
        const int mb_h = py_end - py_start;
        const int mb_w = px_end - px_start;
        const int npix = mb_h * mb_w;

#if GSPLAT_HAS_NEON
        if (mb_h == 4 && mb_w == 8) {
            MbAccum acc;
            init_mb_accum(acc);
            // iter-045: load t into 8 NEON registers, keep them live across
            // every Gaussian's apply, write back once at the end. Saves
            // 8 vld1q + 8 vst1q (= 16 ops × ~1 issue slot each) per Gaussian
            // — the highest-frequency redundant memory ops in the inner loop.
            TVec8 tv;
            load_t_regs(acc, tv);
            // iter-035: max_t every 4 Gaussians (now read from tv regs).
            int32_t k = 0;
            for (; k + 4 <= kn; k += 4) {
                for (int j = 0; j < 4; ++j) {
                    const std::size_t gs = static_cast<std::size_t>(kg_data[k + j]);
                    const GaussianCullRec& rec = gauss_rec[gs];
                    apply_gaussian_t_regs(acc, tv,
                        rec.ci_a, rec.ci_b, rec.ci_c,
                        rec.mx, rec.my,
                        rec.opacity,
                        colors[gs * 3 + 0], colors[gs * 3 + 1], colors[gs * 3 + 2],
                        px_start, py_start);
                }
                if (max_t_regs(tv) < 0.0001f) { k = kn; break; }
            }
            for (; k < kn; ++k) {
                const std::size_t gs = static_cast<std::size_t>(kg_data[k]);
                const GaussianCullRec& rec = gauss_rec[gs];
                apply_gaussian_t_regs(acc, tv,
                    rec.ci_a, rec.ci_b, rec.ci_c,
                    rec.mx, rec.my,
                    rec.opacity,
                    colors[gs * 3 + 0], colors[gs * 3 + 1], colors[gs * 3 + 2],
                    px_start, py_start);
            }
            store_t_regs(acc, tv);
            for (int i = 0; i < 4; ++i) {
                const int gy = py_start + i;
                for (int j = 0; j < 8; ++j) {
                    const int gx = px_start + j;
                    const int out_base = (gy * image_width + gx) * 3;
                    const int ij = i * 8 + j;
                    image_out[out_base + 0] = acc.r[ij];
                    image_out[out_base + 1] = acc.g[ij];
                    image_out[out_base + 2] = acc.b[ij];
                }
            }
            continue;
        }
#endif

        // Scalar fallback for boundary microblocks.
        float T[4 * 8];
        float accum[4 * 8 * 3];
        for (int k = 0; k < npix; ++k) T[k] = 1.0f;
        for (int k = 0; k < npix * 3; ++k) accum[k] = 0.0f;
        for (int32_t kk = 0; kk < kn; ++kk) {
            const std::size_t gs = static_cast<std::size_t>(kg_data[kk]);
            const GaussianCullRec& rec = gauss_rec[gs];
            const float ci_a = rec.ci_a;
            const float ci_b = rec.ci_b;
            const float ci_c = rec.ci_c;
            const float mx = rec.mx;
            const float my = rec.my;
            const float opacity = rec.opacity;
            const float cr = colors[gs * 3 + 0];
            const float cg = colors[gs * 3 + 1];
            const float cb = colors[gs * 3 + 2];
            for (int i = 0; i < mb_h; ++i) {
                const float py = static_cast<float>(py_start + i) + 0.5f;
                for (int j = 0; j < mb_w; ++j) {
                    const float px = static_cast<float>(px_start + j) + 0.5f;
                    const float dx = px - mx;
                    const float dy = py - my;
                    const float power = -0.5f * (ci_a * dx * dx + 2.0f * ci_b * dx * dy + ci_c * dy * dy);
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
            for (int k = 0; k < npix; ++k) tmax = std::max(tmax, T[k]);
            if (tmax < 0.0001f) break;
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

    *tile_dropped_count = dropped;
    *tile_kept_count = kept_total;
}

}  // namespace

CullAndBlendResult cull_and_blend(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const std::size_t M,
    const std::size_t P,
    const int tiles_x,
    const int tiles_y,
    const int tile_size,
    const int image_height,
    const int image_width,
    const float mb_contrib_floor,
    ThreadPool& pool) {
    const int num_tiles = tiles_x * tiles_y;

    CullAndBlendResult result;
    result.pairs_in = static_cast<int64_t>(P);
    result.image.assign(static_cast<std::size_t>(image_height) *
                            static_cast<std::size_t>(image_width) * 3,
                        0.0f);

    // Precompute per-Gaussian cull data into one AoS array, parallel.
    // Pulls log/sqrt out of the per-tile cull inner loop AND packs every
    // field the inner loop reads (mx,my, ci_a/b/c, log_thresh, x_half/y_half,
    // opacity) into one 40-byte record. Cull loop now does 1 cache-line
    // fetch per Gaussian instead of 5-6 strided fetches across separate
    // arrays — meaningful at the 294k (Gaussian, tile) pairs scanned per
    // hero frame.
    std::vector<GaussianCullRec> gauss_rec(M);
    if (M > 0) {
        const std::size_t W = pool.size();
        auto compute_one = [&](std::size_t i) {
            const float a = covs_2d[i * 4 + 0];
            const float b = covs_2d[i * 4 + 1];
            const float c = covs_2d[i * 4 + 3];
            const float det = std::max(a * c - b * b, 1e-6f);

            GaussianCullRec& r = gauss_rec[i];
            r.mx = means_2d[i * 2 + 0];
            r.my = means_2d[i * 2 + 1];
            r.ci_a = c / det;
            r.ci_b = -b / det;
            r.ci_c = a / det;
            r.opacity = opacities[i];

            if (r.opacity <= mb_contrib_floor) {
                r.log_thresh = 0.0f;  // sentinel: drop in per-tile cull
                r.x_half = 0.0f;
                r.y_half = 0.0f;
            } else {
                const float lt = std::log(mb_contrib_floor / r.opacity);
                r.log_thresh = lt;  // <= 0 for visible
                const float rd = std::sqrt(-2.0f * lt);
                r.x_half = rd * std::sqrt(std::max(a, 0.0f));
                r.y_half = rd * std::sqrt(std::max(c, 0.0f));
            }
        };
        if (W > 1 && M >= 4096) {
            for (std::size_t w = 0; w < W; ++w) {
                pool.submit([&, w, W]() {
                    for (std::size_t i = w; i < M; i += W) compute_one(i);
                });
            }
            pool.wait();
        } else {
            for (std::size_t i = 0; i < M; ++i) compute_one(i);
        }
    }

    std::vector<int64_t> tile_dropped(static_cast<std::size_t>(num_tiles), 0);
    std::vector<int64_t> tile_kept(static_cast<std::size_t>(num_tiles), 0);

    // LPT scheduling + reduced submits: sort tile indices by descending
    // Gaussian count, then dispatch only W tasks (one per worker). Each
    // worker greedily pulls the next-heaviest tile from a shared atomic
    // counter. Two wins over 256 individual submits:
    //   - 256 mutex acquisitions on the pool's task queue collapse to W (~10).
    //   - True LPT: workers race to grab heavy tiles first; light tail
    //     dispatched last.
    std::vector<int> tile_order(static_cast<std::size_t>(num_tiles));
    for (int i = 0; i < num_tiles; ++i) tile_order[static_cast<std::size_t>(i)] = i;
    std::sort(tile_order.begin(), tile_order.end(), [&](int a, int b) {
        const int64_t la = tile_ranges[static_cast<std::size_t>(a) * 2 + 1] -
                           tile_ranges[static_cast<std::size_t>(a) * 2 + 0];
        const int64_t lb = tile_ranges[static_cast<std::size_t>(b) * 2 + 1] -
                           tile_ranges[static_cast<std::size_t>(b) * 2 + 0];
        return la > lb;
    });

    std::atomic<int> next_idx{0};
    const std::size_t W = std::max<std::size_t>(1, pool.size());
    // Per-worker scratch lives for the duration of the parallel section.
    // Resized lazily by each tile based on that tile's Gaussian count L.
    std::vector<TileScratch> scratches(W);
    for (std::size_t w = 0; w < W; ++w) {
        pool.submit([&, w]() {
            TileScratch& sc = scratches[w];
            for (;;) {
                const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_tiles) return;
                const int tile_id = tile_order[static_cast<std::size_t>(idx)];
                cull_and_blend_tile(
                    tile_id, tiles_x, tile_size, image_height, image_width,
                    means_2d, covs_2d, colors, opacities,
                    sorted_gaussian_ids, tile_ranges,
                    mb_contrib_floor,
                    gauss_rec.data(),
                    result.image.data(),
                    &tile_dropped[static_cast<std::size_t>(tile_id)],
                    &tile_kept[static_cast<std::size_t>(tile_id)],
                    sc);
            }
        });
    }
    pool.wait();

    int64_t total_dropped = 0, total_kept = 0;
    for (int i = 0; i < num_tiles; ++i) {
        total_dropped += tile_dropped[static_cast<std::size_t>(i)];
        total_kept += tile_kept[static_cast<std::size_t>(i)];
    }
    result.pairs_dropped_all_mb = total_dropped;
    result.pairs_kept_per_mb = total_kept;
    return result;
}

}  // namespace gsplat_cpu

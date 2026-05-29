#include "gsplat_tt/render_blend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gsplat_cpu/cull_and_blend.h"
#include "gsplat_cpu/thread_pool.h"
#include "gsplat_tt/blend.h"
#include "gsplat_tt/mb_payload.h"

namespace gsplat_tt {

namespace {

// Build the legacy 9-float-per-entry kernel payload (packs/offsets/px/py) the
// existing on-device blend kernel (blend_device.cpp process_frame) consumes,
// from the projected gaussians + depth-sorted per-tile lists. This is the
// plan-sanctioned full-32x32-tile device blend baseline (tt-001a): correct,
// proven on-device, ~47 dB. The mb-major 4x8 fp32 kernel supersedes it for
// the >=60 dB gate.
struct KernelInputs {
    std::vector<float> packs;    // P * 9
    std::vector<float> offsets;  // num_tiles + 1
    std::vector<float> px;       // num_tiles * 32 * 32
    std::vector<float> py;       // num_tiles * 32 * 32
};

KernelInputs build_kernel_inputs(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size) {
    const int num_tiles = tiles_x * tiles_y;
    KernelInputs k;
    k.packs.assign(P * 9, 0.0f);

    // Means are stored TILE-LOCAL (mean - tile_origin) and px/py are also
    // tile-local (0..31). The kernel stores px/py + mean as bf16; global
    // coords (up to ~5000 for bicycle) round to multiples of 16-32 in bf16,
    // destroying per-pixel dx. Tile-local coords stay in [0,32) where bf16 is
    // sub-0.25 accurate. dx = px_local - mean_local == global dx.
    for (int t = 0; t < num_tiles; ++t) {
        const int64_t start = tile_ranges[static_cast<std::size_t>(t) * 2 + 0];
        const int64_t end = tile_ranges[static_cast<std::size_t>(t) * 2 + 1];
        const float ox = static_cast<float>((t % tiles_x) * tile_size);
        const float oy = static_cast<float>((t / tiles_x) * tile_size);
        for (int64_t e = start; e < end; ++e) {
            const int64_t g = sorted_gaussian_ids[e];
            const float a = covs_2d[static_cast<std::size_t>(g) * 4 + 0];
            const float b = covs_2d[static_cast<std::size_t>(g) * 4 + 1];
            const float c = covs_2d[static_cast<std::size_t>(g) * 4 + 3];
            const float det = std::max(a * c - b * b, 1e-6f);
            const float ci_a = c / det;
            const float ci_b = -b / det;
            const float ci_c = a / det;
            float* row = &k.packs[static_cast<std::size_t>(e) * 9];
            row[0] = means_2d[static_cast<std::size_t>(g) * 2 + 0] - ox;
            row[1] = means_2d[static_cast<std::size_t>(g) * 2 + 1] - oy;
            row[2] = ci_a;
            row[3] = 2.0f * ci_b;
            row[4] = ci_c;
            row[5] = colors[static_cast<std::size_t>(g) * 3 + 0];
            row[6] = colors[static_cast<std::size_t>(g) * 3 + 1];
            row[7] = colors[static_cast<std::size_t>(g) * 3 + 2];
            row[8] = opacities[static_cast<std::size_t>(g)];
        }
    }

    k.offsets.assign(static_cast<std::size_t>(num_tiles) + 1, 0.0f);
    for (int t = 0; t < num_tiles; ++t) {
        k.offsets[static_cast<std::size_t>(t)] =
            static_cast<float>(tile_ranges[static_cast<std::size_t>(t) * 2 + 0]);
    }
    k.offsets[static_cast<std::size_t>(num_tiles)] = static_cast<float>(P);

    // Tile-local px/py grid is identical for every tile.
    const std::size_t ppt = static_cast<std::size_t>(tile_size) * tile_size;
    k.px.assign(static_cast<std::size_t>(num_tiles) * ppt, 0.0f);
    k.py.assign(static_cast<std::size_t>(num_tiles) * ppt, 0.0f);
    for (int t = 0; t < num_tiles; ++t) {
        float* pxt = &k.px[static_cast<std::size_t>(t) * ppt];
        float* pyt = &k.py[static_cast<std::size_t>(t) * ppt];
        for (int r = 0; r < tile_size; ++r) {
            for (int c = 0; c < tile_size; ++c) {
                const std::size_t i = static_cast<std::size_t>(r) * tile_size + c;
                pxt[i] = static_cast<float>(c) + 0.5f;
                pyt[i] = static_cast<float>(r) + 0.5f;
            }
        }
    }
    return k;
}

}  // namespace

gsplat_cpu::CullAndBlendResult render_blend_tt(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t M,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int image_height,
    int image_width,
    float mb_contrib_floor,
    gsplat_cpu::ThreadPool& pool,
    float* image_out_external,
    bool cull_disabled,
    float transmittance_threshold,
    int blend_mode) {
    gsplat_cpu::CullAndBlendResult result;
    result.pairs_in = static_cast<int64_t>(P);

    // --- Synthetic single-splat debug injection (GSPLAT_TT_SYNTH=1) ---
    // Replaces the projected gaussians with ONE small gaussian in tile 0 so the
    // expected output is trivial to reason about (a small blob in the top-left,
    // everything else black). The identical tiny payload feeds both the device
    // mb kernel (GSPLAT_TT_MB_KERNEL=1) and the CPU oracle (GSPLAT_TT_DEVICE_BLEND=0),
    // isolating device-kernel bugs from the projection/sort/cull pipeline.
    //   GSPLAT_TT_SYNTH_X / _Y   : center in image pixels (default 4, 2)
    //   GSPLAT_TT_SYNTH_VAR      : isotropic 2D variance in px^2 (default 2.0)
    std::vector<float> syn_means, syn_covs, syn_colors, syn_op;
    std::vector<int64_t> syn_ids, syn_ranges;
    if (const char* s = std::getenv("GSPLAT_TT_SYNTH"); s != nullptr && s[0] == '1') {
        const int num_tiles = tiles_x * tiles_y;
        float cx = 4.0f, cy = 2.0f, var = 2.0f;
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_X")) cx = static_cast<float>(std::atof(e));
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_Y")) cy = static_cast<float>(std::atof(e));
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_VAR")) var = static_cast<float>(std::atof(e));
        syn_means  = {cx, cy};
        syn_covs   = {var, 0.0f, 0.0f, var};  // a, b, (unused), c
        syn_colors = {1.0f, 0.0f, 0.0f};      // red
        syn_op     = {0.9f};
        syn_ids    = {0};
        syn_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);
        syn_ranges[1] = 1;  // tile 0 owns gaussian 0: [0, 1); all other tiles empty
        means_2d = syn_means.data();
        covs_2d = syn_covs.data();
        colors = syn_colors.data();
        opacities = syn_op.data();
        sorted_gaussian_ids = syn_ids.data();
        tile_ranges = syn_ranges.data();
        M = 1;
        P = 1;
        result.pairs_in = 1;
    }

    float* image_out = image_out_external;
    if (image_out == nullptr) {
        result.image.assign(static_cast<std::size_t>(image_height) *
                                static_cast<std::size_t>(image_width) * 3,
                            0.0f);
        image_out = result.image.data();
    } else {
        std::memset(image_out, 0,
                    static_cast<std::size_t>(image_height) *
                        static_cast<std::size_t>(image_width) * 3 * sizeof(float));
    }

    if (blend_mode == 1 /* CPU reference blend from basis payload */) {
        MbPayload p = build_mb_payload(
            means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
            tile_ranges, M, P, tiles_x, tiles_y, tile_size, image_height,
            image_width, mb_contrib_floor, pool, cull_disabled);
        blend_from_mb_payload_cpu(p, image_height, image_width,
                                  transmittance_threshold, image_out);
        result.pairs_dropped_all_mb = p.pairs_dropped_all_mb;
        result.pairs_kept_per_mb = p.pairs_kept_per_mb;
        return result;
    }

    if (blend_mode == 2) {
        const bool mb_timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;

        // Microblock-major (4x8) device kernel path (amendment-003 step 3).
        // Opt in with GSPLAT_TT_MB_KERNEL=1. Builds the GAUSSIAN-MAJOR stream:
        // ONE 16-word row per (tile, gaussian) that survives the microblock cull,
        // carrying the 10 basis coeffs plus a 32-bit microblock mask (bit m set =>
        // gaussian touches microblock m). The device reads each gaussian's coeffs
        // ONCE and dispatches the SFPU blend to just the masked microblocks. The
        // rows are now produced DIRECTLY during the cull (build_gaussian_major_payload),
        // fusing what used to be a separate build_mb_payload + two-pass mask/emit
        // stage into a single parallel pass.
        const char* mb_kernel = std::getenv("GSPLAT_TT_MB_KERNEL");
        if (mb_kernel != nullptr && mb_kernel[0] == '1') {
            constexpr uint32_t GM_LANES = 16;  // 10 coeff + mask + pad, 64B row
            const int num_tiles = tiles_x * tiles_y;

            const auto _t0 = std::chrono::steady_clock::now();
            GaussianMajorPayload gp = build_gaussian_major_payload(
                means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
                tile_ranges, P, tiles_x, tiles_y, tile_size, mb_contrib_floor,
                pool, cull_disabled);
            const auto _t1 = std::chrono::steady_clock::now();
            result.pairs_in = gp.pairs_in;
            result.pairs_dropped_all_mb = gp.pairs_dropped_all_mb;
            result.pairs_kept_per_mb = gp.pairs_kept_per_mb;
            const uint32_t row = gp.mb_coeff_off[static_cast<std::size_t>(num_tiles)];

            if (const char* d = std::getenv("GSPLAT_TT_MB_DUMP"); d && d[0] == '1') {
                uint32_t maxpt = 0, nonempty = 0;
                double sumpt = 0;
                for (int t = 0; t < num_tiles; ++t) {
                    const uint32_t pt = gp.mb_coeff_off[t + 1] - gp.mb_coeff_off[t];
                    if (pt > maxpt) maxpt = pt;
                    if (pt > 0) ++nonempty;
                    sumpt += pt;
                }
                uint64_t bits = 0;
                for (std::size_t e = 0; e * GM_LANES < gp.mb_coeff_stream.size(); ++e) {
                    uint32_t mk;
                    std::memcpy(&mk, &gp.mb_coeff_stream[e * GM_LANES + 10], 4);
                    bits += static_cast<uint64_t>(__builtin_popcount(mk));
                }
                std::fprintf(stderr, "[MB_STAT] gaussian-major: num_tiles=%d nonempty=%u total_gaussian_rows=%u max/tile=%u mean/nonempty=%.1f sum_popcount=%llu (should==pairs_kept_per_mb=%lld)\n",
                             num_tiles, nonempty, row, maxpt, nonempty ? sumpt / nonempty : 0.0,
                             (unsigned long long)bits, (long long)gp.pairs_kept_per_mb);
            }

            const auto _t2 = std::chrono::steady_clock::now();
            std::vector<float> img;
            blend_mb_from_payload(gp.mb_counts, gp.mb_coeff_off, gp.mb_coeff_stream,
                                  num_tiles, tiles_x, image_height, image_width, img);
            const auto _t3 = std::chrono::steady_clock::now();
            if (mb_timing) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                std::fprintf(stderr,
                    "[BLEND_HOST] fused_build=%.1f device_blend=%.1f (rows=%u) total=%.1f ms\n",
                    ms(_t0, _t1), ms(_t2, _t3), row, ms(_t0, _t3));
            }
            const std::size_t n = static_cast<std::size_t>(image_height) *
                                  static_cast<std::size_t>(image_width) * 3;
            if (img.size() == n) {
                std::memcpy(image_out, img.data(), n * sizeof(float));
            }
            return result;
        }

        // Non-MB_KERNEL paths still need the per-microblock payload (CPU oracle
        // reference + the device fallback below both consume it).
        MbPayload p = build_mb_payload(
            means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
            tile_ranges, M, P, tiles_x, tiles_y, tile_size, image_height,
            image_width, mb_contrib_floor, pool, cull_disabled);
        result.pairs_dropped_all_mb = p.pairs_dropped_all_mb;
        result.pairs_kept_per_mb = p.pairs_kept_per_mb;

        // blend_mode=2 runs the TT DEVICE blend by default (DST-persistent fp32
        // full-tile kernel: 56-64 dB vs cpu_cpp_mb, runs entirely on-device per
        // the "no CPU/Python in the render loop" goal). Set GSPLAT_TT_DEVICE_BLEND=0
        // to fall back to the C++ CPU basis oracle (validation only).
        const char* dev_blend = std::getenv("GSPLAT_TT_DEVICE_BLEND");
        const bool use_cpu_oracle = dev_blend != nullptr && dev_blend[0] == '0';
        if (!use_cpu_oracle) {
            KernelInputs k = build_kernel_inputs(
                means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
                tile_ranges, P, tiles_x, tiles_y, tile_size);
            setenv("GSPLAT_TT_DEVICE_KERNEL", "1", 1);
            std::vector<float> img;
            std::vector<float> empty_coeff;
            std::vector<uint32_t> empty_u32;
            blend_from_payload(k.packs, k.offsets, k.px, k.py, empty_coeff,
                               empty_u32, empty_u32, image_height, image_width,
                               img);
            const std::size_t n = static_cast<std::size_t>(image_height) *
                                  static_cast<std::size_t>(image_width) * 3;
            if (img.size() == n) {
                std::memcpy(image_out, img.data(), n * sizeof(float));
            }
        } else {
            blend_from_mb_payload_cpu(p, image_height, image_width,
                                      transmittance_threshold, image_out);
        }
        return result;
    }

    // Fallback: bit-identical CPU cull+blend.
    return gsplat_cpu::cull_and_blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges,
        M, P, tiles_x, tiles_y, tile_size, image_height, image_width,
        mb_contrib_floor, pool, image_out_external, cull_disabled,
        transmittance_threshold);
}

}  // namespace gsplat_tt
